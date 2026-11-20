#!/usr/bin/env python3
"""Scaling study: how per-query latency grows from one day to multi-week data.

Runs the five queries (DuckDB, in-memory) at a sequence of day counts, holding
everything else fixed, and records median latency and dataset size at each point.
The output is the scaling curve in ``results/scaling.json`` / ``scaling.md`` — the
"one day to multi-week" experiment the project promises.

Only DuckDB is scaled by default: the pandas baseline's book-reconstruction replay
is linear but slow, and running it at multi-week scale adds minutes without changing
the story (the compute-speedup table already establishes the per-query gap). Pass
``--with-pandas`` to include it.

Usage:
    python scripts/run_scaling.py                     # 1,2,5,10,20 days
    python scripts/run_scaling.py --days 1,3,7,14,28 --with-pandas
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import duckdb  # noqa: E402

from tickstack import engine  # noqa: E402
from tickstack import queries_pandas as qp  # noqa: E402
from tickstack.bench import QUERIES, _duckdb_runners, _time_call  # noqa: E402
from tickstack.generator import SimConfig, Simulator  # noqa: E402
from tickstack.layouts import LayoutPaths, build_all_layouts, dataset_size_bytes  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=str, default="1,2,5,10,20")
    p.add_argument("--events", type=int, default=40_000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--layout", type=str, default="sorted")
    p.add_argument("--with-pandas", action="store_true")
    p.add_argument("--data-dir", type=Path, default=_ROOT / "data" / "scaling")
    p.add_argument("--results-dir", type=Path, default=_ROOT / "results")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    day_counts = [int(d) for d in args.days.split(",") if d.strip()]
    layout = args.layout
    points = []

    for n_days in day_counts:
        cfg = SimConfig(n_days=n_days, events_per_symbol_day=args.events)
        print(f"[days={n_days}] generating + writing '{layout}'...", flush=True)
        build_all_layouts(args.data_dir, Simulator(cfg).simulate(), layouts=(layout,))
        paths = LayoutPaths(root=args.data_dir / layout, kind=layout)

        src = engine.connect(threads=args.threads)
        engine.register_layout(src, paths)
        n_trades = src.execute("SELECT count(*) FROM trades").fetchall()[0][0]
        n_quotes = src.execute("SELECT count(*) FROM quotes").fetchall()[0][0]
        n_book = src.execute("SELECT count(*) FROM book_events").fetchall()[0][0]
        size = sum(dataset_size_bytes(paths, t) for t in ("trades", "quotes", "book_events"))

        con = engine.load_native(src, threads=args.threads)
        engine.build_sample_grid(con)
        runners = _duckdb_runners(con)
        duck_ms = {}
        for q in QUERIES:
            samples, _ = _time_call(runners[q], args.repeats)
            duck_ms[q] = statistics.median(samples)

        pandas_ms = {}
        if args.with_pandas:
            frames = {k: v.to_pandas() for k, v in engine.load_tables_arrow(src).items()}
            grid_df = engine.build_sample_grid(con).to_pandas()
            # Bind frames/grid_df as defaults so the closures don't capture the loop
            # variable (they are called in-iteration, but binding is the clean form).
            preg = {
                "q1_asof": lambda f=frames: qp.q1_asof_trades_quotes(f["trades"], f["quotes"]),
                "q2_vwap": lambda f=frames: qp.q2_vwap_bars(f["trades"]),
                "q3_book": lambda f=frames, g=grid_df: qp.q3_book_reconstruction(f["book_events"], g),
                "q4_event": lambda f=frames: qp.q4_event_study(f["trades"], f["quotes"]),
                "q5_corr": lambda f=frames: qp.q5_cross_symbol_corr(f["quotes"]),
            }
            for q in QUERIES:
                samples, _ = _time_call(preg[q], max(3, args.repeats // 2))
                pandas_ms[q] = statistics.median(samples)

        points.append({
            "n_days": n_days,
            "rows": {"trades": n_trades, "quotes": n_quotes, "book_events": n_book},
            "total_bytes": size,
            "duckdb_ms": duck_ms,
            "pandas_ms": pandas_ms,
        })
        print(f"[days={n_days}] trades={n_trades:,} size={size/1024/1024:.1f}MB "
              f"q1={duck_ms['q1_asof']:.1f}ms q3={duck_ms['q3_book']:.1f}ms", flush=True)
        src.close()
        con.close()

    out = {
        "meta": {
            "layout": layout, "events_per_symbol_day": args.events,
            "repeats": args.repeats, "threads": args.threads,
            "with_pandas": args.with_pandas,
            "platform": f"{platform.system()} {platform.machine()}",
            "duckdb": duckdb.__version__,
        },
        "points": points,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "scaling.json").write_text(json.dumps(out, indent=2))
    (args.results_dir / "scaling.md").write_text(_render_md(out))
    print(f"wrote {args.results_dir/'scaling.json'} and scaling.md")

    import shutil
    shutil.rmtree(args.data_dir, ignore_errors=True)
    return 0


def _render_md(out: dict) -> str:
    L = ["# Scaling study\n",
         f"Query latency vs dataset size on the `{out['meta']['layout']}` layout, "
         f"{out['meta']['events_per_symbol_day']:,} events/symbol/day, "
         f"{out['meta']['repeats']} timed runs (median). DuckDB in memory.\n"]
    L.append("| Days | Trades | Size | Q1 as-of | Q2 VWAP | Q3 book | Q4 event | Q5 corr |")
    L.append("|--:|--:|--:|--:|--:|--:|--:|--:|")
    for p in out["points"]:
        d = p["duckdb_ms"]
        size_mb = p["total_bytes"] / 1024 / 1024
        L.append(
            f"| {p['n_days']} | {p['rows']['trades']:,} | {size_mb:.1f} MB "
            f"| {d['q1_asof']:.1f} ms | {d['q2_vwap']:.1f} ms | {d['q3_book']:.1f} ms "
            f"| {d['q4_event']:.1f} ms | {d['q5_corr']:.1f} ms |"
        )
    L.append("")
    if out["meta"]["with_pandas"] and out["points"] and out["points"][0]["pandas_ms"]:
        L.append("## pandas, same points (median ms)\n")
        L.append("| Days | Q1 as-of | Q2 VWAP | Q3 book | Q4 event | Q5 corr |")
        L.append("|--:|--:|--:|--:|--:|--:|")
        for p in out["points"]:
            m = p["pandas_ms"]
            L.append(
                f"| {p['n_days']} | {m['q1_asof']:.1f} ms | {m['q2_vwap']:.1f} ms "
                f"| {m['q3_book']:.1f} ms | {m['q4_event']:.1f} ms | {m['q5_corr']:.1f} ms |"
            )
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
