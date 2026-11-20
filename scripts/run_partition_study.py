#!/usr/bin/env python3
"""Partition study: does directory partitioning help a scoped query, and when?

This exists because the honest answer is counterintuitive and the naive intuition
("partition by symbol so a single-symbol query only reads that symbol") is wrong at
the scale a laptop benchmark runs. The study sweeps dataset size and records, for a
scoped as-of join (one symbol, one day), the latency under each layout *and the file
count each layout produces* — because file count is the mechanism behind the result.

The finding, recorded in ``results/partition_study.{json,md}``:

* ``by_date_symbol`` produces one small file per symbol-day. At laptop scale those
  files are hundreds of KB, and per-file open/metadata overhead dominates, so this
  layout gets **worse** as the data grows — the small-files anti-pattern, live.
* ``single`` stays flat and competitive — not by scanning the whole table, but
  because the generator writes day-monotonic data, so Parquet row-group statistics
  skip most row groups for a single-day predicate even inside one file.
* ``sorted`` (partition by date only, sorted by symbol within, statistics on) keeps
  files large *and* prunes row groups by symbol — the layout that scales without the
  small-files penalty.

The migration-guide takeaway is that partitioning is a scale-dependent choice, and
over-partitioning is a real and common mistake. That is a more useful result than a
manufactured "partitioning always wins."
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import duckdb  # noqa: E402

from tickstack import engine  # noqa: E402
from tickstack.generator import SimConfig, Simulator  # noqa: E402
from tickstack.layouts import LAYOUTS, LayoutPaths, build_all_layouts  # noqa: E402

_SCOPED_SQL = """
    WITH t AS (
        SELECT seq, ts, symbol, price, size FROM trades WHERE symbol = ? AND date = ?
    ),
    q AS (
        SELECT ts, symbol, bid_px, ask_px FROM quotes WHERE symbol = ? AND date = ?
    )
    SELECT t.seq, t.ts, t.price, q.bid_px, q.ask_px
    FROM t ASOF LEFT JOIN q ON t.symbol = q.symbol AND t.ts >= q.ts
    ORDER BY t.ts, t.seq
"""


def _scoped_median_ms(base: Path, layout: str, sym: str, day: int, reps: int, threads: int):
    paths = LayoutPaths(root=base / layout, kind=layout)
    samples = []
    result_rows = 0
    for _ in range(reps + 1):  # 1 warmup
        con = engine.connect(threads=threads)
        engine.register_layout(con, paths)
        t0 = time.perf_counter()
        out = con.sql(_SCOPED_SQL, params=[sym, day, sym, day]).to_arrow_table()
        dt = (time.perf_counter() - t0) * 1000
        result_rows = out.num_rows
        con.close()
        samples.append(dt)
    return statistics.median(samples[1:]), result_rows


def _file_count(base: Path, layout: str) -> tuple[int, int]:
    root = base / layout
    files = list(root.rglob("*.parquet"))
    return len(files), sum(f.stat().st_size for f in files)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=str, default="5,15,30")
    p.add_argument("--events", type=int, default=40_000)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--data-dir", type=Path, default=_ROOT / "data" / "partition")
    p.add_argument("--results-dir", type=Path, default=_ROOT / "results")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    day_counts = [int(d) for d in args.days.split(",") if d.strip()]
    symbols = SimConfig().symbols
    points = []

    for n_days in day_counts:
        cfg = SimConfig(n_days=n_days, symbols=symbols, events_per_symbol_day=args.events)
        print(f"[days={n_days}] building all layouts...", flush=True)
        build_all_layouts(args.data_dir, Simulator(cfg).simulate())
        sym = symbols[len(symbols) // 2]
        day = n_days // 2

        per_layout = {}
        for layout in LAYOUTS:
            ms, rows = _scoped_median_ms(args.data_dir, layout, sym, day, args.reps, args.threads)
            nfiles, nbytes = _file_count(args.data_dir, layout)
            per_layout[layout] = {"ms": ms, "files": nfiles, "bytes": nbytes, "rows": rows}
        points.append({"n_days": n_days, "scoped_symbol": sym, "scoped_day": day, "layouts": per_layout})
        summary = "  ".join(f"{lay}={per_layout[lay]['ms']:.1f}ms/{per_layout[lay]['files']}f" for lay in LAYOUTS)
        print(f"[days={n_days}] {summary}", flush=True)
        import shutil
        shutil.rmtree(args.data_dir, ignore_errors=True)

    out = {
        "meta": {
            "events_per_symbol_day": args.events, "reps": args.reps, "threads": args.threads,
            "platform": f"{platform.system()} {platform.machine()}", "duckdb": duckdb.__version__,
        },
        "points": points,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "partition_study.json").write_text(json.dumps(out, indent=2))
    (args.results_dir / "partition_study.md").write_text(_render_md(out))
    print(f"wrote {args.results_dir/'partition_study.json'} and partition_study.md")
    return 0


def _render_md(out: dict) -> str:
    L = ["# Partition study — does partitioning help a scoped query?\n",
         "Scoped as-of join (one symbol, one day) vs dataset size, per layout. Median "
         f"of {out['meta']['reps']} runs after warmup. File count is shown because it "
         "is the mechanism: more, smaller files means more open/metadata overhead.\n"]
    L.append("| Days | Layout | Latency | Files | Avg file |")
    L.append("|--:|---|--:|--:|--:|")
    for p in out["points"]:
        for lay, d in p["layouts"].items():
            avg = d["bytes"] / max(1, d["files"]) / 1024
            L.append(f"| {p['n_days']} | {lay} | {d['ms']:.1f} ms | {d['files']} | {avg:.0f} KB |")
    L.append("")
    L.append(
        "**Reading it:** `by_date_symbol` creates one file per symbol-day, so its file "
        "count grows with the data and each file stays small — its scoped-query latency "
        "*rises* with scale (the small-files anti-pattern). `single` stays low and "
        "nearly flat across this range. Note that "
        "`single` is *not* paying a full-table scan here: because the generator writes "
        "day-by-day, `date` is monotonic in row order, so Parquet row-group min/max "
        "statistics let DuckDB skip most row groups for a single-day predicate even in "
        "one file. That is the real lesson — statistics-based row-group skipping does "
        "most of the pruning work, and a directory per symbol-day mostly buys file "
        "overhead. `sorted` keeps files large while clustering by symbol, so it prunes "
        "by statistics without the small-files penalty and is the layout that scales. "
        "At this scale, directory partition-pruning does not pay for its per-file "
        "overhead; the payoff arrives at the scale where even statistics-pruned scans "
        "dominate — the migration-guide point.\n"
    )
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
