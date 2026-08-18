#!/usr/bin/env python3
"""End-to-end benchmark driver: generate → write layouts → verify → time → report.

Usage:
    python scripts/run_bench.py --preset quick        # fast, for CI / smoke
    python scripts/run_bench.py --preset full         # the committed results
    python scripts/run_bench.py --days 10 --events 60000 --repeats 9

Everything is seeded, so the same arguments reproduce the same dataset and — noise
aside — the same timings. The script writes:

    results/benchmark.json   machine-readable timings, sizes, correctness, provenance
    results/benchmark.md     the human-readable report (five-query table, layouts,
                             scaling note, correctness summary)

It never reports a timing for a query whose two engine implementations disagree:
:func:`tickstack.bench.run_benchmark` raises before timing if verification fails.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

# Make ``src`` importable whether run from the repo root or elsewhere.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import duckdb  # noqa: E402
import numpy  # noqa: E402
import pandas  # noqa: E402
import pyarrow  # noqa: E402

from tickstack import engine  # noqa: E402
from tickstack.bench import run_benchmark  # noqa: E402
from tickstack.generator import SimConfig, Simulator  # noqa: E402
from tickstack.layouts import LAYOUTS, LayoutPaths, build_all_layouts  # noqa: E402
from tickstack.report import write_reports  # noqa: E402
from tickstack.verify import verify_q3_against_ground_truth  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=["quick", "full"], default=None,
                   help="quick: tiny CI dataset; full: the committed-results dataset")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--events", type=int, default=None, help="events per symbol per day")
    p.add_argument("--symbols", type=str, default=None, help="comma-separated symbols")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--data-dir", type=Path, default=_ROOT / "data" / "parquet")
    p.add_argument("--results-dir", type=Path, default=_ROOT / "results")
    p.add_argument("--no-pandas", action="store_true", help="skip the pandas baseline")
    p.add_argument("--layouts", type=str, default=",".join(LAYOUTS))
    p.add_argument("--keep-data", action="store_true", help="do not delete data dir after run")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> SimConfig:
    # 'full' preset == SimConfig defaults (5 days, 8 symbols, 40k events/symbol/day).
    base = SimConfig.quick() if args.preset == "quick" else SimConfig()
    kw = {}
    if args.days is not None:
        kw["n_days"] = args.days
    if args.events is not None:
        kw["events_per_symbol_day"] = args.events
    if args.symbols is not None:
        kw["symbols"] = tuple(s.strip() for s in args.symbols.split(",") if s.strip())
    if args.seed is not None:
        kw["seed"] = args.seed
    return SimConfig(**{**base.__dict__, **kw}) if kw else base


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    layouts = tuple(s.strip() for s in args.layouts.split(",") if s.strip())

    print(f"[1/4] generating: {cfg.n_days} days x {len(cfg.symbols)} symbols "
          f"x {cfg.events_per_symbol_day} events (seed {cfg.seed})", flush=True)
    sim = Simulator(cfg)
    print(f"[2/4] writing layouts {layouts} to {args.data_dir}", flush=True)
    build_all_layouts(args.data_dir, sim.simulate(), layouts=layouts)

    # Ground-truth Q3 check on the full generated data (not just the benchmark's).
    con = engine.connect(threads=args.threads)
    engine.register_layout(con, LayoutPaths(root=args.data_dir / "single", kind="single"))
    matched, total = verify_q3_against_ground_truth(con)
    con.close()
    print(f"[3/4] Q3 ground-truth: {matched}/{total} quote timestamps reconstructed exactly", flush=True)
    if matched != total:
        print("ERROR: book reconstruction does not match generator quotes", file=sys.stderr)
        return 2

    print(f"[4/4] benchmarking (repeats={args.repeats}, threads={args.threads})", flush=True)
    res = run_benchmark(
        args.data_dir, layouts=layouts, repeats=args.repeats,
        threads=args.threads, run_pandas=not args.no_pandas,
    )
    res.meta.update(
        {
            "config": {
                "seed": cfg.seed, "n_days": cfg.n_days, "symbols": list(cfg.symbols),
                "events_per_symbol_day": cfg.events_per_symbol_day,
            },
            "q3_ground_truth": {"matched": matched, "total": total},
            "versions": {
                "python": platform.python_version(),
                "duckdb": duckdb.__version__,
                "clickhouse": "26.3 (eval)",
                "questdb": "8.2 (eval)",
                "pyarrow": pyarrow.__version__,
                "pandas": pandas.__version__,
                "numpy": numpy.__version__,
            },
            "platform": f"{platform.system()} {platform.machine()}",
        }
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "benchmark.json").write_text(json.dumps(res.to_dict(), indent=2))
    write_reports(res, args.results_dir)
    print(f"wrote {args.results_dir / 'benchmark.json'} and benchmark.md", flush=True)

    if not args.keep_data:
        import shutil
        shutil.rmtree(args.data_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
