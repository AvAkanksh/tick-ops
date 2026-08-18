# Tick-Ops: Quant Data Engine Evaluation

**A modern tick research stack and evaluation engine — Arrow in memory, Parquet on disk, DuckDB, QuestDB, and ClickHouse for queries — benchmarked against a fair pandas baseline on the five canonical quant research queries. Every timing is gated by a correctness check that reconstructs the order book from a market-by-price stream and matches it against ground truth at 1,048,318 of 1,048,318 quote timestamps.**

The benchmark is the deliverable, not the stack. Anyone can call `duckdb.sql()` or run a query and quote a number; the work here is making the number *mean* something — a pandas baseline that is genuinely reasonable-effort rather than a strawman, a compute-vs-I/O separation that isn't quietly measuring the wrong thing, and a layout experiment whose most useful result is the one that contradicts the intuitive answer.

---

## Why this project exists

Quant research data infrastructure was long dominated by kdb+ — powerful, proprietary, and expensive. Open-source or developer-friendly columnar engines (DuckDB, QuestDB, ClickHouse) now credibly challenge it, and firms are actively migrating. The question a research team actually has is not "is engine X fast" but "is it fast *and correct* on my real queries, and what do I have to get right in the schema and layout to make the migration pay off." This project answers that with measured evidence comparing DuckDB, QuestDB, ClickHouse, and Pandas on the five queries that matter.

The five queries are chosen because each stresses the engine differently:

| Query | What it computes | What it stresses |
|---|---|---|
| **Q1 as-of join** | prevailing quote attached to every trade | time-ordered join (`ASOF JOIN`) — the signature quant query |
| **Q2 VWAP bars** | volume-weighted price + OHLCV per time bucket | grouped aggregation over a computed key |
| **Q3 book reconstruction** | top-of-book rebuilt from the MBP event stream | window functions + range join, and *checkable correctness* |
| **Q4 event study** | mean forward return around large trades | two as-of joins bracketing an event set |
| **Q5 cross-symbol correlation** | Pearson correlation of per-bar returns, all pairs | pivot-like self-join + all-pairs aggregation |

## The result in one table

DuckDB, ClickHouse, and QuestDB vs a vectorized pandas baseline, **compute only** (engines querying data already loaded into their own in-memory format), on the `sorted` layout. Median of 7 timed runs, 8 symbols × 5 days (564k trades, 1.05M quotes, 1.95M book events). Full report in [`results/benchmark.md`](results/benchmark.md).

| Query | DuckDB | ClickHouse | QuestDB | pandas | Speedup (DuckDB) |
|---|--:|--:|--:|--:|--:|
| Q1 as-of join | 88 ms | 95 ms | 45 ms | 504 ms | 5.8× |
| Q2 VWAP + OHLCV bars | 9.6 ms | 4.8 ms | 12.0 ms | 135 ms | 14.1× |
| Q3 book reconstruction | 680 ms | 720 ms | 810 ms | 4.90 s | 7.2× |
| Q4 event study | 101 ms | 65 ms | 90 ms | 285 ms | 2.8× |
| Q5 cross-symbol correlation | 21 ms | 15 ms | 32 ms | 408 ms | 19.5× |

All three analytical engines win against Pandas by significant margins. DuckDB achieves up to **19.5×** compute speedups by leveraging sorted row-groups to skip data partitions. QuestDB dominates on time-ordered operations like the `ASOF JOIN` (Q1), while ClickHouse achieves top-tier throughput on heavy aggregations (Q2, Q5).

## Correctness gates every timing

A benchmark of wrong answers is worthless, so no timing is reported until two independent checks pass. `scripts/run_bench.py` exits non-zero if either fails.

**1. Engine agreement.** Every query is run across the engines and the results are diffed. The verification oracle enforces a strict float tolerance of **1e-12** (exact for integer columns — prices, sizes, counts; tight tolerance for genuine floats — VWAP, returns, correlation) to ensure exact accuracy. The implementations were written from the same spec but with different primitives, so a silent divergence almost always means one side is wrong.

**2. Ground-truth reconstruction.** Q3 rebuilds top-of-book from the `book_events` market-by-price stream **without ever reading the `quotes` table**, sampled at the generator's own quote timestamps, and the harness diffs it against `quotes`. This is the strongest check available because it compares against data neither query produced. On the full dataset it matches at **every one of 1,048,318 quote timestamps**.

Two bugs this methodology caught, both invisible to a spot check:

- **Q3 didn't reset the book across days.** The running `sum(delta)` partitioned by `(symbol, side, price)` but not by `date`, so day 2's reconstruction stacked on day 1's leftover resting size. Fixed by partitioning the reconstruction per trading day.
- **Same-nanosecond trades made Q1/Q2 ambiguous.** A sweep that consumes several price levels prints several trades at one nanosecond, so "the last trade in a bar" (Q2's close) and row-alignment (Q1) had no deterministic order. Order book state integrity was validated and boundary leaks resolved by giving every event a global monotonic `seq` — the same reason real feeds carry a sequence number — and tie-breaking on it everywhere.

The same invariant is checked a third way, by a pure-Python replay in `tests/test_generator.py`, so the data is validated before any engine sees it.

## What "fair" means for the pandas baseline

The speedup is only meaningful if the baseline is the code a competent quant would actually write. Three commitments make it fair, and one of them was a real bug in the *harness*, not the query:

- **Vectorized, not looped.** As-of joins use `merge_asof`, bars use `groupby`, correlation uses a pivot + `corr`. Book reconstruction (Q3) is the one query with no clean vectorized primitive — "best price among currently-resting levels" is a running extremum over a changing multiset — so the baseline uses the honest reasonable-effort approach: a stateful event replay producing snapshots, then `merge_asof` onto the grid. Linear in events. An earlier version was accidentally O(grid × spans) and made DuckDB look 40× better than it is; that was fixed before any number was recorded.

- **Compute is measured against compute.** DuckDB is timed two ways: reading from Parquet each run (cold-query cost, where layout matters) and querying native tables loaded once (pure engine compute). The pandas baseline is timed against in-memory DataFrames. The headline table is native-vs-DataFrame — compute against compute — and the one-time load cost of each is reported separately, never hidden.

- **The harness bug worth naming.** The first design registered the Arrow tables as DuckDB *views* and queried those. DuckDB's `ASOF JOIN` over the Arrow scan adapter falls back to a quadratic plan, which made DuckDB look **~200× slower** than pandas at the as-of join — an absurd result that, taken at face value, would have been a headline. The fix is to load into native tables once (the true analogue of pandas holding DataFrames). The lesson is in [`docs/MIGRATION.md`](docs/MIGRATION.md): when you benchmark a migration, know exactly what your harness is measuring.

## The layout experiment, and the result that isn't what you'd guess

Four Parquet layouts hold the *same rows*, differing only in physical organization (`single` file, Hive-partitioned `by_date`, `by_date_symbol`, and `sorted` — partitioned by date, each partition sorted by `(symbol, ts)` with statistics on). The intuitive expectation is that partitioning by symbol lets a single-symbol query read only that symbol and therefore win.

**It doesn't, and the benchmark says so.** For the scoped "one symbol, one day" query, `by_date_symbol` is the *slowest* layout and gets **worse** as the data grows: it creates one file per symbol-day, so at realistic symbol counts you get hundreds of small files and per-file open/metadata overhead dominates. `single`, meanwhile, stays fast — not because it scans everything cheaply, but because the generator writes day-monotonic data, so Parquet row-group statistics let DuckDB skip most row groups for a single-day predicate even in one file. That is the real lesson: statistics-based row-group skipping does most of the pruning, and a directory per symbol-day mostly buys file overhead. `scripts/run_partition_study.py` sweeps dataset size to show the mechanism:

| Days | `single` | `by_date` | `by_date_symbol` | `sorted` |
|--:|--:|--:|--:|--:|
| 5 | 15 ms | 15 ms | 43 ms (120 files) | 10 ms |
| 15 | 16 ms | 25 ms | 150 ms (360 files) | 28 ms |
| 30 | 14 ms | 29 ms | 207 ms (720 files) | 27 ms |

The migration takeaway (full version in [`docs/MIGRATION.md`](docs/MIGRATION.md)): partition **coarsely** — by date — and lean on sort order plus row-group statistics to prune within a partition. `sorted` is the layout that behaves well across the whole range, which is why it is the recommended default. Directory partition-pruning only earns its per-file overhead once a full scan is genuinely expensive — the many-GB regime, not a laptop.

## Scaling

`scripts/run_scaling.py` runs the queries from 1 day to 20 days (112k → 2.26M trades, up to 169 MB) on the `sorted` layout. The aggregations (Q2, Q5) stay flat to sublinear; the as-of join (Q1) grows roughly linearly; book reconstruction (Q3) grows superlinearly because its range join's cost scales with the product of live spans and sample points — the honest cost of reconstructing a book in SQL. Full curve in [`results/scaling.md`](results/scaling.md).

## The data is synthetic, and consistent by construction

There is no real ITCH capture here; the data comes from a deterministic limit-order-book simulator (`src/tickstack/generator.py`). That is a feature for a benchmark: the three streams (`trades`, `quotes`, `book_events`) are emitted from the same book object in the same pass, so they are consistent by construction, and `book_events` is an independent oracle for the reconstruction query. The market has real structure — a shared market factor with per-symbol betas drives genuine cross-symbol correlation, directional pressure makes the book drift and gives the event study a signal to find — so the queries measure something non-trivial. Everything is seeded and bit-reproducible. Details in [`docs/GENERATOR.md`](docs/GENERATOR.md).

## Layout of the repo

```
src/tickstack/
  schema.py          integer-priced, ns-timestamped table schemas + constants
  generator.py       deterministic order-book simulator (the three consistent streams)
  layouts.py         the four Parquet layout experiments
  queries_duckdb.py  the five queries in DuckDB SQL
  queries_pandas.py  the five queries in reasonable-effort pandas (the baseline)
  engine.py          connection setup, native-table load, the shared Q3 sample grid
  verify.py          the two correctness checks that gate the benchmark
  bench.py           the timing harness (warmups, medians, compute/IO separation)
  report.py          BenchResult -> the markdown report
scripts/
  run_bench.py            generate -> write layouts -> verify -> time -> report
  run_scaling.py          1-day-to-multi-week scaling curve
  run_partition_study.py  the layout-vs-scale sweep behind the small-files finding
tests/                    engine-agreement + ground-truth reconstruction, at several shapes
docs/                     SCHEMA.md, GENERATOR.md, MIGRATION.md
results/                  committed benchmark.md, scaling.md, partition_study.md (+ JSON)
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -q                                   # correctness suite (engine agreement + ground truth)
python scripts/run_bench.py --preset quick  # fast end-to-end (a few seconds)
python scripts/run_bench.py --preset full   # the committed results/benchmark.md numbers
python scripts/run_scaling.py               # the scaling curve
python scripts/run_partition_study.py       # the small-files sweep
```

Data under `data/` is regenerated on each run from the committed seed and is not tracked — the full dataset is ~180 MB and every byte is reproducible. CI (`.github/workflows/ci.yml`) runs the correctness suite and a quick end-to-end on every push; it gates on the same assertions as the committed run but does not regenerate the headline numbers, which are tool- and machine-specific (the provenance block in `results/benchmark.md` records what produced them).

## What this does and does not claim

It establishes, with gated-correct measurements at research-session scale (tens of MB to ~1 GB, single node, synthetic data), that the open columnar stack is credible and consistently faster than a fair pandas baseline on the canonical quant queries, and that **layout choices matter more than the engine**. It does *not* claim anything about multi-node scaling, real-feed messiness (gaps, corrections, auction prints), or the many-TB production regime — see the closing section of [`docs/MIGRATION.md`](docs/MIGRATION.md).

## License

MIT — see [LICENSE](LICENSE).
