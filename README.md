# Tick-Ops: Quant Data Engine Evaluation

**A tick research stack — Arrow in memory, Parquet on disk, DuckDB for queries — measured against pandas, Polars, ClickHouse and QuestDB on the five canonical quant research queries. All six configurations run for real, and every timing is gated by a correctness oracle: integer columns exact, float columns within 1e-12, and an order-book reconstruction that matches ground truth at 1,048,318 of 1,048,318 quote timestamps.**

The benchmark is the deliverable, not the stack. Anyone can call `duckdb.sql()` and quote a number; the work here is making the number *mean* something — baselines that are genuinely reasonable-effort rather than strawmen, a compute-vs-I/O separation that isn't quietly measuring the wrong thing, ports to other engines that are proven equivalent before they are timed, and a willingness to report results that contradict the intuitive (or the previously published) answer.

---

## Why this project exists

Quant research data infrastructure was long dominated by kdb+ — powerful, proprietary, and expensive. Open columnar engines (DuckDB, ClickHouse, QuestDB) and modern DataFrame libraries (Polars) now credibly challenge it, and firms are actively migrating. The question a research team actually has is not "is engine X fast" but "is it fast *and correct* on my real queries, and what do I have to get right in the schema, layout and SQL to make the migration pay off." This project answers that with measured evidence.

The five queries are chosen because each stresses an engine differently:

| Query | What it computes | What it stresses |
|---|---|---|
| **Q1 as-of join** | prevailing quote attached to every trade | time-ordered join (`ASOF JOIN`) — the signature quant query |
| **Q2 VWAP bars** | volume-weighted price + OHLCV per time bucket | grouped aggregation over a computed key |
| **Q3 book reconstruction** | top-of-book rebuilt from the MBP event stream | window functions + range join, and *checkable correctness* |
| **Q4 event study** | mean forward return around large trades | two as-of joins bracketing an event set |
| **Q5 cross-symbol correlation** | Pearson correlation of per-bar returns, all pairs | self-join + all-pairs aggregation |

## The result in one table

**Compute only**: each engine queries data already loaded into its own format, from the `sorted` layout. Median of 7 timed runs after a warmup; 4 threads per engine (ClickHouse and QuestDB containers pinned to 4 CPUs). 8 symbols × 5 days: 564k trades, 1.05M quotes, 1.95M book events. Inter-quartile ranges, load costs and the layout experiment are in [`results/benchmark.md`](results/benchmark.md).

| Query | DuckDB | DuckDB 1 thread | Polars | ClickHouse | QuestDB | pandas |
|---|--:|--:|--:|--:|--:|--:|
| Q1 as-of join | **126 ms** | 406 ms | 231 ms | 322 ms | 1.06 s | 159 ms |
| Q2 VWAP + OHLCV bars | **14 ms** | 28 ms | 36 ms | 43 ms | 51 ms | 60 ms |
| Q3 book reconstruction | **842 ms** | 3.06 s | 935 ms | 1.58 s | 6.68 s | 4.33 s |
| Q4 event study | **139 ms** | 345 ms | **139 ms** | 271 ms | 581 ms | 151 ms |
| Q5 cross-symbol correlation | **24 ms** | 60 ms | 49 ms | 80 ms | 63 ms | 141 ms |

How to read it, honestly:

- **DuckDB vs pandas: 1.1× to 6.0×**, largest on the aggregations (Q2 4.3×, Q5 6.0×) and on book reconstruction (Q3 5.1×, where pandas has no set-based primitive and must replay events in Python). On the join-dominated queries (Q1 1.3×, Q4 1.1×) pandas' purpose-built `merge_asof` is nearly as fast.
- **Part of DuckDB's edge is parallelism.** At one thread DuckDB is *slower* than pandas on Q1 and Q4; the single-thread column is there so "DuckDB is faster" isn't silently "4 cores are faster than 1".
- **Polars is the stronger baseline.** It ties DuckDB on Q4 and is within 11% on Q3; DuckDB leads it 1.8–2.5× on Q1, Q2 and Q5.
- **The servers pay a wire tax.** ClickHouse and QuestDB are queried over HTTP, so their numbers include the round-trip and decoding the result into Arrow in Python; the in-process engines pay none of that. Q1 returns one row per trade, so it is dominated by transfer. Read the server columns as "what a Python researcher experiences", not as pure engine compute.
- **QuestDB's Q3 is a rewrite.** The direct range-join port is correct but exceeds QuestDB's 60 s query timeout at this scale; the as-of-join formulation (same answer, verified) takes 6.7 s.

## Correctness gates every timing

A benchmark of wrong answers is worthless, so no timing is reported until every engine has passed. `scripts/run_bench.py` exits non-zero if any check fails, and refuses to run a server it cannot reach rather than estimating it.

**1. Engine agreement.** DuckDB is diffed against pandas; then Polars, ClickHouse and QuestDB are each diffed against the DuckDB reference — integer columns (prices, sizes, counts) exactly, float columns (VWAP, returns, correlations) within **1e-12**. The implementations use different primitives (SQL windows, `merge_asof`, Polars expressions, ClickHouse/QuestDB dialects), so a silent divergence almost always means one side is wrong.

**2. Ground-truth reconstruction.** Q3 rebuilds top-of-book from the `book_events` market-by-price stream **without ever reading the `quotes` table**, sampled at the generator's own quote timestamps, and the harness diffs it against `quotes`. It compares against data no query produced, so two implementations sharing a mistake cannot pass it. On the full dataset it matches at **every one of 1,048,318 quote timestamps**.

Bugs this methodology caught, all invisible to a spot check:

- **Q3 didn't reset the book across days.** The running `sum(delta)` partitioned by `(symbol, side, price)` but not by `date`, so day 2 stacked on day 1's leftover size. Fixed by partitioning per trading day.
- **Same-nanosecond trades made Q1/Q2 ambiguous.** A sweep prints several trades at one nanosecond, so "the last trade in a bar" had no deterministic order. Fixed with a global monotonic `seq` — the same reason real feeds carry one — tie-broken on everywhere.
- **ClickHouse's `log` is approximate.** ~1e-9 relative error near 1, which moved Q5 correlations by 2e-10 and failed the oracle. `log1p(x - 1)` is correctly rounded. Three more ClickHouse traps (zero-filled unmatched joins, approximate `quantile`, frame-bound `leadInFrame`) and QuestDB's are in [`docs/MIGRATION.md`](docs/MIGRATION.md).

The data is also validated a third way, by a pure-Python replay in `tests/test_generator.py`, before any engine sees it.

## What "fair" means for the baselines

- **Vectorized, not looped.** pandas uses `merge_asof`, `groupby`, pivot + `corr`; Polars uses lazy frames, `join_asof` and expression `group_by`. Book reconstruction is the exception for pandas — "best price among currently-resting levels" is a running extremum over a changing multiset, with no vectorized primitive — so it uses the honest reasonable-effort approach: a linear stateful replay, then `merge_asof` onto the grid. Polars gets a genuinely vectorized formulation (as-of probes per price level).
- **Compute against compute.** DuckDB is timed reading Parquet each run (cold cost, where layout matters) and against native tables loaded once (compute). Every other engine is timed against data in its own format, and every engine's one-time load (Parquet → Arrow → engine) is reported separately.
- **Same resources.** 4 threads everywhere — DuckDB `PRAGMA threads`, `POLARS_MAX_THREADS`, ClickHouse `max_threads`, and 4-CPU containers — plus the 1-thread DuckDB column.
- **Harness bugs worth naming.** (a) Registering Arrow tables as DuckDB *views* sends `ASOF JOIN` through the Arrow scan adapter's quadratic plan — DuckDB looked ~200× *slower* than pandas. Fixed by loading native tables. (b) In DuckDB 2.0, Hive-partition-pruned scans are estimated at "~1 row", which makes the planner swap the ASOF join for a nested-loop join: the scoped query ran **40× slower** (1.2 s vs 30 ms) on exactly the partitioned layouts, which would have "proved" partitioning harmful. `engine.connect` sets `asof_loop_join_threshold = 0`, and a regression test asserts the plan keeps the ASOF operator. The lesson is in [`docs/MIGRATION.md`](docs/MIGRATION.md): when you benchmark a migration, know exactly what your harness — and the planner — is measuring.

## The layout experiment

Four Parquet layouts hold the *same rows*, differing only in physical organization: one `single` file per table, Hive-partitioned `by_date`, `by_date_symbol`, and `sorted` (partitioned by date, each partition sorted by `(symbol, ts)`, statistics on). The intuition is that partitioning by symbol makes a single-symbol query fast.

**At research-session scale it buys nothing, and as data grows it costs.** For the scoped "one symbol, one day" query at 5 days, all four layouts land between 19 and 28 ms. Pruning works everywhere: Parquet row-group statistics skip most of even the `single` file, because the generator writes day-monotonic data. What `by_date_symbol` adds is files — one per symbol-day, 8× as many as `by_date` — and `scripts/run_partition_study.py` shows that cost growing with history:

| Days | `single` | `by_date` | `by_date_symbol` | `sorted` |
|--:|--:|--:|--:|--:|
| 5 | 24 ms | 19 ms | 24 ms (120 files) | **17 ms** |
| 15 | 26 ms | 21 ms | 42 ms (360 files) | **18 ms** |
| 30 | 28 ms | 28 ms | 70 ms (720 files) | **26 ms** |

The migration takeaway (full version in [`docs/MIGRATION.md`](docs/MIGRATION.md)): partition **coarsely** — by date — and lean on sort order plus row-group statistics to prune within a partition. `sorted` is fastest at every size, which is why it is the recommended default. Directory partitioning by symbol only earns its per-file overhead once a full scan is genuinely expensive — the many-GB regime, not a laptop.

## Scaling

`scripts/run_scaling.py` runs DuckDB from 1 to 20 days (112k → 2.26M trades, 8 → 169 MB) on the `sorted` layout. Every query scales roughly linearly or better: going from 1 to 20 days (20× the data), Q1 grows 9.6×, Q3 18.7×, Q4 11.9×, and the aggregations Q2/Q5 grow 4.6×/6.2× because a fixed per-query overhead dominates at small sizes. Q3 stays linear because its range join only ever matches spans within one symbol-day. Full curve in [`results/scaling.md`](results/scaling.md).

## The data is synthetic — what it does and doesn't reproduce

There is no real ITCH capture here; the data comes from a deterministic limit-order-book simulator (`src/tickstack/generator.py`). For a benchmark that is a feature: the three streams (`trades`, `quotes`, `book_events`) are emitted from the same book in the same pass, so they are consistent by construction, and `book_events` is an independent oracle for Q3. Everything is seeded and bit-reproducible. Details in [`docs/GENERATOR.md`](docs/GENERATOR.md).

Whether it *looks like a market* is measured, not asserted — [`results/stylized_facts.md`](results/stylized_facts.md) checks the standard list from Cont (2001):

| Fact | Measured | Reproduced? |
|---|--:|:-:|
| Heavy tails (excess kurtosis of 1-min returns) | +1.64 | yes |
| Cross-sectional correlation (mean pairwise) | +0.60 | yes |
| Bid-ask bounce (lag-1 ACF of trade price changes) | −0.16 | yes |
| Volatility clustering (ACF of \|r\|, lag 1) | +0.07, ~0 by lag 10 | **no** — constant σ by design |
| No return autocorrelation (lag-1 ACF of r) | +0.14 | **no** — mid adjusts to fair value gradually |

None of this affects any timing or correctness result. It does mean Q4/Q5 *values* are properties of the simulator, not market evidence.

## Layout of the repo

```
src/tickstack/
  schema.py              integer-priced, ns-timestamped table schemas + constants
  generator.py           deterministic order-book simulator (the three consistent streams)
  layouts.py             the four Parquet layout experiments
  queries_duckdb.py      the five queries in DuckDB SQL (the reference)
  queries_pandas.py      the five queries in reasonable-effort pandas
  queries_polars.py      the five queries in idiomatic Polars
  queries_clickhouse.py  the five queries in ClickHouse SQL (+ the porting traps)
  queries_questdb.py     the five queries in QuestDB SQL (+ the Q3 rewrite)
  servers.py             ClickHouse/QuestDB adapters: load, query -> Arrow, version
  engine.py              DuckDB connection setup, native-table load, the shared Q3 grid
  verify.py              the correctness checks that gate the benchmark
  bench.py               the timing harness (warmups, medians + IQR, compute/IO split)
  report.py              BenchResult -> the markdown report
scripts/
  run_bench.py            generate -> write layouts -> verify every engine -> time -> report
  run_scaling.py          1-day-to-multi-week scaling curve
  run_partition_study.py  the layout-vs-scale sweep behind the small-files finding
  stylized_facts.py       which empirical market facts the generator reproduces
tests/                    engine agreement (all engines), ground truth, planner regression
docs/                     SCHEMA.md, GENERATOR.md, MIGRATION.md
results/                  committed benchmark / scaling / partition_study / stylized_facts (+ JSON)
docker-compose.yml        ClickHouse 25.8 + QuestDB 9.1, 4 CPUs each, localhost-only ports
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                     # includes polars + server clients
docker compose up -d                        # ClickHouse + QuestDB (needed for their columns/tests)

pytest -q -rs                               # correctness suite; server tests skip if Docker is down
python scripts/run_bench.py --preset quick  # fast end-to-end, every engine gated
python scripts/run_bench.py --preset full   # the committed results/benchmark.md numbers
python scripts/run_bench.py --preset quick --no-servers   # no Docker needed
python scripts/run_scaling.py               # the scaling curve
python scripts/run_partition_study.py       # the small-files sweep
python scripts/stylized_facts.py            # the stylized-facts check
```

Data under `data/` is regenerated on each run from the committed seed and is not tracked. CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) starts ClickHouse and QuestDB as service containers, then runs lint, the full correctness suite (servers included) and a quick end-to-end benchmark on every push. It gates on the same assertions as the committed run but does not regenerate the headline numbers, which are machine-specific; the provenance block in `results/benchmark.md` records exactly what produced them.

## What this does and does not claim

It establishes, with gated-correct measurements at research-session scale (tens of MB to ~1 GB, single node, synthetic data), that in-process DuckDB is the fastest of the six configurations on every query here (tied with Polars on Q4). Its margin over pandas ranges from 1.1× on the join-heavy queries to 6× on the aggregations, and part of that margin is multi-threading. At this scale layout choices matter mostly by *not* over-partitioning. It does *not* claim anything about multi-node scaling, server engines at the data sizes they are built for (where the wire tax amortizes and ClickHouse/QuestDB's storage engines matter), real-feed messiness (gaps, corrections, auction prints), or the many-TB regime — see the closing section of [`docs/MIGRATION.md`](docs/MIGRATION.md).

## License

MIT — see [LICENSE](LICENSE).
