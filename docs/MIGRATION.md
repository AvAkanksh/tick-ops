# Migrating tick research from kdb+ to the open columnar stack

The audience for this is a team running quant research on kdb+ (or a bespoke
in-memory store) and weighing a move to Arrow + Parquet + DuckDB. It is written from
what this benchmark actually measured, not from vendor copy — including the places
where the open stack's naive setup is a trap.

## What transfers cleanly

**The signature query — the as-of join — is a first-class operation.** The single
biggest worry migrating off kdb+ is `aj`, the as-of join that attaches the
prevailing quote to each trade. DuckDB has `ASOF JOIN` as native syntax with a
purpose-built operator; Q1 is a few lines of SQL and runs in ~125 ms in-memory
(~300 ms cold from Parquet) on ~half a million trades (see `results/benchmark.md`).
Be precise about what that buys: pandas' `merge_asof` is also a purpose-built
as-of primitive and is only 1.3× slower, and single-threaded DuckDB is *slower*
than it. The as-of join is now table stakes in open tools — the migration case is
that you get it *inside a SQL engine* alongside everything else, not that it is
dramatically faster than what a pandas desk already has.

**Columnar aggregation is fast and expressive.** VWAP bars (Q2), event studies (Q4),
and cross-symbol correlation (Q5) are ordinary `GROUP BY` / window / self-join SQL.
DuckDB beats a vectorized pandas baseline on every one — by 4–6× on the aggregations
and book reconstruction, by only ~1.1× on the join-dominated event study — and beats
Polars, the stronger baseline, on four of five (tying on Q4). No bespoke primitives
required.

**Arrow is the zero-copy handoff.** DuckDB reads Parquet and hands results to Python
research code as Arrow with no serialization step; `to_pandas()` on the result is
milliseconds. The "query in SQL, analyze in Python" workflow has no impedance layer.

## Where the naive migration goes wrong

**Do not over-partition.** The intuitive Hive layout — `date=/symbol=` — feels right
("a single-symbol query reads only that symbol") and is a documented trap. It
produces one file per symbol-day; at realistic symbol counts that is thousands of
small files, and per-file open/metadata overhead then dominates. In this benchmark
`by_date_symbol` buys nothing for a scoped query at 5 days (all layouts 19–28 ms,
because row-group statistics prune the others just as well) and it gets steadily
worse as history grows — 24 → 42 → 70 ms from 5 to 30 days while `sorted` stays at
17–26 ms (`results/partition_study.md`). Partition coarsely — by date — and rely on
sort order plus row-group statistics to prune within a partition.

**Sort before you write.** The `sorted` layout (partition by date, sort each
partition by `(symbol, ts)`, keep statistics) is the one that behaves well across
scales: files stay large, but row-group min/max on `symbol` lets DuckDB skip row
groups without a directory per symbol. This is the recommended default.

**Know what your benchmark is measuring.** The first version of this project's
harness registered Arrow tables as DuckDB views and queried those. DuckDB's ASOF
join over the Arrow scan adapter falls back to a quadratic plan — it looked like
DuckDB was ~200× *slower* than pandas at the as-of join, which is absurd. The fix is
to load into native DuckDB tables once (the analogue of pandas holding DataFrames)
and query those. If you are benchmarking a migration, measure compute against
compute and I/O against I/O; conflating them will make you draw the wrong conclusion.

**Check the plan, not just the clock.** After upgrading to DuckDB 2.0, the scoped
one-symbol-day as-of join became ~40× slower (1.2 s vs 30 ms) — but only on the
Hive-partitioned layouts. `EXPLAIN` showed why: a partition-pruned Parquet scan is
estimated at "~1 row" regardless of contents, and DuckDB rewrites an ASOF join with
a tiny estimated left side into a nested-loop join + `arg_max` (183M intermediate
rows here). Taken at face value it "proved" partitioning is harmful. The fix is
`SET asof_loop_join_threshold = 0` (in `engine.connect`), guarded by a test that
asserts the plan keeps the ASOF operator. An engine upgrade is a migration too.

## Porting the same five queries to ClickHouse and QuestDB

Both servers run the five queries for real in this benchmark (`docker-compose.yml`),
and every port is diffed against the DuckDB reference — integers exact, floats
within 1e-12 — before it is timed. Getting there surfaced the traps below. Each one
produces a **plausible-looking wrong answer**, not an error, which is why a
migration needs an oracle rather than eyeballing a few rows.

**ClickHouse**

- *Unmatched joins return zeros, not NULLs.* A trade before its symbol's first quote
  gets `bid_px = 0` under the default settings. `join_use_nulls = 1` restores SQL
  semantics.
- *`quantile` is approximate* (reservoir-sampled). The Q4 "top 2% by size" cut needs
  `quantileExactInclusive`, which matches DuckDB's `quantile_cont` interpolation.
- *`log` is approximate.* ClickHouse's `log` is a fast SIMD implementation with
  ~1e-9 relative error near 1 (measured against libm). On 1-minute returns that
  moved correlations by 2e-10 and failed the oracle. `log1p(x - 1)` is correctly
  rounded and fixes it — a trap you only find with a tight tolerance.
- *`leadInFrame`/`lagInFrame` obey the window frame*, whose default ends at the
  current row, so `leadInFrame` silently sees nothing without an explicit
  `UNBOUNDED FOLLOWING`; and their "no row" default is the type's zero, not NULL.

**QuestDB**

- *Nanoseconds need `TIMESTAMP_NS`.* A plain `TIMESTAMP` is microseconds and would
  truncate the ns feed, breaking as-of joins on same-microsecond events.
- *The as-of join is the native idiom.* With designated timestamps on both sides,
  `ASOF JOIN ... ON (symbol)` *is* `aj` — no time predicate needed. Q1 is the
  shortest of all the ports.
- *A range join is not.* The direct port of Q3 (spans + `start <= t < end`) is valid
  QuestDB SQL and correct on small data, but it plans as a hash join plus filter and
  hits the 60 s query timeout at benchmark scale. Rewriting Q3 as "for each (grid
  point, price level), as-of join the level's last event" asks the question QuestDB
  is built for and finishes in seconds — same answer, verified.
- *Missing general-SQL pieces*: no `arg_max` (Q3 packs `price·2³² + size` into one
  int64 and takes `max`), no exact percentile (Q4 computes it from `row_number()` /
  `count()` windows), no `FILTER` clause (`CASE` instead), and no window function
  nested inside an expression (materialize the `lag` one CTE up).

The practical reading: ClickHouse ports with a handful of semantic settings, and
QuestDB rewards rewriting into its time-series idioms (designated timestamps,
`SAMPLE BY`, `ASOF JOIN`) and punishes generic relational shapes.

## A decision checklist

1. **Storage:** Parquet with ZSTD. Integer-encode prices (ticks) and use ns-since-
   epoch timestamps — both compress better and remove float/timezone hazards.
2. **Layout:** partition by date only; sort each partition by `(symbol, time)`; keep
   Parquet statistics on. Revisit finer partitioning only when a full scan is your
   measured bottleneck.
3. **Row-group size:** large enough that files aren't fragmented, small enough that
   statistics prune usefully — a few hundred K rows is a reasonable start.
4. **Query engine:** DuckDB, using `ASOF JOIN` for prevailing-quote joins and window
   functions for book reconstruction. Load the working set into native tables for a
   session of repeated queries.
5. **Validate the migration:** reconstruct a known quantity (top-of-book from the
   MBP stream) and diff it against an independent source (your TBBO feed) before
   trusting any downstream number. That is the check this project builds its whole
   correctness story on, and it is the one that catches a silent schema misread.

## What this benchmark does not claim

It runs at laptop scale (tens of MB to ~1 GB), single-node, on synthetic data. It
does **not** measure multi-node scaling, real-feed messiness (gaps, corrections,
auction prints), or the many-TB regime where directory pruning finally earns its
overhead. The numbers here establish that the open stack is *credible and often
faster* on the canonical queries at research-session scale, and that the layout
choices matter more than the engine — not that it wins every production workload.
