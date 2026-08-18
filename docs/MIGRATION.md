# Migrating tick research from kdb+ to the open columnar stack

The audience for this is a team running quant research on kdb+ (or a bespoke
in-memory store) and weighing a move to Arrow + Parquet + DuckDB. It is written from
what this benchmark actually measured, not from vendor copy — including the places
where the open stack's naive setup is a trap.

## What transfers cleanly

**The signature query — the as-of join — is a first-class operation.** The single
biggest worry migrating off kdb+ is `aj`, the as-of join that attaches the
prevailing quote to each trade. DuckDB has `ASOF JOIN` as native syntax with a
purpose-built operator; Q1 is a few lines of SQL and runs in ~90 ms in-memory
(~130 ms cold from Parquet) on ~half a million trades — and it beats a `merge_asof`
pandas baseline by ~5–6× (see `results/benchmark.md`). This is the query that used
to justify a proprietary engine, and it is now table stakes in an open one.

**Columnar aggregation is fast and expressive.** VWAP bars (Q2), event studies (Q4),
and cross-symbol correlation (Q5) are ordinary `GROUP BY` / window / self-join SQL,
and DuckDB beats a vectorized pandas baseline on every one (see
`results/benchmark.md`). No bespoke primitives required.

**Arrow is the zero-copy handoff.** DuckDB reads Parquet and hands results to Python
research code as Arrow with no serialization step; `to_pandas()` on the result is
milliseconds. The "query in SQL, analyze in Python" workflow has no impedance layer.

## Where the naive migration goes wrong

**Do not over-partition.** The intuitive Hive layout — `date=/symbol=` — feels right
("a single-symbol query reads only that symbol") and is a documented trap. It
produces one file per symbol-day; at realistic symbol counts that is thousands of
small files, and per-file open/metadata overhead then dominates. In this benchmark
`by_date_symbol` is the *slowest* layout for a scoped query and gets worse as data
grows (`results/partition_study.md`). Partition coarsely — by date — and rely on
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
