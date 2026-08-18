# Schema and partitioning design

This document is the "documented schema/partitioning design" the project promises.
It states the on-disk contract, the reasoning behind each decision, and the layout
experiments — including the ones whose result was *not* what intuition predicts.

## The three tables

The dataset models a continuous, integer-priced equity-style market. Three streams,
consistent by construction (see `docs/GENERATOR.md`):

### `trades` — executed prints

| column | type | meaning |
|---|---|---|
| `seq` | int64 | global monotonic sequence; the tie-break within a nanosecond |
| `ts` | int64 | nanoseconds since session open |
| `symbol` | string | instrument |
| `price` | int32 | execution price, in ticks |
| `size` | int32 | shares |
| `aggressor` | string | `'B'` = buyer lifted the ask, `'A'` = seller hit the bid |
| `date` | int32 | trading-day index; the partition key |

### `quotes` — top of book (TBBO)

One row each time the best bid or offer changes (price or size). This is the
generator's own record of the book top, and the oracle the reconstruction query
(Q3) is checked against.

| column | type | meaning |
|---|---|---|
| `ts` | int64 | ns since session open |
| `symbol` | string | instrument |
| `bid_px`, `bid_sz` | int32 | best bid price (ticks) and size |
| `ask_px`, `ask_sz` | int32 | best ask price (ticks) and size |
| `date` | int32 | partition key |

### `book_events` — market-by-price increments

One row per change to resting size at a price level, as a signed `delta`. The live
size at a level is the running sum of its deltas in `(ts, seq)` order; a level is
present while that sum is positive. Replaying these reconstructs the full book.

| column | type | meaning |
|---|---|---|
| `seq` | int64 | global monotonic sequence; replay order |
| `ts` | int64 | ns since session open |
| `symbol` | string | instrument |
| `side` | string | `'B'` bid / `'A'` ask |
| `price` | int32 | level price, in ticks |
| `delta` | int32 | signed size change at the level |
| `date` | int32 | partition key |

## Two load-bearing type decisions

**Prices are integers (ticks), not floats.** The book is keyed on price level, and
reconstruction groups by `(symbol, side, price)`. Floating-point dollars as `GROUP
BY` or dict keys are a correctness hazard — `0.1 + 0.2 != 0.3`, so two economically
equal prices can hash apart. Prices live on disk as `int32` tick counts; one tick is
`TICK_SIZE` (one cent). Dollar values are derived only at the edges (VWAP output,
returns, display), never used as keys. This also compresses better: small integers
with a narrow range dictionary-encode tightly.

**Timestamps are `int64` nanoseconds since session open.** Nanosecond epochs are
what real feeds carry (ITCH, Databento), they sort and range-join with no timezone
ambiguity, and they keep the as-of joins exact. Measuring from session open means
day *N* starts at `N * SESSION_NS`, so multi-day data sorts globally by `ts` and the
day index is recoverable as `ts // SESSION_NS` — which Q3 uses to scope
reconstruction per day.

**Why `seq` exists.** An aggressive order that sweeps several price levels produces
several prints and several book events *at the same nanosecond*. `ts` alone
therefore does not order events, which makes "the last trade in a bar" (Q2 close) or
row-alignment (Q1) ambiguous. `seq` is a single global monotonic counter shared by
trades and book events, so every event has a unique, stable order — exactly why real
feeds carry a sequence number. The generator tie-breaks on `seq`; so do the queries.

## The four layouts

All four hold the *same rows* (same values, verified by `test_layouts_hold_identical_data`),
differing only in physical organization. The `row_group_size` *parameter* is held
constant (128Ki rows) and ZSTD is used throughout, so codec is not the variable. One
honest caveat: partitioning splits the data into smaller files, and a file shorter
than the row-group target ends up with fewer, smaller row groups — so the effective
row-group granularity is *not* identical across layouts (`single` has ~131k-row
groups; `by_date_symbol`'s tiny files have ~4k-row groups). That granularity
difference is part of what the layout comparison measures, not a confound to wish
away: it is precisely why over-partitioning hurts.

| layout | organization | intent |
|---|---|---|
| `single` | one Parquet file per table | the naive default |
| `by_date` | Hive-partitioned `date=` | prune whole days |
| `by_date_symbol` | Hive-partitioned `date=/symbol=` | prune to one symbol-day |
| `sorted` | partitioned by `date`, each partition sorted by `(symbol, ts)`, statistics on | large files that still prune by symbol via row-group min/max |

## What the layout experiment actually found

The intuitive story — "partition by symbol so a single-symbol query reads only that
symbol" — is **wrong at laptop scale**, and the benchmark says so rather than hiding
it. See `results/partition_study.md` for the sweep; the mechanism:

- `by_date_symbol` creates one file per symbol-day. At a 5–30 day, 8-symbol dataset
  those files are a few hundred KB each and number in the hundreds. Per-file open and
  metadata cost then dominates, so the scoped-query latency of this layout *rises*
  with dataset size — the small-files anti-pattern, reproduced live.
- `single` stays low and shows no growth trend across the swept range. It is *not*
  paying a full-table scan: because the generator writes day-by-day, `date` is
  monotonic in row order, so Parquet row-group statistics let DuckDB skip most row
  groups for a single-day predicate even inside one file (measured: 12 of 13 row
  groups skipped at 15 days). This is the real lesson — statistics-based row-group
  skipping does most of the pruning, and a directory per symbol-day mostly adds file
  overhead on top.
- `sorted` keeps files large (one per day, not per symbol-day) while clustering rows
  by symbol so row-group statistics prune by symbol *within* a file. This is the
  layout that scales: it prunes by statistics without the small-files penalty.

The migration takeaway: partitioning is a **scale-dependent** decision, and
over-partitioning is a common, real mistake. Directory partition-pruning only pays
for its per-file overhead once a full scan is expensive enough — which, for tick
data, is at the many-GB scale a production store reaches, not a laptop benchmark.
`sorted` is the layout that behaves well across the range, which is why it is the
default the migration guide recommends.
