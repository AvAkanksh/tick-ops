# Partition study — does partitioning help a scoped query?

Scoped as-of join (one symbol, one day) vs dataset size, per layout. Median of 7 runs after warmup. File count is shown because it is the mechanism: more, smaller files means more open/metadata overhead.

| Days | Layout | Latency | Files | Avg file |
|--:|---|--:|--:|--:|
| 5 | single | 24.3 ms | 3 | 16263 KB |
| 5 | by_date | 18.7 ms | 15 | 3199 KB |
| 5 | by_date_symbol | 24.1 ms | 120 | 361 KB |
| 5 | sorted | 17.3 ms | 15 | 2864 KB |
| 15 | single | 26.4 ms | 3 | 48701 KB |
| 15 | by_date | 21.0 ms | 45 | 3215 KB |
| 15 | by_date_symbol | 42.4 ms | 360 | 362 KB |
| 15 | sorted | 18.4 ms | 45 | 2877 KB |
| 30 | single | 27.5 ms | 3 | 97604 KB |
| 30 | by_date | 27.8 ms | 90 | 3232 KB |
| 30 | by_date_symbol | 70.2 ms | 720 | 364 KB |
| 30 | sorted | 26.1 ms | 90 | 2894 KB |

**Reading it:** `by_date_symbol` creates one file per symbol-day, so its file count grows with the data and each file stays small — its scoped-query latency *rises* with scale (the small-files anti-pattern). `single` stays low and nearly flat across this range. Note that `single` is *not* paying a full-table scan here: because the generator writes day-by-day, `date` is monotonic in row order, so Parquet row-group min/max statistics let DuckDB skip most row groups for a single-day predicate even in one file. That is the real lesson — statistics-based row-group skipping does most of the pruning work, and a directory per symbol-day mostly buys file overhead. `sorted` keeps files large while clustering by symbol, so it prunes by statistics without the small-files penalty and is the layout that scales. At this scale, directory partition-pruning does not pay for its per-file overhead; the payoff arrives at the scale where even statistics-pruned scans dominate — the migration-guide point.
