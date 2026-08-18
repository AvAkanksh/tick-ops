# Scaling study

Query latency vs dataset size on the `sorted` layout, 40,000 events/symbol/day, 5 timed runs (median). DuckDB in memory.

| Days | Trades | Size | Q1 as-of | Q2 VWAP | Q3 book | Q4 event | Q5 corr |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 112,503 | 8.4 MB | 30.1 ms | 7.7 ms | 190.2 ms | 23.9 ms | 9.3 ms |
| 2 | 226,613 | 16.8 MB | 49.6 ms | 9.3 ms | 354.9 ms | 44.4 ms | 10.1 ms |
| 5 | 564,379 | 42.0 MB | 126.7 ms | 15.5 ms | 846.2 ms | 90.5 ms | 15.3 ms |
| 10 | 1,131,150 | 83.8 MB | 239.7 ms | 23.7 ms | 1612.3 ms | 179.4 ms | 29.0 ms |
| 20 | 2,257,674 | 169.0 MB | 488.1 ms | 40.4 ms | 3795.0 ms | 451.6 ms | 69.6 ms |
