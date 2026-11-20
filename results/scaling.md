# Scaling study

Query latency vs dataset size on the `sorted` layout, 40,000 events/symbol/day, 5 timed runs (median). DuckDB in memory.

| Days | Trades | Size | Q1 as-of | Q2 VWAP | Q3 book | Q4 event | Q5 corr |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 112,503 | 8.4 MB | 53.1 ms | 8.5 ms | 182.6 ms | 40.4 ms | 11.8 ms |
| 2 | 226,613 | 16.8 MB | 76.1 ms | 9.5 ms | 370.3 ms | 67.3 ms | 12.9 ms |
| 5 | 564,379 | 42.0 MB | 124.9 ms | 14.6 ms | 834.9 ms | 137.8 ms | 23.7 ms |
| 10 | 1,131,150 | 83.8 MB | 238.2 ms | 21.3 ms | 1714.9 ms | 243.7 ms | 38.7 ms |
| 20 | 2,257,674 | 169.0 MB | 508.6 ms | 39.2 ms | 3423.5 ms | 479.0 ms | 72.7 ms |
