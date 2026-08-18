"""tickstack — a modern tick research stack (Arrow / Parquet / DuckDB).

A benchmark of the open columnar quant stack against a fair pandas baseline on five
canonical research queries, with the layout experiments and correctness cross-checks
that make the numbers trustworthy. See the package modules:

* :mod:`tickstack.schema` — integer-priced, ns-timestamped table schemas.
* :mod:`tickstack.generator` — a deterministic order-book simulator whose three
  streams are consistent by construction.
* :mod:`tickstack.layouts` — the four Parquet layout experiments.
* :mod:`tickstack.queries_duckdb` / :mod:`tickstack.queries_pandas` — the five
  queries on each engine, computing the same logical answer.
* :mod:`tickstack.verify` — the correctness checks that gate the benchmark.
* :mod:`tickstack.bench` — the timing harness.
"""

from __future__ import annotations

__version__ = "0.1.0"
