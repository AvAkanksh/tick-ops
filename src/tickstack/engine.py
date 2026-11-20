"""DuckDB connection setup and the shared sample grid.

The queries reference three views — ``trades``, ``quotes``, ``book_events`` — and Q3
also references a ``sample_grid(symbol, sample_ts)`` relation. This module wires a
layout's Parquet files to those names so a query is written once and run against any
layout. It also builds the sample grid, which *must* be identical for DuckDB and the
pandas baseline or Q3's cross-check is meaningless — so both sides call
:func:`build_sample_grid` here.
"""

from __future__ import annotations

import contextlib

import duckdb
import pyarrow as pa

from .layouts import LayoutPaths
from .schema import NS_PER_MIN


def connect(threads: int | None = None, memory_limit: str | None = None) -> duckdb.DuckDBPyConnection:
    """A fresh in-memory DuckDB connection with optional resource caps. Benchmarks
    pass ``threads`` explicitly so single- vs multi-threaded runs are labeled."""
    con = duckdb.connect()
    # DuckDB rewrites an ASOF join whose left side is *estimated* at <= 64 rows into a
    # nested-loop join + arg_max. Hive-partition-pruned Parquet scans are estimated at
    # "~1 row" whatever they hold, so on the partitioned layouts the scoped
    # one-symbol-day as-of (14k trades x 26k quotes) took that path and ran ~40x
    # slower (1.2 s vs 30 ms; same rows) — a planner misestimate, not a layout cost.
    # Threshold 0 keeps the real ASOF operator everywhere; the single-file layout,
    # whose estimates are right, is unaffected.
    with contextlib.suppress(duckdb.Error):  # setting absent on older DuckDB
        con.execute("SET asof_loop_join_threshold = 0")
    if threads is not None:
        con.execute(f"PRAGMA threads={threads}")
    if memory_limit is not None:
        con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    return con


def register_layout(con: duckdb.DuckDBPyConnection, paths: LayoutPaths) -> None:
    """Create the ``trades``/``quotes``/``book_events`` views over a layout's files.

    Views (not tables) so DuckDB reads Parquet lazily and can push filters and
    projections into the scan — which is the behavior a real research query relies
    on and what the layout experiment is meant to exercise. ``hive_partitioning`` is
    turned on only for the Hive layouts so the partition columns reappear as
    regular columns (``date``, and for one layout ``symbol``)."""
    for table in ("trades", "quotes", "book_events"):
        glob = paths.table_glob(table)
        hive = "true" if paths.is_hive(table) else "false"
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning={hive})"
        )


def build_sample_grid(
    con: duckdb.DuckDBPyConnection, sample_ns: int = NS_PER_MIN
) -> pa.Table:
    """Build and register the ``sample_grid(symbol, sample_ts)`` relation used by Q3.

    The grid is the cross product of the distinct symbols with a regular time grid
    spanning the data, at ``sample_ns`` spacing. Returning it as Arrow lets the
    pandas baseline reconstruct at the *same* points. The grid is registered as a
    DuckDB view named ``sample_grid``."""
    # Distinct symbols and the [min_ts, max_ts] envelope, straight from the data.
    bounds = con.execute(
        "SELECT min(ts) AS lo, max(ts) AS hi FROM "
        "(SELECT ts FROM trades UNION ALL SELECT ts FROM quotes)"
    ).fetchall()[0]
    lo, hi = int(bounds[0]), int(bounds[1])
    symbols = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM quotes ORDER BY symbol"
    ).fetchall()]

    # Grid points aligned to multiples of sample_ns inside [lo, hi]. Start at the
    # first multiple >= lo so points are on a clean grid, not offset by lo.
    first = ((lo + sample_ns - 1) // sample_ns) * sample_ns
    grid_ts = list(range(first, hi + 1, sample_ns))

    syms: list[str] = []
    tss: list[int] = []
    for s in symbols:
        for t in grid_ts:
            syms.append(s)
            tss.append(t)
    grid = pa.table(
        {"symbol": pa.array(syms, pa.string()), "sample_ts": pa.array(tss, pa.int64())}
    )
    con.register("sample_grid_arrow", grid)
    con.execute("CREATE OR REPLACE VIEW sample_grid AS SELECT * FROM sample_grid_arrow")
    return grid


def load_native(
    src: duckdb.DuckDBPyConnection, threads: int | None = None
) -> duckdb.DuckDBPyConnection:
    """Copy the three views from ``src`` into a fresh connection as native DuckDB
    tables, and return that connection.

    This is the in-memory analogue of what pandas does when it holds DataFrames: the
    data is loaded once into the engine's own format, then queried repeatedly. It
    matters for correctness of the *benchmark*, not just speed — querying Arrow tables
    registered as views routes through the Arrow scan adapter, and DuckDB's ASOF join
    over that adapter degrades to a quadratic plan. Native tables are what a research
    session actually queries."""
    dst = connect(threads=threads)
    for table in ("trades", "quotes", "book_events"):
        arrow_tbl = src.sql(f"SELECT * FROM {table}").to_arrow_table()
        dst.register(f"_src_{table}", arrow_tbl)
        dst.execute(f"CREATE TABLE {table} AS SELECT * FROM _src_{table}")
        dst.unregister(f"_src_{table}")
    return dst


def load_tables_arrow(con: duckdb.DuckDBPyConnection) -> dict[str, pa.Table]:
    """Materialize the three views as Arrow tables — the zero-copy handoff the
    pandas baseline consumes. This is the 'Arrow-based handoff to Python research
    code' the project promises: DuckDB reads Parquet, hands Arrow to pandas with no
    intermediate serialization."""
    return {
        "trades": con.sql("SELECT * FROM trades").to_arrow_table(),
        "quotes": con.sql("SELECT * FROM quotes").to_arrow_table(),
        "book_events": con.sql("SELECT * FROM book_events").to_arrow_table(),
    }
