"""Parquet layout experiments.

Partitioning and sort order dominate query performance far more than engine choice,
which is exactly the thing this project sets out to show. Four layouts are defined,
holding the *same rows* (identical values) and differing only in physical
organization:

* ``single`` — one Parquet file per table. The naive default.
* ``by_date`` — Hive-partitioned by ``date=``. Prunes whole days for date-scoped
  queries but leaves each day's file unsorted beyond the generator's time order.
* ``by_date_symbol`` — Hive-partitioned by ``date=`` then ``symbol=``. Prunes to a
  single symbol-day; the file is tiny and already time-sorted within it.
* ``sorted`` — partitioned by date, and each partition **sorted by (symbol, ts)**
  and written with statistics, so DuckDB's row-group min/max lets it skip row
  groups for a symbol without a directory per symbol. This is the layout a real
  research store would use: it keeps files large (good for scans) while still
  pruning by symbol via statistics.

The point of holding four is the benchmark table: the same query runs against each
and the spread is the headline result. The four hold the *same rows* (same values —
verified by a test), not the same bytes: partitioning splits the data into smaller
files, so a partition shorter than ``row_group_size`` ends up with fewer, smaller row
groups. That granularity difference is part of what partitioning changes, not a
confound — it is why over-partitioning hurts.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .generator import DayBatch

LAYOUTS = ("single", "by_date", "by_date_symbol", "sorted")

# Compression + row-group size are held constant across layouts so the comparison
# isolates *organization*. ZSTD is the sensible default for a research store: it
# roughly halves Parquet size vs Snappy at read speeds that stay I/O-bound.
_COMPRESSION = "zstd"
_ROW_GROUP = 128 * 1024


@dataclass
class LayoutPaths:
    """Where a layout's three tables live. A path may be a file or a directory
    (Hive dataset); the query layer opens both via ``read_parquet(..., hive_partitioning)``."""

    root: Path
    kind: str

    def table_glob(self, table: str) -> str:
        if self.kind == "single":
            return str(self.root / f"{table}.parquet")
        # Hive datasets: a directory tree of part files.
        return str(self.root / table / "**" / "*.parquet")

    def is_hive(self, table: str) -> bool:
        # 'sorted' is also written as a Hive dataset (partitioned by date), so the
        # date column lives in the path, not the file, and must be recovered by
        # hive_partitioning=true on read.
        return self.kind in ("by_date", "by_date_symbol", "sorted")


class LayoutWriter:
    """Streams :class:`DayBatch` objects to disk in every requested layout.

    Writing is streaming (one day at a time) so that generating a large dataset
    never holds more than a day in memory. The ``sorted`` and ``single`` layouts
    accumulate references to per-day tables and concatenate at ``finish`` — cheap,
    because Arrow tables are zero-copy views over the same buffers.
    """

    def __init__(self, base: Path, layouts: Iterable[str] = LAYOUTS) -> None:
        self.base = Path(base)
        self.layouts = tuple(layouts)
        for lay in self.layouts:
            if lay not in LAYOUTS:
                raise ValueError(f"unknown layout {lay!r}; known: {LAYOUTS}")
        # Accumulators for layouts that write once at the end.
        self._acc: dict[tuple[str, str], list[pa.Table]] = {}

    def paths(self, layout: str) -> LayoutPaths:
        return LayoutPaths(root=self.base / layout, kind=layout)

    def _tables(self, batch: DayBatch) -> dict[str, pa.Table]:
        return {"trades": batch.trades, "quotes": batch.quotes, "book_events": batch.book_events}

    def write_day(self, batch: DayBatch) -> None:
        tables = self._tables(batch)
        for lay in self.layouts:
            for name, tbl in tables.items():
                if lay in ("single", "sorted"):
                    self._acc.setdefault((lay, name), []).append(tbl)
                elif lay == "by_date":
                    self._write_hive(lay, name, tbl, ["date"])
                elif lay == "by_date_symbol":
                    self._write_hive(lay, name, tbl, ["date", "symbol"])

    def finish(self) -> None:
        for (lay, name), parts in self._acc.items():
            full = pa.concat_tables(parts)
            if lay == "single":
                out = self.base / lay / f"{name}.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(
                    full, out, compression=_COMPRESSION, row_group_size=_ROW_GROUP
                )
            elif lay == "sorted":
                # Global sort by (symbol, ts) then partition by date. Sorting before
                # partitioning means each date's file is symbol-clustered and time-
                # ordered, so row-group statistics on symbol are tight.
                ts_col = "seq" if name == "book_events" else "ts"
                full = full.sort_by(
                    [("symbol", "ascending"), (ts_col, "ascending")]
                )
                self._write_hive(lay, name, full, ["date"], preserve_order=True)
        self._acc.clear()

    def _write_hive(
        self, layout: str, name: str, table: pa.Table, cols: list[str], preserve_order: bool = False
    ) -> None:
        root = self.base / layout / name
        pq.write_to_dataset(
            table,
            root_path=str(root),
            partition_cols=cols,
            compression=_COMPRESSION,
            row_group_size=_ROW_GROUP,
            existing_data_behavior="overwrite_or_ignore",
            # Keep row groups sorted as passed in for the 'sorted' layout.
            use_threads=not preserve_order,
        )


def build_all_layouts(base: Path, batches: Iterable[DayBatch], layouts=LAYOUTS) -> LayoutWriter:
    """Write every batch into every layout under ``base`` and finalize."""
    base = Path(base)
    if base.exists():
        shutil.rmtree(base)
    writer = LayoutWriter(base, layouts)
    for batch in batches:
        writer.write_day(batch)
    writer.finish()
    return writer


def dataset_size_bytes(paths: LayoutPaths, table: str) -> int:
    """Total on-disk bytes for a table under a layout (sum of part files)."""
    if paths.kind == "single":
        f = paths.root / f"{table}.parquet"
        return f.stat().st_size if f.exists() else 0
    root = paths.root / table
    return sum(p.stat().st_size for p in root.rglob("*.parquet"))
