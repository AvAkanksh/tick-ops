"""Adapters for the two server engines: ClickHouse and QuestDB.

Both run as real servers (see ``docker-compose.yml``); nothing here estimates or
scales a timing. Each adapter does three things:

1. **load** the same Arrow tables the DuckDB/pandas paths use into the server's own
   storage (ClickHouse ``MergeTree``; QuestDB WAL tables with a designated
   nanosecond timestamp), timed end-to-end from Arrow in Python to "rows visible".
2. **run** each of the five queries and hand the result back as an Arrow table, so
   the same :func:`tickstack.verify.frames_match` oracle that gates DuckDB vs pandas
   also gates every server result.
3. report the **server version** for provenance.

What a server timing includes, stated plainly: the client round-trip, server-side
execution, and deserializing the result into Arrow in this process. DuckDB and
Polars are in-process and pay none of the wire cost, so for large results (Q1
returns one row per trade) the server number is an upper bound on engine compute,
not a pure compute figure. The report says so next to the table.
"""

from __future__ import annotations

import io
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pyarrow as pa
import pyarrow.csv as pacsv

from . import queries_clickhouse as qch
from . import queries_questdb as qqdb

CH_HOST = os.environ.get("TICKSTACK_CH_HOST", "localhost")
CH_PORT = int(os.environ.get("TICKSTACK_CH_PORT", "18123"))
QDB_HOST = os.environ.get("TICKSTACK_QDB_HOST", "localhost")
QDB_HTTP_PORT = int(os.environ.get("TICKSTACK_QDB_HTTP_PORT", "19000"))


# ---------------------------------------------------------------------------
# ClickHouse
# ---------------------------------------------------------------------------

_CH_DDL = {
    "trades": """
        CREATE TABLE trades (
            seq Int64, ts Int64, symbol LowCardinality(String), price Int32,
            size Int32, aggressor LowCardinality(String), date Int32
        ) ENGINE = MergeTree ORDER BY (symbol, ts, seq)
    """,
    "quotes": """
        CREATE TABLE quotes (
            ts Int64, symbol LowCardinality(String), bid_px Int32, bid_sz Int32,
            ask_px Int32, ask_sz Int32, date Int32
        ) ENGINE = MergeTree ORDER BY (symbol, ts)
    """,
    "book_events": """
        CREATE TABLE book_events (
            seq Int64, ts Int64, symbol LowCardinality(String),
            side LowCardinality(String), price Int32, delta Int32, date Int32
        ) ENGINE = MergeTree ORDER BY (symbol, date, seq)
    """,
    "sample_grid": """
        CREATE TABLE sample_grid (symbol LowCardinality(String), sample_ts Int64)
        ENGINE = MergeTree ORDER BY (symbol, sample_ts)
    """,
}


class ClickHouseEngine:
    """A ClickHouse server reached over HTTP via ``clickhouse-connect``."""

    name = "clickhouse"

    def __init__(
        self,
        threads: int = 4,
        host: str = CH_HOST,
        port: int = CH_PORT,
        database: str = "tickstack",
    ):
        import clickhouse_connect

        self.threads = threads
        admin = clickhouse_connect.get_client(host=host, port=port)
        admin.command(f"CREATE DATABASE IF NOT EXISTS {database}")
        admin.close()
        # join_use_nulls: an unmatched LEFT/ASOF row yields NULL (SQL semantics),
        # not ClickHouse's default of the type's zero value — the oracle requires it.
        self.client = clickhouse_connect.get_client(
            host=host,
            port=port,
            database=database,
            settings={"join_use_nulls": 1, "max_threads": threads},
        )

    def version(self) -> str:
        return str(self.client.command("SELECT version()"))

    def load(self, tables: dict[str, pa.Table], grid: pa.Table) -> float:
        """(Re)create and fill the four tables; return wall-clock load ms. The
        timer covers insert plus ``OPTIMIZE ... FINAL`` (merge into one part), which
        is what a loaded-once research table looks like before querying."""
        src = {**tables, "sample_grid": grid}
        for name in _CH_DDL:
            self.client.command(f"DROP TABLE IF EXISTS {name}")
            self.client.command(_CH_DDL[name])
        t0 = time.perf_counter()
        for name, tbl in src.items():
            self.client.insert_arrow(name, tbl.select([f.name for f in tbl.schema]))
            self.client.command(f"OPTIMIZE TABLE {name} FINAL")
        return (time.perf_counter() - t0) * 1000.0

    def query(self, sql: str) -> pa.Table:
        return self.client.query_arrow(sql, use_strings=True)

    def runners(self) -> dict:
        sqls = {
            "q1_asof": qch.q1_asof_trades_quotes_sql(),
            "q2_vwap": qch.q2_vwap_bars_sql(),
            "q3_book": qch.q3_book_reconstruction_sql(),
            "q4_event": qch.q4_event_study_sql(),
            "q5_corr": qch.q5_cross_symbol_corr_sql(),
        }
        return {q: (lambda s=s: self.query(s)) for q, s in sqls.items()}

    def close(self) -> None:
        self.client.close()


# ---------------------------------------------------------------------------
# QuestDB
# ---------------------------------------------------------------------------

# Designated timestamp is TIMESTAMP_NS so the generator's nanosecond ``ts`` is kept
# exactly (a microsecond TIMESTAMP would truncate and break the as-of joins).
_QDB_DDL = {
    "trades": """
        CREATE TABLE trades (
            seq LONG, ts TIMESTAMP_NS, symbol SYMBOL, price INT, size INT,
            aggressor SYMBOL, date INT
        ) TIMESTAMP(ts) PARTITION BY DAY WAL
    """,
    "quotes": """
        CREATE TABLE quotes (
            ts TIMESTAMP_NS, symbol SYMBOL, bid_px INT, bid_sz INT, ask_px INT,
            ask_sz INT, date INT
        ) TIMESTAMP(ts) PARTITION BY DAY WAL
    """,
    "book_events": """
        CREATE TABLE book_events (
            seq LONG, ts TIMESTAMP_NS, symbol SYMBOL, side SYMBOL, price INT,
            delta INT, date INT
        ) TIMESTAMP(ts) PARTITION BY DAY WAL
    """,
    "sample_grid": """
        CREATE TABLE sample_grid (symbol SYMBOL, sample_ts TIMESTAMP_NS)
        TIMESTAMP(sample_ts) PARTITION BY DAY WAL
    """,
}

_QDB_SORT = {
    "trades": [("ts", "ascending"), ("seq", "ascending")],
    "quotes": [("ts", "ascending"), ("symbol", "ascending")],
    "book_events": [("ts", "ascending"), ("seq", "ascending")],
    "sample_grid": [("sample_ts", "ascending"), ("symbol", "ascending")],
}
_QDB_TS = {"trades": "ts", "quotes": "ts", "book_events": "ts", "sample_grid": "sample_ts"}
_QDB_SYMBOLS = {
    "trades": ["symbol", "aggressor"],
    "quotes": ["symbol"],
    "book_events": ["symbol", "side"],
    "sample_grid": ["symbol"],
}


class QuestDBEngine:
    """A QuestDB server: DDL and queries over its HTTP REST API, ingestion over
    InfluxDB Line Protocol (the officially recommended bulk path)."""

    name = "questdb"

    def __init__(self, host: str = QDB_HOST, http_port: int = QDB_HTTP_PORT):
        self.host = host
        self.http_port = http_port
        self.base = f"http://{host}:{http_port}"

    # -- HTTP helpers -------------------------------------------------------
    def _get(self, path: str, params: dict) -> bytes:
        url = f"{self.base}{path}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=600) as r:
            return r.read()

    def exec(self, sql: str) -> dict:
        import json

        out = json.loads(self._get("/exec", {"query": sql}))
        if "error" in out:
            raise RuntimeError(f"QuestDB: {out['error']} (at {out.get('position')})\n{sql}")
        return out

    def version(self) -> str:
        b = self.exec("SELECT build()")["dataset"][0][0]
        return b.split(",")[0].replace("Build Information: QuestDB ", "")

    def load(self, tables: dict[str, pa.Table], grid: pa.Table) -> float:
        """(Re)create and fill the four tables over ILP; return wall-clock load ms.
        The timer stops only once every row is visible to queries (WAL applied), so
        it is directly comparable to the other engines' "ready to query" cost."""
        from questdb import Sender

        src = {**tables, "sample_grid": grid}
        for name in _QDB_DDL:
            self.exec(f"DROP TABLE IF EXISTS {name}")
            self.exec(_QDB_DDL[name])
        t0 = time.perf_counter()
        conf = f"http::addr={self.host}:{self.http_port};auto_flush_rows=200000;"
        with Sender.from_conf(conf) as sender:
            for name, tbl in src.items():
                ts_col = _QDB_TS[name]
                df = tbl.sort_by(_QDB_SORT[name]).to_pandas()
                df[ts_col] = df[ts_col].astype("datetime64[ns]")
                for c in _QDB_SYMBOLS[name]:
                    df[c] = df[c].astype("category")
                sender.dataframe(df, table_name=name, symbols=_QDB_SYMBOLS[name], at=ts_col)
            sender.flush()
        for name, tbl in src.items():
            self._wait_rows(name, tbl.num_rows)
        return (time.perf_counter() - t0) * 1000.0

    def _wait_rows(self, table: str, n: int, timeout_s: float = 300.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            got = self.exec(f"SELECT count() FROM {table}")["dataset"][0][0]
            if got == n:
                return
            time.sleep(0.05)
        raise TimeoutError(f"QuestDB {table}: {got} rows visible, expected {n}")

    def query(self, sql: str) -> pa.Table:
        """Run ``sql`` and return Arrow, via the CSV export endpoint (QuestDB's
        bulk result path; the JSON endpoint is paginated and slower)."""
        try:
            body = self._get("/exp", {"query": sql})
        except urllib.error.HTTPError as e:  # pragma: no cover - surfaced to caller
            import json

            try:
                info = json.loads(e.read())
                msg = f"{info.get('error')} (at {info.get('position')})"
            except Exception:
                msg = "unparseable error body"
            raise RuntimeError(f"QuestDB /exp {e.code}: {msg}\n{sql}") from None
        if not body.strip():
            return pa.table({})
        return pacsv.read_csv(
            io.BytesIO(body),
            convert_options=pacsv.ConvertOptions(strings_can_be_null=False),
        )

    def runners(self) -> dict:
        sqls = {
            "q1_asof": qqdb.q1_asof_trades_quotes_sql(),
            "q2_vwap": qqdb.q2_vwap_bars_sql(),
            "q3_book": qqdb.q3_book_reconstruction_sql(),
            "q4_event": qqdb.q4_event_study_sql(),
            "q5_corr": qqdb.q5_cross_symbol_corr_sql(),
        }
        return {q: (lambda s=s: self.query(s)) for q, s in sqls.items()}

    def close(self) -> None:
        pass


def available(engine_name: str) -> bool:
    """True if the named server answers a ping — lets tests and the benchmark skip
    cleanly (and say so) instead of failing when Docker isn't running."""
    try:
        if engine_name == "clickhouse":
            urllib.request.urlopen(f"http://{CH_HOST}:{CH_PORT}/ping", timeout=2).read()
        elif engine_name == "questdb":
            urllib.request.urlopen(
                f"http://{QDB_HOST}:{QDB_HTTP_PORT}/exec?query=select%201", timeout=2
            ).read()
        else:
            return False
        return True
    except Exception:
        return False
