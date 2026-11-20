"""The five canonical quant queries, in QuestDB SQL.

Same logical answers as :mod:`tickstack.queries_duckdb`; every result is diffed
against the DuckDB/pandas oracle before it is timed. QuestDB is a time-series
database first, so the ports lean on its native idioms where they exist and work
around missing general-SQL pieces where they don't:

* **Designated timestamp.** Every table has a ``TIMESTAMP_NS`` designated
  timestamp. ``ASOF JOIN ... ON (symbol)`` then means exactly "last row with
  ``right.ts <= left.ts`` for the same symbol" — no explicit time predicate. Output
  timestamps are cast to ``long`` so they compare as the same int64 nanoseconds the
  other engines return.
* **Time bars use ``SAMPLE BY``.** Q2/Q5 bucket with ``SAMPLE BY <n>s ALIGN TO
  CALENDAR``; because the data's epoch is 1970-01-01, a calendar-aligned bucket
  start is exactly ``(ts // bar_ns) * bar_ns``. ``first``/``last`` inside
  ``SAMPLE BY`` follow designated-timestamp order, and rows are ingested sorted by
  ``(ts, seq)``, so they are the open/close trade.
* **No ``arg_max``/``arg_min``.** Q3 needs "the size at the best price". It packs
  ``price * 2^32 + size`` into one int64 and takes ``max``/``min`` of the packed key
  — exact, because both are non-negative int32 and a price is unique per side at
  any instant.
* **No exact percentile.** QuestDB ships only ``approx_percentile``. Q4's per-symbol
  cut is computed exactly with ``row_number()``/``count()`` windows and the same
  linear interpolation ``quantile_cont`` uses, so the event set matches DuckDB.
* **No ``FILTER`` clause.** Conditional aggregates use ``CASE``.
"""

from __future__ import annotations

from .schema import NS_PER_MIN, NS_PER_SEC, SESSION_NS

# End sentinel for a book level that is still live at the end of the data.
_END_SENTINEL = 9_223_372_036_854_775_807
_PACK = 4_294_967_296  # 2^32


def _sample_by(bar_ns: int) -> str:
    if bar_ns % NS_PER_SEC:
        raise ValueError("QuestDB SAMPLE BY port supports whole-second bars only")
    return f"{bar_ns // NS_PER_SEC}s"


def q1_asof_trades_quotes_sql() -> str:
    return """
    SELECT
        t.seq AS seq, cast(t.ts AS long) AS ts, t.symbol AS symbol,
        t.price AS trade_px, t.size AS trade_sz, t.aggressor AS aggressor,
        q.bid_px AS bid_px, q.bid_sz AS bid_sz, q.ask_px AS ask_px, q.ask_sz AS ask_sz
    FROM trades t
    ASOF JOIN quotes q ON (symbol)
    ORDER BY symbol, ts, seq
    """


def q2_vwap_bars_sql(bar_ns: int = NS_PER_MIN) -> str:
    return f"""
    SELECT
        symbol,
        cast(ts AS long) / {bar_ns} AS bar,
        cast(ts AS long) AS bar_start_ns,
        pv, volume,
        cast(pv AS double) / volume AS vwap_ticks,
        open_px, high_px, low_px, close_px, n_trades
    FROM (
        SELECT
            ts, symbol,
            sum(cast(price AS long) * size) AS pv,
            sum(cast(size AS long)) AS volume,
            first(price) AS open_px,
            max(price) AS high_px,
            min(price) AS low_px,
            last(price) AS close_px,
            count() AS n_trades
        FROM trades
        SAMPLE BY {_sample_by(bar_ns)} ALIGN TO CALENDAR
    )
    ORDER BY symbol, bar
    """


def q3_book_reconstruction_sql() -> str:
    """Book reconstruction the QuestDB way: as-of lookups, not a range join.

    The direct port of the DuckDB query (running size + ``lead`` spans, then a
    ``start_ts <= sample_ts < end_ts`` range join) is valid QuestDB SQL and returns
    the right answer on small data, but QuestDB plans the range join as a hash join
    on ``(symbol, date)`` followed by a filter — ~750M candidate pairs at the
    benchmark scale — and hits the server's 60 s query timeout. That plan is kept as
    :func:`q3_book_reconstruction_rangejoin_sql` for the record.

    The idiomatic rewrite asks the question QuestDB is built to answer — *what was
    the state at time t?* — with ``ASOF JOIN``: cross each grid point with the price
    levels that exist that symbol-day, as-of-join each (grid point, level) probe to
    that level's last event at or before the sample time, and keep levels whose
    running size is positive. Same answer (diffed against the oracle), different
    algorithm."""
    return f"""
    WITH running AS (
        SELECT * FROM (
            SELECT
                ts, date, symbol, side, price,
                sum(delta) OVER (
                    PARTITION BY date, symbol, side, price ORDER BY seq
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS level_size
            FROM book_events
            ORDER BY ts, seq
        ) TIMESTAMP(ts)
    ),
    levels AS (
        SELECT DISTINCT date, symbol, side, price FROM book_events
    ),
    grid AS (
        SELECT sample_ts, symbol,
               cast(cast(sample_ts AS long) / {SESSION_NS} AS int) AS date
        FROM sample_grid
    ),
    probes AS (
        SELECT * FROM (
            SELECT g.sample_ts AS ts, g.symbol AS symbol, l.date AS date,
                   l.side AS side, l.price AS price
            FROM grid g
            JOIN levels l ON g.symbol = l.symbol AND g.date = l.date
            ORDER BY g.sample_ts
        ) TIMESTAMP(ts)
    ),
    state AS (
        SELECT p.ts AS ts, p.symbol AS symbol, p.side AS side,
               cast(p.price AS long) * {_PACK} + cast(r.level_size AS long) AS packed,
               r.level_size AS level_size
        FROM probes p
        ASOF JOIN running r ON (symbol, side, price, date)
    ),
    tops AS (
        SELECT symbol, cast(ts AS long) AS sample_ts,
               max(CASE WHEN side = 'B' AND level_size > 0 THEN packed END) AS bid_key,
               min(CASE WHEN side = 'A' AND level_size > 0 THEN packed END) AS ask_key
        FROM state
        GROUP BY symbol, sample_ts
    )
    SELECT
        symbol, sample_ts,
        bid_key / {_PACK} AS bid_px, bid_key % {_PACK} AS bid_sz,
        ask_key / {_PACK} AS ask_px, ask_key % {_PACK} AS ask_sz
    FROM tops
    WHERE bid_key IS NOT NULL OR ask_key IS NOT NULL
    ORDER BY symbol, sample_ts
    """


def q3_book_reconstruction_rangejoin_sql() -> str:
    """Direct port of the DuckDB range-join plan. Correct, but times out at
    benchmark scale in QuestDB 9.1 (see :func:`q3_book_reconstruction_sql`)."""
    return f"""
    WITH running AS (
        SELECT
            date, symbol, side, price,
            cast(ts AS long) AS start_ts,
            sum(delta) OVER (
                PARTITION BY date, symbol, side, price ORDER BY seq
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS level_size,
            lead(cast(ts AS long)) OVER (
                PARTITION BY date, symbol, side, price ORDER BY seq
            ) AS next_ts
        FROM book_events
    ),
    spans AS (
        SELECT
            date, symbol, side, start_ts,
            coalesce(next_ts, {_END_SENTINEL}) AS end_ts,
            cast(price AS long) * {_PACK} + cast(level_size AS long) AS packed
        FROM running
        WHERE level_size > 0
    ),
    grid AS (
        SELECT symbol, cast(sample_ts AS long) AS sample_ts,
               cast(cast(sample_ts AS long) / {SESSION_NS} AS int) AS date
        FROM sample_grid
    ),
    tops AS (
        SELECT
            g.symbol AS symbol, g.sample_ts AS sample_ts,
            max(CASE WHEN s.side = 'B' THEN s.packed END) AS bid_key,
            min(CASE WHEN s.side = 'A' THEN s.packed END) AS ask_key
        FROM grid g
        JOIN spans s
          ON g.symbol = s.symbol AND g.date = s.date
        WHERE s.start_ts <= g.sample_ts AND g.sample_ts < s.end_ts
        GROUP BY g.symbol, g.sample_ts
    )
    SELECT
        symbol, sample_ts,
        bid_key / {_PACK} AS bid_px, bid_key % {_PACK} AS bid_sz,
        ask_key / {_PACK} AS ask_px, ask_key % {_PACK} AS ask_sz
    FROM tops
    ORDER BY symbol, sample_ts
    """


def q4_event_study_sql(horizon_ns: int = 5 * NS_PER_SEC, top_frac: float = 0.02) -> str:
    p = 1.0 - top_frac
    return f"""
    WITH ranked AS (
        SELECT symbol, size,
               row_number() OVER (PARTITION BY symbol ORDER BY size) AS rn,
               count() OVER (PARTITION BY symbol) AS n
        FROM trades
    ),
    pos AS (
        SELECT symbol, size, rn, n,
               floor((n - 1) * {p}) AS lo,
               (n - 1) * {p} - floor((n - 1) * {p}) AS frac
        FROM ranked
    ),
    thresh AS (
        SELECT symbol,
               max(CASE WHEN rn = lo + 1 THEN size END) AS v_lo,
               max(CASE WHEN rn = least(lo + 2, n) THEN size END) AS v_hi,
               max(frac) AS frac
        FROM pos
        GROUP BY symbol
    ),
    events AS (
        SELECT * FROM (
            SELECT t.ts AS ts, t.symbol AS symbol
            FROM trades t
            JOIN thresh th ON (symbol)
            WHERE t.size >= th.v_lo + th.frac * (th.v_hi - th.v_lo)
            ORDER BY t.ts
        ) TIMESTAMP(ts)
    ),
    mids AS (
        SELECT ts, symbol, (bid_px + ask_px) / 2.0 AS mid FROM quotes
    ),
    entry AS (
        SELECT * FROM (
            SELECT e.ts + {horizon_ns} AS ts, e.symbol AS symbol, m.mid AS entry_mid
            FROM events e
            ASOF JOIN mids m ON (symbol)
        ) TIMESTAMP(ts)
    ),
    bracketed AS (
        SELECT en.symbol AS symbol, en.entry_mid AS entry_mid, m.mid AS exit_mid
        FROM entry en
        ASOF JOIN mids m ON (symbol)
    )
    SELECT
        symbol,
        count() AS n_events,
        avg((exit_mid - entry_mid) / entry_mid * 10000.0) AS mean_fwd_bps
    FROM bracketed
    WHERE entry_mid IS NOT NULL AND exit_mid IS NOT NULL AND entry_mid > 0
    GROUP BY symbol
    ORDER BY symbol
    """


def q5_cross_symbol_corr_sql(bar_ns: int = NS_PER_MIN) -> str:
    return f"""
    WITH bar_mid AS (
        SELECT ts, symbol, last((bid_px + ask_px) / 2.0) AS mid
        FROM quotes
        SAMPLE BY {_sample_by(bar_ns)} ALIGN TO CALENDAR
    ),
    lagged AS (
        SELECT symbol, ts, mid,
               lag(mid) OVER (PARTITION BY symbol ORDER BY ts) AS prev_mid
        FROM bar_mid
    ),
    rets AS (
        -- QuestDB cannot nest a window call inside an expression, so the lag is
        -- materialized one CTE up and the log-return taken here.
        SELECT cast(symbol AS string) AS symbol,
               cast(ts AS long) / {bar_ns} AS bar,
               ln(mid / prev_mid) AS r
        FROM lagged
    )
    SELECT a.symbol AS sym_a, b.symbol AS sym_b,
           corr(a.r, b.r) AS correlation,
           count() AS n_bars
    FROM rets a
    JOIN rets b ON a.bar = b.bar
    WHERE a.symbol < b.symbol AND a.r IS NOT NULL AND b.r IS NOT NULL
    GROUP BY a.symbol, b.symbol
    ORDER BY sym_a, sym_b
    """
