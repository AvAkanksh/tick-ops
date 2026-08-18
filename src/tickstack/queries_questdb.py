"""The five canonical quant queries, in QuestDB SQL.

These are the QuestDB SQL equivalents of the queries defined in queries_duckdb.py.
"""

from __future__ import annotations

# End sentinel for a book level that is still live at the end of the data.
_END_SENTINEL = 9_223_372_036_854_775_807
SESSION_NS = 86_400_000_000_000


def q1_asof_trades_quotes_sql() -> str:
    # QuestDB uses ASOF JOIN with timestamps automatically matched and symbol matched in the ON clause.
    return """
    SELECT
        t.seq, t.ts, t.symbol, t.price AS trade_px, t.size AS trade_sz, t.aggressor,
        q.bid_px, q.bid_sz, q.ask_px, q.ask_sz
    FROM trades t
    ASOF JOIN quotes q ON (symbol)
    ORDER BY t.symbol, t.ts, t.seq
    """


def q2_vwap_bars_sql(bar_min: int) -> str:
    # QuestDB uses SAMPLE BY for time-bucketed queries.
    return f"""
    SELECT
        symbol,
        timestamp_floor('{bar_min}m', ts) AS bar_start,
        sum(price * size) AS pv,
        sum(size) AS volume,
        sum(price * size) / sum(size) AS vwap_ticks,
        first(price) AS open_px,
        max(price) AS high_px,
        min(price) AS low_px,
        last(price) AS close_px,
        count(*) AS n_trades
    FROM trades
    SAMPLE BY {bar_min}m ALIGN TO CALENDAR
    """


def q3_book_reconstruction_sql(session_ns: int = SESSION_NS) -> str:
    # Reconstructing MBP book in QuestDB uses timestamps and custom aggregation over temporal windows.
    return """
    WITH running AS (
        SELECT
            symbol, side, price, seq, ts,
            sum(delta) OVER (
                PARTITION BY symbol, side, price ORDER BY seq
            ) AS level_size
        FROM book_events
    ),
    spans AS (
        SELECT
            symbol, side, price, level_size,
            ts AS start_ts,
            coalesce(
                lead(ts) OVER (PARTITION BY symbol, side, price ORDER BY seq),
                9223372036854775807
            ) AS end_ts
        FROM running
        WHERE level_size > 0
    )
    SELECT
        g.symbol,
        g.sample_ts,
        max(s.price) FILTER (WHERE s.side = 'B') AS bid_px,
        last(s.level_size) FILTER (WHERE s.side = 'B') AS bid_sz,
        min(s.price) FILTER (WHERE s.side = 'A') AS ask_px,
        last(s.level_size) FILTER (WHERE s.side = 'A') AS ask_sz
    FROM sample_grid g
    JOIN spans s
      ON s.symbol = g.symbol
     AND s.start_ts <= g.sample_ts
     AND g.sample_ts < s.end_ts
    GROUP BY g.symbol, g.sample_ts
    ORDER BY g.symbol, g.sample_ts
    """


def q4_event_study_sql(horizon_ns: int) -> str:
    # QuestDB event study using windowed quantiles and asof joins.
    return f"""
    WITH mids AS (
        SELECT ts, symbol, (bid_px + ask_px) / 2.0 AS mid FROM quotes
    ),
    events AS (
        SELECT t.ts, t.symbol, t.size
        FROM trades t
        WHERE t.size >= 10000 -- threshold for large trades
    ),
    entry AS (
        SELECT e.ts, e.symbol, e.size, m.mid AS entry_mid
        FROM events e
        ASOF JOIN mids m ON (symbol)
    ),
    bracketed AS (
        SELECT en.ts, en.symbol, en.size, en.entry_mid, m.mid AS exit_mid
        FROM entry en
        ASOF JOIN mids m ON (symbol) -- exit timestamp matches
    )
    SELECT
        symbol,
        count(*) AS n_events,
        avg((exit_mid - entry_mid) / entry_mid * 10000.0) AS mean_fwd_bps
    FROM bracketed
    GROUP BY symbol
    ORDER BY symbol
    """


def q5_cross_symbol_corr_sql(bar_min: int) -> str:
    return f"""
    WITH bar_mid AS (
        SELECT
            symbol,
            timestamp_floor('{bar_min}m', ts) AS bar,
            last((bid_px + ask_px) / 2.0) AS mid
        FROM quotes
        SAMPLE BY {bar_min}m ALIGN TO CALENDAR
    ),
    rets AS (
        SELECT symbol, bar,
               log(mid / lag(mid) OVER (PARTITION BY symbol ORDER BY bar)) AS r
        FROM bar_mid
    )
    SELECT a.symbol AS sym_a, b.symbol AS sym_b,
           corr(a.r, b.r) AS correlation,
           count(*)       AS n_bars
    FROM rets a JOIN rets b ON (bar)
    WHERE a.symbol < b.symbol AND a.r IS NOT NULL AND b.r IS NOT NULL
    GROUP BY sym_a, sym_b
    ORDER BY sym_a, sym_b
    """
