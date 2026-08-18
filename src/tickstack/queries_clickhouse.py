"""The five canonical quant queries, in ClickHouse SQL.

These are the ClickHouse SQL equivalents of the queries defined in queries_duckdb.py.
"""

from __future__ import annotations

# End sentinel for a book level that is still live at the end of the data.
_END_SENTINEL = 9_223_372_036_854_775_807
SESSION_NS = 86_400_000_000_000


def q1_asof_trades_quotes_sql() -> str:
    return """
    SELECT
        t.seq, t.ts, t.symbol, t.price AS trade_px, t.size AS trade_sz, t.aggressor,
        q.bid_px, q.bid_sz, q.ask_px, q.ask_sz
    FROM trades t
    ASOF LEFT JOIN quotes q
      ON t.symbol = q.symbol
     AND t.ts >= q.ts
    ORDER BY t.symbol, t.ts, t.seq
    """


def q2_vwap_bars_sql(bar_ns: int) -> str:
    return f"""
    SELECT
        symbol,
        intDiv(ts, {bar_ns}) AS bar,
        intDiv(ts, {bar_ns}) * {bar_ns} AS bar_start_ns,
        sum(toBigInt(price) * size) AS pv,
        sum(size) AS volume,
        toDouble(sum(toBigInt(price) * size)) / sum(size) AS vwap_ticks,
        argMin(price, seq) AS open_px,
        max(price) AS high_px,
        min(price) AS low_px,
        argMax(price, seq) AS close_px,
        count(*) AS n_trades
    FROM trades
    GROUP BY symbol, bar
    ORDER BY symbol, bar
    """


def q3_book_reconstruction_sql(bar_ns: int) -> str:
    return f"""
    WITH running AS (
        SELECT
            date, symbol, side, price, seq, ts,
            sum(delta) OVER (
                PARTITION BY date, symbol, side, price ORDER BY seq
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS level_size
        FROM book_events
    ),
    spans AS (
        SELECT
            date, symbol, side, price, level_size,
            ts AS start_ts,
            coalesce(
                lead(ts) OVER (PARTITION BY date, symbol, side, price ORDER BY seq),
                {_END_SENTINEL}
            ) AS end_ts
        FROM running
    )
    SELECT
        g.symbol,
        g.sample_ts,
        maxIf(s.price, s.side = 'B') AS bid_px,
        argMaxIf(s.level_size, s.price, s.side = 'B') AS bid_sz,
        minIf(s.price, s.side = 'A') AS ask_px,
        argMinIf(s.level_size, s.price, s.side = 'A') AS ask_sz
    FROM sample_grid g
    JOIN (SELECT * FROM spans WHERE level_size > 0) s
      ON s.symbol = g.symbol
     AND s.date = intDiv(g.sample_ts, {SESSION_NS})
     AND s.start_ts <= g.sample_ts
     AND g.sample_ts < s.end_ts
    GROUP BY g.symbol, g.sample_ts
    ORDER BY g.symbol, g.sample_ts
    """


def q4_event_study_sql(horizon_ns: int, top_frac: float) -> str:
    return f"""
    WITH mids AS (
        SELECT ts, symbol, (bid_px + ask_px) / 2.0 AS mid FROM quotes
    ),
    thresh AS (
        SELECT symbol, quantile({1.0 - top_frac})(size) AS cut
        FROM trades GROUP BY symbol
    ),
    events AS (
        SELECT t.ts, t.symbol, t.size
        FROM trades t JOIN thresh th ON t.symbol = th.symbol
        WHERE t.size >= th.cut
    ),
    entry AS (
        SELECT e.ts, e.symbol, e.size, m.mid AS entry_mid
        FROM events e
        ASOF LEFT JOIN mids m
          ON e.symbol = m.symbol AND e.ts >= m.ts
    ),
    bracketed AS (
        SELECT en.ts, en.symbol, en.size, en.entry_mid, m.mid AS exit_mid
        FROM entry en
        ASOF LEFT JOIN mids m
          ON en.symbol = m.symbol AND (en.ts + {horizon_ns}) >= m.ts
    )
    SELECT
        symbol,
        count(*) AS n_events,
        avg((exit_mid - entry_mid) / entry_mid * 10000.0) AS mean_fwd_bps
    FROM bracketed
    WHERE entry_mid IS NOT NULL AND exit_mid IS NOT NULL AND entry_mid > 0
    GROUP BY symbol
    ORDER BY symbol
    """


def q5_cross_symbol_corr_sql(bar_ns: int) -> str:
    return f"""
    WITH binned AS (
        SELECT symbol, ts, (bid_px + ask_px) / 2.0 AS mid_px, intDiv(ts, {bar_ns}) AS bar
        FROM quotes
    ),
    bar_mid AS (
        SELECT
            symbol,
            bar,
            argMax(mid_px, ts) AS mid
        FROM binned
        GROUP BY symbol, bar
    ),
    rets AS (
        SELECT symbol, bar,
               log(mid / lagInFrame(mid) OVER (PARTITION BY symbol ORDER BY bar)) AS r
        FROM bar_mid
    )
    SELECT a.symbol AS sym_a, b.symbol AS sym_b,
           corr(a.r, b.r) AS correlation,
           count(*)       AS n_bars
FROM rets a JOIN rets b
  ON a.bar = b.bar AND a.symbol < b.symbol
    WHERE a.r IS NOT NULL AND b.r IS NOT NULL
    GROUP BY sym_a, sym_b
    ORDER BY sym_a, sym_b
    """
