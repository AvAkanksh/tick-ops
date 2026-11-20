"""The five canonical quant queries, in ClickHouse SQL.

Same logical answers as :mod:`tickstack.queries_duckdb`; each result is diffed
against the DuckDB/pandas oracle before it is timed. The translations are direct
ports, and every place ClickHouse semantics differ from DuckDB's is handled
explicitly — these are exactly the traps that silently produce *plausible but
wrong* numbers if you port SQL by eye:

* **Unmatched joins return zeros by default.** ClickHouse fills a missing ASOF/LEFT
  match with the column type's default (0), not NULL. The adapter sets
  ``join_use_nulls=1`` so a trade before its symbol's first quote gets NULL quote
  fields, as in DuckDB.
* **``quantile`` is approximate.** Plain ``quantile`` is reservoir-sampled; Q4's
  "top 2% by size" cut must be exact and linearly interpolated like DuckDB's
  ``quantile_cont``, which is ``quantileExactInclusive``.
* **``leadInFrame``/``lagInFrame`` respect the window frame.** The default frame
  ends at the current row, so ``leadInFrame`` would always see nothing; Q3 states
  the full frame. And the "no previous row" default is the type's zero, not NULL —
  ``lagInFrame(toNullable(mid))`` keeps Q5's first bar NULL instead of ``log(mid/0)``.
* **``log`` is approximate.** ClickHouse's ``log`` trades accuracy for SIMD speed
  (~1e-9 relative error near 1, measured against libm); Q5 uses ``log1p(x - 1)``,
  which is correctly rounded, so its correlations pass the 1e-12 oracle.
* **int32 × int32 stays int32.** ``price`` is widened with ``toInt64`` before the
  VWAP product, as in DuckDB.
"""

from __future__ import annotations

from .schema import NS_PER_MIN, NS_PER_SEC, SESSION_NS

# End sentinel for a book level that is still live at the end of the data.
_END_SENTINEL = 9_223_372_036_854_775_807


def q1_asof_trades_quotes_sql() -> str:
    return """
    SELECT
        t.seq AS seq, t.ts AS ts, t.symbol AS symbol, t.price AS trade_px,
        t.size AS trade_sz, t.aggressor AS aggressor,
        q.bid_px AS bid_px, q.bid_sz AS bid_sz, q.ask_px AS ask_px, q.ask_sz AS ask_sz
    FROM trades t
    ASOF LEFT JOIN quotes q
      ON t.symbol = q.symbol
     AND t.ts >= q.ts
    ORDER BY symbol, ts, seq
    """


def q2_vwap_bars_sql(bar_ns: int = NS_PER_MIN) -> str:
    return f"""
    SELECT
        symbol,
        intDiv(ts, {bar_ns}) AS bar,
        intDiv(ts, {bar_ns}) * {bar_ns} AS bar_start_ns,
        sum(toInt64(price) * size) AS pv,
        sum(toInt64(size)) AS volume,
        toFloat64(sum(toInt64(price) * size)) / sum(toInt64(size)) AS vwap_ticks,
        argMin(price, seq) AS open_px,
        max(price) AS high_px,
        min(price) AS low_px,
        argMax(price, seq) AS close_px,
        count() AS n_trades
    FROM trades
    GROUP BY symbol, bar
    ORDER BY symbol, bar
    """


def q3_book_reconstruction_sql() -> str:
    return f"""
    WITH running AS (
        SELECT
            date, symbol, side, price, seq, ts,
            sum(delta) OVER (
                PARTITION BY date, symbol, side, price ORDER BY seq
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS level_size,
            leadInFrame(ts, 1, toInt64({_END_SENTINEL})) OVER (
                PARTITION BY date, symbol, side, price ORDER BY seq
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) AS end_ts
        FROM book_events
    ),
    spans AS (
        SELECT date, symbol, side, price, level_size, ts AS start_ts, end_ts
        FROM running
        WHERE level_size > 0
    )
    SELECT
        g.symbol AS symbol,
        g.sample_ts AS sample_ts,
        maxIf(s.price, s.side = 'B') AS bid_px,
        argMaxIf(s.level_size, s.price, s.side = 'B') AS bid_sz,
        minIf(s.price, s.side = 'A') AS ask_px,
        argMinIf(s.level_size, s.price, s.side = 'A') AS ask_sz
    FROM sample_grid g
    INNER JOIN spans s
      ON s.symbol = g.symbol
     AND s.date = intDiv(g.sample_ts, {SESSION_NS})
     AND s.start_ts <= g.sample_ts
     AND g.sample_ts < s.end_ts
    GROUP BY symbol, sample_ts
    ORDER BY symbol, sample_ts
    """


def q4_event_study_sql(horizon_ns: int = 5 * NS_PER_SEC, top_frac: float = 0.02) -> str:
    return f"""
    WITH mids AS (
        SELECT ts, symbol, (bid_px + ask_px) / 2.0 AS mid FROM quotes
    ),
    thresh AS (
        SELECT symbol, quantileExactInclusive({1.0 - top_frac})(size) AS cut
        FROM trades GROUP BY symbol
    ),
    events AS (
        SELECT t.ts AS ts, t.symbol AS symbol, t.ts + {horizon_ns} AS exit_ts
        FROM trades t INNER JOIN thresh th ON t.symbol = th.symbol
        WHERE t.size >= th.cut
    ),
    entry AS (
        SELECT e.ts AS ts, e.symbol AS symbol, e.exit_ts AS exit_ts, m.mid AS entry_mid
        FROM events e
        ASOF LEFT JOIN mids m
          ON e.symbol = m.symbol AND e.ts >= m.ts
    ),
    bracketed AS (
        SELECT en.symbol AS symbol, en.entry_mid AS entry_mid, m.mid AS exit_mid
        FROM entry en
        ASOF LEFT JOIN mids m
          ON en.symbol = m.symbol AND en.exit_ts >= m.ts
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
        SELECT symbol, intDiv(ts, {bar_ns}) AS bar,
               argMax((bid_px + ask_px) / 2.0, ts) AS mid
        FROM quotes
        GROUP BY symbol, bar
    ),
    rets AS (
        SELECT symbol, bar,
               -- log1p(x - 1), not log(x): ClickHouse's log() is a fast vectorized
               -- approximation (~1e-9 relative error for x near 1, measured), which
               -- is visible in a correlation of 1-minute returns. log1p is correctly
               -- rounded, and x - 1 is exact for x near 1 (Sterbenz).
               log1p(mid / lagInFrame(toNullable(mid)) OVER (
                   PARTITION BY symbol ORDER BY bar
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) - 1) AS r
        FROM bar_mid
    )
    SELECT a.symbol AS sym_a, b.symbol AS sym_b,
           corrStable(a.r, b.r) AS correlation,
           count() AS n_bars
    FROM rets a INNER JOIN rets b ON a.bar = b.bar
    WHERE a.symbol < b.symbol AND a.r IS NOT NULL AND b.r IS NOT NULL
    GROUP BY sym_a, sym_b
    ORDER BY sym_a, sym_b
    """
