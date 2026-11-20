"""The five canonical quant queries, in DuckDB SQL.

These are the queries a research desk actually runs, chosen because each stresses a
different engine capability:

* **Q1 as-of join** — attach the prevailing quote to every trade. The signature
  quant query; stresses the engine's time-ordered join. DuckDB does it with ``ASOF
  JOIN``, which is a large part of why the open stack is interesting.
* **Q2 VWAP bars** — time-bucketed volume-weighted average price plus OHLCV.
  Stresses grouped aggregation over a computed bucket key.
* **Q3 book reconstruction** — turn the market-by-price event stream back into
  top-of-book on a sample grid. Stresses window functions (running sum, ``lead``,
  ``QUALIFY``) and a range join, and is the query whose *correctness* is checkable
  against the generator's own quotes.
* **Q4 event study** — mean forward mid return around large trades. Stresses two
  as-of joins bracketing an event set.
* **Q5 cross-symbol correlation** — Pearson correlation of per-bar returns across
  every symbol pair. Stresses a pivot-like self-join and an all-pairs aggregation.

Every function takes an open DuckDB connection with the three tables registered as
views (see :mod:`tickstack.engine`) and returns an Arrow table. Q3 additionally
reads a ``sample_grid(symbol, sample_ts)`` relation so DuckDB and the pandas
baseline reconstruct at *identical* sample points. Each result is the same logical
answer the pandas baseline computes; the harness diffs the two, so a divergence is
a bug on one side, not a tolerance to smooth over.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa

from .schema import NS_PER_MIN, NS_PER_SEC, SESSION_NS

# End sentinel for a book level that is still live at the end of the data. int64
# max, so any sample time compares as "before the level ever closes".
_END_SENTINEL = 9_223_372_036_854_775_807


def q1_asof_trades_quotes(con: duckdb.DuckDBPyConnection) -> pa.Table:
    """Q1 — as-of join: for each trade, the most recent quote at or before the
    trade's timestamp, matched within the same symbol.

    Prevailing-quote semantics are *the last quote with ``quote.ts <= trade.ts``*,
    which is exactly what ``ASOF ... ON t.ts >= q.ts`` selects. A LEFT ASOF join
    keeps trades that precede their symbol's first quote, with NULL quote fields.
    Deterministic tie-break: ties on ts are broken by ``quote.ts`` being 'known'
    at the trade, which mirrors a feed logging the quote update before the print."""
    return con.sql(
        """
        SELECT
            t.seq, t.ts, t.symbol, t.price AS trade_px, t.size AS trade_sz, t.aggressor,
            q.bid_px, q.bid_sz, q.ask_px, q.ask_sz
        FROM trades t
        ASOF LEFT JOIN quotes q
          ON t.symbol = q.symbol
         AND t.ts >= q.ts
        ORDER BY t.symbol, t.ts, t.seq
        """
    ).to_arrow_table()


def q2_vwap_bars(con: duckdb.DuckDBPyConnection, bar_ns: int = NS_PER_MIN) -> pa.Table:
    """Q2 — VWAP + OHLCV bars: bucket trades into fixed ``bar_ns`` windows per
    symbol and compute volume-weighted average price with open/high/low/close/volume.

    VWAP numerator is summed in integers and divided once at the end to a double.
    ``price`` is cast to BIGINT *before* the product so ``price * size`` is computed
    in int64, not int32 — an int32×int32 product would overflow at ~2.1e9 (a
    $210k×10k-share print), and the pandas baseline casts identically, so both are
    exact and equal. The bucket key is integer floor division ``ts // bar_ns``, so
    bar edges match the baseline bit-for-bit. Open and close use ``arg_min`` /
    ``arg_max`` on ``seq`` (not ``ts``) so they are the first and last *trade* in the
    bar even when several prints share a nanosecond — ``ts`` alone would leave a
    sweep's close ambiguous."""
    return con.sql(
        """
        WITH binned AS (
            SELECT symbol, seq, ts, price, size, (ts // ?) AS bar
            FROM trades
        )
        SELECT
            symbol,
            bar,
            bar * ?                                       AS bar_start_ns,
            sum(price::BIGINT * size)                     AS pv,   -- int64, no overflow
            sum(size)                                     AS volume,
            sum(price::BIGINT * size)::DOUBLE / sum(size) AS vwap_ticks,
            arg_min(price, seq)                           AS open_px,
            max(price)                                    AS high_px,
            min(price)                                    AS low_px,
            arg_max(price, seq)                           AS close_px,
            count(*)                                      AS n_trades
        FROM binned
        GROUP BY symbol, bar
        ORDER BY symbol, bar
        """,
        params=[bar_ns, bar_ns],
    ).to_arrow_table()


def q3_book_reconstruction(con: duckdb.DuckDBPyConnection) -> pa.Table:
    """Q3 — book reconstruction: rebuild top-of-book at every ``(symbol, sample_ts)``
    in the registered ``sample_grid`` relation, from the market-by-price event stream
    alone — the ``quotes`` table is not consulted.

    The method is fully set-based:

    1. **running size** — cumulative ``sum(delta)`` per ``(symbol, side, price)`` over
       ``seq`` order gives the resting size at each level after each event.
    2. **active spans** — ``lead(ts)`` gives the timestamp of that level's next event,
       so each event opens a half-open time span ``[ts, next_ts)`` during which the
       level holds a constant size. ``QUALIFY level_size > 0`` keeps only the spans in
       which the level is actually resting. Non-degenerate because a symbol's events
       carry strictly increasing per-level timestamps.
    3. **range join** — each grid point joins the spans covering it
       (``start_ts <= sample_ts < end_ts``); the best bid is the max live bid price and
       the best ask the min live ask price, with ``arg_max``/``arg_min`` reading the
       size at that extreme price. Only one span per price covers any instant, so the
       extreme price is unique.

    The result matches the generator's own top-of-book — the correctness harness
    reconstructs at quote timestamps and diffs against ``quotes``."""
    return con.sql(
        f"""
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
            QUALIFY level_size > 0
        )
        SELECT
            g.symbol,
            g.sample_ts,
            max(s.price)          FILTER (WHERE s.side = 'B')            AS bid_px,
            arg_max(s.level_size, s.price) FILTER (WHERE s.side = 'B')   AS bid_sz,
            min(s.price)          FILTER (WHERE s.side = 'A')            AS ask_px,
            arg_min(s.level_size, s.price) FILTER (WHERE s.side = 'A')   AS ask_sz
        FROM sample_grid g
        JOIN spans s
          ON s.symbol = g.symbol
         AND s.date = (g.sample_ts // {SESSION_NS})
         AND s.start_ts <= g.sample_ts
         AND g.sample_ts < s.end_ts
        GROUP BY g.symbol, g.sample_ts
        ORDER BY g.symbol, g.sample_ts
        """
    ).to_arrow_table()


def q4_event_study(
    con: duckdb.DuckDBPyConnection,
    horizon_ns: int = 5 * NS_PER_SEC,
    top_frac: float = 0.02,
) -> pa.Table:
    """Q4 — event study: mean forward mid-price return over ``horizon_ns`` following
    large trades, per symbol.

    'Large' is the top ``top_frac`` of trades by size *within each symbol*, so a
    big name and a small name are each judged against their own size distribution.
    The entry mid is the prevailing quote mid at the trade (an as-of join), and the
    exit mid is the prevailing quote mid ``horizon_ns`` later (a second as-of join
    on ``ts + horizon``). Return is basis points of the entry mid. Events whose exit
    falls after the last quote, or with a non-positive entry mid, are dropped."""
    return con.sql(
        """
        WITH mids AS (
            SELECT ts, symbol, (bid_px + ask_px) / 2.0 AS mid FROM quotes
        ),
        thresh AS (
            SELECT symbol, quantile_cont(size, 1.0 - ?) AS cut
            FROM trades GROUP BY symbol
        ),
        events AS (
            SELECT t.ts, t.symbol, t.size
            FROM trades t JOIN thresh th USING (symbol)
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
              ON en.symbol = m.symbol AND (en.ts + ?) >= m.ts
        )
        SELECT
            symbol,
            count(*) AS n_events,
            avg((exit_mid - entry_mid) / entry_mid * 10000.0) AS mean_fwd_bps
        FROM bracketed
        WHERE entry_mid IS NOT NULL AND exit_mid IS NOT NULL AND entry_mid > 0
        GROUP BY symbol
        ORDER BY symbol
        """,
        params=[top_frac, horizon_ns],
    ).to_arrow_table()


def q5_cross_symbol_corr(
    con: duckdb.DuckDBPyConnection, bar_ns: int = NS_PER_MIN
) -> pa.Table:
    """Q5 — cross-symbol correlation: Pearson correlation of per-bar mid-price
    log-returns between every symbol pair (a < b).

    The mid in each ``bar_ns`` bucket is the *last* quote mid in the bucket (a
    windowed pick by ts), the return is the log ratio of consecutive bar mids, and
    the correlation runs over the aligned return series — bars where either symbol
    lacks a return are dropped for that pair (inner join on bar). ``a.symbol <
    b.symbol`` yields each unordered pair once."""
    return con.sql(
        """
        WITH binned AS (
            SELECT symbol, ts, (bid_px + ask_px) / 2.0 AS mid_px, (ts // ?) AS bar
            FROM quotes
        ),
        bar_mid AS (
            SELECT
                symbol,
                bar,
                last(mid_px ORDER BY ts) AS mid
            FROM binned
            GROUP BY symbol, bar
        ),
        rets AS (
            SELECT symbol, bar,
                   ln(mid / lag(mid) OVER (PARTITION BY symbol ORDER BY bar)) AS r
            FROM bar_mid
        )
        SELECT a.symbol AS sym_a, b.symbol AS sym_b,
               corr(a.r, b.r) AS correlation,
               count(*)       AS n_bars
        FROM rets a JOIN rets b
          ON a.bar = b.bar AND a.symbol < b.symbol
        WHERE a.r IS NOT NULL AND b.r IS NOT NULL
        GROUP BY a.symbol, b.symbol
        ORDER BY a.symbol, b.symbol
        """,
        params=[bar_ns],
    ).to_arrow_table()
