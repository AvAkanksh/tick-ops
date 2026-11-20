"""The five canonical queries in Polars — the "modern DataFrame" baseline.

pandas is the baseline a desk already has; Polars is the one a desk would reach for
next instead of adopting a database, so it is the more demanding comparison for the
DuckDB stack. Same rules as :mod:`tickstack.queries_pandas`: identical logical
answer (diffed against the oracle before any timing), idiomatic and vectorized
(lazy frames, ``join_asof``, expression ``group_by``), and inputs already in memory
(``pl.from_arrow``) so it measures compute.

Q3 is the interesting one. Polars has no range join, and the pandas approach is a
Python event loop. The vectorized Polars formulation is the same as-of idea used
for QuestDB: cross each grid point with the price levels that exist that
symbol-day, ``join_asof`` each probe to the level's last event at or before the
sample time, keep levels with positive running size, and take the best bid/ask.
"""

from __future__ import annotations

import polars as pl

from .schema import NS_PER_MIN, NS_PER_SEC, SESSION_NS


def frames_from_arrow(tables: dict) -> dict[str, pl.DataFrame]:
    """Arrow -> Polars (zero-copy for numeric columns)."""
    return {k: pl.from_arrow(v) for k, v in tables.items()}


def q1_asof_trades_quotes(trades: pl.DataFrame, quotes: pl.DataFrame) -> pl.DataFrame:
    """Prevailing quote per trade: backward ``join_asof`` by symbol."""
    t = trades.lazy().sort(["ts", "seq"])
    q = quotes.lazy().select(["ts", "symbol", "bid_px", "bid_sz", "ask_px", "ask_sz"]).sort("ts")
    return (
        t.join_asof(q, on="ts", by="symbol", strategy="backward", check_sortedness=False)
        .select(
            "seq",
            "ts",
            "symbol",
            pl.col("price").alias("trade_px"),
            pl.col("size").alias("trade_sz"),
            "aggressor",
            "bid_px",
            "bid_sz",
            "ask_px",
            "ask_sz",
        )
        .sort(["symbol", "ts", "seq"])
        .collect()
    )


def q2_vwap_bars(trades: pl.DataFrame, bar_ns: int = NS_PER_MIN) -> pl.DataFrame:
    """VWAP + OHLCV per (symbol, bar). Integer numerator, open/close by ``seq``."""
    pv = pl.col("price").cast(pl.Int64) * pl.col("size").cast(pl.Int64)
    return (
        trades.lazy()
        .with_columns((pl.col("ts") // bar_ns).alias("bar"))
        .group_by(["symbol", "bar"])
        .agg(
            pv.sum().alias("pv"),
            pl.col("size").cast(pl.Int64).sum().alias("volume"),
            pl.col("price").get(pl.col("seq").arg_min()).alias("open_px"),
            pl.col("price").max().alias("high_px"),
            pl.col("price").min().alias("low_px"),
            pl.col("price").get(pl.col("seq").arg_max()).alias("close_px"),
            pl.len().alias("n_trades"),
        )
        .with_columns(
            (pl.col("bar") * bar_ns).alias("bar_start_ns"),
            (pl.col("pv").cast(pl.Float64) / pl.col("volume")).alias("vwap_ticks"),
        )
        .select(
            [
                "symbol",
                "bar",
                "bar_start_ns",
                "pv",
                "volume",
                "vwap_ticks",
                "open_px",
                "high_px",
                "low_px",
                "close_px",
                "n_trades",
            ]
        )
        .sort(["symbol", "bar"])
        .collect()
    )


def q3_book_reconstruction(book_events: pl.DataFrame, sample_grid: pl.DataFrame) -> pl.DataFrame:
    """Top-of-book at each grid point via as-of probes (see module docstring)."""
    keys = ["date", "symbol", "side", "price"]
    # Hive-partitioned layouts surface ``date`` as int64 (a partition column) while
    # the single-file layout stores int32; asof ``by`` keys must match exactly.
    book_events = book_events.with_columns(pl.col("date").cast(pl.Int32))
    running = (
        book_events.lazy()
        .sort("seq")
        .with_columns(pl.col("delta").cum_sum().over(keys).alias("level_size"))
        .select(["ts", "seq", *keys, "level_size"])
        .sort(["ts", "seq"])  # ties on ts resolve to the highest seq (last applied)
    )
    levels = book_events.lazy().select(keys).unique()
    grid = sample_grid.lazy().with_columns(
        (pl.col("sample_ts") // SESSION_NS).cast(pl.Int32).alias("date")
    )
    probes = grid.join(levels, on=["symbol", "date"], how="inner").sort("sample_ts")
    state = probes.join_asof(
        running,
        left_on="sample_ts",
        right_on="ts",
        by=keys,
        strategy="backward",
        check_sortedness=False,
    ).filter(pl.col("level_size") > 0)

    is_b = pl.col("side") == "B"
    is_a = pl.col("side") == "A"
    return (
        state.group_by(["symbol", "sample_ts"])
        .agg(
            pl.col("price").filter(is_b).max().alias("bid_px"),
            pl.col("level_size")
            .filter(is_b)
            .get(pl.col("price").filter(is_b).arg_max())
            .alias("bid_sz"),
            pl.col("price").filter(is_a).min().alias("ask_px"),
            pl.col("level_size")
            .filter(is_a)
            .get(pl.col("price").filter(is_a).arg_min())
            .alias("ask_sz"),
        )
        .sort(["symbol", "sample_ts"])
        .collect()
    )


def q4_event_study(
    trades: pl.DataFrame,
    quotes: pl.DataFrame,
    horizon_ns: int = 5 * NS_PER_SEC,
    top_frac: float = 0.02,
) -> pl.DataFrame:
    """Mean forward mid return (bps) after each symbol's top ``top_frac`` trades."""
    mids = (
        quotes.lazy()
        .select("ts", "symbol", ((pl.col("bid_px") + pl.col("ask_px")) / 2.0).alias("mid"))
        .sort("ts")
    )
    cut = pl.col("size").quantile(1.0 - top_frac, interpolation="linear").over("symbol")
    events = trades.lazy().filter(pl.col("size") >= cut).select("ts", "symbol").sort("ts")
    entry = events.join_asof(
        mids, on="ts", by="symbol", strategy="backward", check_sortedness=False
    ).rename({"mid": "entry_mid"})
    exits = (
        entry.with_columns(pl.col("ts") + horizon_ns)
        .sort("ts")
        .join_asof(mids, on="ts", by="symbol", strategy="backward", check_sortedness=False)
        .rename({"mid": "exit_mid"})
    )
    return (
        exits.filter(
            pl.col("entry_mid").is_not_null()
            & pl.col("exit_mid").is_not_null()
            & (pl.col("entry_mid") > 0)
        )
        .with_columns(
            ((pl.col("exit_mid") - pl.col("entry_mid")) / pl.col("entry_mid") * 10000.0).alias(
                "fwd_bps"
            )
        )
        .group_by("symbol")
        .agg(pl.len().alias("n_events"), pl.col("fwd_bps").mean().alias("mean_fwd_bps"))
        .sort("symbol")
        .collect()
    )


def q5_cross_symbol_corr(quotes: pl.DataFrame, bar_ns: int = NS_PER_MIN) -> pl.DataFrame:
    """Pearson correlation of per-bar mid log-returns for each symbol pair (a<b)."""
    rets = (
        quotes.lazy()
        .with_columns(
            ((pl.col("bid_px") + pl.col("ask_px")) / 2.0).alias("mid"),
            (pl.col("ts") // bar_ns).alias("bar"),
        )
        .group_by(["symbol", "bar"])
        .agg(pl.col("mid").get(pl.col("ts").arg_max()).alias("mid"))
        .sort(["symbol", "bar"])
        .with_columns((pl.col("mid") / pl.col("mid").shift(1)).log().over("symbol").alias("r"))
        .filter(pl.col("r").is_not_null())
        .select("symbol", "bar", "r")
    )
    pairs = rets.join(rets, on="bar", suffix="_b").filter(pl.col("symbol") < pl.col("symbol_b"))
    return (
        pairs.group_by([pl.col("symbol").alias("sym_a"), pl.col("symbol_b").alias("sym_b")])
        .agg(pl.corr("r", "r_b").alias("correlation"), pl.len().alias("n_bars"))
        .sort(["sym_a", "sym_b"])
        .collect()
    )
