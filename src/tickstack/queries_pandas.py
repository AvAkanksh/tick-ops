"""The five canonical queries again, in reasonable-effort pandas.

This is the baseline the DuckDB stack is measured against, so it has to be *fair* —
the implementation a competent quant would actually write, not a strawman. Fairness
here means three concrete commitments:

* **Same answer.** Each function computes the identical logical result to its DuckDB
  twin; the harness asserts they match before timing means anything. Where DuckDB
  is exact-integer, pandas is too (VWAP numerator stays in int64).
* **Vectorized, not looped.** As-of joins use ``merge_asof`` (pandas' purpose-built
  primitive), bars use ``groupby``, reconstruction is vectorized per level. No
  ``iterrows`` strawman — that would make DuckDB look good for the wrong reason.
* **Data already in memory.** Inputs are pandas DataFrames handed over from Arrow
  (``Table.to_pandas``), so this measures *compute*, not pandas' Parquet reader. The
  DuckDB side is timed both ways (from Parquet, and from in-memory Arrow) so the
  comparison is stated honestly rather than conflating I/O with compute.

The signatures take plain DataFrames. The harness converts the Arrow tables from
:func:`tickstack.engine.load_tables_arrow` once and reuses them across repetitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import NS_PER_MIN, NS_PER_SEC, SESSION_NS


def q1_asof_trades_quotes(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Q1 — prevailing quote per trade via ``merge_asof`` (backward: last quote with
    ts <= trade ts), grouped by symbol via the ``by=`` key. ``merge_asof`` requires
    both sides sorted on the ``on`` key, so we sort by ts first."""
    t = trades.sort_values(["ts", "seq"], kind="stable")
    q = quotes.sort_values("ts", kind="stable")
    merged = pd.merge_asof(
        t, q, on="ts", by="symbol", direction="backward", suffixes=("", "_q")
    )
    out = merged[
        ["seq", "ts", "symbol", "price", "size", "aggressor",
         "bid_px", "bid_sz", "ask_px", "ask_sz"]
    ].rename(columns={"price": "trade_px", "size": "trade_sz"})
    return out.sort_values(["symbol", "ts", "seq"], kind="stable").reset_index(drop=True)


def q2_vwap_bars(trades: pd.DataFrame, bar_ns: int = NS_PER_MIN) -> pd.DataFrame:
    """Q2 — VWAP + OHLCV per (symbol, time bucket). Bucket key is integer floor
    division so edges match DuckDB. VWAP numerator ``price*size`` is accumulated in
    int64 (exact), divided once. Open/close are the first/last trade *by ts* within
    the bar, obtained by sorting on ts and taking first/last — identical to DuckDB's
    ``arg_min``/``arg_max`` on ts because a bar's rows are contiguous after sort."""
    df = trades.copy()
    df["bar"] = df["ts"] // bar_ns
    # int64 numerator: cast before multiply so int32*int32 can't overflow.
    df["pv"] = df["price"].astype("int64") * df["size"].astype("int64")
    # Sort by seq within a bar so first/last == open/close even when prints share
    # a ns, matching DuckDB's arg_min/arg_max on seq.
    df = df.sort_values(["symbol", "bar", "seq"], kind="stable")

    grouped = df.groupby(["symbol", "bar"], sort=True)
    agg = grouped.agg(
        pv=("pv", "sum"),
        volume=("size", "sum"),
        open_px=("price", "first"),
        high_px=("price", "max"),
        low_px=("price", "min"),
        close_px=("price", "last"),
        n_trades=("price", "size"),
    ).reset_index()
    agg["bar_start_ns"] = agg["bar"] * bar_ns
    agg["volume"] = agg["volume"].astype("int64")
    agg["vwap_ticks"] = agg["pv"].astype("float64") / agg["volume"]
    cols = ["symbol", "bar", "bar_start_ns", "pv", "volume", "vwap_ticks",
            "open_px", "high_px", "low_px", "close_px", "n_trades"]
    return agg[cols].sort_values(["symbol", "bar"], kind="stable").reset_index(drop=True)


def q3_book_reconstruction(book_events: pd.DataFrame, sample_grid: pd.DataFrame) -> pd.DataFrame:
    """Q3 — top-of-book at each grid point from the MBP stream, independent of quotes.

    Book reconstruction is the one query with no clean vectorized pandas primitive:
    "best price among currently-resting levels" is a running max/min over a *changing
    multiset*, which neither ``groupby`` nor ``merge_asof`` expresses. The honest
    reasonable-effort pandas approach — and the one a quant actually writes — is a
    stateful replay: walk each symbol-day's events in ``seq`` order maintaining the
    two level dicts, emit a top-of-book snapshot whenever the top changes, then
    ``merge_asof`` the sample grid onto those snapshots. Linear in events, and it is
    exactly the DuckDB set logic's imperative twin, so agreement is a real check.

    This is where DuckDB's range join structurally wins: the same reconstruction is
    a single set-based query there, and a Python event loop here."""
    if book_events.empty:
        return pd.DataFrame(
            columns=["symbol", "sample_ts", "bid_px", "bid_sz", "ask_px", "ask_sz"]
        )

    be = book_events.sort_values(["date", "symbol", "seq"], kind="stable")
    # Materialize to Python lists, not numpy arrays: the replay is a pure-Python loop
    # that indexes element-by-element and uses the values as dict keys and in tuple
    # comparisons. Indexing a numpy array with a Python int boxes a numpy scalar on
    # every access — the worst case for this access pattern — whereas .tolist() hands
    # back native ints/strs once. Same values, same output, ~1.5x faster loop; keeping
    # the slow form would flatter DuckDB's Q3 speedup for no real reason.
    date = be["date"].tolist()
    symbol = be["symbol"].tolist()
    side = be["side"].tolist()
    price = be["price"].tolist()
    delta = be["delta"].tolist()
    ts = be["ts"].tolist()

    # Replay per (date, symbol), snapshot the top whenever it moves.
    snap_sym: list[str] = []
    snap_ts: list[int] = []
    snap_bpx: list[int] = []
    snap_bsz: list[int] = []
    snap_apx: list[int] = []
    snap_asz: list[int] = []

    bids: dict[int, int] = {}
    asks: dict[int, int] = {}
    cur_key = None
    last_top = None

    def flush_top(sym, t):
        nonlocal last_top
        if not bids or not asks:
            return
        bpx = max(bids)
        apx = min(asks)
        top = (bpx, bids[bpx], apx, asks[apx])
        if top == last_top:
            return
        last_top = top
        snap_sym.append(sym)
        snap_ts.append(t)
        snap_bpx.append(top[0])
        snap_bsz.append(top[1])
        snap_apx.append(top[2])
        snap_asz.append(top[3])

    n = len(be)
    i = 0
    while i < n:
        key = (date[i], symbol[i])
        if key != cur_key:
            bids = {}
            asks = {}
            cur_key = key
            last_top = None
        # Apply all events sharing this ts (a sweep touches several levels at once),
        # then snapshot once — matching the [ts, next_ts) span semantics.
        t = ts[i]
        sym = symbol[i]
        while i < n and ts[i] == t and (date[i], symbol[i]) == key:
            book = bids if side[i] == "B" else asks
            lvl = book.get(price[i], 0) + delta[i]
            if lvl > 0:
                book[price[i]] = lvl
            elif price[i] in book:
                del book[price[i]]
            i += 1
        flush_top(sym, int(t))

    snaps = pd.DataFrame(
        {
            "symbol": snap_sym, "ts": snap_ts,
            "bid_px": snap_bpx, "bid_sz": snap_bsz,
            "ask_px": snap_apx, "ask_sz": snap_asz,
        }
    )
    if snaps.empty:
        return pd.DataFrame(
            columns=["symbol", "sample_ts", "bid_px", "bid_sz", "ask_px", "ask_sz"]
        )

    # merge_asof each grid point to the last snapshot at/<= it, within the same day.
    grid = sample_grid.copy()
    grid["date"] = grid["sample_ts"] // SESSION_NS
    snaps["date"] = snaps["ts"] // SESSION_NS
    grid = grid.sort_values("sample_ts", kind="stable")
    snaps = snaps.sort_values("ts", kind="stable")
    merged = pd.merge_asof(
        grid, snaps, left_on="sample_ts", right_on="ts",
        by=["symbol", "date"], direction="backward",
    )
    out = merged.dropna(subset=["bid_px", "ask_px"])[
        ["symbol", "sample_ts", "bid_px", "bid_sz", "ask_px", "ask_sz"]
    ].copy()
    for c in ("bid_px", "bid_sz", "ask_px", "ask_sz"):
        out[c] = out[c].astype("int64")
    return out.sort_values(["symbol", "sample_ts"], kind="stable").reset_index(drop=True)


def q4_event_study(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    horizon_ns: int = 5 * NS_PER_SEC,
    top_frac: float = 0.02,
) -> pd.DataFrame:
    """Q4 — mean forward mid return after large (top ``top_frac`` per symbol) trades.

    Per-symbol size cutoff via ``quantile``; entry and exit mids via two
    ``merge_asof`` backward joins (at trade ts, and at trade ts + horizon). Return in
    bps of entry mid. Uses the same quantile definition as DuckDB's
    ``quantile_cont`` (linear interpolation) so the event sets match exactly."""
    q = quotes.sort_values("ts", kind="stable").copy()
    q["mid"] = (q["bid_px"] + q["ask_px"]) / 2.0
    mids = q[["ts", "symbol", "mid"]]

    # Per-symbol threshold: top_frac tail => (1 - top_frac) quantile, linear interp.
    cut = trades.groupby("symbol")["size"].quantile(1.0 - top_frac, interpolation="linear")
    ev = trades.join(cut.rename("cut"), on="symbol")
    ev = ev[ev["size"] >= ev["cut"]][["ts", "symbol", "size"]].sort_values("ts", kind="stable")

    entry = pd.merge_asof(ev, mids, on="ts", by="symbol", direction="backward")
    entry = entry.rename(columns={"mid": "entry_mid"})

    ev_exit = entry.copy()
    ev_exit["ts"] = ev_exit["ts"] + horizon_ns
    ev_exit = ev_exit.sort_values("ts", kind="stable")
    exit_join = pd.merge_asof(
        ev_exit[["ts", "symbol", "entry_mid"]], mids, on="ts", by="symbol", direction="backward"
    ).rename(columns={"mid": "exit_mid"})

    valid = exit_join[
        exit_join["entry_mid"].notna()
        & exit_join["exit_mid"].notna()
        & (exit_join["entry_mid"] > 0)
    ].copy()
    valid["fwd_bps"] = (valid["exit_mid"] - valid["entry_mid"]) / valid["entry_mid"] * 10000.0
    out = valid.groupby("symbol").agg(
        n_events=("fwd_bps", "size"), mean_fwd_bps=("fwd_bps", "mean")
    ).reset_index()
    return out.sort_values("symbol", kind="stable").reset_index(drop=True)


def q5_cross_symbol_corr(quotes: pd.DataFrame, bar_ns: int = NS_PER_MIN) -> pd.DataFrame:
    """Q5 — Pearson correlation of per-bar log-returns for each symbol pair (a<b).

    Last mid per (symbol, bar), log return of consecutive bars, then a pivot to a
    (bar x symbol) return matrix and ``DataFrame.corr`` — which is pairwise-complete
    by default, matching DuckDB's inner join on bar per pair. ``n_bars`` is the count
    of jointly-present returns per pair."""
    q = quotes.copy()
    q["mid"] = (q["bid_px"] + q["ask_px"]) / 2.0
    q["bar"] = q["ts"] // bar_ns
    q = q.sort_values(["symbol", "bar", "ts"], kind="stable")
    bar_mid = q.groupby(["symbol", "bar"]).agg(mid=("mid", "last")).reset_index()
    bar_mid = bar_mid.sort_values(["symbol", "bar"], kind="stable")
    bar_mid["r"] = bar_mid.groupby("symbol")["mid"].transform(lambda s: np.log(s / s.shift(1)))

    mat = bar_mid.pivot(index="bar", columns="symbol", values="r")
    symbols = sorted(mat.columns)
    rows = []
    for i, a in enumerate(symbols):
        for b in symbols[i + 1 :]:
            pair = mat[[a, b]].dropna()
            n = len(pair)
            corr = pair[a].corr(pair[b]) if n > 1 else np.nan
            rows.append((a, b, corr, n))
    return pd.DataFrame(rows, columns=["sym_a", "sym_b", "correlation", "n_bars"])
