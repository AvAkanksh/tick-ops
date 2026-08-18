"""A deterministic limit-order-book market simulator.

The whole benchmark rests on this file, because a benchmark on *inconsistent* data
measures nothing. The simulator maintains a real integer-priced order book per
symbol and emits three streams that are consistent *by construction*:

* ``book_events`` — every change to resting size at a price level, as a signed
  ``delta``. Summing a level's deltas in ``(ts, seq)`` order gives its live size.
* ``quotes`` — the top of book (best bid/ask price and size) recorded every time
  it changes. This is the generator's own record of the book top.
* ``trades`` — one print per price level consumed by an aggressive order, at the
  resting price it lifted.

The load-bearing invariant, which query Q3 checks and which makes the reconstruction
benchmark honest: **replaying ``book_events`` up to any quote's timestamp
reproduces that quote exactly.** To keep "up to a timestamp" unambiguous, every
*action* within a symbol gets a strictly increasing, unique ``ts`` — all the book
events of one action share it, and no two actions collide — so reconstruction
keyed on ``ts`` alone is well defined.

Cross-symbol correlation (query Q5) is real, not decorative: every symbol's fair
value tracks a shared market factor with a symbol-specific beta plus idiosyncratic
noise, so the correlation structure is a known function of the betas.

Everything is seeded. ``default_rng([seed, day, symbol])`` makes each symbol-day
independently reproducible and the whole dataset bit-reproducible from the seed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

from .schema import (
    BOOK_EVENTS_SCHEMA,
    NS_PER_SEC,
    QUOTES_SCHEMA,
    SESSION_NS,
    SESSION_SECONDS,
    SIDE_ASK,
    SIDE_BID,
    TRADES_SCHEMA,
)


@dataclass(frozen=True)
class SimConfig:
    """Knobs for the simulator. Defaults produce a laptop-scale but non-trivial
    dataset; ``quick()`` is the fast preset used by the unit tests."""

    seed: int = 20260731
    n_days: int = 5
    symbols: tuple[str, ...] = ("AAPL", "MSFT", "AMZN", "GOOG", "META", "NVDA", "TSLA", "JPM")
    events_per_symbol_day: int = 40_000

    # Base price per symbol in ticks (cents). Filled from ``symbols`` if empty.
    base_price_ticks: dict[str, int] = field(default_factory=dict)

    # Action mix. Adds slightly exceed cancels+trades so the book stays populated.
    p_add: float = 0.50
    p_cancel: float = 0.30
    p_trade: float = 0.20

    # Price-formation parameters, all in ticks.
    market_sigma: float = 0.9  # per-second common-factor step
    idio_sigma: float = 0.25  # per-event idiosyncratic step
    beta_low: float = 0.6
    beta_high: float = 1.4

    # Book shape.
    init_levels: int = 8  # levels seeded per side at session open
    max_band: int = 60  # levels never posted more than this many ticks from fair value
    max_trade_levels: int = 5  # an aggressive order sweeps at most this many levels

    def __post_init__(self) -> None:
        # The action selector uses cumulative thresholds p_add and p_add+p_cancel,
        # so p_trade is the implied remainder. Validate that the three sum to 1 (and
        # p_trade matches the remainder) so a mis-specified config fails loudly at
        # construction rather than silently changing the trade rate.
        total = self.p_add + self.p_cancel + self.p_trade
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"p_add + p_cancel + p_trade must sum to 1.0, got {total} "
                f"({self.p_add} + {self.p_cancel} + {self.p_trade})"
            )
        if min(self.p_add, self.p_cancel, self.p_trade) < 0:
            raise ValueError("action probabilities must be non-negative")

    @classmethod
    def quick(cls) -> SimConfig:
        return cls(n_days=2, symbols=("AAA", "BBB", "CCC"), events_per_symbol_day=4_000)

    def resolved_bases(self) -> dict[str, int]:
        """Base price per symbol, defaulting to a spread of round-number prices."""
        if self.base_price_ticks:
            return dict(self.base_price_ticks)
        # Deterministic spread: 50.00, 75.00, 100.00, ... so symbols don't overlap.
        return {sym: 5000 + 2500 * i for i, sym in enumerate(self.symbols)}


@dataclass
class DayBatch:
    """One trading day's three tables, already sorted for writing."""

    date: int
    trades: pa.Table
    quotes: pa.Table
    book_events: pa.Table


class _ColumnBuffers:
    """Plain Python-list column accumulators, converted to Arrow once per day."""

    def __init__(self) -> None:
        self.tr_seq: list[int] = []
        self.tr_ts: list[int] = []
        self.tr_sym: list[str] = []
        self.tr_px: list[int] = []
        self.tr_sz: list[int] = []
        self.tr_agg: list[str] = []

        self.q_ts: list[int] = []
        self.q_sym: list[str] = []
        self.q_bpx: list[int] = []
        self.q_bsz: list[int] = []
        self.q_apx: list[int] = []
        self.q_asz: list[int] = []

        self.be_seq: list[int] = []
        self.be_ts: list[int] = []
        self.be_sym: list[str] = []
        self.be_side: list[str] = []
        self.be_px: list[int] = []
        self.be_delta: list[int] = []

    def build(self, date: int) -> DayBatch:
        n_tr = len(self.tr_ts)
        n_q = len(self.q_ts)
        n_be = len(self.be_seq)

        trades = pa.table(
            {
                "seq": pa.array(self.tr_seq, pa.int64()),
                "ts": pa.array(self.tr_ts, pa.int64()),
                "symbol": pa.array(self.tr_sym, pa.string()),
                "price": pa.array(self.tr_px, pa.int32()),
                "size": pa.array(self.tr_sz, pa.int32()),
                "aggressor": pa.array(self.tr_agg, pa.string()),
                "date": pa.array([date] * n_tr, pa.int32()),
            },
            schema=TRADES_SCHEMA,
        )
        quotes = pa.table(
            {
                "ts": pa.array(self.q_ts, pa.int64()),
                "symbol": pa.array(self.q_sym, pa.string()),
                "bid_px": pa.array(self.q_bpx, pa.int32()),
                "bid_sz": pa.array(self.q_bsz, pa.int32()),
                "ask_px": pa.array(self.q_apx, pa.int32()),
                "ask_sz": pa.array(self.q_asz, pa.int32()),
                "date": pa.array([date] * n_q, pa.int32()),
            },
            schema=QUOTES_SCHEMA,
        )
        book = pa.table(
            {
                "seq": pa.array(self.be_seq, pa.int64()),
                "ts": pa.array(self.be_ts, pa.int64()),
                "symbol": pa.array(self.be_sym, pa.string()),
                "side": pa.array(self.be_side, pa.string()),
                "price": pa.array(self.be_px, pa.int32()),
                "delta": pa.array(self.be_delta, pa.int32()),
                "date": pa.array([date] * n_be, pa.int32()),
            },
            schema=BOOK_EVENTS_SCHEMA,
        )

        # Sort each table by time so the on-disk data is time-ordered within a day.
        # book_events break ts ties by seq to preserve intra-action replay order.
        trades = trades.sort_by([("ts", "ascending"), ("seq", "ascending")])
        quotes = quotes.sort_by([("ts", "ascending"), ("symbol", "ascending")])
        book = book.sort_by([("ts", "ascending"), ("seq", "ascending")])
        return DayBatch(date=date, trades=trades, quotes=quotes, book_events=book)


class _Book:
    """One symbol's order book: two ``price -> size`` dicts with incrementally
    tracked best bid/ask. Zero-size levels are deleted so ``max``/``min`` rescans
    (needed only when the top empties) stay cheap."""

    __slots__ = ("bids", "asks", "best_bid", "best_ask")

    # Emptiness is structural: ``best_bid``/``best_ask`` are ``None`` iff the side is
    # empty. An earlier version used -1 as an "empty" sentinel, which aliases a real
    # (negative) price level and silently desyncs the tracked top from the dict. None
    # can never collide with a price, so the invariant holds for any price range.
    def __init__(self) -> None:
        self.bids: dict[int, int] = {}
        self.asks: dict[int, int] = {}
        self.best_bid: int | None = None
        self.best_ask: int | None = None

    def add(self, side: str, price: int, size: int) -> None:
        book = self.bids if side == SIDE_BID else self.asks
        book[price] = book.get(price, 0) + size
        if side == SIDE_BID:
            if self.best_bid is None or price > self.best_bid:
                self.best_bid = price
        elif self.best_ask is None or price < self.best_ask:
            self.best_ask = price

    def remove(self, side: str, price: int, size: int) -> int:
        """Remove up to ``size`` from a level; return the amount actually removed."""
        book = self.bids if side == SIDE_BID else self.asks
        have = book.get(price, 0)
        take = size if size < have else have
        if take <= 0:
            return 0
        remaining = have - take
        if remaining == 0:
            del book[price]
            if side == SIDE_BID and price == self.best_bid:
                self.best_bid = max(book) if book else None
            elif side == SIDE_ASK and price == self.best_ask:
                self.best_ask = min(book) if book else None
        else:
            book[price] = remaining
        return take


class Simulator:
    """Generates the dataset described by a :class:`SimConfig`."""

    def __init__(self, config: SimConfig) -> None:
        self.cfg = config
        self.bases = config.resolved_bases()
        # A shared per-second market path across the whole span drives cross-symbol
        # correlation. Seeded off the base seed so the whole dataset is reproducible.
        mkt_rng = np.random.default_rng([config.seed, 0x4D4B54])  # 0x4D4B54 = "MKT"
        grid = config.n_days * SESSION_SECONDS
        self.market_path = np.cumsum(
            mkt_rng.normal(0.0, config.market_sigma, size=grid)
        )
        # Betas are fixed per symbol (deterministic), spanning [beta_low, beta_high].
        n = len(config.symbols)
        self.betas = {
            sym: config.beta_low
            + (config.beta_high - config.beta_low) * (i / max(1, n - 1))
            for i, sym in enumerate(config.symbols)
        }
        self._seq = 0  # global monotonic sequence for book_events

    def simulate(self) -> Iterator[DayBatch]:
        for day in range(self.cfg.n_days):
            buf = _ColumnBuffers()
            for sym_idx, sym in enumerate(self.cfg.symbols):
                self._simulate_symbol_day(buf, day, sym_idx, sym)
            yield buf.build(day)

    # -- internals ----------------------------------------------------------

    def _market_at(self, ts: int) -> float:
        sec = ts // NS_PER_SEC
        if sec >= self.market_path.shape[0]:
            sec = self.market_path.shape[0] - 1
        return float(self.market_path[sec])

    def _next_seq(self) -> int:
        s = self._seq
        self._seq += 1
        return s

    def _simulate_symbol_day(
        self, buf: _ColumnBuffers, day: int, sym_idx: int, sym: str
    ) -> None:
        cfg = self.cfg
        rng = np.random.default_rng([cfg.seed, day, sym_idx])
        n = cfg.events_per_symbol_day
        day_start = day * SESSION_NS
        beta = self.betas[sym]
        base = self.bases[sym]

        # Fair value at session open, from the shared market factor.
        fv = float(base + beta * self._market_at(day_start))
        book = _Book()

        # Seed a two-sided book around fair value so the first action has a top.
        # Bid levels are floored at 1 tick so a low base price with many levels can
        # never post a non-positive price (prices are strictly positive by contract).
        mid0 = int(round(fv))
        for lvl in range(cfg.init_levels):
            bid_px = mid0 - 1 - lvl
            ask_px = mid0 + 1 + lvl
            if bid_px >= 1:
                bsz = int(rng.integers(100, 800))
                book.add(SIDE_BID, bid_px, bsz)
                self._emit_book(buf, day_start, sym, SIDE_BID, bid_px, bsz)
            asz = int(rng.integers(100, 800))
            book.add(SIDE_ASK, ask_px, asz)
            self._emit_book(buf, day_start, sym, SIDE_ASK, ask_px, asz)
        last_quote = self._record_quote(buf, day_start, sym, book, None)

        # Pre-draw randomness in bulk; index into it in the loop.
        gaps = rng.exponential(SESSION_NS / (n + 1), size=n)
        action_roll = rng.random(size=n)
        side_roll = rng.random(size=n)
        offsets = rng.geometric(0.55, size=n) - 1  # 0,1,2,... ticks behind best
        improve_roll = rng.random(size=n)
        add_sizes = rng.integers(50, 600, size=n)
        trade_sizes = rng.integers(50, 1500, size=n)
        idio_steps = rng.normal(0.0, cfg.idio_sigma, size=n)
        cancel_frac = rng.random(size=n)

        clock = day_start
        idio = 0.0
        p_add = cfg.p_add
        p_add_cancel = cfg.p_add + cfg.p_cancel

        for i in range(n):
            # Advance the per-symbol clock; strictly increasing, unique per action.
            # step >= 2 so the action's resulting quote can sit at clock+1 without
            # colliding with the next action's clock (see _record_quote call below).
            step = int(gaps[i]) + 2
            clock += step
            if clock >= day_start + SESSION_NS:
                break

            # Update fair value from the shared market factor plus idiosyncratic drift.
            idio += idio_steps[i]
            fv = base + beta * self._market_at(clock) + idio

            # Directional pressure: which way is fair value relative to mid?
            mid = (book.best_bid + book.best_ask) / 2.0
            drift = fv - mid  # >0 => upward pressure (favor buys / bid improvement)

            roll = action_roll[i]
            if roll < p_add:
                self._do_add(
                    buf, clock, sym, book, drift, int(offsets[i]),
                    improve_roll[i], int(add_sizes[i]), side_roll[i], fv,
                )
            elif roll < p_add_cancel:
                self._do_cancel(buf, clock, sym, book, side_roll[i], cancel_frac[i], drift)
            else:
                self._do_trade(buf, clock, sym, book, drift, int(trade_sizes[i]), side_roll[i], fv)

            # Keep the book two-sided; reseed a side if it emptied.
            self._ensure_two_sided(buf, clock, sym, book, fv, rng)

            # Record the resulting top-of-book at clock+1, strictly after the trade
            # prints and book events at `clock`. This makes the "prevailing quote"
            # for a trade the state *before* it — an inclusive as-of join (ts >= q.ts)
            # then attaches the pre-trade quote, not the quote the trade itself
            # produced. Q3 reconstruction still matches exactly: it reconstructs at
            # the quote's own timestamp (clock+1), after the events at clock are
            # applied. The +2 clock step above guarantees clock+1 never collides with
            # the next action.
            last_quote = self._record_quote(buf, clock + 1, sym, book, last_quote)

    # -- actions ------------------------------------------------------------

    def _do_add(
        self, buf, ts, sym, book, drift, offset, improve_roll, size, side_roll, fv
    ) -> None:
        cfg = self.cfg
        # Directional pressure tilts which side receives liquidity and how often it
        # improves the quote. p_bid rises when fair value is above mid.
        p_bid = 0.5 + 0.35 * np.tanh(drift / 3.0)
        side = SIDE_BID if side_roll < p_bid else SIDE_ASK
        improve_prob = 0.15 + 0.25 * (p_bid if side == SIDE_BID else (1.0 - p_bid))

        if side == SIDE_BID:
            price = book.best_bid + 1 if improve_roll < improve_prob else book.best_bid - offset
            price = min(price, book.best_ask - 1)  # never cross
        else:
            price = book.best_ask - 1 if improve_roll < improve_prob else book.best_ask + offset
            price = max(price, book.best_bid + 1)

        # Stay within a band around fair value.
        lo, hi = int(fv - cfg.max_band), int(fv + cfg.max_band)
        if price < lo or price > hi or price <= 0:
            return
        book.add(side, price, size)
        self._emit_book(buf, ts, sym, side, price, size)

    def _do_cancel(self, buf, ts, sym, book, side_roll, frac, drift) -> None:
        # Cancel pressure leans opposite to add pressure so the book breathes.
        p_bid = 0.5 - 0.35 * np.tanh(drift / 3.0)
        side = SIDE_BID if side_roll < p_bid else SIDE_ASK
        levels = book.bids if side == SIDE_BID else book.asks
        if not levels:
            return
        # Pick a level; bias toward canceling near the top where quotes churn most.
        prices = list(levels.keys())
        price = prices[int(frac * len(prices))]
        have = levels[price]
        amount = max(1, int(have * (0.3 + 0.7 * frac)))
        removed = book.remove(side, price, amount)
        if removed > 0:
            self._emit_book(buf, ts, sym, side, price, -removed)

    def _do_trade(self, buf, ts, sym, book, drift, size, side_roll, fv) -> None:
        cfg = self.cfg
        # Aggressor side: upward pressure favors buy aggressors lifting the ask.
        p_buy = 0.5 + 0.35 * np.tanh(drift / 3.0)
        buy = side_roll < p_buy
        agg = SIDE_BID if buy else SIDE_ASK  # 'B' = buyer initiated
        resting_side = SIDE_ASK if buy else SIDE_BID

        size_left = size
        levels_walked = 0
        while size_left > 0 and levels_walked < cfg.max_trade_levels:
            best = book.best_ask if buy else book.best_bid
            if best is None:
                break
            avail = book.asks.get(best, 0) if buy else book.bids.get(best, 0)
            if avail <= 0:
                break
            take = size_left if size_left < avail else avail
            removed = book.remove(resting_side, best, take)
            if removed <= 0:
                break
            self._emit_book(buf, ts, sym, resting_side, best, -removed)
            self._emit_trade(buf, ts, sym, best, removed, agg)
            size_left -= removed
            levels_walked += 1

    def _ensure_two_sided(self, buf, ts, sym, book, fv, rng) -> None:
        # Prices are floored at 1 tick: a price <= 0 is economically meaningless and
        # would (before the structural-emptiness fix) also collide with sentinels.
        if book.best_bid is None:
            price = max(1, int(fv) - 1)
            if book.best_ask is not None:
                price = min(price, book.best_ask - 1)
            if price >= 1:
                size = int(rng.integers(100, 600))
                book.add(SIDE_BID, price, size)
                self._emit_book(buf, ts, sym, SIDE_BID, price, size)
        if book.best_ask is None:
            price = int(fv) + 1
            if book.best_bid is not None:
                price = max(price, book.best_bid + 1)
            price = max(price, 1)
            size = int(rng.integers(100, 600))
            book.add(SIDE_ASK, price, size)
            self._emit_book(buf, ts, sym, SIDE_ASK, price, size)

    # -- emit helpers -------------------------------------------------------

    def _emit_book(self, buf, ts, sym, side, price, delta) -> None:
        buf.be_seq.append(self._next_seq())
        buf.be_ts.append(ts)
        buf.be_sym.append(sym)
        buf.be_side.append(side)
        buf.be_px.append(price)
        buf.be_delta.append(delta)

    def _emit_trade(self, buf, ts, sym, price, size, agg) -> None:
        # Trades draw from the same global seq as book events, so a trade and the
        # book event that recorded the level it consumed keep their generation order
        # and every trade has a unique, monotonic seq for tie-breaking.
        buf.tr_seq.append(self._next_seq())
        buf.tr_ts.append(ts)
        buf.tr_sym.append(sym)
        buf.tr_px.append(price)
        buf.tr_sz.append(size)
        buf.tr_agg.append(agg)

    def _record_quote(self, buf, ts, sym, book, last):
        """Append a quote row iff the top of book changed since ``last``."""
        if book.best_bid is None or book.best_ask is None:
            return last
        top = (book.best_bid, book.bids[book.best_bid], book.best_ask, book.asks[book.best_ask])
        if last is not None and top == last:
            return last
        buf.q_ts.append(ts)
        buf.q_sym.append(sym)
        buf.q_bpx.append(top[0])
        buf.q_bsz.append(top[1])
        buf.q_apx.append(top[2])
        buf.q_asz.append(top[3])
        return top


def generate(config: SimConfig) -> Iterator[DayBatch]:
    """Convenience wrapper: yield per-day batches for ``config``."""
    return Simulator(config).simulate()
