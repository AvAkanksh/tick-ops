"""Canonical schemas and constants for the tick research stack.

Everything downstream — the simulator, the Parquet layouts, the five queries and
their pandas baselines — agrees on the column names and dtypes defined here. Two
decisions are load-bearing and worth stating up front:

1. **Prices are integers.** The book is keyed on price level, and reconstruction
   groups events by ``(symbol, side, price)``. Floating-point dollars as dict or
   GROUP BY keys are a correctness hazard (``0.1 + 0.2 != 0.3``), so prices live
   on disk as ``int32`` counts of ticks. One tick is ``TICK_SIZE`` dollars. Dollar
   values are derived at the edges (VWAP, returns, display) — never used as keys.

2. **Timestamps are ``int64`` nanoseconds since session open.** Nanosecond epochs
   are what real feeds (ITCH, Databento) carry, they sort and range-join without
   timezone ambiguity, and they keep the as-of joins exact.
"""

from __future__ import annotations

import pyarrow as pa

# One tick = one cent. Prices on disk are integer tick counts; multiply by this
# to get dollars. Chosen so that price*size stays comfortably inside int64 for
# VWAP accumulation without overflow at realistic sizes.
TICK_SIZE: float = 0.01

# Nanoseconds per second / minute — used to build sampling grids and bar edges.
NS_PER_SEC: int = 1_000_000_000
NS_PER_MIN: int = 60 * NS_PER_SEC

# A regular US-equity-style continuous session is 6.5 hours. The simulator lays
# events across this window; the epoch (ns) is measured from session open so that
# day N starts at N * SESSION_NS and multi-day data sorts globally by ts.
SESSION_SECONDS: int = int(6.5 * 3600)
SESSION_NS: int = SESSION_SECONDS * NS_PER_SEC

# Side is stored as a single byte to keep it branch-cheap and unambiguous:
# 'B' = bid/buy resting or buyer-aggressor, 'A' = ask/sell.
SIDE_BID: str = "B"
SIDE_ASK: str = "A"

# ---------------------------------------------------------------------------
# Arrow schemas. These are the on-disk Parquet schemas and the in-memory Arrow
# contract handed to DuckDB and pandas. Field order matters for the columnar
# layout (hot columns first is a minor read win) but the queries name columns
# explicitly, so order is for readability.
# ---------------------------------------------------------------------------

# One executed trade. ``aggressor`` is the side of the order that crossed the
# spread ('B' = a buy market order lifted the ask). ``price`` is the execution
# price in ticks.
TRADES_SCHEMA = pa.schema(
    [
        # Global monotonic sequence. Multiple prints share a nanosecond when an
        # aggressive order sweeps several levels, so ``ts`` alone does not order
        # trades; ``seq`` is the tie-break that makes "first/last trade in a bar"
        # and any trade-level ordering deterministic — exactly why real feeds carry
        # a sequence number.
        pa.field("seq", pa.int64(), nullable=False),
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("price", pa.int32(), nullable=False),
        pa.field("size", pa.int32(), nullable=False),
        pa.field("aggressor", pa.string(), nullable=False),
        pa.field("date", pa.int32(), nullable=False),  # trading day index, for partitioning
    ]
)

# Top-of-book (TBBO). One row every time the best bid or offer — price or size —
# changes. This is the generator's *own record* of the book top and serves as an
# independent oracle for the reconstruction query (Q3).
QUOTES_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("bid_px", pa.int32(), nullable=False),
        pa.field("bid_sz", pa.int32(), nullable=False),
        pa.field("ask_px", pa.int32(), nullable=False),
        pa.field("ask_sz", pa.int32(), nullable=False),
        pa.field("date", pa.int32(), nullable=False),
    ]
)

# Market-by-price incremental updates. Each row changes the resting size at one
# price level by ``delta`` (signed). The resting size at a level is the running
# sum of its deltas; a level is present while that sum is > 0. Reconstruction
# (Q3) turns this stream back into a book. ``seq`` is a global monotonic tie-
# breaker so that same-nanosecond events replay in generation order.
BOOK_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("seq", pa.int64(), nullable=False),
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("side", pa.string(), nullable=False),
        pa.field("price", pa.int32(), nullable=False),
        pa.field("delta", pa.int32(), nullable=False),
        pa.field("date", pa.int32(), nullable=False),
    ]
)

# The three tables of the dataset, by name.
TABLES = ("trades", "quotes", "book_events")


def ticks_to_dollars(ticks) -> float:
    """Convert an integer tick count (or array) to dollars."""
    return ticks * TICK_SIZE
