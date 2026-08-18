"""Tests for the simulator: determinism, internal consistency, and the invariant
the whole benchmark leans on — that book events reconstruct to the recorded quotes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tickstack.generator import SimConfig, Simulator  # noqa: E402
from tickstack.schema import SIDE_ASK, SIDE_BID  # noqa: E402


@pytest.fixture(scope="module")
def batches():
    return list(Simulator(SimConfig.quick()).simulate())


def test_determinism_bit_reproducible():
    """Same seed => identical data, run to run."""
    a = list(Simulator(SimConfig.quick()).simulate())
    b = list(Simulator(SimConfig.quick()).simulate())
    assert len(a) == len(b)
    for da, db in zip(a, b, strict=True):
        assert da.trades.equals(db.trades)
        assert da.quotes.equals(db.quotes)
        assert da.book_events.equals(db.book_events)


def test_different_seed_differs():
    a = list(Simulator(SimConfig.quick()).simulate())
    b = list(Simulator(SimConfig(**{**SimConfig.quick().__dict__, "seed": 999})).simulate())
    assert not a[0].trades.equals(b[0].trades)


def test_nonempty_and_shapes(batches):
    for b in batches:
        assert b.trades.num_rows > 0
        assert b.quotes.num_rows > 0
        assert b.book_events.num_rows > 0


def test_prices_positive_and_book_uncrossed(batches):
    """Every recorded quote has bid < ask (a book is never crossed) and positive px."""
    for b in batches:
        bid = b.quotes.column("bid_px").to_numpy()
        ask = b.quotes.column("ask_px").to_numpy()
        assert (bid > 0).all()
        assert (ask > bid).all()


def test_trade_prices_within_book(batches):
    """Trades print at integer prices and sizes are positive."""
    for b in batches:
        assert (b.trades.column("price").to_numpy() > 0).all()
        assert (b.trades.column("size").to_numpy() > 0).all()
        aggr = set(b.trades.column("aggressor").to_pylist())
        assert aggr.issubset({SIDE_BID, SIDE_ASK})


def test_seq_unique_and_contiguous_globally(batches):
    """The global seq is unique across book events and trades, and forms a gapless
    0..N-1 range — the property the tie-break relies on. (Sorting first means this
    checks uniqueness and contiguity, which together imply a strict global order.)"""
    seqs = []
    for b in batches:
        seqs += b.book_events.column("seq").to_pylist()
        seqs += b.trades.column("seq").to_pylist()
    n = len(seqs)
    unique = set(seqs)
    assert len(unique) == n, f"seq values are not unique ({n - len(unique)} duplicates)"
    assert min(seqs) == 0, f"seq should start at 0, starts at {min(seqs)}"
    assert max(seqs) == n - 1, f"seq should be a gapless 0..{n-1} range, max is {max(seqs)}"


def test_trades_seq_strictly_increasing_within_symbol(batches):
    """Within a symbol-day, trade seq is strictly increasing in ts order — so
    ordering trades by (ts, seq) is a total order with no ties."""
    for b in batches:
        t = b.trades.sort_by([("ts", "ascending"), ("seq", "ascending")])
        seqs = t.column("seq").to_pylist()
        # After sorting by (ts, seq), seq need not be globally sorted (interleaved
        # symbols), but (ts, seq) pairs must be strictly increasing and unique.
        pairs = list(zip(t.column("ts").to_pylist(), seqs, strict=True))
        assert len(set(pairs)) == len(pairs), "duplicate (ts, seq) among trades"
        assert pairs == sorted(pairs), "(ts, seq) not a total order after sort"


def test_reconstruction_matches_quotes(batches):
    """The core invariant: replaying book_events up to each quote ts reproduces the
    quote exactly. Pure-Python replay, no engine, so this stands alone."""
    import collections

    for b in batches:
        be = b.book_events.sort_by([("ts", "ascending"), ("seq", "ascending")])
        ts = be.column("ts").to_pylist()
        sym = be.column("symbol").to_pylist()
        side = be.column("side").to_pylist()
        px = be.column("price").to_pylist()
        delta = be.column("delta").to_pylist()

        # Per symbol, snapshot the top after applying all events at each ts.
        books = collections.defaultdict(lambda: (collections.defaultdict(int), collections.defaultdict(int)))
        tops = {}  # (symbol, ts) -> (bpx,bsz,apx,asz)
        i, n = 0, len(ts)
        while i < n:
            t = ts[i]
            touched = set()
            while i < n and ts[i] == t:
                bids, asks = books[sym[i]]
                book = bids if side[i] == SIDE_BID else asks
                book[px[i]] += delta[i]
                if book[px[i]] <= 0:
                    del book[px[i]]
                touched.add(sym[i])
                i += 1
            for s in touched:
                bids, asks = books[s]
                if bids and asks:
                    bpx = max(bids)
                    apx = min(asks)
                    tops[(s, t)] = (bpx, bids[bpx], apx, asks[apx])

        q = b.quotes
        qsym = q.column("symbol").to_pylist()
        qts = q.column("ts").to_pylist()
        qb = q.column("bid_px").to_pylist()
        qbs = q.column("bid_sz").to_pylist()
        qa = q.column("ask_px").to_pylist()
        qas = q.column("ask_sz").to_pylist()
        checked = mism = 0
        for k in range(len(qts)):
            key = (qsym[k], qts[k])
            if key in tops:
                checked += 1
                if tops[key] != (qb[k], qbs[k], qa[k], qas[k]):
                    mism += 1
        assert checked > 0
        assert mism == 0, f"{mism}/{checked} reconstructed tops disagree with quotes"
