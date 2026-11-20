"""Correctness checks that gate the benchmark.

Two independent kinds of check, both of which must pass before any timing is
reported:

1. **Engine agreement** — every query's DuckDB result equals its pandas result.
   This catches a bug in *either* implementation: they were written from the same
   English spec but with different primitives (SQL windows vs pandas groupby), so a
   silent divergence almost always means one side is wrong.

2. **Ground-truth agreement for Q3** — the reconstructed book, sampled at the
   generator's own quote timestamps, equals the generator's ``quotes`` table. This
   is the strongest check available: it compares against data neither query
   produced, so it cannot be satisfied by two implementations sharing a mistake.

Comparisons are exact where the math is exact (integer prices, sizes, counts) and
use a tight tolerance only for genuinely floating-point outputs (VWAP, returns,
correlation), where DuckDB and pandas may sum in a different order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from . import queries_duckdb as qd
from . import queries_pandas as qp

# Per-query comparison spec: key columns, exact-integer columns, float columns.
_SPECS = {
    "q1_asof": {
        "keys": ["symbol", "seq"],
        "exact": ["trade_px", "trade_sz", "bid_px", "bid_sz", "ask_px", "ask_sz"],
        "approx": [],
    },
    "q2_vwap": {
        "keys": ["symbol", "bar"],
        "exact": ["pv", "volume", "open_px", "high_px", "low_px", "close_px", "n_trades"],
        "approx": ["vwap_ticks"],
    },
    "q3_book": {
        "keys": ["symbol", "sample_ts"],
        "exact": ["bid_px", "bid_sz", "ask_px", "ask_sz"],
        "approx": [],
    },
    "q4_event": {
        "keys": ["symbol"],
        "exact": ["n_events"],
        "approx": ["mean_fwd_bps"],
    },
    "q5_corr": {
        "keys": ["sym_a", "sym_b"],
        "exact": ["n_bars"],
        "approx": ["correlation"],
    },
}


def _to_pandas(obj) -> pd.DataFrame:
    return obj.to_pandas() if isinstance(obj, pa.Table) else obj


def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns and str(df[c].dtype) == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def frames_match(
    duck, pan, keys, exact_cols, approx_cols, atol: float = 1e-12, rtol: float = 1e-12
) -> tuple[bool, str]:
    """Return (matched, message) for two result frames."""
    d = _coerce_numeric(_to_pandas(duck), exact_cols + approx_cols + keys)
    p = _coerce_numeric(_to_pandas(pan), exact_cols + approx_cols + keys)
    if len(d) != len(p):
        return False, f"row count {len(d)} != {len(p)}"
    d = d.sort_values(keys, kind="stable").reset_index(drop=True)
    p = p.sort_values(keys, kind="stable").reset_index(drop=True)

    for c in keys:
        if not (d[c].to_numpy() == p[c].to_numpy()).all():
            return False, f"key column {c!r} differs after sort"
    for c in exact_cols:
        dv, pv = d[c].to_numpy(), p[c].to_numpy()
        # Exact integer equality (allowing NaN==NaN for LEFT-join misses).
        dv_f = dv.astype("float64")
        pv_f = pv.astype("float64")
        if not np.array_equal(dv_f, pv_f, equal_nan=True):
            n_bad = int((~((dv_f == pv_f) | (np.isnan(dv_f) & np.isnan(pv_f)))).sum())
            return False, f"exact column {c!r}: {n_bad} mismatches"
    for c in approx_cols:
        dv = d[c].to_numpy().astype("float64")
        pv = p[c].to_numpy().astype("float64")
        if not np.allclose(dv, pv, atol=atol, rtol=rtol, equal_nan=True):
            n_bad = int((~np.isclose(dv, pv, atol=atol, rtol=rtol, equal_nan=True)).sum())
            return False, f"approx column {c!r}: {n_bad} beyond tol (atol={atol})"
    return True, "ok"


def verify_all(con, frames: dict, grid_df: pd.DataFrame) -> dict[str, bool]:
    """Run every query on both engines and return {query: matched}."""
    tr, qo, be = frames["trades"], frames["quotes"], frames["book_events"]
    duck = {
        "q1_asof": qd.q1_asof_trades_quotes(con),
        "q2_vwap": qd.q2_vwap_bars(con),
        "q3_book": qd.q3_book_reconstruction(con),
        "q4_event": qd.q4_event_study(con),
        "q5_corr": qd.q5_cross_symbol_corr(con),
    }
    pan = {
        "q1_asof": qp.q1_asof_trades_quotes(tr, qo),
        "q2_vwap": qp.q2_vwap_bars(tr),
        "q3_book": qp.q3_book_reconstruction(be, grid_df),
        "q4_event": qp.q4_event_study(tr, qo),
        "q5_corr": qp.q5_cross_symbol_corr(qo),
    }
    out = {}
    for q, spec in _SPECS.items():
        ok, _ = frames_match(duck[q], pan[q], spec["keys"], spec["exact"], spec["approx"])
        out[q] = ok
    return out


def verify_q3_against_ground_truth(con) -> tuple[int, int]:
    """Reconstruct the book at the generator's own quote timestamps and compare to
    ``quotes``. Returns (matched, total). This is the ground-truth check: it does
    not consult the pandas baseline at all.

    Temporarily points ``sample_grid`` at the distinct quote timestamps, so the
    caller should rebuild the minute grid afterward if it needs it."""
    con.execute(
        "CREATE OR REPLACE VIEW sample_grid AS "
        "SELECT DISTINCT symbol, ts AS sample_ts FROM quotes"
    )
    recon = qd.q3_book_reconstruction(con)
    con.register("_recon_gt", recon)
    # LEFT JOIN from quotes so the denominator is the true number of distinct quote
    # timestamps, not the inner-join count. A quote timestamp where Q3 produced *no*
    # row (r.* is NULL) then counts as a failure rather than being silently dropped
    # from both numerator and denominator — the check must catch omissions, not only
    # wrong values.
    row = con.execute(
        """
        WITH q AS (SELECT DISTINCT symbol, ts, bid_px, bid_sz, ask_px, ask_sz FROM quotes)
        SELECT count(*) AS n,
               sum(CASE WHEN r.bid_px=q.bid_px AND r.bid_sz=q.bid_sz
                         AND r.ask_px=q.ask_px AND r.ask_sz=q.ask_sz
                        THEN 1 ELSE 0 END) AS matches
        FROM q
        LEFT JOIN _recon_gt r
          ON r.symbol=q.symbol AND r.sample_ts=q.ts
        """
    ).fetchall()[0]
    con.unregister("_recon_gt")
    total, matches = int(row[0]), int(row[1] or 0)
    return matches, total
