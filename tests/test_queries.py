"""Tests that the DuckDB and pandas implementations of all five queries agree, and
that Q3 reconstructs the generator's own quotes exactly. These are the checks the
benchmark itself runs before reporting a timing; here they run at multiple dataset
shapes and query parameters so a divergence that only shows up at a particular bar
width or horizon is still caught."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tickstack import engine  # noqa: E402
from tickstack import queries_duckdb as qd
from tickstack import queries_pandas as qp
from tickstack.generator import SimConfig, Simulator  # noqa: E402
from tickstack.layouts import LAYOUTS, LayoutPaths, build_all_layouts  # noqa: E402
from tickstack.schema import NS_PER_MIN, NS_PER_SEC  # noqa: E402
from tickstack.verify import (  # noqa: E402
    frames_match,
    verify_all,
    verify_q3_against_ground_truth,
)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("layouts")
    cfg = SimConfig.quick()
    build_all_layouts(base, Simulator(cfg).simulate())
    con = engine.connect(threads=2)
    engine.register_layout(con, LayoutPaths(root=base / "single", kind="single"))
    grid = engine.build_sample_grid(con)
    frames = {k: v.to_pandas() for k, v in engine.load_tables_arrow(con).items()}
    return {"base": base, "con": con, "grid": grid.to_pandas(), "frames": frames}


def test_all_queries_agree_default_params(env):
    res = verify_all(env["con"], env["frames"], env["grid"])
    assert all(res.values()), res


def test_q3_matches_generator_ground_truth(env):
    matched, total = verify_q3_against_ground_truth(env["con"])
    assert total > 0
    assert matched == total, f"{matched}/{total} reconstructed tops match quotes"
    # Rebuild the minute grid that verify_q3 replaced, for later tests.
    engine.build_sample_grid(env["con"])


@pytest.mark.parametrize("bar_ns", [NS_PER_MIN, 5 * NS_PER_MIN, 30 * NS_PER_SEC])
def test_q2_vwap_agrees_across_bar_widths(env, bar_ns):
    d = qd.q2_vwap_bars(env["con"], bar_ns=bar_ns)
    p = qp.q2_vwap_bars(env["frames"]["trades"], bar_ns=bar_ns)
    ok, msg = frames_match(
        d, p, ["symbol", "bar"],
        ["pv", "volume", "open_px", "high_px", "low_px", "close_px", "n_trades"],
        ["vwap_ticks"],
    )
    assert ok, msg


@pytest.mark.parametrize("bar_ns", [NS_PER_MIN, 2 * NS_PER_MIN])
def test_q5_corr_agrees_across_bar_widths(env, bar_ns):
    d = qd.q5_cross_symbol_corr(env["con"], bar_ns=bar_ns)
    p = qp.q5_cross_symbol_corr(env["frames"]["quotes"], bar_ns=bar_ns)
    ok, msg = frames_match(d, p, ["sym_a", "sym_b"], ["n_bars"], ["correlation"])
    assert ok, msg


@pytest.mark.parametrize("horizon_ns,top_frac", [(5 * NS_PER_SEC, 0.02), (1 * NS_PER_SEC, 0.05)])
def test_q4_event_study_agrees_across_params(env, horizon_ns, top_frac):
    d = qd.q4_event_study(env["con"], horizon_ns=horizon_ns, top_frac=top_frac)
    p = qp.q4_event_study(env["frames"]["trades"], env["frames"]["quotes"],
                          horizon_ns=horizon_ns, top_frac=top_frac)
    ok, msg = frames_match(d, p, ["symbol"], ["n_events"], ["mean_fwd_bps"])
    assert ok, msg


def test_q1_asof_left_join_keeps_all_trades(env):
    """A LEFT ASOF join must return exactly one row per trade."""
    d = qd.q1_asof_trades_quotes(env["con"])
    n_trades = env["con"].execute("SELECT count(*) FROM trades").fetchone()[0]
    assert d.num_rows == n_trades


@pytest.mark.parametrize("layout", LAYOUTS)
def test_layouts_hold_identical_data(env, layout):
    """Every layout must expose identical row counts and the same columns — they are
    the same bytes, physically reorganized."""
    con = engine.connect(threads=2)
    engine.register_layout(con, LayoutPaths(root=env["base"] / layout, kind=layout))
    for table in ("trades", "quotes", "book_events"):
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        n0 = env["con"].execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        assert n == n0, f"{layout}.{table}: {n} != {n0}"
    con.close()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_q1_agrees_across_layouts(env, layout):
    """The as-of join result must be layout-independent (same answer, any layout)."""
    con = engine.connect(threads=2)
    engine.register_layout(con, LayoutPaths(root=env["base"] / layout, kind=layout))
    d_layout = qd.q1_asof_trades_quotes(con)
    d_single = qd.q1_asof_trades_quotes(env["con"])
    ok, msg = frames_match(
        d_layout, d_single, ["symbol", "seq"],
        ["trade_px", "trade_sz", "bid_px", "bid_sz", "ask_px", "ask_sz"], [],
    )
    con.close()
    assert ok, f"{layout}: {msg}"
