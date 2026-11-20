"""Every non-reference engine — Polars in-process, ClickHouse and QuestDB as servers —
must return the DuckDB reference answer on all five queries (integers exact, floats
within 1e-12). These are the same checks ``run_benchmark`` performs before timing.

The server tests need ``docker compose up -d``; without it they are *skipped* (and
pytest says so), never silently passed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tickstack import engine, servers  # noqa: E402
from tickstack.bench import (  # noqa: E402
    QUERIES,
    _check_against_reference,
    _duckdb_runners,
    _polars_runners,  # noqa: E402
)
from tickstack.generator import SimConfig, Simulator  # noqa: E402
from tickstack.layouts import LayoutPaths, build_all_layouts  # noqa: E402


@pytest.fixture(scope="module")
def ref_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("eng")
    build_all_layouts(base, Simulator(SimConfig.quick()).simulate(), layouts=("single",))
    con = engine.connect(threads=2)
    engine.register_layout(con, LayoutPaths(root=base / "single", kind="single"))
    grid = engine.build_sample_grid(con)
    tables = engine.load_tables_arrow(con)
    ref = {q: fn() for q, fn in _duckdb_runners(con).items()}
    yield {"tables": tables, "grid": grid, "ref": ref}
    con.close()


def test_polars_matches_reference(ref_env):
    pl = pytest.importorskip("polars")
    from tickstack import queries_polars as ql

    runners = _polars_runners(
        ql.frames_from_arrow(ref_env["tables"]), pl.from_arrow(ref_env["grid"])
    )
    assert _check_against_reference(ref_env["ref"], runners) == {q: True for q in QUERIES}


@pytest.mark.parametrize("name", ["clickhouse", "questdb"])
def test_server_matches_reference(ref_env, name):
    if not servers.available(name):
        pytest.skip(f"{name} not running (docker compose up -d)")
    srv = servers.ClickHouseEngine(threads=2) if name == "clickhouse" else servers.QuestDBEngine()
    srv.load(ref_env["tables"], ref_env["grid"])
    assert _check_against_reference(ref_env["ref"], srv.runners()) == {q: True for q in QUERIES}
    srv.close()


def test_clickhouse_log_is_not_trusted_for_returns():
    """Regression guard for the porting trap documented in queries_clickhouse: Q5
    must not use ClickHouse's approximate ``log`` for returns."""
    from tickstack import queries_clickhouse as qch

    sql = "\n".join(line.split("--")[0] for line in qch.q5_cross_symbol_corr_sql().splitlines())
    assert "log1p(" in sql
    assert " log(" not in sql and "(log(" not in sql
