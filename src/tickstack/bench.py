"""Benchmark harness: time the five queries across engines and layouts, fairly.

"Fairly" is the whole game, and it drives every choice here:

* **Correctness gates timing.** Before any number is reported, each query's DuckDB
  and pandas results are diffed by :mod:`tickstack.verify`. A query that disagrees
  is not timed — a fast wrong answer is not a result.

* **I/O and compute are separated, not conflated.** DuckDB is timed two ways:
  ``duckdb_parquet`` reads from the Parquet files each run (what you pay in a cold
  research query), and ``duckdb_mem`` runs against native DuckDB tables loaded once
  (pure engine compute). The parallel is exact: pandas converts Arrow to DataFrames
  once and reruns; DuckDB loads Parquet into native tables once and reruns. Both
  one-time load costs are reported separately, so the headline ``duckdb_mem`` vs
  ``pandas`` number is compute against compute, with ``duckdb_parquet`` alongside to
  show what layout buys on the read.

  Registering the Arrow tables as *views* and querying those was the first design,
  and it was wrong: DuckDB's ASOF join over the Arrow scan adapter falls back to a
  quadratic plan (~200x slower than native), which measured the adapter, not the
  engine. Loading into native tables once — the thing a researcher's session does —
  is both faster and the honest analogue of pandas holding DataFrames.

* **Warmups, medians, and repeats.** Each cell is a warmup followed by ``repeats``
  timed runs; we report the median (robust to a stray GC pause) plus min and the
  inter-quartile spread. The number of repeats is a knob so CI can run cheaply.

* **Every engine is real and every engine is gated.** DuckDB, pandas and Polars
  run in-process; ClickHouse and QuestDB run as servers (``docker-compose.yml``) and
  are queried over their own client protocols. Each engine's five results are diffed
  against the DuckDB reference (itself checked against pandas and against the
  generator's ground truth) before a single timing is taken. An unreachable server
  is an error unless it was explicitly excluded — the harness never estimates a
  number it did not measure.

* **Same resources.** DuckDB, Polars and both servers are capped at ``threads``
  (DuckDB ``PRAGMA threads``, ``POLARS_MAX_THREADS``, ClickHouse ``max_threads``,
  and the compose file pins each server container to that many CPUs). DuckDB is
  also timed at one thread (``duckdb_1t``) so the parallelism share of its speedup is
  visible rather than folded into "DuckDB is faster".

* **The pandas conversion is amortized honestly.** ``to_pandas`` on the Arrow
  tables happens once per (dataset) and is reused across repeats, because a
  researcher converts once and runs many queries. That conversion cost is reported
  separately so it is visible, not hidden.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow as pa

from . import engine
from . import queries_duckdb as qd
from . import queries_pandas as qp
from .layouts import LAYOUTS, LayoutPaths, dataset_size_bytes
from .verify import _SPECS, frames_match, verify_all

# Query registry: name -> (duckdb callable(con), pandas callable(dfs, grid)).
# The pandas callables take the in-memory frames dict and the sample grid.
QUERIES = ("q1_asof", "q2_vwap", "q3_book", "q4_event", "q5_corr")


@dataclass
class Timing:
    """Timing summary for one (query, engine, layout) cell, in milliseconds."""

    query: str
    engine: str
    layout: str
    median_ms: float
    min_ms: float
    p25_ms: float
    p75_ms: float
    repeats: int
    result_rows: int


@dataclass
class BenchResult:
    """Everything a report needs: timings, sizes, correctness, and provenance."""

    timings: list[Timing] = field(default_factory=list)
    scoped_timings: list[Timing] = field(default_factory=list)  # single symbol-day scan
    scoped_meta: dict = field(default_factory=dict)  # which symbol/day, result rows
    sizes_bytes: dict[str, dict[str, int]] = field(default_factory=dict)  # layout -> table -> bytes
    conversion_ms: dict[str, float] = field(default_factory=dict)  # layout -> to_pandas ms
    load_ms: dict[str, dict[str, float]] = field(default_factory=dict)  # layout -> engine -> load ms
    correctness: dict[str, bool] = field(default_factory=dict)  # query -> duckdb==pandas
    # engine -> query -> matched the DuckDB reference (pandas, polars, servers)
    engine_correctness: dict[str, dict[str, bool]] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)  # table -> row count
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timings"] = [asdict(t) for t in self.timings]
        return d


def _time_call(fn, repeats: int, warmup: int = 1):
    """Run ``fn`` ``warmup`` then ``repeats`` times; return (samples_ms, last_result)."""
    result = None
    for _ in range(warmup):
        result = fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples, result


def _percentile(sorted_vals, p):
    """Linear-interpolated percentile (p in [0,1]) over an already-sorted list.
    Matches numpy's default 'linear' method; avoids the low bias of floored
    nearest-rank indexing at small sample counts."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = p * (n - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= n:
        return sorted_vals[-1]
    return sorted_vals[lo] + frac * (sorted_vals[lo + 1] - sorted_vals[lo])


def _summarize(query, engine_name, layout, samples, rows) -> Timing:
    samples_sorted = sorted(samples)
    q25 = _percentile(samples_sorted, 0.25)
    q75 = _percentile(samples_sorted, 0.75)
    n = len(samples_sorted)
    return Timing(
        query=query,
        engine=engine_name,
        layout=layout,
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        p25_ms=q25,
        p75_ms=q75,
        repeats=n,
        result_rows=rows,
    )


def _rows(obj) -> int:
    if isinstance(obj, pa.Table):
        return obj.num_rows
    try:
        return len(obj)
    except TypeError:
        return -1


def _duckdb_runners(con):
    """Map query name -> zero-arg callable that runs it on ``con``."""
    return {
        "q1_asof": lambda: qd.q1_asof_trades_quotes(con),
        "q2_vwap": lambda: qd.q2_vwap_bars(con),
        "q3_book": lambda: qd.q3_book_reconstruction(con),
        "q4_event": lambda: qd.q4_event_study(con),
        "q5_corr": lambda: qd.q5_cross_symbol_corr(con),
    }


def _pandas_runners(frames, grid_df):
    tr, qo, be = frames["trades"], frames["quotes"], frames["book_events"]
    return {
        "q1_asof": lambda: qp.q1_asof_trades_quotes(tr, qo),
        "q2_vwap": lambda: qp.q2_vwap_bars(tr),
        "q3_book": lambda: qp.q3_book_reconstruction(be, grid_df),
        "q4_event": lambda: qp.q4_event_study(tr, qo),
        "q5_corr": lambda: qp.q5_cross_symbol_corr(qo),
    }


def _polars_runners(pframes, grid_pl):
    from . import queries_polars as ql

    tr, qo, be = pframes["trades"], pframes["quotes"], pframes["book_events"]
    return {
        "q1_asof": lambda: ql.q1_asof_trades_quotes(tr, qo),
        "q2_vwap": lambda: ql.q2_vwap_bars(tr),
        "q3_book": lambda: ql.q3_book_reconstruction(be, grid_pl),
        "q4_event": lambda: ql.q4_event_study(tr, qo),
        "q5_corr": lambda: ql.q5_cross_symbol_corr(qo),
    }


def _as_arrow(obj) -> pa.Table:
    if isinstance(obj, pa.Table):
        return obj
    if hasattr(obj, "to_arrow"):  # polars
        return obj.to_arrow()
    return pa.Table.from_pandas(obj, preserve_index=False)


def _check_against_reference(ref: dict, runners: dict) -> dict[str, bool]:
    """Run each query once via ``runners`` and diff it against the reference result
    with the same per-query spec (exact integers, 1e-12 floats) as DuckDB-vs-pandas."""
    out = {}
    for q in QUERIES:
        spec = _SPECS[q]
        ok, _msg = frames_match(
            _as_arrow(runners[q]()), ref[q], spec["keys"], spec["exact"], spec["approx"]
        )
        out[q] = ok
    return out


SERVER_ENGINES = ("clickhouse", "questdb")


def _make_server(name: str, threads: int):
    from . import servers

    if not servers.available(name):
        raise RuntimeError(
            f"{name} server is not reachable. Start it with `docker compose up -d` "
            f"or exclude it with --no-servers / --engines."
        )
    return servers.ClickHouseEngine(threads=threads) if name == "clickhouse" \
        else servers.QuestDBEngine()


def run_scoped_benchmark(
    base: Path,
    layouts=LAYOUTS,
    repeats: int = 7,
    threads: int = 4,
) -> list[Timing]:
    """The far more common desk query — "pull one name for one day" — and the layout
    result that is *not* what intuition predicts.

    The full-history table times queries that scan everything, where partitioning can
    only add file-open overhead. This scoped pass is the counterpart. The intuitive
    expectation is that ``by_date_symbol`` wins by pruning to one file; the measured
    result is that it buys nothing at this scale (every layout is within a few ms)
    and costs more as history grows (``scripts/run_partition_study.py``), because one
    file per symbol-day is the small-files anti-pattern. ``single`` stays competitive because row-group
    statistics prune it even without a directory layout (the generator writes
    day-monotonic, so ``date`` min/max skips most row groups). The takeaway is that at
    this scale directory partitioning does not earn its overhead — a scale-dependent
    lesson, not a "partitioning wins" one. Reported next to the full-scan table.

    Timed from Parquet (fresh connection per run) because pruning is a *scan-time*
    decision — loading into memory first would erase the very effect being measured.
    The timer starts *after* connection + view registration so it measures the query,
    not registration metadata cost (which scales with file count). The scoped query is
    the Q1 as-of join restricted to the median symbol on the middle day, chosen
    deterministically from the data."""
    timings: list[Timing] = []

    # Pick a deterministic (symbol, date): middle day, lexicographically middle symbol.
    probe = engine.connect(threads=threads)
    engine.register_layout(probe, LayoutPaths(root=base / "single", kind="single"))
    syms = [r[0] for r in probe.execute(
        "SELECT DISTINCT symbol FROM trades ORDER BY symbol"
    ).fetchall()]
    dates = [r[0] for r in probe.execute(
        "SELECT DISTINCT date FROM trades ORDER BY date"
    ).fetchall()]
    probe.close()
    sym = syms[len(syms) // 2]
    day = dates[len(dates) // 2]

    # Scope BOTH sides to (symbol, date) *before* the ASOF join, in derived tables,
    # so the date/symbol predicate pushes into each Parquet scan (enabling partition
    # pruning and row-group skipping) and the join only ever sees same-day quotes.
    # Filtering after the join would be both slower (no pruning) and wrong (the join
    # could pick a prior-day quote that the filter then nulls out).
    scoped_sql = """
        WITH t AS (
            SELECT seq, ts, symbol, price, size
            FROM trades WHERE symbol = ? AND date = ?
        ),
        q AS (
            SELECT ts, symbol, bid_px, ask_px
            FROM quotes WHERE symbol = ? AND date = ?
        )
        SELECT t.seq, t.ts, t.symbol, t.price AS trade_px, t.size AS trade_sz,
               q.bid_px, q.ask_px
        FROM t
        ASOF LEFT JOIN q
          ON t.symbol = q.symbol AND t.ts >= q.ts
        ORDER BY t.ts, t.seq
    """

    # Time only the query, not connection setup or view registration. Registering a
    # Hive dataset opens Parquet footers for every matched file, so if that sat inside
    # the timer it would scale with file count — inflating the many-files layouts by a
    # constant that has nothing to do with the scoped query's scan/prune cost. A fresh
    # connection per run still gives a cold query (no cached native tables); the timer
    # starts after registration. This matches scripts/run_partition_study.py exactly.
    def scoped_factory(paths):
        def run(con):
            return con.sql(scoped_sql, params=[sym, day, sym, day]).to_arrow_table()
        return run

    for layout in layouts:
        paths = LayoutPaths(root=base / layout, kind=layout)
        fn = scoped_factory(paths)
        samples = []
        out = None
        for r in range(repeats + 1):  # 1 warmup
            con = engine.connect(threads=threads)
            engine.register_layout(con, paths)
            t0 = time.perf_counter()
            out = fn(con)
            dt = (time.perf_counter() - t0) * 1000.0
            con.close()
            if r > 0:
                samples.append(dt)
        timings.append(_summarize("scoped_asof", "duckdb_parquet", layout, samples, _rows(out)))
    return timings


def run_benchmark(
    base: Path,
    layouts=LAYOUTS,
    repeats: int = 7,
    threads: int = 4,
    run_pandas: bool = True,
    run_polars: bool = True,
    server_engines=SERVER_ENGINES,
    compute_layout: str | None = None,
) -> BenchResult:
    """Run the full benchmark against pre-written layouts under ``base``.

    ``compute_layout`` (default ``sorted``) is the layout whose data feeds the
    headline compute table. DuckDB-in-memory and pandas are timed on every layout
    (as before); DuckDB at 1 thread, Polars and the servers run their queries on
    ``compute_layout`` only — once data is inside an engine its on-disk layout no
    longer matters — while their one-time *load* is measured on every layout."""
    compute_layout = compute_layout or ("sorted" if "sorted" in layouts else layouts[0])
    res = BenchResult()
    res.meta = {
        "repeats": repeats, "threads": threads, "layouts": list(layouts),
        "compute_layout": compute_layout,
    }

    # Correctness first, on the canonical single-file layout, so timing never
    # reports a wrong answer. Uses the default query params throughout.
    con0 = engine.connect(threads=threads)
    engine.register_layout(con0, LayoutPaths(root=base / "single", kind="single"))
    grid0 = engine.build_sample_grid(con0)
    arrow0 = engine.load_tables_arrow(con0)
    frames0 = {k: v.to_pandas() for k, v in arrow0.items()}
    res.rows = {k: int(v.shape[0]) for k, v in frames0.items()}
    res.correctness = verify_all(con0, frames0, grid0.to_pandas())
    res.engine_correctness["pandas"] = dict(res.correctness)
    ref = {q: fn() for q, fn in _duckdb_runners(con0).items()}

    if run_polars:
        import polars as pl

        from . import queries_polars as ql

        res.engine_correctness["polars"] = _check_against_reference(
            ref, _polars_runners(ql.frames_from_arrow(arrow0), pl.from_arrow(grid0))
        )
    servers_live = {}
    for name in server_engines:
        srv = _make_server(name, threads)
        srv.load(arrow0, grid0)
        res.engine_correctness[name] = _check_against_reference(ref, srv.runners())
        res.meta.setdefault("server_versions", {})[name] = srv.version()
        servers_live[name] = srv
    con0.close()

    bad = [f"{e}:{q}" for e, qs in res.engine_correctness.items() for q, ok in qs.items() if not ok]
    if bad:
        raise AssertionError(f"correctness check failed for {bad}; refusing to report timings")

    # Build the Q3 sample grid once, up front, as an Arrow table. It is an input to
    # Q3, not part of the query, and it is layout-independent — so rebuilding it
    # inside a timed run (as an earlier version did) measured grid construction, not
    # the query, and did so more for layouts with more files. Every timed path below
    # registers this precomputed grid cheaply instead of recomputing it.
    grid_arrow = grid0

    def _register_grid(con):
        con.register("sample_grid_arrow", grid_arrow)
        con.execute("CREATE OR REPLACE VIEW sample_grid AS SELECT * FROM sample_grid_arrow")

    for layout in layouts:
        paths = LayoutPaths(root=base / layout, kind=layout)
        is_compute = layout == compute_layout
        res.sizes_bytes[layout] = {
            t: dataset_size_bytes(paths, t) for t in ("trades", "quotes", "book_events")
        }

        # --- duckdb_parquet: fresh connection + view per run, reads Parquet ---
        # The timed region is the cold-query cost: connection + view registration +
        # the query. The sample grid is registered from the precomputed Arrow table
        # (a cheap, layout-independent constant), not rebuilt, so the Q3 row measures
        # the query and not grid construction.
        def duck_parquet_factory(qname, paths=paths):
            def run():
                con = engine.connect(threads=threads)
                engine.register_layout(con, paths)
                if qname == "q3_book":
                    _register_grid(con)
                out = _duckdb_runners(con)[qname]()
                con.close()
                return out
            return run

        for qname in QUERIES:
            samples, out = _time_call(duck_parquet_factory(qname), repeats)
            res.timings.append(_summarize(qname, "duckdb_parquet", layout, samples, _rows(out)))

        # --- one-time in-memory load, measured symmetrically for every engine ---
        # Every engine starts from the same Parquet-backed views and is timed over
        # the SAME work: read Parquet -> Arrow, then materialize into the engine's own
        # format (native DuckDB tables / DataFrames / server tables). The shared
        # Parquet read is inside every timer.
        con = engine.connect(threads=threads)
        engine.register_layout(con, paths)
        lm = res.load_ms.setdefault(layout, {})

        t0 = time.perf_counter()
        con2 = engine.load_native(con, threads=threads)  # Parquet->Arrow + native copy
        lm["duckdb"] = (time.perf_counter() - t0) * 1000.0
        _register_grid(con2)
        for qname in QUERIES:
            samples, out = _time_call(_duckdb_runners(con2)[qname], repeats)
            res.timings.append(_summarize(qname, "duckdb_mem", layout, samples, _rows(out)))
        con2.close()

        # --- duckdb_1t: same native tables, one thread — isolates parallelism ---
        if is_compute:
            con1 = engine.load_native(con, threads=1)
            _register_grid(con1)
            for qname in QUERIES:
                samples, out = _time_call(_duckdb_runners(con1)[qname], repeats)
                res.timings.append(_summarize(qname, "duckdb_1t", layout, samples, _rows(out)))
            con1.close()

        # --- pandas: read Parquet->Arrow AND convert to DataFrames, both timed ---
        if run_pandas:
            t0 = time.perf_counter()
            arrs = engine.load_tables_arrow(con)  # Parquet->Arrow (the same read)
            frames = {k: v.to_pandas() for k, v in arrs.items()}
            lm["pandas"] = (time.perf_counter() - t0) * 1000.0
            res.conversion_ms[layout] = lm["pandas"]
            pruns = _pandas_runners(frames, grid_arrow.to_pandas())
            for qname in QUERIES:
                samples, out = _time_call(pruns[qname], repeats)
                res.timings.append(_summarize(qname, "pandas", layout, samples, _rows(out)))
            del frames, arrs

        # --- polars: Parquet->Arrow + from_arrow timed; queries on compute layout ---
        if run_polars:
            import polars as pl

            from . import queries_polars as ql

            t0 = time.perf_counter()
            pframes = ql.frames_from_arrow(engine.load_tables_arrow(con))
            lm["polars"] = (time.perf_counter() - t0) * 1000.0
            if is_compute:
                pruns = _polars_runners(pframes, pl.from_arrow(grid_arrow))
                for qname in QUERIES:
                    samples, out = _time_call(pruns[qname], repeats)
                    res.timings.append(_summarize(qname, "polars", layout, samples, _rows(out)))
            del pframes

        # --- servers: Parquet->Arrow + server ingest timed; queries on compute layout ---
        for name, srv in servers_live.items():
            t0 = time.perf_counter()
            arrs = engine.load_tables_arrow(con)
            read_ms = (time.perf_counter() - t0) * 1000.0
            lm[name] = read_ms + srv.load(arrs, grid_arrow)
            if is_compute:
                sruns = srv.runners()
                for qname in QUERIES:
                    samples, out = _time_call(sruns[qname], repeats)
                    res.timings.append(_summarize(qname, name, layout, samples, _rows(out)))
        con.close()

    for srv in servers_live.values():
        srv.close()

    # The scoped (single symbol-day) pass — the layout result that isn't intuitive.
    res.scoped_timings = run_scoped_benchmark(base, layouts=layouts, repeats=repeats, threads=threads)
    if res.scoped_timings:
        res.scoped_meta = {"result_rows": res.scoped_timings[0].result_rows}

    return res
