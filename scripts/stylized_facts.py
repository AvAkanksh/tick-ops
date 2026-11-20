#!/usr/bin/env python3
"""Measure which empirical stylized facts the synthetic market does — and does not —
reproduce.

The generator is built for *benchmark* properties (consistent streams, an exact
reconstruction oracle, real cross-symbol structure). Whether it also looks like a
real market statistically is a separate question, and it should be answered with
numbers rather than asserted. This script computes, on the default dataset, the
standard checklist from Cont (2001), "Empirical properties of asset returns":

* **No linear autocorrelation of returns** — lag-1 ACF of 1-minute mid log-returns ≈ 0.
* **Heavy tails** — excess kurtosis of 1-minute returns > 0 (Gaussian = 0).
* **Volatility clustering** — ACF of |r| positive and slowly decaying across lags.
* **Cross-sectional correlation** — positive average pairwise return correlation.
* **Bid-ask bounce in trades** — negative lag-1 ACF of trade-to-trade price changes.

Output: ``results/stylized_facts.md`` and ``.json``. The verdicts are mechanical
(thresholds stated in the table), so the doc cannot overclaim.

Usage:
    python scripts/stylized_facts.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from tickstack import engine  # noqa: E402
from tickstack.generator import SimConfig, Simulator  # noqa: E402
from tickstack.layouts import LayoutPaths, build_all_layouts  # noqa: E402
from tickstack.schema import NS_PER_MIN  # noqa: E402


def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    d = float((x * x).sum())
    return float((x[:-lag] * x[lag:]).sum() / d) if d > 0 else float("nan")


def _excess_kurtosis(x: np.ndarray) -> float:
    x = x - x.mean()
    m2 = float((x**2).mean())
    return float((x**4).mean() / m2**2 - 3.0)


def main() -> int:
    data_dir = _ROOT / "data" / "stylized"
    cfg = SimConfig()
    build_all_layouts(data_dir, Simulator(cfg).simulate(), layouts=("single",))
    con = engine.connect(threads=4)
    engine.register_layout(con, LayoutPaths(root=data_dir / "single", kind="single"))

    # 1-minute last-mid log returns per symbol, within a day (no overnight return).
    rets = con.sql(
        f"""
        WITH bar_mid AS (
            SELECT symbol, date, ts // {NS_PER_MIN} AS bar,
                   last((bid_px + ask_px) / 2.0 ORDER BY ts) AS mid
            FROM quotes GROUP BY ALL
        )
        SELECT symbol, date, bar,
               ln(mid / lag(mid) OVER (PARTITION BY symbol, date ORDER BY bar)) AS r
        FROM bar_mid
        QUALIFY r IS NOT NULL
        ORDER BY symbol, date, bar
        """
    ).df()
    trades = con.sql(
        "SELECT symbol, date, price FROM trades ORDER BY symbol, date, ts, seq"
    ).df()

    per_symbol = {}
    for sym, g in rets.groupby("symbol"):
        r = g["r"].to_numpy()
        a = np.abs(r)
        t = trades[trades["symbol"] == sym]
        dp = np.concatenate(
            [np.diff(d["price"].to_numpy().astype("float64")) for _, d in t.groupby("date")]
        )
        per_symbol[sym] = {
            "n_returns": int(r.size),
            "acf_r_lag1": _acf(r, 1),
            "excess_kurtosis": _excess_kurtosis(r),
            "acf_abs_r": {str(k): _acf(a, k) for k in (1, 5, 10, 30)},
            "acf_trade_dp_lag1": _acf(dp, 1),
        }

    wide = rets.pivot_table(index=["date", "bar"], columns="symbol", values="r")
    c = wide.corr().to_numpy()
    mean_pair_corr = float(c[np.triu_indices_from(c, k=1)].mean())

    med = lambda key: float(np.median([v[key] for v in per_symbol.values()]))  # noqa: E731
    med_abs = {k: float(np.median([v["acf_abs_r"][k] for v in per_symbol.values()]))
               for k in ("1", "5", "10", "30")}
    facts = [
        ("No linear autocorrelation of returns", "median lag-1 ACF(r)",
         med("acf_r_lag1"), "|x| < 0.05", abs(med("acf_r_lag1")) < 0.05),
        ("Heavy tails", "median excess kurtosis(r)",
         med("excess_kurtosis"), "> 1", med("excess_kurtosis") > 1.0),
        ("Volatility clustering", "median ACF(|r|) at lag 1 / 5 / 10 / 30",
         med_abs, "lag-1 > 0.1 and lag-10 > 0.05",
         med_abs["1"] > 0.1 and med_abs["10"] > 0.05),
        ("Cross-sectional correlation", "mean pairwise corr(r)",
         mean_pair_corr, "> 0.1", mean_pair_corr > 0.1),
        ("Bid-ask bounce", "median lag-1 ACF of trade price changes",
         med("acf_trade_dp_lag1"), "< -0.05", med("acf_trade_dp_lag1") < -0.05),
    ]

    out = {
        "config": {"seed": cfg.seed, "n_days": cfg.n_days, "symbols": list(cfg.symbols),
                   "events_per_symbol_day": cfg.events_per_symbol_day, "bar": "1 min"},
        "facts": [{"fact": f, "statistic": s, "value": v, "threshold": th, "reproduced": ok}
                  for f, s, v, th, ok in facts],
        "per_symbol": per_symbol,
    }
    res_dir = _ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "stylized_facts.json").write_text(json.dumps(out, indent=2))

    def fmt(v):
        if isinstance(v, dict):
            return " / ".join(f"{x:+.3f}" for x in v.values())
        return f"{v:+.3f}"

    L = ["# Stylized facts of the synthetic market\n",
         "Generated by `scripts/stylized_facts.py` on the default benchmark dataset "
         f"({cfg.n_days} days × {len(cfg.symbols)} symbols, seed {cfg.seed}), 1-minute "
         "mid log-returns within each day. Checklist from Cont (2001). Verdicts are "
         "mechanical against the stated thresholds.\n",
         "| Fact | Statistic | Value | Threshold | Reproduced? |",
         "|---|---|--:|---|:-:|"]
    for f, s, v, th, ok in facts:
        L.append(f"| {f} | {s} | {fmt(v)} | {th} | {'yes' if ok else '**no**'} |")
    L.append("")
    missing = [f for f, *_rest, ok in facts if not ok]
    if missing:
        L.append("**Not reproduced:** " + ", ".join(missing) + ".\n")
    if "Volatility clustering" in missing:
        L.append(
            "- *No volatility clustering — by design.* `generator.py` uses a constant-σ "
            "Gaussian market factor, Gaussian idiosyncratic steps and constant-rate "
            "exponential inter-arrival times; there is no GARCH-like, Hawkes or regime "
            "process, so |r| has essentially no memory.\n"
        )
    if "No linear autocorrelation of returns" in missing:
        L.append(
            "- *Positive return autocorrelation — a side effect of how prices form.* The "
            "mid is not set to fair value; it is pulled toward it by order-flow pressure "
            "(`p_bid = 0.5 + 0.35·tanh((fv − mid)/3)` in `_do_add`). That partial "
            "adjustment spreads one fair-value move across more than one bar, so consecutive "
            "1-minute returns share sign — momentum a real market would arbitrage away.\n"
        )
    L.append(
        "**What this means for the benchmark:** nothing for timings or correctness — the "
        "queries run on realistic *shapes* (consistent streams, real cross-symbol "
        "structure, same-nanosecond sweeps, bid-ask bounce). It does mean the Q4/Q5 "
        "*values* (event-study drift, correlations) are properties of the simulator, "
        "not evidence about markets.\n"
    )
    (res_dir / "stylized_facts.md").write_text("\n".join(L))
    print("\n".join(L))
    con.close()
    shutil.rmtree(data_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
