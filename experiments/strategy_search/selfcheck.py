"""Synthetic-data checks of the simulator: timing (no look-ahead), costs, funding, delisting.

Run: python experiments/strategy_search/selfcheck.py   (exits non-zero on failure)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine as eng  # noqa: E402
import strategies as strat  # noqa: E402
from data import Market  # noqa: E402


def _frames(close: pd.DataFrame, funding: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    fund = pd.DataFrame(funding, index=close.index, columns=close.columns)
    rank = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    return fund, rank


def check_timing_and_costs() -> None:
    idx = pd.date_range("2021-01-01", periods=10, freq="D", tz="UTC")
    close = pd.DataFrame({"A": 100 * 1.01 ** np.arange(10)}, index=idx)
    fund, rank = _frames(close)
    tg = pd.DataFrame(np.nan, index=idx, columns=["A"])
    tg.iloc[2] = 1.0  # signal on day 2 -> filled at close of day 3 -> earns from day 4
    res = eng.simulate(close, fund, rank, tg)
    assert np.allclose(res.returns.iloc[:3], 0.0)
    assert np.isclose(res.returns.iloc[3], -(eng.PERP_FEE + 0.0005)), res.returns.iloc[3]  # entry cost only
    assert np.allclose(res.returns.iloc[4:], 0.01), res.returns.iloc[4:].to_list()
    same = eng.simulate(close, fund, rank, tg, lag=0)
    assert np.isclose(same.returns.iloc[3], 0.01)


def check_funding_and_short() -> None:
    idx = pd.date_range("2021-01-01", periods=6, freq="D", tz="UTC")
    close = pd.DataFrame({"A": [100.0] * 6}, index=idx)
    fund, rank = _frames(close, funding=0.001)
    tg = pd.DataFrame(np.nan, index=idx, columns=["A"])
    tg.iloc[0] = -1.0
    res = eng.simulate(close, fund, rank, tg)
    # short receives +0.1%/day while funding is positive
    assert np.allclose(res.returns.iloc[2:], 0.001, atol=1e-5), res.returns.to_list()  # weight drifts as equity grows
    tg.iloc[0] = 1.0
    res = eng.simulate(close, fund, rank, tg)
    assert np.allclose(res.returns.iloc[2:], -0.001, atol=1e-5)


def check_delisting() -> None:
    idx = pd.date_range("2021-01-01", periods=6, freq="D", tz="UTC")
    close = pd.DataFrame({"A": [100, 100, 100, 50, np.nan, np.nan]}, index=idx, dtype=float)
    fund, rank = _frames(close)
    tg = pd.DataFrame(np.nan, index=idx, columns=["A"])
    tg.iloc[0] = 1.0
    res = eng.simulate(close, fund, rank, tg)
    assert np.isclose(res.returns.iloc[3], -0.5)
    assert res.gross.iloc[4] == 0.0 and res.returns.iloc[5] == 0.0


def check_no_lookahead_on_noise() -> None:
    """On i.i.d. returns every family must earn ~0 before costs; a future-peeking signal would not."""
    rng = np.random.default_rng(7)
    n_days, n_coins = 1500, 40
    idx = pd.date_range("2020-01-01", periods=n_days, freq="D", tz="UTC")
    cols = ["BTCUSDT"] + [f"C{i}USDT" for i in range(n_coins - 1)]
    rets = rng.normal(0.0, 0.04, size=(n_days, n_coins))
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=cols)
    for j in range(10, n_coins):  # staggered listings, so the new-listing family has something to trade
        close.iloc[: int(rng.integers(100, n_days - 200)), j] = np.nan
    qvol = pd.DataFrame(rng.uniform(1e6, 1e8, size=(n_days, n_coins)), index=idx, columns=cols)
    fund = pd.DataFrame(rng.normal(0.0, 0.0003, size=(n_days, n_coins)), index=idx, columns=cols)
    m = Market(close, close, close, close, qvol, fund, close)
    rank, univ = strat.prepare(m)
    zero_fund = fund * 0.0
    worst = 0.0
    for fam in strat.FAMILIES.values():
        for name, spec in fam.items():
            if spec[0] == "cash_and_carry":
                continue
            targets = strat.build_targets(m, univ, spec)
            res = eng.simulate(close, zero_fund, rank, targets)
            g = res.gross_returns.iloc[200:]
            t_stat = g.mean() / g.std() * np.sqrt(len(g))
            worst = max(worst, abs(t_stat))
            assert abs(t_stat) < 3.5, f"{name}: gross t-stat {t_stat:.2f} on pure noise (look-ahead?)"
            # the same targets shifted to peek one day ahead must look brilliant, proving the test has power
    peek = strat.xs_reversal(m, univ, 1).shift(-2)
    res = eng.simulate(close, zero_fund, rank, -peek)
    g = res.gross_returns.iloc[200:]
    assert g.mean() / g.std() * np.sqrt(len(g)) > 10, "peeking control failed to show an edge"
    print(f"noise test: worst |t| across families = {worst:.2f}")


if __name__ == "__main__":
    check_timing_and_costs()
    check_funding_and_short()
    check_delisting()
    check_no_lookahead_on_noise()
    print("selfcheck OK")
