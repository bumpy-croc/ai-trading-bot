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
from data import Market  # noqa: E402

import strategies as strat  # noqa: E402


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


def _noise_market(seed: int = 7, n_days: int = 1500, n_coins: int = 40) -> Market:
    """i.i.d. daily returns, staggered listings, random volume and funding: nothing here is predictable."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="D", tz="UTC")
    cols = ["BTCUSDT"] + [f"C{i}USDT" for i in range(n_coins - 1)]
    rets = rng.normal(0.0, 0.04, size=(n_days, n_coins))
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=cols)
    for j in range(10, n_coins):  # staggered listings, so the new-listing family has something to trade
        close.iloc[: int(rng.integers(100, n_days - 200)), j] = np.nan
    qvol = pd.DataFrame(rng.uniform(1e6, 1e8, size=(n_days, n_coins)), index=idx, columns=cols)
    fund = pd.DataFrame(rng.normal(0.0, 0.0003, size=(n_days, n_coins)), index=idx, columns=cols)
    return Market(close, close, close, close, qvol, fund, close)


def _t_stat(g: pd.Series) -> float:
    return float(g.mean() / g.std() * np.sqrt(len(g)))


def check_no_lookahead_on_noise() -> None:
    """On i.i.d. returns every family must earn ~0 before costs; a future-peeking signal would not."""
    m = _noise_market()
    close, fund = m.close, m.funding
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


def check_cross_venue() -> None:
    """Same price on both venues, Hyperliquid funding 0.1%/day higher: the rule must short Hyperliquid
    and earn notional x 0.1% a day once in, paying entry costs once."""
    import xvenue as xv

    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    px = pd.DataFrame({"AAA": 100 * np.cumprod(1 + rng.normal(0, 0.03, 60)), "BBB": 50.0}, index=idx)
    fund_hl = pd.DataFrame({"AAA": 0.001, "BBB": 0.0}, index=idx)
    fund_bn = pd.DataFrame(0.0, index=idx, columns=px.columns)
    rank = pd.DataFrame(1.0, index=idx, columns=px.columns)
    elig = pd.DataFrame(True, index=idx, columns=px.columns)
    p = xv.Pair(px, px, px, fund_hl, px, px, px, fund_bn, rank, elig)
    res = xv.simulate(p)
    notional = 0.1
    # signal needs 7 days (index 6), executed at index 7, earns from index 8
    assert np.allclose(res.returns.iloc[:7], 0.0)
    entry_cost = notional * (eng.PERP_FEE + 0.0005 + xv.HL_FEE + xv.HL_SLIP)
    assert np.isclose(res.returns.iloc[7], -entry_cost), res.returns.iloc[7]
    assert np.allclose(res.returns.iloc[8:], notional * 0.001), res.returns.iloc[8:12].to_list()
    assert res.extra["slots"].iloc[8:].eq(1).all()  # BBB has no gap and is never entered
    # flip the sign: funding higher on Binance -> long Hyperliquid, short Binance, same income
    p2 = xv.Pair(px, px, px, fund_bn, px, px, px, fund_hl, rank, elig)
    res2 = xv.simulate(p2)
    assert np.allclose(res2.returns.iloc[8:], notional * 0.001)


def check_events() -> None:
    import events as ev

    # E1: flat price, 8-hourly settlements; the trade must use the PREVIOUS print to decide and book the actual one
    idx = pd.date_range("2024-03-01", periods=80, freq="h", tz="UTC")
    hourly = pd.DataFrame({"open": 100.0, "close": 100.0, "qvol": 1e6}, index=idx)
    times = pd.DatetimeIndex([idx[8], idx[16], idx[24], idx[32], idx[40]])
    funding = pd.Series([0.001, 0.01, 0.01, 0.0001, 0.0001], index=times)
    rank = pd.Series(1.0, index=pd.date_range("2024-02-28", periods=10, freq="D", tz="UTC"))
    tr = ev.trades_e1("X", hourly, funding, rank)
    assert [t["entry"] for t in tr] == [idx[23], idx[31]], [t["entry"] for t in tr]
    cost = 2 * (eng.PERP_FEE + 0.0005)
    assert np.isclose(tr[0]["net"], 0.01 - cost) and tr[0]["side"] == -1.0
    assert np.isclose(tr[1]["net"], 0.0001 - cost)
    # negative previous print -> long, and a sign flip at settlement costs the long
    tr = ev.trades_e1("X", hourly, pd.Series([-0.01, 0.002], index=times[:2]), rank)
    assert tr[0]["side"] == 1.0 and np.isclose(tr[0]["funding"], -0.002)
    assert ev.trades_e1("X", hourly, funding, rank * 500) == []  # illiquid coin: no trade

    # E2 / E3: one -5% hour on 5x volume, then +1% for four hours
    rng = np.random.default_rng(11)
    n = 400
    idx = pd.date_range("2024-03-01", periods=n, freq="h", tz="UTC")
    r = rng.normal(0, 0.002, n)
    r[300] = -0.05
    r[301:305] = 0.01
    close = 100 * np.cumprod(1 + r)
    qvol = np.full(n, 1e6)
    qvol[300] = 5e6
    hourly = pd.DataFrame({"open": np.r_[100.0, close[:-1]], "close": close, "qvol": qvol}, index=idx)
    rank = pd.Series(1.0, index=pd.date_range("2024-02-28", periods=30, freq="D", tz="UTC"))
    none = pd.Series(dtype=float)
    tr = ev.trades_e23("X", hourly, none, rank, direction=-1)
    assert len(tr) == 1 and tr[0]["entry"] == idx[301] and tr[0]["exit"] == idx[305]
    assert np.isclose(tr[0]["price_ret"], 1.01**4 - 1.0)
    assert np.isclose(tr[0]["cost"], 2 * (eng.PERP_FEE + 2 * 0.0005))
    assert ev.trades_e23("X", hourly, none, rank, direction=+1) == []
    # a volume spike is required
    quiet = hourly.assign(qvol=1e6)
    assert ev.trades_e23("X", quiet, none, rank, direction=-1) == []
    # funding inside the hold is charged to the long
    fund = pd.Series([0.003], index=[idx[303]])
    tr = ev.trades_e23("X", hourly, fund, rank, direction=-1)
    assert np.isclose(tr[0]["net"], 1.01**4 - 1.0 - 0.003 - tr[0]["cost"])
    # on pure noise with fat tails removed, no trigger fires before the event (no look-ahead into the event hour's sigma)
    assert all(t["entry"] >= idx[301] for t in ev.trades_e23("X", hourly, none, rank, direction=-1))


def check_adaptive_books() -> None:
    """The slow adaptive books (adaptive.py): selectivity, the regime switch's timing, and the ML ranker's
    training cut. On pure noise the ranker must earn nothing; given features dated a week ahead it must
    earn a great deal, which shows the noise test would catch a leak."""
    import adaptive as ad

    m = _noise_market()
    close = m.close
    rank, univ = strat.prepare(m)
    zero_fund = m.funding * 0.0

    # V1 holds a coin only when the 30-, 60- and 120-day returns agree
    sig = ad.trend_signal(m)
    v1 = ad.selective_targets(m, univ)
    day = v1.dropna(how="all").index[-1]
    held = v1.loc[day] != 0
    assert held.any() and (sig.loc[day][held].abs() > 0.99).all()
    assert (v1.loc[day][univ.loc[day] & (sig.loc[day].abs() < 0.99)] == 0).all()
    assert np.isclose(v1.loc[day].abs().sum(), 1.0)

    # V3: the switch on day t must not move when day t's own value or any later value changes the median
    er = ad.trendiness(m, univ)
    on = ad.regime_on(er)
    assert not on.iloc[: ad.ER_WINDOW + ad.ER_MIN_HISTORY].any()  # off until there is enough history
    assert 0.3 < on.iloc[600:].mean() < 0.7
    k = 900
    future = er.copy()
    future.iloc[k + 1:] = 0.99
    assert ad.regime_on(future).iloc[: k + 1].equals(on.iloc[: k + 1])
    past_median = er.expanding(min_periods=ad.ER_MIN_HISTORY).median()
    assert bool(on.iloc[k]) == bool(er.iloc[k] > past_median.iloc[k - 1])
    # the switched book carries the un-switched book's leverage and is flat while the switch is off
    base = ad.run_book(m, rank, strat.trend(m, univ, ad.LOOKBACKS))
    gated = ad.run_book(m, rank, strat.trend(m, univ, ad.LOOKBACKS), gate=on)
    signal_days = base.weights.dropna(how="all").index
    off_days = signal_days[~on.loc[signal_days].to_numpy()]
    on_days = signal_days[on.loc[signal_days].to_numpy()]
    assert (gated.weights.loc[off_days].abs().sum(axis=1) == 0).all()
    assert np.allclose(gated.weights.loc[on_days].fillna(0.0), base.weights.loc[on_days].fillna(0.0))

    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("adaptive books: V1 and V3 OK; ML ranker check SKIPPED (scikit-learn is not installed)")
        return
    scores, info = ad.ml_scores(m, rank, univ)
    assert len(info["refits"]) >= 8 and info["first_prediction"] is not None
    start = pd.Timestamp(info["first_prediction"], tz="UTC")
    res = eng.simulate(close, zero_fund, rank, ad.ml_targets(m, univ, scores))
    t_noise = _t_stat(res.gross_returns.loc[start:])
    assert abs(t_noise) < 3.5, f"ML ranker: gross t-stat {t_noise:.2f} on pure noise (leak?)"
    peek, _ = ad.ml_scores(m, rank, univ, peek_days=ad.ML_LABEL_LAG)
    res = eng.simulate(close, zero_fund, rank, ad.ml_targets(m, univ, peek))
    t_peek = _t_stat(res.gross_returns.loc[start:])
    assert t_peek > 10, f"ML ranker leak control failed to show an edge (t = {t_peek:.2f})"
    again, _ = ad.ml_scores(m, rank, univ)
    assert again.equals(scores), "ML ranker is not deterministic"
    print(f"adaptive books OK: ML ranker t = {t_noise:.2f} on noise, {t_peek:.1f} with features dated a week ahead")


if __name__ == "__main__":
    check_adaptive_books()
    check_events()
    check_cross_venue()
    check_timing_and_costs()
    check_funding_and_short()
    check_delisting()
    check_no_lookahead_on_noise()
    print("selfcheck OK")
