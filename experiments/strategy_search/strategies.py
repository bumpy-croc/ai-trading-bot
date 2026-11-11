"""The eight preregistered families (docs/research/experiments/2026-11-10_strategy-search-prereg.md).

Every function returns target weights (date x symbol) with rows only on signal days, at
gross exposure 1 (or less when legs are empty). CC is the exception: it has a spot leg, so
it is simulated here and returns an engine.Result directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from data import Market
from engine import (
    ANN,
    EXCLUDE,
    PERP_FEE,
    SPOT_FEE,
    Result,
    liquidity_rank,
    on_days,
    slippage_for_rank,
    universe,
    weekly_days,
)

BTC = "BTCUSDT"


def _rets(m: Market) -> pd.DataFrame:
    return m.close.pct_change(fill_method=None)


def _vol30(m: Market) -> pd.DataFrame:
    return _rets(m).rolling(30, min_periods=30).std()


def _quintile_legs(score: pd.DataFrame, univ: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Boolean frames (top-score leg, bottom-score leg) within the universe, per day."""
    s = score.where(univ)
    n = s.notna().sum(axis=1)
    k = np.maximum(3, np.round(n / 5.0)).astype(int)
    enough = n >= 6
    hi = s.rank(axis=1, ascending=False, method="first").le(k, axis=0) & enough.to_numpy()[:, None]
    lo = s.rank(axis=1, ascending=True, method="first").le(k, axis=0) & enough.to_numpy()[:, None]
    return hi, lo


def _long_short(long_leg: pd.DataFrame, short_leg: pd.DataFrame) -> pd.DataFrame:
    """Equal weight inside each leg, 0.5 gross per leg."""
    lw = long_leg.div(long_leg.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 0.5
    sw = short_leg.div(short_leg.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 0.5
    return lw - sw


def _normalise(w: pd.DataFrame) -> pd.DataFrame:
    return w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


# ------------------------------------------------------------------------ families


def trend(m: Market, univ: pd.DataFrame, lookbacks: tuple[int, ...]) -> pd.DataFrame:
    sig = sum(np.sign(m.close / m.close.shift(L) - 1.0) for L in lookbacks) / len(lookbacks)
    raw = (sig / _vol30(m)).where(univ)
    return on_days(_normalise(raw.fillna(0.0)), weekly_days(m.close.index))


def xs_momentum(m: Market, univ: pd.DataFrame, lookback: int) -> pd.DataFrame:
    score = m.close.shift(1) / m.close.shift(lookback) - 1.0
    hi, lo = _quintile_legs(score, univ)
    return on_days(_long_short(hi, lo), weekly_days(m.close.index))


def xs_reversal(m: Market, univ: pd.DataFrame, lookback: int) -> pd.DataFrame:
    score = m.close / m.close.shift(lookback) - 1.0
    hi, lo = _quintile_legs(score, univ)
    days = weekly_days(m.close.index) if lookback >= 7 else m.close.index
    return on_days(_long_short(lo, hi), days)


def xs_carry(m: Market, univ: pd.DataFrame, lookback: int) -> pd.DataFrame:
    score = m.funding.rolling(lookback, min_periods=lookback).mean()
    hi, lo = _quintile_legs(score, univ)
    return on_days(_long_short(lo, hi), weekly_days(m.close.index))


def btc_vs_alts(m: Market, univ: pd.DataFrame, beta_neutral: bool) -> pd.DataFrame:
    alts = univ.copy()
    alts[BTC] = False
    basket_w = alts.div(alts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if beta_neutral:
        r = _rets(m)
        # basket return uses the previous day's membership, so it is known at each close
        basket_r = (basket_w.shift(1) * r).sum(axis=1)
        beta = basket_r.rolling(90, min_periods=90).cov(r[BTC]) / r[BTC].rolling(90, min_periods=90).var()
        short_scale = (1.0 / beta.clip(lower=0.25, upper=4.0)).fillna(1.0)
    else:
        short_scale = pd.Series(1.0, index=m.close.index)
    w = -basket_w.mul(short_scale, axis=0)
    w[BTC] = np.where(alts.any(axis=1) & m.close[BTC].notna(), 1.0, 0.0)
    w = w.where(alts.any(axis=1), 0.0)
    return on_days(_normalise(w), weekly_days(m.close.index))


def low_vol(m: Market, univ: pd.DataFrame, variant: str) -> pd.DataFrame:
    vol = _vol30(m)
    if variant == "max":
        score = _rets(m).rolling(30, min_periods=30).max()
    else:
        score = vol
    hi, lo = _quintile_legs(score, univ)
    if variant == "vol_equal_risk":
        # equal weight inside a leg; each leg sized inversely to its mean coin vol (equal ex-ante vol)
        l_vol = vol.where(lo).mean(axis=1)
        s_vol = vol.where(hi).mean(axis=1)
        lw = lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).div(l_vol, axis=0)
        sw = hi.div(hi.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).div(s_vol, axis=0)
        w = _normalise((lw - sw).fillna(0.0))
    else:
        w = _long_short(lo, hi)
    return on_days(w, weekly_days(m.close.index))


def new_listings(m: Market, d1: int, d2: int) -> pd.DataFrame:
    age = m.close.notna().cumsum()  # trading days since listing (1 on the first day)
    tradable = [c for c in m.close.columns if c not in EXCLUDE and c != BTC]
    short = ((age > d1) & (age <= d2) & m.close.notna()).reindex(columns=m.close.columns, fill_value=False)
    short.loc[:, [c for c in m.close.columns if c not in tradable]] = False
    sw = short.div(short.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 0.5
    w = -sw
    w[BTC] = np.where(short.any(axis=1), 0.5, 0.0)
    return on_days(w, weekly_days(m.close.index))


# ---------------------------------------------------------------- cash and carry


def cash_and_carry(m: Market, univ: pd.DataFrame, rank: pd.DataFrame, lookback: int, lag: int = 1) -> Result:
    """Spot long + perp short on the highest-funding universe coins. K = 5 slots.

    Notional per slot = 0.5 / K of capital (half the slot buys spot, half is perp margin).
    Membership is reviewed on Monday closes and executed `lag` days later.
    """
    k, enter_rank, exit_rank, enter_thr = 5, 5, 10, 0.15 / ANN
    idx = m.close.index
    cols = m.close.columns
    trail = m.funding.rolling(lookback, min_periods=lookback).mean()
    has_spot = m.spot_close.notna() & m.close.notna()
    cand = trail.where(univ & has_spot)
    frank = cand.rank(axis=1, ascending=False, method="first")
    review = set(weekly_days(idx))

    members: set[str] = set()
    target = pd.DataFrame(False, index=idx, columns=cols)
    for day in idx:
        if day in review:
            tr, fr = trail.loc[day], frank.loc[day]
            members = {c for c in members if fr.get(c, np.nan) <= exit_rank and tr.get(c, np.nan) >= 0}
            for c in fr.dropna().sort_values().index:
                if len(members) >= k:
                    break
                if fr[c] <= enter_rank and tr[c] > enter_thr:
                    members.add(c)
        if members:
            target.loc[day, list(members)] = True
    held = target.shift(lag, fill_value=False) & has_spot  # position after the close of each day
    pos = held.shift(1, fill_value=False)  # position carried through the day
    notional = 0.5 / k

    spot_r = m.spot_close.pct_change(fill_method=None).fillna(0.0)
    perp_r = m.close.pct_change(fill_method=None).fillna(0.0)
    fund = m.funding.fillna(0.0)  # a short receives nothing when the print is missing
    gross_ret = ((spot_r - perp_r).where(pos, 0.0) * notional).sum(axis=1)
    fund_recv = (fund.where(pos, 0.0) * notional).sum(axis=1)
    slip = pd.DataFrame(slippage_for_rank(rank.shift(lag).fillna(10**6).to_numpy()), index=idx, columns=cols)
    change = held.astype(int).diff().abs().fillna(held.astype(int))
    trade_cost = (change * notional * (SPOT_FEE + PERP_FEE + 2 * slip)).sum(axis=1)
    hedge_cost = (spot_r.abs().where(pos, 0.0) * notional * (SPOT_FEE + slip)).sum(axis=1)
    cost = trade_cost + hedge_cost
    net = (1.0 + gross_ret + fund_recv) * (1.0 - cost) - 1.0
    res = Result(
        returns=net,
        gross_returns=gross_ret,
        costs=cost,
        funding=-fund_recv,
        turnover=(change * notional * 2).sum(axis=1),
        gross=held.sum(axis=1) * notional * 2,
        net=pd.Series(0.0, index=idx),
    )
    res.extra["slots_used"] = held.sum(axis=1)
    return res


# ------------------------------------------------------------------------ registry

FAMILIES: dict[str, dict[str, tuple]] = {
    "T": {"T_fast_7_14_28": ("trend", (7, 14, 28)), "T_slow_30_60_120": ("trend", (30, 60, 120))},
    "XM": {f"XM_{L}": ("xs_momentum", L) for L in (7, 14, 28, 56)},
    "XR": {f"XR_{L}": ("xs_reversal", L) for L in (1, 3, 7)},
    "XC": {f"XC_{L}": ("xs_carry", L) for L in (3, 7, 14)},
    "CC": {f"CC_{L}": ("cash_and_carry", L) for L in (7, 14)},
    "AB": {"AB_dollar": ("btc_vs_alts", False), "AB_beta": ("btc_vs_alts", True)},
    "LV": {"LV_vol": ("low_vol", "vol"), "LV_vol_equal_risk": ("low_vol", "vol_equal_risk"), "LV_max": ("low_vol", "max")},
    "NL": {"NL_7_60": ("new_listings", (7, 60)), "NL_7_120": ("new_listings", (7, 120))},
}


def build_targets(m: Market, univ: pd.DataFrame, spec: tuple) -> pd.DataFrame:
    kind, arg = spec
    if kind == "trend":
        return trend(m, univ, arg)
    if kind == "xs_momentum":
        return xs_momentum(m, univ, arg)
    if kind == "xs_reversal":
        return xs_reversal(m, univ, arg)
    if kind == "xs_carry":
        return xs_carry(m, univ, arg)
    if kind == "btc_vs_alts":
        return btc_vs_alts(m, univ, arg)
    if kind == "low_vol":
        return low_vol(m, univ, arg)
    if kind == "new_listings":
        return new_listings(m, *arg)
    raise ValueError(kind)


def prepare(m: Market) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank = liquidity_rank(m.close, m.qvol)
    return rank, universe(rank)
