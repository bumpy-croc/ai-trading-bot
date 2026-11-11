"""Portfolio simulator and metrics for the wide strategy search.

Perpetual-futures portfolio, daily bars, signed weights (notional / equity).
A target computed on day t's close is filled at day t+LAG's close (LAG = 1 by default,
the same conservative rule as experiments/strategy_bakeoff.py).

Costs, per side, on traded notional:
  fee 5 bps (taker) + slippage by the coin's liquidity rank on the signal day
  (rank 1-10: 5 bps, 11-30: 10 bps, 31+: 20 bps).
Funding is charged on the position held through each day at the real historical rate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ANN = 365
RF_ANNUAL = 0.04  # what idle USDT earns in a flexible savings product; the hurdle
RF_DAILY = (1.0 + RF_ANNUAL) ** (1.0 / ANN) - 1.0
PERP_FEE = 0.0005
SPOT_FEE = 0.0010
SLIP_TIERS = ((10, 0.0005), (30, 0.0010), (10**9, 0.0020))
MISSING_FUNDING_LONG = 0.0003  # per day, charged to longs when a funding print is missing; shorts get 0
VOL_TARGET = 0.15
VOL_LOOKBACK = 60
MAX_GROSS = 2.0
MIN_HISTORY = 60
UNIVERSE_N = 30
LIQ_LOOKBACK = 30

# Not crypto-asset perps: stablecoin pairs, index products, tokenised metals/equities.
EXCLUDE = {
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "USDPUSDT", "DAIUSDT", "USDEUSDT", "USD1USDT",
    "BTCDOMUSDT", "DEFIUSDT", "FOOTBALLUSDT", "BLUEBIRDUSDT", "ALLUSDT",
    "PAXGUSDT", "XAUUSDT", "XAGUSDT", "XAUTUSDT",
    "BTCSTUSDT",
}
# Plus everything Binance's own exchangeInfo classifies as TradFi / index / pre-market
# (tokenised equities, commodities, FX). Written by data.fetch_non_crypto(), committed.
_NON_CRYPTO = Path(__file__).resolve().parent / "non_crypto_symbols.json"
if _NON_CRYPTO.exists():
    EXCLUDE |= set(json.loads(_NON_CRYPTO.read_text()))


def slippage_for_rank(rank: np.ndarray) -> np.ndarray:
    out = np.full(rank.shape, SLIP_TIERS[-1][1])
    for limit, slip in reversed(SLIP_TIERS):
        out = np.where(rank <= limit, slip, out)
    return out


def liquidity(qvol: pd.DataFrame) -> pd.DataFrame:
    """Trailing 30-day median quote volume, known at each day's close."""
    return qvol.rolling(LIQ_LOOKBACK, min_periods=LIQ_LOOKBACK).median()


def liquidity_rank(close: pd.DataFrame, qvol: pd.DataFrame) -> pd.DataFrame:
    """Rank 1 = most liquid among coins that are tradable and seasoned on that day."""
    ok = (close.notna().cumsum() >= MIN_HISTORY) & close.notna()
    ok = ok.loc[:, [c for c in close.columns if c not in EXCLUDE]].reindex(columns=close.columns, fill_value=False)
    liq = liquidity(qvol).where(ok)
    return liq.rank(axis=1, ascending=False, method="first")


def universe(rank: pd.DataFrame, n: int = UNIVERSE_N) -> pd.DataFrame:
    return rank <= n


@dataclass
class Result:
    returns: pd.Series  # daily net return on equity
    gross_returns: pd.Series  # before trading costs and funding
    costs: pd.Series  # daily trading cost as a fraction of equity
    funding: pd.Series  # daily funding paid (+) or received (-), fraction of equity
    turnover: pd.Series  # traded notional / equity
    gross: pd.Series  # sum |w| after the close
    net: pd.Series  # sum w after the close
    extra: dict = field(default_factory=dict)


def simulate(
    close: pd.DataFrame,
    funding: pd.DataFrame,
    rank: pd.DataFrame,
    targets: pd.DataFrame,
    lag: int = 1,
) -> Result:
    """`targets`: signed target weights on signal days, all-NaN rows = no rebalance."""
    idx = close.index
    rets = close.pct_change(fill_method=None).to_numpy()
    alive = close.notna().to_numpy()
    fund = funding.to_numpy()
    tg = targets.shift(lag).to_numpy()
    slip = slippage_for_rank(rank.shift(lag).fillna(10**6).to_numpy())
    cost_rate = PERP_FEE + slip
    n_days, n_assets = rets.shape
    w = np.zeros(n_assets)
    o_ret, o_gross_ret, o_cost, o_fund, o_turn, o_gross, o_net = (np.zeros(n_days) for _ in range(7))
    for d in range(n_days):
        held = w != 0
        cost = 0.0
        fund_paid = 0.0
        port = 0.0
        if held.any():
            r = np.where(np.isnan(rets[d]), 0.0, rets[d])
            f = fund[d]
            f_eff = np.where(np.isnan(f), np.where(w > 0, MISSING_FUNDING_LONG, 0.0), f)
            port = float((w * r).sum())
            fund_paid = float((w * f_eff)[held].sum())
            w = w * (1.0 + r) / (1.0 + port - fund_paid)
            # a contract that stopped trading is settled at its last price
            dead = held & ~alive[d]
            if dead.any():
                cost += float((np.abs(w[dead]) * cost_rate[d][dead]).sum())
                w = np.where(dead, 0.0, w)
        turn = 0.0
        row = tg[d]
        if not np.isnan(row).all():
            tgt = np.where(np.isnan(row) | ~alive[d], 0.0, row)
            delta = np.abs(tgt - w)
            turn = float(delta.sum())
            cost += float((delta * cost_rate[d]).sum())
            w = tgt
        o_gross_ret[d] = port
        o_fund[d] = fund_paid
        o_cost[d] = cost
        o_ret[d] = (1.0 + port - fund_paid) * (1.0 - cost) - 1.0
        o_turn[d] = turn
        o_gross[d] = np.abs(w).sum()
        o_net[d] = w.sum()
    s = lambda a: pd.Series(a, index=idx)  # noqa: E731
    return Result(s(o_ret), s(o_gross_ret), s(o_cost), s(o_fund), s(o_turn), s(o_gross), s(o_net))


def vol_targeted(
    close: pd.DataFrame,
    funding: pd.DataFrame,
    rank: pd.DataFrame,
    targets: pd.DataFrame,
    lag: int = 1,
) -> Result:
    """Scale the strategy to VOL_TARGET annualised using its own trailing 60-day realised vol.

    The scale is recomputed on each signal day from unlevered returns known at that close,
    and gross exposure is capped at MAX_GROSS.
    """
    base = simulate(close, funding, rank, targets, lag)
    vol = base.returns.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std() * np.sqrt(ANN)
    lev = (VOL_TARGET / vol.replace(0.0, np.nan)).fillna(0.0)
    gross = targets.abs().sum(axis=1).replace(0.0, np.nan)
    lev = np.minimum(lev, MAX_GROSS / gross).fillna(0.0)
    scaled = targets.mul(lev, axis=0)
    out = simulate(close, funding, rank, scaled, lag)
    out.extra["unlevered"] = base
    out.extra["leverage"] = lev.where(targets.notna().any(axis=1))
    return out


# ------------------------------------------------------------------------ metrics


def window(s: pd.Series, start: str, end: str) -> pd.Series:
    return s.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]


def metrics(res: Result, start: str, end: str) -> dict:
    r = window(res.returns, start, end)
    if len(r) < 30:
        return {}
    eq = (1.0 + r).cumprod()
    years = len(r) / ANN
    sd = r.std()
    ex = r - RF_DAILY
    return {
        "cagr": float(eq.iloc[-1] ** (1.0 / years) - 1.0) if eq.iloc[-1] > 0 else -1.0,
        "vol": float(sd * np.sqrt(ANN)),
        "sharpe_rf0": float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else float("nan"),
        "sharpe": float(ex.mean() / sd * np.sqrt(ANN)) if sd > 0 else float("nan"),  # excess of idle USDT
        "max_dd": float(-(eq / eq.cummax().clip(lower=1.0) - 1.0).min()),
        "worst_day": float(r.min()),
        "cost_per_year": float(window(res.costs, start, end).sum() / years),
        "funding_per_year": float(window(res.funding, start, end).sum() / years),
        "turnover_per_year": float(window(res.turnover, start, end).sum() / years),
        "avg_gross": float(window(res.gross, start, end).mean()),
        "avg_net": float(window(res.net, start, end).mean()),
        "days": int(len(r)),
    }


def yearly(res: Result) -> dict[int, float]:
    return {int(y): float((1.0 + r).prod() - 1.0) for y, r in res.returns.groupby(res.returns.index.year)}


def weekly_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index[index.dayofweek == 0]  # Monday close (UTC)


def on_days(weights: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Keep target rows only on rebalance days."""
    out = pd.DataFrame(np.nan, index=weights.index, columns=weights.columns)
    out.loc[days] = weights.loc[days].fillna(0.0)
    return out


def window_frame(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]
