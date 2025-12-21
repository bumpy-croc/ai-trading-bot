"""Shared performance and risk metrics used across back-tester, live engine and dashboard.

All functions are *pure* (no side-effects), strictly typed and unit-testable.
Numbers are expressed as decimal fractions (e.g. +0.02 = +2 %).
Any percentage outputs are marked in the docstrings.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

Number = int | float


class Side(str, Enum):
    """Directional enum shared by trading components."""

    LONG = "long"
    SHORT = "short"


# ────────────────────── Trade-level helpers ──────────────────────


def pnl_percent(
    entry_price: Number, exit_price: Number, side: Side, fraction: float = 1.0
) -> float:
    """Sized percentage return (decimal).

    Example
    -------
    >>> pnl_percent(100, 105, Side.LONG, 0.5)
    0.025   # +2.5 % on *total* balance (5 % move × 50 % position size)
    """
    if entry_price == 0:
        return 0.0

    raw = (
        (exit_price - entry_price) / entry_price
        if side == Side.LONG
        else (entry_price - exit_price) / entry_price
    )
    return raw * fraction


def cash_pnl(pnl_pct: float, balance_before: Number) -> float:
    """Convert a sized percentage PnL (decimal) into currency units."""

    return float(pnl_pct) * float(balance_before)


# ───────────────────── Equity-curve helpers ─────────────────────


def total_return(initial_balance: Number, final_balance: Number) -> float:
    """Total *percentage* return over the whole period."""

    if initial_balance == 0:
        return 0.0
    return (float(final_balance) / float(initial_balance) - 1.0) * 100.0


def cagr(initial_balance: Number, final_balance: Number, days: int) -> float:
    """Compound annual growth rate (percentage)."""

    if initial_balance == 0 or days <= 0:
        return 0.0
    return ((float(final_balance) / float(initial_balance)) ** (365.0 / days) - 1.0) * 100.0


def sharpe(daily_balance: pd.Series) -> float:
    """Annualised Sharpe ratio (risk-free rate = 0).

    Parameters
    ----------
    daily_balance : pd.Series
        Equity curve resampled at *daily* frequency (index monotonic).
    """

    if daily_balance.empty or len(daily_balance) < 2:
        return 0.0

    daily_returns = daily_balance.pct_change().dropna()
    std = daily_returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return (daily_returns.mean() / std) * np.sqrt(365.0)


def max_drawdown(balance_series: pd.Series) -> float:
    """Maximum drawdown of an equity curve (percentage)."""

    if balance_series.empty:
        return 0.0

    running_max = balance_series.cummax()
    drawdowns = (running_max - balance_series) / running_max
    return drawdowns.max() * 100.0


def directional_accuracy(pred_prices: pd.Series, actual_next_close: pd.Series) -> float:
    """Directional accuracy (% correct up/down vs next close)."""
    if pred_prices is None or actual_next_close is None:
        return 0.0
    if len(pred_prices) == 0 or len(actual_next_close) == 0:
        return 0.0
    common = pred_prices.index.intersection(actual_next_close.index)
    if len(common) == 0:
        return 0.0
    p = pred_prices.loc[common]
    a = actual_next_close.loc[common]
    # Compare sign(pred - current) vs sign(next - current). We need current close at t.
    # Assume a is aligned to t+1 close; for approximation, compare pred vs a directly.
    pred_dir = np.sign(p.diff())  # fallback if current not available
    true_dir = np.sign(a.diff())
    mask = (~np.isnan(pred_dir)) & (~np.isnan(true_dir))
    if mask.sum() == 0:
        return 0.0
    acc = (pred_dir[mask] == true_dir[mask]).mean()
    return float(acc * 100.0)


def mean_absolute_error(pred: pd.Series, actual: pd.Series) -> float:
    """MAE between prediction and actual."""
    common = pred.index.intersection(actual.index)
    if len(common) == 0:
        return 0.0
    return float((pred.loc[common] - actual.loc[common]).abs().mean())


def mean_absolute_percentage_error(pred: pd.Series, actual: pd.Series) -> float:
    """MAPE (%) between prediction and actual."""
    common = pred.index.intersection(actual.index)
    if len(common) == 0:
        return 0.0
    a = actual.loc[common]
    p = pred.loc[common]
    den = a.replace(0, np.nan).abs()
    mape = ((p - a).abs() / den).dropna().mean()
    return float((mape if not np.isnan(mape) else 0.0) * 100.0)


def brier_score_direction(prob_up: pd.Series, actual_up: pd.Series) -> float:
    """Brier score for binary direction (lower is better).

    prob_up: probability of up direction in [0,1]
    actual_up: 1 if next close up, else 0
    """
    common = prob_up.index.intersection(actual_up.index)
    if len(common) == 0:
        return 0.0
    p = prob_up.loc[common].clip(0.0, 1.0)
    y = actual_up.loc[common].clip(0.0, 1.0)
    return float(((p - y) ** 2).mean())
