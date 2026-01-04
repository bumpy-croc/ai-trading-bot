"""Shared performance and risk metrics used across back-tester, live engine and dashboard.

All functions are *pure* (no side-effects), strictly typed and unit-testable.
Numbers are expressed as decimal fractions (e.g. +0.02 = +2 %).
Any percentage outputs are marked in the docstrings.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
import pandas as pd

Number = int | float

# Module-level constants for financial calculations
DAYS_PER_YEAR = 365.0  # Standard year for financial annualization
MAX_FINITE_RATIO = 999.0  # Cap for ratios to avoid infinity in database/JSON storage


class Side(str, Enum):
    """Directional enum shared by trading components."""

    LONG = "long"
    SHORT = "short"


# ────────────────────── Trade-level helpers ──────────────────────


def pnl_percent(
    entry_price: Number, exit_price: Number, side: Side, fraction: float = 1.0
) -> float:
    """Sized percentage return (decimal).

    Parameters
    ----------
    entry_price : Number
        Entry price (must be positive and finite).
    exit_price : Number
        Exit price (must be positive and finite).
    side : Side
        Trade direction (LONG or SHORT).
    fraction : float
        Position size as fraction of balance (must be in [0, 1]).

    Returns
    -------
    float
        Sized percentage return as decimal.

    Raises
    ------
    ValueError
        If inputs are invalid (non-positive prices, non-finite values,
        or fraction outside [0, 1]).

    Example
    -------
    >>> pnl_percent(100, 105, Side.LONG, 0.5)
    0.025   # +2.5 % on *total* balance (5 % move × 50 % position size)
    """
    if entry_price <= 0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if exit_price <= 0:
        raise ValueError(f"exit_price must be positive, got {exit_price}")
    if not math.isfinite(entry_price) or not math.isfinite(exit_price):
        raise ValueError(f"Prices must be finite: entry={entry_price}, exit={exit_price}")
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    raw = (
        (exit_price - entry_price) / entry_price
        if side == Side.LONG
        else (entry_price - exit_price) / entry_price
    )
    return raw * fraction


def cash_pnl(pnl_pct: float, balance_before: Number) -> float:
    """Convert a sized percentage PnL (decimal) into currency units.

    Parameters
    ----------
    pnl_pct : float
        PnL as decimal (must be finite).
    balance_before : Number
        Balance before trade (must be non-negative and finite).

    Returns
    -------
    float
        PnL in currency units.

    Raises
    ------
    ValueError
        If pnl_pct or balance_before is not finite, or balance_before is negative.
    """
    if not math.isfinite(pnl_pct):
        raise ValueError(f"pnl_pct must be finite, got {pnl_pct}")
    if balance_before < 0:
        raise ValueError(f"balance_before must be non-negative, got {balance_before}")
    if not math.isfinite(balance_before):
        raise ValueError(f"balance_before must be finite, got {balance_before}")

    return float(pnl_pct) * float(balance_before)


# ───────────────────── Equity-curve helpers ─────────────────────


def total_return(initial_balance: Number, final_balance: Number) -> float:
    """Total *percentage* return over the whole period.

    Parameters
    ----------
    initial_balance : Number
        Initial balance (must be positive and finite).
    final_balance : Number
        Final balance (must be non-negative and finite).

    Returns
    -------
    float
        Total return as percentage.

    Raises
    ------
    ValueError
        If balances are not finite, initial_balance is not positive,
        or final_balance is negative.
    """
    if initial_balance <= 0:
        raise ValueError(f"initial_balance must be positive, got {initial_balance}")
    if final_balance < 0:
        raise ValueError(f"final_balance must be non-negative, got {final_balance}")
    if not math.isfinite(initial_balance) or not math.isfinite(final_balance):
        raise ValueError(
            f"Balances must be finite: initial={initial_balance}, final={final_balance}"
        )

    return (float(final_balance) / float(initial_balance) - 1.0) * 100.0


def cagr(initial_balance: Number, final_balance: Number, days: int) -> float:
    """Compound annual growth rate (percentage).

    Parameters
    ----------
    initial_balance : Number
        Initial balance (must be positive and finite).
    final_balance : Number
        Final balance (must be non-negative and finite).
    days : int
        Number of days (must be positive).

    Returns
    -------
    float
        Annualized return as percentage.

    Raises
    ------
    ValueError
        If balances are not finite, initial_balance is not positive,
        or final_balance is negative.
    """
    if initial_balance <= 0:
        raise ValueError(f"initial_balance must be positive, got {initial_balance}")
    if final_balance < 0:
        raise ValueError(f"final_balance must be non-negative, got {final_balance}")
    if not math.isfinite(initial_balance) or not math.isfinite(final_balance):
        raise ValueError(
            f"Balances must be finite: initial={initial_balance}, final={final_balance}"
        )
    if days < 1:
        # Less than 1 day - return 0 to avoid unrealistic annualized returns
        return 0.0

    return ((float(final_balance) / float(initial_balance)) ** (DAYS_PER_YEAR / days) - 1.0) * 100.0


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
    return (daily_returns.mean() / std) * np.sqrt(DAYS_PER_YEAR)


def max_drawdown(balance_series: pd.Series) -> float:
    """Maximum drawdown of an equity curve (percentage)."""

    if balance_series.empty:
        return 0.0

    running_max = balance_series.cummax()

    # Prevent division by zero when balance reaches zero
    # Only calculate drawdowns where running_max > 0 to avoid inf/NaN
    mask = running_max > 0
    if not mask.any():
        # All balances are zero or negative - return 100% drawdown
        return 100.0

    drawdowns = pd.Series(0.0, index=balance_series.index)
    drawdowns[mask] = (running_max[mask] - balance_series[mask]) / running_max[mask]

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


# ───────────────────── Risk-adjusted metrics ─────────────────────


def sortino_ratio(daily_balance: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio using downside deviation.

    Measures risk-adjusted return using only downside volatility.

    Parameters
    ----------
    daily_balance : pd.Series
        Equity curve resampled at daily frequency (index monotonic).
    risk_free_rate : float
        Annual risk-free rate as decimal (must be finite, e.g., 0.02 for 2%).

    Returns
    -------
    float
        Annualized Sortino ratio.

    Raises
    ------
    ValueError
        If risk_free_rate is not finite.
    """
    if not math.isfinite(risk_free_rate):
        raise ValueError(f"risk_free_rate must be finite, got {risk_free_rate}")

    if daily_balance.empty or len(daily_balance) < 2:
        return 0.0

    daily_returns = daily_balance.pct_change().dropna()
    if daily_returns.empty:
        return 0.0

    mean_return = daily_returns.mean()
    daily_rf = risk_free_rate / DAYS_PER_YEAR

    # Calculate downside returns (only negative returns)
    downside_returns = daily_returns[daily_returns < daily_rf]

    if downside_returns.empty or len(downside_returns) < 2:
        # No downside volatility - cap at large finite value
        # Using MAX_FINITE_RATIO instead of infinity to avoid database/JSON issues
        return MAX_FINITE_RATIO if mean_return > daily_rf else 0.0

    downside_std = downside_returns.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    # Annualize
    return (mean_return - daily_rf) / downside_std * np.sqrt(DAYS_PER_YEAR)


def calmar_ratio(annualized_return: float, max_drawdown_pct: float) -> float:
    """Calmar ratio: annualized return divided by maximum drawdown.

    Measures return per unit of drawdown risk.

    Parameters
    ----------
    annualized_return : float
        Annualized return as percentage (must be finite, e.g., 15.0 for 15%).
    max_drawdown_pct : float
        Maximum drawdown as percentage (must be finite and non-negative, e.g., 20.0 for 20%).

    Returns
    -------
    float
        Calmar ratio. Returns large finite value (MAX_FINITE_RATIO) for zero drawdown
        with positive returns to avoid infinity/database issues.

    Raises
    ------
    ValueError
        If inputs are not finite or max_drawdown_pct is negative.
    """
    if not math.isfinite(annualized_return) or not math.isfinite(max_drawdown_pct):
        raise ValueError(
            f"Parameters must be finite: return={annualized_return}, drawdown={max_drawdown_pct}"
        )
    if max_drawdown_pct < 0:
        raise ValueError(f"max_drawdown_pct must be non-negative, got {max_drawdown_pct}")

    if max_drawdown_pct <= 0:
        # No drawdown case - return large value if positive returns
        # Using MAX_FINITE_RATIO instead of infinity to avoid database/JSON issues
        return MAX_FINITE_RATIO if annualized_return > 0 else 0.0
    return annualized_return / max_drawdown_pct


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Value at Risk (VaR) at given confidence level.

    Calculates the maximum expected loss at the specified confidence level.

    Parameters
    ----------
    returns : pd.Series
        Series of returns (as decimals).
    confidence : float
        Confidence level (must be in (0, 1), e.g., 0.95 for 95% VaR).

    Returns
    -------
    float
        VaR as decimal (negative value indicates loss).

    Raises
    ------
    ValueError
        If confidence is not in (0, 1).
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    if returns.empty:
        return 0.0

    # VaR is the percentile at (1 - confidence)
    # E.g., 95% VaR is the 5th percentile
    percentile = (1.0 - confidence) * 100.0
    var = float(np.percentile(returns, percentile))
    return var


def expectancy(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Expected value per trade.

    Calculates the average expected profit/loss per trade.

    Parameters
    ----------
    win_rate : float
        Win rate as decimal (must be in [0, 1], e.g., 0.6 for 60%).
    avg_win : float
        Average winning trade PnL (must be non-negative and finite).
    avg_loss : float
        Average losing trade PnL (must be negative or zero and finite).

    Returns
    -------
    float
        Expected value per trade.

    Raises
    ------
    ValueError
        If win_rate is not in [0, 1], avg_loss is positive,
        avg_win is negative, or values are not finite.
    """
    if not (0.0 <= win_rate <= 1.0):
        raise ValueError(f"win_rate must be in [0, 1], got {win_rate}")
    if not math.isfinite(avg_win) or not math.isfinite(avg_loss):
        raise ValueError(f"avg_win and avg_loss must be finite: win={avg_win}, loss={avg_loss}")
    if avg_loss > 0:
        raise ValueError(f"avg_loss must be negative or zero, got {avg_loss}")
    if avg_win < 0:
        raise ValueError(f"avg_win must be non-negative, got {avg_win}")

    return (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)
