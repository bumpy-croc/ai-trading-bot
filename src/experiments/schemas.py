from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ParameterSet:
    """A concrete set of tunable parameters.
    Values are kept generic to support strategies, risk, and engine toggles.
    """

    name: str
    values: dict[str, Any]


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment/backtest run."""

    strategy_name: str
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    initial_balance: float
    risk_parameters: dict[str, Any] = field(default_factory=dict)
    feature_flags: dict[str, Any] = field(default_factory=dict)
    parameters: ParameterSet | None = None
    use_cache: bool = True
    provider: str = "binance"
    random_seed: int | None = None
    # Keyword arguments forwarded to the strategy factory at construction time
    # by :meth:`ExperimentRunner._load_strategy`. Use this for knobs the
    # strategy only honors in ``__init__`` (e.g. ``model_type`` on
    # hyper_growth, which swaps the underlying signal generator and cannot be
    # changed by setattr afterwards).
    factory_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results of a backtest/experiment with key KPIs."""

    config: ExperimentConfig
    total_trades: int
    win_rate: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    final_balance: float
    session_id: int | None = None
    artifacts_path: str | None = None
    # Per-trade P&L sequence (as fractional returns in the order trades
    # closed). The reporter uses this to distinguish "different trades, same
    # aggregate" from "literally the same trades" when variants tie the
    # baseline on the headline metrics — a critical tie-breaker for
    # diagnosing dead-code overrides.
    trade_pnl_pcts: list[float] = field(default_factory=list)
