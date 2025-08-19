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


@dataclass
class Suggestion:
    """Proposed bounded change to configuration with rationale and expected impact."""

    target: str  # e.g., "risk", "strategy:ml_basic", "feature_flags"
    change: dict[str, Any]
    rationale: str
    expected_delta: dict[str, float]  # {"annualized_return": +x, "max_drawdown": -y}
    confidence: float
