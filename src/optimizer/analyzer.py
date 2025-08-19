from __future__ import annotations

from dataclasses import dataclass

from src.config.constants import (
    DEFAULT_DRAWDOWN_THRESHOLD,
    DEFAULT_MAX_PARAMETER_CHANGE,
    DEFAULT_SHARPE_RATIO_THRESHOLD,
    DEFAULT_WIN_RATE_THRESHOLD,
)
from src.optimizer.schemas import ExperimentResult, Suggestion


@dataclass
class AnalyzerConfig:
    max_parameter_change: float = DEFAULT_MAX_PARAMETER_CHANGE
    win_rate_threshold: float = DEFAULT_WIN_RATE_THRESHOLD
    sharpe_threshold: float = DEFAULT_SHARPE_RATIO_THRESHOLD
    drawdown_threshold: float = DEFAULT_DRAWDOWN_THRESHOLD


class PerformanceAnalyzer:
    """Analyze experiment results and generate bounded improvement suggestions."""

    def __init__(self, config: AnalyzerConfig | None = None):
        self.cfg = config or AnalyzerConfig()

    def analyze(self, results: list[ExperimentResult]) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        if not results:
            return suggestions

        # Use last result as current baseline
        baseline = results[-1]

        # 1) If drawdown too high, suggest reducing max_position_size and risk per trade
        if baseline.max_drawdown > self.cfg.drawdown_threshold * 100.0:
            change = {
                "risk.max_position_size": -(self.cfg.max_parameter_change),
                "risk.base_risk_per_trade": -(self.cfg.max_parameter_change / 2.0),
            }
            suggestions.append(
                Suggestion(
                    target="risk",
                    change=change,
                    rationale=f"Max drawdown {baseline.max_drawdown:.2f}% exceeds threshold. Reduce exposure.",
                    expected_delta={
                        "annualized_return": -1.0,
                        "max_drawdown": -5.0,
                    },
                    confidence=0.6,
                )
            )

        # 2) If Sharpe low but drawdown acceptable, increase position size bounds slightly
        if (
            baseline.sharpe_ratio < self.cfg.sharpe_threshold
            and baseline.max_drawdown <= self.cfg.drawdown_threshold * 100.0
        ):
            change = {"risk.max_position_size": +(self.cfg.max_parameter_change / 4.0)}
            suggestions.append(
                Suggestion(
                    target="risk",
                    change=change,
                    rationale=f"Sharpe {baseline.sharpe_ratio:.2f} below target with acceptable drawdown; try modest size increase.",
                    expected_delta={
                        "annualized_return": +2.0,
                        "max_drawdown": +1.0,
                    },
                    confidence=0.5,
                )
            )

        # 3) If win rate low, tighten stop loss / raise take profit or thresholds in strategies
        if baseline.win_rate < self.cfg.win_rate_threshold * 100.0:
            suggestions.append(
                Suggestion(
                    target="strategy:ml_basic",
                    change={
                        "MlBasic.stop_loss_pct": -0.05,  # tighten 5% of current value (relative intent)
                        "MlBasic.take_profit_pct": +0.05,
                    },
                    rationale=f"Win rate {baseline.win_rate:.2f}% below threshold; consider tighter stops and higher TP.",
                    expected_delta={
                        "annualized_return": +1.0,
                        "max_drawdown": -1.0,
                    },
                    confidence=0.5,
                )
            )

        return suggestions
