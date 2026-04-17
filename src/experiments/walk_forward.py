from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from src.experiments.runner import ExperimentRunner
from src.experiments.schemas import ExperimentConfig, ExperimentResult


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis.

    Attributes:
        strategy_name: Strategy to evaluate.
        symbol: Trading pair symbol (e.g. BTCUSDT).
        timeframe: Candle timeframe (e.g. 1h, 4h).
        train_days: In-sample window size in days.
        test_days: Out-of-sample window size in days.
        step_days: Step size between folds in days.  Defaults to ``test_days``.
        num_folds: Number of rolling folds.  When set, ``step_days`` is derived
            automatically so that all folds fit within ``total_days``.
        total_days: Total data span.  Derived from folds when not provided.
        initial_balance: Starting capital for each fold.
        provider: Data provider name (binance, coinbase, mock, fixture).
        use_cache: Whether to use cached market data.
        random_seed: Seed for reproducible mock data.
        robustness_good: Minimum OOS/IS Sharpe ratio considered acceptable.
        robustness_strong: Minimum OOS/IS Sharpe ratio considered robust.
    """

    strategy_name: str = "ml_basic"
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    train_days: int = 180
    test_days: int = 30
    step_days: int | None = None
    num_folds: int | None = None
    total_days: int | None = None
    initial_balance: float = 10_000.0
    provider: str = "binance"
    use_cache: bool = True
    random_seed: int | None = None
    robustness_good: float = 0.5
    robustness_strong: float = 0.7
    risk_parameters: dict | None = None

    def __post_init__(self) -> None:
        if self.train_days <= 0:
            raise ValueError(f"train_days must be positive, got {self.train_days}")
        if self.test_days <= 0:
            raise ValueError(f"test_days must be positive, got {self.test_days}")
        if self.initial_balance <= 0:
            raise ValueError(f"initial_balance must be positive, got {self.initial_balance}")
        if self.num_folds is not None and self.num_folds <= 0:
            raise ValueError(f"num_folds must be positive, got {self.num_folds}")
        if self.step_days is not None and self.step_days <= 0:
            raise ValueError(f"step_days must be positive, got {self.step_days}")


@dataclass
class FoldResult:
    """Metrics for a single in-sample / out-of-sample fold."""

    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    is_sharpe: float
    oos_sharpe: float
    is_return: float
    oos_return: float
    is_max_drawdown: float
    oos_max_drawdown: float
    is_win_rate: float
    oos_win_rate: float
    is_total_trades: int
    oos_total_trades: int
    robustness_ratio: float


@dataclass
class WalkForwardResult:
    """Aggregate results of a walk-forward analysis run."""

    config: WalkForwardConfig
    folds: list[FoldResult] = field(default_factory=list)
    mean_is_sharpe: float = 0.0
    mean_oos_sharpe: float = 0.0
    mean_robustness_ratio: float = 0.0
    median_robustness_ratio: float = 0.0
    overfitting_detected: bool = False
    robustness_label: str = "POOR"

    @property
    def num_folds(self) -> int:
        return len(self.folds)


def compute_windows(
    end: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
    num_folds: int,
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """Compute rolling train/test windows walking backward from *end*.

    Returns a list of ``(train_start, train_end, test_start, test_end)`` tuples
    ordered chronologically (earliest fold first).
    """
    windows: list[tuple[datetime, datetime, datetime, datetime]] = []
    for i in range(num_folds):
        offset = i * step_days
        test_end = end - timedelta(days=offset)
        test_start = test_end - timedelta(days=test_days)
        train_end = test_start
        train_start = train_end - timedelta(days=train_days)
        windows.append((train_start, train_end, test_start, test_end))

    # Return in chronological order (earliest first)
    windows.reverse()
    return windows


def robustness_label(ratio: float, good: float, strong: float) -> str:
    """Classify a robustness ratio into a human-readable label."""
    if ratio >= strong:
        return "ROBUST"
    if ratio >= good:
        return "ACCEPTABLE"
    if ratio > 0:
        return "WEAK"
    return "POOR"


class WalkForwardAnalyzer:
    """Run walk-forward analysis to validate strategy robustness and detect overfitting.

    Walk-forward analysis splits historical data into rolling in-sample (IS) and
    out-of-sample (OOS) windows.  For each fold the strategy is backtested on
    both periods and the OOS/IS Sharpe ratio (robustness ratio) is computed.
    A mean robustness ratio above 0.5 suggests the strategy generalises; above
    0.7 is considered robust.
    """

    def __init__(self, config: WalkForwardConfig, runner: ExperimentRunner | None = None):
        self.config = config
        self.runner = runner or ExperimentRunner()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _resolve_folds(self) -> tuple[int, int]:
        """Return ``(num_folds, step_days)`` from config, applying defaults."""
        cfg = self.config
        step = cfg.step_days or cfg.test_days

        if cfg.num_folds is not None and cfg.total_days is not None:
            raise ValueError(
                "Cannot set both num_folds and total_days — they are mutually exclusive"
            )

        if cfg.num_folds is not None:
            return cfg.num_folds, step

        # Derive num_folds from total_days if provided
        if cfg.total_days is not None:
            available = cfg.total_days - cfg.train_days - cfg.test_days
            if available < 0:
                raise ValueError(
                    f"total_days ({cfg.total_days}) too small for "
                    f"train_days ({cfg.train_days}) + test_days ({cfg.test_days})"
                )
            num_folds = available // step + 1
            return max(num_folds, 1), step

        # Default: 6 folds
        return 6, step

    def _build_experiment(self, start: datetime, end: datetime) -> ExperimentConfig:
        """Create an ExperimentConfig for a single window."""
        return ExperimentConfig(
            strategy_name=self.config.strategy_name,
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            start=start,
            end=end,
            initial_balance=self.config.initial_balance,
            provider=self.config.provider,
            use_cache=self.config.use_cache,
            random_seed=self.config.random_seed,
            risk_parameters=self.config.risk_parameters or {},
        )

    @staticmethod
    def _safe_robustness(oos_sharpe: float, is_sharpe: float) -> float:
        """Compute OOS/IS Sharpe ratio, handling edge cases."""
        if is_sharpe <= 0:
            # Negative or zero IS Sharpe provides no valid baseline for comparison
            return 0.0
        ratio = oos_sharpe / is_sharpe
        return max(-1.0, min(ratio, 2.0))

    def run(self, end: datetime | None = None) -> WalkForwardResult:
        """Execute the walk-forward analysis and return aggregate results."""
        end = end or datetime.now(UTC)
        num_folds, step_days = self._resolve_folds()

        windows = compute_windows(
            end=end,
            train_days=self.config.train_days,
            test_days=self.config.test_days,
            step_days=step_days,
            num_folds=num_folds,
        )

        self.logger.info(
            "Starting walk-forward analysis: %d folds, train=%dd, test=%dd, step=%dd",
            len(windows),
            self.config.train_days,
            self.config.test_days,
            step_days,
        )

        folds: list[FoldResult] = []
        for idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            self.logger.info(
                "Fold %d/%d: IS %s–%s  OOS %s–%s",
                idx + 1,
                len(windows),
                train_start.date(),
                train_end.date(),
                test_start.date(),
                test_end.date(),
            )

            is_cfg = self._build_experiment(train_start, train_end)
            oos_cfg = self._build_experiment(test_start, test_end)

            is_result: ExperimentResult = self.runner.run(is_cfg)
            oos_result: ExperimentResult = self.runner.run(oos_cfg)

            ratio = self._safe_robustness(oos_result.sharpe_ratio, is_result.sharpe_ratio)

            folds.append(
                FoldResult(
                    fold_index=idx,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    is_sharpe=is_result.sharpe_ratio,
                    oos_sharpe=oos_result.sharpe_ratio,
                    is_return=is_result.total_return,
                    oos_return=oos_result.total_return,
                    is_max_drawdown=is_result.max_drawdown,
                    oos_max_drawdown=oos_result.max_drawdown,
                    is_win_rate=is_result.win_rate,
                    oos_win_rate=oos_result.win_rate,
                    is_total_trades=is_result.total_trades,
                    oos_total_trades=oos_result.total_trades,
                    robustness_ratio=ratio,
                )
            )

        return self._aggregate(folds)

    def _aggregate(self, folds: list[FoldResult]) -> WalkForwardResult:
        """Build the final WalkForwardResult from per-fold data."""
        if not folds:
            return WalkForwardResult(config=self.config)

        ratios = [f.robustness_ratio for f in folds]
        is_sharpes = [f.is_sharpe for f in folds]
        oos_sharpes = [f.oos_sharpe for f in folds]

        mean_ratio = sum(ratios) / len(ratios)
        sorted_ratios = sorted(ratios)
        mid = len(sorted_ratios) // 2
        if len(sorted_ratios) % 2 == 0:
            median_ratio = (sorted_ratios[mid - 1] + sorted_ratios[mid]) / 2
        else:
            median_ratio = sorted_ratios[mid]

        mean_is = sum(is_sharpes) / len(is_sharpes)
        mean_oos = sum(oos_sharpes) / len(oos_sharpes)

        # Overfitting: IS performance substantially exceeds OOS
        # Use a generous threshold: mean IS Sharpe > 2x mean OOS Sharpe
        overfitting = mean_is > 0 and mean_oos < mean_is * 0.5

        label = robustness_label(
            mean_ratio, self.config.robustness_good, self.config.robustness_strong
        )

        return WalkForwardResult(
            config=self.config,
            folds=folds,
            mean_is_sharpe=mean_is,
            mean_oos_sharpe=mean_oos,
            mean_robustness_ratio=mean_ratio,
            median_robustness_ratio=median_ratio,
            overfitting_detected=overfitting,
            robustness_label=label,
        )
