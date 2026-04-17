from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider
from src.data_providers.coinbase_provider import CoinbaseProvider
from src.data_providers.offline import FixtureProvider, RandomWalkProvider
from src.engines.backtest.engine import Backtester
from src.experiments.schemas import ExperimentConfig, ExperimentResult
from src.risk.risk_manager import RiskParameters
from src.strategies.components import Strategy
from src.strategies.hyper_growth import create_hyper_growth_strategy
from src.strategies.ml_adaptive import create_ml_adaptive_strategy
from src.strategies.ml_basic import create_ml_basic_strategy
from src.strategies.ml_sentiment import create_ml_sentiment_strategy


class ExperimentRunner:
    """Runs backtests for given experiment configurations."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_provider(
        self,
        name: str,
        use_cache: bool,
        cache_ttl_hours: int = 24,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        timeframe: str = "1h",
        seed: int | None = None,
    ):
        name = (name or "binance").lower()
        if name == "mock":
            return RandomWalkProvider(
                start or (datetime.now(UTC) - timedelta(days=30)),
                end or datetime.now(UTC),
                timeframe=timeframe,
                seed=seed,
            )
        if name == "fixture":
            fixture_path = Path("tests/data/BTCUSDT_1h_2023-01-01_2024-12-31.feather")
            return FixtureProvider(fixture_path)

        if name == "coinbase":
            provider = CoinbaseProvider()
        else:
            provider = BinanceProvider()
        if use_cache:
            return CachedDataProvider(provider, cache_ttl_hours=cache_ttl_hours)
        return provider

    def _load_strategy(self, strategy_name: str) -> Strategy:
        strategies = {
            "ml_basic": create_ml_basic_strategy,
            "ml_adaptive": create_ml_adaptive_strategy,
            "ml_sentiment": create_ml_sentiment_strategy,
            "hyper_growth": create_hyper_growth_strategy,
        }
        builder = strategies.get(strategy_name)
        if builder is not None:
            return builder()
        raise ValueError(f"Unknown strategy: {strategy_name}")

    def _apply_parameter_overrides(self, strategy: Strategy, config: ExperimentConfig) -> None:
        if not (config.parameters and config.parameters.values):
            return

        strategy_key = config.strategy_name.replace("_", "").lower()

        for key, value in config.parameters.values.items():
            if "." not in key:
                continue

            namespace, attr = key.split(".", 1)
            if namespace.replace("_", "").lower() != strategy_key:
                continue

            if not self._apply_strategy_attribute(strategy, attr, value):
                self.logger.debug(
                    "Failed to apply override %s for strategy %s", key, config.strategy_name
                )

    def _apply_strategy_attribute(self, strategy: Strategy, attr: str, value: object) -> bool:
        """Apply overrides to the correct component on a component-based strategy."""

        signal_generator = getattr(strategy, "signal_generator", None)
        risk_manager = getattr(strategy, "risk_manager", None)
        position_sizer = getattr(strategy, "position_sizer", None)

        component_targets: dict[str, list[Strategy | object | None]] = {
            # Risk manager attributes
            "stop_loss_pct": [strategy, risk_manager],
            "risk_per_trade": [risk_manager],
            "trailing_stop_pct": [strategy, risk_manager],
            "atr_multiplier": [strategy, risk_manager],
            # Position sizer attributes
            "base_fraction": [position_sizer],
            "min_confidence": [position_sizer],
            "min_confidence_floor": [position_sizer],
            # Signal generator attributes
            "sequence_length": [signal_generator],
            "model_path": [strategy, signal_generator],
            "use_prediction_engine": [strategy, signal_generator],
            "model_name": [strategy, signal_generator],
            "model_type": [strategy, signal_generator],
            "timeframe": [strategy, signal_generator],
            "long_entry_threshold": [signal_generator],
            "short_entry_threshold": [signal_generator],
            "confidence_multiplier": [signal_generator],
            "short_threshold_trend_up": [signal_generator],
            "short_threshold_trend_down": [signal_generator],
            "short_threshold_range": [signal_generator],
            "short_threshold_high_vol": [signal_generator],
            "short_threshold_low_vol": [signal_generator],
            "short_threshold_confidence_multiplier": [signal_generator],
            # Strategy level attributes
            "take_profit_pct": [strategy],
        }

        targets = component_targets.get(attr, [strategy])
        applied = False
        coerced_value: object = value

        for target in targets:
            if target is None or not hasattr(target, attr):
                continue
            try:
                current = getattr(target, attr)
                coerced_value = self._coerce_value(current, value)
                setattr(target, attr, coerced_value)
                applied = True
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.debug("Failed to set %s on %s: %s", attr, target, exc)

        # Risk-related attributes must always land in the strategy's risk
        # overrides mapping — the risk manager consults this at runtime, and
        # some strategies (e.g. ml_basic) do not expose them as instance attrs.
        if attr in {"stop_loss_pct", "take_profit_pct"}:
            overrides = getattr(strategy, "_risk_overrides", None) or {}
            overrides[attr] = self._coerce_value(overrides.get(attr), value)
            strategy._risk_overrides = overrides
            applied = True

        return applied

    @staticmethod
    def _coerce_value(current: object, new_value: object) -> object:
        """Coerce overrides to the same type as the existing attribute when possible."""

        if current is None:
            return new_value

        target_type = type(current)
        try:
            return target_type(new_value)
        except Exception:
            return new_value

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        strategy = self._load_strategy(config.strategy_name)
        # Apply any parameter overrides for strategy-level tuning
        self._apply_parameter_overrides(strategy, config)
        provider = self._load_provider(
            config.provider,
            config.use_cache,
            start=config.start,
            end=config.end,
            timeframe=config.timeframe,
            seed=config.random_seed,
        )

        risk_params = (
            RiskParameters(**config.risk_parameters) if config.risk_parameters else RiskParameters()
        )

        backtester = Backtester(
            strategy=strategy,
            data_provider=provider,
            sentiment_provider=None,
            risk_parameters=risk_params,
            initial_balance=config.initial_balance,
            log_to_database=False,
        )

        results = backtester.run(
            symbol=config.symbol,
            timeframe=config.timeframe,
            start=config.start,
            end=config.end,
        )

        return ExperimentResult(
            config=config,
            total_trades=int(results.get("total_trades", 0)),
            win_rate=float(results.get("win_rate", 0.0)),
            total_return=float(results.get("total_return", 0.0)),
            annualized_return=float(results.get("annualized_return", 0.0)),
            max_drawdown=float(results.get("max_drawdown", 0.0)),
            sharpe_ratio=float(results.get("sharpe_ratio", 0.0)),
            final_balance=float(results.get("final_balance", config.initial_balance)),
            session_id=results.get("session_id"),
        )

    def run_sweep(self, configs: list[ExperimentConfig]) -> list[ExperimentResult]:
        return [self.run(cfg) for cfg in configs]
