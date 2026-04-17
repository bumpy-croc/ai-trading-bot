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
                raise ValueError(f"Override key {key!r} is not in '<strategy>.<attr>' form")

            namespace, attr = key.split(".", 1)
            if namespace.replace("_", "").lower() != strategy_key:
                # Namespace mismatch is almost always a copy-paste bug — fail
                # loudly so the user notices rather than running the variant
                # with unintended baseline-like behavior.
                raise ValueError(
                    f"Override {key!r} targets strategy namespace "
                    f"{namespace!r} but suite strategy is {config.strategy_name!r}"
                )

            if not self._apply_strategy_attribute(strategy, attr, value):
                raise ValueError(
                    f"Unknown override attribute {attr!r} for strategy "
                    f"{config.strategy_name!r}; no component accepts it."
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
            # atr_multiplier exists on VolatilityRiskManager only; the map tries
            # risk_manager first and falls through with a clear error if the
            # active strategy's risk manager does not accept it.
            "atr_multiplier": [risk_manager],
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
            except (TypeError, ValueError) as exc:
                # Property setters with validation raise these. Propagate so
                # invalid overrides fail the suite instead of silently
                # leaving the component in a half-configured state.
                raise ValueError(
                    f"Failed to apply override {attr!r}={value!r} to "
                    f"{type(target).__name__}: {exc}"
                ) from exc

        # Risk-related attributes must land in the strategy's risk-overrides
        # mapping AND the active risk manager must actually consume it at
        # trade time. Only CoreRiskAdapter reads ``context["strategy_overrides"]``
        # (via ``_resolve_overrides``). Other risk managers
        # (``RegimeAdaptiveRiskManager`` for ml_adaptive,
        # ``VolatilityRiskManager`` / ``FixedRiskManager``) derive stops
        # from regime / ATR / constants and would silently ignore the
        # override — producing meaningless variant rankings.
        if attr in {"stop_loss_pct", "take_profit_pct"}:
            try:
                numeric_value = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Override {attr!r} must be numeric, got {value!r}") from exc
            if risk_manager is not None and not hasattr(risk_manager, "_strategy_overrides"):
                raise ValueError(
                    f"Override {attr!r} is not supported for "
                    f"{type(risk_manager).__name__}; the risk manager does "
                    "not consume strategy_overrides at trade time. Only "
                    "CoreRiskAdapter-backed strategies (ml_basic, "
                    "ml_sentiment, hyper_growth) honor this knob."
                )
            overrides = getattr(strategy, "_risk_overrides", None) or {}
            overrides[attr] = numeric_value
            strategy._risk_overrides = overrides
            # Mirror onto the risk manager's internal overrides map so that
            # ``_resolve_overrides`` (merge adapter + context) sees the
            # change even for code paths that don't pass context through.
            if risk_manager is not None and hasattr(risk_manager, "_strategy_overrides"):
                risk_manager._strategy_overrides[attr] = numeric_value
            applied = True

        return applied

    @staticmethod
    def _coerce_value(current: object, new_value: object) -> object:
        """Coerce overrides to match the existing attribute type when possible.

        Int → float widening: when the existing attribute is an ``int`` but
        the override is a non-integer ``float``, keep the float precision
        rather than silently truncating (e.g. ``confidence_multiplier = 12``
        overridden to ``20.5`` must not become ``20``).

        For numeric targets, coercion failures raise ``ValueError`` — a
        silent fallback to the raw string would corrupt the attribute and
        push the failure deep into the backtest.
        """

        if current is None:
            return new_value

        target_type = type(current)
        if target_type is int and isinstance(new_value, float) and not new_value.is_integer():
            return new_value
        try:
            return target_type(new_value)  # type: ignore[call-arg]
        except (TypeError, ValueError) as exc:
            if target_type in (int, float):
                raise ValueError(
                    f"override value {new_value!r} is not convertible to " f"{target_type.__name__}"
                ) from exc
            return new_value

    @staticmethod
    def _validate_post_override_invariants(strategy: Strategy) -> None:
        """Re-check component invariants after override setattrs ran.

        ``setattr`` bypasses ``__init__`` so override combinations that
        violate a component's construction invariants must be re-validated
        here. Bounds mirror what the components enforce at construction:

        * ``ConfidenceWeightedSizer``: ``base_fraction`` ∈ [0.001, 0.5],
          ``min_confidence`` / ``min_confidence_floor`` ∈ [0, 1],
          ``min_confidence_floor`` ≤ ``min_confidence``.
        * ``VolatilityRiskManager``: ``base_risk`` ∈ [0.001, 0.1],
          ``atr_multiplier`` ∈ [0.5, 5.0], ``min_risk`` ≤ ``max_risk``.
        """
        signal_generator = getattr(strategy, "signal_generator", None)
        if signal_generator is not None:
            # ``setattr`` on these fields bypasses the construction-time
            # finite check in :func:`_require_finite`. Rechecking here
            # keeps NaN/Inf overrides from silently corrupting signal
            # decisions (NaN comparisons return False).
            for attr in (
                "long_entry_threshold",
                "short_entry_threshold",
                "short_threshold_trend_up",
                "short_threshold_trend_down",
                "short_threshold_range",
                "short_threshold_high_vol",
                "short_threshold_low_vol",
                "short_threshold_confidence_multiplier",
            ):
                _require_finite_attr(signal_generator, attr)
            _require_finite_attr(signal_generator, "confidence_multiplier", positive=True)
            # sequence_length slices DataFrames via ``.iloc[index - n : index]``
            # and must stay an integer. The ``_coerce_value`` widen-on-non-int
            # rule preserves float precision for multipliers; we undo that
            # here for this specific index-like attribute.
            seq_len = getattr(signal_generator, "sequence_length", None)
            if seq_len is not None:
                if isinstance(seq_len, bool) or not isinstance(seq_len, int | float):
                    raise ValueError(f"sequence_length must be numeric, got {seq_len!r}")
                if isinstance(seq_len, float) and not seq_len.is_integer():
                    raise ValueError(f"sequence_length must be a whole number, got {seq_len!r}")
                if int(seq_len) < 1:
                    raise ValueError(f"sequence_length must be >= 1, got {seq_len!r}")

        position_sizer = getattr(strategy, "position_sizer", None)
        if position_sizer is not None:
            _check_numeric_bound(position_sizer, "base_fraction", 0.001, 0.5)
            _check_numeric_bound(position_sizer, "min_confidence", 0.0, 1.0)
            _check_numeric_bound(position_sizer, "min_confidence_floor", 0.0, 1.0)
            floor = getattr(position_sizer, "min_confidence_floor", None)
            gate = getattr(position_sizer, "min_confidence", None)
            if isinstance(floor, int | float) and isinstance(gate, int | float) and floor > gate:
                raise ValueError(
                    f"Invalid sizer state after overrides: "
                    f"min_confidence_floor ({floor}) must be <= "
                    f"min_confidence ({gate}); floor exceeding gate would "
                    f"over-size low-confidence signals."
                )

        risk_manager = getattr(strategy, "risk_manager", None)
        if risk_manager is not None:
            _check_numeric_bound(risk_manager, "base_risk", 0.001, 0.1)
            _check_numeric_bound(risk_manager, "atr_multiplier", 0.5, 5.0)
            _check_numeric_bound(risk_manager, "min_risk", 0.001, 0.2)
            _check_numeric_bound(risk_manager, "max_risk", 0.001, 0.2)
            min_risk = getattr(risk_manager, "min_risk", None)
            max_risk = getattr(risk_manager, "max_risk", None)
            if (
                isinstance(min_risk, int | float)
                and isinstance(max_risk, int | float)
                and min_risk > max_risk
            ):
                raise ValueError(
                    f"Invalid risk state after overrides: "
                    f"min_risk ({min_risk}) must be <= max_risk ({max_risk})."
                )

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        strategy = self._load_strategy(config.strategy_name)
        # Apply any parameter overrides for strategy-level tuning
        self._apply_parameter_overrides(strategy, config)
        self._validate_post_override_invariants(strategy)
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


def _check_numeric_bound(
    target: object,
    attr: str,
    lower: float,
    upper: float,
) -> None:
    """Raise ValueError when ``target.attr`` is set and outside [lower, upper].

    Also raises when the attribute exists but holds a non-numeric value
    (string override slipping past ``_coerce_value``), or a non-finite one
    (NaN fails both ``< lower`` and ``> upper`` comparisons, so an earlier
    version silently accepted it).
    """
    import math as _math

    if not hasattr(target, attr):
        return
    value = getattr(target, attr)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(
            f"Invalid state after overrides: {type(target).__name__}.{attr} "
            f"must be numeric, got {value!r}."
        )
    if not _math.isfinite(value):
        raise ValueError(
            f"Invalid state after overrides: {type(target).__name__}.{attr} "
            f"must be finite, got {value!r}."
        )
    if value < lower or value > upper:
        raise ValueError(
            f"Invalid state after overrides: {type(target).__name__}.{attr} "
            f"({value}) must be in [{lower}, {upper}]."
        )


def _require_finite_attr(target: object, attr: str, *, positive: bool = False) -> None:
    """Raise ValueError when ``target.attr`` is non-numeric or non-finite.

    ``positive=True`` additionally requires the value to be > 0. A string
    override to a numeric knob (e.g. YAML ``long_entry_threshold: "abc"``)
    no longer slips through: we require an ``int | float`` and fail loudly.
    """
    import math as _math  # local import to avoid polluting module-level name

    if not hasattr(target, attr):
        return
    value = getattr(target, attr)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(
            f"Invalid state after overrides: {type(target).__name__}.{attr} "
            f"must be numeric, got {value!r}."
        )
    if not _math.isfinite(value):
        raise ValueError(
            f"Invalid state after overrides: {type(target).__name__}.{attr} "
            f"must be finite, got {value!r}."
        )
    if positive and value <= 0:
        raise ValueError(
            f"Invalid state after overrides: {type(target).__name__}.{attr} "
            f"must be > 0, got {value!r}."
        )
