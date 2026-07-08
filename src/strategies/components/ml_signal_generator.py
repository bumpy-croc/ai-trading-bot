"""
ML Signal Generator Components

This module contains ML-based signal generators extracted from existing strategies.
These components generate trading signals using machine learning models with
regime-aware threshold adjustments and confidence calculations.
"""

import logging
import math
import time
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from src.prediction.models.registry import PredictionModelRegistry, StrategyModel

from src.config.config_manager import get_config
from src.config.feature_flags import is_enabled
from src.prediction import PredictionConfig, PredictionEngine
from src.prediction.features.pipeline import FeaturePipeline
from src.prediction.features.price_only import PriceOnlyFeatureExtractor
from src.prediction.models.exceptions import ModelNotAvailableError
from src.regime.detector import TrendLabel, VolLabel
from src.strategies.components.regime_context import RegimeContext
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator
from src.trading.symbols.factory import SymbolFactory

logger = logging.getLogger(__name__)


def _require_finite(name: str, value: Any) -> float:
    """Coerce ``value`` to ``float`` and reject NaN/Inf.

    NaN comparisons are always False, so a NaN threshold would silently
    disable an entry side; a NaN multiplier poisons downstream confidence
    calculations. Rejecting at construction time makes tuning-override
    mistakes fail loudly.
    """
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(coerced):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return coerced


def _require_finite_positive(name: str, value: Any) -> float:
    """Like :func:`_require_finite` but also requires ``value > 0``."""
    coerced = _require_finite(name, value)
    if coerced <= 0:
        raise ValueError(f"{name} must be > 0, got {coerced}")
    return coerced


class MLSignalGenerator(SignalGenerator):
    """
    ML Signal Generator extracted from MlAdaptive strategy

    This signal generator uses machine learning models to predict price movements
    and generates trading signals with regime-aware threshold adjustments.

    Features:
    - ONNX model inference for price predictions
    - Regime-aware dynamic threshold adjustment
    - Confidence calculation based on prediction quality
    - Optional prediction engine integration
    """

    # Default thresholds (overridable per-instance for experiments).
    # LONG_ENTRY_THRESHOLD=0.0 preserves pre-refactor behavior (any positive
    # predicted return triggers BUY). Experiments tighten via overrides.
    LONG_ENTRY_THRESHOLD = 0.0
    SHORT_ENTRY_THRESHOLD = -0.0005  # Base threshold for short entries (-0.05%)
    CONFIDENCE_MULTIPLIER = 12.0  # Multiplier for confidence calculation

    # Dynamic threshold configuration for different market regimes
    SHORT_THRESHOLD_TREND_UP = -0.0003  # Less conservative in uptrend (-0.03%)
    SHORT_THRESHOLD_TREND_DOWN = -0.0007  # More conservative in downtrend (-0.07%)
    SHORT_THRESHOLD_RANGE = -0.0005  # Standard threshold in range-bound market (-0.05%)
    SHORT_THRESHOLD_HIGH_VOL = -0.0004  # Less conservative in high volatility (-0.04%)
    SHORT_THRESHOLD_LOW_VOL = -0.0006  # More conservative in low volatility (-0.06%)
    SHORT_THRESHOLD_CONFIDENCE_MULTIPLIER = 0.2  # Adjust threshold based on regime confidence

    # Registry-selection hints optionally attached by strategy factories
    # (e.g. create_ml_sentiment_strategy). Bare annotations so hasattr()
    # behavior is unchanged until a factory assigns them.
    model_type: str
    model_timeframe: str

    def __init__(
        self,
        name: str = "ml_signal_generator",
        sequence_length: int = 120,
        model_name: str | None = None,
        *,
        long_entry_threshold: float | None = None,
        short_entry_threshold: float | None = None,
        confidence_multiplier: float | None = None,
        short_threshold_trend_up: float | None = None,
        short_threshold_trend_down: float | None = None,
        short_threshold_range: float | None = None,
        short_threshold_high_vol: float | None = None,
        short_threshold_low_vol: float | None = None,
        short_threshold_confidence_multiplier: float | None = None,
    ):
        """
        Initialize ML Signal Generator

        Args:
            name: Name for this signal generator
            sequence_length: Sequence length for model input
            model_name: Model name for prediction engine (optional)
            long_entry_threshold: Minimum predicted return to open a long.
            short_entry_threshold: Maximum predicted return to open a short
                (expected to be negative).
            confidence_multiplier: Scales |predicted_return| → confidence score.
            short_threshold_*: Regime-specific short thresholds; defaults fall
                back to the corresponding class constants.
        """
        super().__init__(name)

        self.sequence_length = sequence_length

        # Instance-level thresholds (experiments override these without mutating
        # class state shared across strategies). Every numeric knob is
        # validated finite via ``_require_finite`` so a NaN/Inf override
        # (trivial to land via YAML) cannot silently corrupt a trading
        # decision — NaN comparisons are always False so a NaN threshold
        # would disable entries, and a NaN multiplier would NaN-poison
        # confidence scores.
        self.long_entry_threshold = _require_finite(
            "long_entry_threshold",
            self.LONG_ENTRY_THRESHOLD if long_entry_threshold is None else long_entry_threshold,
        )
        self.short_entry_threshold = _require_finite(
            "short_entry_threshold",
            self.SHORT_ENTRY_THRESHOLD if short_entry_threshold is None else short_entry_threshold,
        )
        self.confidence_multiplier = _require_finite_positive(
            "confidence_multiplier",
            self.CONFIDENCE_MULTIPLIER if confidence_multiplier is None else confidence_multiplier,
        )
        self.short_threshold_trend_up = _require_finite(
            "short_threshold_trend_up",
            (
                self.SHORT_THRESHOLD_TREND_UP
                if short_threshold_trend_up is None
                else short_threshold_trend_up
            ),
        )
        self.short_threshold_trend_down = _require_finite(
            "short_threshold_trend_down",
            (
                self.SHORT_THRESHOLD_TREND_DOWN
                if short_threshold_trend_down is None
                else short_threshold_trend_down
            ),
        )
        self.short_threshold_range = _require_finite(
            "short_threshold_range",
            self.SHORT_THRESHOLD_RANGE if short_threshold_range is None else short_threshold_range,
        )
        self.short_threshold_high_vol = _require_finite(
            "short_threshold_high_vol",
            (
                self.SHORT_THRESHOLD_HIGH_VOL
                if short_threshold_high_vol is None
                else short_threshold_high_vol
            ),
        )
        self.short_threshold_low_vol = _require_finite(
            "short_threshold_low_vol",
            (
                self.SHORT_THRESHOLD_LOW_VOL
                if short_threshold_low_vol is None
                else short_threshold_low_vol
            ),
        )
        self.short_threshold_confidence_multiplier = _require_finite(
            "short_threshold_confidence_multiplier",
            (
                self.SHORT_THRESHOLD_CONFIDENCE_MULTIPLIER
                if short_threshold_confidence_multiplier is None
                else short_threshold_confidence_multiplier
            ),
        )

        # Model name configuration
        cfg = get_config()
        self.model_name = model_name
        if self.model_name is None:
            self.model_name = cfg.get("PREDICTION_ENGINE_MODEL_NAME", default=None)

        self.prediction_engine: PredictionEngine | None = None
        self._engine_warning_emitted = False
        self.use_engine_batch = get_config().get_bool("ENGINE_BATCH_INFERENCE", default=False)
        # True when the most recent failed prediction was a live inference
        # timeout — stamped into the HOLD signal for attributability.
        self._last_prediction_timed_out = False

        # Initialize feature pipeline
        self._setup_feature_pipeline()

        # Initialize prediction engine (always enabled)
        self._initialize_prediction_engine()

    def _setup_feature_pipeline(self):
        """Setup feature pipeline for data preprocessing"""
        # Use price-only extractor for prediction engine
        price_only = PriceOnlyFeatureExtractor(normalization_window=self.sequence_length)
        config = {
            "technical_features": {"enabled": False},
            "sentiment_features": {"enabled": False},
            "market_features": {"enabled": False},
            "price_only_features": {"enabled": False},
        }
        self.feature_pipeline = FeaturePipeline(
            config=config,
            custom_extractors=[price_only],
        )

    def _initialize_prediction_engine(self):
        """Initialize prediction engine with health check"""
        try:
            config = PredictionConfig.from_config_manager()
            config.enable_sentiment = False
            config.enable_market_microstructure = False
            engine = PredictionEngine(config)

            # Setup feature pipeline for engine
            pipeline_config = {
                "technical_features": {"enabled": False},
                "sentiment_features": {"enabled": False},
                "market_features": {"enabled": False},
                "price_only_features": {"enabled": False},
            }
            engine.feature_pipeline = FeaturePipeline(
                config=pipeline_config,
                custom_extractors=[
                    PriceOnlyFeatureExtractor(normalization_window=self.sequence_length)
                ],
            )

            # Health check
            health = engine.health_check()
            if health.get("status") != "healthy" and not self._engine_warning_emitted:
                logger.warning("MLSignalGenerator: Prediction engine health degraded: %s", health)
                self._engine_warning_emitted = True

            self.prediction_engine = engine

        except Exception:
            if not self._engine_warning_emitted:
                logger.exception("MLSignalGenerator: Prediction engine initialization failed")
                self._engine_warning_emitted = True
            self.prediction_engine = None

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: RegimeContext | None = None
    ) -> Signal:
        """
        Generate trading signal based on ML prediction

        Args:
            df: DataFrame with OHLCV data and calculated indicators
            index: Current index position in the DataFrame
            regime: Optional regime context for regime-aware signal generation

        Returns:
            Signal object with direction, strength, confidence, and metadata
        """
        self.validate_inputs(df, index)

        # Ensure we have enough history for prediction
        if index < self.sequence_length:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    "generator": self.name,
                    "reason": "insufficient_history",
                    "index": index,
                    "required_length": self.sequence_length,
                },
            )

        # Get ML prediction
        prediction = self._get_ml_prediction(df, index)
        if prediction is None:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    "generator": self.name,
                    "reason": "prediction_failed",
                    "timed_out": self._last_prediction_timed_out,
                    "index": index,
                },
            )

        current_price = df["close"].iloc[index]

        # Validate prediction and current_price are finite to prevent NaN/Infinity propagation
        if not (math.isfinite(prediction) and math.isfinite(current_price) and current_price > 0):
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    "generator": self.name,
                    "reason": "invalid_prediction_or_price",
                    "prediction": prediction,
                    "current_price": current_price,
                    "predicted_return": 0,  # Set to 0 when price is invalid
                    "index": index,
                },
            )

        predicted_return = (prediction - current_price) / current_price

        # Determine signal direction
        if predicted_return > self.long_entry_threshold:
            direction = SignalDirection.BUY
            strength = min(1.0, abs(predicted_return) * 10)  # Scale to 0-1
        elif self._should_generate_short_signal(predicted_return, regime):
            direction = SignalDirection.SELL
            strength = min(1.0, abs(predicted_return) * 10)  # Scale to 0-1
        else:
            direction = SignalDirection.HOLD
            strength = 0.0

        # Calculate confidence
        confidence = self._calculate_confidence(predicted_return)

        # Create metadata
        metadata = {
            "generator": self.name,
            "prediction": prediction,
            "current_price": current_price,
            "predicted_return": predicted_return,
            "index": index,
            "sequence_length": self.sequence_length,
            "long_entry_threshold": self.long_entry_threshold,
            "engine_model_name": self.model_name,
            "engine_batch": self.use_engine_batch,
        }

        # Add regime information if available
        if regime:
            metadata.update(
                {
                    "regime_trend": regime.trend.value,
                    "regime_volatility": regime.volatility.value,
                    "regime_confidence": regime.confidence,
                    "dynamic_threshold": self._calculate_dynamic_short_threshold(regime),
                }
            )

        return Signal(
            direction=direction, strength=strength, confidence=confidence, metadata=metadata
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """
        Get confidence score for signal generation at the given index

        Args:
            df: DataFrame containing OHLCV data with calculated indicators
            index: Current index position in the DataFrame

        Returns:
            Confidence score between 0.0 and 1.0
        """
        self.validate_inputs(df, index)

        if index < self.sequence_length:
            return 0.0

        # Get ML prediction
        prediction = self._get_ml_prediction(df, index)
        if prediction is None:
            return 0.0

        current_price = df["close"].iloc[index]

        # Return zero confidence if prediction or price are invalid (prevents NaN propagation)
        if not (math.isfinite(prediction) and math.isfinite(current_price) and current_price > 0):
            return 0.0

        predicted_return = (prediction - current_price) / current_price

        return self._calculate_confidence(predicted_return)

    def _get_ml_prediction(self, df: pd.DataFrame, index: int) -> float | None:
        """
        Get ML prediction for the given index

        Args:
            df: DataFrame with processed features
            index: Current index position

        Returns:
            Predicted price or None if prediction fails
        """
        self._last_prediction_timed_out = False
        try:
            # Feature pipeline transformation currently not used
            # (prediction engine handles raw price data directly)

            # Get prediction from prediction engine
            if self.prediction_engine is None:
                logger.error("Prediction engine not initialized for symbol prediction")
                return None

            window_df = df[["open", "high", "low", "close", "volume"]].iloc[
                index - self.sequence_length : index
            ]
            result = self.prediction_engine.predict(window_df, model_name=self.model_name)

            # An errored result carries a placeholder price of 0.0 — treating
            # it as real would fabricate a -100% predicted return (phantom
            # SELL). Degrade to a failed prediction (caller emits HOLD).
            if result.error is not None:
                self._last_prediction_timed_out = bool(result.metadata.get("timed_out", False))
                logger.warning(
                    "MLSignalGenerator: prediction failed at index %d (timed_out=%s): %s",
                    index,
                    self._last_prediction_timed_out,
                    result.error,
                )
                return None

            pred = float(result.price)

            # Prediction engine returns real prices
            return pred

        except Exception:
            logger.exception("MLSignalGenerator: Prediction error at index %d", index)
            return None

    def _should_generate_short_signal(
        self, predicted_return: float, regime: RegimeContext | None
    ) -> bool:
        """
        Determine if a short signal should be generated based on predicted return and regime

        Args:
            predicted_return: Predicted return value
            regime: Current regime context

        Returns:
            True if short signal should be generated
        """
        if regime is not None:
            dynamic_threshold = self._calculate_dynamic_short_threshold(regime)
        else:
            dynamic_threshold = self.short_entry_threshold

        return predicted_return < dynamic_threshold

    def _calculate_dynamic_short_threshold(self, regime: RegimeContext) -> float:
        """
        Calculate dynamic short entry threshold based on current market regime

        Args:
            regime: Current regime context

        Returns:
            Dynamic threshold for short entries
        """
        # Start with base threshold based on trend
        if regime.trend == TrendLabel.TREND_UP:
            base_threshold = self.short_threshold_trend_up
        elif regime.trend == TrendLabel.TREND_DOWN:
            base_threshold = self.short_threshold_trend_down
        else:  # range
            base_threshold = self.short_threshold_range

        # Adjust for volatility
        if regime.volatility == VolLabel.HIGH:
            vol_adjustment = self.short_threshold_high_vol
        else:  # low_vol
            vol_adjustment = self.short_threshold_low_vol

        # Combine trend and volatility adjustments (weighted average)
        threshold = (base_threshold + vol_adjustment) / 2

        # Adjust based on regime confidence
        # Higher confidence = more aggressive threshold (closer to 0)
        # Lower confidence = more conservative threshold (further from 0)
        # Since threshold is negative, we scale it: high confidence reduces magnitude, low confidence increases magnitude
        confidence_factor = 1 + regime.confidence * self.short_threshold_confidence_multiplier
        threshold = threshold / confidence_factor

        # Ensure threshold is within reasonable bounds
        threshold = max(-0.01, min(-0.0001, threshold))  # Between -1% and -0.01%

        return threshold

    def _calculate_confidence(self, predicted_return: float) -> float:
        """
        Calculate confidence score based on predicted return magnitude

        Args:
            predicted_return: Predicted return value

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = min(1.0, abs(predicted_return) * self.confidence_multiplier)
        return max(0.0, confidence)

    @property
    def warmup_period(self) -> int:
        """Return the minimum history required before producing valid signals."""
        return self.sequence_length

    def get_parameters(self) -> dict[str, Any]:
        """Get signal generator parameters for logging and serialization"""
        params = super().get_parameters()
        params.update(
            {
                "sequence_length": self.sequence_length,
                "model_name": self.model_name,
                "long_entry_threshold": self.long_entry_threshold,
                "short_entry_threshold": self.short_entry_threshold,
                "confidence_multiplier": self.confidence_multiplier,
                "short_threshold_trend_up": self.short_threshold_trend_up,
                "short_threshold_trend_down": self.short_threshold_trend_down,
                "short_threshold_range": self.short_threshold_range,
                "short_threshold_high_vol": self.short_threshold_high_vol,
                "short_threshold_low_vol": self.short_threshold_low_vol,
                "short_threshold_confidence_multiplier": (
                    self.short_threshold_confidence_multiplier
                ),
            }
        )
        return params


class MLBasicSignalGenerator(SignalGenerator):
    """
    ML Basic Signal Generator extracted from MlBasic strategy

    This signal generator uses machine learning models to predict price movements
    and generates basic trading signals without regime awareness.

    Features:
    - ONNX model inference for price predictions
    - Basic signal generation without regime-specific adjustments
    - Confidence calculation based on prediction quality
    - Optional prediction engine integration with model registry support
    """

    # Default thresholds (overridable per-instance for experiments).
    # LONG_ENTRY_THRESHOLD=0.0 preserves pre-refactor behavior (any positive
    # predicted return triggers BUY). Experiments tighten via overrides.
    LONG_ENTRY_THRESHOLD = 0.0
    SHORT_ENTRY_THRESHOLD = -0.0005  # -0.05% threshold for short entries
    CONFIDENCE_MULTIPLIER = 12.0  # Multiplier for confidence calculation

    # Default symbol used for registry selection when none specified
    DEFAULT_SYMBOL = "BTCUSDT"

    # Feature flag that explicitly opts into scoring one symbol with another
    # symbol's model when no model exists for the trading symbol.
    CROSS_SYMBOL_FLAG = "allow_cross_symbol_model"

    # Guard logs (mismatch/substitution) are emitted at most once per interval
    # per generator instance so a 1m loop cannot flood the logs.
    SYMBOL_GUARD_LOG_INTERVAL_SECONDS = 300.0

    def __init__(
        self,
        name: str = "ml_basic_signal_generator",
        sequence_length: int = 120,
        model_name: str | None = None,
        model_type: str | None = None,
        timeframe: str | None = None,
        symbol: str | None = None,
        *,
        long_entry_threshold: float | None = None,
        short_entry_threshold: float | None = None,
        confidence_multiplier: float | None = None,
    ):
        """
        Initialize ML Basic Signal Generator

        Args:
            name: Name for this signal generator
            sequence_length: Sequence length for model input
            model_name: Model name for prediction engine (optional)
            model_type: Model type for registry selection (optional)
            timeframe: Timeframe for registry selection (optional)
            symbol: Trading symbol for registry selection (optional, defaults to BTCUSDT)
            long_entry_threshold: Minimum predicted return for long entry.
            short_entry_threshold: Maximum predicted return for short entry.
            confidence_multiplier: Scales |predicted_return| → confidence.
        """
        super().__init__(name)

        self.sequence_length = sequence_length

        # Instance-level thresholds (experiments override these without mutating
        # class state shared across strategies). Validate finite to block
        # NaN/Inf overrides from corrupting signal decisions.
        self.long_entry_threshold = _require_finite(
            "long_entry_threshold",
            self.LONG_ENTRY_THRESHOLD if long_entry_threshold is None else long_entry_threshold,
        )
        self.short_entry_threshold = _require_finite(
            "short_entry_threshold",
            self.SHORT_ENTRY_THRESHOLD if short_entry_threshold is None else short_entry_threshold,
        )
        self.confidence_multiplier = _require_finite_positive(
            "confidence_multiplier",
            self.CONFIDENCE_MULTIPLIER if confidence_multiplier is None else confidence_multiplier,
        )

        # Registry model selection preferences. Normalize to the registry's
        # Binance-style directory convention so provider-specific formats
        # (e.g. Coinbase "ETH-USD") select the same model bundle.
        self.model_type = model_type or "basic"
        self.model_timeframe = timeframe or "1h"
        raw_symbol = symbol or self.DEFAULT_SYMBOL
        try:
            self.symbol = SymbolFactory.to_exchange_symbol(raw_symbol, "binance")
        except ValueError as exc:
            raise ValueError(f"Invalid trading symbol {raw_symbol!r}: {exc}") from exc

        # Model name configuration
        cfg = get_config()
        self.model_name = model_name
        if self.model_name is None:
            self.model_name = cfg.get("PREDICTION_ENGINE_MODEL_NAME", default=None)

        self.prediction_engine: PredictionEngine | None = None
        self._registry: PredictionModelRegistry | None = None
        self._engine_warning_emitted = False
        self.use_engine_batch = get_config().get_bool("ENGINE_BATCH_INFERENCE", default=False)
        # True when the most recent failed prediction was a live inference
        # timeout — stamped into the HOLD signal for attributability.
        self._last_prediction_timed_out = False

        # Cross-symbol guard state: which model symbol actually scored the
        # last prediction, the pinned substitute bundle (flag opt-in only),
        # and per-condition rate-limit clocks for guard logs (keyed by kind
        # so one condition's log cannot swallow another's).
        self._model_symbol: str | None = None
        self._cross_symbol_bundle_key: str | None = None
        self._symbol_guard_log_ts: dict[str, float] = {}

        # Initialize feature pipeline
        self._setup_feature_pipeline()

        # Initialize prediction engine (always enabled)
        self._initialize_prediction_engine()

        # Fail fast if the registry has no model for the trading symbol
        self._validate_model_availability()

    def _setup_feature_pipeline(self):
        """Setup feature pipeline for data preprocessing"""
        # Use price-only extractor for prediction engine
        price_only = PriceOnlyFeatureExtractor(normalization_window=self.sequence_length)
        config = {
            "technical_features": {"enabled": False},
            "sentiment_features": {"enabled": False},
            "market_features": {"enabled": False},
            "price_only_features": {"enabled": False},
        }
        self.feature_pipeline = FeaturePipeline(
            config=config,
            custom_extractors=[price_only],
        )

    def _initialize_prediction_engine(self):
        """Initialize prediction engine with health check and registry support"""
        try:
            config = PredictionConfig.from_config_manager()
            config.enable_sentiment = False
            config.enable_market_microstructure = False
            engine = PredictionEngine(config)

            # Setup feature pipeline for engine
            pipeline_config = {
                "technical_features": {"enabled": False},
                "sentiment_features": {"enabled": False},
                "market_features": {"enabled": False},
                "price_only_features": {"enabled": False},
            }
            engine.feature_pipeline = FeaturePipeline(
                config=pipeline_config,
                custom_extractors=[
                    PriceOnlyFeatureExtractor(normalization_window=self.sequence_length)
                ],
            )

            # Health check
            health = engine.health_check()
            if health.get("status") != "healthy" and not self._engine_warning_emitted:
                logger.warning(
                    "MLBasicSignalGenerator: Prediction engine health degraded: %s", health
                )
                self._engine_warning_emitted = True

            self.prediction_engine = engine

            # Initialize registry for structured selection
            try:
                self._registry = engine.model_registry
            except AttributeError:
                # Engine doesn't have model_registry attribute
                self._registry = None

        except Exception:
            if not self._engine_warning_emitted:
                logger.exception("MLBasicSignalGenerator: Prediction engine initialization failed")
                self._engine_warning_emitted = True
            self.prediction_engine = None

    def _validate_model_availability(self) -> None:
        """Fail fast when the registry has no model for the trading symbol.

        Without this check the prediction engine silently falls back to the
        default bundle — typically another symbol's model — and every signal
        is scored by a model trained on a different market.
        FEATURE_ALLOW_CROSS_SYMBOL_MODEL=true explicitly opts into that
        substitution as a transition path while the symbol's own model has
        not shipped yet.
        """
        if self._registry is None:
            # Engine degraded/unavailable — prediction path already fails
            # safe (returns None → HOLD), so nothing to validate against.
            return
        try:
            self._registry.select_bundle(
                symbol=self.symbol,
                model_type=self.model_type,
                timeframe=self.model_timeframe,
            )
            return
        except ModelNotAvailableError:
            pass
        except Exception:
            # Registry probing failure (not a missing model) — leave it to
            # the prediction path, which degrades to HOLD.
            return

        available = self._available_bundle_keys()
        if not is_enabled(self.CROSS_SYMBOL_FLAG, default=False):
            raise ModelNotAvailableError(
                f"No {self.model_type}/{self.model_timeframe} model exists for trading "
                f"symbol {self.symbol}. Available models: {', '.join(available) or 'none'}. "
                f"Train and deploy a {self.symbol} model, or set "
                f"FEATURE_ALLOW_CROSS_SYMBOL_MODEL=true to explicitly accept scoring "
                f"{self.symbol} with another symbol's model."
            )

        fallback = self._select_cross_symbol_fallback()
        if fallback is None:
            raise ModelNotAvailableError(
                f"FEATURE_ALLOW_CROSS_SYMBOL_MODEL=true but no "
                f"{self.model_type}/{self.model_timeframe} model exists for any symbol "
                f"to substitute for {self.symbol}. "
                f"Available models: {', '.join(available) or 'none'}."
            )
        self._cross_symbol_bundle_key = fallback.key
        self._model_symbol = fallback.symbol
        logger.critical(
            "CROSS-SYMBOL MODEL SUBSTITUTION ACTIVE: trading %s but scoring with %s "
            "model %s (FEATURE_ALLOW_CROSS_SYMBOL_MODEL=true). Predictions are not "
            "trained on %s — deploy a %s model and unset the flag as soon as possible.",
            self.symbol,
            fallback.symbol,
            fallback.key,
            self.symbol,
            self.symbol,
        )

    def _available_bundle_keys(self) -> list[str]:
        """List loaded bundle keys for error messages; empty on failure."""
        if self._registry is None:
            return []
        try:
            return sorted(str(bundle.key) for bundle in self._registry.list_bundles())
        except Exception:
            return []

    def _select_cross_symbol_fallback(self) -> "StrategyModel | None":
        """Pick a deterministic same-type/timeframe bundle from another symbol.

        Prefers DEFAULT_SYMBOL, then lexicographic order, so restarts always
        substitute the same model.
        """
        if self._registry is None:
            return None
        try:
            candidates = [
                bundle
                for bundle in self._registry.list_bundles()
                if bundle.model_type == self.model_type and bundle.timeframe == self.model_timeframe
            ]
        except Exception:
            return None
        if not candidates:
            return None
        candidates.sort(key=lambda bundle: (bundle.symbol != self.DEFAULT_SYMBOL, bundle.symbol))
        return candidates[0]

    def _log_symbol_guard(self, kind: str, level: int, msg: str, *args: Any) -> None:
        """Emit a guard log, rate-limited per instance and per condition kind.

        Separate clocks per ``kind`` so distinct conditions (substitution
        warning, mismatch, bundle vanished) each announce at least once per
        window instead of suppressing one another.
        """
        now = time.monotonic()
        last = self._symbol_guard_log_ts.get(kind)
        if last is not None and now - last < self.SYMBOL_GUARD_LOG_INTERVAL_SECONDS:
            return
        self._symbol_guard_log_ts[kind] = now
        logger.log(level, msg, *args)

    @staticmethod
    def _parse_model_symbol(model_name: Any) -> str | None:
        """Extract the symbol from a bundle key like ``BTCUSDT:1h:basic:v1``."""
        if isinstance(model_name, str) and ":" in model_name:
            return model_name.split(":", 1)[0]
        return None

    def _symbol_guard_stamps(self) -> dict[str, str | None]:
        """Cross-symbol guard stamps included in every Signal's metadata.

        ``model_symbol`` is the symbol whose model scored the most recent
        prediction (None when no prediction has resolved, e.g. HOLD paths).
        """
        return {"trading_symbol": self.symbol, "model_symbol": self._model_symbol}

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: RegimeContext | None = None
    ) -> Signal:
        """
        Generate trading signal based on ML prediction (basic, no regime awareness)

        Args:
            df: DataFrame with OHLCV data and calculated indicators
            index: Current index position in the DataFrame
            regime: Optional regime context (ignored in basic implementation)

        Returns:
            Signal object with direction, strength, confidence, and metadata
        """
        self.validate_inputs(df, index)

        # Ensure we have enough history for prediction
        if index < self.sequence_length:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    "generator": self.name,
                    "reason": "insufficient_history",
                    "index": index,
                    "required_length": self.sequence_length,
                    **self._symbol_guard_stamps(),
                },
            )

        # Get ML prediction
        prediction = self._get_ml_prediction(df, index)
        if prediction is None:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    "generator": self.name,
                    "reason": "prediction_failed",
                    "timed_out": self._last_prediction_timed_out,
                    "index": index,
                    **self._symbol_guard_stamps(),
                },
            )

        current_price = df["close"].iloc[index]

        # Validate prediction and current_price are finite to prevent NaN/Infinity propagation
        if not (math.isfinite(prediction) and math.isfinite(current_price) and current_price > 0):
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    "generator": self.name,
                    "reason": "invalid_prediction_or_price",
                    "prediction": prediction,
                    "current_price": current_price,
                    "predicted_return": 0,  # Set to 0 when price is invalid
                    "index": index,
                    **self._symbol_guard_stamps(),
                },
            )

        predicted_return = (prediction - current_price) / current_price

        # Determine signal direction (basic logic without regime awareness)
        if predicted_return > self.long_entry_threshold:
            direction = SignalDirection.BUY
            strength = min(1.0, abs(predicted_return) * 10)  # Scale to 0-1
        elif predicted_return < self.short_entry_threshold:
            direction = SignalDirection.SELL
            strength = min(1.0, abs(predicted_return) * 10)  # Scale to 0-1
        else:
            direction = SignalDirection.HOLD
            strength = 0.0

        # Calculate confidence
        confidence = self._calculate_confidence(predicted_return)

        # Create metadata
        metadata = {
            "generator": self.name,
            "prediction": prediction,
            "current_price": current_price,
            "predicted_return": predicted_return,
            "index": index,
            "sequence_length": self.sequence_length,
            "long_entry_threshold": self.long_entry_threshold,
            "short_entry_threshold": self.short_entry_threshold,
            "engine_model_name": self.model_name,
            "engine_batch": self.use_engine_batch,
            "model_type": self.model_type,
            "model_timeframe": self.model_timeframe,
            **self._symbol_guard_stamps(),
        }

        # Enable short entries for SELL signals
        if direction == SignalDirection.SELL:
            metadata["enter_short"] = True

        return Signal(
            direction=direction, strength=strength, confidence=confidence, metadata=metadata
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """
        Get confidence score for signal generation at the given index

        Args:
            df: DataFrame containing OHLCV data with calculated indicators
            index: Current index position in the DataFrame

        Returns:
            Confidence score between 0.0 and 1.0
        """
        self.validate_inputs(df, index)

        if index < self.sequence_length:
            return 0.0

        # Get ML prediction
        prediction = self._get_ml_prediction(df, index)
        if prediction is None:
            return 0.0

        current_price = df["close"].iloc[index]

        # Return zero confidence if prediction or price are invalid (prevents NaN propagation)
        if not (math.isfinite(prediction) and math.isfinite(current_price) and current_price > 0):
            return 0.0

        predicted_return = (prediction - current_price) / current_price

        return self._calculate_confidence(predicted_return)

    def _get_ml_prediction(self, df: pd.DataFrame, index: int) -> float | None:
        """
        Get ML prediction for the given index using prediction engine

        Args:
            df: DataFrame with processed features
            index: Current index position

        Returns:
            Predicted price or None if prediction fails
        """
        self._last_prediction_timed_out = False
        try:
            # Get prediction from prediction engine
            if self.prediction_engine is None:
                logger.warning("MLBasicSignalGenerator: Prediction engine not initialized")
                return None

            # Use prediction engine with registry selection
            window_df = df[["open", "high", "low", "close", "volume"]].iloc[
                index - self.sequence_length : index
            ]

            # Try to select bundle using registry
            selected_bundle_key = None
            resolved_model_symbol: str | None = None
            if self._cross_symbol_bundle_key is not None:
                # Startup explicitly opted into substitution via
                # FEATURE_ALLOW_CROSS_SYMBOL_MODEL — score with the pinned
                # bundle and keep reminding the operator.
                selected_bundle_key = self._cross_symbol_bundle_key
                resolved_model_symbol = self._model_symbol
                self._log_symbol_guard(
                    "cross_symbol_substitution",
                    logging.WARNING,
                    "Cross-symbol model substitution: scoring %s with %s model %s "
                    "(FEATURE_ALLOW_CROSS_SYMBOL_MODEL=true)",
                    self.symbol,
                    resolved_model_symbol,
                    selected_bundle_key,
                )
            elif self._registry is not None:
                try:
                    bundle = self._registry.select_bundle(
                        symbol=self.symbol,
                        model_type=self.model_type,
                        timeframe=self.model_timeframe,
                    )
                    selected_bundle_key = bundle.key
                    bundle_symbol = getattr(bundle, "symbol", None)
                    if isinstance(bundle_symbol, str):
                        resolved_model_symbol = bundle_symbol
                except ModelNotAvailableError:
                    # Bundle vanished after startup validation (e.g. registry
                    # reload). Refuse to silently score another symbol's
                    # model — fail the prediction (caller emits HOLD).
                    self._log_symbol_guard(
                        "model_unavailable",
                        logging.ERROR,
                        "No %s/%s model available for %s at prediction time — holding "
                        "(refusing cross-symbol fallback)",
                        self.model_type,
                        self.model_timeframe,
                        self.symbol,
                    )
                    self._model_symbol = None
                    return None
                except (KeyError, ValueError, AttributeError):
                    # Bundle not found or invalid - fall back to default
                    selected_bundle_key = None

            # Prefer registry selection by symbol/type/timeframe when available
            engine_model_name = selected_bundle_key or self.model_name
            try:
                if engine_model_name:
                    result = self.prediction_engine.predict(window_df, model_name=engine_model_name)
                else:
                    result = self.prediction_engine.predict(window_df)
            except (KeyError, ValueError):
                # Fall back to default registry resolution if explicit lookup fails
                result = self.prediction_engine.predict(window_df)

            # An errored result carries a placeholder price of 0.0 — treating
            # it as real would fabricate a -100% predicted return (phantom
            # SELL). Degrade to a failed prediction (caller emits HOLD).
            if result.error is not None:
                self._last_prediction_timed_out = bool(result.metadata.get("timed_out", False))
                self._model_symbol = None
                logger.warning(
                    "MLBasicSignalGenerator: prediction failed at index %d (timed_out=%s): %s",
                    index,
                    self._last_prediction_timed_out,
                    result.error,
                )
                return None

            # Detect which symbol's model actually scored this prediction so
            # signals can be stamped and mismatches surfaced loudly.
            if resolved_model_symbol is None:
                resolved_model_symbol = self._parse_model_symbol(
                    getattr(result, "model_name", None)
                )
            self._model_symbol = resolved_model_symbol
            if (
                self._cross_symbol_bundle_key is None
                and resolved_model_symbol is not None
                and resolved_model_symbol != self.symbol
            ):
                self._log_symbol_guard(
                    "symbol_mismatch",
                    logging.ERROR,
                    "MODEL/SYMBOL MISMATCH: trading %s but the resolved model bundle is "
                    "for %s (model=%s). Signals are scored by a model trained on a "
                    "different symbol.",
                    self.symbol,
                    resolved_model_symbol,
                    engine_model_name,
                )

            pred = float(result.price)

            # Prediction engine returns real prices
            return pred

        except Exception:
            # No model scored this bar — clear the stamp so the HOLD signal
            # doesn't carry the previous bar's model_symbol.
            self._model_symbol = None
            logger.exception("MLBasicSignalGenerator: Prediction error at index %d", index)
            return None

    def _calculate_confidence(self, predicted_return: float) -> float:
        """
        Calculate confidence score based on predicted return magnitude

        Args:
            predicted_return: Predicted return value

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = min(1.0, abs(predicted_return) * self.confidence_multiplier)
        return max(0.0, confidence)

    @property
    def warmup_period(self) -> int:
        """Return the minimum history required before producing valid signals."""
        return self.sequence_length

    def get_parameters(self) -> dict[str, Any]:
        """Get signal generator parameters for logging and serialization"""
        params = super().get_parameters()
        params.update(
            {
                "sequence_length": self.sequence_length,
                "model_name": self.model_name,
                "model_type": self.model_type,
                "model_timeframe": self.model_timeframe,
                "symbol": self.symbol,
                "long_entry_threshold": self.long_entry_threshold,
                "short_entry_threshold": self.short_entry_threshold,
                "confidence_multiplier": self.confidence_multiplier,
            }
        )
        return params
