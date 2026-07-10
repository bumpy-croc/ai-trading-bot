"""Exam-only classification-native and percentile-rank signal generators.

Built for the TARGET-REDESIGN tournament exam harness (preregistration §5):
entrants (a)/(b)/(c) (classifiers) plug in via ``ClassificationExamSignalGenerator``,
entrant (d) (smoothed forward return) via ``SmoothedReturnExamSignalGenerator``.
Both consume ``PredictionResult`` from the classification-native prediction
path (``src/prediction/engine.py``/``onnx_runner.py``) and both convert raw
model output to signal confidence via the harness-wide rule: statistics of
the model's OWN training-set target distribution, never a hardcoded
constant. The ``confidence = |return| * 12`` formula
(``ml_signal_generator.py``'s ``CONFIDENCE_MULTIPLIER``) is prohibited here.

Deliberately a SEPARATE module from ``ml_signal_generator.py``
(``MLBasicSignalGenerator``/``MLSignalGenerator``) -- those are live,
money-path code with their own cross-symbol-fallback hardening this
exam-only research harness does not need, and this scaffolding does not
modify them.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from src.prediction import PredictionConfig, PredictionEngine, PredictionResult
from src.prediction.distribution_stats import FrozenDistribution, percentile_rank_confidence
from src.prediction.features.pipeline import FeaturePipeline
from src.prediction.features.price_only import PriceOnlyFeatureExtractor
from src.strategies.components.regime_context import RegimeContext
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator

logger = logging.getLogger(__name__)

_PRICE_ONLY_PIPELINE_CONFIG = {
    "technical_features": {"enabled": False},
    "sentiment_features": {"enabled": False},
    "market_features": {"enabled": False},
    "price_only_features": {"enabled": False},
}


def _build_price_only_prediction_engine(sequence_length: int) -> PredictionEngine | None:
    """Build a price-only PredictionEngine for the exam harness.

    Mirrors MLBasicSignalGenerator's feature-pipeline setup (price-only
    extractor, no sentiment/technical/market features) without its
    cross-symbol-substitution guard machinery -- that hardening exists for
    live trading and is not needed for this exam-only research harness.
    """
    try:
        config = PredictionConfig.from_config_manager()
        config.enable_sentiment = False
        config.enable_market_microstructure = False
        engine = PredictionEngine(config)
        engine.feature_pipeline = FeaturePipeline(
            config=_PRICE_ONLY_PIPELINE_CONFIG,
            custom_extractors=[PriceOnlyFeatureExtractor(normalization_window=sequence_length)],
        )
        health = engine.health_check()
        if health.get("status") != "healthy":
            logger.warning("Exam signal generator: prediction engine health degraded: %s", health)
        return engine
    except Exception:
        logger.exception("Exam signal generator: prediction engine initialization failed")
        return None


class ClassificationExamSignalGenerator(SignalGenerator):
    """Consumes a classifier bundle's probability output directly.

    ``direction`` is already mapped to {-1, 0, 1} by the bundle's
    ``class_labels`` metadata (see ``OnnxRunner._process_classification_output``);
    ``confidence`` is the raw P(argmax class) -- bounded [0, 1] by
    construction, no percentile-rank/multiplier transform needed for the
    *value* itself (still subject to a calibration-correction step fit at
    training time, per the preregistration §2 -- this signal generator
    consumes whatever probability the bundle reports).
    """

    def __init__(
        self,
        name: str = "classification_exam_signal_generator",
        sequence_length: int = 120,
        model_name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.sequence_length = sequence_length
        self.model_name = model_name
        self.prediction_engine = _build_price_only_prediction_engine(sequence_length)

    def generate_signal(self, df: Any, index: int, regime: RegimeContext | None = None) -> Signal:
        self.validate_inputs(df, index)

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

        result = self._predict(df, index)
        if result is None or result.error is not None or result.probabilities is None:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    "generator": self.name,
                    "reason": "prediction_failed_or_not_classification",
                    "index": index,
                },
            )

        confidence = max(0.0, min(1.0, float(result.confidence)))
        raw_direction = int(result.direction)
        if raw_direction > 0:
            direction = SignalDirection.BUY
        elif raw_direction < 0:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD

        metadata: dict[str, Any] = {
            "generator": self.name,
            "probabilities": result.probabilities,
            "raw_direction": raw_direction,
            "index": index,
            "sequence_length": self.sequence_length,
            "engine_model_name": self.model_name,
        }
        if direction == SignalDirection.SELL:
            metadata["enter_short"] = True

        # No separate "magnitude" for a classifier output -- confidence
        # doubles as strength (same convention as the incumbent
        # regression signal generators, which scale strength from the same
        # signal that drives confidence).
        return Signal(
            direction=direction, strength=confidence, confidence=confidence, metadata=metadata
        )

    def get_confidence(self, df: Any, index: int) -> float:
        self.validate_inputs(df, index)
        if index < self.sequence_length:
            return 0.0
        result = self._predict(df, index)
        if result is None or result.error is not None or result.probabilities is None:
            return 0.0
        return max(0.0, min(1.0, float(result.confidence)))

    def _predict(self, df: Any, index: int) -> PredictionResult | None:
        if self.prediction_engine is None:
            return None
        window_df = df[["open", "high", "low", "close", "volume"]].iloc[
            index - self.sequence_length : index
        ]
        try:
            if self.model_name:
                return self.prediction_engine.predict(window_df, model_name=self.model_name)
            return self.prediction_engine.predict(window_df)
        except Exception:
            logger.exception(
                "ClassificationExamSignalGenerator: prediction error at index %d", index
            )
            return None

    @property
    def warmup_period(self) -> int:
        return self.sequence_length

    def get_parameters(self) -> dict[str, Any]:
        params = super().get_parameters()
        params.update({"sequence_length": self.sequence_length, "model_name": self.model_name})
        return params


class SmoothedReturnExamSignalGenerator(SignalGenerator):
    """Entrant (d): regression output, confidence via percentile-rank.

    Confidence is ``percentile_rank_confidence(|predicted_return|,
    distribution)`` against a FROZEN training-set distribution loaded from
    the resolved bundle's metadata (``metadata["target_distribution"]``,
    a ``FrozenDistribution.to_metadata()`` payload persisted at training
    time) -- the harness-wide rule's clearest application, and the direct
    fix for the prohibited ``× 12`` formula. A bundle with no
    ``target_distribution`` metadata degrades to HOLD rather than falling
    back to any hardcoded formula.
    """

    def __init__(
        self,
        name: str = "smoothed_return_exam_signal_generator",
        sequence_length: int = 120,
        model_name: str | None = None,
        long_entry_threshold: float = 0.0,
        short_entry_threshold: float = 0.0,
    ) -> None:
        super().__init__(name)
        self.sequence_length = sequence_length
        self.model_name = model_name
        self.long_entry_threshold = long_entry_threshold
        self.short_entry_threshold = short_entry_threshold
        self.prediction_engine = _build_price_only_prediction_engine(sequence_length)

    def generate_signal(self, df: Any, index: int, regime: RegimeContext | None = None) -> Signal:
        self.validate_inputs(df, index)

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

        result = self._predict(df, index)
        if result is None or result.error is not None:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={"generator": self.name, "reason": "prediction_failed", "index": index},
            )

        current_price = df["close"].iloc[index]
        # result.price for a smoothed_return-target model IS the predicted
        # return already (the model was trained on
        # smoothed_forward_return_labels, a return-scale target -- see
        # labels.py). It is NOT a price level: smoothed_return metadata
        # carries no price_normalization, so OnnxRunner never denormalizes
        # it, and re-deriving (prediction-current_price)/current_price here
        # would treat a ~0.002-scale return as if it were a ~60000-scale
        # price, saturating to ~-1.0 (all-SELL, confidence clamped to 1.0)
        # on virtually every bar. current_price is still validated as a
        # basic data-sanity check (a non-finite/non-positive close means
        # something upstream is broken), just no longer used arithmetically.
        predicted_return = result.price
        if not (math.isfinite(predicted_return) and math.isfinite(current_price) and current_price > 0):
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    "generator": self.name,
                    "reason": "invalid_prediction_or_price",
                    "prediction": predicted_return,
                    "current_price": current_price,
                    "index": index,
                },
            )

        distribution = self._distribution_for(result)
        if distribution is None:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    "generator": self.name,
                    "reason": "missing_target_distribution",
                    "predicted_return": predicted_return,
                    "index": index,
                },
            )

        confidence = percentile_rank_confidence(abs(predicted_return), distribution)

        if predicted_return > self.long_entry_threshold:
            direction = SignalDirection.BUY
        elif predicted_return < -self.short_entry_threshold:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD

        metadata: dict[str, Any] = {
            "generator": self.name,
            "prediction": predicted_return,
            "current_price": current_price,
            "predicted_return": predicted_return,
            "index": index,
            "sequence_length": self.sequence_length,
            "engine_model_name": self.model_name,
        }
        if direction == SignalDirection.SELL:
            metadata["enter_short"] = True

        return Signal(
            direction=direction, strength=confidence, confidence=confidence, metadata=metadata
        )

    def get_confidence(self, df: Any, index: int) -> float:
        self.validate_inputs(df, index)
        if index < self.sequence_length:
            return 0.0
        result = self._predict(df, index)
        if result is None or result.error is not None:
            return 0.0
        current_price = df["close"].iloc[index]
        # See generate_signal: result.price IS the predicted return already.
        predicted_return = result.price
        if not (math.isfinite(predicted_return) and math.isfinite(current_price) and current_price > 0):
            return 0.0
        distribution = self._distribution_for(result)
        if distribution is None:
            return 0.0
        return percentile_rank_confidence(abs(predicted_return), distribution)

    def _predict(self, df: Any, index: int) -> PredictionResult | None:
        if self.prediction_engine is None:
            return None
        window_df = df[["open", "high", "low", "close", "volume"]].iloc[
            index - self.sequence_length : index
        ]
        try:
            if self.model_name:
                return self.prediction_engine.predict(window_df, model_name=self.model_name)
            return self.prediction_engine.predict(window_df)
        except Exception:
            logger.exception(
                "SmoothedReturnExamSignalGenerator: prediction error at index %d", index
            )
            return None

    def _distribution_for(self, result: PredictionResult) -> FrozenDistribution | None:
        if self.prediction_engine is None:
            return None
        info = self.prediction_engine.get_model_info(result.model_name)
        bundle_metadata = info.get("metadata") or {}
        raw = bundle_metadata.get("target_distribution")
        if not raw:
            return None
        try:
            return FrozenDistribution.from_metadata(raw)
        except (KeyError, ValueError, TypeError):
            logger.warning(
                "SmoothedReturnExamSignalGenerator: invalid target_distribution metadata for %s",
                result.model_name,
            )
            return None

    @property
    def warmup_period(self) -> int:
        return self.sequence_length

    def get_parameters(self) -> dict[str, Any]:
        params = super().get_parameters()
        params.update(
            {
                "sequence_length": self.sequence_length,
                "model_name": self.model_name,
                "long_entry_threshold": self.long_entry_threshold,
                "short_entry_threshold": self.short_entry_threshold,
            }
        )
        return params


__all__ = ["ClassificationExamSignalGenerator", "SmoothedReturnExamSignalGenerator"]
