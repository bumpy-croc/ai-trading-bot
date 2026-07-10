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
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config.constants import (
    DEFAULT_MAX_HOLDING_HOURS,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
)
from src.engines.shared.barrier_touch import check_barrier_touch
from src.engines.shared.cost_calculator import CostCalculator
from src.ml.training_pipeline.meta_labels import (
    HIT_RATE_LOOKBACK_FIRES,
    REALIZED_VOL_WINDOW_BARS,
    _session_cyclical_encoding,
    encode_meta_label_feature_row,
)
from src.performance.metrics import Side, pnl_percent
from src.prediction import PredictionConfig, PredictionEngine, PredictionResult
from src.prediction.distribution_stats import FrozenDistribution, percentile_rank_confidence
from src.prediction.features.pipeline import FeaturePipeline
from src.prediction.features.price_only import PriceOnlyFeatureExtractor
from src.prediction.models.registry import PredictionModelRegistry
from src.regime.enhanced_detector import EnhancedRegimeDetector
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
        if not (
            math.isfinite(predicted_return) and math.isfinite(current_price) and current_price > 0
        ):
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
        if not (
            math.isfinite(predicted_return) and math.isfinite(current_price) and current_price > 0
        ):
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
        # Prefer this generator's OWN model_name (the registry key a caller
        # pinned, e.g. exam_target_redesign.py's ATB_MODEL_VERSION_OVERRIDE
        # support) over result.model_name -- PredictionResult.model_name is
        # the ONNX file's basename (OnnxRunner.predict() sets it from
        # os.path.basename(self.model_path), e.g. "model.onnx"), which never
        # matches a registry bundle.key and made target_distribution
        # permanently unreachable whenever a specific version was pinned.
        # Falls back to result.model_name when unpinned (self.model_name is
        # None), matching prior behavior for the registry-default-bundle case.
        lookup_name = self.model_name or result.model_name
        info = self.prediction_engine.get_model_info(lookup_name)
        bundle_metadata = info.get("metadata") or {}
        raw = bundle_metadata.get("target_distribution")
        if not raw:
            return None
        try:
            return FrozenDistribution.from_metadata(raw)
        except (KeyError, ValueError, TypeError):
            logger.warning(
                "SmoothedReturnExamSignalGenerator: invalid target_distribution metadata for %s",
                lookup_name,
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


@dataclass
class _OpenFire:
    """A primary signal fire whose trade outcome hasn't resolved yet."""

    fire_index: int
    direction: int  # 1 (long) or -1 (short)
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    vertical_exit_index: int


class MetaLabelExamSignalGenerator(SignalGenerator):
    """Entrant (a): gates a wrapped primary SignalGenerator by P(profitable).

    Unlike ``ClassificationExamSignalGenerator``, this classifier's output
    is NOT a direction prediction -- it's P(the primary signal's fired trade
    is profitable). The primary signal supplies direction; this generator
    only decides whether to act on it.

    STATEFUL and causal BY CONSTRUCTION, which is the entire point:
    ``resolve_fired_trade``/``build_meta_label_features``
    (``src/ml/training_pipeline/meta_labels.py``) are TRAINING-TIME-ONLY --
    they read forward bars deliberately, which is fine for labeling a
    COMPLETE historical corpus but would be a genuine lookahead bug if
    called from inside ``generate_signal()`` for the CURRENT bar (it would
    read bars strictly after "now" in a live/backtest loop). This class
    instead resolves open fires bar-by-bar via ``check_barrier_touch`` (the
    same intrabar high/low logic ``ExitHandler``/``triple_barrier_labels``
    use) as later bars naturally arrive, reconstructing the training-time
    ``rolling_hit_rate_20`` feature online. ``generate_signal`` MUST be
    called in strictly increasing ``index`` order -- the same contract every
    backtest/live engine in this codebase already guarantees.
    """

    def __init__(
        self,
        primary_signal_generator: SignalGenerator,
        name: str = "meta_label_exam_signal_generator",
        model_name: str | None = None,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        min_confidence: float = 0.5,
        take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
        max_holding_bars: int = DEFAULT_MAX_HOLDING_HOURS,
        resolved_history_maxlen: int = 500,
    ) -> None:
        super().__init__(name)
        self.primary_signal_generator = primary_signal_generator
        self.model_name = model_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.min_confidence = min_confidence
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_bars = max_holding_bars
        self._open_fires: list[_OpenFire] = []
        self._resolved_history: deque[bool] = deque(maxlen=resolved_history_maxlen)
        self._regime_detector = EnhancedRegimeDetector()
        self._cost_calculator = CostCalculator()
        self._bundle = self._load_bundle()
        # generate_signal has side effects (fire registration/resolution);
        # cache the last (index, Signal) pair so a same-index re-call (e.g.
        # get_confidence(df, index) after generate_signal(df, index)) never
        # double-registers or double-resolves.
        self._last_signal_index: int | None = None
        self._last_signal: Signal | None = None

    def _load_bundle(self) -> Any:
        """Resolve the meta_label registry bundle once at construction.

        Bypasses PredictionEngine/FeaturePipeline entirely -- meta-label
        features are pre-computed tabular values (encode_meta_label_feature_row),
        not something a generic OHLCV feature pipeline should re-derive.
        Calls the ONNX runner directly, same as PredictionEngine does
        internally after its own bundle resolution.
        """
        try:
            config = PredictionConfig.from_config_manager()
            registry = PredictionModelRegistry(config)
            if self.model_name:
                bundle = registry.get_bundle_by_key(self.model_name)
                if bundle is not None:
                    return bundle
                logger.warning(
                    "MetaLabelExamSignalGenerator: pinned model_name=%r not found, "
                    "falling back to latest meta_label bundle",
                    self.model_name,
                )
            return registry.select_bundle(
                symbol=self.symbol, model_type="meta_label", timeframe=self.timeframe
            )
        except Exception:
            logger.exception("MetaLabelExamSignalGenerator: failed to load meta_label bundle")
            return None

    def _resolve_open_fires(self, df: Any, index: int) -> None:
        """Resolve any open fires against bar `index`'s OHLC, causally.

        Only ever looks at df.iloc[index] (the bar that just "arrived") --
        never forward. Fires that resolve are appended to
        _resolved_history; fires that don't stay open for a later call.
        """
        high = float(df["high"].iloc[index])
        low = float(df["low"].iloc[index])
        still_open: list[_OpenFire] = []
        for fire in self._open_fires:
            side = "long" if fire.direction == 1 else "short"
            touch = check_barrier_touch(
                side=side,
                candle_high=high,
                candle_low=low,
                stop_loss_price=fire.stop_loss_price,
                take_profit_price=fire.take_profit_price,
            )
            exit_price: float | None = None
            if touch.hit_stop_loss:
                exit_price = touch.stop_loss_exit_price
            elif touch.hit_take_profit:
                exit_price = touch.take_profit_exit_price
            elif index >= fire.vertical_exit_index:
                exit_price = float(df["close"].iloc[index])

            if exit_price is None:
                still_open.append(fire)
                continue

            side_enum = Side.LONG if fire.direction == 1 else Side.SHORT
            raw_pct_return = pnl_percent(fire.entry_price, exit_price, side_enum, fraction=1.0)
            round_trip_cost_pct = 2.0 * self._cost_calculator.fee_rate
            profitable = bool((raw_pct_return - round_trip_cost_pct) > 0.0)
            self._resolved_history.append(profitable)

        self._open_fires = still_open

    def _register_fire(self, df: Any, index: int, direction: int) -> None:
        entry_price = float(df["close"].iloc[index])
        if direction == 1:
            stop_loss_price = entry_price * (1.0 - self.stop_loss_pct)
            take_profit_price = entry_price * (1.0 + self.take_profit_pct)
        else:
            stop_loss_price = entry_price * (1.0 + self.stop_loss_pct)
            take_profit_price = entry_price * (1.0 - self.take_profit_pct)
        self._open_fires.append(
            _OpenFire(
                fire_index=index,
                direction=direction,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                vertical_exit_index=index + self.max_holding_bars,
            )
        )

    def _rolling_hit_rate(self) -> float:
        if not self._resolved_history:
            return float("nan")
        recent = list(self._resolved_history)[-HIT_RATE_LOOKBACK_FIRES:]
        return float(np.mean(recent))

    def _realized_volatility(self, df: Any, index: int) -> float:
        """Causal 48-bar trailing realized volatility, ending at `index`."""
        window_start = max(0, index - REALIZED_VOL_WINDOW_BARS)
        close_window = df["close"].iloc[window_start : index + 1]
        returns = close_window.pct_change()
        return float(returns.std())

    def generate_signal(self, df: Any, index: int, regime: RegimeContext | None = None) -> Signal:
        """Caching wrapper: generate_signal has side effects (fire
        registration/resolution), so a same-index re-call (e.g. a caller's
        own get_confidence(df, index) after generate_signal(df, index))
        must return the CACHED result rather than double-registering the
        fire or double-resolving open fires against the same bar."""
        if index == self._last_signal_index and self._last_signal is not None:
            return self._last_signal
        signal = self._generate_signal_impl(df, index, regime)
        self._last_signal_index = index
        self._last_signal = signal
        return signal

    def _generate_signal_impl(
        self, df: Any, index: int, regime: RegimeContext | None = None
    ) -> Signal:
        self.validate_inputs(df, index)
        self._resolve_open_fires(df, index)

        primary_signal = self.primary_signal_generator.generate_signal(df, index, regime)
        if primary_signal.direction == SignalDirection.HOLD:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={"generator": self.name, "reason": "primary_hold", "index": index},
            )

        direction = 1 if primary_signal.direction == SignalDirection.BUY else -1
        # Register the fire unconditionally (regardless of the meta-gate's
        # eventual verdict) -- rolling_hit_rate_20 tracks the PRIMARY
        # signal's own track record, matching training-time semantics
        # (run_primary_signal_forward records every fire, not just ones a
        # meta-classifier would later approve).
        self._register_fire(df, index, direction)

        if self._bundle is None:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={"generator": self.name, "reason": "no_meta_label_bundle", "index": index},
            )

        timestamp = df.index[index]
        session_sin, session_cos = _session_cyclical_encoding(pd.Timestamp(timestamp))
        regime_ctx = self._regime_detector.detect_regime(df, index)
        predicted_return_magnitude = abs(
            float(primary_signal.metadata.get("predicted_return", 0.0))
        )

        feature_row = encode_meta_label_feature_row(
            realized_vol_48=self._realized_volatility(df, index),
            rolling_hit_rate_20=self._rolling_hit_rate(),
            session_sin=session_sin,
            session_cos=session_cos,
            regime_trend=regime_ctx.trend.value,
            regime_volatility=regime_ctx.volatility.value,
            regime_confidence=float(regime_ctx.confidence),
            predicted_return_magnitude=predicted_return_magnitude,
        )

        try:
            prediction = self._bundle.runner.predict(feature_row)
        except Exception:
            logger.exception("MetaLabelExamSignalGenerator: prediction error at index %d", index)
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={"generator": self.name, "reason": "prediction_failed", "index": index},
            )

        probabilities = getattr(prediction, "probabilities", None)
        if probabilities is None or len(probabilities) != 2:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    "generator": self.name,
                    "reason": "invalid_meta_label_output",
                    "index": index,
                },
            )

        # class_labels=[0, 1] ("not profitable", "profitable") -- probabilities[1]
        # IS P(profitable) directly, regardless of which class argmax picked
        # (see pipeline.py::_run_meta_label_pipeline's metadata comment and
        # task_types.py's TARGET_CLASS_LABELS docstring: meta_label's output
        # is a profitability GATE, not a direction-value classifier).
        p_profitable = float(probabilities[1])

        if p_profitable < self.min_confidence:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=p_profitable,
                metadata={
                    "generator": self.name,
                    "reason": "meta_label_gate",
                    "p_profitable": p_profitable,
                    "primary_direction": direction,
                    "index": index,
                },
            )

        metadata: dict[str, Any] = {
            "generator": self.name,
            "p_profitable": p_profitable,
            "primary_direction": direction,
            "predicted_return": primary_signal.metadata.get("predicted_return", 0.0),
            "index": index,
        }
        if primary_signal.direction == SignalDirection.SELL:
            metadata["enter_short"] = True

        return Signal(
            direction=primary_signal.direction,
            strength=p_profitable,
            confidence=p_profitable,
            metadata=metadata,
        )

    def get_confidence(self, df: Any, index: int) -> float:
        """Passive accessor -- never re-runs generate_signal's side effects.

        Unlike ClassificationExamSignalGenerator/SmoothedReturnExamSignalGenerator
        (stateless, safe to recompute), this generator is STATEFUL: calling
        generate_signal for a bar it hasn't seen yet from get_confidence
        would register/resolve fires out of the caller's control. Returns
        the cached confidence from the most recent generate_signal(df, index)
        call for this exact index, or 0.0 if that bar hasn't been processed
        via generate_signal yet.
        """
        if index == self._last_signal_index and self._last_signal is not None:
            return self._last_signal.confidence
        return 0.0

    @property
    def warmup_period(self) -> int:
        return max(self.primary_signal_generator.warmup_period, REALIZED_VOL_WINDOW_BARS)

    def get_parameters(self) -> dict[str, Any]:
        params = super().get_parameters()
        params.update(
            {
                "model_name": self.model_name,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "min_confidence": self.min_confidence,
                "primary_signal_generator": type(self.primary_signal_generator).__name__,
            }
        )
        return params


__all__ = [
    "ClassificationExamSignalGenerator",
    "MetaLabelExamSignalGenerator",
    "SmoothedReturnExamSignalGenerator",
]
