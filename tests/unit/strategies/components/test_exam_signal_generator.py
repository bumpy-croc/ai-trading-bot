"""Unit tests for the exam-only classification-native signal generators.

Built for the TARGET-REDESIGN tournament exam harness (§5): entrants
(a)/(b)/(c) (classifiers) via ClassificationExamSignalGenerator, entrant (d)
(smoothed forward return) via SmoothedReturnExamSignalGenerator. Deliberately
separate from MLBasicSignalGenerator/MLSignalGenerator (money-path, live) --
these tests never import or touch that module.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.engines.shared.barrier_touch import BarrierTouchResult
from src.ml.training_pipeline.meta_labels import (
    HIT_RATE_LOOKBACK_FIRES,
    PrimarySignalRecord,
    TradeResolution,
    build_meta_label_features,
)
from src.prediction import PredictionResult
from src.prediction.distribution_stats import FrozenDistribution
from src.prediction.models.onnx_runner import ModelPrediction
from src.strategies.components.exam_signal_generator import (
    ClassificationExamSignalGenerator,
    MetaLabelExamSignalGenerator,
    SmoothedReturnExamSignalGenerator,
    TargetDistributionUnavailableError,
)
from src.strategies.components.signal_generator import (
    Signal,
    SignalDirection,
    SignalGenerator,
)


def _make_df(length=150):
    dates = pd.date_range("2023-01-01", periods=length, freq="1h")
    rng = np.random.default_rng(42)
    base_price = 50000.0
    prices = [base_price]
    for change in rng.normal(0, 0.02, length - 1):
        prices.append(max(prices[-1] * (1 + change), 1000))
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": rng.uniform(1000, 10000, length),
        },
        index=dates,
    )


@patch("src.strategies.components.exam_signal_generator.PredictionEngine")
@patch("src.strategies.components.exam_signal_generator.PredictionConfig")
class TestClassificationExamSignalGenerator:
    def test_initialization(self, mock_config_class, mock_engine_class):
        mock_engine = MagicMock()
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = ClassificationExamSignalGenerator(sequence_length=120)

        assert generator.sequence_length == 120
        assert generator.warmup_period == 120

    def test_insufficient_history_holds(self, mock_config_class, mock_engine_class):
        mock_engine = MagicMock()
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = ClassificationExamSignalGenerator(sequence_length=120)
        df = _make_df(150)

        signal = generator.generate_signal(df, 50)

        assert signal.direction == SignalDirection.HOLD
        assert signal.confidence == 0.0

    def test_up_class_wins_produces_buy(self, mock_config_class, mock_engine_class):
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        mock_result.error = None
        mock_result.direction = 1
        mock_result.confidence = 0.82
        mock_result.probabilities = (0.18, 0.82)
        mock_engine.predict.return_value = mock_result
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = ClassificationExamSignalGenerator(sequence_length=120)
        df = _make_df(150)

        signal = generator.generate_signal(df, 130)

        assert signal.direction == SignalDirection.BUY
        assert signal.confidence == pytest.approx(0.82)
        assert signal.metadata["probabilities"] == (0.18, 0.82)

    def test_down_class_wins_produces_sell_with_enter_short_flag(
        self, mock_config_class, mock_engine_class
    ):
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        mock_result.error = None
        mock_result.direction = -1
        mock_result.confidence = 0.7
        mock_result.probabilities = (0.7, 0.3)
        mock_engine.predict.return_value = mock_result
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = ClassificationExamSignalGenerator(sequence_length=120)
        df = _make_df(150)

        signal = generator.generate_signal(df, 130)

        assert signal.direction == SignalDirection.SELL
        assert signal.metadata["enter_short"] is True

    def test_hold_class_wins(self, mock_config_class, mock_engine_class):
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        mock_result.error = None
        mock_result.direction = 0
        mock_result.confidence = 0.6
        mock_result.probabilities = (0.2, 0.6, 0.2)
        mock_engine.predict.return_value = mock_result
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = ClassificationExamSignalGenerator(sequence_length=120)
        df = _make_df(150)

        signal = generator.generate_signal(df, 130)

        assert signal.direction == SignalDirection.HOLD

    def test_regression_bundle_no_probabilities_holds(self, mock_config_class, mock_engine_class):
        """A regression bundle (probabilities=None) resolved by mistake must
        degrade to HOLD, not crash or fabricate a direction from price."""
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        mock_result.error = None
        mock_result.probabilities = None
        mock_engine.predict.return_value = mock_result
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = ClassificationExamSignalGenerator(sequence_length=120)
        df = _make_df(150)

        signal = generator.generate_signal(df, 130)

        assert signal.direction == SignalDirection.HOLD
        assert signal.confidence == 0.0

    def test_prediction_error_holds(self, mock_config_class, mock_engine_class):
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        mock_result.error = "boom"
        mock_result.metadata = {}
        mock_engine.predict.return_value = mock_result
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = ClassificationExamSignalGenerator(sequence_length=120)
        df = _make_df(150)

        signal = generator.generate_signal(df, 130)

        assert signal.direction == SignalDirection.HOLD

    def test_get_confidence(self, mock_config_class, mock_engine_class):
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        mock_result.error = None
        mock_result.direction = 1
        mock_result.confidence = 0.55
        mock_result.probabilities = (0.45, 0.55)
        mock_engine.predict.return_value = mock_result
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = ClassificationExamSignalGenerator(sequence_length=120)
        df = _make_df(150)

        assert generator.get_confidence(df, 130) == pytest.approx(0.55)
        assert generator.get_confidence(df, 50) == 0.0


@patch("src.strategies.components.exam_signal_generator.PredictionEngine")
@patch("src.strategies.components.exam_signal_generator.PredictionConfig")
class TestSmoothedReturnExamSignalGenerator:
    def _distribution_metadata(self):
        dist = FrozenDistribution(
            values=(0.0, 0.01, 0.02, 0.05, 0.10),
            percentiles=(0.0, 25.0, 50.0, 90.0, 100.0),
        )
        return dist.to_metadata()

    def test_initialization(self, mock_config_class, mock_engine_class):
        mock_engine = MagicMock()
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = SmoothedReturnExamSignalGenerator(sequence_length=120)
        assert generator.warmup_period == 120

    def test_confidence_uses_percentile_rank_not_multiplier(
        self, mock_config_class, mock_engine_class
    ):
        """The harness-wide fix for the prohibited ×12 formula: confidence
        must come from the frozen training-set distribution, never a fixed
        multiplier on |predicted_return|."""
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        mock_result.error = None
        mock_result.model_name = "BTCUSDT:1h:smoothed_return:v1"
        mock_engine.get_model_info.return_value = {
            "metadata": {"target_distribution": self._distribution_metadata()}
        }
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = SmoothedReturnExamSignalGenerator(sequence_length=120)
        df = _make_df(150)
        # result.price IS the predicted return directly (the model was
        # trained on smoothed_forward_return_labels, a return-scale target)
        # -- NOT a price level to re-derive a return from.
        mock_result.price = 0.03  # controlled +3% predicted return
        mock_engine.predict.return_value = mock_result

        signal = generator.generate_signal(df, 130)

        # Distribution grid (0%,25th=1%,50th=2%,90th=5%,100th=10%) -> a 3%
        # move interpolates to ~63.3% confidence. A naive `|x| * 12`
        # formula would instead give min(1, 0.03*12) = 0.36 -- assert the
        # percentile-rank result is NOT that.
        assert 0.0 <= signal.confidence <= 1.0
        naive_multiplier_confidence = min(1.0, 0.03 * 12.0)
        assert signal.confidence != pytest.approx(naive_multiplier_confidence)
        assert signal.confidence == pytest.approx(0.6333, abs=0.01)

    def test_units_bug_regression_small_return_does_not_saturate_all_sell(
        self, mock_config_class, mock_engine_class
    ):
        """#948 fix-round bug (flagged by claude[bot] PR review, missed in
        the first fix round): treating result.price as a PRICE and
        re-deriving (prediction-current_price)/current_price produced
        ~-0.9999 on virtually every bar (saturated all-SELL, confidence
        clamped to 1.0), since a return-scale value (~0.002) is ~4 orders
        of magnitude smaller than a real price (~60000). A typical small
        predicted return must produce a typical small predicted_return in
        the signal metadata, not a near -1.0 value."""
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        mock_result.error = None
        mock_result.model_name = "BTCUSDT:1h:smoothed_return:v1"
        mock_engine.get_model_info.return_value = {
            "metadata": {"target_distribution": self._distribution_metadata()}
        }
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = SmoothedReturnExamSignalGenerator(sequence_length=120)
        df = _make_df(150)
        mock_result.price = 0.002  # a typical small smoothed-return prediction
        mock_engine.predict.return_value = mock_result

        signal = generator.generate_signal(df, 130)

        assert signal.metadata["predicted_return"] == pytest.approx(0.002, abs=1e-9)
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence < 1.0

    def test_unresolvable_bundle_raises_instead_of_silent_hold(
        self, mock_config_class, mock_engine_class
    ):
        """PR #950 review item 4: when running UNPINNED (self.model_name is
        None), lookup falls back to result.model_name -- the ONNX file's
        basename (e.g. "model.onnx") -- which can never match a registry
        key, so get_model_info returns {} (nothing resolvable at all). This
        must fail loud (TargetDistributionUnavailableError), not silently
        HOLD every bar forever -- a full backtest finishing with zero
        trades and no error is indistinguishable from "the model
        legitimately never signals" (the #927 silent-failure pattern)."""
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        mock_result.error = None
        mock_result.price = 0.02
        mock_result.model_name = "model.onnx"  # unpinned: the ONNX basename
        mock_engine.predict.return_value = mock_result
        mock_engine.get_model_info.return_value = {}  # nothing resolvable at all
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = SmoothedReturnExamSignalGenerator(sequence_length=120)  # no model_name pinned
        df = _make_df(150)

        with pytest.raises(TargetDistributionUnavailableError):
            generator.generate_signal(df, 130)

    def test_missing_target_distribution_holds(self, mock_config_class, mock_engine_class):
        """Distinct from the unresolvable-bundle case above: a REAL,
        resolved bundle (get_model_info returns non-empty) that simply
        predates the target_distribution metadata field is the class's
        documented, intentional graceful-degrade case (no hardcoded-formula
        fallback) -- must still HOLD, not raise."""
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        mock_result.error = None
        mock_result.price = 0.02
        mock_result.model_name = "BTCUSDT:1h:smoothed_return:v1"
        mock_engine.predict.return_value = mock_result
        mock_engine.get_model_info.return_value = {"metadata": {}}
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = SmoothedReturnExamSignalGenerator(sequence_length=120)
        df = _make_df(150)

        signal = generator.generate_signal(df, 130)

        assert signal.direction == SignalDirection.HOLD
        assert signal.metadata["reason"] == "missing_target_distribution"

    def test_prediction_error_holds(self, mock_config_class, mock_engine_class):
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        mock_result.error = "boom"
        mock_result.metadata = {}
        mock_engine.predict.return_value = mock_result
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = SmoothedReturnExamSignalGenerator(sequence_length=120)
        df = _make_df(150)

        signal = generator.generate_signal(df, 130)

        assert signal.direction == SignalDirection.HOLD

    def test_distribution_lookup_uses_own_model_name_not_result_model_name(
        self, mock_config_class, mock_engine_class
    ):
        """Regression guard found via Phase 2b item 6's real end-to-end
        acceptance run: PredictionResult.model_name is the ONNX file's
        basename (OnnxRunner.predict() sets it from
        os.path.basename(self.model_path), e.g. "model.onnx") -- it NEVER
        matches a registry bundle.key (format
        "{symbol}:{timeframe}:{model_type}:{version}"), so
        get_model_info(result.model_name) could never find target_distribution
        for a PINNED (ATB_MODEL_VERSION_OVERRIDE) bundle in real production
        code -- only in unit tests whose mocks don't distinguish the two
        strings. This generator's own self.model_name (set at construction,
        e.g. by exam_target_redesign.py's _exam_model_name) IS the correct
        registry key and must be preferred when set."""
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        mock_result.error = None
        # Deliberately NOT a valid registry key -- exactly what
        # OnnxRunner.predict() actually produces in real code.
        mock_result.model_name = "model.onnx"
        mock_engine.get_model_info.return_value = {
            "metadata": {"target_distribution": self._distribution_metadata()}
        }
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        pinned_key = "BTCUSDT:1h:price:2026-01-01_1h_v1"
        generator = SmoothedReturnExamSignalGenerator(sequence_length=120, model_name=pinned_key)
        df = _make_df(150)
        mock_result.price = 0.05
        mock_engine.predict.return_value = mock_result

        signal = generator.generate_signal(df, 130)

        mock_engine.get_model_info.assert_called_once_with(pinned_key)
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence > 0.0

    def test_insufficient_history_holds(self, mock_config_class, mock_engine_class):
        mock_engine = MagicMock()
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = SmoothedReturnExamSignalGenerator(sequence_length=120)
        df = _make_df(150)

        signal = generator.generate_signal(df, 50)

        assert signal.direction == SignalDirection.HOLD
        assert signal.confidence == 0.0

    def test_up_prediction_produces_buy_signal(self, mock_config_class, mock_engine_class):
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        mock_result.error = None
        mock_result.model_name = "BTCUSDT:1h:smoothed_return:v1"
        mock_engine.get_model_info.return_value = {
            "metadata": {"target_distribution": self._distribution_metadata()}
        }
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = SmoothedReturnExamSignalGenerator(sequence_length=120)
        df = _make_df(150)
        mock_result.price = 0.05  # +5% predicted return, consumed directly
        mock_engine.predict.return_value = mock_result

        signal = generator.generate_signal(df, 130)

        assert signal.direction == SignalDirection.BUY
        assert signal.metadata["predicted_return"] == pytest.approx(0.05, abs=1e-6)

    def test_down_prediction_produces_sell_signal_with_enter_short(
        self, mock_config_class, mock_engine_class
    ):
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        mock_result.error = None
        mock_result.model_name = "BTCUSDT:1h:smoothed_return:v1"
        mock_engine.get_model_info.return_value = {
            "metadata": {"target_distribution": self._distribution_metadata()}
        }
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = SmoothedReturnExamSignalGenerator(sequence_length=120)
        df = _make_df(150)
        mock_result.price = -0.05  # -5% predicted return, consumed directly
        mock_engine.predict.return_value = mock_result

        signal = generator.generate_signal(df, 130)

        assert signal.direction == SignalDirection.SELL
        assert signal.metadata["enter_short"] is True


class _CannedPrimarySignalGenerator(SignalGenerator):
    """Fires exactly at the configured indices with a fixed direction/
    predicted_return; HOLD everywhere else."""

    def __init__(self, fire_at: dict[int, tuple[SignalDirection, float]], warmup: int = 5):
        super().__init__("canned_primary")
        self._fire_at = fire_at
        self._warmup = warmup

    def generate_signal(self, df, index, regime=None) -> Signal:
        if index in self._fire_at:
            direction, predicted_return = self._fire_at[index]
            return Signal(
                direction=direction,
                strength=0.5,
                confidence=0.5,
                metadata={"predicted_return": predicted_return},
            )
        return Signal(direction=SignalDirection.HOLD, strength=0.0, confidence=0.0, metadata={})

    def get_confidence(self, df, index) -> float:
        return 0.5

    @property
    def warmup_period(self) -> int:
        return self._warmup


def _make_meta_label_prediction(p_profitable: float) -> ModelPrediction:
    """A ModelPrediction shaped like OnnxRunner's binary_classification
    output for meta_label's class_labels=[0,1] convention: probabilities is
    (P(not profitable), P(profitable))."""
    return ModelPrediction(
        price=float("nan"),
        confidence=max(p_profitable, 1.0 - p_profitable),
        direction=1 if p_profitable >= 0.5 else 0,
        model_name="BTCUSDT:1h:meta_label:v1",
        inference_time=0.001,
        probabilities=(1.0 - p_profitable, p_profitable),
    )


@patch("src.strategies.components.exam_signal_generator.PredictionModelRegistry")
@patch("src.strategies.components.exam_signal_generator.PredictionConfig")
class TestMetaLabelExamSignalGenerator:
    """entrant (a): gates a wrapped primary SignalGenerator by P(profitable).
    STATEFUL and causal -- resolves fires bar-by-bar via check_barrier_touch,
    never by scanning forward (that would leak future bars into the CURRENT
    bar's signal, the exact bug class resolve_fired_trade/
    build_meta_label_features are training-time-only to avoid)."""

    def _mock_bundle(self, mock_registry_class, predictions: list[float]):
        """predictions: queue of P(profitable) values, consumed in order,
        one per generate_signal call that reaches meta-label prediction."""
        mock_bundle = MagicMock()
        mock_bundle.runner.predict.side_effect = [
            _make_meta_label_prediction(p) for p in predictions
        ]
        mock_registry = MagicMock()
        mock_registry.select_bundle.return_value = mock_bundle
        mock_registry_class.return_value = mock_registry
        return mock_bundle

    def test_hold_when_primary_holds(self, mock_config_class, mock_registry_class):
        self._mock_bundle(mock_registry_class, [])
        primary = _CannedPrimarySignalGenerator(fire_at={})
        generator = MetaLabelExamSignalGenerator(primary_signal_generator=primary)
        df = _make_df(150)

        signal = generator.generate_signal(df, 100)

        assert signal.direction == SignalDirection.HOLD

    def test_passes_through_primary_direction_when_confident(
        self, mock_config_class, mock_registry_class
    ):
        self._mock_bundle(mock_registry_class, [0.9])
        primary = _CannedPrimarySignalGenerator(fire_at={100: (SignalDirection.BUY, 0.02)})
        generator = MetaLabelExamSignalGenerator(
            primary_signal_generator=primary, min_confidence=0.5
        )
        df = _make_df(150)

        signal = generator.generate_signal(df, 100)

        assert signal.direction == SignalDirection.BUY
        assert signal.confidence == pytest.approx(0.9)

    def test_gates_to_hold_when_not_confident(self, mock_config_class, mock_registry_class):
        self._mock_bundle(mock_registry_class, [0.2])
        primary = _CannedPrimarySignalGenerator(fire_at={100: (SignalDirection.BUY, 0.02)})
        generator = MetaLabelExamSignalGenerator(
            primary_signal_generator=primary, min_confidence=0.5
        )
        df = _make_df(150)

        signal = generator.generate_signal(df, 100)

        assert signal.direction == SignalDirection.HOLD
        assert signal.metadata["reason"] == "meta_label_gate"

    def test_mutating_a_future_bar_does_not_change_the_current_signal(
        self, mock_config_class, mock_registry_class
    ):
        """The core leak-safety property: generate_signal(df, index) must be
        identical whether or not bars strictly after `index` are mutated."""
        self._mock_bundle(mock_registry_class, [0.9, 0.9])
        primary = _CannedPrimarySignalGenerator(fire_at={100: (SignalDirection.BUY, 0.02)})

        df_orig = _make_df(150)
        df_mutated = df_orig.copy()
        df_mutated.loc[df_mutated.index[105:], ["open", "high", "low", "close"]] *= 5.0

        gen_a = MetaLabelExamSignalGenerator(primary_signal_generator=primary, min_confidence=0.5)
        signal_orig = gen_a.generate_signal(df_orig, 100)

        gen_b = MetaLabelExamSignalGenerator(
            primary_signal_generator=_CannedPrimarySignalGenerator(
                fire_at={100: (SignalDirection.BUY, 0.02)}
            ),
            min_confidence=0.5,
        )
        signal_mutated = gen_b.generate_signal(df_mutated, 100)

        assert signal_orig.direction == signal_mutated.direction
        assert signal_orig.confidence == pytest.approx(signal_mutated.confidence)

    def test_no_bundle_available_holds(self, mock_config_class, mock_registry_class):
        mock_registry = MagicMock()
        mock_registry.select_bundle.side_effect = RuntimeError("no meta_label bundle")
        mock_registry_class.return_value = mock_registry
        primary = _CannedPrimarySignalGenerator(fire_at={100: (SignalDirection.BUY, 0.02)})

        generator = MetaLabelExamSignalGenerator(primary_signal_generator=primary)
        df = _make_df(150)

        signal = generator.generate_signal(df, 100)

        assert signal.direction == SignalDirection.HOLD
        assert signal.metadata["reason"] == "no_meta_label_bundle"

    def test_open_fire_resolves_and_feeds_rolling_hit_rate(
        self, mock_config_class, mock_registry_class
    ):
        """A fire registered at bar 100 must resolve (via barrier touch or
        vertical exit) as later bars are processed, and the resolved
        outcome must feed rolling_hit_rate_20 for a LATER fire's feature
        row -- never influencing the fire's own decision retroactively."""
        self._mock_bundle(mock_registry_class, [0.9, 0.9])
        primary = _CannedPrimarySignalGenerator(
            fire_at={100: (SignalDirection.BUY, 0.02), 110: (SignalDirection.BUY, 0.01)}
        )
        generator = MetaLabelExamSignalGenerator(
            primary_signal_generator=primary, min_confidence=0.5
        )
        df = _make_df(150)
        # Force a take-profit touch shortly after the first fire so it
        # resolves well before the second fire at index 110.
        df.loc[df.index[102], "high"] = df["close"].iloc[100] * 1.10

        generator.generate_signal(df, 100)
        for i in range(101, 110):
            generator.generate_signal(df, i)
        generator.generate_signal(df, 110)

        assert len(generator._resolved_history) == 1
        assert len(generator._open_fires) == 1  # the second fire, still open


@patch("src.strategies.components.exam_signal_generator.check_barrier_touch")
@patch("src.strategies.components.exam_signal_generator.PredictionModelRegistry")
@patch("src.strategies.components.exam_signal_generator.PredictionConfig")
class TestMetaLabelHitRateTrainServeParity:
    """PR #950 review finding: rolling_hit_rate_20 windowed in FIRE order at
    training time (build_meta_label_features) but RESOLUTION order at
    serve time (MetaLabelExamSignalGenerator) selected different trailing-20
    subsets once >20 fires had resolved, silently skewing entrant (a)'s
    features relative to what it was trained on. Both sites now delegate to
    the single shared ``resolution_ordered_hit_rate`` -- this test drives
    BOTH real code paths (training's build_meta_label_features and the
    serving generator's actual bar-by-bar state machine, real
    check_barrier_touch calls only stubbed at the OHLC-touch boundary) over
    an IDENTICAL, deliberately overlapping fire/resolution history (>20
    resolved fires, fire-order and resolution-order genuinely diverging)
    and asserts they compute the identical rolling_hit_rate_20 for the same
    final fire."""

    _N_FIRES = 30
    _FIRE_SPACING = 2
    _WARMUP = 60
    _MAX_HOLDING_BARS = 20
    _TAKE_PROFIT_PCT = 0.04
    _STOP_LOSS_PCT = 0.05

    @classmethod
    def _fire_index(cls, position: int) -> int:
        return cls._WARMUP + cls._FIRE_SPACING * position

    @classmethod
    def _is_fast(cls, position: int) -> bool:
        """Even positions resolve almost immediately (barrier touch, 1 bar
        after firing); odd positions resolve only after the full
        max_holding_bars vertical exit -- with fire spacing (2) far smaller
        than the slow holding period (20), many slow fires stay open
        concurrently while later fast fires fire AND resolve, producing the
        overlapping-trades divergence this test targets."""
        return position % 2 == 0

    @classmethod
    def _make_df(cls, periods: int = 250) -> pd.DataFrame:
        # Tiny strictly-increasing drift so every fire's entry price (and
        # therefore its derived stop-loss/take-profit price pair) is unique
        # -- lets the check_barrier_touch stub identify which fire a given
        # call belongs to by its (stop_loss_price, take_profit_price) pair,
        # without needing real, mutually-interfering OHLC price action for
        # 25 overlapping trades.
        closes = 100.0 + np.arange(periods) * 1e-4
        return pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": np.full(periods, 1000.0),
            },
            index=pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC"),
        )

    @classmethod
    def _expected_resolutions(cls) -> list[TradeResolution]:
        resolutions = []
        for position in range(cls._N_FIRES):
            fire_index = cls._fire_index(position)
            if cls._is_fast(position):
                resolutions.append(TradeResolution(profitable=True, exit_index=fire_index + 1))
            else:
                resolutions.append(
                    TradeResolution(profitable=False, exit_index=fire_index + cls._MAX_HOLDING_BARS)
                )
        return resolutions

    def test_training_and_serving_agree_on_overlapping_trailing_window(
        self, mock_config_class, mock_registry_class, mock_check_barrier_touch
    ):
        df = self._make_df()

        # --- Training side: hand-built fires/resolutions matching exactly
        # what the (real, unmocked) check_barrier_touch-driven serving path
        # below will independently produce (verified by construction: fast
        # fires touch take-profit at fire_index+1 for a net gain of
        # take_profit_pct minus round-trip fees, well above zero; slow
        # fires vertical-exit after max_holding_bars at a ~0.002% price
        # drift, well below round-trip fees) -- see _is_fast's docstring.
        fires = [
            PrimarySignalRecord(index=self._fire_index(p), direction=1, predicted_return=0.01)
            for p in range(self._N_FIRES)
        ]
        resolutions = self._expected_resolutions()
        features = build_meta_label_features(
            df, fires, resolutions, hit_rate_lookback=HIT_RATE_LOOKBACK_FIRES
        )
        training_hit_rate = features["rolling_hit_rate_20"].iloc[-1]

        # Sanity check this scenario is actually a meaningful regression
        # guard: resolution-order and fire-order windowing must disagree
        # here, or this test would pass even without the fix.
        buggy_fire_ordered = [
            resolutions[i].profitable
            for i in range(self._N_FIRES - 1)
            if resolutions[i].exit_index <= fires[-1].index
        ][-HIT_RATE_LOOKBACK_FIRES:]
        assert training_hit_rate != pytest.approx(float(np.mean(buggy_fire_ordered)))

        # --- Serving side: drive the REAL MetaLabelExamSignalGenerator
        # bar-by-bar. Only check_barrier_touch (the OHLC-touch primitive) is
        # stubbed, keyed by each fire's own (stop_loss_price,
        # take_profit_price) pair (unique per fire thanks to the drifting
        # entry prices) -- everything else (fire registration, resolution
        # bookkeeping, _rolling_hit_rate, encode_meta_label_feature_row) is
        # the actual production code.
        fast_signatures: set[tuple[float, float]] = set()
        for position in range(self._N_FIRES):
            if not self._is_fast(position):
                continue
            entry_price = float(df["close"].iloc[self._fire_index(position)])
            sl = entry_price * (1.0 - self._STOP_LOSS_PCT)
            tp = entry_price * (1.0 + self._TAKE_PROFIT_PCT)
            fast_signatures.add((round(sl, 8), round(tp, 8)))

        def fake_touch(side, candle_high, candle_low, stop_loss_price, take_profit_price):
            key = (round(stop_loss_price, 8), round(take_profit_price, 8))
            hit_take_profit = key in fast_signatures
            return BarrierTouchResult(
                hit_stop_loss=False,
                hit_take_profit=hit_take_profit,
                stop_loss_exit_price=stop_loss_price,
                take_profit_exit_price=take_profit_price,
            )

        mock_check_barrier_touch.side_effect = fake_touch

        mock_bundle = MagicMock()
        mock_bundle.runner.predict.side_effect = [
            _make_meta_label_prediction(0.9) for _ in range(self._N_FIRES)
        ]
        mock_registry = MagicMock()
        mock_registry.select_bundle.return_value = mock_bundle
        mock_registry_class.return_value = mock_registry

        fire_at = {self._fire_index(p): (SignalDirection.BUY, 0.01) for p in range(self._N_FIRES)}
        primary = _CannedPrimarySignalGenerator(fire_at=fire_at, warmup=self._WARMUP)
        generator = MetaLabelExamSignalGenerator(
            primary_signal_generator=primary,
            min_confidence=0.0,
            take_profit_pct=self._TAKE_PROFIT_PCT,
            stop_loss_pct=self._STOP_LOSS_PCT,
            max_holding_bars=self._MAX_HOLDING_BARS,
        )

        last_fire_index = self._fire_index(self._N_FIRES - 1)
        for index in range(self._WARMUP, last_fire_index + 1):
            generator.generate_signal(df, index)

        serving_hit_rate = generator._rolling_hit_rate()

        assert serving_hit_rate == pytest.approx(training_hit_rate)
