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

from src.prediction import PredictionResult
from src.prediction.distribution_stats import FrozenDistribution
from src.strategies.components.exam_signal_generator import (
    ClassificationExamSignalGenerator,
    SmoothedReturnExamSignalGenerator,
)
from src.strategies.components.signal_generator import SignalDirection


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

    def test_missing_target_distribution_holds(self, mock_config_class, mock_engine_class):
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
