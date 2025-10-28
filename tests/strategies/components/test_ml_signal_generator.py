"""
Unit tests for ML Signal Generator components
"""

from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.prediction import PredictionResult
from src.regime.detector import TrendLabel, VolLabel
from src.strategies.components.ml_signal_generator import MLBasicSignalGenerator, MLSignalGenerator
from src.strategies.components.regime_context import RegimeContext
from src.strategies.components.signal_generator import Signal, SignalDirection


class TestMLSignalGenerator:
    """Test MLSignalGenerator implementation"""

    def create_test_dataframe(self, length=150):
        """Create test DataFrame with OHLCV data and normalized features"""
        dates = pd.date_range("2023-01-01", periods=length, freq="1H")

        # Create realistic price data
        np.random.seed(42)
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, length)
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))  # Minimum price

        data = {
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "close": prices,
            "volume": np.random.uniform(1000, 10000, length),
        }

        df = pd.DataFrame(data, index=dates)

        # Add normalized features that would be created by feature pipeline
        for feature in ["close", "volume", "high", "low", "open"]:
            df[f"{feature}_normalized"] = (df[feature] - df[feature].min()) / (
                df[feature].max() - df[feature].min()
            )

        return df

    def create_regime_context(self):
        """Create test regime context"""
        return RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=15,
            strength=0.7,
        )

    @patch("src.strategies.components.ml_signal_generator.PredictionEngine")
    @patch("src.strategies.components.ml_signal_generator.PredictionConfig")
    def test_ml_signal_generator_initialization(self, mock_config_class, mock_engine_class):
        """Test MLSignalGenerator initialization with prediction engine"""
        mock_engine = MagicMock()
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = MLSignalGenerator(
            name="test_ml_generator",
            sequence_length=120,
        )

        assert generator.name == "test_ml_generator"
        assert generator.sequence_length == 120
        # Prediction engine should be initialized
        assert generator.prediction_engine is not None

    @patch("src.strategies.components.ml_signal_generator.PredictionEngine")
    @patch("src.strategies.components.ml_signal_generator.PredictionConfig")
    def test_lazy_onnx_session_initialization(self, mock_config_class, mock_engine_class):
        """Test that prediction engine is properly initialized"""
        # This test was originally for ONNX lazy loading,
        # but the new architecture always uses prediction engine
        mock_engine = MagicMock()
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        # Create generator
        generator = MLSignalGenerator(
            name="test_ml_generator",
            sequence_length=120,
        )

        # Prediction engine should be initialized during construction
        assert generator.prediction_engine is not None

        # Create test data
        df = self.create_test_dataframe(200)

        # Mock prediction result
        mock_result = Mock(spec=PredictionResult)
        mock_result.price = 51000.0
        mock_engine.predict.return_value = mock_result

        # Generate signal - should use prediction engine
        signal = generator.generate_signal(df, 150)

        # Verify prediction was called
        assert mock_engine.predict.called
        assert signal is not None

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_generate_signal_insufficient_history(self, mock_ort):
        """Test signal generation with insufficient history"""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_ort.return_value = mock_session

        generator = MLSignalGenerator(sequence_length=120)
        df = self.create_test_dataframe(50)  # Less than sequence_length

        signal = generator.generate_signal(df, 30)  # Valid index within dataframe

        assert signal.direction == SignalDirection.HOLD
        assert signal.strength == 0.0
        assert signal.confidence == 0.0
        assert signal.metadata["reason"] == "insufficient_history"

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_generate_signal_prediction_success(self, mock_ort):
        """Test successful signal generation with ML prediction"""
        # Mock ONNX session
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_session.run.return_value = [[[[0.6]]]]  # Normalized prediction
        mock_ort.return_value = mock_session

        generator = MLSignalGenerator(sequence_length=120)
        df = self.create_test_dataframe(150)

        signal = generator.generate_signal(df, 130)

        assert isinstance(signal.direction, SignalDirection)
        assert 0.0 <= signal.strength <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert "prediction" in signal.metadata
        assert "predicted_return" in signal.metadata

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_generate_signal_with_regime_awareness(self, mock_ort):
        """Test signal generation with regime-aware threshold adjustment"""
        # Mock ONNX session to return negative prediction (short signal)
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_session.run.return_value = [[[[0.4]]]]  # Prediction lower than current price
        mock_ort.return_value = mock_session

        generator = MLSignalGenerator(sequence_length=120)
        df = self.create_test_dataframe(150)
        regime = self.create_regime_context()

        signal = generator.generate_signal(df, 130, regime)

        assert "regime_trend" in signal.metadata
        assert "regime_volatility" in signal.metadata
        assert "regime_confidence" in signal.metadata
        assert "dynamic_threshold" in signal.metadata

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_calculate_dynamic_short_threshold(self, mock_ort):
        """Test dynamic short threshold calculation"""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_ort.return_value = mock_session

        generator = MLSignalGenerator()

        # Test different regime combinations
        regimes = [
            RegimeContext(TrendLabel.TREND_UP, VolLabel.LOW, 0.8, 10, 0.7),
            RegimeContext(TrendLabel.TREND_DOWN, VolLabel.HIGH, 0.6, 5, 0.5),
            RegimeContext(TrendLabel.RANGE, VolLabel.LOW, 0.9, 20, 0.8),
        ]

        for regime in regimes:
            threshold = generator._calculate_dynamic_short_threshold(regime)

            # Threshold should be negative and within reasonable bounds
            assert -0.01 <= threshold <= -0.0001
            assert isinstance(threshold, float)

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_confidence_scaling_for_short_threshold(self, mock_ort):
        """Test that confidence scaling works correctly for short thresholds"""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_ort.return_value = mock_session

        generator = MLSignalGenerator()

        # Test with same trend/volatility but different confidence levels
        base_regime_params = {
            "trend": TrendLabel.RANGE,
            "volatility": VolLabel.LOW,
            "duration": 10,
            "strength": 0.7,
        }

        # High confidence regime
        high_confidence_regime = RegimeContext(confidence=0.9, **base_regime_params)

        # Low confidence regime
        low_confidence_regime = RegimeContext(confidence=0.3, **base_regime_params)

        high_conf_threshold = generator._calculate_dynamic_short_threshold(high_confidence_regime)
        low_conf_threshold = generator._calculate_dynamic_short_threshold(low_confidence_regime)

        # High confidence should result in more aggressive threshold (closer to 0)
        assert (
            high_conf_threshold > low_conf_threshold
        ), f"High confidence threshold ({high_conf_threshold}) should be more aggressive (closer to 0) than low confidence ({low_conf_threshold})"

        # Both should be negative and within bounds
        assert high_conf_threshold < 0 and low_conf_threshold < 0
        assert -0.01 <= high_conf_threshold <= -0.0001
        assert -0.01 <= low_conf_threshold <= -0.0001

        # Test extreme cases
        perfect_confidence_regime = RegimeContext(confidence=1.0, **base_regime_params)
        no_confidence_regime = RegimeContext(confidence=0.0, **base_regime_params)

        perfect_threshold = generator._calculate_dynamic_short_threshold(perfect_confidence_regime)
        no_conf_threshold = generator._calculate_dynamic_short_threshold(no_confidence_regime)

        # Perfect confidence should be most aggressive
        assert (
            perfect_threshold > high_conf_threshold > low_conf_threshold > no_conf_threshold
        ), "Thresholds should be ordered by confidence level (higher confidence = more aggressive)"

    @patch("src.strategies.components.ml_signal_generator.PredictionEngine")
    @patch("src.strategies.components.ml_signal_generator.PredictionConfig")
    def test_prediction_engine_no_denormalization(self, mock_config_class, mock_engine_class):
        """Test that prediction engine results are not denormalized"""
        # Mock prediction engine
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        real_price = 50000.0  # Real price from prediction engine
        mock_result.price = real_price
        mock_engine.predict.return_value = mock_result
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = MLSignalGenerator(model_name="test_model")
        generator.prediction_engine = mock_engine

        # Create test data
        df = self.create_test_dataframe(200)

        # Get prediction
        prediction = generator._get_ml_prediction(df, 150)

        # Should return the real price directly (no denormalization)
        assert (
            prediction == real_price
        ), f"Prediction engine result should not be denormalized: expected {real_price}, got {prediction}"

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_confidence_calculation(self, mock_ort):
        """Test confidence calculation based on predicted return"""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_ort.return_value = mock_session

        generator = MLSignalGenerator()

        # Test different predicted returns
        test_returns = [0.001, 0.005, 0.01, 0.02, 0.05]

        for predicted_return in test_returns:
            confidence = generator._calculate_confidence(predicted_return)

            assert 0.0 <= confidence <= 1.0
            assert isinstance(confidence, float)

            # Higher returns should generally give higher confidence
            if predicted_return > 0.01:
                assert confidence > 0.1

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_get_confidence(self, mock_ort):
        """Test get_confidence method"""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_session.run.return_value = [[[[0.55]]]]
        mock_ort.return_value = mock_session

        generator = MLSignalGenerator(sequence_length=120)
        df = self.create_test_dataframe(150)

        confidence = generator.get_confidence(df, 130)

        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)

    @patch("src.strategies.components.ml_signal_generator.PredictionEngine")
    @patch("src.strategies.components.ml_signal_generator.PredictionConfig")
    def test_get_parameters(self, mock_config_class, mock_engine_class):
        """Test get_parameters method"""
        mock_engine = MagicMock()
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = MLSignalGenerator(
            name="test_generator", sequence_length=100
        )

        params = generator.get_parameters()

        assert params["name"] == "test_generator"
        assert params["sequence_length"] == 100
        assert "short_entry_threshold" in params
        assert "confidence_multiplier" in params

    @patch("src.strategies.components.ml_signal_generator.PredictionEngine")
    @patch("src.strategies.components.ml_signal_generator.PredictionConfig")
    def test_prediction_failure_handling(self, mock_config_class, mock_engine_class):
        """Test handling of prediction failures"""
        mock_engine = MagicMock()
        mock_engine.health_check.return_value = {"status": "healthy"}
        # Simulate prediction engine initialization failure
        mock_engine.predict.side_effect = Exception("Prediction failed")
        mock_engine_class.return_value = mock_engine

        generator = MLSignalGenerator(sequence_length=120)
        generator.prediction_engine = mock_engine
        df = self.create_test_dataframe(150)

        signal = generator.generate_signal(df, 130)

        assert signal.direction == SignalDirection.HOLD
        assert signal.strength == 0.0
        assert signal.confidence == 0.0
        assert signal.metadata["reason"] == "prediction_failed"


class TestMLBasicSignalGenerator:
    """Test MLBasicSignalGenerator implementation"""

    def create_test_dataframe(self, length=150):
        """Create test DataFrame with OHLCV data and normalized features"""
        dates = pd.date_range("2023-01-01", periods=length, freq="1H")

        # Create realistic price data
        np.random.seed(42)
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, length)
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))  # Minimum price

        data = {
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "close": prices,
            "volume": np.random.uniform(1000, 10000, length),
        }

        df = pd.DataFrame(data, index=dates)

        # Add normalized features
        for feature in ["close", "volume", "high", "low", "open"]:
            df[f"{feature}_normalized"] = (df[feature] - df[feature].min()) / (
                df[feature].max() - df[feature].min()
            )

        return df

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_ml_basic_signal_generator_initialization(self, mock_ort):
        """Test MLBasicSignalGenerator initialization"""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_ort.return_value = mock_session

        generator = MLBasicSignalGenerator(
            name="test_basic_generator",
            model_path="test_model.onnx",
            sequence_length=120,
            model_type="basic",
            timeframe="1h",
        )

        assert generator.name == "test_basic_generator"
        assert generator.model_path == "test_model.onnx"
        assert generator.sequence_length == 120
        assert generator.model_type == "basic"
        assert generator.model_timeframe == "1h"

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_generate_signal_basic_logic(self, mock_ort):
        """Test basic signal generation without regime awareness"""
        # Mock ONNX session
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_session.run.return_value = [[[[0.6]]]]  # Normalized prediction
        mock_ort.return_value = mock_session

        generator = MLBasicSignalGenerator(sequence_length=120)
        df = self.create_test_dataframe(150)

        # Create regime context (should be ignored in basic implementation)
        regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=15,
            strength=0.7,
        )

        signal = generator.generate_signal(df, 130, regime)

        assert isinstance(signal.direction, SignalDirection)
        assert 0.0 <= signal.strength <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert "prediction" in signal.metadata
        assert "short_threshold" in signal.metadata
        # Should not have regime-specific metadata
        assert "dynamic_threshold" not in signal.metadata

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_generate_signal_buy_condition(self, mock_ort):
        """Test buy signal generation"""
        # Mock ONNX session to return higher prediction than current price
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_session.run.return_value = [[[[0.9]]]]  # Very high normalized prediction
        mock_ort.return_value = mock_session

        generator = MLBasicSignalGenerator(sequence_length=120, use_prediction_engine=False)
        df = self.create_test_dataframe(150)

        signal = generator.generate_signal(df, 130)

        # Should generate buy signal for high prediction
        assert signal.direction == SignalDirection.BUY
        assert signal.strength > 0.0
        assert signal.confidence > 0.0

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_generate_signal_sell_condition(self, mock_ort):
        """Test sell signal generation"""
        # Mock ONNX session to return much lower prediction than current price
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_session.run.return_value = [[[[0.1]]]]  # Very low normalized prediction
        mock_ort.return_value = mock_session

        generator = MLBasicSignalGenerator(sequence_length=120)
        df = self.create_test_dataframe(150)

        signal = generator.generate_signal(df, 130)

        # Should generate sell signal if predicted return is below threshold
        assert signal.direction in [SignalDirection.SELL, SignalDirection.HOLD]
        if signal.direction == SignalDirection.SELL:
            assert signal.strength > 0.0

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_generate_signal_hold_condition(self, mock_ort):
        """Test hold signal generation"""
        # Mock ONNX session to return prediction close to current price
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_session.run.return_value = [[[[0.5]]]]  # Neutral prediction
        mock_ort.return_value = mock_session

        generator = MLBasicSignalGenerator(sequence_length=120)
        df = self.create_test_dataframe(150)

        signal = generator.generate_signal(df, 130)

        # Should generate hold signal for neutral predictions
        if signal.direction == SignalDirection.HOLD:
            assert signal.strength == 0.0

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_insufficient_history(self, mock_ort):
        """Test behavior with insufficient history"""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_ort.return_value = mock_session

        generator = MLBasicSignalGenerator(sequence_length=120)
        df = self.create_test_dataframe(50)  # Less than sequence_length

        signal = generator.generate_signal(df, 30)

        assert signal.direction == SignalDirection.HOLD
        assert signal.strength == 0.0
        assert signal.confidence == 0.0
        assert signal.metadata["reason"] == "insufficient_history"

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_get_confidence_basic(self, mock_ort):
        """Test get_confidence method for basic generator"""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_session.run.return_value = [[[[0.7]]]]
        mock_ort.return_value = mock_session

        generator = MLBasicSignalGenerator(sequence_length=120)
        df = self.create_test_dataframe(150)

        confidence = generator.get_confidence(df, 130)

        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_get_parameters_basic(self, mock_ort):
        """Test get_parameters method for basic generator"""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_ort.return_value = mock_session

        generator = MLBasicSignalGenerator(name="test_basic", model_type="advanced", timeframe="4h")

        params = generator.get_parameters()

        assert params["name"] == "test_basic"
        assert params["model_type"] == "advanced"
        assert params["model_timeframe"] == "4h"
        assert "short_entry_threshold" in params
        assert "confidence_multiplier" in params

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_prediction_engine_metadata(self, mock_ort):
        """Test prediction engine metadata inclusion"""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_session.run.return_value = [[[[0.6]]]]
        mock_ort.return_value = mock_session

        generator = MLBasicSignalGenerator(
            sequence_length=120, use_prediction_engine=True, model_name="test_model"
        )
        df = self.create_test_dataframe(150)

        signal = generator.generate_signal(df, 130)

        # Should include engine metadata even if engine is not actually initialized
        assert "engine_enabled" in signal.metadata
        assert "engine_model_name" in signal.metadata
        assert "engine_batch" in signal.metadata

    @patch("src.strategies.components.ml_signal_generator.PredictionEngine")
    @patch("src.strategies.components.ml_signal_generator.PredictionConfig")
    def test_mlbasic_prediction_engine_no_denormalization(self, mock_config_class, mock_engine_class):
        """Test that MLBasicSignalGenerator prediction engine results are not denormalized"""
        # Mock prediction engine
        mock_engine = MagicMock()
        mock_result = Mock(spec=PredictionResult)
        real_price = 45000.0  # Real price from prediction engine
        mock_result.price = real_price
        mock_engine.predict.return_value = mock_result
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_engine_class.return_value = mock_engine

        generator = MLBasicSignalGenerator(model_name="test_model")
        generator.prediction_engine = mock_engine

        # Create test data
        df = self.create_test_dataframe(200)

        # Get prediction
        prediction = generator._get_ml_prediction(df, 150)

        # Should return the real price directly (no denormalization)
        assert (
            prediction == real_price
        ), f"MLBasic prediction engine result should not be denormalized: expected {real_price}, got {prediction}"


class TestMLSignalGeneratorEdgeCases:
    """Test edge cases and error conditions for ML signal generators"""

    def create_test_dataframe(self, length=150):
        """Create test DataFrame with OHLCV data"""
        dates = pd.date_range("2023-01-01", periods=length, freq="1H")
        data = {
            "open": np.random.uniform(100, 110, length),
            "high": np.random.uniform(110, 120, length),
            "low": np.random.uniform(90, 100, length),
            "close": np.random.uniform(100, 110, length),
            "volume": np.random.uniform(1000, 10000, length),
        }
        return pd.DataFrame(data, index=dates)

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_invalid_dataframe(self, mock_ort):
        """Test behavior with invalid DataFrame"""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_ort.return_value = mock_session

        generator = MLSignalGenerator()

        # Test empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            generator.generate_signal(empty_df, 0)

        # Test DataFrame missing required columns
        invalid_df = pd.DataFrame({"price": [100, 101, 102]})
        with pytest.raises(ValueError, match="DataFrame missing required columns"):
            generator.generate_signal(invalid_df, 0)

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_invalid_index(self, mock_ort):
        """Test behavior with invalid index"""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_ort.return_value = mock_session

        generator = MLSignalGenerator()
        df = self.create_test_dataframe(100)

        # Test negative index
        with pytest.raises(IndexError, match="Index -1 is out of bounds"):
            generator.generate_signal(df, -1)

        # Test index >= length
        with pytest.raises(IndexError, match="Index 100 is out of bounds"):
            generator.generate_signal(df, 100)

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_nan_prediction_handling(self, mock_ort):
        """Test handling of NaN predictions"""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_session.run.side_effect = Exception("NaN prediction error")
        mock_ort.return_value = mock_session

        generator = MLSignalGenerator(sequence_length=50, use_prediction_engine=False)
        df = self.create_test_dataframe(100)

        # Add normalized features
        for feature in ["close", "volume", "high", "low", "open"]:
            df[f"{feature}_normalized"] = (df[feature] - df[feature].min()) / (
                df[feature].max() - df[feature].min()
            )

        signal = generator.generate_signal(df, 60)

        # Should handle prediction failure gracefully
        assert signal.direction == SignalDirection.HOLD
        assert signal.strength == 0.0
        assert signal.confidence == 0.0
        assert signal.metadata["reason"] == "prediction_failed"

    @patch("src.strategies.components.ml_signal_generator.ort.InferenceSession")
    def test_zero_price_handling(self, mock_ort):
        """Test handling of zero prices"""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_session.run.return_value = [[[[0.5]]]]
        mock_ort.return_value = mock_session

        generator = MLSignalGenerator(sequence_length=50)
        df = self.create_test_dataframe(100)

        # Set some prices to zero
        df.loc[df.index[60], "close"] = 0.0

        # Add normalized features
        for feature in ["close", "volume", "high", "low", "open"]:
            df[f"{feature}_normalized"] = (df[feature] - df[feature].min()) / (
                df[feature].max() - df[feature].min()
            )

        signal = generator.generate_signal(df, 60)

        # Should handle zero prices gracefully
        assert isinstance(signal, Signal)
        assert signal.metadata["predicted_return"] == 0  # Should be 0 when current_price is 0
