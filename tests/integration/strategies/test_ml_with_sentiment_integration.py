from unittest.mock import Mock

import numpy as np
import pytest

from strategies.ml_with_sentiment import MlWithSentiment

pytestmark = pytest.mark.integration


class TestMlWithSentimentIntegration:
    def test_enhanced_strategy_initialization(self):
        strategy = MlWithSentiment()
        assert hasattr(strategy, "name")
        assert getattr(strategy, "trading_pair", "BTCUSDT") is not None
        assert hasattr(strategy, "model_path")
        assert hasattr(strategy, "sequence_length")
        assert hasattr(strategy, "use_sentiment")
        assert hasattr(strategy, "stop_loss_pct")
        assert hasattr(strategy, "take_profit_pct")

    def test_enhanced_strategy_execution_logging(self, sample_ohlcv_data):
        strategy = MlWithSentiment()
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=101)
        df = strategy.calculate_indicators(sample_ohlcv_data)
        valid_indices = range(max(120, len(df) - 3), len(df))
        for i in valid_indices:
            if i < len(df):
                result = strategy.check_entry_conditions(df, i)
                assert isinstance(result, (bool, np.bool_))
                mock_db_manager.log_strategy_execution.assert_called()
                _, kwargs = mock_db_manager.log_strategy_execution.call_args
                assert kwargs["strategy_name"] == strategy.name
                assert kwargs["signal_type"] == "entry"
                assert kwargs["price"] > 0
                assert isinstance(kwargs["reasons"], list) and len(kwargs["reasons"]) > 0

    def test_enhanced_strategy_missing_prediction_logging(self, sample_ohlcv_data):
        strategy = MlWithSentiment()
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=202)
        df_no_predictions = sample_ohlcv_data.copy()
        df_no_predictions["ml_prediction"] = np.nan
        if len(df_no_predictions) > 1:
            result = strategy.check_entry_conditions(df_no_predictions, 1)
            assert result is False
            mock_db_manager.log_strategy_execution.assert_called()
            _, kwargs = mock_db_manager.log_strategy_execution.call_args
            assert "missing_ml_prediction" in kwargs["reasons"]
            assert "prediction_available=False" in kwargs["reasons"]
