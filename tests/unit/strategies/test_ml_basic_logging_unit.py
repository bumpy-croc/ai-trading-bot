from unittest.mock import Mock

import numpy as np
import pytest

from strategies.ml_basic import MlBasic

pytestmark = pytest.mark.unit


class TestMlBasicLogging:
    def test_ml_basic_strategy_execution_logging(self, sample_ohlcv_data):
        strategy = MlBasic()
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=456)
        df = strategy.calculate_indicators(sample_ohlcv_data)
        valid_indices = range(max(120, len(df) - 3), len(df))
        for i in valid_indices:
            if i < len(df):
                result = strategy.check_entry_conditions(df, i)
                assert isinstance(result, bool)
                mock_db_manager.log_strategy_execution.assert_called()
                _, kwargs = mock_db_manager.log_strategy_execution.call_args
                assert kwargs["strategy_name"] == strategy.name
                assert kwargs["signal_type"] == "entry"
                assert kwargs["price"] > 0
                assert isinstance(kwargs["reasons"], list)
                assert len(kwargs["reasons"]) > 0
                assert "additional_context" in kwargs
                ctx = kwargs["additional_context"]
                assert "model_type" in ctx

    def test_ml_basic_strategy_missing_prediction_logging(self, sample_ohlcv_data):
        strategy = MlBasic()
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=789)
        df_no_predictions = sample_ohlcv_data.copy()
        df_no_predictions["onnx_pred"] = np.nan
        if len(df_no_predictions) > 1:
            result = strategy.check_entry_conditions(df_no_predictions, 1)
            assert result is False
            mock_db_manager.log_strategy_execution.assert_called()
            _, kwargs = mock_db_manager.log_strategy_execution.call_args
            assert "missing_ml_prediction" in kwargs["reasons"]
            assert any("prediction_available" in r for r in kwargs["reasons"])
