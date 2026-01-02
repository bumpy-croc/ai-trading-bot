"""Tests for infrastructure.logging.decision_logger module."""

from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.logging.decision_logger import log_strategy_execution


class TestLogStrategyExecution:
    """Tests for log_strategy_execution function."""

    def test_calls_db_manager_with_all_params(self):
        """Test that db_manager.log_strategy_execution is called with all params."""
        mock_db = MagicMock()

        log_strategy_execution(
            mock_db,
            strategy_name="ml_basic",
            symbol="BTCUSDT",
            signal_type="BUY",
            action_taken="OPEN_LONG",
            price=50000.0,
            timeframe="1h",
            signal_strength=0.8,
            confidence_score=0.85,
            indicators={"rsi": 45, "macd": 100},
            sentiment_data={"score": 0.7},
            ml_predictions={"direction": "up", "confidence": 0.9},
            position_size=0.1,
            reasons=["Strong momentum", "Bullish sentiment"],
            volume=1000000.0,
            volatility=0.02,
            session_id=123,
        )

        mock_db.log_strategy_execution.assert_called_once()
        call_kwargs = mock_db.log_strategy_execution.call_args[1]

        assert call_kwargs["strategy_name"] == "ml_basic"
        assert call_kwargs["symbol"] == "BTCUSDT"
        assert call_kwargs["signal_type"] == "BUY"
        assert call_kwargs["action_taken"] == "OPEN_LONG"
        assert call_kwargs["price"] == 50000.0
        assert call_kwargs["timeframe"] == "1h"
        assert call_kwargs["signal_strength"] == 0.8
        assert call_kwargs["confidence_score"] == 0.85
        assert call_kwargs["indicators"] == {"rsi": 45, "macd": 100}
        assert call_kwargs["sentiment_data"] == {"score": 0.7}
        assert call_kwargs["ml_predictions"] == {"direction": "up", "confidence": 0.9}
        assert call_kwargs["position_size"] == 0.1
        assert call_kwargs["reasons"] == ["Strong momentum", "Bullish sentiment"]
        assert call_kwargs["volume"] == 1000000.0
        assert call_kwargs["volatility"] == 0.02
        assert call_kwargs["session_id"] == 123

    def test_returns_none_when_db_manager_is_none(self):
        """Test that function returns early when db_manager is None."""
        # Should not raise, just return
        result = log_strategy_execution(
            None,
            strategy_name="ml_basic",
            symbol="BTCUSDT",
            signal_type="BUY",
            action_taken="OPEN_LONG",
            price=50000.0,
            timeframe="1h",
            signal_strength=0.8,
            confidence_score=0.85,
        )
        assert result is None

    def test_handles_db_exception_silently(self):
        """Test that DB exceptions are caught and don't propagate."""
        mock_db = MagicMock()
        mock_db.log_strategy_execution.side_effect = Exception("DB connection failed")

        # Should not raise
        result = log_strategy_execution(
            mock_db,
            strategy_name="ml_basic",
            symbol="BTCUSDT",
            signal_type="BUY",
            action_taken="OPEN_LONG",
            price=50000.0,
            timeframe="1h",
            signal_strength=0.8,
            confidence_score=0.85,
        )
        assert result is None

    def test_empty_sentiment_data_becomes_none(self):
        """Test that empty sentiment_data dict becomes None."""
        mock_db = MagicMock()

        log_strategy_execution(
            mock_db,
            strategy_name="ml_basic",
            symbol="BTCUSDT",
            signal_type="BUY",
            action_taken="OPEN_LONG",
            price=50000.0,
            timeframe="1h",
            signal_strength=0.8,
            confidence_score=0.85,
            sentiment_data={},
        )

        call_kwargs = mock_db.log_strategy_execution.call_args[1]
        assert call_kwargs["sentiment_data"] is None

    def test_empty_ml_predictions_becomes_none(self):
        """Test that empty ml_predictions dict becomes None."""
        mock_db = MagicMock()

        log_strategy_execution(
            mock_db,
            strategy_name="ml_basic",
            symbol="BTCUSDT",
            signal_type="BUY",
            action_taken="OPEN_LONG",
            price=50000.0,
            timeframe="1h",
            signal_strength=0.8,
            confidence_score=0.85,
            ml_predictions={},
        )

        call_kwargs = mock_db.log_strategy_execution.call_args[1]
        assert call_kwargs["ml_predictions"] is None

    def test_none_reasons_becomes_empty_list(self):
        """Test that None reasons becomes empty list."""
        mock_db = MagicMock()

        log_strategy_execution(
            mock_db,
            strategy_name="ml_basic",
            symbol="BTCUSDT",
            signal_type="BUY",
            action_taken="OPEN_LONG",
            price=50000.0,
            timeframe="1h",
            signal_strength=0.8,
            confidence_score=0.85,
            reasons=None,
        )

        call_kwargs = mock_db.log_strategy_execution.call_args[1]
        assert call_kwargs["reasons"] == []

    def test_optional_params_default_to_none(self):
        """Test that optional parameters default to None."""
        mock_db = MagicMock()

        log_strategy_execution(
            mock_db,
            strategy_name="ml_basic",
            symbol="BTCUSDT",
            signal_type="BUY",
            action_taken="OPEN_LONG",
            price=50000.0,
            timeframe="1h",
            signal_strength=0.8,
            confidence_score=0.85,
        )

        call_kwargs = mock_db.log_strategy_execution.call_args[1]
        assert call_kwargs["indicators"] is None
        assert call_kwargs["sentiment_data"] is None
        assert call_kwargs["ml_predictions"] is None
        assert call_kwargs["position_size"] is None
        assert call_kwargs["volume"] is None
        assert call_kwargs["volatility"] is None
        assert call_kwargs["session_id"] is None


@pytest.mark.fast
class TestDecisionLoggerIntegration:
    """Integration tests for decision logger."""

    def test_full_logging_workflow(self):
        """Test a complete decision logging workflow."""
        mock_db = MagicMock()

        # Log multiple decisions
        log_strategy_execution(
            mock_db,
            strategy_name="ml_basic",
            symbol="BTCUSDT",
            signal_type="BUY",
            action_taken="OPEN_LONG",
            price=50000.0,
            timeframe="1h",
            signal_strength=0.8,
            confidence_score=0.85,
        )

        log_strategy_execution(
            mock_db,
            strategy_name="ml_basic",
            symbol="BTCUSDT",
            signal_type="SELL",
            action_taken="CLOSE_LONG",
            price=51000.0,
            timeframe="1h",
            signal_strength=0.7,
            confidence_score=0.75,
        )

        assert mock_db.log_strategy_execution.call_count == 2

    def test_resilient_to_db_failures(self):
        """Test that function is resilient to DB failures."""
        mock_db = MagicMock()
        mock_db.log_strategy_execution.side_effect = [
            Exception("Connection lost"),
            None,  # Second call succeeds
        ]

        # First call - should not raise
        log_strategy_execution(
            mock_db,
            strategy_name="ml_basic",
            symbol="BTCUSDT",
            signal_type="BUY",
            action_taken="OPEN_LONG",
            price=50000.0,
            timeframe="1h",
            signal_strength=0.8,
            confidence_score=0.85,
        )

        # Second call - should succeed
        log_strategy_execution(
            mock_db,
            strategy_name="ml_basic",
            symbol="BTCUSDT",
            signal_type="SELL",
            action_taken="CLOSE_LONG",
            price=51000.0,
            timeframe="1h",
            signal_strength=0.7,
            confidence_score=0.75,
        )

        assert mock_db.log_strategy_execution.call_count == 2
