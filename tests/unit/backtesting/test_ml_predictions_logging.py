"""Tests for wiring ML prediction outputs into strategy-execution logging (#914).

``strategy_executions.ml_predictions`` was JSON null in every row ever written
because the column-based extractor looks for dataframe columns the strategies
never produce — model outputs live on Signal metadata. These tests pin the
signal-metadata extractor and the backtest engine call sites that merge it
into the logged ``ml_predictions`` dict, including the failed-prediction case
(which must be visible in the DB rather than logged as null).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.engines.backtest.utils import extract_ml_predictions_from_signal
from src.strategies.components import Signal, SignalDirection

pytestmark = pytest.mark.fast


def _ml_success_signal() -> Signal:
    """Signal shaped like MLBasicSignalGenerator's successful-prediction output."""
    return Signal(
        direction=SignalDirection.BUY,
        strength=0.5,
        confidence=0.42,
        metadata={
            "generator": "ml_basic_signal_generator",
            "prediction": 101.5,
            "current_price": 100.0,
            "predicted_return": 0.015,
            "index": 42,
            "sequence_length": 120,
            "long_entry_threshold": 0.001,
            "engine_model_name": "BTCUSDT:1h:basic:v1",
            "engine_batch": False,
            "model_type": "basic",
            "model_timeframe": "1h",
            "trading_symbol": "BTCUSDT",
            "model_symbol": "BTCUSDT",
        },
    )


def _ml_failure_signal() -> Signal:
    """Signal shaped like the ML generators' prediction-failed output."""
    return Signal(
        direction=SignalDirection.HOLD,
        strength=0.0,
        confidence=0.0,
        metadata={
            "generator": "ml_basic_signal_generator",
            "reason": "prediction_failed",
            "index": 42,
            "trading_symbol": "BTCUSDT",
            "model_symbol": None,
        },
    )


class TestExtractMlPredictionsFromSignal:
    def test_successful_prediction_extracts_model_outputs(self):
        result = extract_ml_predictions_from_signal(_ml_success_signal())

        assert result["prediction"] == 101.5
        assert result["predicted_return"] == 0.015
        assert result["current_price"] == 100.0
        assert result["engine_model_name"] == "BTCUSDT:1h:basic:v1"
        assert result["model_type"] == "basic"
        assert result["model_timeframe"] == "1h"
        assert result["model_symbol"] == "BTCUSDT"
        assert result["generator"] == "ml_basic_signal_generator"
        assert "prediction_failed" not in result

    def test_failed_prediction_is_visible_not_empty(self):
        result = extract_ml_predictions_from_signal(_ml_failure_signal())

        assert result["prediction_failed"] is True
        assert result["reason"] == "prediction_failed"
        assert result["generator"] == "ml_basic_signal_generator"

    def test_invalid_prediction_or_price_is_flagged_as_failure(self):
        signal = Signal(
            direction=SignalDirection.HOLD,
            strength=0.0,
            confidence=0.0,
            metadata={
                "generator": "ml_signal_generator",
                "reason": "invalid_prediction_or_price",
                "prediction": float("nan"),
                "current_price": 100.0,
                "predicted_return": 0,
                "index": 42,
            },
        )

        result = extract_ml_predictions_from_signal(signal)

        assert result["prediction_failed"] is True
        assert result["reason"] == "invalid_prediction_or_price"

    def test_error_fields_flow_through_when_present(self):
        signal = Signal(
            direction=SignalDirection.HOLD,
            strength=0.0,
            confidence=0.0,
            metadata={
                "generator": "ml_basic_signal_generator",
                "reason": "prediction_failed",
                "error": "ONNX session timed out",
                "error_type": "TimeoutError",
            },
        )

        result = extract_ml_predictions_from_signal(signal)

        assert result["error"] == "ONNX session timed out"
        assert result["error_type"] == "TimeoutError"

    def test_non_ml_signal_returns_empty(self):
        # "insufficient_history" is also emitted by technical generators, so a
        # bare reason without prediction keys must not be treated as ML output.
        signal = Signal(
            direction=SignalDirection.HOLD,
            strength=0.0,
            confidence=0.0,
            metadata={"generator": "rsi_signal_generator", "reason": "insufficient_history"},
        )

        assert extract_ml_predictions_from_signal(signal) == {}

    def test_none_signal_returns_empty(self):
        assert extract_ml_predictions_from_signal(None) == {}

    def test_non_dict_metadata_returns_empty(self):
        assert extract_ml_predictions_from_signal(MagicMock(metadata="not-a-dict")) == {}


def _sample_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="1h")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000.0, 1100.0, 1200.0],
        },
        index=idx,
    )


def _make_backtester_stub():
    """Backtester with only the attributes the decision-logging paths touch."""
    from src.engines.backtest.engine import Backtester

    bt = object.__new__(Backtester)
    bt.balance = 1000.0
    bt.peak_balance = 1000.0
    bt.trading_session_id = None
    bt.enable_dynamic_risk = False
    bt.dynamic_risk_adjustments = []
    bt.strategy = MagicMock()
    bt._component_strategy = None
    bt.entry_handler = MagicMock()
    bt.exit_handler = MagicMock()
    bt.position_tracker = MagicMock()
    bt.event_logger = MagicMock()
    return bt


def _runtime_decision(signal: Signal) -> MagicMock:
    decision = MagicMock()
    decision.signal = signal
    decision.regime = None
    decision.risk_metrics = None
    decision.metadata = {}
    return decision


class TestBacktestEngineMlPredictionsLogging:
    def test_entry_no_action_logs_signal_ml_predictions(self):
        bt = _make_backtester_stub()
        entry_signal = MagicMock()
        entry_signal.should_enter = False
        entry_signal.reasons = ["entry_conditions_not_met"]
        bt.entry_handler.process_runtime_decision.return_value = entry_signal
        bt.event_logger.should_log_candle.return_value = True

        df = _sample_df()
        bt._process_entry_signal(
            runtime_decision=_runtime_decision(_ml_success_signal()),
            df=df,
            index=2,
            candle=df.iloc[2],
            current_price=102.5,
            current_time=datetime(2024, 1, 1, 2, tzinfo=UTC),
            symbol="BTCUSDT",
            timeframe="1h",
        )

        kwargs = bt.event_logger.log_entry_decision.call_args.kwargs
        assert kwargs["ml_predictions"]["prediction"] == 101.5
        assert kwargs["ml_predictions"]["predicted_return"] == 0.015
        assert kwargs["ml_predictions"]["engine_model_name"] == "BTCUSDT:1h:basic:v1"

    def test_entry_no_action_logs_prediction_failure(self):
        bt = _make_backtester_stub()
        entry_signal = MagicMock()
        entry_signal.should_enter = False
        entry_signal.reasons = ["entry_conditions_not_met"]
        bt.entry_handler.process_runtime_decision.return_value = entry_signal
        bt.event_logger.should_log_candle.return_value = True

        df = _sample_df()
        bt._process_entry_signal(
            runtime_decision=_runtime_decision(_ml_failure_signal()),
            df=df,
            index=2,
            candle=df.iloc[2],
            current_price=102.5,
            current_time=datetime(2024, 1, 1, 2, tzinfo=UTC),
            symbol="BTCUSDT",
            timeframe="1h",
        )

        kwargs = bt.event_logger.log_entry_decision.call_args.kwargs
        assert kwargs["ml_predictions"]["prediction_failed"] is True
        assert kwargs["ml_predictions"]["reason"] == "prediction_failed"

    def test_exit_decision_logs_signal_ml_predictions(self):
        bt = _make_backtester_stub()
        bt.exit_handler.update_trailing_stop.return_value = (False, None)
        partial_result = MagicMock()
        partial_result.realized_pnl = 0.0
        partial_result.scale_in_fees = 0.0
        bt.exit_handler.check_partial_operations.return_value = partial_result
        bt.position_tracker.current_trade = None
        exit_check = MagicMock()
        exit_check.should_exit = False
        bt.exit_handler.check_exit_conditions.return_value = exit_check
        bt.exit_handler.calculate_current_pnl_pct.return_value = 0.01

        df = _sample_df()
        exited, trade = bt._process_position_exit(
            runtime_decision=_runtime_decision(_ml_success_signal()),
            candle=df.iloc[2],
            current_price=102.5,
            current_time=datetime(2024, 1, 1, 2, tzinfo=UTC),
            df=df,
            index=2,
            symbol="BTCUSDT",
            timeframe="1h",
        )

        assert exited is False
        assert trade is None
        kwargs = bt.event_logger.log_exit_decision.call_args.kwargs
        assert kwargs["ml_predictions"]["prediction"] == 101.5
        assert kwargs["ml_predictions"]["engine_model_name"] == "BTCUSDT:1h:basic:v1"
