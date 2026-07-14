"""Backtest behavior of the HyperGrowth/ETHUSDT long-only config (GH #1020).

End-to-end through the real factory -> deployment config -> signal generator
-> backtest engine chain:

- With the deployment default (allow_shorts=False for ETHUSDT), a backtest
  fed persistent SELL predictions opens ZERO short trades — the post-ship
  proof-monitoring bar from the risk review.
- The explicit ``allow_shorts=True`` override (counterfactual/research arm)
  still opens shorts on identical inputs, proving the harness would have
  shorted and the flag is the only thing stopping it.
- The backtest engine writes NO shadow suppression events (C6: live-only).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.data_providers.data_provider import DataProvider
from src.database.models import EventType
from src.engines.backtest.engine import Backtester
from src.engines.shared.models import PositionSide
from src.prediction import PredictionResult
from src.strategies.hyper_growth import create_hyper_growth_strategy

pytestmark = [pytest.mark.unit, pytest.mark.mock_only]

ENGINE_PATH = "src.strategies.components.ml_signal_generator.PredictionEngine"
CONFIG_PATH = "src.strategies.components.ml_signal_generator.PredictionConfig"

_BARS = 140  # sequence_length (120) warmup + a window of decision bars


class _FrameProvider(DataProvider):
    """Data provider returning one prepared DataFrame."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def get_historical_data(self, symbol, timeframe, start, end=None):  # type: ignore[override]
        return self._frame.copy()

    def get_current_price(self, symbol):  # type: ignore[override]
        return float(self._frame["close"].iloc[-1])

    def get_live_data(self, symbol, timeframe, limit=500):  # type: ignore[override]
        return self._frame.tail(limit).copy()

    def update_live_data(self, symbol, timeframe):  # type: ignore[override]
        return self._frame.copy()


def _downtrend_frame() -> pd.DataFrame:
    """Steady 0.4%/bar downtrend so predictions below close mean SELL bars."""
    start = datetime(2024, 1, 1)
    closes = [2000.0 * (0.996**i) for i in range(_BARS)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [1_000.0] * _BARS,
        },
        index=[start + timedelta(hours=i) for i in range(_BARS)],
    )


def _bearish_prediction_engine() -> MagicMock:
    """Mock engine predicting 2% below the latest close on every bar."""

    def _predict(window_df: pd.DataFrame, model_name: Any = None) -> Mock:
        result = Mock(spec=PredictionResult)
        result.error = None
        result.metadata = {}
        result.price = float(window_df["close"].iloc[-1]) * 0.98
        return result

    engine = MagicMock()
    engine.predict.side_effect = _predict
    engine.health_check.return_value = {"status": "healthy"}
    return engine


def _run_backtest(strategy) -> Backtester:
    backtester = Backtester(
        strategy,
        _FrameProvider(_downtrend_frame()),
        log_to_database=False,
        enable_dynamic_risk=False,
        enable_engine_risk_exits=False,
        use_next_bar_execution=False,
    )
    # Spy DB: proves the backtest engine writes no shadow suppression events.
    backtester.db_manager = MagicMock()
    frame = _downtrend_frame()
    backtester.run(symbol="ETHUSDT", timeframe="1h", start=frame.index[0], end=frame.index[-1])
    return backtester


def _shorts_opened(backtester: Backtester) -> int:
    """Count shorts the run opened: closed short trades + an open short."""
    count = len([t for t in backtester.trades if t.side == PositionSide.SHORT])
    if backtester.position_tracker.position_side == PositionSide.SHORT:
        count += 1
    return count


def _suppression_events(backtester: Backtester) -> list:
    return [
        call
        for call in backtester.db_manager.log_event.call_args_list
        if call.kwargs.get("event_type") == EventType.SHORT_ENTRY_SUPPRESSED
    ]


@patch(ENGINE_PATH)
@patch(CONFIG_PATH)
def test_deployment_default_opens_no_shorts_and_writes_no_events(_cfg, engine_cls):
    engine_cls.return_value = _bearish_prediction_engine()
    strategy = create_hyper_growth_strategy(symbol="ETHUSDT")

    backtester = _run_backtest(strategy)

    assert _shorts_opened(backtester) == 0
    assert backtester.position_tracker.has_position is False  # SELL-only feed
    assert _suppression_events(backtester) == []


@patch(ENGINE_PATH)
@patch(CONFIG_PATH)
def test_explicit_allow_shorts_true_still_opens_shorts(_cfg, engine_cls):
    """Control arm: identical inputs short when the flag is re-enabled."""
    engine_cls.return_value = _bearish_prediction_engine()
    strategy = create_hyper_growth_strategy(symbol="ETHUSDT", allow_shorts=True)

    backtester = _run_backtest(strategy)

    assert _shorts_opened(backtester) >= 1
    assert _suppression_events(backtester) == []
