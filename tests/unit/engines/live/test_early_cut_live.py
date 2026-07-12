"""LiveExitHandler integration for the MFE early-cut policy (#971).

Same shared policy implementation as backtest (parity requirement): the
live handler must produce the same decision and the same exit-reason string
from the same candles.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from src.engines.live.execution.exit_handler import LiveExitHandler
from src.engines.live.execution.position_tracker import LivePosition, PositionSide
from src.engines.shared.execution.execution_model import ExecutionModel
from src.engines.shared.execution.fill_policy import default_fill_policy
from src.position_management.early_cut import EarlyCutPolicy

pytestmark = [pytest.mark.unit, pytest.mark.fast]

ENTRY_PRICE = 100.0


def _make_df(highs: list[float], lows: list[float], start: datetime) -> pd.DataFrame:
    index = pd.date_range(start=start.replace(tzinfo=None), periods=len(highs), freq="1h")
    closes = [(h + lo) / 2 for h, lo in zip(highs, lows)]
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=index,
    )


def _make_handler(early_cut_policy: EarlyCutPolicy | None) -> LiveExitHandler:
    return LiveExitHandler(
        execution_engine=Mock(),
        position_tracker=Mock(),
        execution_model=ExecutionModel(default_fill_policy()),
        early_cut_policy=early_cut_policy,
    )


def _position(entry_time: datetime) -> LivePosition:
    return LivePosition(
        symbol="TESTUSDT",
        side=PositionSide.LONG,
        entry_price=ENTRY_PRICE,
        entry_time=entry_time,
        size=0.1,
        order_id="order-1",
    )


class TestLiveEarlyCut:
    def test_cuts_when_window_elapsed_and_mfe_below_threshold(self) -> None:
        # Entry 4 hours ago; window 3h; window MFE 1.9% < 2% => cut now.
        entry_time = datetime.now(UTC) - timedelta(hours=4)
        df = _make_df(
            highs=[100.8, 101.5, 101.9, 100.9, 100.9],
            lows=[99.5, 99.8, 100.1, 100.0, 100.0],
            start=entry_time,
        )
        handler = _make_handler(
            EarlyCutPolicy(mfe_threshold_pct=0.02, evaluation_window_hours=3)
        )

        result = handler.check_exit_conditions(
            position=_position(entry_time),
            current_price=100.5,
            candle_high=100.9,
            candle_low=100.0,
            df=df,
        )
        assert result.should_exit is True
        assert result.exit_reason.startswith("Early cut")
        assert result.limit_price is None

    def test_holds_before_window_elapses(self) -> None:
        entry_time = datetime.now(UTC) - timedelta(hours=2)
        df = _make_df(
            highs=[100.8, 101.5, 101.9],
            lows=[99.5, 99.8, 100.1],
            start=entry_time,
        )
        handler = _make_handler(
            EarlyCutPolicy(mfe_threshold_pct=0.02, evaluation_window_hours=3)
        )

        result = handler.check_exit_conditions(
            position=_position(entry_time),
            current_price=100.5,
            candle_high=101.9,
            candle_low=100.1,
            df=df,
        )
        assert result.should_exit is False

    def test_holds_when_mfe_reached_threshold(self) -> None:
        entry_time = datetime.now(UTC) - timedelta(hours=4)
        df = _make_df(
            highs=[100.8, 101.5, 102.1, 100.9, 100.9],
            lows=[99.5, 99.8, 100.1, 100.0, 100.0],
            start=entry_time,
        )
        handler = _make_handler(
            EarlyCutPolicy(mfe_threshold_pct=0.02, evaluation_window_hours=3)
        )

        result = handler.check_exit_conditions(
            position=_position(entry_time),
            current_price=100.5,
            candle_high=100.9,
            candle_low=100.0,
            df=df,
        )
        assert result.should_exit is False

    def test_holds_when_history_does_not_cover_entry(self) -> None:
        """Restart safety: a short kline buffer must never trigger a cut."""
        entry_time = datetime.now(UTC) - timedelta(hours=10)
        # Buffer starts 5 hours after entry — window MFE unknowable.
        df = _make_df(
            highs=[100.5, 100.5, 100.5],
            lows=[99.9, 99.9, 99.9],
            start=entry_time + timedelta(hours=5),
        )
        handler = _make_handler(
            EarlyCutPolicy(mfe_threshold_pct=0.02, evaluation_window_hours=3)
        )

        result = handler.check_exit_conditions(
            position=_position(entry_time),
            current_price=100.0,
            candle_high=100.5,
            candle_low=99.9,
            df=df,
        )
        assert result.should_exit is False

    def test_stop_loss_priority_over_early_cut(self) -> None:
        entry_time = datetime.now(UTC) - timedelta(hours=4)
        df = _make_df(
            highs=[100.8, 101.0, 101.0, 100.9, 95.5],
            lows=[99.5, 99.8, 99.9, 100.0, 94.0],
            start=entry_time,
        )
        handler = _make_handler(
            EarlyCutPolicy(mfe_threshold_pct=0.02, evaluation_window_hours=3)
        )
        position = _position(entry_time)
        position.stop_loss = 95.0

        result = handler.check_exit_conditions(
            position=position,
            current_price=94.5,
            candle_high=95.5,
            candle_low=94.0,
            df=df,
        )
        assert result.should_exit is True
        assert result.exit_reason == "Stop loss"

    def test_no_policy_no_df_unchanged(self) -> None:
        entry_time = datetime.now(UTC) - timedelta(hours=4)
        handler = _make_handler(None)

        result = handler.check_exit_conditions(
            position=_position(entry_time),
            current_price=100.5,
            candle_high=100.9,
            candle_low=100.0,
        )
        assert result.should_exit is False
