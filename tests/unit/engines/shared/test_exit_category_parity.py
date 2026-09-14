"""Exit-category taxonomy: parity between engines, and no prose-driven control flow (#1115).

The property under test is that the live and backtest engines assign the *same*
:class:`ExitReason` to the same logical exit. Before #1115 the two engines classified stop
exits by substring-matching the free-text reason — live case-insensitively, backtest against
the exact string ``"Stop loss"`` — so ``stop_loss`` and ``stop_loss_filled_offline`` silently
skipped the backtest branch entirely.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from src.engines.backtest.execution.exit_handler import ExitHandler
from src.engines.backtest.execution.position_tracker import PositionTracker
from src.engines.backtest.models import ActiveTrade
from src.engines.live.execution.exit_handler import LiveExitHandler
from src.engines.live.execution.position_tracker import LivePosition, LivePositionTracker
from src.engines.shared.execution.execution_model import ExecutionModel
from src.engines.shared.execution.fill_policy import default_fill_policy
from src.engines.shared.execution.order_intent import OrderType
from src.engines.shared.models import PositionSide
from src.trading.exit_reason import (
    LEGACY_EXIT_REASON_CATEGORIES,
    STOP_EXIT_CATEGORIES,
    ExitReason,
    classify_stop_exit,
    coerce_exit_category,
    infer_legacy_category,
)

ENTRY_PRICE = 100.0
STOP_PRICE = 95.0
ENTRY_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _backtest_handler(tracker: PositionTracker) -> ExitHandler:
    return ExitHandler(
        execution_engine=Mock(),
        position_tracker=tracker,
        risk_manager=Mock(),
        execution_model=ExecutionModel(default_fill_policy()),
        use_high_low_for_stops=True,
    )


def _live_handler(tracker: LivePositionTracker) -> LiveExitHandler:
    return LiveExitHandler(
        execution_engine=Mock(),
        position_tracker=tracker,
        execution_model=ExecutionModel(default_fill_policy()),
        use_high_low_for_stops=True,
    )


def _backtest_stop_category(*, trailing: bool, breakeven: bool) -> ExitReason:
    """Drive the backtest handler into a stop-loss exit and return its category."""
    tracker = PositionTracker()
    tracker.open_position(
        ActiveTrade(
            symbol="TEST",
            side=PositionSide.LONG,
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            size=0.1,
            stop_loss=STOP_PRICE,
            trailing_stop_activated=trailing,
            breakeven_triggered=breakeven,
        )
    )
    candle = pd.Series(
        {"open": 99.0, "high": 99.5, "low": 94.0, "close": 94.5, "volume": 1000.0},
        name=ENTRY_TIME + timedelta(hours=1),
    )
    result = _backtest_handler(tracker).check_exit_conditions(
        runtime_decision=None,
        candle=candle,
        current_price=94.5,
        symbol="TEST",
    )
    assert result.should_exit is True
    assert result.is_stop_loss is True
    return result.exit_category


def _live_stop_category(*, trailing: bool, breakeven: bool) -> ExitReason:
    """Drive the live handler into a stop-loss exit and return its category."""
    tracker = LivePositionTracker()
    position = LivePosition(
        symbol="TEST",
        side=PositionSide.LONG,
        entry_price=ENTRY_PRICE,
        entry_time=ENTRY_TIME,
        size=0.1,
        order_id="order-1",
        stop_loss=STOP_PRICE,
        trailing_stop_activated=trailing,
        breakeven_triggered=breakeven,
    )
    tracker.positions[position.order_id] = position
    result = _live_handler(tracker).check_exit_conditions(
        position=position,
        current_price=94.5,
        candle_high=99.5,
        candle_low=94.0,
    )
    assert result.should_exit is True
    return result.exit_category


@pytest.mark.fast
class TestEngineParity:
    """Both engines must categorize the same logical exit identically."""

    @pytest.mark.parametrize(
        ("trailing", "breakeven", "expected"),
        [
            (False, False, ExitReason.STOP_LOSS),
            (False, True, ExitReason.BREAKEVEN_STOP),
            (True, False, ExitReason.TRAILING_STOP),
            (True, True, ExitReason.TRAILING_STOP),
        ],
    )
    def test_stop_loss_exit_categories_agree(
        self, trailing: bool, breakeven: bool, expected: ExitReason
    ) -> None:
        backtest = _backtest_stop_category(trailing=trailing, breakeven=breakeven)
        live = _live_stop_category(trailing=trailing, breakeven=breakeven)
        assert backtest == live == expected

    def test_protective_and_trailing_stops_are_distinguishable(self) -> None:
        """The distinction exit-quality analysis needs must not collapse."""
        assert _live_stop_category(trailing=False, breakeven=False) != _live_stop_category(
            trailing=True, breakeven=False
        )

    def test_both_engines_pick_a_stop_order_for_every_stop_category(self) -> None:
        """Order-type selection keys off the category, not the reason prose.

        Pre-#1115 the backtest branch required the literal ``"Stop loss"``, so the
        ``stop_loss`` and ``stop_loss_filled_offline`` spellings fell through to a market
        order with no gap pricing.
        """
        bt_tracker = PositionTracker()
        trade = ActiveTrade(
            symbol="TEST",
            side=PositionSide.LONG,
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            size=0.1,
            stop_loss=STOP_PRICE,
        )
        bt_tracker.open_position(trade)
        backtest = _backtest_handler(bt_tracker)

        live_tracker = LivePositionTracker()
        position = LivePosition(
            symbol="TEST",
            side=PositionSide.LONG,
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            size=0.1,
            order_id="order-1",
            stop_loss=STOP_PRICE,
        )
        live = _live_handler(live_tracker)

        for category in STOP_EXIT_CATEGORIES:
            # A snake_case detail string that the old substring branch would have missed.
            detail = "stop_loss_filled_offline"
            bt_intent = backtest._build_exit_intent(
                trade, detail, category, backtest._map_exit_order_side(trade)
            )
            live_intent = live._build_exit_intent(position, detail, category, None)
            assert bt_intent.order_type is OrderType.STOP_LOSS
            assert live_intent.order_type is OrderType.STOP_LOSS


@pytest.mark.fast
class TestClassifier:
    """classify_stop_exit is the single source of the protected/trailing distinction."""

    def test_defaults_to_protective_stop(self) -> None:
        assert classify_stop_exit(object()) is ExitReason.STOP_LOSS

    def test_mock_attributes_do_not_read_as_activated(self) -> None:
        """A test double's truthy MagicMock attributes must not fake a trailing stop."""
        assert classify_stop_exit(Mock()) is ExitReason.STOP_LOSS


@pytest.mark.fast
class TestCoercion:
    """Category coercion never raises — a bad value degrades a report, not a close."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (ExitReason.TRAILING_STOP, ExitReason.TRAILING_STOP),
            ("trailing_stop", ExitReason.TRAILING_STOP),
            ("Stop loss", ExitReason.UNKNOWN),
            (None, ExitReason.UNKNOWN),
            (object(), ExitReason.UNKNOWN),
        ],
    )
    def test_coerce(self, value: object, expected: ExitReason) -> None:
        assert coerce_exit_category(value) is expected

    def test_mock_coerces_to_unknown(self) -> None:
        assert coerce_exit_category(Mock()) is ExitReason.UNKNOWN


@pytest.mark.fast
class TestLegacyMapping:
    """The historical mapping must cover every string prod actually holds."""

    #: Distinct exit_reason values in the production trades table on 2026-08-20 (#1115).
    PROD_VALUES = (
        "Stop loss",
        "stop_loss",
        "stop_loss_filled_offline",
        "Stop-loss placement failed - emergency close",
        "Engine shutdown",
    )

    @pytest.mark.parametrize("reason", PROD_VALUES)
    def test_every_prod_value_maps(self, reason: str) -> None:
        assert infer_legacy_category(reason) is not ExitReason.UNKNOWN

    def test_all_three_stop_spellings_collapse_to_one_category(self) -> None:
        categories = {
            infer_legacy_category(r)
            for r in ("Stop loss", "stop_loss", "stop_loss_filled_offline", "stop_loss_offline")
        }
        assert categories == {ExitReason.STOP_LOSS}

    def test_prefixed_reasons_map(self) -> None:
        assert infer_legacy_category("Early cut @ 12h, MFE 0.3%") is ExitReason.EARLY_CUT
        assert (
            infer_legacy_category("Partial exits complete @ level 2")
            is ExitReason.PARTIAL_EXIT_COMPLETE
        )

    def test_unknown_and_empty(self) -> None:
        assert infer_legacy_category("") is ExitReason.UNKNOWN
        assert infer_legacy_category(None) is ExitReason.UNKNOWN
        assert infer_legacy_category("something new") is ExitReason.UNKNOWN

    def test_map_targets_are_real_categories(self) -> None:
        assert all(isinstance(v, ExitReason) for v in LEGACY_EXIT_REASON_CATEGORIES.values())
