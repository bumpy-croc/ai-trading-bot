"""Unit tests for the shared OHLC barrier-touch helper.

This is the pure function extracted from ExitHandler.check_exit_conditions's
stop-loss/take-profit high/low comparison — the single source of truth reused
by both the live/backtest exit handler and the triple-barrier label
simulator (see docs/research/experiments/2026-07-10_target-redesign-tournament-prereg.md
Deviation 2 / #8 engineering inventory).
"""

import math

import pytest

from src.engines.shared.barrier_touch import BarrierTouchResult, check_barrier_touch


class TestLongStopLoss:
    def test_stop_loss_triggered_when_low_at_or_below_stop(self):
        result = check_barrier_touch(
            side="long",
            candle_high=101.0,
            candle_low=94.0,
            stop_loss_price=95.0,
            take_profit_price=None,
        )
        assert result.hit_stop_loss is True
        # Worst-case (gap-through) fill: candle low, not the stop price itself.
        assert result.stop_loss_exit_price == pytest.approx(94.0)

    def test_stop_loss_not_triggered_when_low_above_stop(self):
        result = check_barrier_touch(
            side="long",
            candle_high=101.0,
            candle_low=96.0,
            stop_loss_price=95.0,
            take_profit_price=None,
        )
        assert result.hit_stop_loss is False

    def test_stop_loss_boundary_is_inclusive(self):
        result = check_barrier_touch(
            side="long",
            candle_high=101.0,
            candle_low=95.0,
            stop_loss_price=95.0,
            take_profit_price=None,
        )
        assert result.hit_stop_loss is True
        assert result.stop_loss_exit_price == pytest.approx(95.0)


class TestLongTakeProfit:
    def test_take_profit_triggered_when_high_at_or_above_target(self):
        result = check_barrier_touch(
            side="long",
            candle_high=106.0,
            candle_low=99.0,
            stop_loss_price=None,
            take_profit_price=104.0,
        )
        assert result.hit_take_profit is True
        # Fills exactly at the take-profit level (not the gap-through high).
        assert result.take_profit_exit_price == pytest.approx(104.0)

    def test_take_profit_not_triggered_when_high_below_target(self):
        result = check_barrier_touch(
            side="long",
            candle_high=103.0,
            candle_low=99.0,
            stop_loss_price=None,
            take_profit_price=104.0,
        )
        assert result.hit_take_profit is False


class TestShortSide:
    def test_short_stop_loss_uses_candle_high(self):
        result = check_barrier_touch(
            side="short",
            candle_high=106.0,
            candle_low=99.0,
            stop_loss_price=105.0,
            take_profit_price=None,
        )
        assert result.hit_stop_loss is True
        assert result.stop_loss_exit_price == pytest.approx(106.0)

    def test_short_take_profit_uses_candle_low(self):
        result = check_barrier_touch(
            side="short",
            candle_high=101.0,
            candle_low=94.0,
            stop_loss_price=None,
            take_profit_price=95.0,
        )
        assert result.hit_take_profit is True
        assert result.take_profit_exit_price == pytest.approx(95.0)


class TestBothBarriersInSameBar:
    def test_both_hit_reports_both_flags_stop_loss_priority_left_to_caller(self):
        """Both flags are reported independently; SL-over-TP priority is a
        caller concern (matches ExitHandler.check_exit_conditions, which
        checks `if hit_stop_loss: ... elif hit_take_profit: ...`)."""
        result = check_barrier_touch(
            side="long",
            candle_high=110.0,
            candle_low=90.0,
            stop_loss_price=95.0,
            take_profit_price=104.0,
        )
        assert result.hit_stop_loss is True
        assert result.hit_take_profit is True


class TestNoLevelsConfigured:
    def test_none_stop_loss_never_triggers(self):
        result = check_barrier_touch(
            side="long",
            candle_high=200.0,
            candle_low=1.0,
            stop_loss_price=None,
            take_profit_price=None,
        )
        assert result.hit_stop_loss is False
        assert result.hit_take_profit is False


class TestInvalidSide:
    def test_unknown_side_raises(self):
        with pytest.raises(ValueError, match="side"):
            check_barrier_touch(
                side="sideways",
                candle_high=101.0,
                candle_low=99.0,
                stop_loss_price=95.0,
                take_profit_price=104.0,
            )


class TestResultIsFrozenDataclass:
    def test_result_fields(self):
        result = check_barrier_touch(
            side="long",
            candle_high=101.0,
            candle_low=99.0,
            stop_loss_price=None,
            take_profit_price=None,
        )
        assert isinstance(result, BarrierTouchResult)
        assert math.isfinite(result.stop_loss_exit_price)
        assert math.isfinite(result.take_profit_exit_price)
