"""Regression tests for #1206: ``position.quantity`` staleness after a scale-in.

``LivePositionTracker.apply_scale_in`` used to grow ``current_size``/``size`` without
ever touching ``quantity`` (the base-asset amount from the entry fill) or
``original_size``, so every consumer that recovers "how much base asset is actually
held" via ``quantity * (current_size / original_size)`` either fell back to a less
accurate path (``_closed_base_quantity`` -> None) or silently under-reported the true
held amount (``held_protection_quantity``).

Worked example pinned by these tests: an entry buys 1.0 unit at price 100 on a
$10,000 entry balance (size 0.01 = 1% of balance). A scale-in adds 0.2% of balance
(0.002) at price 50 (a lower price, as scale-ins on a drawdown typically are) — that
buys an additional 0.002 * 10,000 / 50 = 0.4 units. After the fix, ``quantity`` reads
1.4, ``original_size`` grows from 0.01 to 0.012 in lockstep with ``current_size``, so
the ratio stays 1.0 (fully held, nothing exited since the scale-in).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.engines.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
    PositionSide,
)
from src.engines.live.execution.stop_loss_manager import LiveStopLossManager
from src.engines.live.trade_close_accounting import _closed_base_quantity

pytestmark = pytest.mark.fast

ENTRY_TIME = datetime(2024, 1, 1, tzinfo=UTC)

ENTRY_BALANCE = 10_000.0
ENTRY_PRICE = 100.0
ENTRY_SIZE = 0.01  # 1% of balance -> 1.0 unit at entry_price=100
ENTRY_QUANTITY = 1.0

SCALE_IN_DELTA_FRACTION = 0.002  # 0.2% of balance
SCALE_IN_PRICE = 50.0
SCALE_IN_ADDED_UNITS = 0.4  # 0.002 * 10_000 / 50


def _make_position(**overrides) -> LivePosition:
    defaults = dict(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=ENTRY_SIZE,
        entry_price=ENTRY_PRICE,
        entry_time=ENTRY_TIME,
        entry_balance=ENTRY_BALANCE,
        quantity=ENTRY_QUANTITY,
        original_size=ENTRY_SIZE,
        current_size=ENTRY_SIZE,
        order_id="order-1",
    )
    defaults.update(overrides)
    return LivePosition(**defaults)


def _tracked_position(tracker: LivePositionTracker, **overrides) -> LivePosition:
    position = _make_position(**overrides)
    tracker._positions["order-1"] = position
    return position


class TestApplyScaleInGrowsQuantity:
    def test_quantity_grows_by_the_units_the_scale_in_bought(self) -> None:
        tracker = LivePositionTracker(db_manager=None)
        position = _tracked_position(tracker)

        result = tracker.apply_scale_in(
            order_id="order-1",
            delta_fraction=SCALE_IN_DELTA_FRACTION,
            price=SCALE_IN_PRICE,
            threshold_level=0,
            max_position_size=1.0,
        )

        assert result is not None
        assert position.quantity == pytest.approx(ENTRY_QUANTITY + SCALE_IN_ADDED_UNITS)

    def test_original_size_grows_in_lockstep_with_current_size(self) -> None:
        tracker = LivePositionTracker(db_manager=None)
        position = _tracked_position(tracker)

        tracker.apply_scale_in(
            order_id="order-1",
            delta_fraction=SCALE_IN_DELTA_FRACTION,
            price=SCALE_IN_PRICE,
            threshold_level=0,
            max_position_size=1.0,
        )

        expected_size = ENTRY_SIZE + SCALE_IN_DELTA_FRACTION
        assert position.current_size == pytest.approx(expected_size)
        assert position.original_size == pytest.approx(expected_size)
        # Ratio stays meaningful: fully held since the scale-in.
        assert position.current_size / position.original_size == pytest.approx(1.0)

    def test_quantity_unchanged_when_scale_in_fully_capped_out(self) -> None:
        """A scale-in that adds nothing (already at the max-position cap) must not
        perturb quantity/original_size — there is no real growth to reflect."""
        tracker = LivePositionTracker(db_manager=None)
        position = _tracked_position(tracker, size=0.10, original_size=0.10, current_size=0.10)

        tracker.apply_scale_in(
            order_id="order-1",
            delta_fraction=0.05,
            price=SCALE_IN_PRICE,
            threshold_level=0,
            max_position_size=0.10,
        )

        assert position.quantity == pytest.approx(ENTRY_QUANTITY)
        assert position.original_size == pytest.approx(0.10)

    def test_missing_entry_balance_leaves_quantity_untouched_but_grows_original_size(
        self,
    ) -> None:
        """Without a valid entry_balance/price basis the added units cannot be derived
        safely — quantity is left as-is (never fabricated) but current_size/original_size
        still grow together so the ratio itself does not overflow."""
        tracker = LivePositionTracker(db_manager=None)
        position = _tracked_position(tracker, entry_balance=None)

        tracker.apply_scale_in(
            order_id="order-1",
            delta_fraction=SCALE_IN_DELTA_FRACTION,
            price=SCALE_IN_PRICE,
            threshold_level=0,
            max_position_size=1.0,
        )

        assert position.quantity == pytest.approx(ENTRY_QUANTITY)
        expected_size = ENTRY_SIZE + SCALE_IN_DELTA_FRACTION
        assert position.original_size == pytest.approx(expected_size)
        assert position.current_size == pytest.approx(expected_size)

    def test_quantity_none_at_entry_stays_none(self) -> None:
        """A position with no tracked quantity (legacy/incomplete recovery) is not
        given a fabricated baseline by a scale-in."""
        tracker = LivePositionTracker(db_manager=None)
        position = _tracked_position(tracker, quantity=None)

        tracker.apply_scale_in(
            order_id="order-1",
            delta_fraction=SCALE_IN_DELTA_FRACTION,
            price=SCALE_IN_PRICE,
            threshold_level=0,
            max_position_size=1.0,
        )

        assert position.quantity is None


class TestScaleInFixesDownstreamConsumers:
    """The consumers named in #1206 recover a correct held quantity once
    ``apply_scale_in`` keeps ``quantity``/``original_size`` in sync."""

    def test_closed_base_quantity_no_longer_none_after_scale_in(self) -> None:
        tracker = LivePositionTracker(db_manager=None)
        position = _tracked_position(tracker)

        tracker.apply_scale_in(
            order_id="order-1",
            delta_fraction=SCALE_IN_DELTA_FRACTION,
            price=SCALE_IN_PRICE,
            threshold_level=0,
            max_position_size=1.0,
        )

        # Before #1206: current_size (0.012) > original_size (0.01) -> None.
        # After #1206: both grew together -> the full held quantity is derivable.
        closed_quantity = _closed_base_quantity(position)
        assert closed_quantity is not None
        assert closed_quantity == pytest.approx(ENTRY_QUANTITY + SCALE_IN_ADDED_UNITS)

    def test_closed_base_quantity_reflects_partial_exit_after_scale_in(self) -> None:
        """A partial exit taken *after* the scale-in reduces current_size below the
        (now larger) original_size, and the closed slice scales correctly off the
        grown quantity — not just the original entry-fill amount."""
        tracker = LivePositionTracker(db_manager=None)
        position = _tracked_position(tracker)

        tracker.apply_scale_in(
            order_id="order-1",
            delta_fraction=SCALE_IN_DELTA_FRACTION,
            price=SCALE_IN_PRICE,
            threshold_level=0,
            max_position_size=1.0,
        )
        tracker.apply_partial_exit(
            order_id="order-1",
            delta_fraction=position.current_size / 2,
            price=SCALE_IN_PRICE,
            target_level=0,
            basis_balance=ENTRY_BALANCE,
        )

        total_quantity = ENTRY_QUANTITY + SCALE_IN_ADDED_UNITS
        assert _closed_base_quantity(position) == pytest.approx(total_quantity * 0.5)

    def test_held_protection_quantity_reflects_true_held_amount_after_scale_in(
        self,
    ) -> None:
        tracker = LivePositionTracker(db_manager=None)
        position = _tracked_position(tracker)

        tracker.apply_scale_in(
            order_id="order-1",
            delta_fraction=SCALE_IN_DELTA_FRACTION,
            price=SCALE_IN_PRICE,
            threshold_level=0,
            max_position_size=1.0,
        )

        # Before #1206: quantity (1.0) * current/original (0.012/0.01 = 1.2) = 1.2,
        # under-stating the true 1.4 units held (scale-in bought at a lower price than
        # entry, so the flat entry-price-implied ratio undercounts the added units).
        # After #1206: quantity itself already carries the added units, ratio is 1.0.
        held = LiveStopLossManager.held_protection_quantity(position)
        assert held == pytest.approx(ENTRY_QUANTITY + SCALE_IN_ADDED_UNITS)
