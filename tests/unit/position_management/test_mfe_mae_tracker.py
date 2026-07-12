"""MFEMAETracker unit tests.

The tracker stores RAW price excursion relative to entry (unsized, gross of
fees/slippage) so that ``mfe``/``mae`` always agree with their
``mfe_price``/``mae_price`` companions:

    mfe == (mfe_price - entry) / entry   (long)
    mae == (mae_price - entry) / entry   (long; mirrored for shorts)

Every expected value below is hand-computable on paper from entry/high/low
alone — position size must never leak into these numbers (the #966 corruption
was ``size * move - exit_costs`` being persisted as ``mfe``).
"""

from datetime import UTC, datetime

import pytest

from src.position_management.mfe_mae_tracker import MFEMAETracker


def test_mfe_mae_tracker_long_hand_computable():
    """entry=100, high=110, low=95 -> mfe=+0.10 @110, mae=-0.05 @95."""
    tracker = MFEMAETracker(precision_decimals=8)
    entry = 100.0
    now = datetime.now(UTC)

    m = tracker.update_position_metrics(
        position_key="p1",
        entry_price=entry,
        current_price=110.0,
        side="long",
        current_time=now,
    )
    assert m.mfe == pytest.approx(0.10)
    assert m.mfe_price == 110.0
    assert m.mae == 0.0
    assert m.mae_price is None

    m = tracker.update_position_metrics(
        position_key="p1",
        entry_price=entry,
        current_price=95.0,
        side="long",
        current_time=now,
    )
    assert m.mae == pytest.approx(-0.05)
    assert m.mae_price == 95.0
    # MFE retained from the earlier peak
    assert m.mfe == pytest.approx(0.10)
    assert m.mfe_price == 110.0


def test_mfe_mae_tracker_short_hand_computable():
    """Short entry=100: drop to 90 -> mfe=+0.10 @90; rise to 105 -> mae=-0.05 @105."""
    tracker = MFEMAETracker(precision_decimals=8)
    entry = 100.0
    now = datetime.now(UTC)

    m = tracker.update_position_metrics(
        position_key="p2",
        entry_price=entry,
        current_price=90.0,
        side="short",
        current_time=now,
    )
    assert m.mfe == pytest.approx(0.10)
    assert m.mfe_price == 90.0
    assert m.mae == 0.0

    m = tracker.update_position_metrics(
        position_key="p2",
        entry_price=entry,
        current_price=105.0,
        side="short",
        current_time=now,
    )
    assert m.mae == pytest.approx(-0.05)
    assert m.mae_price == 105.0
    assert m.mfe == pytest.approx(0.10)


def test_mfe_agrees_with_mfe_price_companion():
    """The stored fraction must be derivable from its own price companion.

    Regression for #966: prod rows disagreed with their companions by a
    non-constant 10-23x because the writer persisted sized, fee-netted
    values. Uses the actual prod trade-9 row values (ETHUSDT).
    """
    tracker = MFEMAETracker(precision_decimals=8)
    entry = 1673.21
    now = datetime.now(UTC)

    m = tracker.update_position_metrics("t9", entry, 1727.84, "long", now)
    m = tracker.update_position_metrics("t9", entry, 1605.11, "long", now)

    assert m.mfe == pytest.approx((m.mfe_price - entry) / entry)
    assert m.mae == pytest.approx((m.mae_price - entry) / entry)
    # The corrupted writer stored 0.00320157 here; the raw excursion is ~3.265%.
    assert m.mfe == pytest.approx(0.0326495, abs=1e-6)
    assert m.mae == pytest.approx(-0.0407001, abs=1e-6)


def test_small_favorable_move_is_not_swallowed():
    """A +0.1% favorable move must register (old writer floored it to 0.

    The sized/fee-netted writer reported mfe=0 whenever
    size * move < fee + slippage, erasing real favorable excursion).
    """
    tracker = MFEMAETracker()
    now = datetime.now(UTC)

    m = tracker.update_position_metrics("p4", 100.0, 100.1, "long", now)
    assert m.mfe == pytest.approx(0.001)
    assert m.mfe_price == 100.1


def test_running_extremes_are_monotonic():
    tracker = MFEMAETracker()
    entry = 100.0
    now = datetime.now(UTC)

    tracker.update_position_metrics("p5", entry, 104.0, "long", now)
    tracker.update_position_metrics("p5", entry, 102.0, "long", now)  # lower peak: no update
    m = tracker.update_position_metrics("p5", entry, 108.0, "long", now)
    assert m.mfe == pytest.approx(0.08)
    assert m.mfe_price == 108.0

    tracker.update_position_metrics("p5", entry, 97.0, "long", now)
    m = tracker.update_position_metrics("p5", entry, 98.0, "long", now)  # shallower: no update
    assert m.mae == pytest.approx(-0.03)
    assert m.mae_price == 97.0


def test_mfe_mae_no_movement_extremes():
    tracker = MFEMAETracker()
    entry = 100.0
    now = datetime.now(UTC)

    # No movement
    m = tracker.update_position_metrics("p3", entry, 100.0, "long", now)
    assert m.mfe == 0.0
    assert m.mae == 0.0

    # Extreme favorable
    m = tracker.update_position_metrics("p3", entry, 1000.0, "long", now)
    assert m.mfe == pytest.approx(9.0)
    # Extreme adverse
    m = tracker.update_position_metrics("p3", entry, 1.0, "long", now)
    assert m.mae == pytest.approx(-0.99)


def test_invalid_prices_do_not_corrupt_metrics():
    tracker = MFEMAETracker()
    now = datetime.now(UTC)

    m = tracker.update_position_metrics("p6", 100.0, 110.0, "long", now)
    assert m.mfe == pytest.approx(0.10)

    for bad in (float("nan"), float("inf"), 0.0, -5.0):
        m = tracker.update_position_metrics("p6", 100.0, bad, "long", now)
        assert m.mfe == pytest.approx(0.10)
        assert m.mae == 0.0


def test_calculate_mfe_mae_static():
    mfe, mae = MFEMAETracker.calculate_mfe_mae(100.0, 110.0, "long")
    assert mfe == pytest.approx(0.10)
    assert mae == 0.0

    mfe, mae = MFEMAETracker.calculate_mfe_mae(100.0, 110.0, "short")
    assert mfe == 0.0
    assert mae == pytest.approx(-0.10)

    assert MFEMAETracker.calculate_mfe_mae(0.0, 110.0, "long") == (0.0, 0.0)
    assert MFEMAETracker.calculate_mfe_mae(100.0, float("nan"), "long") == (0.0, 0.0)


def test_position_key_type_validation():
    tracker = MFEMAETracker()
    with pytest.raises(TypeError):
        tracker.update_position_metrics(
            position_key=1.5,  # type: ignore[arg-type]
            entry_price=100.0,
            current_price=101.0,
            side="long",
            current_time=datetime.now(UTC),
        )


def test_clear_single_and_all_keys():
    tracker = MFEMAETracker()
    now = datetime.now(UTC)
    tracker.update_position_metrics("a", 100.0, 105.0, "long", now)
    tracker.update_position_metrics("b", 100.0, 105.0, "long", now)

    tracker.clear("a")
    assert tracker.get_position_metrics("a") is None
    assert tracker.get_position_metrics("b") is not None

    tracker.clear()
    assert tracker.get_position_metrics("b") is None
