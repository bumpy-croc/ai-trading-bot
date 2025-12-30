from datetime import UTC, datetime

import pytest

from src.position_management.mfe_mae_tracker import MFEMAETracker


def test_mfe_mae_tracker_long_case():
    tracker = MFEMAETracker(precision_decimals=8)
    entry = 100.0
    now = datetime.now(UTC)

    # Move up +10%
    m = tracker.update_position_metrics(
        position_key="p1",
        entry_price=entry,
        current_price=110.0,
        side="long",
        position_fraction=0.5,
        current_time=now,
    )
    # Sized: 10% move * 0.5 = 0.05, minus exit costs (0.001 + 0.0005) = 0.0485
    assert m.mfe == pytest.approx(0.0485)
    assert m.mae == 0.0

    # Move down -4% from entry -> MAE -0.02 sized, minus exit costs = -0.0215
    m = tracker.update_position_metrics(
        position_key="p1",
        entry_price=entry,
        current_price=96.0,
        side="long",
        position_fraction=0.5,
        current_time=now,
    )
    assert m.mae == pytest.approx(-0.0215)
    # MFE remains from the prior +10% move (net of exit costs)
    assert m.mfe == pytest.approx(0.0485)


def test_mfe_mae_tracker_short_case():
    tracker = MFEMAETracker(precision_decimals=8)
    entry = 100.0
    now = datetime.now(UTC)

    # Price drops 10% is favorable for short: +10% * 0.3 = 0.03, minus exit costs = 0.0285
    m = tracker.update_position_metrics(
        position_key="p2",
        entry_price=entry,
        current_price=90.0,
        side="short",
        position_fraction=0.3,
        current_time=now,
    )
    assert m.mfe == pytest.approx(0.0285)
    assert m.mae == 0.0

    # Price rises 5% vs entry is adverse for short: -5% * 0.3 = -0.015, minus exit costs = -0.0165
    m = tracker.update_position_metrics(
        position_key="p2",
        entry_price=entry,
        current_price=105.0,
        side="short",
        position_fraction=0.3,
        current_time=now,
    )
    assert m.mae == pytest.approx(-0.0165)
    # MFE remains from the prior favorable move (net of exit costs)
    assert m.mfe == pytest.approx(0.0285)


def test_mfe_mae_no_movement_extremes():
    tracker = MFEMAETracker()
    entry = 100.0
    now = datetime.now(UTC)

    # No movement
    m = tracker.update_position_metrics("p3", entry, 100.0, "long", 1.0, now)
    assert m.mfe == 0.0
    assert m.mae == 0.0

    # Extreme favorable
    m = tracker.update_position_metrics("p3", entry, 1000.0, "long", 1.0, now)
    assert m.mfe > 0
    # Extreme adverse
    m = tracker.update_position_metrics("p3", entry, 1.0, "long", 1.0, now)
    assert m.mae < 0
