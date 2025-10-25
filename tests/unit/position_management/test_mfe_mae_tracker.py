from datetime import datetime

import pytest

from src.position_management.mfe_mae_tracker import MFEMAETracker


def test_mfe_mae_tracker_long_case():
    tracker = MFEMAETracker(precision_decimals=8)
    entry = 100.0
    now = datetime.utcnow()

    # Move up +10%
    m = tracker.update_position_metrics(
        position_key="p1",
        entry_price=entry,
        current_price=110.0,
        side="long",
        position_fraction=0.5,
        current_time=now,
    )
    # Sized: 10% move * 0.5 = 0.05
    assert m.mfe == pytest.approx(0.05)
    assert m.mae == 0.0

    # Move down -4% from entry -> MAE -0.02 sized
    m = tracker.update_position_metrics(
        position_key="p1",
        entry_price=entry,
        current_price=96.0,
        side="long",
        position_fraction=0.5,
        current_time=now,
    )
    assert m.mae == pytest.approx(-0.02)
    # MFE remains from the prior +10% move
    assert m.mfe == pytest.approx(0.05)


def test_mfe_mae_tracker_short_case():
    tracker = MFEMAETracker(precision_decimals=8)
    entry = 100.0
    now = datetime.utcnow()

    # Price drops 10% is favorable for short: +10% * 0.3 = +0.03
    m = tracker.update_position_metrics(
        position_key="p2",
        entry_price=entry,
        current_price=90.0,
        side="short",
        position_fraction=0.3,
        current_time=now,
    )
    assert m.mfe == pytest.approx(0.03)
    assert m.mae == 0.0

    # Price rises 5% vs entry is adverse for short: -5% * 0.3 = -0.015
    m = tracker.update_position_metrics(
        position_key="p2",
        entry_price=entry,
        current_price=105.0,
        side="short",
        position_fraction=0.3,
        current_time=now,
    )
    assert m.mae == pytest.approx(-0.015)
    assert m.mfe == pytest.approx(0.03)


def test_mfe_mae_no_movement_extremes():
    tracker = MFEMAETracker()
    entry = 100.0
    now = datetime.utcnow()

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
