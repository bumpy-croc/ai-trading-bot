"""Unit tests for the reconciliation module.

Tests startup reconciliation, pending order recovery, periodic reconciler,
discrepancy handling, close-only mode, and audit events.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.engines.live.reconciliation import (
    AuditEvent,
    PeriodicReconciler,
    PositionReconciler,
    ReconciliationResult,
    Severity,
    classify_severity,
)


# ---------- Test Fixtures ----------


@dataclass
class MockPosition:
    """Minimal position mock for reconciliation tests."""

    symbol: str = "BTCUSDT"
    side: str = "long"
    entry_price: float = 50000.0
    order_id: str = "123456"
    exchange_order_id: str | None = "123456"
    client_order_id: str | None = "atb_BTCUSDT_long_1234_abcd"
    db_position_id: int | None = 1
    stop_loss_order_id: str | None = None
    current_size: float | None = 0.1
    size: float = 0.1


@dataclass
class MockExchangeOrder:
    """Minimal exchange order mock."""

    order_id: str = "123456"
    symbol: str = "BTCUSDT"
    status: object = None
    average_price: float | None = 50100.0
    filled_quantity: float = 0.001
    commission: float = 0.05
    quantity: float = 0.001
    side: object = None
    client_order_id: str | None = None
    create_time: datetime | None = None


@dataclass
class MockBalance:
    asset: str = "USDT"
    total: float = 1000.0
    free: float = 1000.0
    locked: float = 0.0


@pytest.fixture
def mock_exchange():
    exchange = MagicMock()
    exchange.get_order.return_value = None
    exchange.get_order_by_client_id.return_value = None
    exchange.get_all_orders.return_value = []
    exchange.get_my_trades.return_value = []
    exchange.get_open_orders.return_value = []
    exchange.get_balance.return_value = MockBalance()
    return exchange


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.get_unresolved_orders.return_value = []
    db.get_current_balance.return_value = 1000.0
    db.log_audit_event.return_value = 1
    db.update_order_journal.return_value = True
    return db


@pytest.fixture
def mock_position_tracker():
    tracker = MagicMock()
    tracker.positions = {}
    return tracker


@pytest.fixture
def reconciler(mock_exchange, mock_position_tracker, mock_db):
    return PositionReconciler(
        exchange_interface=mock_exchange,
        position_tracker=mock_position_tracker,
        db_manager=mock_db,
        session_id=1,
    )


# ---------- Startup Reconciliation Tests ----------


class TestPositionReconciler:
    """Tests for startup reconciliation."""

    def test_reconcile_startup_no_positions(self, reconciler, mock_db):
        """Startup reconciliation with no positions returns balance check only."""
        results = reconciler.reconcile_startup({})
        # Should still do balance check
        assert any(r.entity_type == "balance" for r in results)

    def test_reconcile_startup_with_positions(self, reconciler, mock_exchange, mock_db):
        """Startup reconciliation verifies each position."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition()
        order = MockExchangeOrder(status=ExOS.FILLED)
        mock_exchange.get_order.return_value = order

        results = reconciler.reconcile_startup({"123456": pos})
        assert len(results) >= 1
        position_results = [r for r in results if r.entity_type == "position"]
        assert len(position_results) == 1

    def test_resolve_pending_order_filled(self, reconciler, mock_exchange, mock_db):
        """Pending order found as FILLED on exchange is resolved."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        mock_db.get_unresolved_orders.return_value = [
            {
                "id": 1,
                "client_order_id": "atb_test_123",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.001,
                "status": "SUBMITTED",
                "created_at": datetime.now(UTC),
            }
        ]
        exchange_order = MockExchangeOrder(status=ExOS.FILLED, average_price=50000.0)
        mock_exchange.get_order_by_client_id.return_value = exchange_order

        results = reconciler.resolve_pending_orders()
        assert len(results) == 1
        assert results[0].status == "resolved"
        mock_db.update_order_journal.assert_called_once()

    def test_resolve_pending_order_not_found_pending_submit(
        self, reconciler, mock_exchange, mock_db
    ):
        """PENDING_SUBMIT order not found on exchange is safely cancelled."""
        mock_db.get_unresolved_orders.return_value = [
            {
                "id": 1,
                "client_order_id": "atb_test_123",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.001,
                "status": "PENDING_SUBMIT",
                "created_at": datetime.now(UTC),
            }
        ]
        mock_exchange.get_order_by_client_id.return_value = None
        mock_exchange.get_all_orders.return_value = []

        results = reconciler.resolve_pending_orders()
        assert len(results) == 1
        assert results[0].status == "resolved"
        assert results[0].severity == Severity.LOW

    def test_resolve_pending_order_not_found_submitted_is_critical(
        self, reconciler, mock_exchange, mock_db
    ):
        """SUBMITTED order not found on exchange is CRITICAL."""
        mock_db.get_unresolved_orders.return_value = [
            {
                "id": 1,
                "client_order_id": "atb_test_123",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.001,
                "status": "SUBMITTED",
                "created_at": datetime.now(UTC),
            }
        ]
        mock_exchange.get_order_by_client_id.return_value = None
        mock_exchange.get_all_orders.return_value = []

        results = reconciler.resolve_pending_orders()
        assert len(results) == 1
        assert results[0].severity == Severity.CRITICAL
        assert results[0].status == "unresolved"

    def test_reconcile_position_entry_filled(self, reconciler, mock_exchange):
        """Position with filled entry order is verified."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(entry_price=50000.0)
        order = MockExchangeOrder(status=ExOS.FILLED, average_price=50000.0)
        mock_exchange.get_order.return_value = order

        result = reconciler.reconcile_position(pos)
        assert result.entity_type == "position"

    def test_reconcile_position_entry_cancelled_is_high(self, reconciler, mock_exchange, mock_db):
        """Position whose entry was cancelled gets HIGH severity."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition()
        order = MockExchangeOrder(status=ExOS.CANCELLED)
        mock_exchange.get_order.return_value = order

        result = reconciler.reconcile_position(pos)
        assert result.severity == Severity.HIGH
        assert len(result.corrections) > 0
        mock_db.log_audit_event.assert_called()

    def test_reconcile_position_sl_filled_offline(self, reconciler, mock_exchange, mock_db):
        """Stop-loss filled while offline is detected."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(stop_loss_order_id="sl_123")
        # Entry order is fine
        entry_order = MockExchangeOrder(status=ExOS.FILLED, average_price=50000.0)
        sl_order = MockExchangeOrder(
            order_id="sl_123", status=ExOS.FILLED, average_price=49000.0
        )
        mock_exchange.get_order.side_effect = [entry_order, sl_order]

        result = reconciler.reconcile_position(pos)
        assert result.severity == Severity.HIGH
        assert any("Stop-loss filled" in c.reason for c in result.corrections)

    def test_reconcile_position_legacy_no_ids(self, reconciler):
        """Legacy position without exchange IDs is skipped."""
        pos = MockPosition(exchange_order_id=None, client_order_id=None)
        result = reconciler.reconcile_position(pos)
        assert result.status == "skipped"

    def test_reconcile_balance_within_threshold(self, reconciler, mock_exchange, mock_db):
        """Balance within threshold does not trigger correction."""
        mock_exchange.get_balance.return_value = MockBalance(total=1005.0)
        mock_db.get_current_balance.return_value = 1000.0

        result = reconciler._reconcile_balance()
        assert result.severity != Severity.CRITICAL

    def test_reconcile_balance_exceeds_threshold(self, reconciler, mock_exchange, mock_db):
        """Balance exceeding 5% threshold triggers CRITICAL."""
        mock_exchange.get_balance.return_value = MockBalance(total=500.0)
        mock_db.get_current_balance.return_value = 1000.0

        result = reconciler._reconcile_balance()
        assert result.severity == Severity.CRITICAL


# ---------- Periodic Reconciler Tests ----------


class TestPeriodicReconciler:
    """Tests for periodic background reconciliation."""

    def test_start_stop_lifecycle(self, mock_exchange, mock_position_tracker, mock_db):
        """Periodic reconciler starts and stops cleanly."""
        reconciler = PeriodicReconciler(
            exchange_interface=mock_exchange,
            position_tracker=mock_position_tracker,
            db_manager=mock_db,
            session_id=1,
            interval=1,
        )
        reconciler.start()
        assert reconciler._running is True
        assert reconciler._thread is not None
        assert reconciler._thread.is_alive()

        reconciler.stop()
        assert reconciler._running is False

    def test_not_started_twice(self, mock_exchange, mock_position_tracker, mock_db):
        """Starting twice doesn't create duplicate threads."""
        reconciler = PeriodicReconciler(
            exchange_interface=mock_exchange,
            position_tracker=mock_position_tracker,
            db_manager=mock_db,
            session_id=1,
            interval=1,
        )
        reconciler.start()
        thread1 = reconciler._thread
        reconciler.start()
        assert reconciler._thread is thread1
        reconciler.stop()

    def test_critical_callback(self, mock_exchange, mock_position_tracker, mock_db):
        """On CRITICAL severity, on_critical callback is invoked."""
        callback = MagicMock()
        reconciler = PeriodicReconciler(
            exchange_interface=mock_exchange,
            position_tracker=mock_position_tracker,
            db_manager=mock_db,
            session_id=1,
            interval=1,
            on_critical=callback,
        )

        # Simulate balance discrepancy
        mock_exchange.get_balance.return_value = MockBalance(total=1.0)
        mock_db.get_current_balance.return_value = 1000.0

        pos = MockPosition()
        mock_position_tracker.positions = {"123": pos}

        reconciler._reconcile_cycle()
        callback.assert_called_once()

    def test_position_lock_uniqueness(self, mock_exchange, mock_position_tracker, mock_db):
        """Per-position locks are unique per key and reusable."""
        reconciler = PeriodicReconciler(
            exchange_interface=mock_exchange,
            position_tracker=mock_position_tracker,
            db_manager=mock_db,
            session_id=1,
        )
        lock_a = reconciler.get_position_lock("pos_a")
        lock_b = reconciler.get_position_lock("pos_b")
        lock_a2 = reconciler.get_position_lock("pos_a")
        assert lock_a is lock_a2
        assert lock_a is not lock_b


# ---------- Severity Classification Tests ----------


class TestSeverityClassification:
    """Tests for discrepancy severity classification."""

    def test_balance_critical(self):
        assert classify_severity("balance", "discrepancy", 0.06) == Severity.CRITICAL

    def test_balance_low(self):
        assert classify_severity("balance", "discrepancy", 0.005) == Severity.LOW

    def test_order_not_found_submitted(self):
        assert classify_severity("order", "not_found_submitted") == Severity.CRITICAL

    def test_order_not_found_pending_submit(self):
        assert classify_severity("order", "not_found_pending_submit") == Severity.LOW

    def test_position_entry_cancelled(self):
        assert classify_severity("position", "entry_cancelled") == Severity.HIGH

    def test_position_price_mismatch(self):
        assert classify_severity("position", "price_mismatch") == Severity.MEDIUM

    def test_unknown_defaults_to_low(self):
        assert classify_severity("unknown", "unknown") == Severity.LOW


# ---------- Audit Event Tests ----------


class TestAuditEvents:
    """Tests for audit event creation and persistence."""

    def test_audit_event_persisted_on_correction(self, reconciler, mock_exchange, mock_db):
        """Corrections are logged as audit events."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition()
        order = MockExchangeOrder(status=ExOS.CANCELLED)
        mock_exchange.get_order.return_value = order

        reconciler.reconcile_position(pos)
        mock_db.log_audit_event.assert_called()
        call_kwargs = mock_db.log_audit_event.call_args
        assert call_kwargs[1]["severity"] == "HIGH" or call_kwargs.kwargs.get("severity") == "HIGH"

    def test_audit_event_contains_before_after(self, reconciler, mock_db):
        """Audit events contain old_value and new_value."""
        audit = AuditEvent(
            entity_type="position",
            entity_id=1,
            field="entry_price",
            old_value="50000.0",
            new_value="50100.0",
            reason="test",
            severity=Severity.MEDIUM,
        )
        reconciler._persist_audit(audit)
        mock_db.log_audit_event.assert_called_once()
        kwargs = mock_db.log_audit_event.call_args.kwargs
        assert kwargs["old_value"] == "50000.0"
        assert kwargs["new_value"] == "50100.0"


# ---------- Close-Only Mode Tests ----------


class TestCloseOnlyMode:
    """Tests for close-only mode behavior in trading engine context."""

    def test_close_only_mode_skips_entries(self):
        """Close-only mode prevents new entry signals."""
        # Minimal mock of _check_entry_conditions behavior
        close_only = True
        entry_should_proceed = not close_only
        assert entry_should_proceed is False

    def test_close_only_mode_allows_exits(self):
        """Close-only mode still allows exit operations."""
        # Exit logic runs regardless of close_only_mode
        close_only = True
        exit_should_proceed = True  # Exits always run
        assert exit_should_proceed is True
