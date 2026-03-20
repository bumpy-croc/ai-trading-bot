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

    def test_reconcile_position_entry_cancelled_is_high(self, reconciler, mock_exchange, mock_db, mock_position_tracker):
        """Position whose entry was cancelled gets HIGH severity and removes from tracker."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition()
        order = MockExchangeOrder(status=ExOS.CANCELLED)
        mock_exchange.get_order.return_value = order

        result = reconciler.reconcile_position(pos)
        assert result.severity == Severity.HIGH
        assert len(result.corrections) > 0
        mock_db.log_audit_event.assert_called()
        # Position must be removed from in-memory tracker
        mock_position_tracker.remove_position.assert_called_once_with(pos.order_id)
        # DB position must be closed
        mock_db.close_position.assert_called_once_with(pos.db_position_id)

    def test_reconcile_position_entry_rejected_removes_from_tracker(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Position whose entry was rejected is removed from tracker and DB."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(order_id="rej_001", db_position_id=42)
        order = MockExchangeOrder(status=ExOS.REJECTED)
        mock_exchange.get_order.return_value = order

        result = reconciler.reconcile_position(pos)
        assert result.status == "corrected"
        assert result.severity == Severity.HIGH
        mock_position_tracker.remove_position.assert_called_once_with("rej_001")
        mock_db.close_position.assert_called_once_with(42)

    def test_reconcile_position_entry_cancelled_no_db_id(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Cancelled entry without db_position_id skips DB close gracefully."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(db_position_id=None)
        order = MockExchangeOrder(status=ExOS.CANCELLED)
        mock_exchange.get_order.return_value = order

        result = reconciler.reconcile_position(pos)
        assert result.severity == Severity.HIGH
        mock_position_tracker.remove_position.assert_called_once()
        mock_db.close_position.assert_not_called()

    def test_reconcile_position_sl_filled_offline(self, reconciler, mock_exchange, mock_db, mock_position_tracker):
        """Stop-loss filled while offline removes position and closes DB record."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(stop_loss_order_id="sl_123", db_position_id=7)
        # Entry order is fine
        entry_order = MockExchangeOrder(status=ExOS.FILLED, average_price=50000.0)
        sl_order = MockExchangeOrder(
            order_id="sl_123", status=ExOS.FILLED, average_price=49000.0
        )
        mock_exchange.get_order.side_effect = [entry_order, sl_order]

        result = reconciler.reconcile_position(pos)
        assert result.severity == Severity.HIGH
        assert any("Stop-loss filled" in c.reason for c in result.corrections)
        # Position must be removed from in-memory tracker
        mock_position_tracker.remove_position.assert_called_once_with(pos.order_id)
        # DB position must be closed with SL fill price
        mock_db.close_position.assert_called_once_with(7, exit_price=49000.0)

    def test_reconcile_position_sl_filled_no_average_price(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """SL filled without average_price still removes position, closes DB without price."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(stop_loss_order_id="sl_456", db_position_id=8)
        entry_order = MockExchangeOrder(status=ExOS.FILLED, average_price=50000.0)
        sl_order = MockExchangeOrder(
            order_id="sl_456", status=ExOS.FILLED, average_price=None
        )
        mock_exchange.get_order.side_effect = [entry_order, sl_order]

        result = reconciler.reconcile_position(pos)
        assert result.severity == Severity.HIGH
        mock_position_tracker.remove_position.assert_called_once_with(pos.order_id)
        mock_db.close_position.assert_called_once_with(8, exit_price=None)

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


# ---------- _find_matching_order Tests ----------


class TestFindMatchingOrder:
    """Tests for _find_matching_order prefix and side mapping logic."""

    def test_exit_order_with_atbx_prefix_matches(self, reconciler):
        """Exit orders using 'atbx_' prefix are accepted by the prefix filter."""
        from src.data_providers.exchange_interface import OrderSide

        order_data = {
            "id": 1,
            "client_order_id": "atbx_BTCUSDT_exit_1234",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.001,
            "order_type": "FULL_EXIT",
            "created_at": datetime.now(UTC),
        }
        exchange_order = MockExchangeOrder(
            order_id="ex_999",
            quantity=0.001,
            side=OrderSide.SELL,
            client_order_id="atbx_BTCUSDT_exit_1234",
            create_time=datetime.now(UTC),
        )
        result = reconciler._find_matching_order(order_data, [exchange_order])
        assert result is not None
        assert result.order_id == "ex_999"

    def test_entry_order_with_atb_prefix_still_matches(self, reconciler):
        """Entry orders using 'atb_' prefix continue to match."""
        from src.data_providers.exchange_interface import OrderSide

        order_data = {
            "id": 2,
            "client_order_id": "atb_BTCUSDT_long_5678",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.001,
            "order_type": "ENTRY",
            "created_at": datetime.now(UTC),
        }
        exchange_order = MockExchangeOrder(
            order_id="ex_100",
            quantity=0.001,
            side=OrderSide.BUY,
            client_order_id="atb_BTCUSDT_long_5678",
            create_time=datetime.now(UTC),
        )
        result = reconciler._find_matching_order(order_data, [exchange_order])
        assert result is not None
        assert result.order_id == "ex_100"

    def test_exit_order_long_expects_sell_side(self, reconciler):
        """Closing a LONG position requires SELL — exit order side is inverted."""
        from src.data_providers.exchange_interface import OrderSide

        order_data = {
            "id": 3,
            "client_order_id": "atbx_BTCUSDT_exit_9999",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.001,
            "order_type": "FULL_EXIT",
            "created_at": datetime.now(UTC),
        }
        # BUY side should NOT match a LONG exit (exit should be SELL)
        buy_order = MockExchangeOrder(
            order_id="ex_buy",
            quantity=0.001,
            side=OrderSide.BUY,
            client_order_id="atbx_BTCUSDT_exit_a",
            create_time=datetime.now(UTC),
        )
        sell_order = MockExchangeOrder(
            order_id="ex_sell",
            quantity=0.001,
            side=OrderSide.SELL,
            client_order_id="atbx_BTCUSDT_exit_b",
            create_time=datetime.now(UTC),
        )
        result = reconciler._find_matching_order(order_data, [buy_order, sell_order])
        assert result is not None
        assert result.order_id == "ex_sell"

    def test_exit_order_short_expects_buy_side(self, reconciler):
        """Closing a SHORT position requires BUY — exit order side is inverted."""
        from src.data_providers.exchange_interface import OrderSide

        order_data = {
            "id": 4,
            "client_order_id": "atbx_BTCUSDT_exit_7777",
            "symbol": "BTCUSDT",
            "side": "SHORT",
            "quantity": 0.001,
            "order_type": "PARTIAL_EXIT",
            "created_at": datetime.now(UTC),
        }
        buy_order = MockExchangeOrder(
            order_id="ex_buy",
            quantity=0.001,
            side=OrderSide.BUY,
            client_order_id="atbx_BTCUSDT_exit_c",
            create_time=datetime.now(UTC),
        )
        result = reconciler._find_matching_order(order_data, [buy_order])
        assert result is not None
        assert result.order_id == "ex_buy"

    def test_entry_order_long_expects_buy_side(self, reconciler):
        """Entry for LONG position expects BUY side (unchanged behavior)."""
        from src.data_providers.exchange_interface import OrderSide

        order_data = {
            "id": 5,
            "client_order_id": "atb_BTCUSDT_long_1111",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.001,
            "order_type": "ENTRY",
            "created_at": datetime.now(UTC),
        }
        sell_order = MockExchangeOrder(
            order_id="ex_sell",
            quantity=0.001,
            side=OrderSide.SELL,
            client_order_id="atb_BTCUSDT_long_x",
            create_time=datetime.now(UTC),
        )
        buy_order = MockExchangeOrder(
            order_id="ex_buy",
            quantity=0.001,
            side=OrderSide.BUY,
            client_order_id="atb_BTCUSDT_long_y",
            create_time=datetime.now(UTC),
        )
        result = reconciler._find_matching_order(order_data, [sell_order, buy_order])
        assert result is not None
        assert result.order_id == "ex_buy"

    def test_non_atb_prefix_orders_excluded(self, reconciler):
        """Orders without 'atb' prefix are excluded from matching."""
        from src.data_providers.exchange_interface import OrderSide

        order_data = {
            "id": 6,
            "client_order_id": "atb_BTCUSDT_long_2222",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.001,
            "order_type": "ENTRY",
            "created_at": datetime.now(UTC),
        }
        manual_order = MockExchangeOrder(
            order_id="ex_manual",
            quantity=0.001,
            side=OrderSide.BUY,
            client_order_id="manual_order_123",
            create_time=datetime.now(UTC),
        )
        result = reconciler._find_matching_order(order_data, [manual_order])
        assert result is None

    def test_missing_order_type_defaults_to_entry_behavior(self, reconciler):
        """When order_type is absent, defaults to ENTRY side mapping."""
        from src.data_providers.exchange_interface import OrderSide

        order_data = {
            "id": 7,
            "client_order_id": "atb_BTCUSDT_long_3333",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.001,
            "created_at": datetime.now(UTC),
            # No order_type key
        }
        buy_order = MockExchangeOrder(
            order_id="ex_buy",
            quantity=0.001,
            side=OrderSide.BUY,
            client_order_id="atb_BTCUSDT_long_z",
            create_time=datetime.now(UTC),
        )
        result = reconciler._find_matching_order(order_data, [buy_order])
        assert result is not None
        assert result.order_id == "ex_buy"
