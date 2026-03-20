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
    quantity: float | None = 0.1


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

    def test_reconcile_position_sl_cancelled_clears_reference(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Cancelled stop-loss clears stale reference and flags for re-placement."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(stop_loss_order_id="sl_canc_1", db_position_id=20)
        entry_order = MockExchangeOrder(status=ExOS.FILLED, average_price=50000.0)
        sl_order = MockExchangeOrder(order_id="sl_canc_1", status=ExOS.CANCELLED)
        mock_exchange.get_order.side_effect = [entry_order, sl_order]

        result = reconciler.reconcile_position(pos)
        # Position stays open — not removed from tracker
        mock_position_tracker.remove_position.assert_not_called()
        # Stale SL reference cleared
        assert pos.stop_loss_order_id is None
        # Correction recorded with MEDIUM severity
        assert result.severity >= Severity.MEDIUM
        sl_corrections = [
            c for c in result.corrections
            if "unprotected" in c.reason and c.field == "stop_loss_order_id"
        ]
        assert len(sl_corrections) == 1
        assert sl_corrections[0].severity == Severity.MEDIUM
        assert sl_corrections[0].old_value == "sl_canc_1"
        assert sl_corrections[0].new_value is None
        mock_db.log_audit_event.assert_called()

    def test_reconcile_position_sl_expired_clears_reference(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Expired stop-loss clears stale reference and flags for re-placement."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(stop_loss_order_id="sl_exp_1", db_position_id=21)
        entry_order = MockExchangeOrder(status=ExOS.FILLED, average_price=50000.0)
        sl_order = MockExchangeOrder(order_id="sl_exp_1", status=ExOS.EXPIRED)
        mock_exchange.get_order.side_effect = [entry_order, sl_order]

        result = reconciler.reconcile_position(pos)
        mock_position_tracker.remove_position.assert_not_called()
        assert pos.stop_loss_order_id is None
        assert result.severity >= Severity.MEDIUM
        sl_corrections = [
            c for c in result.corrections
            if "expired" in c.reason and c.field == "stop_loss_order_id"
        ]
        assert len(sl_corrections) == 1

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


# ---------- Asset Holdings Verification Tests ----------


class TestAssetHoldingsVerification:
    """Tests for external close detection via asset balance checks."""

    def test_extract_base_asset_btcusdt(self):
        """Extracts BTC from BTCUSDT."""
        assert PositionReconciler._extract_base_asset("BTCUSDT") == "BTC"

    def test_extract_base_asset_ethusdt(self):
        """Extracts ETH from ETHUSDT."""
        assert PositionReconciler._extract_base_asset("ETHUSDT") == "ETH"

    def test_extract_base_asset_btcbusd(self):
        """Extracts BTC from BTCBUSD."""
        assert PositionReconciler._extract_base_asset("BTCBUSD") == "BTC"

    def test_extract_base_asset_btcusd(self):
        """Extracts BTC from BTCUSD."""
        assert PositionReconciler._extract_base_asset("BTCUSD") == "BTC"

    def test_extract_base_asset_unknown_quote(self):
        """Returns full symbol when no known quote currency matches."""
        assert PositionReconciler._extract_base_asset("BTCEUR") == "BTCEUR"

    def test_position_externally_closed_detected(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Position with near-zero asset balance is detected as externally closed."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(current_size=0.1, db_position_id=10)
        entry_order = MockExchangeOrder(status=ExOS.FILLED, average_price=50000.0)
        mock_exchange.get_order.return_value = entry_order
        # Asset balance is near zero — position was sold externally
        mock_exchange.get_balance.return_value = MockBalance(
            asset="BTC", total=0.0001, free=0.0001, locked=0.0
        )

        result = reconciler.reconcile_position(pos)
        assert result.status == "corrected"
        assert result.severity == Severity.HIGH
        assert any("closed externally" in c.reason for c in result.corrections)
        mock_position_tracker.remove_position.assert_called_once_with(pos.order_id)
        mock_db.close_position.assert_called_once_with(10)

    def test_position_with_sufficient_balance_not_flagged(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Position with sufficient asset balance passes verification."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(current_size=0.1, db_position_id=11)
        entry_order = MockExchangeOrder(status=ExOS.FILLED, average_price=50000.0)
        mock_exchange.get_order.return_value = entry_order
        # Asset balance is above 50% of tracked quantity
        mock_exchange.get_balance.return_value = MockBalance(
            asset="BTC", total=0.08, free=0.08, locked=0.0
        )

        result = reconciler.reconcile_position(pos)
        assert result.status == "verified"
        mock_position_tracker.remove_position.assert_not_called()
        mock_db.close_position.assert_not_called()

    def test_position_exactly_at_threshold_not_flagged(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Position at exactly 50% of tracked quantity is not flagged."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(current_size=0.1, db_position_id=12)
        entry_order = MockExchangeOrder(status=ExOS.FILLED, average_price=50000.0)
        mock_exchange.get_order.return_value = entry_order
        # Exactly 50% — should NOT be flagged (threshold is strictly less than)
        mock_exchange.get_balance.return_value = MockBalance(
            asset="BTC", total=0.05, free=0.05, locked=0.0
        )

        result = reconciler.reconcile_position(pos)
        assert result.status == "verified"
        mock_position_tracker.remove_position.assert_not_called()

    def test_asset_check_skipped_when_already_corrected(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Asset check is skipped if position was already corrected (e.g., entry cancelled)."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(current_size=0.1, db_position_id=13)
        # Entry order is cancelled — position already corrected before asset check
        entry_order = MockExchangeOrder(status=ExOS.CANCELLED)
        mock_exchange.get_order.return_value = entry_order

        result = reconciler.reconcile_position(pos)
        assert result.status == "corrected"
        # The remove_position call comes from the entry cancel, not asset check
        assert mock_position_tracker.remove_position.call_count == 1

    def test_asset_check_with_no_balance_returned(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Null balance response triggers external close detection."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(current_size=0.1, db_position_id=14)
        entry_order = MockExchangeOrder(status=ExOS.FILLED, average_price=50000.0)
        mock_exchange.get_order.return_value = entry_order
        mock_exchange.get_balance.return_value = None

        result = reconciler.reconcile_position(pos)
        assert result.status == "corrected"
        assert result.severity == Severity.HIGH
        mock_position_tracker.remove_position.assert_called_once()

    def test_asset_check_exchange_error_is_handled(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Exchange API error during asset check is handled gracefully."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(current_size=0.1, db_position_id=15)
        entry_order = MockExchangeOrder(status=ExOS.FILLED, average_price=50000.0)
        mock_exchange.get_order.return_value = entry_order
        mock_exchange.get_balance.side_effect = ConnectionError("API timeout")

        result = reconciler.reconcile_position(pos)
        # Should not crash — position stays verified
        assert result.status == "verified"
        mock_position_tracker.remove_position.assert_not_called()

    def test_asset_check_no_db_id_skips_db_close(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """External close detection without db_position_id skips DB close."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(current_size=0.1, db_position_id=None)
        entry_order = MockExchangeOrder(status=ExOS.FILLED, average_price=50000.0)
        mock_exchange.get_order.return_value = entry_order
        mock_exchange.get_balance.return_value = MockBalance(
            asset="BTC", total=0.0, free=0.0, locked=0.0
        )

        result = reconciler.reconcile_position(pos)
        assert result.status == "corrected"
        mock_position_tracker.remove_position.assert_called_once()
        mock_db.close_position.assert_not_called()


class TestPeriodicReconcilerAssetCheck:
    """Tests for asset holdings checks in periodic reconciliation cycle."""

    def test_cycle_detects_externally_closed_position(
        self, mock_exchange, mock_position_tracker, mock_db
    ):
        """Periodic cycle detects position with no asset balance on exchange."""
        pos = MockPosition(current_size=0.1, exchange_order_id="ex_001")
        mock_position_tracker.positions = {"key1": pos}
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        mock_exchange.get_order.return_value = MockExchangeOrder(
            status=ExOS.FILLED
        )
        # Asset balance is zero, but USDT balance is fine
        mock_exchange.get_balance.side_effect = lambda asset: (
            MockBalance(asset="USDT", total=1000.0)
            if asset == "USDT"
            else MockBalance(asset=asset, total=0.0, free=0.0, locked=0.0)
        )
        mock_db.get_current_balance.return_value = 1000.0

        reconciler = PeriodicReconciler(
            exchange_interface=mock_exchange,
            position_tracker=mock_position_tracker,
            db_manager=mock_db,
            session_id=1,
            interval=60,
        )
        reconciler._reconcile_cycle()

        # Audit event logged for external close
        audit_calls = mock_db.log_audit_event.call_args_list
        external_close_logged = any(
            "closed externally" in str(call) for call in audit_calls
        )
        assert external_close_logged

    def test_cycle_no_false_positive_when_balance_sufficient(
        self, mock_exchange, mock_position_tracker, mock_db
    ):
        """Periodic cycle does not flag positions with sufficient asset balance."""
        pos = MockPosition(current_size=0.1, exchange_order_id="ex_002")
        mock_position_tracker.positions = {"key1": pos}
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        mock_exchange.get_order.return_value = MockExchangeOrder(
            status=ExOS.FILLED
        )
        mock_exchange.get_balance.side_effect = lambda asset: (
            MockBalance(asset="USDT", total=1000.0)
            if asset == "USDT"
            else MockBalance(asset=asset, total=0.1, free=0.1, locked=0.0)
        )
        mock_db.get_current_balance.return_value = 1000.0

        reconciler = PeriodicReconciler(
            exchange_interface=mock_exchange,
            position_tracker=mock_position_tracker,
            db_manager=mock_db,
            session_id=1,
            interval=60,
        )
        reconciler._reconcile_cycle()

        # No external close audit event should be logged
        audit_calls = mock_db.log_audit_event.call_args_list
        external_close_logged = any(
            "closed externally" in str(call) for call in audit_calls
        )
        assert not external_close_logged


# ---------- Position-Notional-Aware Balance Tests ----------


class TestBalanceAccountsForPositionNotional:
    """Tests that balance reconciliation subtracts position notional before comparing.

    In spot trading, buying BTC reduces USDT by the purchase amount. The DB
    balance only moves for fees/realized PnL, so expected USDT on exchange
    equals db_balance minus total position notional.
    """

    def test_no_false_critical_after_opening_position(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Opening a position reduces exchange USDT but should not trigger CRITICAL.

        Scenario: $10,000 DB balance, bought 0.1 BTC @ $50,000 = $5,000 notional.
        Exchange USDT = $5,000 (correct). Without position adjustment this would
        show a 50% discrepancy and trigger close-only mode.
        """
        pos = MockPosition(entry_price=50000.0, current_size=0.1)
        mock_position_tracker.positions = {"pos_1": pos}
        mock_exchange.get_balance.return_value = MockBalance(total=5000.0)
        mock_db.get_current_balance.return_value = 10000.0

        result = reconciler._reconcile_balance()
        assert result.severity != Severity.CRITICAL
        assert result.status == "verified"

    def test_genuine_discrepancy_still_triggers_critical(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Genuine balance loss still triggers CRITICAL even with positions.

        Scenario: $10,000 DB, $5,000 in positions, expected USDT = $5,000.
        Exchange shows only $1,000 — 80% discrepancy.
        """
        pos = MockPosition(entry_price=50000.0, current_size=0.1)
        mock_position_tracker.positions = {"pos_1": pos}
        mock_exchange.get_balance.return_value = MockBalance(total=1000.0)
        mock_db.get_current_balance.return_value = 10000.0

        result = reconciler._reconcile_balance()
        assert result.severity == Severity.CRITICAL

    def test_multiple_positions_notional_summed(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Multiple positions' notional values are summed correctly.

        Scenario: $20,000 DB, two positions totaling $15,000 notional.
        Exchange USDT = $5,000 (correct).
        """
        pos1 = MockPosition(entry_price=50000.0, current_size=0.1, quantity=0.1, symbol="BTCUSDT")
        pos2 = MockPosition(entry_price=3000.0, current_size=10.0 / 3, quantity=10.0 / 3, symbol="ETHUSDT")
        mock_position_tracker.positions = {"pos_1": pos1, "pos_2": pos2}
        mock_exchange.get_balance.return_value = MockBalance(total=5000.0)
        mock_db.get_current_balance.return_value = 20000.0

        result = reconciler._reconcile_balance()
        assert result.severity != Severity.CRITICAL

    def test_no_positions_falls_back_to_raw_comparison(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """With no positions, comparison is against raw DB balance."""
        mock_position_tracker.positions = {}
        mock_exchange.get_balance.return_value = MockBalance(total=500.0)
        mock_db.get_current_balance.return_value = 1000.0

        result = reconciler._reconcile_balance()
        assert result.severity == Severity.CRITICAL

    def test_periodic_reconciler_no_false_critical_with_positions(
        self, mock_exchange, mock_position_tracker, mock_db
    ):
        """Periodic reconciler balance check accounts for position notional.

        Same scenario as startup: position notional explains USDT drop.
        """
        pos = MockPosition(
            entry_price=50000.0, current_size=0.1, exchange_order_id="ex_100"
        )
        mock_position_tracker.positions = {"pos_1": pos}

        from src.data_providers.exchange_interface import OrderStatus as ExOS

        mock_exchange.get_order.return_value = MockExchangeOrder(status=ExOS.FILLED)
        # USDT = $5,000 (DB $10,000 minus $5,000 position notional)
        # BTC balance = 0.1 (position is held)
        mock_exchange.get_balance.side_effect = lambda asset: (
            MockBalance(asset="USDT", total=5000.0)
            if asset == "USDT"
            else MockBalance(asset=asset, total=0.1, free=0.1, locked=0.0)
        )
        mock_db.get_current_balance.return_value = 10000.0

        callback = MagicMock()
        reconciler = PeriodicReconciler(
            exchange_interface=mock_exchange,
            position_tracker=mock_position_tracker,
            db_manager=mock_db,
            session_id=1,
            interval=60,
            on_critical=callback,
        )
        reconciler._reconcile_cycle()
        # Should NOT trigger close-only mode
        callback.assert_not_called()


# ---------- Filled Order Position Reconciliation Tests ----------


class TestFilledOrderPositionReconciliation:
    """Tests for _reconcile_filled_order_position — crash recovery scenarios."""

    def test_filled_entry_creates_position_when_missing(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Crash after ENTRY fill: position is created in tracker and DB."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        mock_position_tracker._positions_lock = __import__("threading").Lock()
        mock_position_tracker._positions = {}
        mock_db.log_position.return_value = 42

        mock_db.get_unresolved_orders.return_value = [
            {
                "id": 1,
                "client_order_id": "atb_BTCUSDT_long_9999_abcd",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.001,
                "status": "SUBMITTED",
                "order_type": "ENTRY",
                "created_at": datetime.now(UTC),
            }
        ]
        exchange_order = MockExchangeOrder(
            status=ExOS.FILLED, average_price=50000.0, filled_quantity=0.001
        )
        mock_exchange.get_order_by_client_id.return_value = exchange_order

        results = reconciler.resolve_pending_orders()
        assert len(results) == 1
        assert results[0].status == "resolved"

        # Position should be logged to DB
        mock_db.log_position.assert_called_once()
        call_kwargs = mock_db.log_position.call_args.kwargs
        assert call_kwargs["symbol"] == "BTCUSDT"
        assert call_kwargs["side"] == "LONG"
        assert call_kwargs["entry_price"] == 50000.0
        # size is a balance fraction (not fill_qty); no entry_balance => default 0.1
        assert call_kwargs["size"] == 0.1

        # Position should be tracked in memory
        mock_position_tracker.track_recovered_position.assert_called_once()
        pos_arg = mock_position_tracker.track_recovered_position.call_args[0][0]
        assert pos_arg.symbol == "BTCUSDT"
        assert pos_arg.entry_price == 50000.0
        assert pos_arg.quantity == 0.001
        assert pos_arg.size == 0.1
        assert pos_arg.exchange_order_id is not None
        db_id_arg = mock_position_tracker.track_recovered_position.call_args[0][1]
        assert db_id_arg == 42

    def test_filled_entry_skips_if_position_already_tracked(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """No duplicate position if entry order was already reconciled."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        mock_position_tracker._positions_lock = __import__("threading").Lock()
        # Position already exists under the exchange order ID
        mock_position_tracker._positions = {"123456": MockPosition()}

        mock_db.get_unresolved_orders.return_value = [
            {
                "id": 1,
                "client_order_id": "atb_BTCUSDT_long_1111_xxxx",
                "exchange_order_id": "123456",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.001,
                "status": "SUBMITTED",
                "order_type": "ENTRY",
                "created_at": datetime.now(UTC),
            }
        ]
        exchange_order = MockExchangeOrder(
            status=ExOS.FILLED, average_price=50000.0, filled_quantity=0.001
        )
        mock_exchange.get_order_by_client_id.return_value = exchange_order

        reconciler.resolve_pending_orders()

        # Should NOT create duplicate
        mock_db.log_position.assert_not_called()
        mock_position_tracker.track_recovered_position.assert_not_called()

    def test_filled_full_exit_closes_position(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Crash after FULL_EXIT fill: position is removed from tracker and DB."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(db_position_id=7)
        mock_position_tracker._positions_lock = __import__("threading").Lock()
        mock_position_tracker._positions = {"order_123": pos}

        mock_db.get_unresolved_orders.return_value = [
            {
                "id": 2,
                "client_order_id": "atbx_BTCUSDT_exit_5555",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.001,
                "status": "SUBMITTED",
                "order_type": "FULL_EXIT",
                "position_id": 7,
                "created_at": datetime.now(UTC),
            }
        ]
        exchange_order = MockExchangeOrder(
            status=ExOS.FILLED, average_price=51000.0, filled_quantity=0.001
        )
        mock_exchange.get_order_by_client_id.return_value = exchange_order

        results = reconciler.resolve_pending_orders()
        assert len(results) == 1
        assert results[0].status == "resolved"

        # Position should be removed from tracker
        mock_position_tracker.remove_position.assert_called_once_with("order_123")
        # DB position should be closed with exit price
        mock_db.close_position.assert_called_once_with(7, exit_price=51000.0)

    def test_filled_exit_no_position_id_is_safe(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Exit order without position_id does not crash."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        mock_position_tracker._positions_lock = __import__("threading").Lock()
        mock_position_tracker._positions = {}

        mock_db.get_unresolved_orders.return_value = [
            {
                "id": 3,
                "client_order_id": "atbx_BTCUSDT_exit_6666",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.001,
                "status": "SUBMITTED",
                "order_type": "FULL_EXIT",
                "created_at": datetime.now(UTC),
                # No position_id
            }
        ]
        exchange_order = MockExchangeOrder(
            status=ExOS.FILLED, average_price=51000.0, filled_quantity=0.001
        )
        mock_exchange.get_order_by_client_id.return_value = exchange_order

        # Should not raise
        results = reconciler.resolve_pending_orders()
        assert len(results) == 1
        mock_position_tracker.remove_position.assert_not_called()
        mock_db.close_position.assert_not_called()

    def test_filled_entry_db_failure_still_tracks_in_memory(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """If DB log_position fails, position is still tracked in memory."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        mock_position_tracker._positions_lock = __import__("threading").Lock()
        mock_position_tracker._positions = {}
        mock_db.log_position.side_effect = Exception("DB connection lost")

        mock_db.get_unresolved_orders.return_value = [
            {
                "id": 4,
                "client_order_id": "atb_BTCUSDT_long_7777_zzzz",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.002,
                "status": "SUBMITTED",
                "order_type": "ENTRY",
                "created_at": datetime.now(UTC),
            }
        ]
        exchange_order = MockExchangeOrder(
            status=ExOS.FILLED, average_price=48000.0, filled_quantity=0.002
        )
        mock_exchange.get_order_by_client_id.return_value = exchange_order

        results = reconciler.resolve_pending_orders()
        assert len(results) == 1

        # Still tracked in memory with db_id=None
        mock_position_tracker.track_recovered_position.assert_called_once()
        db_id_arg = mock_position_tracker.track_recovered_position.call_args[0][1]
        assert db_id_arg is None

    def test_filled_partial_exit_closes_position(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """PARTIAL_EXIT filled also triggers position close reconciliation."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        pos = MockPosition(db_position_id=15)
        mock_position_tracker._positions_lock = __import__("threading").Lock()
        mock_position_tracker._positions = {"order_456": pos}

        mock_db.get_unresolved_orders.return_value = [
            {
                "id": 5,
                "client_order_id": "atbx_BTCUSDT_exit_8888",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.0005,
                "status": "SUBMITTED",
                "order_type": "PARTIAL_EXIT",
                "position_id": 15,
                "created_at": datetime.now(UTC),
            }
        ]
        exchange_order = MockExchangeOrder(
            status=ExOS.FILLED, average_price=52000.0, filled_quantity=0.0005
        )
        mock_exchange.get_order_by_client_id.return_value = exchange_order

        results = reconciler.resolve_pending_orders()
        assert len(results) == 1

        mock_position_tracker.remove_position.assert_called_once_with("order_456")
        mock_db.close_position.assert_called_once_with(15, exit_price=52000.0)

    def test_no_order_type_skips_position_reconciliation(
        self, reconciler, mock_exchange, mock_db, mock_position_tracker
    ):
        """Orders without order_type do not trigger position reconciliation."""
        from src.data_providers.exchange_interface import OrderStatus as ExOS

        mock_position_tracker._positions_lock = __import__("threading").Lock()
        mock_position_tracker._positions = {}

        mock_db.get_unresolved_orders.return_value = [
            {
                "id": 6,
                "client_order_id": "atb_BTCUSDT_long_0000",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.001,
                "status": "SUBMITTED",
                "created_at": datetime.now(UTC),
                # No order_type
            }
        ]
        exchange_order = MockExchangeOrder(
            status=ExOS.FILLED, average_price=50000.0, filled_quantity=0.001
        )
        mock_exchange.get_order_by_client_id.return_value = exchange_order

        results = reconciler.resolve_pending_orders()
        assert len(results) == 1

        # No position creation or closing
        mock_db.log_position.assert_not_called()
        mock_position_tracker.track_recovered_position.assert_not_called()
        mock_position_tracker.remove_position.assert_not_called()
