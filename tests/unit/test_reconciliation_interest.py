"""Unit tests for margin interest during reconciliation.

Verifies that the periodic reconciler logs accumulated margin interest
for open short positions (informational only — no corrections), and
that _realize_pnl_on_close deducts margin interest from realized PnL
for short positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.engines.live.reconciliation import PeriodicReconciler, PositionReconciler

pytestmark = pytest.mark.fast


# ---------- Fixtures ----------


@dataclass
class _MockBalance:
    asset: str = "USDT"
    total: float = 1000.0
    free: float = 1000.0
    locked: float = 0.0


@dataclass
class _MockPosition:
    symbol: str = "ETHUSDT"
    side: str = "short"
    entry_price: float = 2000.0
    order_id: str = "o1"
    exchange_order_id: str | None = "o1"
    client_order_id: str | None = "atb_ETHUSDT_short_1_abcd"
    db_position_id: int | None = 1
    stop_loss: float | None = None
    stop_loss_order_id: str | None = None
    current_size: float | None = 0.1
    size: float = 0.1
    quantity: float | None = 0.1
    partial_exits_taken: int = 0
    last_partial_exit_price: float | None = None
    original_size: float | None = 0.1
    entry_balance: float | None = None
    entry_time: datetime | None = None


@pytest.fixture
def mock_exchange():
    exchange = MagicMock()
    exchange.get_order.return_value = MagicMock(
        status=MagicMock(__eq__=lambda s, o: True)
    )
    exchange.get_order_by_client_id.return_value = None
    exchange.get_all_orders.return_value = []
    exchange.get_my_trades.return_value = []
    exchange.get_open_orders.return_value = []
    exchange.get_balance.return_value = _MockBalance()
    exchange.get_margin_borrowed.return_value = 0.1
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
def mock_tracker():
    return MagicMock()


def _make_reconciler(exchange, tracker, db, *, use_margin: bool = True):
    return PeriodicReconciler(
        exchange_interface=exchange,
        position_tracker=tracker,
        db_manager=db,
        session_id=1,
        interval=60,
        use_margin=use_margin,
    )


# ---------- Tests ----------


class TestReconciliationInterestLogging:
    """Margin interest logging during reconcile cycle."""

    def test_interest_logged_for_short_positions(
        self, mock_exchange, mock_tracker, mock_db, caplog
    ):
        """Interest accrued on open shorts is logged at INFO level."""
        pos = _MockPosition(
            symbol="ETHUSDT",
            side="short",
            entry_time=datetime(2025, 1, 1, tzinfo=UTC),
        )
        mock_tracker.positions = {"o1": pos}

        reconciler = _make_reconciler(mock_exchange, mock_tracker, mock_db)

        with patch(
            "src.engines.live.reconciliation.MarginInterestTracker"
        ) as MockMIT:
            instance = MockMIT.return_value
            instance.get_position_interest_cost.return_value = 0.0523

            with caplog.at_level(logging.INFO):
                reconciler._reconcile_cycle()

            MockMIT.assert_called_once_with(mock_exchange)
            instance.get_position_interest_cost.assert_called_once_with(
                "ETH", pos.entry_time
            )
            assert any(
                "Margin interest accrued for ETHUSDT" in msg
                and "0.05230000 ETH" in msg
                for msg in caplog.messages
            )

    def test_interest_not_logged_for_long_positions(
        self, mock_exchange, mock_tracker, mock_db, caplog
    ):
        """Long positions are skipped — no interest query."""
        pos = _MockPosition(symbol="BTCUSDT", side="long")
        mock_tracker.positions = {"o1": pos}

        # Long in margin mode: balance check path, not interest
        mock_exchange.get_balance.return_value = _MockBalance(
            asset="BTC", total=0.1
        )

        reconciler = _make_reconciler(mock_exchange, mock_tracker, mock_db)

        with patch(
            "src.engines.live.reconciliation.MarginInterestTracker"
        ) as MockMIT:
            with caplog.at_level(logging.INFO):
                reconciler._reconcile_cycle()

            MockMIT.return_value.get_position_interest_cost.assert_not_called()

    def test_interest_not_logged_when_not_margin_mode(
        self, mock_exchange, mock_tracker, mock_db, caplog
    ):
        """Interest logging is skipped entirely outside margin mode."""
        pos = _MockPosition(symbol="ETHUSDT", side="short")
        mock_tracker.positions = {"o1": pos}

        # Need balance for spot mode asset check
        mock_exchange.get_balance.return_value = _MockBalance(
            asset="ETH", total=0.0
        )

        reconciler = _make_reconciler(
            mock_exchange, mock_tracker, mock_db, use_margin=False
        )

        with patch(
            "src.engines.live.reconciliation.MarginInterestTracker"
        ) as MockMIT:
            with caplog.at_level(logging.INFO):
                reconciler._reconcile_cycle()

            MockMIT.assert_not_called()
            assert not any(
                "Margin interest accrued" in msg for msg in caplog.messages
            )

    def test_interest_error_does_not_crash_reconciliation(
        self, mock_exchange, mock_tracker, mock_db, caplog
    ):
        """Errors during interest query are caught — reconciliation continues."""
        pos = _MockPosition(
            symbol="ETHUSDT",
            side="short",
            entry_time=datetime(2025, 1, 1, tzinfo=UTC),
        )
        mock_tracker.positions = {"o1": pos}

        reconciler = _make_reconciler(mock_exchange, mock_tracker, mock_db)

        with patch(
            "src.engines.live.reconciliation.MarginInterestTracker"
        ) as MockMIT:
            instance = MockMIT.return_value
            instance.get_position_interest_cost.side_effect = RuntimeError(
                "API timeout"
            )

            with caplog.at_level(logging.WARNING):
                # Should NOT raise
                reconciler._reconcile_cycle()

            assert any(
                "interest" in msg.lower() and "ETHUSDT" in msg
                for msg in caplog.messages
            )


def _make_position_reconciler(exchange, tracker, db, *, use_margin: bool = True):
    return PositionReconciler(
        exchange_interface=exchange,
        position_tracker=tracker,
        db_manager=db,
        session_id=1,
        use_margin=use_margin,
    )


class TestRealizePnlOnCloseInterestDeduction:
    """Margin interest deducted from PnL when reconciliation closes a short."""

    def test_interest_deducted_from_short_pnl(
        self, mock_exchange, mock_tracker, mock_db, caplog
    ):
        """Short position PnL is reduced by margin interest cost."""
        pos = _MockPosition(
            symbol="ETHUSDT",
            side="short",
            entry_price=2000.0,
            quantity=0.1,
            current_size=0.1,
            original_size=0.1,
            entry_time=datetime(2025, 1, 1, tzinfo=UTC),
        )
        reconciler = _make_position_reconciler(mock_exchange, mock_tracker, mock_db)

        with patch(
            "src.engines.live.reconciliation.MarginInterestTracker"
        ) as MockMIT:
            instance = MockMIT.return_value
            instance.get_position_interest_cost.return_value = 0.002

            with caplog.at_level(logging.INFO):
                reconciler._realize_pnl_on_close(pos, exit_price=1900.0, reason="test")

            # PnL = (2000 - 1900) * 0.1 = 10.0
            # Interest = 0.002 base * 1900.0 exit = 3.80 USDT
            # Net PnL = 10.0 - 3.80 = 6.20
            mock_db.update_balance.assert_called_once()
            call_args = mock_db.update_balance.call_args
            new_balance = call_args[0][0]
            assert new_balance == pytest.approx(1000.0 + 6.20)

            assert any(
                "Deducted margin interest $3.80" in msg
                for msg in caplog.messages
            )

    def test_no_interest_deduction_for_long(
        self, mock_exchange, mock_tracker, mock_db
    ):
        """Long positions do not have interest deducted."""
        pos = _MockPosition(
            symbol="ETHUSDT",
            side="long",
            entry_price=1900.0,
            quantity=0.1,
            current_size=0.1,
            original_size=0.1,
            entry_time=datetime(2025, 1, 1, tzinfo=UTC),
        )
        reconciler = _make_position_reconciler(mock_exchange, mock_tracker, mock_db)

        with patch(
            "src.engines.live.reconciliation.MarginInterestTracker"
        ) as MockMIT:
            reconciler._realize_pnl_on_close(pos, exit_price=2000.0, reason="test")

            # Should not even instantiate the tracker for a long position
            MockMIT.assert_not_called()

            # PnL = (2000 - 1900) * 0.1 = 10.0
            mock_db.update_balance.assert_called_once()
            new_balance = mock_db.update_balance.call_args[0][0]
            assert new_balance == pytest.approx(1000.0 + 10.0)

    def test_no_interest_deduction_when_not_margin_mode(
        self, mock_exchange, mock_tracker, mock_db
    ):
        """Interest is not deducted when use_margin is False."""
        pos = _MockPosition(
            symbol="ETHUSDT",
            side="short",
            entry_price=2000.0,
            quantity=0.1,
            current_size=0.1,
            original_size=0.1,
            entry_time=datetime(2025, 1, 1, tzinfo=UTC),
        )
        reconciler = _make_position_reconciler(
            mock_exchange, mock_tracker, mock_db, use_margin=False
        )

        with patch(
            "src.engines.live.reconciliation.MarginInterestTracker"
        ) as MockMIT:
            reconciler._realize_pnl_on_close(pos, exit_price=1900.0, reason="test")

            MockMIT.assert_not_called()

            # PnL = (2000 - 1900) * 0.1 = 10.0, no interest deduction
            mock_db.update_balance.assert_called_once()
            new_balance = mock_db.update_balance.call_args[0][0]
            assert new_balance == pytest.approx(1000.0 + 10.0)

    def test_interest_error_does_not_block_balance_update(
        self, mock_exchange, mock_tracker, mock_db, caplog
    ):
        """Interest query failure still allows balance update to proceed."""
        pos = _MockPosition(
            symbol="ETHUSDT",
            side="short",
            entry_price=2000.0,
            quantity=0.1,
            current_size=0.1,
            original_size=0.1,
            entry_time=datetime(2025, 1, 1, tzinfo=UTC),
        )
        reconciler = _make_position_reconciler(mock_exchange, mock_tracker, mock_db)

        with patch(
            "src.engines.live.reconciliation.MarginInterestTracker"
        ) as MockMIT:
            instance = MockMIT.return_value
            instance.get_position_interest_cost.side_effect = RuntimeError("API down")

            with caplog.at_level(logging.WARNING):
                reconciler._realize_pnl_on_close(pos, exit_price=1900.0, reason="test")

            # Balance update must still happen with raw PnL (no interest deduction)
            # PnL = (2000 - 1900) * 0.1 = 10.0
            mock_db.update_balance.assert_called_once()
            new_balance = mock_db.update_balance.call_args[0][0]
            assert new_balance == pytest.approx(1000.0 + 10.0)

            assert any(
                "Failed to query margin interest" in msg
                for msg in caplog.messages
            )
