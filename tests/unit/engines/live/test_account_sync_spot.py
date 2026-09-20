"""Spot-mode regression tests for AccountSynchronizer (#746, #747).

Covers position side/unit matching, the balance-sync/reconciler split, the
invalid-balance guard, and the trade-recovery ID space and timezone handling.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from src.data_providers.exchange_interface import (
    AccountBalance,
    OrderSide,
    Position,
    Trade,
)
from src.database.manager import DatabaseManager
from src.database.models import PositionSide, TradeSource
from src.engines.live.account_sync import AccountSynchronizer

pytestmark = pytest.mark.fast


def _exchange_position(symbol="BTCUSDT", side="long", size=0.5, price=50_000.0) -> Position:
    now = datetime.now(UTC)
    return Position(
        symbol=symbol,
        side=side,
        size=size,
        entry_price=price,
        current_price=price,
        unrealized_pnl=0.0,
        margin_type="spot",
        leverage=1.0,
        order_id="",
        open_time=now,
        last_update_time=now,
    )


def _db_position(**overrides) -> dict:
    row = {
        "id": 7,
        "symbol": "BTCUSDT",
        "side": "LONG",  # DatabaseManager.get_active_positions emits p.side.value
        "size": 0.025,  # balance fraction
        "quantity": 0.5,  # asset amount
        "entry_price": 50_000.0,
        "current_size": None,
        "original_size": None,
    }
    row.update(overrides)
    return row


def _usdt(total) -> AccountBalance:
    return AccountBalance(
        asset="USDT", free=total, locked=0.0, total=total, last_updated=datetime.now(UTC)
    )


@pytest.fixture
def db():
    manager = create_autospec(DatabaseManager, instance=True)
    manager.get_current_balance.return_value = 1_000.0
    manager.get_active_positions.return_value = []
    manager.log_position.return_value = 99
    return manager


@pytest.fixture
def sync(db):
    return AccountSynchronizer(exchange=MagicMock(), db_manager=db, session_id=1)


class TestSyncPositions:
    def test_matching_position_with_different_side_casing_is_left_alone(self, sync, db):
        db.get_active_positions.return_value = [_db_position()]

        result = sync._sync_positions([_exchange_position(side="long")])

        assert result["synced_positions"] == 1
        assert result["new_positions"] == 0
        assert result["missing_positions"] == 0
        db.close_position.assert_not_called()
        db.log_position.assert_not_called()

    def test_size_fraction_is_not_compared_with_asset_quantity(self, sync, db):
        # DB size 0.025 (fraction) vs exchange 0.5 (BTC) must not look like a change.
        db.get_active_positions.return_value = [_db_position()]

        sync._sync_positions([_exchange_position(size=0.5)])

        db.update_position.assert_not_called()

    def test_quantity_mismatch_refreshes_mark_but_never_writes_size(self, sync, db):
        db.get_active_positions.return_value = [_db_position()]

        sync._sync_positions([_exchange_position(size=0.75)])

        db.update_position.assert_called_once()
        kwargs = db.update_position.call_args.kwargs
        assert "size" not in kwargs
        assert kwargs["current_price"] == 50_000.0

    def test_partial_exit_quantity_is_scaled_before_comparison(self, sync, db):
        db.get_active_positions.return_value = [
            _db_position(quantity=1.0, original_size=0.04, current_size=0.02)
        ]

        sync._sync_positions([_exchange_position(size=0.5)])

        db.update_position.assert_not_called()

    def test_missing_db_quantity_skips_comparison(self, sync, db):
        db.get_active_positions.return_value = [_db_position(quantity=None)]

        sync._sync_positions([_exchange_position(size=0.75)])

        db.update_position.assert_not_called()

    def test_new_exchange_position_is_logged_with_fraction_and_quantity(self, sync, db):
        # notional 0.5 * 50k = 25k against a 1k balance -> capped at 1.0
        sync._sync_positions([_exchange_position(size=0.5)])
        assert db.log_position.call_args.kwargs["size"] == pytest.approx(1.0)
        assert db.log_position.call_args.kwargs["quantity"] == 0.5
        assert db.log_position.call_args.kwargs["side"] == PositionSide.LONG

    def test_new_exchange_position_fraction_below_cap(self, sync, db):
        db.get_current_balance.return_value = 2_000.0
        sync._sync_positions([_exchange_position(size=0.01)])  # notional 500

        assert db.log_position.call_args.kwargs["size"] == pytest.approx(0.25)

    def test_db_position_absent_on_exchange_is_reported_not_closed(self, sync, db):
        db.get_active_positions.return_value = [_db_position()]

        result = sync._sync_positions([])

        assert result["missing_positions"] == 1
        db.close_position.assert_not_called()
        db.log_event.assert_called_once()
        assert db.log_event.call_args.kwargs["error_code"] == "DB_POSITION_MISSING_ON_EXCHANGE"

    def test_size_fraction_guards_zero_capital_and_notional(self, sync, db):
        db.get_current_balance.return_value = 0.0

        assert sync._size_fraction(0.0) == 1.0

    def test_symbol_scoped_sync_ignores_other_symbols(self, sync, db):
        db.get_active_positions.return_value = [
            _db_position(id=1),
            _db_position(id=2, symbol="ETHUSDT"),
        ]

        result = sync._sync_positions([_exchange_position("BTCUSDT")], symbol="BTCUSDT")

        assert result["total_db_positions"] == 1
        db.close_position.assert_not_called()


class TestSyncBalances:
    def test_open_position_notional_is_not_a_discrepancy(self, sync, db):
        # 1000 total capital, 400 in BTC -> 600 cash on the exchange is exactly right.
        db.get_active_positions.return_value = [_db_position(quantity=0.008)]

        result = sync._sync_balances([_usdt(600.0)])

        assert result["synced"] is True
        assert result["corrected"] is False
        assert "deferred_to_reconciler" not in result
        assert result["expected_cash"] == pytest.approx(600.0)
        db.atomic_balance_correction.assert_not_called()

    def test_real_discrepancy_is_reported_and_left_to_the_reconciler(self, sync, db):
        result = sync._sync_balances([_usdt(700.0)])

        assert result["corrected"] is False
        assert result["deferred_to_reconciler"] is True
        assert result["difference_percent"] == pytest.approx(30.0)
        db.atomic_balance_correction.assert_not_called()
        db.update_balance.assert_not_called()

    def test_small_drift_below_reconciler_threshold_is_not_flagged(self, sync, db):
        # 3% drift: above the old 1% sync threshold, below the reconciler's 5%.
        result = sync._sync_balances([_usdt(1_030.0)])

        assert result["difference_percent"] == pytest.approx(3.0)
        assert "deferred_to_reconciler" not in result

    @pytest.mark.parametrize("bad_total", [None, "12", float("nan")])
    def test_invalid_usdt_total_returns_error_dict(self, sync, bad_total):
        balance = MagicMock()
        balance.asset = "USDT"
        balance.total = bad_total

        result = sync._sync_balances([balance])

        assert result["synced"] is False
        assert result["error"].startswith("Invalid balance data from exchange")


class TestRecoverMissingTrades:
    @staticmethod
    def _fill(trade_id, order_id, qty, price, at=None) -> Trade:
        return Trade(
            trade_id=trade_id,
            order_id=order_id,
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=qty,
            price=price,
            commission=0.0,
            commission_asset="USDT",
            time=at or datetime.now(UTC),
        )

    def test_orders_already_recorded_are_not_recovered(self, sync, db):
        sync.exchange.get_recent_trades.return_value = [self._fill("5001", "O-1", 0.1, 50_000.0)]
        db.get_known_exchange_order_ids.return_value = {"O-1"}

        result = sync.recover_missing_trades("BTCUSDT")

        assert result["missing_trades"] == 0
        db.log_trade.assert_not_called()

    def test_fills_of_one_order_become_a_single_vwap_row(self, sync, db):
        now = datetime.now(UTC)
        sync.exchange.get_recent_trades.return_value = [
            self._fill("1", "O-9", 0.1, 50_000.0, now - timedelta(seconds=5)),
            self._fill("2", "O-9", 0.3, 51_000.0, now),
        ]
        db.get_known_exchange_order_ids.return_value = set()

        result = sync.recover_missing_trades("BTCUSDT")

        assert result["recovered_trades"] == 1
        kwargs = db.log_trade.call_args.kwargs
        assert kwargs["exit_order_id"] == "O-9"
        assert kwargs["quantity"] == pytest.approx(0.4)
        assert kwargs["exit_price"] == pytest.approx(50_750.0)
        assert kwargs["side"] == PositionSide.LONG
        assert kwargs["strategy_name"] == "exchange_recovery"
        assert kwargs["exit_reason"] == "recovered_from_exchange"
        assert kwargs["pnl"] == 0.0
        assert result["details"][0]["trade_ids"] == ["1", "2"]

    def test_naive_exchange_timestamps_are_treated_as_utc(self, sync, db):
        naive_now = datetime.now(UTC).replace(tzinfo=None)
        sync.exchange.get_recent_trades.return_value = [
            self._fill("1", "O-naive", 0.1, 50_000.0, naive_now)
        ]
        db.get_known_exchange_order_ids.return_value = set()

        result = sync.recover_missing_trades("BTCUSDT")

        assert result["recovered_trades"] == 1

    def test_exchange_trades_older_than_the_window_are_ignored(self, sync, db):
        old = datetime.now(UTC) - timedelta(days=30)
        sync.exchange.get_recent_trades.return_value = [self._fill("1", "O-old", 0.1, 1.0, old)]
        db.get_known_exchange_order_ids.return_value = set()

        result = sync.recover_missing_trades("BTCUSDT", days_back=7)

        assert result["missing_trades"] == 0

    def test_recovery_dedups_against_real_db_across_sessions(self):
        """A fill already booked in a previous session is not inserted again."""
        real_db = DatabaseManager("sqlite:///:memory:")
        old_session = real_db.create_trading_session(
            strategy_name="S",
            symbol="BTCUSDT",
            timeframe="1h",
            mode=TradeSource.LIVE,
            initial_balance=1_000.0,
        )
        now = datetime.now(UTC).replace(tzinfo=None)
        real_db.log_trade(
            symbol="BTCUSDT",
            side="long",
            entry_price=50_000.0,
            exit_price=51_000.0,
            size=0.1,
            entry_time=now - timedelta(hours=2),
            exit_time=now - timedelta(hours=1),
            pnl=10.0,
            exit_reason="take_profit",
            strategy_name="S",
            exit_order_id="1234",
            session_id=old_session,
        )
        new_session = real_db.create_trading_session(
            strategy_name="S",
            symbol="BTCUSDT",
            timeframe="1h",
            mode=TradeSource.LIVE,
            initial_balance=1_000.0,
        )
        exchange = MagicMock()
        exchange.get_recent_trades.return_value = [
            self._fill("777", "1234", 0.02, 51_000.0),  # already booked (order 1234)
            self._fill("778", "9999", 0.02, 52_000.0),  # genuinely missing
        ]
        synchronizer = AccountSynchronizer(exchange, real_db, session_id=new_session)

        first = synchronizer.recover_missing_trades("BTCUSDT")
        second = synchronizer.recover_missing_trades("BTCUSDT")

        assert first["recovered_trades"] == 1
        assert first["details"][0]["order_id"] == "9999"
        assert second["missing_trades"] == 0
        recovered = real_db.get_trades_by_symbol_and_date(
            "BTCUSDT", now - timedelta(days=1), new_session
        )
        assert [t["order_id"] for t in recovered] == ["9999"]


class TestKnownOrderIds:
    def test_order_placed_before_window_but_in_db_is_known(self):
        real_db = DatabaseManager("sqlite:///:memory:")
        session_id = real_db.create_trading_session(
            strategy_name="S",
            symbol="BTCUSDT",
            timeframe="1h",
            mode=TradeSource.LIVE,
            initial_balance=1_000.0,
        )
        position_id = real_db.log_position(
            symbol="BTCUSDT",
            side="long",
            entry_price=100.0,
            size=0.1,
            strategy_name="S",
            entry_order_id="ENTRY-1",
            quantity=1.0,
            session_id=session_id,
        )
        assert position_id
        since = datetime.now(UTC) + timedelta(hours=6)  # order created "before the window"

        known = real_db.get_known_exchange_order_ids("BTCUSDT", since)

        assert "ENTRY-1" in known


class TestBinanceRecentTradesTimezone:
    def test_get_recent_trades_returns_aware_utc_timestamps(self):
        from src.data_providers import binance_provider
        from src.data_providers.binance_provider import BinanceProvider

        provider = BinanceProvider.__new__(BinanceProvider)
        provider._client = object()
        raw = [
            {
                "id": 1,
                "orderId": 2,
                "symbol": "BTCUSDT",
                "isBuyer": False,
                "qty": "0.1",
                "price": "50000",
                "commission": "0",
                "commissionAsset": "USDT",
                "time": 1_700_000_000_000,
            }
        ]
        with (
            patch.object(binance_provider, "BINANCE_AVAILABLE", True),
            patch.object(provider, "_call_get_my_trades", return_value=raw),
        ):
            trades = provider.get_recent_trades("BTCUSDT")

        assert trades[0].time.tzinfo is UTC
        assert trades[0].time >= datetime.now(UTC) - timedelta(days=365 * 10)  # aware compare
