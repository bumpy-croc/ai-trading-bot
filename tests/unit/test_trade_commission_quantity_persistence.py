"""Regression tests: closed trade rows must persist commission (USD) and filled quantity.

Bug: the live close path called ``db_manager.log_trade`` without ``commission`` or
``quantity``, so every persisted ``trades`` row had ``commission = 0`` and
``quantity = NULL`` even though entry+exit fees were paid. Consumers of the
``trades`` table then under-reported fees and could not compute true net P&L.

Unit convention (see Trade.commission in src/database/models.py and
docs/live_trading.md): ``trades.commission`` is the USD total of
``entry_fee + exit_fee`` — exactly the values booked to ``account_balances``
(entry as the ``entry_fee_<SYMBOL>`` ledger event; exit folded into the
``realized_pnl_<SYMBOL>`` event). ``trades.quantity`` is the actual filled base
quantity for the portion being closed.
"""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.fast

from src.engines.live.trading_engine import (
    LiveTradingEngine,
    Position,
    PositionSide,
    _close_entry_fee_usd,
    _closed_base_quantity,
)
from tests.mocks import MockDatabaseManager


@pytest.fixture(autouse=True)
def mock_database_manager(monkeypatch):
    """Mock the DatabaseManager for all tests in this module."""
    monkeypatch.setattr("src.engines.live.trading_engine.DatabaseManager", MockDatabaseManager)


def _make_engine(fee_rate=0.001):
    """Create a paper LiveTradingEngine with a non-zero fee rate."""
    strategy = Mock()
    strategy.get_risk_overrides.return_value = None
    data_provider = Mock()
    data_provider.get_current_price.return_value = 100.0

    return LiveTradingEngine(
        strategy=strategy,
        data_provider=data_provider,
        initial_balance=1_000.0,
        enable_live_trading=False,
        log_trades=False,
        fee_rate=fee_rate,
        slippage_rate=0.0,
    )


def _make_position(
    *,
    quantity,
    entry_fee,
    side=PositionSide.LONG,
    symbol="ETHUSDT",
    entry_price=100.0,
    size=0.25,
    current_size=None,
):
    """Create a recovered-style position carrying an entry fill (quantity + entry fee)."""
    position = Position(
        symbol=symbol,
        side=side,
        size=size,
        entry_price=entry_price,
        entry_time=datetime(2025, 1, 1, tzinfo=UTC),
        order_id="order-1",
        original_size=size,
        current_size=size if current_size is None else current_size,
    )
    position.quantity = quantity
    if entry_fee is not None:
        position.metadata["entry_fee"] = entry_fee
    return position


def _close(engine, position, exit_price=110.0):
    engine.trading_session_id = 1
    engine.live_position_tracker.track_recovered_position(position, db_id=None)
    engine._execute_exit(
        position=position,
        reason="test-close",
        limit_price=None,
        current_price=exit_price,
        candle_high=None,
        candle_low=None,
        candle=None,
        skip_live_close=True,
    )
    trades = engine.db_manager._trades
    assert len(trades) == 1
    return list(trades.values())[0]


def test_closed_trade_persists_commission_and_quantity():
    """A full close persists commission = entry_fee + exit_fee (USD) and the filled quantity."""
    engine = _make_engine(fee_rate=0.001)
    # size 0.25 of $1000 = $250 entry notional at price 100 -> 2.5 base units
    position = _make_position(quantity=2.5, entry_fee=0.25)

    trade = _close(engine, position, exit_price=110.0)

    # exit notional = basis_balance(1000) * fraction(0.25) * (110/100) = 275
    # exit_fee = 275 * 0.001 = 0.275 ; commission = entry_fee(0.25) + 0.275 = 0.525
    assert trade["commission"] is not None, "commission must be persisted, not NULL/0"
    assert trade["commission"] == pytest.approx(0.525, abs=0.01)
    assert trade["commission"] > 0.25  # strictly more than entry fee -> exit fee included
    assert trade["quantity"] == pytest.approx(2.5)


def test_closed_trade_quantity_scaled_for_partially_exited_position():
    """quantity reflects the closed portion: position.quantity scaled by current/original size."""
    engine = _make_engine(fee_rate=0.001)
    # Original 2.5 units at size 0.25; 0.10 of original remains to be closed here.
    position = _make_position(quantity=2.5, entry_fee=0.25, size=0.25, current_size=0.10)

    trade = _close(engine, position, exit_price=110.0)

    # closed base quantity = 2.5 * (current_size 0.10 / original_size 0.25) = 1.0
    assert trade["quantity"] == pytest.approx(1.0)
    assert trade["commission"] is not None
    assert trade["commission"] > 0.0


def test_recovered_position_close_reconstructs_entry_fee():
    """A restart-recovered position (no entry-fee metadata) still persists the entry leg.

    The entry fee is reconstructed from the fee model applied to the recovered entry
    notional, so commission is not understated to exit-fee-only vs the account_balances
    ledger.
    """
    engine = _make_engine(fee_rate=0.001)
    # No entry_fee metadata (entry_fee=None) -> simulates a position recovered from the
    # positions table, which does not persist entry fee. quantity + entry_price are
    # persisted/recovered, so the entry notional (2.5 * 100 = 250) is reconstructable.
    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0)
    assert "entry_fee" not in position.metadata

    trade = _close(engine, position, exit_price=110.0)

    # reconstructed entry_fee = calc_entry_fee(2.5 * 100) = 250 * 0.001 = 0.25
    # exit_fee = 275 * 0.001 = 0.275 ; commission = 0.25 + 0.275 = 0.525
    assert trade["commission"] == pytest.approx(0.525, abs=0.01)
    assert trade["commission"] > 0.275  # entry leg included, not exit-only
    assert trade["quantity"] == pytest.approx(2.5)


def test_close_entry_fee_usd_prefers_metadata_over_reconstruction():
    """When entry-fee metadata exists it is used verbatim (exact booked value)."""
    engine = _make_engine(fee_rate=0.001)
    position = _make_position(quantity=2.5, entry_fee=0.123, entry_price=100.0)

    fee = _close_entry_fee_usd(position, engine.live_execution_engine)

    assert fee == pytest.approx(0.123)  # metadata, not the reconstructed 0.25


def test_close_entry_fee_usd_returns_zero_when_unreconstructable():
    """No metadata and no usable notional -> 0.0 (never raises)."""
    engine = _make_engine(fee_rate=0.001)
    position = _make_position(quantity=None, entry_fee=None, entry_price=100.0)
    position.entry_balance = None
    position.size = 0.0

    fee = _close_entry_fee_usd(position, engine.live_execution_engine)

    assert fee == 0.0


@pytest.mark.parametrize(
    "quantity, original_size, current_size",
    [
        (None, 0.25, 0.25),  # unknown filled quantity
        (0.0, 0.25, 0.25),  # non-positive quantity
        (-2.5, 0.25, 0.25),  # negative quantity
        (2.5, 0.0, 0.25),  # explicit zero original_size (valid size) -> no fabrication
        (2.5, 0.25, -0.10),  # negative current_size -> would fabricate negative qty
        (float("inf"), 0.25, 0.25),  # non-finite quantity
        (2.5, 0.25, float("nan")),  # non-finite current_size
    ],
)
def test_closed_base_quantity_returns_none_for_corrupt_sizing(
    quantity, original_size, current_size
):
    """Corrupt sizing inputs yield NULL, never a fabricated or negative quantity."""
    position = _make_position(quantity=quantity, entry_fee=0.25, current_size=current_size)
    position.original_size = original_size

    assert _closed_base_quantity(position) is None


def test_closed_base_quantity_none_when_no_scaling_basis():
    """original_size and size both non-positive -> no scaling basis -> None (not a fabricated qty)."""
    position = _make_position(quantity=2.5, entry_fee=0.25, size=0.0)
    position.original_size = 0.0
    assert _closed_base_quantity(position) is None


def test_closed_base_quantity_scales_valid_inputs():
    """Sanity: valid inputs scale by current/original and are strictly positive."""
    position = _make_position(quantity=2.5, entry_fee=0.25, size=0.25, current_size=0.10)
    assert _closed_base_quantity(position) == pytest.approx(1.0)


def test_closed_base_quantity_none_for_scaled_in_position():
    """current_size > original_size (scale-in) -> quantity not derivable -> None.

    Scale-ins grow current_size but do not update position.quantity, so the held base
    quantity cannot be derived by scaling; NULL is correct rather than over-reporting.
    """
    position = _make_position(quantity=2.5, entry_fee=0.25, size=0.40, current_size=0.40)
    position.original_size = 0.25  # scaled in beyond the original fraction
    assert _closed_base_quantity(position) is None


def test_recover_active_positions_hydrates_partial_state(monkeypatch):
    """Recovered positions carry original_size/current_size so a partially-exited
    position closes at its remaining size (and logs the remaining quantity)."""
    engine = _make_engine()
    engine.trading_session_id = 1
    pos_data = {
        "id": 42,
        "symbol": "ETHUSDT",
        "side": "long",
        "size": 0.10,
        "entry_price": 100.0,
        "entry_time": datetime(2025, 1, 1, tzinfo=UTC),
        "entry_order_id": "order-42",
        "quantity": 2.5,
        "entry_balance": 1000.0,
        "original_size": 0.25,
        "current_size": 0.10,
        "partial_exits_taken": 1,
        "scale_ins_taken": 0,
        "last_partial_exit_price": 105.0,
        "last_scale_in_price": None,
    }
    monkeypatch.setattr(engine.db_manager, "get_active_positions", lambda *a, **k: [pos_data])
    monkeypatch.setattr(
        engine.db_manager, "heal_positions_with_terminal_trades", lambda *a, **k: 0, raising=False
    )

    engine._recover_active_positions()

    positions = engine.live_position_tracker.positions
    assert len(positions) == 1
    pos = next(iter(positions.values()))
    assert float(pos.original_size) == pytest.approx(0.25)
    assert float(pos.current_size) == pytest.approx(0.10)
    assert pos.partial_exits_taken == 1
    # The remaining base quantity is the original fill scaled by current/original.
    assert _closed_base_quantity(pos) == pytest.approx(2.5 * (0.10 / 0.25))


def test_trade_net_pnl_subtracts_commission_and_interest():
    """_trade_net_pnl returns gross pnl minus commission and margin interest (USD)."""
    from types import SimpleNamespace

    from src.database.manager import _trade_net_pnl

    trade = SimpleNamespace(pnl=10.0, commission=0.5, margin_interest_cost=0.2)
    assert _trade_net_pnl(trade) == pytest.approx(9.3)

    # Missing / invalid commission and interest are treated as 0.0 (legacy rows).
    legacy = SimpleNamespace(pnl=10.0, commission=None, margin_interest_cost=None)
    assert _trade_net_pnl(legacy) == pytest.approx(10.0)


def test_commission_scales_entry_fee_for_partial_close():
    """A partial final close persists commission with the entry leg scaled to its portion."""
    engine = _make_engine(fee_rate=0.001)
    # current_size 0.10 of original 0.25 -> entry leg scaled to 0.40 of the full entry fee.
    position = _make_position(quantity=2.5, entry_fee=0.25, size=0.25, current_size=0.10)

    trade = _close(engine, position, exit_price=110.0)

    # entry leg = 0.25 * (0.10 / 0.25) = 0.10 ; exit notional = 1000 * 0.10 * 1.1 = 110,
    # exit_fee = 0.11 ; commission = 0.10 + 0.11 = 0.21 (NOT the full 0.25 + 0.11 = 0.36).
    assert trade["commission"] == pytest.approx(0.21, abs=0.01)
    assert trade["commission"] < 0.25  # entry leg scaled below the full entry fee


@pytest.mark.parametrize(
    "commission, commission_asset, symbol, price, expected",
    [
        (2.1, "USDT", "ETHUSDT", 2100.0, 2.1),  # quote asset -> as-is (USD)
        (0.001, "ETH", "ETHUSDT", 2100.0, 2.1),  # base asset on a buy -> price into USD
        (0.0, "ETH", "ETHUSDT", 2100.0, 0.0),  # no commission -> 0
    ],
)
def test_order_commission_usd_converts_by_asset(
    commission, commission_asset, symbol, price, expected
):
    """Exchange commission is normalized to USD via its commission_asset."""
    from types import SimpleNamespace as NS

    from src.engines.shared.commission import order_commission_usd as _order_commission_usd

    order = NS(commission=commission, commission_asset=commission_asset)
    assert _order_commission_usd(order, symbol, price) == pytest.approx(expected)


@pytest.mark.parametrize("asset", ["BNB", ""])
def test_order_commission_usd_returns_none_for_unconvertible_asset(asset):
    """A discount/unknown commission asset is not convertible here -> None (use model)."""
    from types import SimpleNamespace as NS

    from src.engines.shared.commission import order_commission_usd as _order_commission_usd

    order = NS(commission=0.05, commission_asset=asset)
    assert _order_commission_usd(order, "ETHUSDT", 2100.0) is None


def _make_reconciler(db, fee_rate=0.001):
    from src.engines.live.reconciliation import PositionReconciler

    return PositionReconciler(
        exchange_interface=Mock(),
        position_tracker=Mock(),
        db_manager=db,
        session_id=1,
        fee_rate=fee_rate,
    )


def test_reconciler_logs_trade_with_commission_and_quantity():
    """The reconciler's offline-close path persists a Trade row (gross pnl, commission, qty)."""
    db = MockDatabaseManager()
    reconciler = _make_reconciler(db, fee_rate=0.001)
    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0)

    reconciler._log_reconciliation_trade(
        position=position,
        entry_price=100.0,
        exit_price=90.0,
        qty=2.5,
        gross_pnl=-25.0,
        exit_fee=0.225,
        interest_cost=0.0,
        reason="stop_loss_filled_offline",
        exit_order_id="sl-order-1",
    )

    trades = db._trades
    assert len(trades) == 1
    t = list(trades.values())[0]
    # entry fee = 2.5 * 100 * 0.001 = 0.25 ; commission = 0.25 + 0.225 = 0.475
    assert t["commission"] == pytest.approx(0.475)
    assert t["quantity"] == pytest.approx(2.5)
    assert t["pnl"] == pytest.approx(-25.0)  # GROSS, parity with engine/backtest


def test_reconciler_realize_pnl_logs_trade_when_opted_in(monkeypatch):
    """_realize_pnl_on_close(log_trade=True) realizes balance AND inserts a Trade row."""
    db = MockDatabaseManager()
    reconciler = _make_reconciler(db, fee_rate=0.001)
    # Audit persistence hits the DB; stub it so the test focuses on trade creation.
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)
    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0)

    reconciler._realize_pnl_on_close(
        position,
        90.0,
        "stop_loss_filled_offline",
        exit_fee=0.2,
        log_trade=True,
        exit_order_id="sl-order-2",
    )

    trades = db._trades
    assert len(trades) == 1
    t = list(trades.values())[0]
    assert t["commission"] is not None and t["commission"] > 0.0
    assert t["quantity"] == pytest.approx(2.5)


def test_reconciler_does_not_log_trade_by_default(monkeypatch):
    """Without opt-in, _realize_pnl_on_close updates balance but inserts no Trade row."""
    db = MockDatabaseManager()
    reconciler = _make_reconciler(db, fee_rate=0.001)
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)
    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0)

    reconciler._realize_pnl_on_close(position, 90.0, "exit_order_recovery", exit_fee=0.2)

    assert len(db._trades) == 0


def test_reconciler_filled_sl_converts_short_commission_to_usd_and_dedups(monkeypatch):
    """Offline SL on a SHORT: the base-asset (ETH) SL commission is converted to USD, and the
    trade is logged with a synthetic, non-NULL dedup key when the SL order has no id."""
    from types import SimpleNamespace as NS

    db = MockDatabaseManager()
    reconciler = _make_reconciler(db, fee_rate=0.001)
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)
    position = _make_position(
        quantity=2.5, entry_fee=None, side=PositionSide.SHORT, entry_price=100.0
    )
    position.db_position_id = 77
    position.order_id = "pos-77"
    # Filled SL with NO order id (forces a synthetic dedup key); commission in base asset ETH.
    sl_order = NS(average_price=90.0, order_id=None, commission=0.001, commission_asset="ETH")

    reconciler._close_position_from_filled_sl(position, sl_order)

    trades = db._trades
    assert len(trades) == 1
    t = list(trades.values())[0]
    # exit fee USD = 0.001 ETH * 90.0 = 0.09 ; entry fee = 2.5*100*0.001 = 0.25 -> 0.34 (NOT
    # 0.25 + 0.001 if the base-asset commission had been mis-booked as USD).
    assert t["commission"] == pytest.approx(0.25 + 0.09, abs=0.01)
    assert t["quantity"] == pytest.approx(2.5)
    assert t["pnl"] == pytest.approx(25.0)  # GROSS short pnl (100-90)*2.5
    # Synthetic, non-NULL dedup key derived from db_position_id (so a re-run collides).
    assert t["order_id"] == "reconcile_sl_77"


def test_reconciler_filled_sl_skips_trade_when_db_close_fails(monkeypatch):
    """If the DB position close fails, NO trade is logged — the row stays OPEN and is
    re-reconciled later; logging now would duplicate when it is re-recovered."""
    from types import SimpleNamespace as NS

    db = MockDatabaseManager()
    reconciler = _make_reconciler(db, fee_rate=0.001)
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)
    monkeypatch.setattr(db, "close_position", Mock(side_effect=RuntimeError("db down")))
    position = _make_position(
        quantity=2.5, entry_fee=None, side=PositionSide.SHORT, entry_price=100.0
    )
    position.db_position_id = 88
    position.order_id = "pos-88"
    sl_order = NS(average_price=90.0, order_id="sl-88", commission=2.0, commission_asset="USDT")

    reconciler._close_position_from_filled_sl(position, sl_order)

    assert len(db._trades) == 0  # DB close failed -> trade not logged (no duplicate risk)
