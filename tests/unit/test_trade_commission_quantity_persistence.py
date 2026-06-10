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


def _make_reconciler(db, fee_rate=0.001, tracker=None):
    from src.engines.live.reconciliation import PositionReconciler

    return PositionReconciler(
        exchange_interface=Mock(),
        position_tracker=tracker if tracker is not None else Mock(),
        db_manager=db,
        session_id=1,
        fee_rate=fee_rate,
    )


def _tracker_with(position):
    """A position_tracker stub exposing the internals the reconciler close paths read:
    a real lock + a {order_id: position} map, a spied remove_position, and a real
    pop_position that removes and returns the position (None if already gone)."""
    import threading
    from types import SimpleNamespace

    positions = {position.order_id: position}
    tracker = SimpleNamespace(
        _positions=positions,
        _positions_lock=threading.Lock(),
        remove_position=Mock(),
    )
    tracker.pop_position = lambda oid: positions.pop(oid, None)
    return tracker


def _recon_result():
    from src.engines.live.reconciliation import ReconciliationResult

    return ReconciliationResult(entity_type="position", entity_id=None, status="resolved")


def _register_open_position(db, position):
    """Register the position in the mock DB so close_position(db_position_id) returns True —
    mirrors a real OPEN row that the reconciler then closes (the close paths now gate on the
    actual return value, so an unregistered row would read as 'not closed')."""
    db._positions[position.db_position_id] = {"id": position.db_position_id, "status": "OPEN"}


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
    _register_open_position(db, position)
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


def test_reconciler_filled_exit_logs_trade_with_exchange_order_id(monkeypatch):
    """Crash-recovery FULL_EXIT: closing a still-tracked position logs a Trade row keyed by the
    real exchange exit order id, with commission + quantity + GROSS pnl (parity with offline SL)."""
    db = MockDatabaseManager()
    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0)
    position.db_position_id = 55
    position.order_id = "pos-55"
    _register_open_position(db, position)
    reconciler = _make_reconciler(db, fee_rate=0.001, tracker=_tracker_with(position))
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)

    reconciler._reconcile_filled_exit(
        {"position_id": 55, "client_order_id": "atbx_exit_1"},
        fill_price=90.0,
        exit_fee=0.2,
        exit_order_id="EXCH-EXIT-999",
    )

    trades = db._trades
    assert len(trades) == 1
    t = list(trades.values())[0]
    assert t["order_id"] == "EXCH-EXIT-999"  # real exchange exit order id as the dedup key
    assert t["quantity"] == pytest.approx(2.5)
    # entry fee = 2.5*100*0.001 = 0.25 ; commission = 0.25 + exit_fee 0.2 = 0.45
    assert t["commission"] == pytest.approx(0.45, abs=0.01)
    assert t["pnl"] == pytest.approx((90.0 - 100.0) * 2.5)  # GROSS long pnl = -25.0


def test_reconciler_filled_exit_synthesizes_dedup_key_when_no_exchange_id(monkeypatch):
    """With no exchange exit id, a synthetic non-NULL key reconcile_exit_<pos_id> is used so a
    re-run collides on uq_trade_order_session instead of inserting a duplicate."""
    db = MockDatabaseManager()
    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0)
    position.db_position_id = 56
    position.order_id = "pos-56"
    _register_open_position(db, position)
    reconciler = _make_reconciler(db, fee_rate=0.001, tracker=_tracker_with(position))
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)

    reconciler._reconcile_filled_exit(
        {"position_id": 56, "client_order_id": "atbx_exit_2"},
        fill_price=110.0,
        exit_fee=0.0,
        exit_order_id=None,
    )

    t = list(db._trades.values())[0]
    assert t["order_id"] == "reconcile_exit_56"


def test_reconciler_filled_exit_skips_trade_when_db_close_fails(monkeypatch):
    """If the DB position close fails, NO trade is logged and the balance is NOT corrected — the
    row stays OPEN and is re-reconciled later (avoids the double balance correction)."""
    db = MockDatabaseManager()
    monkeypatch.setattr(db, "close_position", Mock(side_effect=RuntimeError("db down")))
    no_balance_write = Mock()
    monkeypatch.setattr(db, "update_balance", no_balance_write)
    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0)
    position.db_position_id = 57
    position.order_id = "pos-57"
    reconciler = _make_reconciler(db, fee_rate=0.001, tracker=_tracker_with(position))
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)

    reconciler._reconcile_filled_exit(
        {"position_id": 57, "client_order_id": "atbx_exit_3"},
        fill_price=90.0,
        exit_fee=0.2,
        exit_order_id="EXCH-1",
    )

    assert len(db._trades) == 0  # DB close failed -> trade not logged (no duplicate risk)
    no_balance_write.assert_not_called()  # and P&L not realized -> no double balance correction


def test_reconciler_external_close_spot_logs_balance_neutral_trade(monkeypatch):
    """SPOT external close (asset sold on the exchange UI): a balance-neutral Trade row is logged
    (audit only) keyed by reconcile_ext_<pos_id>, priced mark-to-market. Capital is owned by
    Step C (_reconcile_balance), so the session balance is NOT written here."""
    from types import SimpleNamespace

    db = MockDatabaseManager()
    no_balance_write = Mock()
    monkeypatch.setattr(db, "update_balance", no_balance_write)
    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0, symbol="ETHUSDT")
    position.db_position_id = 61
    position.order_id = "pos-61"
    _register_open_position(db, position)
    reconciler = _make_reconciler(db, fee_rate=0.001, tracker=_tracker_with(position))
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)
    # Exchange holds ~0 ETH -> external close detected (held < 50% of tracked 2.5).
    reconciler.exchange.get_balance = Mock(return_value=SimpleNamespace(total=0.0))
    # Current market price feeds the mark-to-market estimate for the audit row.
    reconciler._data_provider = SimpleNamespace(get_current_price=lambda _s: 110.0)

    reconciler._verify_asset_holdings(position, _recon_result())

    trades = db._trades
    assert len(trades) == 1
    t = list(trades.values())[0]
    assert t["order_id"] == "reconcile_ext_61"  # synthetic, non-NULL, stable across re-runs
    assert t["quantity"] == pytest.approx(2.5)
    assert t["commission"] is not None and t["commission"] > 0.0  # reconstructed entry fee
    assert t["pnl"] == pytest.approx((110.0 - 100.0) * 2.5)  # GROSS long pnl = 25.0
    no_balance_write.assert_not_called()  # balance-neutral; capital owned by Step C


def test_reconciler_external_close_margin_logs_balance_neutral_trade(monkeypatch):
    """MARGIN external close/liquidation via _remove_phantom_position: a balance-neutral Trade row
    is logged. Capital is owned by account_sync margin-equity sync, so balance is NOT written."""
    from types import SimpleNamespace

    db = MockDatabaseManager()
    no_balance_write = Mock()
    monkeypatch.setattr(db, "update_balance", no_balance_write)
    position = _make_position(
        quantity=2.5, entry_fee=None, side=PositionSide.SHORT, entry_price=100.0
    )
    position.db_position_id = 62
    position.order_id = "pos-62"
    position.stop_loss_order_id = None
    _register_open_position(db, position)
    reconciler = _make_reconciler(db, fee_rate=0.001, tracker=_tracker_with(position))
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)
    reconciler._data_provider = SimpleNamespace(get_current_price=lambda _s: 90.0)

    reconciler._remove_phantom_position(position, _recon_result())

    t = list(db._trades.values())[0]
    assert t["order_id"] == "reconcile_ext_62"
    assert t["pnl"] == pytest.approx((100.0 - 90.0) * 2.5)  # GROSS short pnl = 25.0
    assert t["quantity"] == pytest.approx(2.5)
    no_balance_write.assert_not_called()


def test_reconciler_external_close_skips_trade_when_db_close_fails(monkeypatch):
    """If the DB position close fails, NO external-close Trade row is logged — the row stays OPEN
    and is re-recovered later; logging now would duplicate it."""
    from types import SimpleNamespace

    db = MockDatabaseManager()
    monkeypatch.setattr(db, "close_position", Mock(side_effect=RuntimeError("db down")))
    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0)
    position.db_position_id = 63
    position.order_id = "pos-63"
    position.stop_loss_order_id = None
    reconciler = _make_reconciler(db, fee_rate=0.001, tracker=_tracker_with(position))
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)
    reconciler._data_provider = SimpleNamespace(get_current_price=lambda _s: 110.0)

    reconciler._remove_phantom_position(position, _recon_result())

    assert len(db._trades) == 0


def test_reconciler_external_close_dedups_across_restart(monkeypatch):
    """Across a restart (position re-loaded into the tracker, DB row re-opened, but the persisted
    trade row remains), re-detecting the same external close inserts only ONE Trade row: the
    synthetic reconcile_ext_<pos_id> key collides on uq_trade_order_session."""
    from types import SimpleNamespace

    db = MockDatabaseManager()
    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0)
    position.db_position_id = 64
    position.order_id = "pos-64"
    position.stop_loss_order_id = None
    _register_open_position(db, position)
    tracker = _tracker_with(position)
    reconciler = _make_reconciler(db, fee_rate=0.001, tracker=tracker)
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)
    reconciler._data_provider = SimpleNamespace(get_current_price=lambda _s: 110.0)

    reconciler._remove_phantom_position(position, _recon_result())  # logs reconcile_ext_64
    # Simulate a later restart: the position is re-tracked and the DB row re-opened, but the
    # persisted trade row remains, so the second attempt dedups instead of duplicating.
    tracker._positions[position.order_id] = position
    _register_open_position(db, position)
    reconciler._remove_phantom_position(position, _recon_result())

    assert len(db._trades) == 1  # synthetic key collided on uq_trade_order_session


def test_external_close_exit_price_falls_back_to_entry_when_no_provider():
    """With no data provider, the external-close exit price degrades to entry_price (GROSS pnl 0),
    so the audit row still records commission + quantity without fabricating a price."""
    db = MockDatabaseManager()
    reconciler = _make_reconciler(db, fee_rate=0.001)  # no data_provider threaded

    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0)

    assert reconciler._external_close_exit_price(position) == pytest.approx(100.0)


def test_reconciler_filled_exit_skips_trade_when_db_close_returns_false(monkeypatch):
    """close_position returns False (row not found / commit rolled back) WITHOUT raising — the gate
    must treat that as 'not closed' and log no trade (else trades/account_balances diverge)."""
    db = MockDatabaseManager()
    monkeypatch.setattr(db, "close_position", Mock(return_value=False))
    no_balance_write = Mock()
    monkeypatch.setattr(db, "update_balance", no_balance_write)
    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0)
    position.db_position_id = 71
    position.order_id = "pos-71"
    reconciler = _make_reconciler(db, fee_rate=0.001, tracker=_tracker_with(position))
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)

    reconciler._reconcile_filled_exit(
        {"position_id": 71, "client_order_id": "atbx_exit_f"},
        fill_price=90.0,
        exit_fee=0.2,
        exit_order_id="EXCH-F",
    )

    assert len(db._trades) == 0  # False return -> not persisted -> no trade
    no_balance_write.assert_not_called()


def test_reconciler_external_close_skips_trade_when_db_close_returns_false(monkeypatch):
    """External close: close_position returning False (not raising) must skip the trade row."""
    from types import SimpleNamespace

    db = MockDatabaseManager()
    monkeypatch.setattr(db, "close_position", Mock(return_value=False))
    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0)
    position.db_position_id = 72
    position.order_id = "pos-72"
    position.stop_loss_order_id = None
    reconciler = _make_reconciler(db, fee_rate=0.001, tracker=_tracker_with(position))
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)
    reconciler._data_provider = SimpleNamespace(get_current_price=lambda _s: 110.0)

    reconciler._remove_phantom_position(position, _recon_result())

    assert len(db._trades) == 0


def test_reconciler_external_close_skips_trade_when_position_already_reconciled(monkeypatch):
    """If an earlier reconcile pass already removed+closed the position (pop_position returns None),
    the external-close path must NOT log a second trade row — the reconcile_exit_/reconcile_ext_
    keys differ, so uq_trade_order_session would not catch the duplicate."""
    from types import SimpleNamespace

    db = MockDatabaseManager()
    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0, symbol="ETHUSDT")
    position.db_position_id = 73
    position.order_id = "pos-73"
    tracker = _tracker_with(position)
    tracker.pop_position(position.order_id)  # simulate Step A already popped it this run
    reconciler = _make_reconciler(db, fee_rate=0.001, tracker=tracker)
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)
    reconciler.exchange.get_balance = Mock(return_value=SimpleNamespace(total=0.0))
    reconciler._data_provider = SimpleNamespace(get_current_price=lambda _s: 110.0)

    reconciler._verify_asset_holdings(position, _recon_result())

    assert len(db._trades) == 0  # not double-logged


def test_reconciler_filled_exit_scale_in_nulls_quantity(monkeypatch):
    """A scaled-in position closed through the FULL_EXIT caller stores NULL quantity (regression
    guard: the caller's qty pre-scaling must not produce a logged quantity for a scale-in)."""
    db = MockDatabaseManager()
    position = _make_position(
        quantity=2.5, entry_fee=None, entry_price=100.0, size=0.40, current_size=0.40
    )
    position.original_size = 0.25  # scaled in beyond the original fraction
    position.db_position_id = 75
    position.order_id = "pos-75"
    _register_open_position(db, position)
    reconciler = _make_reconciler(db, fee_rate=0.001, tracker=_tracker_with(position))
    monkeypatch.setattr(reconciler, "_persist_audit", lambda *a, **k: None)

    reconciler._reconcile_filled_exit(
        {"position_id": 75, "client_order_id": "atbx_exit_si"},
        fill_price=110.0,
        exit_fee=0.11,
        exit_order_id="EXCH-SI",
    )

    t = list(db._trades.values())[0]
    assert t["quantity"] is None  # scale-in -> NULL, not the inflated 4.0
    assert t["order_id"] == "EXCH-SI"


def test_reconciler_dedups_duplicate_trade_on_rerun():
    """A re-run with the same exit_order_id + session inserts only ONE trade row.

    The mock enforces uq_trade_order_session, so the second log_trade raises IntegrityError
    which _log_reconciliation_trade swallows — exercising the dedup branch (the #657/#668
    phantom-trade guard) end to end, not just the key string.
    """
    db = MockDatabaseManager()
    reconciler = _make_reconciler(db, fee_rate=0.001)
    position = _make_position(quantity=2.5, entry_fee=None, entry_price=100.0)

    for _ in range(2):
        reconciler._log_reconciliation_trade(
            position=position,
            entry_price=100.0,
            exit_price=90.0,
            qty=2.5,
            gross_pnl=-25.0,
            exit_fee=0.225,
            interest_cost=0.0,
            reason="stop_loss_filled_offline",
            exit_order_id="sl-dedup-1",
        )

    assert len(db._trades) == 1  # second insert deduped, not duplicated


def test_reconciler_scale_in_close_nulls_quantity_and_does_not_inflate_commission():
    """A scaled-in position (current_size > original_size) closed by the reconciler stores
    NULL quantity and the un-inflated entry fee — matching the engine close path's policy."""
    db = MockDatabaseManager()
    reconciler = _make_reconciler(db, fee_rate=0.001)
    # Scaled in: held grew to 0.40 of balance from an original 0.25; quantity=2.5 is the
    # ORIGINAL fill (not updated on scale-in). _realize_pnl_on_close would scale qty to
    # 2.5 * 0.40/0.25 = 4.0; the engine path NULLs this case rather than over-reporting.
    position = _make_position(
        quantity=2.5, entry_fee=None, entry_price=100.0, size=0.40, current_size=0.40
    )
    position.original_size = 0.25

    reconciler._log_reconciliation_trade(
        position=position,
        entry_price=100.0,
        exit_price=110.0,
        qty=4.0,  # the scaled (inflated) qty the caller would pass
        gross_pnl=40.0,
        exit_fee=0.11,
        interest_cost=0.0,
        reason="stop_loss_filled_offline",
        exit_order_id="sl-scalein-1",
    )

    trade = list(db._trades.values())[0]
    assert trade["quantity"] is None  # not the over-reported 4.0
    # entry fee is on the ORIGINAL fill (2.5 * 100 * 0.001 = 0.25), not the inflated 4.0.
    assert trade["commission"] == pytest.approx(0.25 + 0.11, abs=0.001)


def test_get_performance_metrics_nets_commission(monkeypatch):
    """get_performance_metrics nets commission: a gross-winner whose commission exceeds its
    gross pnl is bucketed as a loss and reduces total_pnl (the _trade_net_pnl integration)."""
    import contextlib
    from datetime import timedelta
    from types import SimpleNamespace
    from unittest.mock import Mock as M

    from src.database.manager import DatabaseManager
    from src.database.models import Trade

    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    trades = [
        # clear winner: net +9.5
        SimpleNamespace(
            pnl=10.0,
            commission=0.5,
            margin_interest_cost=0.0,
            entry_time=t0,
            exit_time=t0 + timedelta(hours=1),
        ),
        # marginal gross-winner that nets to a LOSS after commission: net -0.10
        SimpleNamespace(
            pnl=0.30,
            commission=0.40,
            margin_interest_cost=0.0,
            entry_time=t0,
            exit_time=t0 + timedelta(hours=1),
        ),
    ]

    def query_side_effect(model):
        q = M()
        q.filter.return_value = q
        q.order_by.return_value = q
        q.all.return_value = trades if model is Trade else []
        return q

    mock_session = M()
    mock_session.query.side_effect = query_side_effect

    @contextlib.contextmanager
    def fake_session(*a, **k):
        yield mock_session

    manager = DatabaseManager.__new__(DatabaseManager)
    manager._current_session_id = 1
    monkeypatch.setattr(DatabaseManager, "get_session_with_timeout", fake_session)

    metrics = manager.get_performance_metrics(session_id=1)

    # total_pnl = net(9.5) + net(-0.10) = 9.4 ; the marginal trade is a LOSER net of fees.
    assert metrics["total_pnl"] == pytest.approx(9.4)
    assert metrics["winning_trades"] == 1
    assert metrics["losing_trades"] == 1
    assert metrics["win_rate"] == pytest.approx(50.0)
