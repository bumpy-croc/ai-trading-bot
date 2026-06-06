"""Unit tests for the orphaned-margin-borrow sweep (reconciliation.run_orphaned_borrow_sweep).

The sweep repays a cross-margin borrow that has no tracked position behind it, but ONLY
when a strict gate chain proves the base asset is flat, order-free, fully covered, and
under a value cap. These tests exercise each gate plus the dry-run/active/over-cap paths.
Money never moves unless every gate passes AND the flag is "active".
"""

from __future__ import annotations

import threading
import time
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.engines.live import reconciliation
from src.engines.live.reconciliation import BaseAssetLockRegistry, run_orphaned_borrow_sweep

pytestmark = pytest.mark.fast

SYMBOL = "ETHUSDT"
BASE = "ETH"

# Sentinel so _run() defaults active-mode tests to a REAL lock registry (matching
# production), while still allowing an explicit lock_registry=None to test fail-closed.
_DEFAULT_REGISTRY = object()


def _snapshot(borrowed="0.003", interest="0", free="0.003", locked="0", net="0.00007"):
    return {
        "asset": BASE,
        "free": free,
        "locked": locked,
        "borrowed": borrowed,
        "interest": interest,
        "netAsset": net,
    }


def _exchange(
    *,
    snapshot=None,
    verify_after=None,
    has_orders: bool | None = False,
    price=1600.0,
    repay_ok=True,
):
    ex = MagicMock()
    snap = snapshot if snapshot is not None else _snapshot()
    if verify_after is None:
        ex.get_margin_account_asset.return_value = snap
    else:
        ex.get_margin_account_asset.side_effect = [snap, verify_after]
    ex.has_open_orders.return_value = has_orders
    ex.get_current_price.return_value = price
    ex.repay_margin_loan.return_value = repay_ok
    return ex


def _tracker(has_position=False):
    t = MagicMock()
    t.has_position_for_symbol.return_value = has_position
    return t


def _db(unresolved=None):
    db = MagicMock()
    db.get_unresolved_orders.return_value = [] if unresolved is None else unresolved
    db.log_audit_event.return_value = 1
    return db


def _run(
    exchange,
    tracker,
    db,
    *,
    mode="active",
    symbols=(SYMBOL,),
    cooldown=None,
    use_margin=True,
    lock_registry=_DEFAULT_REGISTRY,
):
    # Default to a real registry so active-mode gate tests exercise the locked path
    # (as production does). Pass lock_registry=None explicitly to test fail-closed.
    if lock_registry is _DEFAULT_REGISTRY:
        lock_registry = BaseAssetLockRegistry()
    with patch.object(reconciliation, "get_flag", return_value=mode):
        return run_orphaned_borrow_sweep(
            exchange=exchange,
            position_tracker=tracker,
            db_manager=db,
            session_id=1,
            use_margin=use_margin,
            symbols=list(symbols),
            cooldown_state={} if cooldown is None else cooldown,
            lock_registry=lock_registry,
        )


# ---- happy paths ----------------------------------------------------------- #


def test_active_repays_when_flat_covered_under_cap():
    ex = _exchange(verify_after=_snapshot(borrowed="0", interest="0", free="0", net="0"))
    results = _run(ex, _tracker(), _db(), mode="active")
    # Repaid the exact Decimal liability (borrowed + interest), not a re-rounded float.
    ex.repay_margin_loan.assert_called_once_with(BASE, Decimal("0.003"))
    # Post-repay re-query happened (snapshot read twice: gate + verify).
    assert ex.get_margin_account_asset.call_count == 2
    assert results and results[0].status == "corrected"


def test_dry_run_detects_but_never_repays():
    ex = _exchange()
    results = _run(ex, _tracker(), _db(), mode="dry_run")
    ex.repay_margin_loan.assert_not_called()
    assert results and results[0].status == "skipped"


def test_off_mode_is_noop():
    ex = _exchange()
    results = _run(ex, _tracker(), _db(), mode="off")
    ex.repay_margin_loan.assert_not_called()
    ex.get_margin_account_asset.assert_not_called()
    assert results == []


def test_not_margin_is_noop():
    ex = _exchange()
    results = _run(ex, _tracker(), _db(), mode="active", use_margin=False)
    ex.repay_margin_loan.assert_not_called()
    assert results == []


# ---- gate: no tracked position --------------------------------------------- #


def test_skip_when_position_tracked():
    ex = _exchange()
    _run(ex, _tracker(has_position=True), _db(), mode="active")
    ex.repay_margin_loan.assert_not_called()


def test_skip_when_other_symbol_sharing_base_has_position():
    # ETHUSDT and ETHUSDC share base ETH; a position on either blocks repaying the ETH borrow.
    ex = _exchange()
    tracker = MagicMock()
    tracker.has_position_for_symbol.side_effect = lambda s: s == "ETHUSDC"
    _run(ex, tracker, _db(), mode="active", symbols=("ETHUSDT", "ETHUSDC"))
    ex.repay_margin_loan.assert_not_called()


# ---- gate: no in-flight order (fail-closed) -------------------------------- #


def test_skip_when_unresolved_journal_row_for_symbol():
    ex = _exchange()
    _run(ex, _tracker(), _db(unresolved=[{"symbol": SYMBOL}]), mode="active")
    ex.repay_margin_loan.assert_not_called()


def test_skip_when_open_orders_present():
    ex = _exchange(has_orders=True)
    _run(ex, _tracker(), _db(), mode="active")
    ex.repay_margin_loan.assert_not_called()


def test_fail_closed_when_open_orders_lookup_unconfirmed():
    ex = _exchange(has_orders=None)  # None = lookup failed -> must skip
    _run(ex, _tracker(), _db(), mode="active")
    ex.repay_margin_loan.assert_not_called()


def test_fail_closed_when_journal_lookup_raises():
    ex = _exchange()
    db = _db()
    db.get_unresolved_orders.side_effect = RuntimeError("db down")
    _run(ex, _tracker(), db, mode="active")
    ex.repay_margin_loan.assert_not_called()


# ---- gate: provably flat & covered ----------------------------------------- #


def test_skip_when_free_below_owed():
    ex = _exchange(snapshot=_snapshot(borrowed="0.003", free="0.001"))
    _run(ex, _tracker(), _db(), mode="active")
    ex.repay_margin_loan.assert_not_called()


def test_skip_when_locked_nonzero():
    ex = _exchange(snapshot=_snapshot(locked="0.001"))
    _run(ex, _tracker(), _db(), mode="active")
    ex.repay_margin_loan.assert_not_called()


def test_skip_when_net_exposure_not_flat():
    # netAsset 0.01 ETH * 1600 = $16 > NET_FLAT_DUST_USD ($1) -> a real position exists.
    ex = _exchange(snapshot=_snapshot(net="0.01"))
    _run(ex, _tracker(), _db(), mode="active")
    ex.repay_margin_loan.assert_not_called()


def test_skip_when_borrow_below_dust():
    ex = _exchange(snapshot=_snapshot(borrowed="0", interest="0"))
    _run(ex, _tracker(), _db(), mode="active")
    ex.repay_margin_loan.assert_not_called()


def test_skip_when_snapshot_unavailable():
    ex = _exchange()
    ex.get_margin_account_asset.return_value = None
    ex.get_margin_account_asset.side_effect = None
    _run(ex, _tracker(), _db(), mode="active")
    ex.repay_margin_loan.assert_not_called()


# ---- gate: valid price + cap ----------------------------------------------- #


@pytest.mark.parametrize("bad_price", [None, 0.0, -5.0, float("nan"), float("inf")])
def test_skip_when_price_invalid(bad_price):
    ex = _exchange(price=bad_price)
    _run(ex, _tracker(), _db(), mode="active")
    ex.repay_margin_loan.assert_not_called()


def test_over_cap_alerts_and_does_not_repay():
    # 1 ETH borrowed * 1600 = $1600 >> $50 cap. But keep it net-flat/covered so only the
    # cap gate trips: free covers it, netAsset ~0.
    ex = _exchange(snapshot=_snapshot(borrowed="1", free="1", net="0"))
    db = _db()
    results = _run(ex, _tracker(), db, mode="active")
    ex.repay_margin_loan.assert_not_called()
    assert results and results[0].severity.value == "CRITICAL"
    db.log_audit_event.assert_called_once()


# ---- active-mode post-repay verification ----------------------------------- #


def test_partial_repay_does_not_emit_success():
    # Repay returns ok, but the borrow is still present afterwards -> unresolved, not corrected.
    ex = _exchange(verify_after=_snapshot(borrowed="0.0015", interest="0"))
    results = _run(ex, _tracker(), _db(), mode="active")
    ex.repay_margin_loan.assert_called_once()
    assert results and results[0].status == "unresolved"


def test_repay_call_false_is_unresolved():
    ex = _exchange(repay_ok=False)
    results = _run(ex, _tracker(), _db(), mode="active")
    assert results and results[0].status == "unresolved"


# ---- cooldown -------------------------------------------------------------- #


def test_cooldown_blocks_second_attempt_in_window():
    ex = _exchange(verify_after=_snapshot(borrowed="0", free="0", net="0"))
    cooldown: dict[str, float] = {}
    _run(ex, _tracker(), _db(), mode="active", cooldown=cooldown)
    assert ex.repay_margin_loan.call_count == 1
    # Second call within cooldown window must not act again.
    ex2 = _exchange()
    _run(ex2, _tracker(), _db(), mode="active", cooldown=cooldown)
    ex2.repay_margin_loan.assert_not_called()


def test_periodic_reconciler_uses_shared_cooldown_dict():
    """The engine passes one cooldown dict to both the startup sweep and the periodic
    reconciler, so the two paths can't both repay within one cooldown window."""
    from src.engines.live.reconciliation import PeriodicReconciler

    shared: dict[str, float] = {}
    pr = PeriodicReconciler(
        exchange_interface=MagicMock(),
        position_tracker=MagicMock(),
        db_manager=MagicMock(),
        session_id=1,
        symbols=["ETHUSDT"],
        sweep_cooldown=shared,
    )
    assert pr._sweep_cooldown is shared


# ---- serialization lock (#703) --------------------------------------------- #


def test_lock_registry_shared_per_base_and_reentrant():
    reg = BaseAssetLockRegistry()
    assert reg.lock_for("ETH") is reg.lock_for("ETH")  # same base -> same lock
    assert reg.lock_for("ETH") is not reg.lock_for("BTC")  # different base -> different lock
    lock = reg.lock_for("ETH")
    with lock:  # re-entrant: a held lock can be re-acquired on the same thread
        with lock:
            pass


def test_sweep_holds_lock_across_repay():
    """The repay happens strictly between acquiring and releasing the base-asset lock."""
    events: list[str] = []

    class _TrackingLock:
        def __enter__(self):
            events.append("acquire")
            return self

        def __exit__(self, *a):
            events.append("release")
            return False

    class _Registry:
        def lock_for(self, base):
            return _TrackingLock()

    ex = _exchange(verify_after=_snapshot(borrowed="0", interest="0", free="0", net="0"))
    ex.repay_margin_loan.side_effect = lambda *a, **k: (events.append("repay"), True)[1]
    _run(ex, _tracker(), _db(), mode="active", lock_registry=_Registry())
    assert events == ["acquire", "repay", "release"]


def test_lock_serialises_sweep_behind_an_entry_holding_the_lock():
    """While a (simulated) entry holds the ETH lock, the sweep's repay is blocked, and
    only proceeds once the entry releases — proving sweep-vs-entry serialization (#703)."""
    reg = BaseAssetLockRegistry()
    order: list[str] = []
    holder_has_lock = threading.Event()
    let_holder_finish = threading.Event()

    def holder():  # stands in for an in-flight entry that placed a borrow but isn't tracked yet
        with reg.lock_for("ETH"):
            order.append("entry_acquired")
            holder_has_lock.set()
            let_holder_finish.wait(2.0)
            order.append("entry_released")

    ex = _exchange(verify_after=_snapshot(borrowed="0", interest="0", free="0", net="0"))
    ex.repay_margin_loan.side_effect = lambda *a, **k: (order.append("repay"), True)[1]

    t = threading.Thread(target=holder)
    t.start()
    assert holder_has_lock.wait(2.0)

    done = threading.Event()

    def sweeper():
        _run(ex, _tracker(), _db(), mode="active", lock_registry=reg)
        done.set()

    s = threading.Thread(target=sweeper)
    s.start()
    time.sleep(0.2)  # give the sweeper time to reach (and block on) the lock
    assert "repay" not in order  # blocked: the entry still holds the lock

    let_holder_finish.set()
    assert done.wait(2.0)
    t.join(2.0)
    s.join(2.0)
    # The repay only happened after the entry released the lock.
    assert order.index("entry_released") < order.index("repay")


def test_active_without_lock_registry_refuses_to_repay():
    """Fail-closed: active mode with no lock registry must NOT repay (can't serialise)."""
    ex = _exchange()
    _run(ex, _tracker(), _db(), mode="active", lock_registry=None)
    ex.repay_margin_loan.assert_not_called()
