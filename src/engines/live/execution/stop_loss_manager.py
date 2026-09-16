"""Exchange-facing stop-loss lifecycle for the live trading engine.

Owns every direct exchange call for stop-loss protection — placement after
entry, cancellation before a close, fill/held-inventory queries, re-protection
after a failed close, and the offline-fill detection used by the legacy
startup reconciliation fallback — so ``LiveTradingEngine`` orchestrates these
operations through one handler instead of talking to the exchange directly
(#486).

Thread-safety / lock ownership: this manager owns no mutable state of its
own and reads ``enable_live_trading``, ``exchange_interface`` and
``order_tracker`` off the engine at call time (tests and the engine's own
startup mutate these after construction); all position mutations go through
``LivePositionTracker``'s internal lock. The one exception is ``move()``:
its cancel-then-place round-trip mutates exchange order state for a base
asset, so — like ``execute_entry``/``execute_exit`` and the periodic
reconciler's own re-placement — it serialises on
``state._base_asset_locks.lock_for(base_asset)`` so it can never race a
concurrent placement for the same base asset and stack a duplicate resting
stop (#1104/#1108/#1167). ``place_protection()`` and ``reprotect()`` are
always invoked by a caller that already holds that lock across the whole
entry/exit sequence, so they do not acquire it themselves.

Any value read from ``position`` before ``move()`` acquires that lock (e.g.
``stop_loss_order_id``) is only a cheap pre-check for the common no-op case —
never a value carried into the locked section. The periodic reconciler's own
stop-loss re-placement serialises on this same lock and mutates that same
field, so a pre-lock snapshot can go stale by the time the lock is acquired;
the locked body re-reads it fresh (#1179).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from src.config.constants import (
    BORROW_DUST_EPSILON,
    DEFAULT_STOP_LOSS_MAX_RETRIES,
    DEFAULT_STOP_LOSS_RETRY_DELAY,
)
from src.data_providers.exchange_interface import OrderSide, SideEffectType
from src.data_providers.exchange_interface import (
    OrderStatus as ExchangeOrderStatus,
)
from src.engines.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
)
from src.engines.live.order_tracker import OrderTracker
from src.engines.live.trade_close_accounting import held_base_quantity
from src.engines.shared.models import PositionSide
from src.infrastructure.logging.events import log_order_event

if TYPE_CHECKING:
    from src.database.manager import DatabaseManager
    from src.engines.live.reconciliation import BaseAssetLockRegistry

logger = logging.getLogger(__name__)


class StopLossEngineState(Protocol):
    """Live engine state the manager reads at call time.

    Attributes are read dynamically (not captured at construction) because the
    engine assigns ``exchange_interface``/``order_tracker`` during ``start()``
    and tests swap them after the engine is built.
    """

    enable_live_trading: bool
    exchange_interface: Any
    order_tracker: OrderTracker | None
    live_position_tracker: LivePositionTracker
    db_manager: DatabaseManager
    trading_session_id: int | None
    _base_asset_locks: BaseAssetLockRegistry

    # Engine helper the manager escalates to on an exchange-wide rate-limit ban
    # (-1003, #738); called via this backref so subclass/test overrides on the
    # engine still apply (mirrors entry_coordinator.py's identical use).
    def _enter_close_only_mode(self, reason: str | None = None) -> None: ...


class LiveStopLossManager:
    """Places, cancels, verifies and re-places server-side stop-loss orders."""

    def __init__(
        self,
        engine_state: StopLossEngineState,
        send_alert: Callable[[str], object],
    ) -> None:
        """Bind to the engine's live state and its alerting hook.

        Args:
            engine_state: Engine attributes read at call time (see protocol).
            send_alert: Webhook alert dispatcher for UNPROTECTED escalations
                (return value, e.g. delivery success, is ignored).
        """
        self._state = engine_state
        self._send_alert = send_alert

    def _on_rate_limit_ban(self, symbol: str) -> Callable[[BaseException], None]:
        """Build the ``place_or_adopt_stop_loss(on_rate_limit_ban=...)`` callback.

        Fires when a placement attempt hits Binance's -1003 (exchange-wide
        rate-limit ban, #738): every order call fails identically for the
        ban's duration, including the emergency-close a caller might attempt
        next, so this is categorically different from an ordinary placement
        failure (which the existing UNPROTECTED-audit-and-alert branch below
        each call site already covers unchanged). Entering close-only mode
        stops the engine from compounding the outage with more failed order
        attempts while the ban clears; the periodic reconciler restores stop-
        loss PROTECTION once it does. Close-only mode itself does NOT self-
        clear when the ban lifts -- it requires a manual ``resume_trading()``
        after review, same as every other close-only trigger. A Binance ban
        can be as short as ~2 minutes, well inside the window an operator
        needs to notice and act, which is deliberate: this condition is meant
        to get human eyes, not silently resolve itself.
        """

        def _callback(exc: BaseException) -> None:
            self._state._enter_close_only_mode(
                f"stop-loss placement for {symbol} hit an exchange-wide "
                f"rate-limit ban (-1003): {exc}"
            )

        return _callback

    def place_protection(
        self,
        position: LivePosition,
        symbol: str,
        side: PositionSide,
        quantity: float,
        stop_price: float,
    ) -> str | None:
        """Place a server-side stop-loss after entry, with retry/backoff.

        On success the stop order id is recorded on the tracked position and
        registered with the order tracker. On total failure (a fail-closed
        refusal, or all retries exhausted) persists an UNPROTECTED audit row
        and returns ``None`` — the caller owns the emergency-close escalation.
        """
        from src.engines.live.reconciliation import (
            StopPlacementDecision,
            place_or_adopt_stop_loss,
            write_unprotected_audit,
        )

        state = self._state
        sl_side = OrderSide.SELL if side == PositionSide.LONG else OrderSide.BUY

        achieved_price: float = stop_price
        refuse_reason: str | None = None

        def _capture_achieved_price(decision: StopPlacementDecision) -> None:
            nonlocal achieved_price
            price = getattr(decision.existing_order, "stop_price", None)
            if price is not None:
                achieved_price = price

        def _capture_refuse_reason(decision: StopPlacementDecision) -> None:
            nonlocal refuse_reason
            refuse_reason = decision.reason

        # Consult the fail-closed resting-stop check BEFORE placing (#1112), with
        # a DEFAULT_STOP_LOSS_MAX_RETRIES-attempt exponential-backoff retry on
        # the exchange call itself: a stale/nulled tracked id must never cause a
        # second protective order to stack on one still resting on the exchange,
        # and stop_price is passed so an untracked resting stop is only adopted
        # when it is actually protecting at (approximately) the intended price
        # -- a same-side but stale orphan at an unrelated price is not a
        # legitimate adoption target, it's the mis-protection this check
        # exists to catch. on_adopt captures the ACHIEVED price (not the
        # intended one) for last_placed_stop_price below (#1179).
        sl_order_id = place_or_adopt_stop_loss(
            state.exchange_interface,
            symbol=symbol,
            side=sl_side,
            quantity=quantity,
            stop_price=stop_price,
            side_effect_type=SideEffectType.AUTO_REPAY,
            max_attempts=DEFAULT_STOP_LOSS_MAX_RETRIES,
            retry_delay=DEFAULT_STOP_LOSS_RETRY_DELAY,
            retry_log_prefix="Stop-loss placement",
            on_adopt=_capture_achieved_price,
            on_refuse=_capture_refuse_reason,
            on_rate_limit_ban=self._on_rate_limit_ban(symbol),
        )

        if sl_order_id:
            logger.info(
                "Server-side stop-loss placed: %s @ $%.2f order_id=%s",
                symbol,
                stop_price,
                sl_order_id,
            )
            if position.order_id is not None:
                state.live_position_tracker.set_stop_loss_order_id(position.order_id, sl_order_id)
                state.live_position_tracker.set_last_placed_stop_price(
                    position.order_id, float(achieved_price)
                )
            if state.order_tracker:
                state.order_tracker.track_order(sl_order_id, symbol)
        else:
            # Refusal and retry-exhaustion both leave the new entry with no
            # protective stop; either way the caller's emergency-close is the
            # escalation, but the position was UNPROTECTED for at least one
            # cycle and that deserves the same persisted trail as the
            # cancel-then-re-place failures below (#1185).
            write_unprotected_audit(
                state.db_manager,
                state.trading_session_id,
                position,
                "post-entry stop-loss placement failed",
                exchange_reason=refuse_reason,
            )
        return sl_order_id

    def cancel(self, position: LivePosition) -> bool:
        """Cancel a position's resting stop-loss order and stop tracking it.

        Returns True only when the exchange confirms the cancel. The close path uses
        this before a market exit so the stop no longer reserves the base asset
        (otherwise the close is rejected -2010 on margin, #710). A False result means
        the order may still rest, or may have just filled, so the caller must NOT
        submit a close (it would -2010, or over-sell an already-closed position).
        """
        state = self._state
        # Capture the id ONCE. position.stop_loss_order_id is a field on an object
        # shared with the OrderTracker callback thread (LivePositionTracker.positions
        # is a shallow dict copy), and re-reading it after the cancel round-trip can
        # yield None — the unexpected-cancel handler clears it (#1104).
        sl_order_id = position.stop_loss_order_id
        if not (state.enable_live_trading and state.exchange_interface and sl_order_id):
            return False
        # Mark the order as self-cancelled BEFORE issuing the cancel. Binance emits the
        # terminal executionReport on the already-open user socket before the DELETE
        # response returns, so without this the tracker escalates OUR cancel as an
        # unexpected one: a false UNPROTECTED page, and the handler nulls
        # stop_loss_order_id, which makes the reconciler stack a duplicate stop on one
        # still resting and orphan it (#1104). The order stays TRACKED throughout —
        # a stop is cancelled at exactly the moment price is touching it, so a genuine
        # FILL in this window is the likely case, not the exotic one, and it must
        # still reach the fill path.
        if state.order_tracker:
            state.order_tracker.mark_self_cancelled(sl_order_id)
        cancelled = False
        try:
            cancelled = bool(state.exchange_interface.cancel_order(sl_order_id, position.symbol))
            if cancelled:
                logger.info(
                    "Cancelled stop-loss order %s for %s before close",
                    sl_order_id,
                    position.symbol,
                )
        except Exception as e:
            logger.warning(
                "Error cancelling stop-loss order %s for %s: %s",
                sl_order_id,
                position.symbol,
                e,
            )
        if state.order_tracker:
            if cancelled:
                # Only stop tracking when the cancel is confirmed; otherwise the order
                # may still be live on the exchange and must remain watched.
                state.order_tracker.stop_tracking(sl_order_id)
            else:
                # Unconfirmed: the order may still rest, so any later terminal status is
                # genuinely unexpected and must escalate.
                state.order_tracker.clear_self_cancelled(sl_order_id)
        return cancelled

    def filled_quantity(self, position: LivePosition) -> float | None:
        """Return the filled (executed) base quantity of a position's stop-loss order.

        ``0.0`` for an unfilled stop, the filled base quantity for a partial/full fill,
        or ``None`` if the order cannot be read (missing / API error). The close path
        treats ``None`` and any non-zero fill as "unsafe to inline-close" and defers to
        the reconciler — a partially-filled stop means held base != tracked size, so a
        full-size close would over-sell (long) / over-buy (short). (#710)
        """
        state = self._state
        if not (
            state.enable_live_trading and state.exchange_interface and position.stop_loss_order_id
        ):
            return 0.0
        try:
            order = state.exchange_interface.get_order(position.stop_loss_order_id, position.symbol)
        except Exception as e:
            logger.warning(
                "Could not read stop-loss order %s for %s: %s",
                position.stop_loss_order_id,
                position.symbol,
                e,
            )
            return None
        if order is None:
            return None
        return float(getattr(order, "filled_quantity", 0.0) or 0.0)

    def position_still_held(self, position: LivePosition) -> bool:
        """Whether the position's inventory is still actually held on the exchange.

        Checked before an inline re-protect so a stop is not re-placed on a position an
        ambiguous / already-executed close has actually closed (which would orphan a
        stop). Conservative: any unreadable/uncertain state returns ``False`` (do not
        re-place; the reconciler reconciles exchange truth). (#710)
        """
        state = self._state
        if not (state.enable_live_trading and state.exchange_interface):
            return False
        from src.engines.live.reconciliation import PositionReconciler

        base = PositionReconciler._extract_base_asset(position.symbol)
        dust = float(BORROW_DUST_EPSILON)
        try:
            get_asset = getattr(state.exchange_interface, "get_margin_account_asset", None)
            if getattr(state.exchange_interface, "is_margin_mode", False) and callable(get_asset):
                asset = get_asset(base)
                if not asset:
                    return False
                if position.side == PositionSide.SHORT:
                    # A short is still held while base remains borrowed (owed).
                    return float(asset.get("borrowed", 0.0) or 0.0) > dust
                free = float(asset.get("free", 0.0) or 0.0)
                locked = float(asset.get("locked", 0.0) or 0.0)
                return (free + locked) > dust
            # Spot / no margin-asset accessor: long inventory is the base balance.
            bal = state.exchange_interface.get_balance(base)
            if not bal:
                return False
            return (
                float(getattr(bal, "free", 0.0) or 0.0) + float(getattr(bal, "locked", 0.0) or 0.0)
            ) > dust
        except Exception as e:
            logger.warning("Could not confirm held inventory for %s: %s", position.symbol, e)
            return False

    @staticmethod
    def held_protection_quantity(position: LivePosition) -> float:
        """Base quantity to protect, scaled for any prior partial exits.

        Delegates to the shared ``held_base_quantity`` so this mirrors the reconciler's
        re-placement sizing off ONE implementation. ``allow_scale_in=True`` preserves this
        method's previous behavior of scaling past 1.0 for a scale-in rather than
        refusing to size the stop — a real held amount must still be
        protected even for legacy/corrupted state (see the helper's docstring). Falls
        back to the raw (unscaled) quantity when the helper cannot scale (missing/
        invalid current_size or original_size), matching the previous inline guard;
        that fallback path now also rejects a non-finite/negative quantity (e.g. NaN),
        which the previous ``not quantity or quantity <= 0`` check silently let through
        (comparisons against NaN are always False) and could have handed the exchange a
        NaN order quantity.
        """
        quantity = getattr(position, "quantity", None)
        current = getattr(position, "current_size", None)
        original = getattr(position, "original_size", None)
        scaled = held_base_quantity(quantity, current, original, allow_scale_in=True)
        if scaled is not None:
            return scaled
        try:
            qty_f = float(quantity) if quantity is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
        return qty_f if math.isfinite(qty_f) and qty_f > 0 else 0.0

    def reprotect(self, position: LivePosition) -> None:
        """Re-place a stop-loss after a failed close left a position momentarily naked.

        Reached only when a market close failed *after* its clean (zero-fill) resting
        stop was cancelled to free the base balance (#710). Re-establish protection
        immediately rather than waiting for the ~120s reconciler — but first verify the
        position is still actually held (the close may be ambiguous / already executed)
        to avoid orphaning a stop, and size for any prior partial exits. The reconciler
        is the ultimate backstop if this attempt cannot run or also fails. Every CRITICAL
        branch below (no valid price/quantity to re-place, or the placement itself
        failing) also persists an UNPROTECTED audit row, matching ``move()``'s identical
        cancel-succeeded/re-place-failed escalation (#1185).
        """
        from src.engines.live.reconciliation import (
            StopPlacementDecision,
            place_or_adopt_stop_loss,
            write_unprotected_audit,
        )

        state = self._state
        if not (state.enable_live_trading and state.exchange_interface):
            return
        if not self.position_still_held(position):
            logger.warning(
                "%s appears no longer held after a failed close — not re-placing a "
                "stop (the reconciler will reconcile exchange state).",
                position.symbol,
            )
            return

        stop_price = getattr(position, "stop_loss", None)
        quantity = self.held_protection_quantity(position)
        if not stop_price or stop_price <= 0 or quantity <= 0:
            logger.critical(
                "CRITICAL: %s close failed after its stop-loss was cancelled and it "
                "cannot be re-protected inline (stop_price=%s, quantity=%s) — position "
                "is UNPROTECTED pending the reconciler. MANUAL REVIEW REQUIRED.",
                position.symbol,
                stop_price,
                quantity,
            )
            self._send_alert(
                f"🚨 {position.symbol} UNPROTECTED: close failed after stop-loss "
                f"cancel and it could not be re-placed inline. Reconciler backstop "
                f"engaged. MANUAL REVIEW REQUIRED."
            )
            write_unprotected_audit(
                state.db_manager,
                state.trading_session_id,
                position,
                "re-protect: no valid stop price or quantity to re-place",
            )
            return

        sl_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY

        achieved_price: float = float(stop_price)
        refuse_reason: str | None = None

        def _capture_achieved_price(decision: StopPlacementDecision) -> None:
            nonlocal achieved_price
            price = getattr(decision.existing_order, "stop_price", None)
            if price is not None:
                achieved_price = price

        def _capture_refuse_reason(decision: StopPlacementDecision) -> None:
            nonlocal refuse_reason
            refuse_reason = decision.reason

        # Consult the fail-closed resting-stop check BEFORE placing (#1112), with
        # a DEFAULT_STOP_LOSS_MAX_RETRIES-attempt exponential-backoff retry on
        # the exchange call itself: the cancel above should have cleared any
        # resting stop, but confirm rather than assume — a stale tracked id
        # must never let a second order stack.
        # exclude_order_id is the id `cancel()` just cancelled (still on
        # position.stop_loss_order_id -- self-cancel suppression keeps it there
        # rather than nulling it, see cancel()'s own comment): the exchange's
        # open-orders view is not guaranteed to reflect that cancel immediately,
        # so without excluding it a re-appearing cancelled order would be
        # silently re-adopted as if it were a genuine untracked resting stop.
        # on_adopt captures the ACHIEVED price for last_placed_stop_price below
        # (#1179). just_cancelled=True (#1173): this call immediately follows
        # the cancel above, so BinanceProvider's free-base read may still see
        # the just-cancelled stop's pre-cancel `locked` amount for a few
        # seconds (the same eventual-consistency window #1165 fixed on the
        # close path) -- without this, that stale read trips the undersized-
        # protection refusal and leaves a fully sellable position visibly
        # UNPROTECTED.
        sl_order_id = place_or_adopt_stop_loss(
            state.exchange_interface,
            symbol=position.symbol,
            side=sl_side,
            quantity=float(quantity),
            stop_price=float(stop_price),
            side_effect_type=SideEffectType.AUTO_REPAY,
            exclude_order_id=position.stop_loss_order_id,
            just_cancelled=True,
            max_attempts=DEFAULT_STOP_LOSS_MAX_RETRIES,
            retry_delay=DEFAULT_STOP_LOSS_RETRY_DELAY,
            retry_log_prefix="Re-protect",
            on_adopt=_capture_achieved_price,
            on_refuse=_capture_refuse_reason,
            on_rate_limit_ban=self._on_rate_limit_ban(position.symbol),
        )

        if sl_order_id:
            if position.order_id is not None:
                state.live_position_tracker.set_stop_loss_order_id(position.order_id, sl_order_id)
                state.live_position_tracker.set_last_placed_stop_price(
                    position.order_id, float(achieved_price)
                )
            if state.order_tracker:
                state.order_tracker.track_order(sl_order_id, position.symbol)
            logger.warning(
                "Re-protected %s after a failed close: new stop-loss %s @ $%.2f (qty=%.8f)",
                position.symbol,
                sl_order_id,
                float(stop_price),
                float(quantity),
            )
        else:
            # A fail-closed refusal and retry-exhaustion both mean the same
            # thing here: the close already cancelled the old stop, so the
            # position is naked in either case (#1185 unifies what were two
            # differently-worded CRITICAL branches into one, since by this
            # point there is no distinct "still protected" outcome to preserve).
            reason_suffix = f" ({refuse_reason})" if refuse_reason else ""
            logger.critical(
                "CRITICAL: %s close failed AND re-placing its stop-loss failed%s — "
                "position is UNPROTECTED pending the periodic reconciler. "
                "MANUAL REVIEW REQUIRED.",
                position.symbol,
                reason_suffix,
            )
            self._send_alert(
                f"🚨 {position.symbol} UNPROTECTED: close failed and stop-loss "
                f"re-placement failed{reason_suffix}. Reconciler is the only backstop. REVIEW NOW."
            )
            write_unprotected_audit(
                state.db_manager,
                state.trading_session_id,
                position,
                "re-protect: cancel succeeded but re-placement failed",
                exchange_reason=refuse_reason,
            )

    def move(self, position: LivePosition, new_stop_price: float) -> bool:
        """Move a position's resting stop-loss order to a new (ratcheted) price.

        Without this, a trailing stop that ratchets ``position.stop_loss`` up
        (or down, for shorts) never reaches the exchange: the resting order
        keeps protecting at its original price forever, while the engine's own
        exit check trusts ``position.stop_loss`` and believes the position is
        protected at a price the exchange will never actually trigger at
        (#1167). Cancels the currently resting stop and places a new one at
        ``new_stop_price`` via the same guarded placement path as
        ``reprotect`` (#1112).

        No-ops (returns False) when live trading is disabled, ``new_stop_price``
        is not a valid price, there is no resting stop to move yet (protection
        not placed — the next placement attempt already picks up the current
        ``position.stop_loss``), or the position is no longer confirmed held.
        A failed cancel leaves the old order in place rather than risk
        stacking a duplicate on one that may still be resting (the next
        ratchet, or the periodic reconciler, retries); a failed re-placement
        after a successful cancel escalates to CRITICAL/alert/audit exactly
        like ``reprotect`` — the periodic reconciler is the backstop in both
        cases.

        The whole cancel-guard-place sequence serialises on the position's
        base-asset lock (see the class docstring) so it can never race the
        periodic reconciler's own re-placement for the same base asset.
        """
        state = self._state
        if not (state.enable_live_trading and state.exchange_interface):
            return False

        if not math.isfinite(new_stop_price) or new_stop_price <= 0:
            logger.warning(
                "%s trailing stop ratchet produced an invalid price %s — not "
                "touching the exchange.",
                position.symbol,
                new_stop_price,
            )
            return False

        # Cheap unlocked pre-check to skip taking the lock for the common
        # no-op case (no resting stop at all yet). This is only an
        # optimization: it must NOT be reused as the id to cancel/exclude
        # once inside the critical section below. The periodic reconciler's
        # own re-placement serialises on this same lock and can cancel and
        # replace this exact order while move() blocks here waiting for it —
        # a snapshot taken before the lock can go stale by the time it is
        # acquired (#1179). ``_move_locked`` re-reads the field fresh.
        if not position.stop_loss_order_id:
            return False

        from src.engines.live.reconciliation import PositionReconciler

        base = PositionReconciler._extract_base_asset(position.symbol)
        with state._base_asset_locks.lock_for(base):
            return self._move_locked(state, position, new_stop_price)

    def _move_locked(
        self,
        state: StopLossEngineState,
        position: LivePosition,
        new_stop_price: float,
    ) -> bool:
        """The cancel-guard-place body of ``move()``, run under the base-asset lock."""
        # Re-read INSIDE the lock — not move()'s pre-lock snapshot (#1179):
        # the periodic reconciler's Step-2 re-placement serialises on this
        # same lock and may have cancelled and replaced this exact order
        # while move() was blocked waiting for it. Using a stale pre-lock id
        # as `exclude_order_id` below would let a lagging Binance
        # open-orders view make move() ADOPT an order it (or the reconciler)
        # just cancelled, rather than genuinely re-placing at the new price.
        old_order_id = position.stop_loss_order_id
        if not old_order_id:
            logger.info(
                "%s trailing stop ratchet found no resting stop-loss to move "
                "once the lock was acquired (cleared concurrently, e.g. by "
                "the periodic reconciler) — skipping this ratchet; the next "
                "one retries.",
                position.symbol,
            )
            return False

        if not self.position_still_held(position):
            logger.warning(
                "%s appears no longer held while trying to move its trailing "
                "stop to $%.2f — not touching the exchange (the reconciler "
                "will reconcile exchange state).",
                position.symbol,
                new_stop_price,
            )
            return False

        quantity = self.held_protection_quantity(position)
        if quantity <= 0:
            logger.warning(
                "%s trailing stop ratcheted to $%.2f but held quantity is "
                "zero — not moving the exchange stop.",
                position.symbol,
                new_stop_price,
            )
            return False

        if not self.cancel(position):
            logger.warning(
                "Could not confirm cancel of stop-loss %s for %s while moving "
                "the trailing stop to $%.2f — it may still be resting (leaving "
                "it in place rather than risk a duplicate; will retry on the "
                "next ratchet), or a prior pass may have already cancelled and "
                "failed to re-place it, in which case the position is naked "
                "pending the periodic reconciler.",
                old_order_id,
                position.symbol,
                new_stop_price,
            )
            return False

        from src.engines.live.reconciliation import (
            StopPlacementDecision,
            _achieved_price_is_safe_to_ratify,
            place_or_adopt_stop_loss,
            write_unprotected_audit,
        )

        side_is_long = position.side == PositionSide.LONG
        sl_side = OrderSide.SELL if side_is_long else OrderSide.BUY

        # The adopted order (if any) is whatever price is actually resting on
        # the exchange, which the guard only guarantees is within
        # _ADOPT_PRICE_TOLERANCE_FRACTION of new_stop_price -- not equal to
        # it. Capture the ACHIEVED price via on_adopt, not the ratchet's
        # intent, so position.stop_loss (which the engine's own exit check
        # trusts) can be corrected to what the exchange will actually trigger
        # at -- but only when that is the tighter of the two (#1213).
        achieved_price: float = new_stop_price
        refuse_reason: str | None = None

        def _capture_achieved_price(decision: StopPlacementDecision) -> None:
            nonlocal achieved_price
            price = getattr(decision.existing_order, "stop_price", None)
            if price is not None:
                achieved_price = price

        def _capture_refuse_reason(decision: StopPlacementDecision) -> None:
            nonlocal refuse_reason
            refuse_reason = decision.reason

        # Same fail-closed check as reprotect() (#1112), with a
        # DEFAULT_STOP_LOSS_MAX_RETRIES-attempt exponential-backoff retry on
        # the exchange call itself, excluding the order just cancelled above —
        # the exchange's open-orders view is not guaranteed to reflect that
        # cancel immediately.
        new_order_id = place_or_adopt_stop_loss(
            state.exchange_interface,
            symbol=position.symbol,
            side=sl_side,
            quantity=float(quantity),
            stop_price=new_stop_price,
            side_effect_type=SideEffectType.AUTO_REPAY,
            exclude_order_id=old_order_id,
            max_attempts=DEFAULT_STOP_LOSS_MAX_RETRIES,
            retry_delay=DEFAULT_STOP_LOSS_RETRY_DELAY,
            retry_log_prefix="Trailing-stop move",
            on_adopt=_capture_achieved_price,
            on_refuse=_capture_refuse_reason,
            on_rate_limit_ban=self._on_rate_limit_ban(position.symbol),
            just_cancelled=True,
        )

        if new_order_id:
            if position.order_id is not None:
                state.live_position_tracker.set_stop_loss_order_id(position.order_id, new_order_id)
                if achieved_price != new_stop_price:
                    if _achieved_price_is_safe_to_ratify(
                        side_is_long, achieved_price, new_stop_price
                    ):
                        state.live_position_tracker.set_stop_loss_price(
                            position.order_id, float(achieved_price)
                        )
                    else:
                        logger.critical(
                            "Adopted trailing-stop move for %s achieved $%.2f, looser "
                            "than the ratcheted $%.2f -- NOT ratifying into "
                            "position.stop_loss (would weaken the engine's own exit "
                            "trigger below where the ratchet had already advanced it); "
                            "the tracked stop stays tighter than the resting order, a "
                            "divergence inside the periodic drift tolerance and so not "
                            "self-correcting (#1214).",
                            position.symbol,
                            achieved_price,
                            new_stop_price,
                        )
                # Unconditional (unlike set_stop_loss_price above): this is the
                # min-trailing-stop-move floor's baseline, and it must always
                # reflect where the exchange order actually landed, even when
                # that equals the ratchet's own intent (#1179).
                state.live_position_tracker.set_last_placed_stop_price(
                    position.order_id, float(achieved_price)
                )
            if state.order_tracker:
                state.order_tracker.track_order(new_order_id, position.symbol)
            logger.info(
                "Moved trailing stop for %s: %s -> %s @ $%.2f",
                position.symbol,
                old_order_id,
                new_order_id,
                achieved_price,
            )
            return True

        # A fail-closed refusal and retry-exhaustion both mean the same thing
        # here: the cancel above already succeeded, so the position is naked
        # in either case (#1185 unifies what were two differently-worded
        # CRITICAL branches into one).
        reason_suffix = f" ({refuse_reason})" if refuse_reason else ""
        logger.critical(
            "CRITICAL: %s trailing-stop cancelled at the old price but "
            "re-placing at the new $%.2f failed%s — position is UNPROTECTED "
            "pending the periodic reconciler. MANUAL REVIEW REQUIRED.",
            position.symbol,
            new_stop_price,
            reason_suffix,
        )
        self._send_alert(
            f"🚨 {position.symbol} UNPROTECTED: trailing-stop cancel succeeded "
            f"but re-placement at the new price failed{reason_suffix}. Reconciler backstop engaged."
        )
        write_unprotected_audit(
            state.db_manager,
            state.trading_session_id,
            position,
            "trailing-stop move: cancel succeeded but re-placement failed",
            exchange_reason=refuse_reason,
        )
        return False

    def check_filled(self, position: LivePosition) -> tuple[bool, float | None]:
        """Check if a stop-loss order already filled on the exchange."""
        state = self._state
        if (
            not state.enable_live_trading
            or not state.exchange_interface
            or not position.stop_loss_order_id
        ):
            return False, None

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                sl_order = state.exchange_interface.get_order(
                    position.stop_loss_order_id, position.symbol
                )
                if sl_order and sl_order.status == ExchangeOrderStatus.FILLED:
                    logger.info(
                        "Stop-loss order %s already filled at $%.2f - using actual fill price",
                        position.stop_loss_order_id,
                        sl_order.average_price,
                    )
                    return True, sl_order.average_price
                return False, None
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(
                    "Transient error checking stop-loss order %s (attempt %s/%s): %s",
                    position.stop_loss_order_id,
                    attempt + 1,
                    max_attempts,
                    e,
                )
                if attempt < max_attempts - 1:
                    time.sleep(2**attempt)
            except Exception as e:
                logger.error(
                    "Unexpected error checking stop-loss order %s: %s",
                    position.stop_loss_order_id,
                    e,
                    exc_info=True,
                )
                return False, None

        logger.error(
            "Failed to check stop-loss order %s after %s attempts; assuming not filled",
            position.stop_loss_order_id,
            max_attempts,
        )
        log_order_event(
            "sl_check_failed",
            order_id=position.stop_loss_order_id,
            symbol=position.symbol,
        )
        return False, None

    def find_offline_filled_stops(
        self, positions_snapshot: Mapping[str, LivePosition]
    ) -> list[tuple[LivePosition, float | None]]:
        """Detect stop-losses that filled while the engine was offline.

        Legacy startup-reconciliation fallback: lists open orders, and for any
        tracked stop-loss id missing from the exchange, confirms via a direct
        order lookup whether it FILLED. Returns ``(position, fill_price)``
        pairs for confirmed fills; bookkeeping (balance/trade/DB updates) stays
        with the caller. Unlike the other methods on this manager (which
        catch-and-degrade), a failed ``get_open_orders`` PROPAGATES — the
        caller must keep its surrounding try/except so a transient listing
        failure degrades to a logged reconciliation error, not a startup crash.
        """
        state = self._state
        exchange_orders = state.exchange_interface.get_open_orders()
        exchange_order_ids = {order.order_id for order in exchange_orders}

        positions_to_close: list[tuple[LivePosition, float | None]] = []
        for _order_id, position in positions_snapshot.items():
            if position.stop_loss_order_id:
                if position.stop_loss_order_id not in exchange_order_ids:
                    logger.warning(
                        "⚠️ Stop-loss order %s not found on exchange for %s - position may have closed",
                        position.stop_loss_order_id,
                        position.symbol,
                    )
                    try:
                        sl_order = state.exchange_interface.get_order(
                            position.stop_loss_order_id, position.symbol
                        )
                        if sl_order and sl_order.status == ExchangeOrderStatus.FILLED:
                            logger.info(
                                "✅ Confirmed: Stop-loss triggered for %s @ $%s",
                                position.symbol,
                                sl_order.average_price or "unknown",
                            )
                            positions_to_close.append((position, sl_order.average_price))
                    except Exception as e:
                        logger.warning("Could not verify stop-loss order status: %s", e)
        return positions_to_close
