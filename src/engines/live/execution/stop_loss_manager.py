"""Exchange-facing stop-loss lifecycle for the live trading engine.

Owns every direct exchange call for stop-loss protection — placement after
entry, cancellation before a close, fill/held-inventory queries, re-protection
after a failed close, and the offline-fill detection used by the legacy
startup reconciliation fallback — so ``LiveTradingEngine`` orchestrates these
operations through one handler instead of talking to the exchange directly
(#486).

Thread-safety / lock ownership: this manager holds no locks and owns no
mutable state of its own. It reads ``enable_live_trading``,
``exchange_interface`` and ``order_tracker`` off the engine at call time
(tests and the engine's own startup mutate these after construction), and all
position mutations go through ``LivePositionTracker``'s internal lock.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from src.config.constants import BORROW_DUST_EPSILON
from src.data_providers.exchange_interface import OrderSide, SideEffectType
from src.data_providers.exchange_interface import (
    OrderStatus as ExchangeOrderStatus,
)
from src.engines.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
)
from src.engines.live.order_tracker import OrderTracker
from src.engines.shared.models import PositionSide
from src.infrastructure.logging.events import log_order_event

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
        registered with the order tracker. Returns the stop order id, or
        ``None`` when all attempts failed (the caller owns the emergency-close
        escalation).
        """
        state = self._state
        sl_side = OrderSide.SELL if side == PositionSide.LONG else OrderSide.BUY
        sl_order_id = None
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                sl_order_id = state.exchange_interface.place_stop_loss_order(
                    symbol=symbol,
                    side=sl_side,
                    quantity=quantity,
                    stop_price=stop_price,
                    side_effect_type=SideEffectType.AUTO_REPAY,
                )
                if sl_order_id:
                    break
            except Exception as sl_err:
                logger.warning(
                    "Stop-loss placement attempt %s/%s failed: %s",
                    attempt + 1,
                    max_retries,
                    sl_err,
                )

            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2

        if sl_order_id:
            logger.info(
                "Server-side stop-loss placed: %s @ $%.2f order_id=%s",
                symbol,
                stop_price,
                sl_order_id,
            )
            if position.order_id is not None:
                state.live_position_tracker.set_stop_loss_order_id(position.order_id, sl_order_id)
            if state.order_tracker:
                state.order_tracker.track_order(sl_order_id, symbol)
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
        if not (
            state.enable_live_trading and state.exchange_interface and position.stop_loss_order_id
        ):
            return False
        cancelled = False
        try:
            cancelled = bool(
                state.exchange_interface.cancel_order(position.stop_loss_order_id, position.symbol)
            )
            if cancelled:
                logger.info(
                    "Cancelled stop-loss order %s for %s before close",
                    position.stop_loss_order_id,
                    position.symbol,
                )
        except Exception as e:
            logger.warning(
                "Error cancelling stop-loss order %s for %s: %s",
                position.stop_loss_order_id,
                position.symbol,
                e,
            )
        # Only stop tracking when the cancel is confirmed; otherwise the order may
        # still be live on the exchange and must remain watched.
        if cancelled and state.order_tracker:
            state.order_tracker.stop_tracking(position.stop_loss_order_id)
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

        Mirrors the reconciler's re-placement sizing ``quantity * current/original`` so
        a re-protected stop covers the *remaining* held size, not the full entry size.
        """
        quantity = getattr(position, "quantity", None)
        if not quantity or quantity <= 0:
            return 0.0
        current = getattr(position, "current_size", None)
        original = getattr(position, "original_size", None)
        if current is not None and original is not None and original > 0:
            return float(quantity) * (float(current) / float(original))
        return float(quantity)

    def reprotect(self, position: LivePosition) -> None:
        """Re-place a stop-loss after a failed close left a position momentarily naked.

        Reached only when a market close failed *after* its clean (zero-fill) resting
        stop was cancelled to free the base balance (#710). Re-establish protection
        immediately rather than waiting for the ~120s reconciler — but first verify the
        position is still actually held (the close may be ambiguous / already executed)
        to avoid orphaning a stop, and size for any prior partial exits. The reconciler
        is the ultimate backstop if this attempt cannot run or also fails.
        """
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
            return

        sl_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        sl_order_id = None
        retry_delay = 1.0
        for attempt in range(3):
            try:
                sl_order_id = state.exchange_interface.place_stop_loss_order(
                    symbol=position.symbol,
                    side=sl_side,
                    quantity=float(quantity),
                    stop_price=float(stop_price),
                    side_effect_type=SideEffectType.AUTO_REPAY,
                )
                if sl_order_id:
                    break
            except Exception as e:
                logger.warning(
                    "Re-protect attempt %s/3 for %s failed: %s",
                    attempt + 1,
                    position.symbol,
                    e,
                )
            if attempt < 2:
                time.sleep(retry_delay)
                retry_delay *= 2

        if sl_order_id:
            if position.order_id is not None:
                state.live_position_tracker.set_stop_loss_order_id(position.order_id, sl_order_id)
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
            logger.critical(
                "CRITICAL: %s close failed AND re-placing its stop-loss failed after "
                "retries — position is UNPROTECTED pending the periodic reconciler. "
                "MANUAL REVIEW REQUIRED.",
                position.symbol,
            )
            self._send_alert(
                f"🚨 {position.symbol} UNPROTECTED: close failed and stop-loss "
                f"re-placement failed. Reconciler is the only backstop. REVIEW NOW."
            )

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
