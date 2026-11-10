"""Exchange verification of OPEN DB positions before a restart adopts them (#683).

A restart must tell two things apart that look identical in the database:

* an orphan: real exchange inventory the engine no longer tracks (adopt it and
  protect it with a stop), and
* a phantom: an OPEN row with no inventory behind it (never adopt it; close it).

Adopting every OPEN row resurrected phantoms (#679), and adopting only the last
session's rows stranded real positions (#677). This module decides per row by
allocating the exchange's actual holdings of the base asset across the rows that
claim it, so two rows can never both be backed by one balance. Rows are ranked by
order evidence first (a resting stop proves a live position), then newest first.

A balance snapshot cannot see exits whose fills the order journal has not booked yet
(an offline stop fill, a partial or full exit filled just before a crash). A row with
such activity is handed to the startup reconciler unchanged (``RECONCILE``) instead of
being closed or resized here, because the reconciler books the real fill price, fee
and P&L and a second resize would apply the same delta twice.

The verifier only reads the exchange; applying a verdict (DB moves, closes, alerts)
belongs to the caller.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from src.engines.live.reconciliation import _margin_net_asset, lookup_order_fail_closed
from src.engines.live.trade_close_accounting import held_base_quantity
from src.engines.shared.commission import split_base_quote

logger = logging.getLogger(__name__)

# A holding within 1% of the tracked quantity is the same position: entry commissions taken
# in the base asset and step-size rounding leave real positions slightly short of the
# recorded fill, and re-sizing them on every restart would be churn.
FULL_ADOPTION_TOLERANCE = 0.99

# DB order types that change the inventory a position holds, and the DB statuses that mean
# the exchange outcome has not been booked yet.
_INVENTORY_ORDER_TYPES = frozenset({"PARTIAL_EXIT", "FULL_EXIT", "SCALE_IN"})
_UNRESOLVED_ORDER_STATUSES = frozenset(
    {"PENDING", "OPEN", "PENDING_SUBMIT", "SUBMITTED", "UNKNOWN", "UNRESOLVED"}
)


class AdoptionVerdict(str, Enum):
    """What a restart does with one OPEN DB position."""

    ADOPT = "adopt"
    ADOPT_REDUCED = "adopt_reduced"
    RECONCILE = "reconcile"
    PHANTOM = "phantom"
    UNVERIFIABLE = "unverifiable"


class _StopEvidence(Enum):
    """What the exchange says about a position's stop-loss order."""

    NONE = "none"
    RESTING = "resting"
    EXECUTED = "executed"
    UNCONFIRMED = "unconfirmed"


@dataclass(frozen=True)
class AdoptionDecision:
    """The verdict for one OPEN DB position and the evidence behind it."""

    row: dict[str, Any]
    verdict: AdoptionVerdict
    reason: str
    expected_quantity: float | None = None
    held_share: float | None = None

    @property
    def position_id(self) -> int:
        """DB id of the position the verdict applies to."""
        return int(self.row["id"])

    @property
    def resized_current_size(self) -> float | None:
        """``current_size`` that makes the held quantity equal the exchange share.

        Only meaningful for ``ADOPT_REDUCED``. Scaling ``current_size`` is how every
        consumer (stop sizing, close quantity, holdings checks) already represents a
        position holding less than it was opened with.
        """
        if (
            self.verdict is not AdoptionVerdict.ADOPT_REDUCED
            or self.expected_quantity is None
            or self.expected_quantity <= 0
            or self.held_share is None
        ):
            return None
        current = self.row.get("current_size")
        if current is None:
            current = self.row.get("size")
        if current is None:
            return None
        return float(current) * (self.held_share / self.expected_quantity)


class ExchangePositionVerifier:
    """Allocates exchange holdings across OPEN DB positions to classify each one."""

    def __init__(
        self,
        exchange: Any,
        *,
        use_margin: bool,
        price_lookup: Callable[[str], float | None] | None = None,
    ) -> None:
        """Bind to the exchange; ``price_lookup`` values dust (entry price when absent)."""
        self._exchange = exchange
        self._use_margin = use_margin
        self._price_lookup = price_lookup
        self._stop_evidence_cache: dict[int, _StopEvidence] = {}

    def verify(self, rows: list[dict[str, Any]]) -> list[AdoptionDecision]:
        """Classify every row; one decision per row, in the order given.

        Rows are grouped by base asset and side and each group shares one exchange
        holding. A group whose holding (or whose stop evidence) cannot be confirmed is
        ``UNVERIFIABLE`` as a whole: nothing is adopted and nothing is closed blind.
        """
        groups: dict[tuple[str, bool], list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(self._group_key(row), []).append(row)

        decisions: dict[int, AdoptionDecision] = {}
        for (base_asset, is_short), group in groups.items():
            for decision in self._verify_group(base_asset, is_short, group):
                decisions[decision.position_id] = decision
        return [decisions[int(row["id"])] for row in rows]

    @staticmethod
    def _group_key(row: dict[str, Any]) -> tuple[str, bool]:
        """(base asset, is-short) the row's exchange claim is scoped to."""
        base_asset = split_base_quote(str(row["symbol"]))[0]
        return base_asset, str(row.get("side")).lower() == "short"

    def _verify_group(
        self, base_asset: str, is_short: bool, group: list[dict[str, Any]]
    ) -> list[AdoptionDecision]:
        """Allocate one asset-and-side holding across ``group``, best-evidenced row first."""
        pool = self._read_pool(base_asset, is_short)
        if pool is None:
            return self._all_unverifiable(
                group, f"exchange holding of {base_asset} could not be confirmed"
            )
        if len(group) > 1:
            # Several rows compete for one balance: a resting stop is the strongest proof of
            # which row is live, so it needs a confirmed answer for every row.
            for row in group:
                if self._stop_evidence(row) is _StopEvidence.UNCONFIRMED:
                    return self._all_unverifiable(
                        group, "a stop-loss order could not be looked up to rank the rows"
                    )
        remaining = pool
        decisions: list[AdoptionDecision] = []
        ranked = sorted(group, key=self._rank, reverse=True) if len(group) > 1 else group
        for row in ranked:
            decision = self._decide(row, base_asset, pool, remaining)
            if decision.held_share is not None:
                remaining = max(0.0, remaining - decision.held_share)
            decisions.append(decision)
        if len(group) > 1 and any(d.verdict is AdoptionVerdict.UNVERIFIABLE for d in decisions):
            # A row of unknown size may own part of the balance, so no other row's share is
            # trustworthy.
            return self._all_unverifiable(group, "a row in the group has no usable quantity")
        return decisions

    @staticmethod
    def _all_unverifiable(group: list[dict[str, Any]], reason: str) -> list[AdoptionDecision]:
        """``UNVERIFIABLE`` for every row of ``group``."""
        return [AdoptionDecision(row, AdoptionVerdict.UNVERIFIABLE, reason) for row in group]

    def _rank(self, row: dict[str, Any]) -> tuple[bool, datetime, int]:
        """Sort key; the caller sorts descending so the highest key claims inventory first.

        A row whose stop is resting on the exchange is live; ties go to the most recently
        opened row (id breaks entry-time ties). The engine opens one position per symbol,
        so a newer row exists only after the older one left the exchange.
        """
        resting = self._stop_evidence(row) is _StopEvidence.RESTING
        return resting, row.get("entry_time") or datetime.min, int(row["id"])

    def _read_pool(self, base_asset: str, is_short: bool) -> float | None:
        """Exchange quantity backing positions in ``base_asset``; None when unconfirmed.

        Margin longs are backed by ``netAsset`` (net of borrow and interest), margin shorts
        by the raw borrowed amount, spot longs by the wallet total. A spot short has no
        holding to read, so it cannot be verified.
        """
        try:
            if self._use_margin and is_short:
                getter = getattr(self._exchange, "get_margin_borrowed", None)
                raw = getter(base_asset) if callable(getter) else None
                pool = None if raw is None else float(raw)
            elif self._use_margin:
                pool = _margin_net_asset(self._exchange, base_asset)
            elif is_short:
                return None
            else:
                balance = self._exchange.get_balance(base_asset)
                pool = None if balance is None else float(balance.total)
        except Exception as e:  # noqa: BLE001 - any lookup failure means "unconfirmed"
            logger.warning("Holding lookup for %s failed during adoption: %s", base_asset, e)
            return None
        if pool is None or not math.isfinite(pool):
            return None
        return max(0.0, pool)

    def _decide(
        self, row: dict[str, Any], base_asset: str, pool: float, remaining: float
    ) -> AdoptionDecision:
        """Verdict for ``row`` given the pool and the exchange quantity still unclaimed."""
        expected = self._expected_quantity(row)
        if expected is None:
            if pool <= 0:
                return self._phantom_or_reconcile(
                    row, f"no {base_asset} on the exchange", expected, 0.0
                )
            return AdoptionDecision(
                row, AdoptionVerdict.UNVERIFIABLE, "tracked quantity is missing or invalid"
            )
        if expected <= 0:
            return self._phantom_or_reconcile(
                row, "position was fully exited; nothing is expected on the exchange", expected, 0.0
            )
        share = min(expected, remaining)
        if share <= 0:
            return self._phantom_or_reconcile(
                row, f"no unclaimed {base_asset} on the exchange", expected, 0.0
            )
        if self._is_dust(row, share):
            return self._phantom_or_reconcile(
                row,
                f"only {share:.8f} {base_asset} remains, below the exchange minimum order",
                expected,
                share,
            )
        reason = f"exchange holds {share:.8f} of the tracked {expected:.8f} {base_asset}"
        if share >= expected * FULL_ADOPTION_TOLERANCE:
            return AdoptionDecision(
                row, AdoptionVerdict.ADOPT, reason, expected_quantity=expected, held_share=share
            )
        return self._reduced_or_reconcile(row, reason, expected, share)

    def _phantom_or_reconcile(
        self, row: dict[str, Any], reason: str, expected: float | None, share: float
    ) -> AdoptionDecision:
        """``PHANTOM`` unless unbooked exit activity explains the missing inventory."""
        override = self._exit_activity_override(row, reason, expected, share)
        if override is not None:
            return override
        return AdoptionDecision(
            row, AdoptionVerdict.PHANTOM, reason, expected_quantity=expected, held_share=share
        )

    def _reduced_or_reconcile(
        self, row: dict[str, Any], reason: str, expected: float, share: float
    ) -> AdoptionDecision:
        """``ADOPT_REDUCED`` unless unbooked exit activity explains the missing inventory."""
        override = self._exit_activity_override(row, reason, expected, share)
        if override is not None:
            return override
        return AdoptionDecision(
            row,
            AdoptionVerdict.ADOPT_REDUCED,
            reason,
            expected_quantity=expected,
            held_share=share,
        )

    def _exit_activity_override(
        self, row: dict[str, Any], reason: str, expected: float | None, share: float
    ) -> AdoptionDecision | None:
        """``RECONCILE``/``UNVERIFIABLE`` when the shortfall may be an unbooked exit fill.

        None means nothing explains the shortfall, so the caller may close or resize.
        """
        if self._has_unresolved_inventory_order(row):
            return AdoptionDecision(
                row,
                AdoptionVerdict.RECONCILE,
                f"{reason}; an exit/scale-in order is unresolved, left to the startup reconciler",
                expected_quantity=expected,
                held_share=share,
            )
        evidence = self._stop_evidence(row)
        if evidence is _StopEvidence.EXECUTED:
            return AdoptionDecision(
                row,
                AdoptionVerdict.RECONCILE,
                f"{reason}; its stop-loss executed, left to the startup reconciler to book",
                expected_quantity=expected,
                held_share=share,
            )
        if evidence is _StopEvidence.UNCONFIRMED:
            return AdoptionDecision(
                row,
                AdoptionVerdict.UNVERIFIABLE,
                f"{reason}; its stop-loss order could not be looked up",
            )
        return None

    @staticmethod
    def _has_unresolved_inventory_order(row: dict[str, Any]) -> bool:
        """True when the DB holds an exit or scale-in order whose outcome is not booked."""
        for order in row.get("orders") or []:
            if (
                str(order.get("order_type")).upper() in _INVENTORY_ORDER_TYPES
                and str(order.get("status")).upper() in _UNRESOLVED_ORDER_STATUSES
            ):
                return True
        return False

    def _stop_evidence(self, row: dict[str, Any]) -> _StopEvidence:
        """What the exchange says about the row's stop-loss order (cached per row)."""
        position_id = int(row["id"])
        cached = self._stop_evidence_cache.get(position_id)
        if cached is not None:
            return cached
        evidence = self._lookup_stop_evidence(row)
        self._stop_evidence_cache[position_id] = evidence
        return evidence

    def _lookup_stop_evidence(self, row: dict[str, Any]) -> _StopEvidence:
        """Classify the row's stop-loss order; ``UNCONFIRMED`` when the lookup fails."""
        stop_id = row.get("stop_loss_order_id")
        if not stop_id:
            return _StopEvidence.NONE
        try:
            order, confirmed = lookup_order_fail_closed(
                self._exchange, str(stop_id), str(row["symbol"])
            )
        except Exception as e:  # noqa: BLE001 - any lookup failure means "unconfirmed"
            logger.warning("Stop-loss %s lookup failed during adoption: %s", stop_id, e)
            return _StopEvidence.UNCONFIRMED
        if not confirmed:
            return _StopEvidence.UNCONFIRMED
        if order is None:
            return _StopEvidence.NONE
        status = str(getattr(getattr(order, "status", None), "value", "")).upper()
        filled = _positive_float(getattr(order, "filled_quantity", None))
        if status in ("FILLED", "PARTIALLY_FILLED") or filled is not None:
            return _StopEvidence.EXECUTED
        if status == "PENDING":
            return _StopEvidence.RESTING
        return _StopEvidence.NONE

    @staticmethod
    def _expected_quantity(row: dict[str, Any]) -> float | None:
        """Base quantity the row should still hold, scaled for partial exits.

        Uses the shared ``held_base_quantity`` like every other holdings check, with
        ``allow_scale_in=True`` because this is a presence threshold, not a persisted record.
        """
        try:
            qty = float(row["quantity"]) if row.get("quantity") is not None else None
        except (TypeError, ValueError):
            return None
        if qty is None or not math.isfinite(qty) or qty <= 0:
            return None
        scaled = held_base_quantity(
            qty, row.get("current_size"), row.get("original_size"), allow_scale_in=True
        )
        return qty if scaled is None else scaled

    def _is_dust(self, row: dict[str, Any], share: float) -> bool:
        """True when ``share`` is below the symbol's minimum order quantity or notional.

        Inventory below the minimum can be neither sold nor protected by a stop, so
        adopting it only produces rejected-order churn. Unknown filters never declare dust.
        """
        symbol = str(row["symbol"])
        try:
            info = self._exchange.get_symbol_info(symbol)
        except Exception as e:  # noqa: BLE001 - filters are best-effort
            logger.warning("Symbol filters for %s unavailable during adoption: %s", symbol, e)
            return False
        if not isinstance(info, dict):
            return False
        min_qty = _positive_float(info.get("min_qty"))
        if min_qty is not None and share < min_qty:
            return True
        min_notional = _positive_float(info.get("min_notional"))
        price = self._reference_price(row)
        return min_notional is not None and price is not None and share * price < min_notional

    def _reference_price(self, row: dict[str, Any]) -> float | None:
        """Current price for valuing dust, falling back to the position's entry price."""
        if self._price_lookup is not None:
            try:
                price = _positive_float(self._price_lookup(str(row["symbol"])))
            except Exception as e:  # noqa: BLE001 - valuation is best-effort
                logger.warning("Price lookup for %s failed during adoption: %s", row["symbol"], e)
                price = None
            if price is not None:
                return price
        return _positive_float(row.get("entry_price"))


def _positive_float(value: Any) -> float | None:
    """``value`` as a finite positive float, else None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) and result > 0 else None
