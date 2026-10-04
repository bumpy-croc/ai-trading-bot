"""Exchange verification of OPEN DB positions before a restart adopts them (#683).

A restart must tell two things apart that look identical in the database:

* an orphan: real exchange inventory the engine no longer tracks (adopt it and
  protect it with a stop), and
* a phantom: an OPEN row with no inventory behind it (never adopt it; close it).

Adopting every OPEN row resurrected phantoms (#679), and adopting only the last
session's rows stranded real positions (#677). This module decides per row by
allocating the exchange's actual holdings of the base asset across the rows that
claim it, newest first, so two rows can never both be backed by one balance.

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

from src.engines.live.reconciliation import _margin_net_asset
from src.engines.live.trade_close_accounting import held_base_quantity
from src.engines.shared.commission import split_base_quote

logger = logging.getLogger(__name__)

# A holding within 1% of the tracked quantity is the same position: entry commissions taken
# in the base asset and step-size rounding leave real positions slightly short of the
# recorded fill, and re-sizing them on every restart would be churn.
FULL_ADOPTION_TOLERANCE = 0.99


class AdoptionVerdict(str, Enum):
    """What a restart does with one OPEN DB position."""

    ADOPT = "adopt"
    ADOPT_REDUCED = "adopt_reduced"
    PHANTOM = "phantom"
    UNVERIFIABLE = "unverifiable"


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
            or not self.expected_quantity
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

    def verify(self, rows: list[dict[str, Any]]) -> list[AdoptionDecision]:
        """Classify every row; one decision per row, in the order given.

        Rows are grouped by base asset and side. Within a group the newest row claims
        the exchange holding first, because phantoms are stale rows and the live
        position is the recent one. A group whose holding cannot be read is
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
        """Allocate one asset-and-side holding across ``group``, newest row first."""
        pool = self._read_pool(base_asset, is_short)
        if pool is None:
            return [
                AdoptionDecision(
                    row,
                    AdoptionVerdict.UNVERIFIABLE,
                    f"exchange holding of {base_asset} could not be confirmed",
                )
                for row in group
            ]
        remaining = pool
        decisions: list[AdoptionDecision] = []
        for row in sorted(group, key=self._recency, reverse=True):
            decision = self._decide(row, base_asset, remaining)
            if decision.held_share is not None:
                remaining = max(0.0, remaining - decision.held_share)
            decisions.append(decision)
        return decisions

    @staticmethod
    def _recency(row: dict[str, Any]) -> tuple[datetime, int]:
        """Sort key placing the most recently opened row last (id breaks entry-time ties)."""
        return row.get("entry_time") or datetime.min, int(row["id"])

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

    def _decide(self, row: dict[str, Any], base_asset: str, remaining: float) -> AdoptionDecision:
        """Verdict for ``row`` given the exchange quantity still unclaimed."""
        expected = self._expected_quantity(row)
        if expected is None:
            return AdoptionDecision(
                row, AdoptionVerdict.UNVERIFIABLE, "tracked quantity is missing or invalid"
            )
        if expected <= 0:
            return AdoptionDecision(
                row,
                AdoptionVerdict.PHANTOM,
                "position was fully exited; nothing is expected on the exchange",
                expected_quantity=expected,
                held_share=0.0,
            )
        share = min(expected, remaining)
        if share <= 0:
            return AdoptionDecision(
                row,
                AdoptionVerdict.PHANTOM,
                f"no unclaimed {base_asset} on the exchange",
                expected_quantity=expected,
                held_share=0.0,
            )
        if self._is_dust(row, share):
            return AdoptionDecision(
                row,
                AdoptionVerdict.PHANTOM,
                f"only {share:.8f} {base_asset} remains, below the exchange minimum order",
                expected_quantity=expected,
                held_share=share,
            )
        verdict = (
            AdoptionVerdict.ADOPT
            if share >= expected * FULL_ADOPTION_TOLERANCE
            else AdoptionVerdict.ADOPT_REDUCED
        )
        return AdoptionDecision(
            row,
            verdict,
            f"exchange holds {share:.8f} of the tracked {expected:.8f} {base_asset}",
            expected_quantity=expected,
            held_share=share,
        )

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
