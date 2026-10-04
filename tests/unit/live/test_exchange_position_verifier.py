"""Unit tests for ExchangePositionVerifier (#683): per-position exchange verification.

The exchange is a hand-written stub (no MagicMock), so a lookup the verifier does not
expect fails loudly instead of returning a truthy auto-attribute.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest

from src.engines.live.position_adoption import (
    FULL_ADOPTION_TOLERANCE,
    AdoptionVerdict,
    ExchangePositionVerifier,
)

pytestmark = pytest.mark.fast

SYMBOL = "ETHUSDT"


@dataclass
class _SpotBalance:
    total: float


class StubExchange:
    """Margin/spot holdings and filters for one base asset."""

    def __init__(
        self,
        *,
        net_asset: float | None = 0.0,
        borrowed: float | None = 0.0,
        spot_total: float | None = 0.0,
        symbol_info: dict[str, Any] | None = None,
        raise_on_lookup: bool = False,
    ) -> None:
        self._net_asset = net_asset
        self._borrowed = borrowed
        self._spot_total = spot_total
        self._symbol_info = symbol_info
        self._raise = raise_on_lookup

    def get_margin_account_asset(self, asset: str) -> dict[str, str] | None:
        if self._raise:
            raise ConnectionError("margin account unavailable")
        if self._net_asset is None:
            return None
        return {"asset": asset, "netAsset": str(self._net_asset)}

    def get_margin_borrowed(self, asset: str) -> float | None:
        if self._raise:
            raise ConnectionError("margin account unavailable")
        return self._borrowed

    def get_balance(self, asset: str) -> _SpotBalance | None:
        if self._raise:
            raise ConnectionError("account unavailable")
        return None if self._spot_total is None else _SpotBalance(self._spot_total)

    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
        return self._symbol_info


def _row(
    position_id: int,
    *,
    quantity: float | None = 1.0,
    side: str = "LONG",
    symbol: str = SYMBOL,
    original_size: float | None = 0.1,
    current_size: float | None = 0.1,
    entry_price: float = 2000.0,
) -> dict[str, Any]:
    return {
        "id": position_id,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "size": 0.1,
        "original_size": original_size,
        "current_size": current_size,
        "entry_price": entry_price,
        "session_id": 1,
    }


def _verify(exchange: StubExchange, rows: list[dict[str, Any]], *, use_margin: bool = True):
    return ExchangePositionVerifier(exchange, use_margin=use_margin).verify(rows)


class TestSinglePosition:
    def test_full_holding_is_adopted_unchanged(self):
        [decision] = _verify(StubExchange(net_asset=1.0), [_row(1, quantity=1.0)])

        assert decision.verdict is AdoptionVerdict.ADOPT
        assert decision.held_share == pytest.approx(1.0)
        assert decision.resized_current_size is None

    def test_holding_within_commission_tolerance_is_adopted_unchanged(self):
        held = 1.0 * (FULL_ADOPTION_TOLERANCE + 0.001)

        [decision] = _verify(StubExchange(net_asset=held), [_row(1, quantity=1.0)])

        assert decision.verdict is AdoptionVerdict.ADOPT

    def test_partial_holding_is_adopted_reduced_with_scaled_current_size(self):
        [decision] = _verify(StubExchange(net_asset=0.4), [_row(1, quantity=1.0)])

        assert decision.verdict is AdoptionVerdict.ADOPT_REDUCED
        assert decision.held_share == pytest.approx(0.4)
        assert decision.expected_quantity == pytest.approx(1.0)
        # current_size 0.1 at quantity 1.0 -> 0.4 held means current_size 0.04.
        assert decision.resized_current_size == pytest.approx(0.04)

    def test_partial_holding_below_old_fifty_percent_threshold_is_still_adopted(self):
        [decision] = _verify(StubExchange(net_asset=0.2), [_row(1, quantity=1.0)])

        assert decision.verdict is AdoptionVerdict.ADOPT_REDUCED
        assert decision.held_share == pytest.approx(0.2)

    def test_expected_quantity_honours_prior_partial_exit(self):
        row = _row(1, quantity=1.0, original_size=0.1, current_size=0.05)

        [decision] = _verify(StubExchange(net_asset=0.5), [row])

        assert decision.expected_quantity == pytest.approx(0.5)
        assert decision.verdict is AdoptionVerdict.ADOPT

    def test_no_holding_is_phantom(self):
        [decision] = _verify(StubExchange(net_asset=0.0), [_row(1)])

        assert decision.verdict is AdoptionVerdict.PHANTOM
        assert decision.held_share == 0.0

    def test_negative_net_asset_is_phantom(self):
        [decision] = _verify(StubExchange(net_asset=-0.3), [_row(1)])

        assert decision.verdict is AdoptionVerdict.PHANTOM

    def test_fully_exited_row_is_phantom(self):
        row = _row(1, quantity=1.0, original_size=0.1, current_size=0.0)

        [decision] = _verify(StubExchange(net_asset=5.0), [row])

        assert decision.verdict is AdoptionVerdict.PHANTOM

    def test_missing_quantity_is_unverifiable(self):
        [decision] = _verify(StubExchange(net_asset=1.0), [_row(1, quantity=None)])

        assert decision.verdict is AdoptionVerdict.UNVERIFIABLE


class TestDust:
    FILTERS = {"min_qty": 0.0001, "min_notional": 5.0}

    def test_holding_below_min_notional_is_phantom(self):
        # 0.001 ETH at $2000 = $2, under the $5 minimum order.
        exchange = StubExchange(net_asset=0.001, symbol_info=self.FILTERS)

        [decision] = _verify(exchange, [_row(1, quantity=0.001)])

        assert decision.verdict is AdoptionVerdict.PHANTOM
        assert decision.held_share == pytest.approx(0.001)

    def test_holding_below_min_qty_is_phantom(self):
        exchange = StubExchange(net_asset=0.00005, symbol_info=self.FILTERS)

        [decision] = _verify(exchange, [_row(1, quantity=0.00005)])

        assert decision.verdict is AdoptionVerdict.PHANTOM

    def test_holding_above_min_notional_is_adopted(self):
        exchange = StubExchange(net_asset=0.01, symbol_info=self.FILTERS)

        [decision] = _verify(exchange, [_row(1, quantity=0.01)])

        assert decision.verdict is AdoptionVerdict.ADOPT

    def test_unknown_filters_never_declare_dust(self):
        exchange = StubExchange(net_asset=0.0001, symbol_info=None)

        [decision] = _verify(exchange, [_row(1, quantity=0.0001)])

        assert decision.verdict is AdoptionVerdict.ADOPT

    def test_reduced_share_is_valued_not_the_tracked_quantity(self):
        # Tracked 1.0 ETH ($2000) but only 0.001 ETH ($2) remains.
        exchange = StubExchange(net_asset=0.001, symbol_info=self.FILTERS)

        [decision] = _verify(exchange, [_row(1, quantity=1.0)])

        assert decision.verdict is AdoptionVerdict.PHANTOM

    def test_dust_is_valued_at_the_price_lookup_when_given(self):
        exchange = StubExchange(net_asset=0.01, symbol_info=self.FILTERS)
        verifier = ExchangePositionVerifier(
            exchange, use_margin=True, price_lookup=lambda _s: 100.0
        )

        [decision] = verifier.verify([_row(1, quantity=0.01)])

        # 0.01 ETH at $100 = $1, under the $5 minimum.
        assert decision.verdict is AdoptionVerdict.PHANTOM


class TestSharedInventoryAcrossSameSymbolRows:
    def test_phantom_cannot_borrow_the_real_positions_inventory(self):
        """The #683 prod case: real #13 (newer) and phantom #12 share one ETH balance."""
        exchange = StubExchange(net_asset=0.0037)
        rows = [_row(12, quantity=0.0033), _row(13, quantity=0.0037)]

        decisions = {d.position_id: d for d in _verify(exchange, rows)}

        assert decisions[13].verdict is AdoptionVerdict.ADOPT
        assert decisions[12].verdict is AdoptionVerdict.PHANTOM

    def test_newest_entry_time_claims_first_even_with_a_lower_id(self):
        exchange = StubExchange(net_asset=1.0)
        older_but_higher_id = {**_row(20, quantity=1.0), "entry_time": datetime(2026, 1, 1)}
        newer_but_lower_id = {**_row(10, quantity=1.0), "entry_time": datetime(2026, 2, 1)}

        decisions = {
            d.position_id: d for d in _verify(exchange, [older_but_higher_id, newer_but_lower_id])
        }

        assert decisions[10].verdict is AdoptionVerdict.ADOPT
        assert decisions[20].verdict is AdoptionVerdict.PHANTOM

    def test_decisions_come_back_in_the_order_rows_were_given(self):
        exchange = StubExchange(net_asset=0.0037)
        rows = [_row(13, quantity=0.0037), _row(12, quantity=0.0033)]

        decisions = _verify(exchange, rows)

        assert [d.position_id for d in decisions] == [13, 12]

    def test_inventory_covering_both_adopts_both(self):
        exchange = StubExchange(net_asset=2.0)
        rows = [_row(1, quantity=1.0), _row(2, quantity=1.0)]

        decisions = _verify(exchange, rows)

        assert [d.verdict for d in decisions] == [AdoptionVerdict.ADOPT, AdoptionVerdict.ADOPT]

    def test_remainder_after_the_newest_claims_goes_to_the_older_row_reduced(self):
        exchange = StubExchange(net_asset=1.5)
        rows = [_row(1, quantity=1.0), _row(2, quantity=1.0)]

        decisions = {d.position_id: d for d in _verify(exchange, rows)}

        assert decisions[2].verdict is AdoptionVerdict.ADOPT
        assert decisions[1].verdict is AdoptionVerdict.ADOPT_REDUCED
        assert decisions[1].held_share == pytest.approx(0.5)


class TestLookupFailureIsFailSafe:
    @pytest.mark.parametrize(
        "exchange",
        [
            StubExchange(net_asset=None),
            StubExchange(raise_on_lookup=True),
        ],
        ids=["unconfirmed", "raises"],
    )
    def test_margin_long_is_unverifiable(self, exchange):
        decisions = _verify(exchange, [_row(1), _row(2)])

        assert [d.verdict for d in decisions] == [AdoptionVerdict.UNVERIFIABLE] * 2

    def test_margin_short_with_unconfirmed_borrow_is_unverifiable(self):
        exchange = StubExchange(borrowed=None)

        [decision] = _verify(exchange, [_row(1, side="SHORT")])

        assert decision.verdict is AdoptionVerdict.UNVERIFIABLE

    def test_spot_balance_lookup_none_is_unverifiable(self):
        exchange = StubExchange(spot_total=None)

        [decision] = _verify(exchange, [_row(1)], use_margin=False)

        assert decision.verdict is AdoptionVerdict.UNVERIFIABLE

    def test_spot_short_cannot_be_verified(self):
        [decision] = _verify(StubExchange(), [_row(1, side="SHORT")], use_margin=False)

        assert decision.verdict is AdoptionVerdict.UNVERIFIABLE


class TestShortsAndSpot:
    def test_margin_short_is_backed_by_borrowed_amount(self):
        exchange = StubExchange(borrowed=1.0, net_asset=0.0)

        [decision] = _verify(exchange, [_row(1, side="SHORT", quantity=1.0)])

        assert decision.verdict is AdoptionVerdict.ADOPT

    def test_margin_short_without_debt_is_phantom(self):
        exchange = StubExchange(borrowed=0.0, net_asset=0.0)

        [decision] = _verify(exchange, [_row(1, side="SHORT", quantity=1.0)])

        assert decision.verdict is AdoptionVerdict.PHANTOM

    def test_spot_long_is_backed_by_wallet_total(self):
        [decision] = _verify(StubExchange(spot_total=1.0), [_row(1)], use_margin=False)

        assert decision.verdict is AdoptionVerdict.ADOPT

    def test_spot_long_without_holding_is_phantom(self):
        [decision] = _verify(StubExchange(spot_total=0.0), [_row(1)], use_margin=False)

        assert decision.verdict is AdoptionVerdict.PHANTOM
