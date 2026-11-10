"""Tests for the quantity-weighted cost basis helper."""

from __future__ import annotations

import math

import pytest

from src.engines.shared.cost_basis import blend_cost_basis

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class TestBlendCostBasis:
    def test_blends_by_units_not_by_notional(self) -> None:
        # 100 bought at 100 (1.0 unit) + 50 bought at 103 (0.4854 units):
        # 150 notional / 1.4854 units = 100.98...
        blended = blend_cost_basis(100.0, 100.0, 103.0, 50.0)

        assert blended == pytest.approx(150.0 / (1.0 + 50.0 / 103.0))

    def test_scale_in_at_the_same_price_keeps_the_cost(self) -> None:
        assert blend_cost_basis(100.0, 0.10, 100.0, 0.03) == pytest.approx(100.0)

    def test_nothing_added_keeps_held_cost(self) -> None:
        assert blend_cost_basis(100.0, 0.10, 120.0, 0.0) == 100.0

    def test_nothing_held_takes_added_price(self) -> None:
        assert blend_cost_basis(100.0, 0.0, 120.0, 0.05) == 120.0

    @pytest.mark.parametrize("bad_price", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_invalid_prices(self, bad_price: float) -> None:
        with pytest.raises(ValueError, match="held_cost_price"):
            blend_cost_basis(bad_price, 0.1, 100.0, 0.1)
        with pytest.raises(ValueError, match="added_price"):
            blend_cost_basis(100.0, 0.1, bad_price, 0.1)

    @pytest.mark.parametrize("bad_size", [-0.1, math.nan, math.inf])
    def test_rejects_invalid_sizes(self, bad_size: float) -> None:
        with pytest.raises(ValueError, match="held_size"):
            blend_cost_basis(100.0, bad_size, 100.0, 0.1)
        with pytest.raises(ValueError, match="added_size"):
            blend_cost_basis(100.0, 0.1, 100.0, bad_size)
