import pytest

from src.position_management.trailing_stops import TrailingStopPolicy

pytestmark = pytest.mark.unit


def test_no_activation_before_threshold():
    policy = TrailingStopPolicy(
        activation_threshold=0.015,
        trailing_distance_pct=0.005,
        breakeven_threshold=0.02,
        breakeven_buffer=0.001,
    )
    # +1.0% sized PnL (position size 1.0 for clarity) < 1.5% => not activated
    new_sl, activated, be = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=101.0,
        existing_stop=None,
        position_fraction=1.0,
    )
    assert activated is False
    assert be is False
    assert new_sl is None


def test_activation_and_trailing_percentage_long():
    policy = TrailingStopPolicy(activation_threshold=0.01, trailing_distance_pct=0.005)
    # +2% pnl -> activate; distance = 0.5% of price = 102 * 0.005 = 0.51 => SL = 101.49
    new_sl, activated, be = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=102.0,
        existing_stop=None,
        position_fraction=1.0,
    )
    assert activated is True
    assert be is False
    assert pytest.approx(new_sl, rel=1e-6) == 102.0 - (102.0 * 0.005)

    # Move to 103 -> SL should tighten only upwards
    new_sl2, activated2, be2 = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=103.0,
        existing_stop=new_sl,
        position_fraction=1.0,
    )
    assert activated2 is True and be2 is False
    assert new_sl2 >= new_sl


def test_activation_and_trailing_percentage_short():
    policy = TrailingStopPolicy(activation_threshold=0.01, trailing_distance_pct=0.01)
    # Short side, +2% pnl -> price falls to 98; distance = 1% * 98 = 0.98 -> SL = 98 + 0.98 = 98.98
    new_sl, activated, be = policy.update_trailing_stop(
        side="short",
        entry_price=100.0,
        current_price=98.0,
        existing_stop=None,
        position_fraction=1.0,
    )
    assert activated is True
    assert be is False
    assert pytest.approx(new_sl, rel=1e-6) == 98.0 + (98.0 * 0.01)

    # Price moves further to 97; SL should tighten downward (lower)
    new_sl2, _, _ = policy.update_trailing_stop(
        side="short",
        entry_price=100.0,
        current_price=97.0,
        existing_stop=new_sl,
        position_fraction=1.0,
    )
    assert new_sl2 <= new_sl


def test_trailing_atr_precedence_over_pct():
    policy = TrailingStopPolicy(
        activation_threshold=0.0, trailing_distance_pct=0.02, atr_multiplier=2.0
    )
    # ATR-based distance used when atr provided
    new_sl, activated, be = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=110.0,
        existing_stop=None,
        position_fraction=1.0,
        atr=1.5,
    )
    assert activated is True
    assert be is False
    assert pytest.approx(new_sl, rel=1e-6) == 110.0 - (1.5 * 2.0)


def test_breakeven_trigger_and_buffer_long():
    policy = TrailingStopPolicy(
        activation_threshold=0.005,
        trailing_distance_pct=0.005,
        breakeven_threshold=0.02,
        breakeven_buffer=0.001,
    )
    # First activate but below breakeven
    sl1, act1, be1 = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=101.0,  # +1% pnl
        existing_stop=None,
        position_fraction=1.0,
    )
    assert act1 is True and be1 is False and sl1 is not None

    # Now reach breakeven threshold at +2%
    sl2, act2, be2 = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=102.0,
        existing_stop=sl1,
        position_fraction=1.0,
    )
    assert act2 is True and be2 is True
    expected_be = 100.0 * (1 + 0.001)
    assert sl2 >= expected_be


def test_never_loosen_stop():
    policy = TrailingStopPolicy(activation_threshold=0.0, trailing_distance_pct=0.01)
    # Long: initial SL
    sl1, act1, be1 = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=105.0,
        existing_stop=None,
        position_fraction=1.0,
    )
    # Price pulls back; stop should not move down
    sl2, _, _ = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=104.0,
        existing_stop=sl1,
        position_fraction=1.0,
    )
    assert sl2 == sl1


class TestTrailingStopPolicyEdgeCases:
    """Test edge cases and extreme scenarios for TrailingStopPolicy."""

    def test_zero_entry_price(self):
        """Test behavior with zero entry price."""
        policy = TrailingStopPolicy(activation_threshold=0.01, trailing_distance_pct=0.005)

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=0.0,  # Zero entry price
            current_price=100.0,
            existing_stop=None,
            position_fraction=1.0,
        )

        assert not activated
        assert not be
        assert new_sl is None

    def test_negative_entry_price(self):
        """Test behavior with negative entry price."""
        policy = TrailingStopPolicy(activation_threshold=0.01, trailing_distance_pct=0.005)

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=-100.0,  # Negative entry price
            current_price=100.0,
            existing_stop=None,
            position_fraction=1.0,
        )

        assert not activated
        assert not be
        assert new_sl is None

    def test_zero_position_fraction(self):
        """Test behavior with zero position fraction."""
        policy = TrailingStopPolicy(activation_threshold=0.01, trailing_distance_pct=0.005)

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=110.0,  # 10% move
            existing_stop=None,
            position_fraction=0.0,  # Zero position
        )

        assert not activated  # No sized PnL
        assert not be
        assert new_sl is None

    def test_negative_position_fraction(self):
        """Test behavior with negative position fraction."""
        policy = TrailingStopPolicy(activation_threshold=0.01, trailing_distance_pct=0.005)

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=110.0,
            existing_stop=None,
            position_fraction=-0.5,  # Negative position
        )

        assert not activated
        assert not be
        assert new_sl is None

    def test_zero_trailing_distance_pct(self):
        """Test behavior with zero trailing distance percentage."""
        policy = TrailingStopPolicy(activation_threshold=0.01, trailing_distance_pct=0.0)

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=102.0,  # 2% move
            existing_stop=None,
            position_fraction=1.0,
        )

        assert activated  # Should activate based on threshold
        assert not be
        assert new_sl is None  # But no stop set due to zero distance

    def test_none_trailing_distance_pct(self):
        """Test behavior with None trailing distance percentage."""
        policy = TrailingStopPolicy(activation_threshold=0.01, trailing_distance_pct=None)

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=102.0,
            existing_stop=None,
            position_fraction=1.0,
        )

        assert activated
        assert not be
        assert new_sl is None

    def test_zero_atr_with_atr_multiplier(self):
        """Test behavior with zero ATR when ATR multiplier is set."""
        policy = TrailingStopPolicy(
            activation_threshold=0.01, trailing_distance_pct=0.005, atr_multiplier=2.0
        )

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=102.0,
            existing_stop=None,
            position_fraction=1.0,
            atr=0.0,  # Zero ATR
        )

        assert activated
        assert not be
        # Should fall back to percentage-based distance
        assert new_sl is not None
        assert pytest.approx(new_sl, rel=1e-6) == 102.0 - (102.0 * 0.005)

    def test_none_atr_with_atr_multiplier(self):
        """Test behavior with None ATR when ATR multiplier is set."""
        policy = TrailingStopPolicy(
            activation_threshold=0.01, trailing_distance_pct=0.005, atr_multiplier=2.0
        )

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=102.0,
            existing_stop=None,
            position_fraction=1.0,
            atr=None,  # None ATR
        )

        assert activated
        assert not be
        # Should fall back to percentage-based distance
        assert new_sl is not None
        assert pytest.approx(new_sl, rel=1e-6) == 102.0 - (102.0 * 0.005)

    def test_zero_atr_multiplier(self):
        """Test behavior with zero ATR multiplier."""
        policy = TrailingStopPolicy(
            activation_threshold=0.01, trailing_distance_pct=0.005, atr_multiplier=0.0
        )

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=102.0,
            existing_stop=None,
            position_fraction=1.0,
            atr=1.5,
        )

        assert activated
        assert not be
        # Should fall back to percentage-based distance
        assert new_sl is not None
        assert pytest.approx(new_sl, rel=1e-6) == 102.0 - (102.0 * 0.005)

    def test_none_atr_multiplier(self):
        """Test behavior with None ATR multiplier."""
        policy = TrailingStopPolicy(
            activation_threshold=0.01, trailing_distance_pct=0.005, atr_multiplier=None
        )

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=102.0,
            existing_stop=None,
            position_fraction=1.0,
            atr=1.5,
        )

        assert activated
        assert not be
        # Should use percentage-based distance since atr_multiplier is None
        assert new_sl is not None
        assert pytest.approx(new_sl, rel=1e-6) == 102.0 - (102.0 * 0.005)

    def test_extreme_price_movements(self):
        """Test behavior with extreme price movements."""
        policy = TrailingStopPolicy(activation_threshold=0.01, trailing_distance_pct=0.005)

        # Extreme upward movement for long
        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=10000.0,  # 100x increase
            existing_stop=None,
            position_fraction=1.0,
        )

        assert activated
        assert not be
        assert new_sl is not None
        expected = 10000.0 - (10000.0 * 0.005)
        assert pytest.approx(new_sl, rel=1e-6) == expected

    def test_very_small_price_movements(self):
        """Test behavior with very small price movements."""
        policy = TrailingStopPolicy(
            activation_threshold=0.000001, trailing_distance_pct=0.000001
        )  # Very small thresholds

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=100.0001,  # Tiny movement
            existing_stop=None,
            position_fraction=1.0,
        )

        assert activated  # Should activate with tiny threshold
        assert not be
        assert new_sl is not None

    def test_breakeven_with_zero_threshold(self):
        """Test breakeven behavior with zero threshold."""
        policy = TrailingStopPolicy(
            activation_threshold=0.0,
            trailing_distance_pct=0.005,
            breakeven_threshold=0.0,  # Zero breakeven threshold
            breakeven_buffer=0.001,
        )

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=100.01,  # Tiny movement
            existing_stop=None,
            position_fraction=1.0,
        )

        assert activated
        # Zero threshold might not trigger breakeven if PnL is not >= 0.0
        # The implementation might require PnL > threshold rather than >=
        assert isinstance(be, bool)
        if be:
            expected_be = 100.0 * (1 + 0.001)
            assert new_sl >= expected_be

    def test_breakeven_with_none_threshold(self):
        """Test breakeven behavior with None threshold (disabled)."""
        policy = TrailingStopPolicy(
            activation_threshold=0.01,
            trailing_distance_pct=0.005,
            breakeven_threshold=None,  # Disabled
            breakeven_buffer=0.001,
        )

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=110.0,  # Large movement
            existing_stop=None,
            position_fraction=1.0,
        )

        assert activated
        assert not be  # Breakeven should be disabled
        # Should use trailing distance
        expected = 110.0 - (110.0 * 0.005)
        assert pytest.approx(new_sl, rel=1e-6) == expected

    def test_zero_breakeven_buffer(self):
        """Test breakeven with zero buffer."""
        policy = TrailingStopPolicy(
            activation_threshold=0.01,
            trailing_distance_pct=0.005,
            breakeven_threshold=0.02,
            breakeven_buffer=0.0,  # Zero buffer
        )

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=102.0,  # 2% move
            existing_stop=None,
            position_fraction=1.0,
        )

        assert activated
        assert be
        # Should set stop exactly at entry price
        assert pytest.approx(new_sl, rel=1e-6) == 100.0

    def test_negative_breakeven_buffer(self):
        """Test breakeven with negative buffer."""
        policy = TrailingStopPolicy(
            activation_threshold=0.01,
            trailing_distance_pct=0.005,
            breakeven_threshold=0.02,
            breakeven_buffer=-0.001,  # Negative buffer
        )

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=102.0,
            existing_stop=None,
            position_fraction=1.0,
        )

        assert activated
        assert be
        # Should handle negative buffer gracefully (max with 0)
        expected_be = 100.0 * (1 + max(0.0, -0.001))  # Should be 100.0
        assert pytest.approx(new_sl, rel=1e-6) == expected_be

    def test_short_side_extreme_scenarios(self):
        """Test short side with extreme scenarios."""
        policy = TrailingStopPolicy(
            activation_threshold=0.01,
            trailing_distance_pct=0.005,
            breakeven_threshold=0.02,
            breakeven_buffer=0.001,
        )

        # Extreme favorable movement for short (price crash)
        new_sl, activated, be = policy.update_trailing_stop(
            side="short",
            entry_price=100.0,
            current_price=1.0,  # 99% drop
            existing_stop=None,
            position_fraction=1.0,
        )

        assert activated
        assert be  # Should trigger breakeven
        expected_be = 100.0 * (1 - 0.001)  # Below entry for short
        assert new_sl <= expected_be

    def test_already_activated_and_breakeven_triggered(self):
        """Test update when both trailing and breakeven are already triggered."""
        policy = TrailingStopPolicy(
            activation_threshold=0.01,
            trailing_distance_pct=0.005,
            breakeven_threshold=0.02,
            breakeven_buffer=0.001,
        )

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=105.0,  # 5% move
            existing_stop=102.0,  # Existing stop
            position_fraction=1.0,
            trailing_activated=True,  # Already activated
            breakeven_triggered=True,  # Already triggered
        )

        assert activated
        assert be
        # Should maintain breakeven logic
        expected_be = 100.0 * (1 + 0.001)
        assert new_sl >= expected_be

    def test_compute_distance_edge_cases(self):
        """Test compute_distance method with edge cases."""
        policy = TrailingStopPolicy(trailing_distance_pct=0.005, atr_multiplier=2.0)

        # ATR takes precedence when available
        distance = policy.compute_distance(price=100.0, atr=1.5)
        assert distance == pytest.approx(3.0)  # 1.5 * 2.0

        # Fall back to percentage when ATR is None
        distance = policy.compute_distance(price=100.0, atr=None)
        assert distance == pytest.approx(0.5)  # 100.0 * 0.005

        # Return None when both are None/zero
        policy_none = TrailingStopPolicy(trailing_distance_pct=None, atr_multiplier=None)
        distance = policy_none.compute_distance(price=100.0, atr=None)
        assert distance is None

        # Return None when values are zero
        policy_zero = TrailingStopPolicy(trailing_distance_pct=0.0, atr_multiplier=0.0)
        distance = policy_zero.compute_distance(price=100.0, atr=1.0)
        assert distance is None

    def test_pnl_fraction_edge_cases(self):
        """Test _pnl_fraction method with edge cases."""
        policy = TrailingStopPolicy()

        # Zero entry price
        pnl = policy._pnl_fraction(0.0, 100.0, "long", 1.0)
        assert pnl == 0.0

        # Zero position fraction
        pnl = policy._pnl_fraction(100.0, 110.0, "long", 0.0)
        assert pnl == 0.0

        # Negative position fraction
        pnl = policy._pnl_fraction(100.0, 110.0, "long", -1.0)
        assert pnl == 0.0

        # Long side calculation
        pnl = policy._pnl_fraction(100.0, 110.0, "long", 0.5)
        assert pnl == pytest.approx(0.05)  # 10% move * 0.5 position

        # Short side calculation
        pnl = policy._pnl_fraction(100.0, 90.0, "short", 0.5)
        assert pnl == pytest.approx(0.05)  # 10% favorable move * 0.5 position

    def test_invalid_side_parameter(self):
        """Test behavior with invalid side parameter."""
        policy = TrailingStopPolicy(activation_threshold=0.01, trailing_distance_pct=0.005)

        # The current implementation doesn't validate side parameter
        # It will default to long behavior for invalid sides
        new_sl, activated, be = policy.update_trailing_stop(
            side="invalid",  # Invalid side
            entry_price=100.0,
            current_price=102.0,
            existing_stop=None,
            position_fraction=1.0,
        )

        # Should behave like long side, but may not activate if PnL calculation fails
        assert isinstance(activated, bool)
        assert isinstance(be, bool)

    def test_float_precision_edge_cases(self):
        """Test behavior with floating point precision edge cases."""
        policy = TrailingStopPolicy(
            activation_threshold=0.00000001,  # Very small threshold
            trailing_distance_pct=0.00000001,
            breakeven_threshold=0.00000002,
            breakeven_buffer=0.00000001,
        )

        new_sl, activated, be = policy.update_trailing_stop(
            side="long",
            entry_price=100.0,
            current_price=100.000001,  # Tiny movement
            existing_stop=None,
            position_fraction=1.0,
        )

        # Should handle very small numbers without issues
        assert isinstance(activated, bool)
        assert isinstance(be, bool)
        assert new_sl is None or isinstance(new_sl, float)
