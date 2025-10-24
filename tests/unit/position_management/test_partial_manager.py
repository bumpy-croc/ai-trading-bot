import pytest

from src.position_management.partial_manager import PartialExitPolicy, PositionState

pytestmark = pytest.mark.unit


def test_partial_exits_multiple_targets_long():
    policy = PartialExitPolicy(
        exit_targets=[0.03, 0.06, 0.10],
        exit_sizes=[0.25, 0.25, 0.50],
    )
    pos = PositionState(entry_price=100.0, side="long", original_size=0.5, current_size=0.5)

    # At +2% no exits
    actions = policy.check_partial_exits(pos, 102.0)
    assert actions == []

    # At +3% first exit
    actions = policy.check_partial_exits(pos, 103.0)
    assert len(actions) == 1
    assert actions[0]["type"] == "partial_exit"
    assert actions[0]["size"] == 0.25
    assert actions[0]["target_level"] == 0
    policy.apply_partial_exit(pos, executed_size_fraction_of_original=0.25, price=103.0)
    # 0.25 of original (0.5) => 0.125 reduction
    assert pos.current_size == pytest.approx(0.5 - 0.125)
    assert pos.partial_exits_taken == 1

    # At +11% both remaining exits are triggered (0.06 and 0.10)
    actions = policy.check_partial_exits(pos, 111.0)
    # Next two levels (0.25 and 0.5 of original)
    assert len(actions) == 2
    sizes = [a["size"] for a in actions]
    assert sizes == [0.25, 0.50]
    for a in actions:
        policy.apply_partial_exit(pos, executed_size_fraction_of_original=a["size"], price=111.0)
    # Total reduction = 0.125 (first) + 0.125 + 0.25 = 0.5 → current_size 0.0
    assert pos.current_size == pytest.approx(0.0, abs=1e-9)
    assert pos.partial_exits_taken == 3


def test_scale_in_long_with_thresholds():
    policy = PartialExitPolicy(
        exit_targets=[0.05],
        exit_sizes=[0.5],
        scale_in_thresholds=[0.02, 0.05],
        scale_in_sizes=[0.25, 0.25],
        max_scale_ins=2,
    )
    pos = PositionState(entry_price=100.0, side="long", original_size=0.2, current_size=0.2)

    # At +1% no scale-in
    assert policy.check_scale_in_opportunity(pos, 101.0) is None

    # At +2% first scale-in (adds 0.25 * original 0.2 = 0.05)
    act = policy.check_scale_in_opportunity(pos, 102.0)
    assert act is not None and act["type"] == "scale_in" and act["size"] == 0.25
    policy.apply_scale_in(pos, add_size_fraction_of_original=0.25, price=102.0)
    assert pos.current_size == pytest.approx(0.25)
    assert pos.scale_ins_taken == 1

    # At +6% second scale-in (adds another 0.05)
    act = policy.check_scale_in_opportunity(pos, 106.0)
    assert act is not None and act["size"] == 0.25
    policy.apply_scale_in(pos, add_size_fraction_of_original=0.25, price=106.0)
    assert pos.current_size == pytest.approx(0.30)
    assert pos.scale_ins_taken == 2

    # Exceed max scale-ins
    assert policy.check_scale_in_opportunity(pos, 110.0) is None


def test_edge_cases_small_position_and_exact_hits():
    policy = PartialExitPolicy(
        exit_targets=[0.01, 0.02],
        exit_sizes=[0.6, 0.4],
        scale_in_thresholds=[0.015],
        scale_in_sizes=[0.2],
        max_scale_ins=1,
    )
    pos = PositionState(entry_price=100.0, side="long", original_size=0.01, current_size=0.01)

    # Exact target hit at +1%
    actions = policy.check_partial_exits(pos, 101.0)
    assert len(actions) == 1 and actions[0]["size"] == 0.6

    # Exact scale-in threshold
    act = policy.check_scale_in_opportunity(pos, 101.5)
    assert act is not None and act["size"] == 0.2


def test_strategy_integration():
    """Test that strategies can provide partial operations configuration"""
    from src.strategies.ml_adaptive import create_ml_adaptive_strategy

    strategy = create_ml_adaptive_strategy()
    overrides = strategy.get_risk_overrides()

    assert overrides is not None
    assert "partial_operations" in overrides

    partial_config = overrides["partial_operations"]
    assert "exit_targets" in partial_config
    assert "exit_sizes" in partial_config
    assert "scale_in_thresholds" in partial_config
    assert "scale_in_sizes" in partial_config
    assert "max_scale_ins" in partial_config

    # Verify the configuration values
    assert partial_config["exit_targets"] == [0.03, 0.06, 0.10]
    assert partial_config["exit_sizes"] == [0.25, 0.25, 0.50]
    assert partial_config["scale_in_thresholds"] == [0.02, 0.05]
    assert partial_config["scale_in_sizes"] == [0.25, 0.25]
    assert partial_config["max_scale_ins"] == 2


class TestPartialExitPolicyEdgeCases:
    """Test edge cases and boundary conditions for PartialExitPolicy."""

    def test_empty_exit_targets(self):
        """Test policy with empty exit targets."""
        policy = PartialExitPolicy(
            exit_targets=[],
            exit_sizes=[],
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=1.0)

        actions = policy.check_partial_exits(pos, 110.0)
        assert actions == []

    def test_mismatched_exit_arrays_length(self):
        """Test policy with mismatched exit targets and sizes arrays."""
        with pytest.raises(ValueError, match="exit_targets and exit_sizes must have equal length"):
            PartialExitPolicy(
                exit_targets=[0.05, 0.10],
                exit_sizes=[0.5],  # Different length
            )

    def test_empty_scale_in_targets(self):
        """Test policy with empty scale-in targets."""
        policy = PartialExitPolicy(
            exit_targets=[0.05],
            exit_sizes=[1.0],
            scale_in_thresholds=[],
            scale_in_sizes=[],
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=1.0)

        action = policy.check_scale_in_opportunity(pos, 105.0)
        assert action is None

    def test_mismatched_scale_in_arrays_length(self):
        """Test policy with mismatched scale-in arrays."""
        with pytest.raises(
            ValueError, match="scale_in_thresholds and scale_in_sizes must have equal length"
        ):
            PartialExitPolicy(
                exit_targets=[0.05],
                exit_sizes=[1.0],
                scale_in_thresholds=[0.02, 0.05],
                scale_in_sizes=[0.25],  # Different length
            )

    def test_negative_exit_targets_validation(self):
        """Test policy validation with negative exit targets."""
        with pytest.raises(ValueError, match="exit_targets must be positive"):
            PartialExitPolicy(
                exit_targets=[-0.05, 0.10],  # Negative target
                exit_sizes=[0.5, 0.5],
            )

    def test_zero_exit_targets_validation(self):
        """Test policy validation with zero exit targets."""
        with pytest.raises(ValueError, match="exit_targets must be positive"):
            PartialExitPolicy(
                exit_targets=[0.0, 0.10],  # Zero target
                exit_sizes=[0.5, 0.5],
            )

    def test_zero_exit_sizes_validation(self):
        """Test policy validation with zero exit sizes."""
        with pytest.raises(ValueError, match="exit_sizes must be in \\(0, 1\\]"):
            PartialExitPolicy(
                exit_targets=[0.05],
                exit_sizes=[0.0],  # Zero size
            )

    def test_exit_sizes_greater_than_one_validation(self):
        """Test policy validation with exit sizes > 1.0."""
        with pytest.raises(ValueError, match="exit_sizes must be in \\(0, 1\\]"):
            PartialExitPolicy(
                exit_targets=[0.05],
                exit_sizes=[1.5],  # > 100%
            )

    def test_negative_exit_sizes_validation(self):
        """Test policy validation with negative exit sizes."""
        with pytest.raises(ValueError, match="exit_sizes must be in \\(0, 1\\]"):
            PartialExitPolicy(
                exit_targets=[0.05],
                exit_sizes=[-0.5],  # Negative size
            )

    def test_negative_scale_in_thresholds_validation(self):
        """Test policy validation with negative scale-in thresholds."""
        with pytest.raises(ValueError, match="scale_in_thresholds must be positive"):
            PartialExitPolicy(
                exit_targets=[0.05],
                exit_sizes=[0.5],
                scale_in_thresholds=[-0.02],
                scale_in_sizes=[0.25],
            )

    def test_invalid_scale_in_sizes_validation(self):
        """Test policy validation with invalid scale-in sizes."""
        with pytest.raises(ValueError, match="scale_in_sizes must be in \\(0, 1\\]"):
            PartialExitPolicy(
                exit_targets=[0.05],
                exit_sizes=[0.5],
                scale_in_thresholds=[0.02],
                scale_in_sizes=[0.0],  # Invalid size
            )

    def test_negative_max_scale_ins_validation(self):
        """Test policy validation with negative max_scale_ins."""
        with pytest.raises(ValueError, match="max_scale_ins must be >= 0"):
            PartialExitPolicy(
                exit_targets=[0.05],
                exit_sizes=[0.5],
                max_scale_ins=-1,
            )

    def test_zero_original_size_position(self):
        """Test with position having zero original size."""
        policy = PartialExitPolicy(
            exit_targets=[0.05],
            exit_sizes=[0.5],
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=0.0, current_size=0.0)

        # The implementation doesn't check original_size in check_partial_exits
        # It only checks if PnL target is met, so it will return an action
        actions = policy.check_partial_exits(pos, 105.0)
        assert len(actions) == 1  # Will trigger based on PnL alone

    def test_negative_current_size_position(self):
        """Test with position having negative current size."""
        policy = PartialExitPolicy(
            exit_targets=[0.05],
            exit_sizes=[0.5],
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=-0.5)

        # Should still check for exits based on original size and entry price
        actions = policy.check_partial_exits(pos, 105.0)
        assert len(actions) == 1

    def test_zero_entry_price(self):
        """Test with zero entry price."""
        policy = PartialExitPolicy(
            exit_targets=[0.05],
            exit_sizes=[0.5],
        )
        pos = PositionState(entry_price=0.0, side="long", original_size=1.0, current_size=1.0)

        # Should not crash but may not work as expected
        actions = policy.check_partial_exits(pos, 105.0)
        # Behavior depends on implementation - should handle gracefully
        assert isinstance(actions, list)

    def test_negative_entry_price(self):
        """Test with negative entry price."""
        policy = PartialExitPolicy(
            exit_targets=[0.05],
            exit_sizes=[0.5],
        )
        pos = PositionState(entry_price=-100.0, side="long", original_size=1.0, current_size=1.0)

        actions = policy.check_partial_exits(pos, 105.0)
        assert isinstance(actions, list)

    def test_short_side_exit_logic(self):
        """Test exit logic for short positions."""
        policy = PartialExitPolicy(
            exit_targets=[0.05, 0.10],  # Favorable moves for short
            exit_sizes=[0.5, 0.5],
        )
        pos = PositionState(entry_price=100.0, side="short", original_size=1.0, current_size=1.0)

        # Price drops 5% (favorable for short)
        actions = policy.check_partial_exits(pos, 95.0)
        assert len(actions) == 1
        assert actions[0]["target_level"] == 0

        # Price drops 12% (both targets hit)
        actions = policy.check_partial_exits(pos, 88.0)
        assert len(actions) == 2

    def test_unsorted_exit_targets(self):
        """Test policy with unsorted exit targets."""
        policy = PartialExitPolicy(
            exit_targets=[0.10, 0.05, 0.15],  # Unsorted
            exit_sizes=[0.3, 0.3, 0.4],
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=1.0)

        # At 12% move, should trigger targets 0 and 2 (0.10 and 0.05)
        actions = policy.check_partial_exits(pos, 112.0)
        target_levels = [a["target_level"] for a in actions]
        assert 0 in target_levels  # 0.10 target
        assert 1 in target_levels  # 0.05 target

    def test_apply_partial_exit_edge_cases(self):
        """Test apply_partial_exit with edge cases."""
        policy = PartialExitPolicy(
            exit_targets=[0.05],
            exit_sizes=[0.5],
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=1.0)

        # Apply zero size exit
        policy.apply_partial_exit(pos, executed_size_fraction_of_original=0.0, price=105.0)
        assert pos.current_size == 1.0  # No change
        assert pos.partial_exits_taken == 1  # Still counted
        assert pos.last_partial_exit_price == 105.0

        # Apply negative size exit (acts like scale-in)
        pos_reset = PositionState(
            entry_price=100.0, side="long", original_size=1.0, current_size=1.0
        )
        policy.apply_partial_exit(pos_reset, executed_size_fraction_of_original=-0.5, price=105.0)
        assert pos_reset.current_size == 1.5  # Increased (negative exit = scale in)
        assert pos_reset.partial_exits_taken == 1

    def test_apply_scale_in_edge_cases(self):
        """Test apply_scale_in with edge cases."""
        policy = PartialExitPolicy(
            exit_targets=[0.05],
            exit_sizes=[0.5],
            scale_in_thresholds=[0.02],
            scale_in_sizes=[0.5],
            max_scale_ins=2,  # Allow scale-ins
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=1.0)

        # Apply zero size scale-in
        policy.apply_scale_in(pos, add_size_fraction_of_original=0.0, price=102.0)
        assert pos.current_size == 1.0  # No change
        assert pos.scale_ins_taken == 1  # Still counted
        assert pos.last_scale_in_price == 102.0

        # Apply negative size scale-in (acts like partial exit)
        pos_reset = PositionState(
            entry_price=100.0, side="long", original_size=1.0, current_size=1.0
        )
        policy.apply_scale_in(pos_reset, add_size_fraction_of_original=-0.5, price=102.0)
        assert pos_reset.current_size == 0.5  # Decreased (negative scale-in = partial exit)
        assert pos_reset.scale_ins_taken == 1

    def test_apply_scale_in_size_limit(self):
        """Test apply_scale_in with size limit."""
        policy = PartialExitPolicy(
            exit_targets=[0.05],
            exit_sizes=[0.5],
            scale_in_thresholds=[0.02],
            scale_in_sizes=[0.5],
            max_scale_ins=1,  # Allow 1 scale-in
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=0.9)

        # Scale-in should be limited to 1.0 max
        policy.apply_scale_in(pos, add_size_fraction_of_original=0.5, price=102.0)
        assert pos.current_size == 1.0  # Capped at 1.0
        assert pos.scale_ins_taken == 1

    def test_max_scale_ins_zero(self):
        """Test with max_scale_ins set to zero."""
        policy = PartialExitPolicy(
            exit_targets=[0.05],
            exit_sizes=[0.5],
            scale_in_thresholds=[0.02],
            scale_in_sizes=[0.5],
            max_scale_ins=0,  # No scale-ins allowed
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=1.0)

        action = policy.check_scale_in_opportunity(pos, 102.0)
        assert action is None

    def test_scale_ins_already_at_limit(self):
        """Test scale-in when already at the limit."""
        policy = PartialExitPolicy(
            exit_targets=[0.05],
            exit_sizes=[0.5],
            scale_in_thresholds=[0.02],
            scale_in_sizes=[0.5],
            max_scale_ins=1,
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=1.0)
        pos.scale_ins_taken = 1  # Already at limit

        action = policy.check_scale_in_opportunity(pos, 102.0)
        assert action is None

    def test_scale_ins_exceeded_limit(self):
        """Test scale-in when already exceeded the limit."""
        policy = PartialExitPolicy(
            exit_targets=[0.05],
            exit_sizes=[0.5],
            scale_in_thresholds=[0.02],
            scale_in_sizes=[0.5],
            max_scale_ins=1,
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=1.0)
        pos.scale_ins_taken = 2  # Exceeded limit

        action = policy.check_scale_in_opportunity(pos, 102.0)
        assert action is None

    def test_multiple_partial_exits_same_level(self):
        """Test multiple partial exits at the same target level."""
        policy = PartialExitPolicy(
            exit_targets=[0.05, 0.05, 0.10],  # Duplicate target
            exit_sizes=[0.3, 0.3, 0.4],
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=1.0)

        # At 5% move, should trigger both level 0 and 1
        actions = policy.check_partial_exits(pos, 105.0)
        assert len(actions) == 2
        target_levels = [a["target_level"] for a in actions]
        assert 0 in target_levels
        assert 1 in target_levels

    def test_all_exits_already_taken(self):
        """Test when all partial exits have been taken."""
        policy = PartialExitPolicy(
            exit_targets=[0.05, 0.10],
            exit_sizes=[0.5, 0.5],
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=1.0)
        pos.partial_exits_taken = 2  # All exits taken

        actions = policy.check_partial_exits(pos, 115.0)  # 15% move
        assert actions == []

    def test_partial_exits_taken_exceeds_targets(self):
        """Test when partial_exits_taken exceeds number of targets."""
        policy = PartialExitPolicy(
            exit_targets=[0.05],
            exit_sizes=[0.5],
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=1.0)
        pos.partial_exits_taken = 5  # More than available targets

        actions = policy.check_partial_exits(pos, 110.0)
        assert actions == []

    def test_very_small_position_sizes(self):
        """Test with very small position sizes."""
        policy = PartialExitPolicy(
            exit_targets=[0.05],
            exit_sizes=[0.5],
        )
        pos = PositionState(
            entry_price=100.0, side="long", original_size=0.000001, current_size=0.000001
        )

        actions = policy.check_partial_exits(pos, 105.0)
        assert len(actions) == 1

        # Apply the exit
        policy.apply_partial_exit(pos, executed_size_fraction_of_original=0.5, price=105.0)
        assert pos.current_size == pytest.approx(0.0000005, abs=1e-9)

    def test_extreme_price_movements(self):
        """Test with extreme price movements."""
        policy = PartialExitPolicy(
            exit_targets=[0.05, 0.10, 0.50, 1.0],  # Up to 100% move
            exit_sizes=[0.25, 0.25, 0.25, 0.25],
        )
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=1.0)

        # Extreme 1000% move
        actions = policy.check_partial_exits(pos, 1100.0)
        assert len(actions) == 4  # All targets hit

    def test_position_state_edge_cases(self):
        """Test PositionState with edge case values."""
        # Test with all zero values
        pos = PositionState(entry_price=0.0, side="long", original_size=0.0, current_size=0.0)
        assert pos.entry_price == 0.0
        assert pos.side == "long"
        assert pos.original_size == 0.0
        assert pos.current_size == 0.0
        assert pos.partial_exits_taken == 0
        assert pos.scale_ins_taken == 0

        # Test with extreme values
        pos_extreme = PositionState(
            entry_price=999999.99, side="short", original_size=1000000.0, current_size=500000.0
        )
        assert pos_extreme.entry_price == 999999.99
        assert pos_extreme.side == "short"
        assert pos_extreme.original_size == 1000000.0
        assert pos_extreme.current_size == 500000.0


class TestPositionStateEdgeCases:
    """Test PositionState dataclass edge cases."""

    def test_position_state_creation(self):
        """Test PositionState creation with various parameters."""
        pos = PositionState(entry_price=100.0, side="long", original_size=1.0, current_size=0.8)

        assert pos.entry_price == 100.0
        assert pos.side == "long"
        assert pos.original_size == 1.0
        assert pos.current_size == 0.8
        assert pos.partial_exits_taken == 0  # Default
        assert pos.scale_ins_taken == 0  # Default

    def test_position_state_with_counters(self):
        """Test PositionState with exit and scale-in counters."""
        pos = PositionState(
            entry_price=100.0,
            side="short",
            original_size=2.0,
            current_size=1.5,
            partial_exits_taken=2,
            scale_ins_taken=1,
        )

        assert pos.partial_exits_taken == 2
        assert pos.scale_ins_taken == 1

    def test_position_state_invalid_side(self):
        """Test PositionState with invalid side."""
        # PositionState doesn't validate side parameter
        pos = PositionState(entry_price=100.0, side="invalid", original_size=1.0, current_size=1.0)

        assert pos.side == "invalid"  # Should accept any value
