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
    from src.strategies.ml_adaptive import MlAdaptive
    
    strategy = MlAdaptive()
    overrides = strategy.get_risk_overrides()
    
    assert overrides is not None
    assert 'partial_operations' in overrides
    
    partial_config = overrides['partial_operations']
    assert 'exit_targets' in partial_config
    assert 'exit_sizes' in partial_config
    assert 'scale_in_thresholds' in partial_config
    assert 'scale_in_sizes' in partial_config
    assert 'max_scale_ins' in partial_config
    
    # Verify the configuration values
    assert partial_config['exit_targets'] == [0.03, 0.06, 0.10]
    assert partial_config['exit_sizes'] == [0.25, 0.25, 0.50]
    assert partial_config['scale_in_thresholds'] == [0.02, 0.05]
    assert partial_config['scale_in_sizes'] == [0.25, 0.25]
    assert partial_config['max_scale_ins'] == 2