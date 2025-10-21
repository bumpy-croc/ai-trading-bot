"""
Unit tests for StrategyFactory and StrategyBuilder
"""

import pytest

from src.strategies.components.position_sizer import (
    ConfidenceWeightedSizer,
    FixedFractionSizer,
    KellySizer,
    RegimeAdaptiveSizer,
)
from src.strategies.components.risk_manager import (
    FixedRiskManager,
    RegimeAdaptiveRiskManager,
    VolatilityRiskManager,
)
from src.strategies.components.signal_generator import (
    HierarchicalSignalGenerator,
    HoldSignalGenerator,
    RandomSignalGenerator,
    RegimeAdaptiveSignalGenerator,
    WeightedVotingSignalGenerator,
)
from src.strategies.components.strategy import Strategy
from src.strategies.components.strategy_factory import (
    StrategyBuilder,
    StrategyFactory,
    create_strategy_template,
    validate_strategy_configuration,
)


class TestStrategyFactory:
    """Test StrategyFactory class"""
    
    def test_create_conservative_strategy(self):
        """Test creating conservative strategy"""
        strategy = StrategyFactory.create_conservative_strategy("TestConservative")
        
        assert isinstance(strategy, Strategy)
        assert strategy.name == "TestConservative"
        assert isinstance(strategy.signal_generator, HoldSignalGenerator)
        assert isinstance(strategy.risk_manager, FixedRiskManager)
        assert isinstance(strategy.position_sizer, FixedFractionSizer)
        
        # Check conservative parameters
        assert strategy.risk_manager.risk_per_trade == 0.01  # 1% risk
        assert strategy.risk_manager.stop_loss_pct == 0.03   # 3% stop loss
        assert strategy.position_sizer.fraction == 0.02      # 2% base position
    
    def test_create_balanced_strategy(self):
        """Test creating balanced strategy"""
        strategy = StrategyFactory.create_balanced_strategy("TestBalanced")
        
        assert isinstance(strategy, Strategy)
        assert strategy.name == "TestBalanced"
        assert isinstance(strategy.signal_generator, RandomSignalGenerator)
        assert isinstance(strategy.risk_manager, VolatilityRiskManager)
        assert isinstance(strategy.position_sizer, ConfidenceWeightedSizer)
        
        # Check balanced parameters
        assert strategy.risk_manager.base_risk == 0.02
        assert strategy.position_sizer.base_fraction == 0.04
        assert strategy.position_sizer.min_confidence == 0.4
    
    def test_create_aggressive_strategy(self):
        """Test creating aggressive strategy"""
        strategy = StrategyFactory.create_aggressive_strategy("TestAggressive")
        
        assert isinstance(strategy, Strategy)
        assert strategy.name == "TestAggressive"
        assert isinstance(strategy.signal_generator, RandomSignalGenerator)
        assert isinstance(strategy.risk_manager, RegimeAdaptiveRiskManager)
        assert isinstance(strategy.position_sizer, KellySizer)
        
        # Check aggressive parameters
        assert strategy.risk_manager.base_risk == 0.03
        assert strategy.position_sizer.kelly_fraction == 0.3
    
    def test_create_regime_adaptive_strategy(self):
        """Test creating regime-adaptive strategy"""
        strategy = StrategyFactory.create_regime_adaptive_strategy("TestRegimeAdaptive")
        
        assert isinstance(strategy, Strategy)
        assert strategy.name == "TestRegimeAdaptive"
        assert isinstance(strategy.signal_generator, RegimeAdaptiveSignalGenerator)
        assert isinstance(strategy.risk_manager, RegimeAdaptiveRiskManager)
        assert isinstance(strategy.position_sizer, RegimeAdaptiveSizer)
        
        # Check regime-adaptive configuration
        assert strategy.signal_generator.confidence_adjustment is True
        assert strategy.risk_manager.base_risk == 0.025
        assert strategy.position_sizer.volatility_adjustment is True
    
    def test_create_ensemble_strategy(self):
        """Test creating ensemble strategy"""
        strategy = StrategyFactory.create_ensemble_strategy("TestEnsemble")
        
        assert isinstance(strategy, Strategy)
        assert strategy.name == "TestEnsemble"
        assert isinstance(strategy.signal_generator, WeightedVotingSignalGenerator)
        assert isinstance(strategy.risk_manager, VolatilityRiskManager)
        assert isinstance(strategy.position_sizer, ConfidenceWeightedSizer)
        
        # Check ensemble configuration
        assert len(strategy.signal_generator.generators) == 3
        assert strategy.signal_generator.consensus_threshold == 0.6
        assert strategy.signal_generator.min_confidence == 0.3
    
    def test_create_hierarchical_strategy(self):
        """Test creating hierarchical strategy"""
        strategy = StrategyFactory.create_hierarchical_strategy("TestHierarchical")
        
        assert isinstance(strategy, Strategy)
        assert strategy.name == "TestHierarchical"
        assert isinstance(strategy.signal_generator, HierarchicalSignalGenerator)
        assert isinstance(strategy.risk_manager, FixedRiskManager)
        assert isinstance(strategy.position_sizer, FixedFractionSizer)
        
        # Check hierarchical configuration
        assert strategy.signal_generator.confirmation_mode is True
        assert strategy.signal_generator.min_primary_confidence == 0.6
    
    def test_create_strategy_with_default_name(self):
        """Test creating strategy with default name"""
        strategy = StrategyFactory.create_conservative_strategy()
        assert strategy.name == "Conservative"


class TestStrategyBuilder:
    """Test StrategyBuilder class"""
    
    def test_builder_initialization(self):
        """Test builder initialization"""
        builder = StrategyBuilder("TestBuilder")
        assert builder.name == "TestBuilder"
        assert builder._signal_generator is None
        assert builder._risk_manager is None
        assert builder._position_sizer is None
        assert builder._enable_logging is True
        assert builder._max_history == 1000
    
    def test_builder_fluent_interface(self):
        """Test builder fluent interface"""
        signal_gen = HoldSignalGenerator()
        risk_mgr = FixedRiskManager()
        pos_sizer = FixedFractionSizer()
        
        builder = (StrategyBuilder("FluentTest")
                  .with_signal_generator(signal_gen)
                  .with_risk_manager(risk_mgr)
                  .with_position_sizer(pos_sizer)
                  .with_logging(False)
                  .with_history_size(500))
        
        assert builder._signal_generator == signal_gen
        assert builder._risk_manager == risk_mgr
        assert builder._position_sizer == pos_sizer
        assert builder._enable_logging is False
        assert builder._max_history == 500
    
    def test_builder_build_success(self):
        """Test successful strategy building"""
        strategy = (StrategyBuilder("BuildTest")
                   .with_signal_generator(HoldSignalGenerator())
                   .with_risk_manager(FixedRiskManager())
                   .with_position_sizer(FixedFractionSizer())
                   .build())
        
        assert isinstance(strategy, Strategy)
        assert strategy.name == "BuildTest"
        assert isinstance(strategy.signal_generator, HoldSignalGenerator)
        assert isinstance(strategy.risk_manager, FixedRiskManager)
        assert isinstance(strategy.position_sizer, FixedFractionSizer)
    
    def test_builder_missing_signal_generator(self):
        """Test building without signal generator"""
        with pytest.raises(ValueError, match="Signal generator is required"):
            (StrategyBuilder("MissingSignalGen")
             .with_risk_manager(FixedRiskManager())
             .with_position_sizer(FixedFractionSizer())
             .build())
    
    def test_builder_missing_risk_manager(self):
        """Test building without risk manager"""
        with pytest.raises(ValueError, match="Risk manager is required"):
            (StrategyBuilder("MissingRiskMgr")
             .with_signal_generator(HoldSignalGenerator())
             .with_position_sizer(FixedFractionSizer())
             .build())
    
    def test_builder_missing_position_sizer(self):
        """Test building without position sizer"""
        with pytest.raises(ValueError, match="Position sizer is required"):
            (StrategyBuilder("MissingPosSizer")
             .with_signal_generator(HoldSignalGenerator())
             .with_risk_manager(FixedRiskManager())
             .build())
    
    def test_builder_with_custom_regime_detector(self):
        """Test building with custom regime detector"""
        from src.strategies.components.regime_context import EnhancedRegimeDetector
        
        custom_detector = EnhancedRegimeDetector()
        
        strategy = (StrategyBuilder("CustomRegime")
                   .with_signal_generator(HoldSignalGenerator())
                   .with_risk_manager(FixedRiskManager())
                   .with_position_sizer(FixedFractionSizer())
                   .with_regime_detector(custom_detector)
                   .build())
        
        assert strategy.regime_detector == custom_detector


class TestStrategyTemplates:
    """Test strategy template functions"""
    
    def test_create_conservative_template(self):
        """Test creating conservative template"""
        template = create_strategy_template("conservative")
        
        assert template['signal_generator']['type'] == 'HoldSignalGenerator'
        assert template['risk_manager']['type'] == 'FixedRiskManager'
        assert template['risk_manager']['risk_per_trade'] == 0.01
        assert template['position_sizer']['type'] == 'FixedFractionSizer'
        assert template['description'] == 'Low-risk conservative strategy'
    
    def test_create_balanced_template(self):
        """Test creating balanced template"""
        template = create_strategy_template("balanced")
        
        assert template['signal_generator']['type'] == 'RandomSignalGenerator'
        assert template['risk_manager']['type'] == 'VolatilityRiskManager'
        assert template['position_sizer']['type'] == 'ConfidenceWeightedSizer'
        assert template['description'] == 'Moderate-risk balanced strategy'
    
    def test_create_aggressive_template(self):
        """Test creating aggressive template"""
        template = create_strategy_template("aggressive")
        
        assert template['signal_generator']['type'] == 'RandomSignalGenerator'
        assert template['risk_manager']['type'] == 'RegimeAdaptiveRiskManager'
        assert template['position_sizer']['type'] == 'KellySizer'
        assert template['description'] == 'Higher-risk aggressive strategy'
    
    def test_create_unknown_template(self):
        """Test creating unknown template"""
        with pytest.raises(ValueError, match="Unknown template: unknown"):
            create_strategy_template("unknown")
    
    def test_template_with_parameter_overrides(self):
        """Test template with parameter overrides"""
        template = create_strategy_template(
            "conservative",
            risk_manager={'risk_per_trade': 0.005},
            position_sizer={'fraction': 0.01}
        )

        assert template['risk_manager']['risk_per_trade'] == 0.005
        assert template['position_sizer']['fraction'] == 0.01
        # Other parameters should remain unchanged
        assert template['risk_manager']['stop_loss_pct'] == 0.03

    def test_template_type_override_resets_default_parameters(self):
        """Overriding the component type should discard template-specific params."""
        template = create_strategy_template(
            "balanced",
            signal_generator={'type': 'HoldSignalGenerator'}
        )

        assert template['signal_generator']['type'] == 'HoldSignalGenerator'
        # Parameters that only make sense for the original template should be removed
        assert 'buy_prob' not in template['signal_generator']
        assert 'sell_prob' not in template['signal_generator']


class TestStrategyValidation:
    """Test strategy configuration validation"""
    
    def test_validate_valid_configuration(self):
        """Test validating valid configuration"""
        config = {
            'signal_generator': {'type': 'HoldSignalGenerator'},
            'risk_manager': {'type': 'FixedRiskManager'},
            'position_sizer': {'type': 'FixedFractionSizer'}
        }
        
        assert validate_strategy_configuration(config) is True
    
    def test_validate_missing_signal_generator(self):
        """Test validation with missing signal generator"""
        config = {
            'risk_manager': {'type': 'FixedRiskManager'},
            'position_sizer': {'type': 'FixedFractionSizer'}
        }
        
        with pytest.raises(ValueError, match="Missing required component: signal_generator"):
            validate_strategy_configuration(config)
    
    def test_validate_missing_risk_manager(self):
        """Test validation with missing risk manager"""
        config = {
            'signal_generator': {'type': 'HoldSignalGenerator'},
            'position_sizer': {'type': 'FixedFractionSizer'}
        }
        
        with pytest.raises(ValueError, match="Missing required component: risk_manager"):
            validate_strategy_configuration(config)
    
    def test_validate_missing_position_sizer(self):
        """Test validation with missing position sizer"""
        config = {
            'signal_generator': {'type': 'HoldSignalGenerator'},
            'risk_manager': {'type': 'FixedRiskManager'}
        }
        
        with pytest.raises(ValueError, match="Missing required component: position_sizer"):
            validate_strategy_configuration(config)
    
    def test_validate_missing_type_field(self):
        """Test validation with missing type field"""
        config = {
            'signal_generator': {'name': 'test'},  # Missing 'type'
            'risk_manager': {'type': 'FixedRiskManager'},
            'position_sizer': {'type': 'FixedFractionSizer'}
        }
        
        with pytest.raises(ValueError, match="Component signal_generator missing 'type' field"):
            validate_strategy_configuration(config)