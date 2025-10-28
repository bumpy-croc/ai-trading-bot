"""
Strategy Factory and Builder

This module provides factory methods and builder patterns for creating
pre-configured strategies and custom strategy compositions.
"""

from copy import deepcopy
from typing import Any, Optional

from .position_sizer import (
    ConfidenceWeightedSizer,
    FixedFractionSizer,
    KellySizer,
    PositionSizer,
    RegimeAdaptiveSizer,
)
from .regime_context import EnhancedRegimeDetector
from .risk_manager import (
    FixedRiskManager,
    RegimeAdaptiveRiskManager,
    RiskManager,
    VolatilityRiskManager,
)
from .signal_generator import (
    HierarchicalSignalGenerator,
    HoldSignalGenerator,
    RandomSignalGenerator,
    RegimeAdaptiveSignalGenerator,
    SignalGenerator,
    WeightedVotingSignalGenerator,
)
from .ml_signal_generator import MLBasicSignalGenerator, MLSignalGenerator
from .momentum_signal_generator import MomentumSignalGenerator
from .strategy import Strategy


class StrategyFactory:
    """
    Factory for creating pre-configured strategies

    Provides convenient methods for creating common strategy configurations
    without needing to manually compose all components.
    """

    @staticmethod
    def create_conservative_strategy(name: str = "Conservative") -> Strategy:
        """
        Create a conservative strategy with low risk parameters

        Args:
            name: Strategy name

        Returns:
            Configured conservative strategy
        """
        signal_generator = HoldSignalGenerator()
        risk_manager = FixedRiskManager(risk_per_trade=0.01, stop_loss_pct=0.03)  # 1% risk, 3% stop
        position_sizer = FixedFractionSizer(fraction=0.02, adjust_for_confidence=True)  # 2% base

        return Strategy(
            name=name,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=EnhancedRegimeDetector(),
        )

    @staticmethod
    def create_balanced_strategy(name: str = "Balanced") -> Strategy:
        """
        Create a balanced strategy with moderate risk parameters

        Args:
            name: Strategy name

        Returns:
            Configured balanced strategy
        """
        signal_generator = RandomSignalGenerator(buy_prob=0.3, sell_prob=0.3, seed=42)
        risk_manager = VolatilityRiskManager(base_risk=0.02, atr_multiplier=2.0)
        position_sizer = ConfidenceWeightedSizer(base_fraction=0.04, min_confidence=0.4)

        return Strategy(
            name=name,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=EnhancedRegimeDetector(),
        )

    @staticmethod
    def create_aggressive_strategy(name: str = "Aggressive") -> Strategy:
        """
        Create an aggressive strategy with higher risk parameters

        Args:
            name: Strategy name

        Returns:
            Configured aggressive strategy
        """
        signal_generator = RandomSignalGenerator(buy_prob=0.4, sell_prob=0.4, seed=123)
        risk_manager = RegimeAdaptiveRiskManager(base_risk=0.03)
        position_sizer = KellySizer(win_rate=0.55, avg_win=0.025, avg_loss=0.02, kelly_fraction=0.3)

        return Strategy(
            name=name,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=EnhancedRegimeDetector(),
        )

    @staticmethod
    def create_regime_adaptive_strategy(name: str = "RegimeAdaptive") -> Strategy:
        """
        Create a regime-adaptive strategy that changes behavior based on market conditions

        Args:
            name: Strategy name

        Returns:
            Configured regime-adaptive strategy
        """
        # Create different generators for different regimes
        bull_generator = RandomSignalGenerator(buy_prob=0.5, sell_prob=0.2, seed=100)
        bear_generator = RandomSignalGenerator(buy_prob=0.1, sell_prob=0.5, seed=200)
        range_generator = HoldSignalGenerator()

        regime_generators = {
            "bull_low_vol": bull_generator,
            "bull_high_vol": bull_generator,
            "bear_low_vol": bear_generator,
            "bear_high_vol": bear_generator,
            "range_low_vol": range_generator,
            "range_high_vol": range_generator,
        }

        signal_generator = RegimeAdaptiveSignalGenerator(
            regime_generators=regime_generators,
            default_generator=HoldSignalGenerator(),
            confidence_adjustment=True,
        )

        risk_manager = RegimeAdaptiveRiskManager(base_risk=0.025)
        position_sizer = RegimeAdaptiveSizer(base_fraction=0.03, volatility_adjustment=True)

        return Strategy(
            name=name,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=EnhancedRegimeDetector(),
        )

    @staticmethod
    def create_ensemble_strategy(name: str = "Ensemble") -> Strategy:
        """
        Create an ensemble strategy that combines multiple signal generators

        Args:
            name: Strategy name

        Returns:
            Configured ensemble strategy
        """
        # Create multiple signal generators
        gen1 = RandomSignalGenerator(buy_prob=0.35, sell_prob=0.35, seed=1)
        gen2 = RandomSignalGenerator(buy_prob=0.3, sell_prob=0.4, seed=2)
        gen3 = HoldSignalGenerator()

        # Combine with weighted voting
        generators = {gen1: 0.4, gen2: 0.4, gen3: 0.2}
        signal_generator = WeightedVotingSignalGenerator(
            generators=generators, min_confidence=0.3, consensus_threshold=0.6
        )

        risk_manager = VolatilityRiskManager(base_risk=0.02, atr_multiplier=1.8)
        position_sizer = ConfidenceWeightedSizer(base_fraction=0.05, min_confidence=0.5)

        return Strategy(
            name=name,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=EnhancedRegimeDetector(),
        )

    @staticmethod
    def create_hierarchical_strategy(name: str = "Hierarchical") -> Strategy:
        """
        Create a hierarchical strategy with primary/secondary signal confirmation

        Args:
            name: Strategy name

        Returns:
            Configured hierarchical strategy
        """
        primary_generator = RandomSignalGenerator(buy_prob=0.4, sell_prob=0.3, seed=10)
        secondary_generator = RandomSignalGenerator(buy_prob=0.3, sell_prob=0.4, seed=20)

        signal_generator = HierarchicalSignalGenerator(
            primary_generator=primary_generator,
            secondary_generator=secondary_generator,
            confirmation_mode=True,
            min_primary_confidence=0.6,
        )

        risk_manager = FixedRiskManager(risk_per_trade=0.025, stop_loss_pct=0.04)
        position_sizer = FixedFractionSizer(fraction=0.03, adjust_for_confidence=True)

        return Strategy(
            name=name,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=EnhancedRegimeDetector(),
        )

    @staticmethod
    def create_ml_basic_strategy(
        name: str = "MLBasic",
        sequence_length: int = 120,
        model_name: Optional[str] = None,
        model_type: str = "basic",
        timeframe: str = "1h",
    ) -> Strategy:
        """
        Create ML Basic strategy with component-based architecture

        Args:
            name: Strategy name
            sequence_length: Sequence length for LSTM
            model_name: Model name for registry
            model_type: Model type
            timeframe: Model timeframe

        Returns:
            Configured ML Basic strategy
        """
        signal_generator = MLBasicSignalGenerator(
            name=f"{name}_signals",
            sequence_length=sequence_length,
            model_name=model_name,
            model_type=model_type,
            timeframe=timeframe,
        )

        risk_manager = FixedRiskManager(
            risk_per_trade=0.02,
            stop_loss_pct=0.02,
        )

        position_sizer = ConfidenceWeightedSizer(
            base_fraction=0.20,
            min_confidence=0.3,
        )

        return Strategy(
            name=name,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=EnhancedRegimeDetector(),
        )

    @staticmethod
    def create_ml_adaptive_strategy(
        name: str = "MLAdaptive",
        sequence_length: int = 120,
        model_name: Optional[str] = None,
    ) -> Strategy:
        """
        Create ML Adaptive strategy with component-based architecture

        Args:
            name: Strategy name
            sequence_length: Sequence length for LSTM
            model_name: Model name for registry

        Returns:
            Configured ML Adaptive strategy
        """
        signal_generator = MLSignalGenerator(
            name=f"{name}_signals",
            sequence_length=sequence_length,
            model_name=model_name,
        )

        risk_manager = RegimeAdaptiveRiskManager(
            base_risk=0.02,
        )

        position_sizer = ConfidenceWeightedSizer(
            base_fraction=0.20,
            min_confidence=0.3,
        )

        return Strategy(
            name=name,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=EnhancedRegimeDetector(),
        )

    @staticmethod
    def create_ml_sentiment_strategy(
        name: str = "MLSentiment",
        sequence_length: int = 120,
        model_name: Optional[str] = None,
        model_type: str = "sentiment",
        timeframe: str = "1h",
    ) -> Strategy:
        """
        Create ML Sentiment strategy with component-based architecture

        Args:
            name: Strategy name
            sequence_length: Sequence length for LSTM
            model_name: Model name for registry
            model_type: Model type
            timeframe: Model timeframe

        Returns:
            Configured ML Sentiment strategy
        """
        signal_generator = MLSignalGenerator(
            name=f"{name}_signals",
            sequence_length=sequence_length,
            model_name=model_name,
        )

        risk_manager = FixedRiskManager(
            risk_per_trade=0.02,
            stop_loss_pct=0.04,
        )

        position_sizer = ConfidenceWeightedSizer(
            base_fraction=0.20,
            min_confidence=0.3,
        )

        return Strategy(
            name=name,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=EnhancedRegimeDetector(),
        )

    @staticmethod
    def create_momentum_leverage_strategy(
        name: str = "MomentumLeverage",
    ) -> Strategy:
        """
        Create Momentum Leverage strategy with component-based architecture

        Args:
            name: Strategy name

        Returns:
            Configured Momentum Leverage strategy
        """
        signal_generator = MomentumSignalGenerator(
            name=f"{name}_signals",
            momentum_entry_threshold=0.01,
            strong_momentum_threshold=0.025,
        )

        risk_manager = VolatilityRiskManager(
            base_risk=0.10,
            atr_multiplier=2.0,
            min_risk=0.05,
            max_risk=0.35,
        )

        position_sizer = ConfidenceWeightedSizer(
            base_fraction=0.50,
            min_confidence=0.3,
        )

        return Strategy(
            name=name,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=EnhancedRegimeDetector(),
        )

    @staticmethod
    def create_ensemble_weighted_strategy(
        name: str = "EnsembleWeighted",
        use_ml_basic: bool = True,
        use_ml_adaptive: bool = True,
        use_ml_sentiment: bool = False,
    ) -> Strategy:
        """
        Create Ensemble Weighted strategy with component-based architecture

        Args:
            name: Strategy name
            use_ml_basic: Whether to include ML Basic
            use_ml_adaptive: Whether to include ML Adaptive
            use_ml_sentiment: Whether to include ML Sentiment

        Returns:
            Configured Ensemble Weighted strategy
        """
        # Create individual signal generators
        generators = {}
        if use_ml_basic:
            generators[MLBasicSignalGenerator(name="ml_basic_signals")] = 0.30
        if use_ml_adaptive:
            generators[MLSignalGenerator(name="ml_adaptive_signals")] = 0.30
        if use_ml_sentiment:
            generators[MLSignalGenerator(name="ml_sentiment_signals")] = 0.15

        signal_generator = WeightedVotingSignalGenerator(
            generators=generators,
            min_confidence=0.3,
            consensus_threshold=0.6,
        )

        risk_manager = VolatilityRiskManager(
            base_risk=0.06,
            atr_multiplier=2.0,
            min_risk=0.03,
            max_risk=0.20,
        )

        position_sizer = ConfidenceWeightedSizer(
            base_fraction=0.50,
            min_confidence=0.3,
        )

        return Strategy(
            name=name,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=EnhancedRegimeDetector(),
        )


class StrategyBuilder:
    """
    Builder pattern for custom strategy composition

    Provides a fluent interface for building custom strategies with
    specific component configurations.
    """

    def __init__(self, name: str):
        """
        Initialize strategy builder

        Args:
            name: Strategy name
        """
        self.name = name
        self._signal_generator: Optional[SignalGenerator] = None
        self._risk_manager: Optional[RiskManager] = None
        self._position_sizer: Optional[PositionSizer] = None
        self._regime_detector: Optional[EnhancedRegimeDetector] = None
        self._enable_logging: bool = True
        self._max_history: int = 1000

    def with_signal_generator(self, signal_generator: SignalGenerator) -> "StrategyBuilder":
        """
        Set signal generator component

        Args:
            signal_generator: Signal generator to use

        Returns:
            Self for method chaining
        """
        self._signal_generator = signal_generator
        return self

    def with_risk_manager(self, risk_manager: RiskManager) -> "StrategyBuilder":
        """
        Set risk manager component

        Args:
            risk_manager: Risk manager to use

        Returns:
            Self for method chaining
        """
        self._risk_manager = risk_manager
        return self

    def with_position_sizer(self, position_sizer: PositionSizer) -> "StrategyBuilder":
        """
        Set position sizer component

        Args:
            position_sizer: Position sizer to use

        Returns:
            Self for method chaining
        """
        self._position_sizer = position_sizer
        return self

    def with_regime_detector(self, regime_detector: EnhancedRegimeDetector) -> "StrategyBuilder":
        """
        Set regime detector component

        Args:
            regime_detector: Regime detector to use

        Returns:
            Self for method chaining
        """
        self._regime_detector = regime_detector
        return self

    def with_logging(self, enable: bool = True) -> "StrategyBuilder":
        """
        Configure logging

        Args:
            enable: Whether to enable logging

        Returns:
            Self for method chaining
        """
        self._enable_logging = enable
        return self

    def with_history_size(self, max_history: int) -> "StrategyBuilder":
        """
        Configure history size

        Args:
            max_history: Maximum number of decisions to keep

        Returns:
            Self for method chaining
        """
        self._max_history = max_history
        return self

    def build(self) -> Strategy:
        """
        Build the strategy with configured components

        Returns:
            Configured strategy

        Raises:
            ValueError: If required components are missing
        """
        # Validate required components
        if self._signal_generator is None:
            raise ValueError("Signal generator is required")

        if self._risk_manager is None:
            raise ValueError("Risk manager is required")

        if self._position_sizer is None:
            raise ValueError("Position sizer is required")

        # Use defaults for optional components
        regime_detector = self._regime_detector or EnhancedRegimeDetector()

        return Strategy(
            name=self.name,
            signal_generator=self._signal_generator,
            risk_manager=self._risk_manager,
            position_sizer=self._position_sizer,
            regime_detector=regime_detector,
            enable_logging=self._enable_logging,
            max_history=self._max_history,
        )


def create_strategy_template(template_name: str, **kwargs) -> dict[str, Any]:
    """
    Create strategy template configuration

    Args:
        template_name: Name of the template
        **kwargs: Template parameters

    Returns:
        Strategy template configuration
    """
    templates = {
        "conservative": {
            "signal_generator": {"type": "HoldSignalGenerator"},
            "risk_manager": {
                "type": "FixedRiskManager",
                "risk_per_trade": 0.01,
                "stop_loss_pct": 0.03,
            },
            "position_sizer": {"type": "FixedFractionSizer", "fraction": 0.02},
            "description": "Low-risk conservative strategy",
        },
        "balanced": {
            "signal_generator": {
                "type": "RandomSignalGenerator",
                "buy_prob": 0.3,
                "sell_prob": 0.3,
            },
            "risk_manager": {"type": "VolatilityRiskManager", "base_risk": 0.02},
            "position_sizer": {"type": "ConfidenceWeightedSizer", "base_fraction": 0.04},
            "description": "Moderate-risk balanced strategy",
        },
        "aggressive": {
            "signal_generator": {
                "type": "RandomSignalGenerator",
                "buy_prob": 0.4,
                "sell_prob": 0.4,
            },
            "risk_manager": {"type": "RegimeAdaptiveRiskManager", "base_risk": 0.03},
            "position_sizer": {"type": "KellySizer", "kelly_fraction": 0.3},
            "description": "Higher-risk aggressive strategy",
        },
    }

    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")

    # Work on a deep copy so modifications don't leak between calls
    template = deepcopy(templates[template_name])

    # Apply any parameter overrides
    for component_type in ["signal_generator", "risk_manager", "position_sizer"]:
        if component_type not in kwargs:
            continue

        overrides = kwargs[component_type]
        component_config = template[component_type].copy()

        # If the caller changes the component type we need to drop template-specific
        # parameters so we don't pass invalid kwargs to the new component class.
        override_type = overrides.get("type")
        if override_type and override_type != component_config.get("type"):
            component_config = {"type": override_type}
            overrides = {k: v for k, v in overrides.items() if k != "type"}

        component_config.update(overrides)
        template[component_type] = component_config

    return template


def validate_strategy_configuration(config: dict[str, Any]) -> bool:
    """
    Validate strategy configuration

    Args:
        config: Strategy configuration dictionary

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_components = ["signal_generator", "risk_manager", "position_sizer"]

    for component in required_components:
        if component not in config:
            raise ValueError(f"Missing required component: {component}")

        if "type" not in config[component]:
            raise ValueError(f"Component {component} missing 'type' field")

    return True
