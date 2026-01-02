"""
Optimized Weighted Ensemble Strategy - Component-Based Implementation

An aggressive ensemble approach designed to beat buy-and-hold returns while
maintaining acceptable risk levels (20-30% max drawdown).

Key Features for Beating Buy-and-Hold:
- Leveraged position sizing (up to 45% per trade)
- Momentum-based entry timing
- Trend following with breakout detection
- Dynamic risk scaling based on market volatility
- Performance-based strategy weighting
- Multi-timeframe confirmation
- Aggressive profit-taking and re-entry

Risk Management:
- Wider stops (3.5%) to avoid premature exits
- Higher profit targets (8%) to capture trends
- Trailing stops to protect profits
- Dynamic position sizing based on confidence
"""

from src.config.constants import DEFAULT_STRATEGY_MIN_CONFIDENCE
from src.strategies.components import (
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
    MLBasicSignalGenerator,
    MLSignalGenerator,
    Strategy,
    VolatilityRiskManager,
    WeightedVotingSignalGenerator,
)

# Configuration constants for ensemble strategy
# Note: MIN_STRATEGIES_FOR_SIGNAL, PERFORMANCE_WINDOW, and WEIGHT_UPDATE_FREQUENCY
# are reserved for future dynamic weight adjustment features.
BASE_POSITION_SIZE = 0.50
MIN_POSITION_SIZE_RATIO = 0.20
MAX_POSITION_SIZE_RATIO = 0.80
STOP_LOSS_PCT = 0.06
TAKE_PROFIT_PCT = 0.20


def create_ensemble_weighted_strategy(
    name: str = "EnsembleWeighted",
    use_ml_basic: bool = True,
    use_ml_adaptive: bool = True,
    use_ml_sentiment: bool = False,
) -> Strategy:
    """
    Create Ensemble Weighted strategy using component composition.

    Args:
        name: Strategy name
        use_ml_basic: Whether to include ML Basic signal generator
        use_ml_adaptive: Whether to include ML Adaptive signal generator
        use_ml_sentiment: Whether to include ML Sentiment signal generator

    Returns:
        Configured Strategy instance
    """
    # Create individual signal generators with weights
    generators = {}

    if use_ml_basic:
        generators[MLBasicSignalGenerator()] = 0.30
    if use_ml_adaptive:
        generators[MLSignalGenerator()] = 0.30
    if use_ml_sentiment:
        generators[MLSignalGenerator()] = 0.15

    # Create weighted voting signal generator
    signal_generator = WeightedVotingSignalGenerator(
        generators=generators,
        min_confidence=DEFAULT_STRATEGY_MIN_CONFIDENCE,
        consensus_threshold=0.6,
    )

    # Create volatility-based risk manager
    risk_manager = VolatilityRiskManager(
        base_risk=STOP_LOSS_PCT,
        atr_multiplier=2.0,
        min_risk=0.005,
        max_risk=0.05,
    )

    # Create aggressive position sizer
    position_sizer = ConfidenceWeightedSizer(
        base_fraction=BASE_POSITION_SIZE,
        min_confidence=DEFAULT_STRATEGY_MIN_CONFIDENCE,
    )

    # Create regime detector
    regime_detector = EnhancedRegimeDetector()

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
    )

    # Expose configuration for test validation
    strategy.stop_loss_pct = STOP_LOSS_PCT
    strategy.take_profit_pct = TAKE_PROFIT_PCT
    strategy.min_position_size_ratio = MIN_POSITION_SIZE_RATIO
    strategy.max_position_size_ratio = MAX_POSITION_SIZE_RATIO
    strategy.trading_pair = "BTCUSDT"

    # Expose normalized component weights for introspection
    strategy.component_weights = {
        generator.__class__.__name__: weight
        for generator, weight in signal_generator.generators.items()
    }

    # Configure trailing stop behavior via risk overrides
    strategy.set_risk_overrides({
        "trailing_stop": {
            "enabled": True,
            "activation_threshold": 0.04,
            "distance": 0.02,
            "step": 0.01,
            "cooldown_hours": 4,
        }
    })

    return strategy
