from .base import BaseStrategy
from .adaptive import AdaptiveStrategy
from .enhanced import EnhancedStrategy
# Optional high-risk strategy (requires TA-Lib)
try:
    from .high_risk_high_reward import HighRiskHighRewardStrategy
except ModuleNotFoundError:
    # TA-Lib may be missing in some environments; skip importing to avoid crashes
    HighRiskHighRewardStrategy = None
from .ml_basic import MlBasic
from .ml_with_sentiment import MlWithSentiment


def get_strategy_class(strategy_name: str):
    """Get strategy class by name"""
    strategy_map = {
        'adaptive': AdaptiveStrategy,
        'enhanced': EnhancedStrategy,
        'high_risk_high_reward': HighRiskHighRewardStrategy,
        'ml_basic': MlBasic,
        'ml_with_sentiment': MlWithSentiment,
    }
    
    return strategy_map.get(strategy_name.lower())


__all__ = ['BaseStrategy', 'AdaptiveStrategy', 'EnhancedStrategy', 'HighRiskHighRewardStrategy', 'MlBasic', 'MlWithSentiment', 'get_strategy_class'] 