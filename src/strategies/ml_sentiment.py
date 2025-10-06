"""
ML Sentiment Strategy - Component-Based Implementation

This strategy uses machine learning models trained with both price data and sentiment analysis.
It leverages the Fear & Greed Index to enhance prediction accuracy and trading decisions.

Key Features:
- Price + sentiment predictions using LSTM neural network
- 120-day sequence length for pattern recognition
- Fear & Greed Index sentiment integration
- Adaptive position sizing based on sentiment confidence
- 2% stop loss, 4% take profit risk management
- Robust fallback when sentiment data is unavailable
- Component-based architecture for better maintainability

Ideal for:
- Enhanced prediction accuracy with market sentiment
- Bull/bear market regime detection
- Trading during high-volatility periods
- Environments with reliable sentiment data access
"""

from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.config.config_manager import get_config
from src.config.constants import DEFAULT_USE_PREDICTION_ENGINE
from src.prediction import PredictionConfig, PredictionEngine
from src.prediction.features.pipeline import FeaturePipeline
from src.prediction.features.sentiment import SentimentFeatureExtractor
from src.prediction.features.technical import TechnicalFeatureExtractor
from src.prediction.models.registry import PredictionModelRegistry
from src.strategies.adapters.legacy_adapter import LegacyStrategyAdapter
from src.strategies.components import (
    Strategy,
    MLSignalGenerator,
    FixedRiskManager,
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
)


class MlSentiment(LegacyStrategyAdapter):
    """
    ML Sentiment Strategy using component-based architecture.
    
    This implementation wraps the component-based strategy with LegacyStrategyAdapter
    to maintain backward compatibility with the existing BaseStrategy interface.
    """
    
    # * Strategy configuration constants
    SHORT_ENTRY_THRESHOLD = -0.0005  # -0.05% threshold for short entries
    CONFIDENCE_MULTIPLIER = 12  # Multiplier for confidence calculation
    BASE_POSITION_SIZE = 0.2  # Base position size (20% of balance)
    MIN_POSITION_SIZE_RATIO = 0.05  # Minimum position size (5% of balance)
    MAX_POSITION_SIZE_RATIO = 0.25  # Maximum position size (25% of balance)
    
    # Sentiment-specific parameters
    SENTIMENT_BOOST_MULTIPLIER = 1.2  # Boost position size when sentiment aligns
    SENTIMENT_REDUCTION_MULTIPLIER = 0.8  # Reduce position size when sentiment contradicts
    MIN_SENTIMENT_CONFIDENCE = 0.6  # Minimum sentiment confidence to use sentiment features

    def __init__(
        self,
        name="MlSentiment",
        model_path="src/ml/btcusdt_sentiment.onnx",
        sequence_length=120,
        use_prediction_engine: Optional[bool] = None,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        timeframe: Optional[str] = None,
    ):
        # Create component-based strategy
        component_strategy = self._create_component_strategy(
            name=name,
            model_path=model_path,
            sequence_length=sequence_length,
            use_prediction_engine=use_prediction_engine,
            model_name=model_name,
            model_type=model_type,
            timeframe=timeframe,
        )
        
        # Initialize adapter with component strategy
        super().__init__(component_strategy, name=name)
        
        # Preserve legacy attributes for backward compatibility
        self.trading_pair = "BTCUSDT"
        self.model_type = model_type or "sentiment"
        self.model_timeframe = timeframe or "1h"
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
        # Optional prediction engine integration
        cfg = get_config()
        self.use_prediction_engine = (
            use_prediction_engine
            if use_prediction_engine is not None
            else cfg.get_bool("USE_PREDICTION_ENGINE", default=DEFAULT_USE_PREDICTION_ENGINE)
        )
        
        # Model name configuration
        self.model_name = model_name
        if self.model_name is None:
            self.model_name = cfg.get("PREDICTION_ENGINE_MODEL_NAME", default=None)
        if self.model_name is None:
            try:
                self.model_name = Path(self.model_path).stem
            except Exception:
                self.model_name = None

    def _create_component_strategy(
        self,
        name: str,
        model_path: str,
        sequence_length: int,
        use_prediction_engine: Optional[bool],
        model_name: Optional[str],
        model_type: Optional[str],
        timeframe: Optional[str],
    ) -> Strategy:
        """Create the component-based strategy with ML Sentiment configuration."""
        
        # Create signal generator with ML Sentiment parameters (sentiment model)
        signal_generator = MLSignalGenerator(
            name=f"{name}_signals",
            model_path=model_path,
            sequence_length=sequence_length,
            use_prediction_engine=use_prediction_engine,
            model_name=model_name,
            model_type=model_type,
            timeframe=timeframe,
            confidence_multiplier=self.CONFIDENCE_MULTIPLIER,
            # Sentiment-specific parameters
            sentiment_boost_multiplier=self.SENTIMENT_BOOST_MULTIPLIER,
            sentiment_reduction_multiplier=self.SENTIMENT_REDUCTION_MULTIPLIER,
            min_sentiment_confidence=self.MIN_SENTIMENT_CONFIDENCE,
        )
        
        # Create fixed risk manager
        risk_manager = FixedRiskManager(
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
        )
        
        # Create position sizer with confidence weighting and sentiment adjustment
        position_sizer = ConfidenceWeightedSizer(
            base_fraction=self.BASE_POSITION_SIZE,
            min_fraction=self.MIN_POSITION_SIZE_RATIO,
            max_fraction=self.MAX_POSITION_SIZE_RATIO,
            confidence_multiplier=self.CONFIDENCE_MULTIPLIER,
            # Sentiment-specific adjustments
            sentiment_boost_multiplier=self.SENTIMENT_BOOST_MULTIPLIER,
            sentiment_reduction_multiplier=self.SENTIMENT_REDUCTION_MULTIPLIER,
        )
        
        # Create regime detector
        regime_detector = EnhancedRegimeDetector()
        
        # Create component-based strategy
        return Strategy(
            name=f"{name}_component",
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=regime_detector,
        )


    def get_parameters(self) -> dict:
        """Return strategy parameters for logging"""
        return {
            "strategy_name": self.name,
            "model_path": self.model_path,
            "sequence_length": self.sequence_length,
            "use_prediction_engine": self.use_prediction_engine,
            "model_name": self.model_name,
            "base_position_size": self.BASE_POSITION_SIZE,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "sentiment_boost_multiplier": self.SENTIMENT_BOOST_MULTIPLIER,
            "sentiment_reduction_multiplier": self.SENTIMENT_REDUCTION_MULTIPLIER,
            "min_sentiment_confidence": self.MIN_SENTIMENT_CONFIDENCE,
        }
