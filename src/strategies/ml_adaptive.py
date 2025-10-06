"""
ML Adaptive Strategy - Component-Based Implementation

This strategy uses a machine learning model trained exclusively on price data (OHLCV).
It provides simple, reliable predictions without external dependencies like sentiment data.

Key Features:
- Price-only predictions using LSTM neural network
- 120-day sequence length for pattern recognition
- Normalized price inputs for better model performance
- 2% stop loss, 4% take profit risk management
- No external API dependencies
- Component-based architecture for better maintainability
- Dynamic regime-aware thresholds

Ideal for:
- Consistent, reliable trading signals
- Backtesting historical periods
- Environments where sentiment data is unavailable
- Simple deployment scenarios
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.config.config_manager import get_config
from src.config.constants import DEFAULT_USE_PREDICTION_ENGINE
from src.prediction import PredictionConfig, PredictionEngine
from src.prediction.features.pipeline import FeaturePipeline
from src.prediction.features.price_only import PriceOnlyFeatureExtractor
from src.prediction.features.technical import TechnicalFeatureExtractor
from src.regime.detector import RegimeDetector, TrendLabel, VolLabel
from src.strategies.adapters.legacy_adapter import LegacyStrategyAdapter
from src.strategies.components import (
    Strategy,
    MLSignalGenerator,
    RegimeAdaptiveRiskManager,
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
)


class MlAdaptive(LegacyStrategyAdapter):
    """
    ML Adaptive Strategy using component-based architecture.
    
    This implementation wraps the component-based strategy with LegacyStrategyAdapter
    to maintain backward compatibility with the existing BaseStrategy interface.
    """
    
    # * Strategy configuration constants
    SHORT_ENTRY_THRESHOLD = -0.0005  # -0.05% threshold for short entries (base threshold)
    CONFIDENCE_MULTIPLIER = 12  # Multiplier for confidence calculation
    BASE_POSITION_SIZE = 0.2  # Base position size (20% of balance)
    MIN_POSITION_SIZE_RATIO = 0.05  # Minimum position size (5% of balance)
    MAX_POSITION_SIZE_RATIO = 0.25  # Maximum position size (25% of balance)
    
    # * Dynamic short entry threshold configuration
    # Base thresholds for different market regimes (more aggressive to match original performance)
    SHORT_THRESHOLD_TREND_UP = -0.0003  # Less conservative in uptrend (-0.03%)
    SHORT_THRESHOLD_TREND_DOWN = -0.0007  # More conservative in downtrend (-0.07%)
    SHORT_THRESHOLD_RANGE = -0.0005  # Standard threshold in range-bound market (-0.05%)
    SHORT_THRESHOLD_HIGH_VOL = -0.0004  # Less conservative in high volatility (-0.04%)
    SHORT_THRESHOLD_LOW_VOL = -0.0006  # More conservative in low volatility (-0.06%)
    # Confidence-based adjustment (reduced impact)
    SHORT_THRESHOLD_CONFIDENCE_MULTIPLIER = 0.2  # Adjust threshold based on regime confidence

    def __init__(
        self,
        name="MlAdaptive",
        model_path="src/ml/btcusdt_price.onnx",
        sequence_length=120,
        use_prediction_engine: Optional[bool] = None,
        model_name: Optional[str] = None,
    ):
        # Initialize legacy attributes first
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
        # Create individual components
        signal_generator, risk_manager, position_sizer, regime_detector = self._create_components(
            name=name,
            model_path=model_path,
            sequence_length=sequence_length,
            use_prediction_engine=use_prediction_engine,
            model_name=model_name,
        )
        
        # Initialize adapter with individual components
        super().__init__(
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=regime_detector,
            name=name,
        )
        
        # Preserve legacy attributes for backward compatibility
        self.trading_pair = "BTCUSDT"
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
        
        # Model name resolution
        self.model_name = model_name
        if self.model_name is None:
            self.model_name = cfg.get("PREDICTION_ENGINE_MODEL_NAME", default=None)
        if self.model_name is None:
            try:
                self.model_name = Path(self.model_path).stem
            except Exception:
                self.model_name = None
        
        # Initialize regime detector for dynamic threshold adjustment
        self.regime_detector = RegimeDetector()

    def _create_components(
        self,
        name: str,
        model_path: str,
        sequence_length: int,
        use_prediction_engine: Optional[bool],
        model_name: Optional[str],
    ) -> tuple[MLSignalGenerator, RegimeAdaptiveRiskManager, ConfidenceWeightedSizer, EnhancedRegimeDetector]:
        """Create the component-based strategy with ML Adaptive configuration."""
        
        # Create signal generator with ML Adaptive parameters (regime-aware thresholds)
        signal_generator = MLSignalGenerator(
            name=f"{name}_signals",
            model_path=model_path,
            sequence_length=sequence_length,
            use_prediction_engine=use_prediction_engine,
            model_name=model_name,
        )
        
        # Create regime-adaptive risk manager
        risk_manager = RegimeAdaptiveRiskManager(
            base_risk=self.stop_loss_pct,
        )
        
        # Create position sizer with confidence weighting
        position_sizer = ConfidenceWeightedSizer(
            base_fraction=self.BASE_POSITION_SIZE,
            min_confidence=0.3,  # Minimum confidence for position sizing
        )
        
        # Create regime detector
        regime_detector = EnhancedRegimeDetector()
        
        # Return individual components
        return signal_generator, risk_manager, position_sizer, regime_detector

    def get_parameters(self) -> dict:
        """Return strategy parameters for logging"""
        return {
            "model_path": self.model_path,
            "sequence_length": self.sequence_length,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "use_prediction_engine": self.use_prediction_engine,
            "model_name": self.model_name,
        }

    def get_risk_overrides(self) -> Optional[dict[str, Any]]:
        """
        Provide risk management overrides including dynamic risk, partial operations, trailing stops, and time-based exits.
        
        This strategy uses:
        
        Dynamic Risk Management:
        - Adaptive position sizing based on performance and market conditions
        - Drawdown-based risk reduction: 5%, 10%, 15% thresholds
        - Recovery-based risk restoration: 2%, 5% positive return thresholds
        - Volatility-based adjustments for high/low volatility periods
        
        Conservative Partial Operations:
        - Take 25% profit at 3% gain, 25% at 6% gain, 50% at 10% gain
        - Scale in 25% at 2% gain, 25% at 5% gain (max 2 scale-ins)
        
        Conservative Trailing Stops:
        - Activate at 1.5% profit
        - 0.5% trailing distance
        - Breakeven at 2% profit
        
        Time-Based Exits:
        - Maximum 24-hour holding period for crypto positions
        - No weekend restrictions (crypto trades 24/7)
        - No overnight restrictions (crypto trades 24/7)
        """
        return {
            "dynamic_risk": {
                "enabled": True,
                "performance_window_days": 30,
                "drawdown_thresholds": [0.05, 0.10, 0.15],  # 5%, 10%, 15%
                "risk_reduction_factors": [0.8, 0.6, 0.4],   # 80%, 60%, 40% of normal size
                "recovery_thresholds": [0.02, 0.05],         # 2%, 5% positive returns
                "volatility_adjustment_enabled": True,
                "volatility_window_days": 30,
                "high_volatility_threshold": 0.03,           # 3% daily volatility
                "low_volatility_threshold": 0.01,            # 1% daily volatility
                "volatility_risk_multipliers": (0.7, 1.3),   # (high_vol, low_vol) multipliers
            },
            "partial_operations": {
                "exit_targets": [0.03, 0.06, 0.10],  # 3%, 6%, 10%
                "exit_sizes": [0.25, 0.25, 0.50],     # 25%, 25%, 50%
                "scale_in_thresholds": [0.02, 0.05],  # 2%, 5%
                "scale_in_sizes": [0.25, 0.25],       # 25%, 25%
                "max_scale_ins": 2,
            },
            "trailing_stop": {
                "activation_threshold": 0.015,  # 1.5%
                "trailing_distance_pct": 0.005,  # 0.5%
                "breakeven_threshold": 0.02,  # 2.0%
                "breakeven_buffer": 0.001,  # 0.1%
            },
            "time_exits": {
                "max_holding_hours": 24,  # Maximum 24-hour holding period
                "end_of_day_flat": False,  # No end-of-day restrictions for crypto
                "weekend_flat": False,     # No weekend restrictions for crypto
                "market_timezone": "UTC",  # Use UTC for crypto markets
                "time_restrictions": {
                    "no_overnight": False,     # No overnight restrictions for crypto
                    "no_weekend": False,       # No weekend restrictions for crypto
                    "trading_hours_only": False,  # No trading hours restrictions for crypto
                }
            }
        }
