"""
ML Basic Strategy - Component-Based Implementation

This strategy uses a machine learning model trained exclusively on price data (OHLCV).
It provides simple, reliable predictions without external dependencies like sentiment data.

Key Features:
- Price-only predictions using LSTM neural network
- 120-day sequence length for pattern recognition
- Normalized price inputs for better model performance
- 2% stop loss, 4% take profit risk management
- No external API dependencies
- Component-based architecture for better maintainability

Ideal for:
- Consistent, reliable trading signals
- Backtesting historical periods
- Environments where sentiment data is unavailable
- Simple deployment scenarios
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
from src.prediction.features.price_only import PriceOnlyFeatureExtractor
from src.prediction.features.technical import TechnicalFeatureExtractor
from src.prediction.models.registry import PredictionModelRegistry
from src.strategies.adapters.legacy_adapter import LegacyStrategyAdapter
from src.strategies.components import (
    Strategy,
    MLBasicSignalGenerator,
    FixedRiskManager,
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
)


class MlBasic(LegacyStrategyAdapter):
    """
    ML Basic Strategy using component-based architecture.
    
    This implementation wraps the component-based strategy with LegacyStrategyAdapter
    to maintain backward compatibility with the existing BaseStrategy interface.
    """
    
    # * Strategy configuration constants
    SHORT_ENTRY_THRESHOLD = -0.0005  # -0.05% threshold for short entries
    CONFIDENCE_MULTIPLIER = 12  # Multiplier for confidence calculation
    BASE_POSITION_SIZE = 0.2  # Base position size (20% of balance)
    MIN_POSITION_SIZE_RATIO = 0.05  # Minimum position size (5% of balance)
    MAX_POSITION_SIZE_RATIO = 0.25  # Maximum position size (25% of balance)

    def __init__(
        self,
        name="MlBasic",
        model_path="src/ml/btcusdt_price.onnx",
        sequence_length=120,
        use_prediction_engine: Optional[bool] = None,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        timeframe: Optional[str] = None,
    ):
        # Initialize legacy attributes first (needed by _create_component_strategy)
        self.trading_pair = "BTCUSDT"
        self.model_type = model_type or "basic"
        self.model_timeframe = timeframe or "1h"
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
        # Create components for the adapter
        signal_generator, risk_manager, position_sizer, regime_detector = self._create_components(
            name=name,
            model_path=model_path,
            sequence_length=sequence_length,
            use_prediction_engine=use_prediction_engine,
            model_name=model_name,
            model_type=model_type,
            timeframe=timeframe,
        )
        
        # Initialize adapter with components
        super().__init__(
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=regime_detector,
            name=name
        )
        
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

    def _create_components(
        self,
        name: str,
        model_path: str,
        sequence_length: int,
        use_prediction_engine: Optional[bool],
        model_name: Optional[str],
        model_type: Optional[str],
        timeframe: Optional[str],
    ) -> tuple[MLBasicSignalGenerator, FixedRiskManager, ConfidenceWeightedSizer, EnhancedRegimeDetector]:
        """Create the components for ML Basic strategy."""
        
        # Create signal generator with ML Basic parameters
        signal_generator = MLBasicSignalGenerator(
            name=f"{name}_signals",
            model_path=model_path,
            sequence_length=sequence_length,
            use_prediction_engine=use_prediction_engine,
            model_name=model_name,
            model_type=model_type,
            timeframe=timeframe,
        )
        
        # Create risk manager with fixed stop loss and take profit
        risk_manager = FixedRiskManager(
            risk_per_trade=self.stop_loss_pct,
            stop_loss_pct=self.stop_loss_pct,
        )
        
        # Create position sizer with confidence weighting
        position_sizer = ConfidenceWeightedSizer(
            base_fraction=self.BASE_POSITION_SIZE,
            min_confidence=0.3,
        )
        
        # Create regime detector
        regime_detector = EnhancedRegimeDetector()
        
        return signal_generator, risk_manager, position_sizer, regime_detector

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators including ML predictions
        
        This method adds the ml_prediction column to maintain backward compatibility
        with tests and existing code that expect this column.
        """
        # First call the parent implementation to get regime annotations
        df = super().calculate_indicators(df)
        
        # Initialize prediction columns with safe defaults
        df["ml_prediction"] = np.nan
        df["prediction_confidence"] = np.nan
        
        # Use the signal generator's feature pipeline if available
        signal_gen = self.signal_generator
        if hasattr(signal_gen, 'feature_pipeline') and signal_gen.feature_pipeline is not None:
            df = signal_gen.feature_pipeline.transform(df)
        
        # Generate predictions for each row using the signal generator's logic
        if hasattr(signal_gen, '_get_ml_prediction'):
            for i in range(len(df)):
                try:
                    prediction = signal_gen._get_ml_prediction(df, i)
                    if prediction is not None:
                        df.loc[df.index[i], "ml_prediction"] = prediction
                except Exception:
                    # Skip rows where prediction fails
                    pass
        
        return df
    
    def get_parameters(self) -> dict:
        """Get strategy parameters for backward compatibility."""
        return {
            "name": self.name,
            "model_path": self.model_path,
            "sequence_length": self.sequence_length,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "use_prediction_engine": self.use_prediction_engine,
            "engine_model_name": self.model_name,
        }
