"""ML Sentiment strategy combining price and sentiment predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.config.config_manager import get_config
from src.config.constants import DEFAULT_USE_PREDICTION_ENGINE
from src.strategies.components import (
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
    FixedRiskManager,
    MLSignalGenerator,
    Strategy,
)


class MlSentiment(Strategy):
    """Strategy that augments ML signals with sentiment-aware sizing."""

    BASE_POSITION_SIZE = 0.2
    SENTIMENT_BOOST_MULTIPLIER = 1.2
    SENTIMENT_REDUCTION_MULTIPLIER = 0.8
    MIN_SENTIMENT_CONFIDENCE = 0.6

    def __init__(
        self,
        name: str = "MlSentiment",
        model_path: str = "src/ml/btcusdt_sentiment.onnx",
        sequence_length: int = 120,
        use_prediction_engine: Optional[bool] = None,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> None:
        self.trading_pair = "BTCUSDT"
        self.model_type = model_type or "sentiment"
        self.model_timeframe = timeframe or "1h"
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04

        cfg = get_config()
        self.use_prediction_engine = (
            use_prediction_engine
            if use_prediction_engine is not None
            else cfg.get_bool("USE_PREDICTION_ENGINE", default=DEFAULT_USE_PREDICTION_ENGINE)
        )

        resolved_model_name = model_name
        if resolved_model_name is None:
            resolved_model_name = cfg.get("PREDICTION_ENGINE_MODEL_NAME", default=None)
        if resolved_model_name is None:
            try:
                resolved_model_name = Path(model_path).stem
            except Exception:
                resolved_model_name = None
        self.model_name = resolved_model_name

        signal_generator = MLSignalGenerator(
            name=f"{name}_signals",
            model_path=model_path,
            sequence_length=sequence_length,
            use_prediction_engine=self.use_prediction_engine,
            model_name=self.model_name,
        )

        risk_manager = FixedRiskManager(
            risk_per_trade=self.stop_loss_pct,
            stop_loss_pct=self.stop_loss_pct,
        )

        position_sizer = ConfidenceWeightedSizer(
            base_fraction=self.BASE_POSITION_SIZE,
            min_confidence=0.3,
        )

        regime_detector = EnhancedRegimeDetector()

        super().__init__(
            name=name,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=regime_detector,
        )

        self.set_risk_overrides(
            {
                "partial_operations": {
                    "exit_targets": [0.03, 0.06, 0.10],
                    "exit_sizes": [0.25, 0.25, 0.50],
                    "scale_in_thresholds": [0.02, 0.05],
                    "scale_in_sizes": [0.25, 0.25],
                    "max_scale_ins": 2,
                },
                "trailing_stop": {
                    "activation_threshold": 0.015,
                    "trailing_distance_pct": 0.005,
                    "breakeven_threshold": 0.02,
                    "breakeven_buffer": 0.001,
                },
            }
        )

    def get_parameters(self) -> dict[str, object]:
        """Return strategy parameters for reporting."""

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
