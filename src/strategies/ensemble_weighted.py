"""Optimised weighted ensemble strategy."""

from __future__ import annotations

from src.strategies.components import (
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
    MLBasicSignalGenerator,
    MLSignalGenerator,
    Strategy,
    VolatilityRiskManager,
    WeightedVotingSignalGenerator,
)


class EnsembleWeighted(Strategy):
    """Strategy that blends multiple ML generators with adaptive weighting."""

    MIN_STRATEGIES_FOR_SIGNAL = 1
    PERFORMANCE_WINDOW = 30
    WEIGHT_UPDATE_FREQUENCY = 10

    BASE_POSITION_SIZE = 0.50
    MIN_POSITION_SIZE_RATIO = 0.20
    MAX_POSITION_SIZE_RATIO = 0.80

    STOP_LOSS_PCT = 0.06
    TAKE_PROFIT_PCT = 0.20

    def __init__(
        self,
        name: str = "EnsembleWeighted",
        use_ml_basic: bool = True,
        use_ml_adaptive: bool = True,
        use_ml_sentiment: bool = False,
    ) -> None:
        self.trading_pair = "BTCUSDT"

        generators: dict[Strategy, float] = {}
        if use_ml_basic:
            generators[MLBasicSignalGenerator()] = 0.30
        if use_ml_adaptive:
            generators[MLSignalGenerator()] = 0.30
        if use_ml_sentiment:
            generators[MLSignalGenerator()] = 0.15

        signal_generator = WeightedVotingSignalGenerator(
            generators=generators,
            min_confidence=0.3,
            consensus_threshold=0.6,
        )

        risk_manager = VolatilityRiskManager(
            base_risk=self.STOP_LOSS_PCT,
            atr_multiplier=2.0,
            min_risk=0.005,
            max_risk=0.05,
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
                "position_sizer": "confidence_weighted",
                "base_fraction": self.BASE_POSITION_SIZE,
                "min_fraction": self.MIN_POSITION_SIZE_RATIO,
                "max_fraction": self.MAX_POSITION_SIZE_RATIO,
                "stop_loss_pct": self.STOP_LOSS_PCT,
                "take_profit_pct": self.TAKE_PROFIT_PCT,
                "dynamic_risk": {
                    "enabled": True,
                    "drawdown_thresholds": [0.15, 0.25, 0.35],
                    "risk_reduction_factors": [0.95, 0.8, 0.6],
                    "recovery_thresholds": [0.08, 0.15],
                },
                "partial_operations": {
                    "exit_targets": [0.06, 0.10, 0.15],
                    "exit_sizes": [0.15, 0.25, 0.60],
                    "scale_in_thresholds": [0.015, 0.03, 0.05],
                    "scale_in_sizes": [0.3, 0.25, 0.2],
                    "max_scale_ins": 4,
                },
                "trailing_stop": {
                    "activation_threshold": 0.04,
                    "trailing_distance_pct": 0.02,
                    "breakeven_threshold": 0.06,
                    "breakeven_buffer": 0.01,
                },
            }
        )

    def get_parameters(self) -> dict[str, object]:
        """Return ensemble configuration details."""

        return {
            "name": self.name,
            "min_strategies_for_signal": self.MIN_STRATEGIES_FOR_SIGNAL,
            "performance_window": self.PERFORMANCE_WINDOW,
            "weight_update_frequency": self.WEIGHT_UPDATE_FREQUENCY,
            "base_position_size": self.BASE_POSITION_SIZE,
            "stop_loss_pct": self.STOP_LOSS_PCT,
            "take_profit_pct": self.TAKE_PROFIT_PCT,
        }
