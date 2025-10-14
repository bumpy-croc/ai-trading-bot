"""Momentum leverage strategy for trend capture."""

from __future__ import annotations

from src.strategies.components import (
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
    MomentumSignalGenerator,
    Strategy,
    VolatilityRiskManager,
)


class MomentumLeverage(Strategy):
    """Aggressive momentum strategy with broad risk tolerances."""

    BASE_POSITION_SIZE = 0.70
    MIN_POSITION_SIZE_RATIO = 0.40
    MAX_POSITION_SIZE_RATIO = 0.95

    STOP_LOSS_PCT = 0.10
    TAKE_PROFIT_PCT = 0.35

    MOMENTUM_ENTRY_THRESHOLD = 0.01
    STRONG_MOMENTUM_THRESHOLD = 0.025
    TREND_LOOKBACK = 15

    def __init__(self, name: str = "MomentumLeverage") -> None:
        self.trading_pair = "BTCUSDT"

        signal_generator = MomentumSignalGenerator(
            name=f"{name}_signals",
            momentum_entry_threshold=self.MOMENTUM_ENTRY_THRESHOLD,
            strong_momentum_threshold=self.STRONG_MOMENTUM_THRESHOLD,
        )

        risk_manager = VolatilityRiskManager(
            base_risk=self.STOP_LOSS_PCT,
            atr_multiplier=2.0,
            min_risk=0.005,
            max_risk=0.15,
        )

        position_sizer = ConfidenceWeightedSizer(
            base_fraction=min(0.5, self.BASE_POSITION_SIZE),
            min_confidence=0.2,
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
                    "drawdown_thresholds": [0.25, 0.35, 0.45],
                    "risk_reduction_factors": [0.95, 0.85, 0.75],
                    "recovery_thresholds": [0.12, 0.25],
                },
                "partial_operations": {
                    "exit_targets": [0.08, 0.15, 0.25],
                    "exit_sizes": [0.20, 0.30, 0.50],
                    "scale_in_thresholds": [0.02, 0.04],
                    "scale_in_sizes": [0.4, 0.3],
                    "max_scale_ins": 3,
                },
                "trailing_stop": {
                    "activation_threshold": 0.06,
                    "trailing_distance_pct": 0.03,
                    "breakeven_threshold": 0.10,
                    "breakeven_buffer": 0.02,
                },
            }
        )

    def get_parameters(self) -> dict[str, object]:
        """Return strategy parameters for reporting."""

        return {
            "name": self.name,
            "base_position_size": self.BASE_POSITION_SIZE,
            "max_position_size": self.MAX_POSITION_SIZE_RATIO,
            "stop_loss_pct": self.STOP_LOSS_PCT,
            "take_profit_pct": self.TAKE_PROFIT_PCT,
            "momentum_entry_threshold": self.MOMENTUM_ENTRY_THRESHOLD,
            "strong_momentum_threshold": self.STRONG_MOMENTUM_THRESHOLD,
            "trend_lookback": self.TREND_LOOKBACK,
        }
