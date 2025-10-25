"""Simplified ML Basic strategy for focused debugging.

This variant bypasses confidence weighting and additional heuristics so we can
observe raw model-driven signals. It always binds to the latest BTCUSDT basic
model bundle (``src/ml/models/BTCUSDT/basic/latest``) unless a custom model path
is provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.risk.risk_manager import RiskManager as EngineRiskManager
from src.risk.risk_manager import RiskParameters
from src.strategies.components import (
    CoreRiskAdapter,
    EnhancedRegimeDetector,
    FixedFractionSizer,
    MLBasicSignalGenerator,
    Strategy,
)


def _resolve_latest_basic_model(symbol: str = "BTCUSDT") -> str:
    """Return the ONNX path for the promoted latest basic model."""
    model_dir = Path("src/ml/models") / symbol / "basic" / "latest"
    if not model_dir.exists():
        raise FileNotFoundError(f"Latest model directory not found: {model_dir}")

    onnx_files = sorted(model_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX models present in {model_dir}")

    return str(onnx_files[0])


def create_ml_basic_simple_strategy(
    name: str = "MlBasicSimple",
    symbol: str = "BTCUSDT",
    sequence_length: int = 120,
    position_fraction: float = 0.02,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
    model_path: Optional[str] = None,
    use_prediction_engine: bool = False,
) -> Strategy:
    """Compose a minimal ml_basic strategy bound to the latest basic model."""

    resolved_model_path = model_path or _resolve_latest_basic_model(symbol)

    signal_generator = MLBasicSignalGenerator(
        name=f"{name}_signals",
        model_path=resolved_model_path,
        sequence_length=sequence_length,
        use_prediction_engine=use_prediction_engine,
        model_type="basic",
        timeframe="1h",
    )

    risk_parameters = RiskParameters(
        base_risk_per_trade=position_fraction,
        default_take_profit_pct=take_profit_pct,
        max_position_size=position_fraction,
    )
    core_risk_manager = EngineRiskManager(risk_parameters)
    risk_adapter = CoreRiskAdapter(core_risk_manager)
    risk_adapter.set_strategy_overrides(
        {
            "position_sizer": "fixed_fraction",
            "base_fraction": position_fraction,
            "max_fraction": position_fraction,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        }
    )

    position_sizer = FixedFractionSizer(fraction=position_fraction)
    regime_detector = EnhancedRegimeDetector()

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_adapter,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
        enable_logging=True,
    )

    return strategy
