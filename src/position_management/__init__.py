"""Position management module for dynamic risk and position sizing."""

from .dynamic_risk import DynamicRiskConfig, DynamicRiskManager

# Re-export new MFE/MAE modules if present
try:
    from .mfe_mae_tracker import MFEMAETracker
except Exception:
    MFEMAETracker = None  # type: ignore

try:
    from .mfe_mae_analyzer import MFEMAEAnalyzer
except Exception:
    MFEMAEAnalyzer = None  # type: ignore

__all__ = ["DynamicRiskConfig", "DynamicRiskManager", "MFEMAETracker", "MFEMAEAnalyzer"]
