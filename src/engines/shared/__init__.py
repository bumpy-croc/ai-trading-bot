"""Shared engine components used by both backtest and live trading engines.

This module provides unified implementations for common trading engine functionality:
- Position and Trade models
- Cost calculations (fees, slippage)
- Trailing stop management
- Dynamic risk handling
- Policy hydration
- Risk configuration
- Partial operations management
- Performance tracking
- Correlation-based position sizing
"""

from src.engines.shared.correlation_handler import CorrelationHandler
from src.engines.shared.cost_calculator import CostCalculator, CostResult
from src.engines.shared.dynamic_risk_handler import (
    DynamicRiskAdjustment,
    DynamicRiskHandler,
)
from src.engines.shared.models import (
    BasePosition,
    BaseTrade,
    OrderStatus,
    PartialExitResult,
    Position,
    PositionSide,
    ScaleInResult,
    Trade,
    normalize_side,
)
from src.engines.shared.partial_operations_manager import (
    PartialExitDecision,
    PartialOperationsManager,
    ScaleInDecision,
)
# Re-export from canonical location for backward compatibility
from src.performance.tracker import (
    PerformanceMetrics,
    PerformanceTracker,
)
from src.engines.shared.policy_hydration import (
    HydratedPolicies,
    PolicyHydrator,
    apply_policies_to_engine,
)
from src.engines.shared.risk_configuration import (
    build_time_exit_policy,
    build_trailing_stop_policy,
    extract_risk_overrides,
    get_risk_parameters,
    merge_dynamic_risk_config,
)
from src.engines.shared.strategy_exit_checker import (
    StrategyExitChecker,
    StrategyExitResult,
)
from src.engines.shared.trailing_stop_manager import (
    TrailingStopManager,
    TrailingStopUpdate,
)

__all__ = [
    # Models
    "PositionSide",
    "OrderStatus",
    "normalize_side",
    "BasePosition",
    "BaseTrade",
    "Position",
    "Trade",
    # Cost calculation
    "CostCalculator",
    "CostResult",
    # Trailing stop management
    "TrailingStopManager",
    "TrailingStopUpdate",
    # Strategy exit checking
    "StrategyExitChecker",
    "StrategyExitResult",
    # Dynamic risk handling
    "DynamicRiskHandler",
    "DynamicRiskAdjustment",
    # Policy hydration
    "PolicyHydrator",
    "HydratedPolicies",
    "apply_policies_to_engine",
    # Risk configuration
    "merge_dynamic_risk_config",
    "build_trailing_stop_policy",
    "build_time_exit_policy",
    "extract_risk_overrides",
    "get_risk_parameters",
    # Partial operations
    "PartialOperationsManager",
    "PartialExitDecision",
    "ScaleInDecision",
    "PartialExitResult",
    "ScaleInResult",
    # Performance tracking
    "PerformanceTracker",
    "PerformanceMetrics",
    # Correlation handling
    "CorrelationHandler",
]
