"""
Multi-Strategy Portfolio Manager

This component manages a portfolio of multiple strategies, allocating capital
across them dynamically based on performance, regime, and risk metrics.

Key Features:
1. Dynamic Weight Allocation
   - Performance-based: Allocate more to recent winners
   - Regime-based: Allocate more to strategies suited for current regime
   - Risk-adjusted: Allocate more to strategies with better Sharpe ratios

2. Portfolio-Level Risk Management
   - Total position limits across all strategies
   - Correlation-aware allocation
   - Drawdown-based rebalancing

3. Rebalancing Logic
   - Periodic rebalancing (daily, weekly)
   - Threshold-based rebalancing (when allocations drift)
   - Emergency rebalancing (on large drawdowns)

Expected Benefits:
- Smoother equity curve (diversification)
- Better Sharpe ratio (risk-adjusted returns)
- Lower maximum drawdown
- More consistent performance across market conditions
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from .regime_context import RegimeContext
from .strategy import Strategy

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """Strategy allocation methods"""

    EQUAL = "equal"  # Equal weights across all strategies
    PERFORMANCE = "performance"  # Weight by recent performance
    REGIME = "regime"  # Weight by regime suitability
    RISK_ADJUSTED = "risk_adjusted"  # Weight by Sharpe ratio
    ADAPTIVE = "adaptive"  # Combination of all methods
    HIERARCHICAL = "hierarchical"  # Regime → Strategy selection


@dataclass
class StrategyAllocation:
    """Allocation for a single strategy"""

    strategy: Strategy
    target_weight: float = 0.0  # Target allocation weight (0-1)
    current_weight: float = 0.0  # Current allocation weight
    current_position_value: float = 0.0  # Current position value
    unrealized_pnl: float = 0.0  # Unrealized P&L
    realized_pnl: float = 0.0  # Realized P&L
    total_return: float = 0.0  # Total return %
    sharpe_ratio: float = 0.0  # Sharpe ratio
    max_drawdown: float = 0.0  # Maximum drawdown
    win_rate: float = 0.0  # Win rate
    trades: int = 0  # Number of trades
    regime_suitability: float = 0.5  # 0-1, how suited for current regime


@dataclass
class PortfolioState:
    """Current state of the multi-strategy portfolio"""

    total_equity: float  # Total portfolio equity
    cash: float  # Available cash
    total_position_value: float  # Sum of all position values
    allocations: dict[str, StrategyAllocation]  # Strategy name → allocation
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    overall_return: float = 0.0
    timestamp: Optional[pd.Timestamp] = None


class MultiStrategyPortfolio:
    """
    Manages a portfolio of multiple strategies with dynamic allocation

    This class runs multiple strategies in parallel and dynamically
    allocates capital between them based on various factors.
    """

    def __init__(
        self,
        strategies: list[Strategy],
        initial_balance: float = 10000.0,
        allocation_method: AllocationMethod = AllocationMethod.ADAPTIVE,
        # Rebalancing parameters
        rebalance_frequency: int = 24,  # Rebalance every N candles
        rebalance_threshold: float = 0.10,  # Rebalance if drift > 10%
        # Performance tracking
        performance_lookback: int = 100,  # Candles for performance calculation
        # Risk limits
        max_total_position: float = 0.80,  # Max 80% of portfolio in positions
        max_per_strategy_position: float = 0.30,  # Max 30% per strategy
        min_per_strategy_allocation: float = 0.05,  # Min 5% per strategy
        # Correlation management
        enable_correlation_adjustment: bool = True,
        correlation_lookback: int = 50,
    ):
        """
        Initialize multi-strategy portfolio

        Args:
            strategies: List of Strategy instances to manage
            initial_balance: Starting portfolio balance
            allocation_method: Method for allocating capital
            rebalance_frequency: How often to rebalance (in candles)
            rebalance_threshold: Rebalance if allocation drifts by this amount
            performance_lookback: Number of candles for performance metrics
            max_total_position: Maximum total position size (as fraction of equity)
            max_per_strategy_position: Maximum position per strategy
            min_per_strategy_allocation: Minimum allocation per strategy
            enable_correlation_adjustment: Reduce allocation to correlated strategies
            correlation_lookback: Period for correlation calculation
        """
        self.strategies = {s.name: s for s in strategies}
        self.initial_balance = initial_balance
        self.allocation_method = allocation_method
        self.rebalance_frequency = rebalance_frequency
        self.rebalance_threshold = rebalance_threshold
        self.performance_lookback = performance_lookback
        self.max_total_position = max_total_position
        self.max_per_strategy_position = max_per_strategy_position
        self.min_per_strategy_allocation = min_per_strategy_allocation
        self.enable_correlation_adjustment = enable_correlation_adjustment
        self.correlation_lookback = correlation_lookback

        # Portfolio state
        self.current_balance = initial_balance
        self.allocations: dict[str, StrategyAllocation] = {}

        # Initialize allocations
        self._initialize_allocations()

        # Performance tracking
        self.equity_curve: list[float] = [initial_balance]
        self.returns: deque = deque(maxlen=performance_lookback)
        self.strategy_returns: dict[str, deque] = {
            name: deque(maxlen=performance_lookback) for name in self.strategies.keys()
        }

        # Rebalancing tracking
        self.candles_since_rebalance = 0
        self.rebalance_count = 0

        logger.info(
            f"MultiStrategyPortfolio initialized with {len(strategies)} strategies, "
            f"${initial_balance:,.2f} initial balance, "
            f"allocation method: {allocation_method.value}"
        )

    def _initialize_allocations(self) -> None:
        """Initialize equal allocations for all strategies"""
        equal_weight = 1.0 / len(self.strategies)

        for name, strategy in self.strategies.items():
            self.allocations[name] = StrategyAllocation(
                strategy=strategy,
                target_weight=equal_weight,
                current_weight=equal_weight,
            )

        logger.info(f"Initialized {len(self.allocations)} allocations with equal weights")

    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state"""
        total_position_value = sum(a.current_position_value for a in self.allocations.values())
        total_unrealized_pnl = sum(a.unrealized_pnl for a in self.allocations.values())
        total_realized_pnl = sum(a.realized_pnl for a in self.allocations.values())
        total_equity = self.current_balance + total_position_value
        overall_return = ((total_equity - self.initial_balance) / self.initial_balance) * 100

        return PortfolioState(
            total_equity=total_equity,
            cash=self.current_balance,
            total_position_value=total_position_value,
            allocations=self.allocations.copy(),
            total_unrealized_pnl=total_unrealized_pnl,
            total_realized_pnl=total_realized_pnl,
            overall_return=overall_return,
        )

    def calculate_target_weights(
        self, df: pd.DataFrame, index: int, regime: Optional[RegimeContext] = None
    ) -> dict[str, float]:
        """
        Calculate target allocation weights for each strategy

        Args:
            df: Price data
            index: Current candle index
            regime: Current market regime

        Returns:
            Dictionary mapping strategy name to target weight (0-1)
        """
        if self.allocation_method == AllocationMethod.EQUAL:
            return self._equal_weights()
        elif self.allocation_method == AllocationMethod.PERFORMANCE:
            return self._performance_based_weights()
        elif self.allocation_method == AllocationMethod.REGIME:
            return self._regime_based_weights(regime)
        elif self.allocation_method == AllocationMethod.RISK_ADJUSTED:
            return self._risk_adjusted_weights()
        elif self.allocation_method == AllocationMethod.ADAPTIVE:
            return self._adaptive_weights(regime)
        elif self.allocation_method == AllocationMethod.HIERARCHICAL:
            return self._hierarchical_weights(regime)
        else:
            return self._equal_weights()

    def _equal_weights(self) -> dict[str, float]:
        """Equal allocation across all strategies"""
        equal_weight = 1.0 / len(self.strategies)
        return {name: equal_weight for name in self.strategies.keys()}

    def _performance_based_weights(self) -> dict[str, float]:
        """Allocate based on recent performance (total return)"""
        if not self.strategy_returns:
            return self._equal_weights()

        # Calculate performance score for each strategy
        performance_scores = {}
        for name, returns in self.strategy_returns.items():
            if len(returns) < 10:  # Need minimum data
                performance_scores[name] = 0.0
            else:
                # Use recent returns (last 20 periods)
                recent_returns = list(returns)[-20:]
                avg_return = np.mean(recent_returns) if recent_returns else 0.0
                performance_scores[name] = max(0.0, avg_return)  # Only positive performance

        # Normalize to weights (if all negative, use equal weights)
        total_score = sum(performance_scores.values())
        if total_score <= 0:
            return self._equal_weights()

        weights = {name: score / total_score for name, score in performance_scores.items()}

        # Apply minimum allocation constraint
        return self._apply_min_allocation(weights)

    def _regime_based_weights(self, regime: Optional[RegimeContext]) -> dict[str, float]:
        """Allocate based on regime suitability"""
        if regime is None:
            return self._equal_weights()

        # Strategy regime suitability (simplified - would be more sophisticated in practice)
        regime_suitability = {}

        for name in self.strategies.keys():
            if "crash" in name.lower():
                # Crash avoiders excel in CRASH_RISK and TRENDING_DOWN
                if hasattr(regime, "primary_regime"):
                    if "CRASH" in regime.primary_regime or "DOWN" in regime.primary_regime:
                        regime_suitability[name] = 0.8
                    else:
                        regime_suitability[name] = 0.2
                else:
                    regime_suitability[name] = 0.5

            elif "trend" in name.lower():
                # Trend followers excel in TRENDING_UP and TRENDING_DOWN
                if hasattr(regime, "primary_regime"):
                    if "TREND" in regime.primary_regime:
                        regime_suitability[name] = 0.8
                    else:
                        regime_suitability[name] = 0.3
                else:
                    regime_suitability[name] = 0.5

            elif "volatility" in name.lower():
                # Volatility exploiters excel in HIGH_VOLATILITY
                if hasattr(regime, "primary_regime"):
                    if "VOLATILITY" in regime.primary_regime or "HIGH" in regime.primary_regime:
                        regime_suitability[name] = 0.8
                    else:
                        regime_suitability[name] = 0.4
                else:
                    regime_suitability[name] = 0.5

            elif "adaptive" in name.lower() or "regime" in name.lower():
                # Adaptive strategies work in all regimes
                regime_suitability[name] = 0.7

            else:
                # Default suitability
                regime_suitability[name] = 0.5

        # Normalize to weights
        total_score = sum(regime_suitability.values())
        weights = {name: score / total_score for name, score in regime_suitability.items()}

        return self._apply_min_allocation(weights)

    def _risk_adjusted_weights(self) -> dict[str, float]:
        """Allocate based on Sharpe ratio (risk-adjusted returns)"""
        # Calculate Sharpe-like metric for each strategy
        sharpe_scores = {}

        for name, returns in self.strategy_returns.items():
            if len(returns) < 20:
                sharpe_scores[name] = 0.0
            else:
                returns_array = np.array(list(returns))
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)

                if std_return > 0:
                    sharpe = mean_return / std_return
                    sharpe_scores[name] = max(0.0, sharpe)  # Only positive Sharpe
                else:
                    sharpe_scores[name] = 0.0

        # Normalize to weights
        total_score = sum(sharpe_scores.values())
        if total_score <= 0:
            return self._equal_weights()

        weights = {name: score / total_score for name, score in sharpe_scores.items()}

        return self._apply_min_allocation(weights)

    def _adaptive_weights(self, regime: Optional[RegimeContext]) -> dict[str, float]:
        """
        Combine multiple allocation methods adaptively

        Weights:
        - 40% performance-based
        - 30% regime-based
        - 30% risk-adjusted
        """
        perf_weights = self._performance_based_weights()
        regime_weights = self._regime_based_weights(regime)
        risk_weights = self._risk_adjusted_weights()

        # Combine with weights
        combined = {}
        for name in self.strategies.keys():
            combined[name] = (
                0.40 * perf_weights.get(name, 0.0)
                + 0.30 * regime_weights.get(name, 0.0)
                + 0.30 * risk_weights.get(name, 0.0)
            )

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {name: weight / total for name, weight in combined.items()}
        else:
            combined = self._equal_weights()

        return self._apply_min_allocation(combined)

    def _hierarchical_weights(self, regime: Optional[RegimeContext]) -> dict[str, float]:
        """
        Hierarchical allocation: First select strategies by regime, then by performance

        1. Filter strategies by regime suitability (>0.6)
        2. Among suitable strategies, allocate by performance
        """
        regime_weights = self._regime_based_weights(regime)

        # Filter to high-suitability strategies
        suitable_strategies = {
            name: weight for name, weight in regime_weights.items() if weight > 0.15
        }

        if not suitable_strategies:
            # Fallback to all strategies if none suitable
            suitable_strategies = regime_weights

        # Among suitable strategies, use performance-based allocation
        perf_weights = self._performance_based_weights()

        # Filter performance weights to only suitable strategies
        filtered_perf = {name: perf_weights.get(name, 0.0) for name in suitable_strategies.keys()}

        # Normalize
        total = sum(filtered_perf.values())
        if total > 0:
            filtered_perf = {name: weight / total for name, weight in filtered_perf.items()}
        else:
            # Equal among suitable
            equal = 1.0 / len(suitable_strategies)
            filtered_perf = {name: equal for name in suitable_strategies.keys()}

        # Set non-suitable strategies to minimum allocation
        final_weights = {
            name: (
                filtered_perf[name]
                if name in filtered_perf
                else self.min_per_strategy_allocation
            )
            for name in self.strategies.keys()
        }

        # Renormalize
        total = sum(final_weights.values())
        final_weights = {name: weight / total for name, weight in final_weights.items()}

        return final_weights

    def _apply_min_allocation(self, weights: dict[str, float]) -> dict[str, float]:
        """Ensure all strategies get at least minimum allocation"""
        adjusted = weights.copy()

        # Apply minimum
        for name in adjusted.keys():
            if adjusted[name] < self.min_per_strategy_allocation:
                adjusted[name] = self.min_per_strategy_allocation

        # Renormalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {name: weight / total for name, weight in adjusted.items()}

        return adjusted

    def should_rebalance(self) -> bool:
        """Determine if portfolio should rebalance"""
        # Time-based rebalancing
        if self.candles_since_rebalance >= self.rebalance_frequency:
            logger.info(f"Rebalancing triggered: {self.candles_since_rebalance} candles elapsed")
            return True

        # Drift-based rebalancing
        max_drift = 0.0
        for name, allocation in self.allocations.items():
            drift = abs(allocation.current_weight - allocation.target_weight)
            max_drift = max(max_drift, drift)

        if max_drift > self.rebalance_threshold:
            logger.info(f"Rebalancing triggered: max drift {max_drift:.2%} > threshold")
            return True

        return False

    def rebalance(
        self, df: pd.DataFrame, index: int, regime: Optional[RegimeContext] = None
    ) -> None:
        """
        Rebalance portfolio to target weights

        Args:
            df: Price data
            index: Current candle index
            regime: Current market regime
        """
        # Calculate new target weights
        new_weights = self.calculate_target_weights(df, index, regime)

        # Update target weights
        for name, weight in new_weights.items():
            if name in self.allocations:
                self.allocations[name].target_weight = weight

        # Reset rebalance counter
        self.candles_since_rebalance = 0
        self.rebalance_count += 1

        logger.info(
            f"Portfolio rebalanced (#{self.rebalance_count}). "
            f"New weights: {', '.join([f'{name}: {weight:.1%}' for name, weight in new_weights.items()])}"
        )

    def get_strategy_capital(self, strategy_name: str) -> float:
        """Get capital allocated to a specific strategy"""
        if strategy_name not in self.allocations:
            return 0.0

        allocation = self.allocations[strategy_name]
        portfolio_state = self.get_portfolio_state()

        # Calculate capital based on target weight and total equity
        allocated_capital = portfolio_state.total_equity * allocation.target_weight

        # Apply maximum position constraint
        max_capital = portfolio_state.total_equity * self.max_per_strategy_position
        allocated_capital = min(allocated_capital, max_capital)

        return allocated_capital
