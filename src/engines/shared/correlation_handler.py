"""Unified CorrelationHandler for both backtest and live trading engines.

Applies correlation control to limit exposure to correlated assets
during position sizing. Shared implementation ensures parity between
backtesting and live trading.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from src.data_providers.data_provider import DataProvider
    from src.position_management.correlation_engine import CorrelationEngine
    from src.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class CorrelationHandler:
    """Handles correlation-based position sizing adjustments.

    This class coordinates with the correlation engine to reduce
    position sizes when correlated exposure is high.
    """

    def __init__(
        self,
        correlation_engine: CorrelationEngine | None,
        risk_manager: RiskManager,
        data_provider: DataProvider,
        strategy: Any | None = None,
    ) -> None:
        """Initialize correlation handler.

        Args:
            correlation_engine: Engine for correlation calculations.
            risk_manager: Risk manager with position tracking.
            data_provider: Provider for historical data.
            strategy: Strategy for risk overrides.
        """
        self.correlation_engine = correlation_engine
        self.risk_manager = risk_manager
        self.data_provider = data_provider
        self.strategy = strategy

    def set_strategy(self, strategy: Any) -> None:
        """Update the strategy (for regime switching).

        Args:
            strategy: New strategy instance.
        """
        self.strategy = strategy

    def apply_correlation_control(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        index: int,
        candidate_fraction: float,
    ) -> float:
        """Apply correlation control to candidate position size.

        Args:
            symbol: Trading symbol.
            timeframe: Candle timeframe.
            df: DataFrame with market data.
            index: Current candle index.
            candidate_fraction: Proposed position size fraction.

        Returns:
            Adjusted position size fraction.
        """
        if candidate_fraction <= 0 or self.correlation_engine is None:
            return candidate_fraction

        # Get strategy overrides
        overrides = self._get_strategy_overrides()
        max_exposure_override = None
        if overrides:
            try:
                corr_cfg = overrides.get("correlation_control", {})
                max_exposure_override = corr_cfg.get("max_correlated_exposure")
            except Exception:
                pass

        # Build price series for correlation calculation
        price_series = self._build_price_series(symbol, timeframe, df, index)
        if not price_series:
            return candidate_fraction

        # Calculate correlation matrix
        corr_matrix = self._calculate_correlation_matrix(price_series, symbol)
        if corr_matrix is None:
            return candidate_fraction

        # Compute size reduction factor
        factor = self._compute_reduction_factor(
            symbol, candidate_fraction, corr_matrix, max_exposure_override
        )

        adjusted = candidate_fraction * factor

        # Apply maximum exposure limits
        adjusted = self._apply_exposure_limits(
            symbol, adjusted, corr_matrix, max_exposure_override, overrides
        )

        return max(0.0, adjusted)

    def _get_strategy_overrides(self) -> dict | None:
        """Get risk overrides from strategy."""
        if self.strategy is None:
            return None

        try:
            if hasattr(self.strategy, "get_risk_overrides"):
                return self.strategy.get_risk_overrides()
        except Exception:
            pass

        return None

    def _build_price_series(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        index: int,
    ) -> dict[str, pd.Series]:
        """Build price series dictionary for correlation calculation.

        Args:
            symbol: Primary trading symbol.
            timeframe: Candle timeframe.
            df: DataFrame with market data.
            index: Current candle index.

        Returns:
            Dictionary mapping symbols to close price series.
        """
        price_series: dict[str, pd.Series] = {}

        # Add primary symbol
        try:
            price_series[str(symbol)] = df["close"].astype(float)
        except Exception:
            try:
                price_series[str(symbol)] = df["close"]
            except Exception:
                return {}

        # Get time bounds
        end_ts = None
        try:
            end_ts = df.index[index]
        except Exception:
            try:
                end_ts = df.index[-1]
            except Exception:
                pass

        start_ts = None
        if isinstance(end_ts, pd.Timestamp):
            try:
                start_ts = end_ts - pd.Timedelta(
                    days=self.risk_manager.params.correlation_window_days
                )
            except Exception:
                pass

        # Get currently open positions
        open_symbols = set()
        try:
            if self.risk_manager.positions:
                open_symbols = set(map(str, self.risk_manager.positions.keys()))
        except Exception:
            pass

        # Fetch price series for open positions
        for sym in open_symbols:
            if sym == str(symbol) or sym in price_series:
                continue

            try:
                if start_ts is not None and isinstance(end_ts, pd.Timestamp):
                    hist = self.data_provider.get_historical_data(
                        sym,
                        timeframe=timeframe,
                        start=start_ts.to_pydatetime(),
                        end=end_ts.to_pydatetime(),
                    )
                else:
                    hist = self.data_provider.get_historical_data(sym, timeframe=timeframe)
            except Exception:
                continue

            if hist is None or hist.empty or "close" not in hist.columns:
                continue

            try:
                price_series[sym] = hist["close"].astype(float)
            except Exception:
                price_series[sym] = hist["close"]

        return price_series

    def _calculate_correlation_matrix(
        self,
        price_series: dict[str, pd.Series],
        symbol: str,
    ) -> pd.DataFrame | None:
        """Calculate correlation matrix from price series.

        Args:
            price_series: Dictionary of symbol to price series.
            symbol: Primary symbol for logging.

        Returns:
            Correlation matrix DataFrame, or None on failure.
        """
        try:
            return self.correlation_engine.calculate_position_correlations(price_series)
        except Exception as e:
            logger.debug("Failed to calculate correlation matrix for %s: %s", symbol, e)
            return None

    def _compute_reduction_factor(
        self,
        symbol: str,
        candidate_fraction: float,
        corr_matrix: pd.DataFrame,
        max_exposure_override: float | None,
    ) -> float:
        """Compute position size reduction factor based on correlations.

        Args:
            symbol: Trading symbol.
            candidate_fraction: Proposed position size.
            corr_matrix: Correlation matrix.
            max_exposure_override: Override for max exposure.

        Returns:
            Reduction factor (0.0 to 1.0).
        """
        try:
            return float(
                self.correlation_engine.compute_size_reduction_factor(
                    positions=self.risk_manager.positions,
                    corr_matrix=corr_matrix,
                    candidate_symbol=str(symbol),
                    candidate_fraction=float(candidate_fraction),
                    max_exposure_override=max_exposure_override,
                )
            )
        except Exception as e:
            logger.debug("Correlation size reduction failed for %s: %s", symbol, e)
            return 1.0

    def _apply_exposure_limits(
        self,
        symbol: str,
        adjusted: float,
        corr_matrix: pd.DataFrame,
        max_exposure_override: float | None,
        overrides: dict | None,
    ) -> float:
        """Apply maximum exposure limits based on correlation groups.

        Args:
            symbol: Trading symbol.
            adjusted: Current adjusted position size.
            corr_matrix: Correlation matrix.
            max_exposure_override: Override for max exposure.
            overrides: Strategy overrides.

        Returns:
            Position size respecting exposure limits.
        """
        # Determine max allowed exposure
        max_allowed = None
        try:
            if overrides and max_exposure_override is not None:
                max_allowed = float(max_exposure_override)
            else:
                max_allowed = float(self.correlation_engine.config.max_correlated_exposure)
        except Exception:
            return adjusted

        if max_allowed is None:
            return adjusted

        # Get correlation groups
        try:
            groups = self.correlation_engine.get_correlation_groups(corr_matrix)
        except Exception as e:
            logger.debug("Failed to derive correlation groups for %s: %s", symbol, e)
            return adjusted

        # Find current group exposure
        candidate_group_exposure = None
        for group in groups:
            if str(symbol) in group:
                current = 0.0
                for sym in group:
                    current += float(
                        self.risk_manager.positions.get(sym, {}).get("size", 0.0)
                    )
                candidate_group_exposure = current
                break

        if candidate_group_exposure is None:
            return adjusted

        # Apply remaining capacity limit
        remaining_capacity = max(0.0, max_allowed - candidate_group_exposure)
        if remaining_capacity <= 0:
            return 0.0

        return max(0.0, min(adjusted, remaining_capacity))
