import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

from src.utils.symbol_factory import SymbolFactory


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    All concrete strategy implementations must inherit from this class.
    Default trading pair is Binance style (e.g., BTCUSDT).

    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)

        # Default trading pair - strategies can override this
        self.trading_pair = "BTCUSDT"

        # Optional model selection preferences for registry-based strategies
        self.model_type: str | None = None
        self.model_timeframe: str | None = None

        # Strategy execution logging
        self.db_manager = None
        self.session_id = None
        self.enable_execution_logging = True

    def set_database_manager(self, db_manager, session_id: Optional[int] = None):
        """Set database manager for strategy execution logging"""
        self.db_manager = db_manager

    def log_execution(
        self,
        signal_type: str,
        action_taken: str,
        price: float,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        signal_strength: Optional[float] = None,
        confidence_score: Optional[float] = None,
        indicators: Optional[dict] = None,
        sentiment_data: Optional[dict] = None,
        ml_predictions: Optional[dict] = None,
        position_size: Optional[float] = None,
        reasons: Optional[list[str]] = None,
        volume: Optional[float] = None,
        volatility: Optional[float] = None,
        additional_context: Optional[dict] = None,
    ):
        """
        Log strategy execution details to database.

        This allows strategies to log their internal decision-making process,
        providing insights into why certain signals were generated.

        """
        try:
            if not self.db_manager or not self.enable_execution_logging:
                return
            # Merge additional context into reasons for auditability
            final_reasons = list(reasons) if reasons else []
            if additional_context:
                try:
                    context_pairs = [f"{k}={v}" for k, v in additional_context.items()]
                    final_reasons.extend(context_pairs)
                except Exception as e:
                    self.logger.debug(f"Failed to process additional context: {e}")
                    pass
            # Use the SymbolFactory for consistent formatting
            # Default to the strategy's trading pair when symbol is omitted or None
            effective_symbol = symbol or self.trading_pair
            # Normalize to Binance-style symbol for consistency
            try:
                symbol_code = SymbolFactory.to_exchange_symbol(effective_symbol, "binance")
            except Exception:
                symbol_code = effective_symbol
            self.db_manager.log_strategy_execution(
                strategy_name=self.__class__.__name__,
                symbol=symbol_code,
                signal_type=signal_type,
                action_taken=action_taken,
                price=price,
                timeframe=timeframe,
                signal_strength=signal_strength,
                confidence_score=confidence_score,
                indicators=indicators,
                sentiment_data=sentiment_data,
                ml_predictions=ml_predictions,
                position_size=position_size,
                reasons=final_reasons,
                volume=volume,
                volatility=volatility,
                session_id=self.session_id,
            )

        except Exception as e:
            self.logger.warning(f"Failed to log strategy execution: {e}")

    def get_trading_pair(self) -> str:
        """Get the trading pair for this strategy"""
        return self.trading_pair

    def set_trading_pair(self, trading_pair: str):
        """Set the trading pair for this strategy"""
        self.trading_pair = trading_pair

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific indicators on the data"""
        pass

    @abstractmethod
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if entry conditions are met at the given index"""
        pass

    @abstractmethod
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check if exit conditions are met at the given index"""
        pass

    @abstractmethod
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate the position size for a new trade"""
        pass

    @abstractmethod
    def calculate_stop_loss(
        self, df: pd.DataFrame, index: int, price: float, side: str = "long"
    ) -> float:
        """Calculate stop loss level for a position"""
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """Return strategy parameters for logging"""
        pass

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for strategy execution"""
        return self.calculate_indicators(df.copy())

    def get_risk_overrides(self) -> Optional[dict[str, Any]]:
        """
        Optional hook: strategies can provide risk/position management overrides.
        Expected keys may include:
          - position_sizer: 'fixed_fraction' | 'confidence_weighted' | 'atr_risk'
          - base_fraction: float (e.g., 0.02 for 2%)
          - min_fraction: float
          - max_fraction: float
          - stop_loss_pct: float
          - take_profit_pct: float or None
          - dynamic_risk: dict with dynamic risk configuration overrides:
            - enabled: bool
            - drawdown_thresholds: List[float]
            - risk_reduction_factors: List[float]
            - recovery_thresholds: List[float]
            - volatility_adjustment_enabled: bool
          - partial_operations: dict with partial exit and scale-in configuration:
            - exit_targets: List[float] (e.g., [0.03, 0.06, 0.10] for 3%, 6%, 10%)
            - exit_sizes: List[float] (e.g., [0.25, 0.25, 0.50] for 25%, 25%, 50%)
            - scale_in_thresholds: List[float] (e.g., [0.02, 0.05] for 2%, 5%)
            - scale_in_sizes: List[float] (e.g., [0.25, 0.25] for 25%, 25%)
            - max_scale_ins: int (e.g., 2)
          - trailing_stop: dict with trailing parameters (all decimals):
            - activation_threshold: float  # e.g., 0.015 for 1.5%
            - trailing_distance_pct: float | None  # e.g., 0.005 for 0.5%
            - trailing_distance_atr_mult: float | None  # e.g., 1.5
            - breakeven_threshold: float  # e.g., 0.02 for 2%
            - breakeven_buffer: float  # e.g., 0.001 for 0.1%
            
        Example:
            return {
                'dynamic_risk': {
                    'enabled': True,
                    'drawdown_thresholds': [0.03, 0.08, 0.15],
                    'risk_reduction_factors': [0.9, 0.7, 0.5]
                },
                'partial_operations': {
                    'exit_targets': [0.03, 0.06, 0.10],
                    'exit_sizes': [0.25, 0.25, 0.50],
                    'scale_in_thresholds': [0.02, 0.05],
                    'scale_in_sizes': [0.25, 0.25],
                    'max_scale_ins': 2
                },
                'trailing_stop': {
                    'activation_threshold': 0.015,
                    'trailing_distance_pct': 0.005,
                    'breakeven_threshold': 0.02,
                    'breakeven_buffer': 0.001,
                }
            }
            
        If None, the RiskManager defaults are used.
        """
        return None
