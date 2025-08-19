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
                except Exception:
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
          - position_sizer: 'fixed_fraction' | 'confidence_weighted'
          - base_fraction: float (e.g., 0.02 for 2%)
          - min_fraction: float
          - max_fraction: float
          - stop_loss_pct: float
          - take_profit_pct: float or None
        If None, the RiskManager defaults are used.
        """
        return None
