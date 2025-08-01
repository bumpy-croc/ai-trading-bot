from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Optional, Dict, List, Any
from utils.symbol_factory import SymbolFactory

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
        self.trading_pair = 'BTCUSDT'
        
        # Strategy execution logging
        self.db_manager = None
        self.session_id = None
        self.enable_execution_logging = True
        
    def set_database_manager(self, db_manager, session_id: Optional[int] = None):
        """Set database manager for strategy execution logging"""
        self.db_manager = db_manager
        self.session_id = session_id
        
    def log_execution(
        self,
        signal_type: str,
        action_taken: str,
        price: float,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        signal_strength: Optional[float] = None,
        confidence_score: Optional[float] = None,
        indicators: Optional[Dict] = None,
        sentiment_data: Optional[Dict] = None,
        ml_predictions: Optional[Dict] = None,
        position_size: Optional[float] = None,
        reasons: Optional[List[str]] = None,
        volume: Optional[float] = None,
        volatility: Optional[float] = None,
        additional_context: Optional[Dict] = None
    ):
        """
        Log strategy execution details to database.
        
        This allows strategies to log their internal decision-making process,
        providing insights into why certain signals were generated.
        """
        if not self.enable_execution_logging or not self.db_manager:
            return
            
        try:
            # Merge additional context into reasons if provided
            final_reasons = reasons or []
            if additional_context:
                context_reasons = [f"{k}={v}" for k, v in additional_context.items()]
                final_reasons.extend(context_reasons)
            
            self.db_manager.log_strategy_execution(
                strategy_name=self.name,
                symbol=symbol or self.trading_pair,
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
                session_id=self.session_id
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
    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = 'long') -> float:
        """Calculate stop loss level for a position"""
        pass
        
    @abstractmethod
    def get_parameters(self) -> dict:
        """Return strategy parameters for logging"""
        pass
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for strategy execution"""
        return self.calculate_indicators(df.copy()) 