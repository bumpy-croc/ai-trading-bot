from abc import ABC, abstractmethod
import pandas as pd
import logging

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    All concrete strategy implementations must inherit from this class.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Default trading pair - strategies can override this
        self.trading_pair = 'BTCUSDT'
        
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