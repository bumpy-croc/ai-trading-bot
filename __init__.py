"""
Bottrade - Cryptocurrency Trading Bot Framework
"""

from core.data_providers import DataProvider, BinanceDataProvider
from core.indicators import (
    calculate_moving_averages,
    calculate_rsi,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd
)
from core.risk import RiskManager, RiskParameters
from strategies import BaseStrategy
from backtesting import Backtester, Trade

__version__ = '0.1.0'

__all__ = [
    # Data providers
    'DataProvider',
    'BinanceDataProvider',
    
    # Technical indicators
    'calculate_moving_averages',
    'calculate_rsi',
    'calculate_atr',
    'calculate_bollinger_bands',
    'calculate_macd',
    
    # Risk management
    'RiskManager',
    'RiskParameters',
    
    # Strategy and backtesting
    'BaseStrategy',
    'Backtester',
    'Trade'
] 