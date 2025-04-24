"""
Configuration settings for the trading bot
"""

import os
from pathlib import Path

# API Configuration
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')

# Ensure API keys are set
if not API_KEY or not API_SECRET:
    raise ValueError(
        "Binance API credentials not found. Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables."
    )

# Data directory for storing historical data, logs, etc.
DATA_DIR = Path(__file__).parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

# Default trading parameters
DEFAULT_SYMBOL = 'BTCUSDT'
DEFAULT_TIMEFRAME = '1h'
DEFAULT_INITIAL_BALANCE = 10000

# Risk management defaults
DEFAULT_RISK_PER_TRADE = 0.02  # 2% risk per trade
DEFAULT_MAX_RISK_PER_TRADE = 0.03  # 3% maximum risk per trade
DEFAULT_MAX_POSITION_SIZE = 0.25  # 25% maximum position size
DEFAULT_MAX_DRAWDOWN = 0.20  # 20% maximum drawdown 