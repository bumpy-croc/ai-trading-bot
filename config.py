import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Binance API Configuration
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Trading Configuration
TRADING_PAIR = os.getenv('TRADING_PAIR', 'BTCUSDT')
QUANTITY = float(os.getenv('QUANTITY', '0.001'))

# Strategy Parameters
SHORT_MA_PERIOD = int(os.getenv('SHORT_MA_PERIOD', '10'))
LONG_MA_PERIOD = int(os.getenv('LONG_MA_PERIOD', '20'))

# Time intervals
INTERVAL = '1h'  # 1 hour candles 