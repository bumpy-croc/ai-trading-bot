# Binance Trading Bot

A simple trading bot that implements a moving average crossover strategy on the Binance platform.

## Features

- Moving Average Crossover Strategy (10 and 20 period)
- Real-time trading on Binance
- Configurable trading parameters
- Error handling and logging

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your Binance API credentials:
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
TRADING_PAIR=BTCUSDT
QUANTITY=0.001
SHORT_MA_PERIOD=10
LONG_MA_PERIOD=20
```

3. Get your API keys from Binance:
   - Log in to your Binance account
   - Go to API Management
   - Create a new API key
   - Make sure to enable trading permissions

## Usage

Run the trading bot:
```bash
python trading_bot.py
```

The bot will:
- Monitor the specified trading pair
- Calculate moving averages
- Execute trades when crossover signals are detected
- Log all activities and errors

## Strategy

The bot uses a simple moving average crossover strategy:
- Buy signal: When the short-term MA (10 periods) crosses above the long-term MA (20 periods)
- Sell signal: When the short-term MA crosses below the long-term MA

## Data Caching

The trading bot includes an intelligent data caching system that significantly speeds up backtesting by avoiding redundant API calls to data providers.

### How It Works

- **Automatic Caching**: Historical data is automatically cached after being fetched from exchanges
- **Smart Cache Keys**: Each data request (symbol, timeframe, date range) gets a unique cache key
- **TTL Management**: Cache files have a configurable time-to-live (default: 24 hours)
- **Transparent Operation**: The caching layer is completely transparent to strategies

### Benefits

- **Faster Backtests**: Subsequent runs with the same parameters use cached data
- **Reduced API Usage**: Fewer calls to exchange APIs, avoiding rate limits
- **Offline Testing**: Run backtests without internet connection using cached data
- **Development Efficiency**: Iterate on strategies without waiting for data downloads

### Usage

Caching is enabled by default. To disable it:

```bash
python run_backtest.py my_strategy --no-cache
```

To set custom cache TTL (in hours):

```bash
python run_backtest.py my_strategy --cache-ttl 48
```

### Cache Management

Use the cache management utility to monitor and manage cached data:

```bash
# Show cache information
python scripts/cache_manager.py info

# List all cache files
python scripts/cache_manager.py list

# List cache files with detailed information
python scripts/cache_manager.py list --detailed

# Clear all cache files
python scripts/cache_manager.py clear

# Clear cache files older than 48 hours
python scripts/cache_manager.py clear-old --hours 48
```

### Cache Location

Cache files are stored in `data/cache/` directory as pickle files. Each file contains a pandas DataFrame with OHLCV data for a specific request.

## Disclaimer

This is a basic trading bot for educational purposes. Always test thoroughly with small amounts before using real funds. The creator is not responsible for any financial losses incurred while using this bot. 