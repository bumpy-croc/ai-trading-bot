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

## Disclaimer

This is a basic trading bot for educational purposes. Always test thoroughly with small amounts before using real funds. The creator is not responsible for any financial losses incurred while using this bot. 