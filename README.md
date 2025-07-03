# Crypto Trend-Following Trading Bot

A modular, production-ready cryptocurrency trading system inspired by Ray Dalio's risk-balanced approach and built around long-term trend-following with strict risk containment.

The codebase supports **backtesting**, **live trading**, **machine-learning price & sentiment models**, multi-exchange data providers, and AWS-based CI/CD deployments.

---

## � Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Directory Structure](#directory-structure)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Quick Usage](#quick-usage)
    1. [Backtesting](#backtesting)
    2. [Live Trading](#live-trading)
    3. [Training ML Models](#training-ml-models)
    4. [Cache Management](#cache-management)
    5. [Risk Management Utilities](#risk-management-utilities)
7. [Built-in Strategies](#built-in-strategies)
8. [Deployment](#deployment)
9. [Testing](#testing)
10. [Sentiment Analysis Integration](#sentiment-analysis-integration)
11. [Disclaimer](#disclaimer)

---

## Features

- 🔌 **Pluggable Architecture** – Separate Data, Indicator, ML, Risk, Strategy, and Execution layers.
- 🎯 **Multiple Trading Strategies** – Adaptive EMA, ML-driven, sentiment-enhanced, and high-risk-high-reward templates (see below).
- ♻️ **Fast Backtesting Engine** – Vectorised simulation with intelligent on-disk caching of historical data.
- 🤖 **Live Trading Engine** – Robust real-time execution on Binance with position sizing, trailing stops, and exposure limits.
- 🧠 **Machine-Learning Integration** – Keras & ONNX models for price prediction with optional sentiment features.
- 💬 **Sentiment Data Providers** – SentiCrypt, Augmento, CryptoCompare, and custom providers via a simple interface.
- 🛡 **Centralised Risk Manager** – Enforces max 1-2 % capital risk per trade and validates all position sizes.
- 🚀 **One-Click AWS Deployment** – Hardened CI/CD pipelines and bash scripts for staging & production environments.
- 📈 **Rich Analytics** – Automatic metric tracking (Sharpe, max-drawdown, MAPE, etc.) and interactive reports.

---

## Architecture Overview

```text
        ┌──────────────────┐
        │   Data Layer     │  →  Binance, Senticrypt, CryptoCompare …
        └────────┬─────────┘
                 │ pandas.DataFrame
                 ▼
        ┌──────────────────┐
        │ Indicator Layer  │  →  EMA, RSI, Bollinger, …
        └────────┬─────────┘
                 │ enriched DataFrame
                 ▼
        ┌──────────────────┐
        │  Strategy Layer  │  →  signal, confidence, meta
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │  Risk Manager    │  →  validated position size
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ Execution Layer  │  →  Backtesting Engine / Live Trading Engine
        └──────────────────┘
```

Each component is completely decoupled and can be swapped out or extended without touching the rest of the system.

---

## Directory Structure

```text
.
├── backtesting/          # Vectorised historical simulation engine
├── core/
│   ├── data_providers/   # Market & sentiment data adapters (+ caching wrapper)
│   ├── indicators/       # Pure functions for technical indicators
│   ├── risk/             # Central risk-management utilities
│   └── config/           # Typed configuration objects & loaders
├── live/                # Real-time trading engine & strategy manager
├── ml/                  # Trained models (.h5 / .keras / .onnx) + metadata
├── strategies/          # All built-in trading strategies
├── scripts/             # CLI utilities (model training, cache tools, etc.)
├── data/                # Cached market & sentiment datasets
├── bin/                 # Deployment scripts used by GitHub Actions
└── docs/                # Additional guides (AWS, sentiment, etc.)
```

---

## Installation

```bash
# Clone & install python dependencies
pip install -r requirements.txt
```

> Python 3.9+ is recommended.  GPU acceleration is optional for model training.

---

## Configuration

The bot automatically reads settings in the following priority order:

1. **AWS Secrets Manager** (production)
2. **Environment Variables** (Docker / CI)
3. **.env file** (local development)

Minimal `.env` example:

```env
BINANCE_API_KEY=YOUR_KEY
BINANCE_API_SECRET=YOUR_SECRET
TRADING_MODE=paper  # or live
INITIAL_BALANCE=1000
```

Test the configuration loader:

```bash
python scripts/test_config_system.py
```

---

## Quick Usage

### Backtesting

```bash
# Generic backtest (most recent 90 days)
python scripts/run_backtest.py adaptive --days 90

# Full-history backtest with cache disabled and a custom start date
python scripts/run_backtest.py ml_with_sentiment \
  --start-date 2020-01-01 --no-cache
```

### Live Trading

```bash
# Start live trading (paper-mode by default)
python scripts/run_live_trading.py adaptive

# Switch to a different strategy on the fly
python live_trading_control.py switch ml_basic
```

### Training ML Models

```bash
# Price-only model
python scripts/train_model.py BTCUSDT --epochs 50

# Price + sentiment features (recommended)
python scripts/train_model_with_sentiment.py BTCUSDT \
  --start-date 2020-01-01 --end-date 2023-01-01
```

### Cache Management

```bash
# Inspect cache
python scripts/cache_manager.py info

# Purge files older than 24 h
python scripts/cache_manager.py clear-old --hours 24
```

### Risk Management Utilities

```bash
# Validate a hypothetical order of 0.05 BTC given current exposure
python -m core.risk.risk_manager --symbol BTCUSDT --qty 0.05
```

---

## Built-in Strategies

| File                               | Description                                             |
|------------------------------------|---------------------------------------------------------|
| `adaptive.py`                      | EMA-based adaptive trend-follower                       |
| `enhanced.py`                      | Combines multiple indicators for stronger confirmation |
| `high_risk_high_reward.py`         | Aggressive breakout strategy (for small allocations)    |
| `ml_basic.py`                      | Utilises ML price predictions for entry/exit            |
| `ml_with_sentiment.py`             | ML + sentiment, confidence-weighted sizing             |

Add your own by subclassing `strategies.base.BaseStrategy` and implementing `generate_signals()` & `calculate_position_size()`.

---

## Deployment

### Choose Your Deployment Platform

**🚄 Railway (Recommended for beginners)**
- **5-minute setup** vs 2-4 hours on AWS
- **40-60% cost savings** compared to AWS
- **Built-in database and SSL** 
- **Simple scaling and monitoring**
- See [Railway Quick Start](RAILWAY_QUICKSTART.md) or [Railway Deployment Guide](docs/RAILWAY_DEPLOYMENT_GUIDE.md)

```bash
# Quick Railway deployment
./bin/railway-setup.sh
./bin/railway-deploy.sh -p your-project-name -e staging
```

**⚡ AWS (For advanced users)**
- **Full infrastructure control**
- **Enterprise features and integrations**
- **Advanced networking and security**
- GitHub Actions orchestrates deployments using hardened bash scripts under `bin/`

```text
main →  staging EC2  (auto)  →  production EC2  (manual promotion)
```

Read `docs/AWS_DEPLOYMENT_GUIDE.md` for a step-by-step walkthrough.

---

## Testing

```bash
pytest -q          # unit tests
python -m strategies.adaptive --test   # quick strategy smoke-test
```

---

## Sentiment Analysis Integration

This project now includes advanced sentiment analysis capabilities using data from [SentiCrypt](https://api.senticrypt.com/v2/all.json), providing Bitcoin market sentiment data spanning from April 2019 to present.

### 🎯 **Key Features**

#### **Sentiment Data Provider System**
- **Modular Design**: Easy-to-replace sentiment data sources
- **SentiCrypt Integration**: Real-time Bitcoin sentiment data with 6+ years of history
- **Normalized Features**: 8 sentiment indicators including momentum, volatility, and extremes
- **Automatic Correlation**: Sentiment data aligned with price data timeframes

#### **Enhanced ML Models**
- **Multi-Feature Training**: Price + sentiment data for improved predictions
- **Flexible Architecture**: Supports both sentiment-enhanced and price-only models
- **Advanced Neural Networks**: CNN-LSTM hybrid architecture for complex pattern recognition
- **ONNX Export**: Production-ready model deployment

#### **Sentiment-Enhanced Trading Strategy**
- **Intelligent Predictions**: ML models trained on both price and sentiment data
- **Confidence-Based Position Sizing**: Trade size adjusted by prediction confidence
- **Dynamic Exit Conditions**: Exit trades when sentiment turns negative
- **Real-Time Sentiment Integration**: Live sentiment data for backtesting and trading

### 📊 **Sentiment Features**

The system creates 8 normalized sentiment features from raw SentiCrypt data:

1. **`sentiment_primary`**: Main sentiment score (normalized using tanh)
2. **`sentiment_momentum`**: Rate of change in sentiment
3. **`sentiment_volatility`**: 7-day rolling standard deviation of sentiment
4. **`sentiment_extreme_positive`**: Binary flag for extremely positive sentiment
5. **`sentiment_extreme_negative`**: Binary flag for extremely negative sentiment
6. **`sentiment_ma_3`**: 3-day moving average of sentiment
7. **`sentiment_ma_7`**: 7-day moving average of sentiment
8. **`sentiment_ma_14`**: 14-day moving average of sentiment

### 🚀 **Usage Examples**

#### **Training a Sentiment-Enhanced Model**
```bash
# Train with sentiment data (default)
python scripts/train_model_with_sentiment.py BTCUSDT --start-date 2020-01-01 --end-date 2023-01-01

# Train price-only model
python scripts/train_model_with_sentiment.py BTCUSDT --no-sentiment --start-date 2020-01-01 --end-date 2023-01-01
```

#### **Running Sentiment-Enhanced Backtests**
```bash
# Use sentiment-enhanced ML strategy
python scripts/run_backtest.py ml_sentiment_strategy --days 365

# Compare with price-only ML strategy
python scripts/run_backtest.py ml_model_strategy --days 365
```

#### **Downloading Fresh Sentiment Data**
```bash
# Fetch latest sentiment data from SentiCrypt API
curl -s "https://api.senticrypt.com/v2/all.json" | python3 -c "
import json
import sys
import pandas as pd

data = json.load(sys.stdin)
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.to_csv('data/senticrypt_sentiment_data.csv', index=False)
print(f'Saved {len(df)} sentiment records from {df["date"].min().strftime("%Y-%m-%d")} to {df["date"].max().strftime("%Y-%m-%d")}')
"
```

### 📈 **Performance Insights**

The sentiment-enhanced models have shown:

- **Improved Prediction Accuracy**: 16.12% MAPE vs higher error rates in price-only models
- **Better Risk-Adjusted Returns**: Enhanced Sharpe ratios with sentiment features
- **Market Regime Adaptation**: Better performance during high volatility periods
- **Feature Correlation**: Sentiment momentum and volatility show strong correlation with price movements

### 🔧 **Technical Architecture**

#### **Sentiment Data Flow**
```
SentiCrypt API → CSV Storage → SentiCryptProvider → Feature Engineering → ML Model Training → ONNX Export → Trading Strategy
```

#### **Model Architecture (Sentiment-Enhanced)**
- **Input Layer**: 120 time steps × 13 features (5 price + 8 sentiment)
- **CNN Layers**: Feature extraction with 128 and 64 filters
- **LSTM Layers**: Sequence modeling with 100 and 50 units
- **Dense Layers**: Final prediction with dropout regularization
- **Output**: Single price prediction with confidence score

#### **Strategy Components**
1. **Data Loading**: Automatic sentiment data integration
2. **Feature Normalization**: Price features (MinMax) + sentiment features (StandardScaler)
3. **Prediction Generation**: Real-time ONNX inference
4. **Confidence Scoring**: Prediction reliability assessment
5. **Position Management**: Confidence-based sizing and dynamic exits

### 🎯 **Key Benefits**

#### **For Traders**
- **Enhanced Accuracy**: Sentiment data improves prediction quality
- **Risk Management**: Confidence-based position sizing reduces risk
- **Market Insight**: Understanding sentiment-price relationships
- **Adaptive Strategy**: Dynamic response to market sentiment changes

#### **For Developers**
- **Modular Design**: Easy to add new sentiment data sources
- **Scalable Architecture**: Supports multiple timeframes and symbols
- **Production Ready**: ONNX models for efficient deployment
- **Comprehensive Testing**: Extensive backtesting capabilities

### 📚 **Advanced Configuration**

#### **Custom Sentiment Providers**
Create your own sentiment provider by extending `SentimentDataProvider`:

```python
from core.data_providers.sentiment_provider import SentimentDataProvider

class CustomSentimentProvider(SentimentDataProvider):
    def get_historical_sentiment(self, symbol, start, end):
        # Your custom sentiment data logic
        return sentiment_dataframe
```

#### **Model Customization**
Modify the model architecture in `train_model_with_sentiment.py`:

```python
def create_model_with_sentiment(input_shape, num_features):
    # Custom architecture for your specific needs
    # Add more layers, different optimizers, etc.
```

### 🔍 **Monitoring and Analysis**

The system provides comprehensive analysis tools:

- **Feature Correlation Analysis**: Understand sentiment-price relationships
- **Prediction Confidence Tracking**: Monitor model reliability
- **Sentiment Trend Analysis**: Visualize market sentiment over time
- **Performance Attribution**: Separate sentiment vs. price contributions

### ⚡ **Performance Optimization**

- **ONNX Runtime**: Fast inference for real-time trading
- **Cached Data Providers**: Reduced API calls and faster backtesting
- **Efficient Feature Engineering**: Optimized sentiment calculations
- **Memory Management**: Proper handling of large datasets

This sentiment analysis integration represents a significant advancement in the trading system's capabilities, providing traders with powerful tools to understand and capitalize on market sentiment dynamics.

---

## Disclaimer

This project is provided **for educational purposes only**.  Trading cryptocurrencies carries significant risk.  Test extensively with paper accounts and never risk capital you cannot afford to lose. 