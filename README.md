# Crypto Trend-Following Trading Bot

A modular, production-ready cryptocurrency trading system inspired by Ray Dalio's risk-balanced approach and built around long-term trend-following with strict risk containment.

The codebase supports **backtesting**, **live trading**, **machine-learning price & sentiment models**, multi-exchange data providers, and AWS-based CI/CD deployments.

---

## Table of Contents

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
8. [Database Architecture](#database-architecture)
9. [Deployment](#deployment)
10. [Testing](#testing)
11. [Sentiment Analysis Integration](#sentiment-analysis-integration)
12. [Disclaimer](#disclaimer)

---

## Features

- 🔌 **Pluggable Architecture** – Separate Data, Indicator, ML, Risk, Strategy, and Execution layers.
- 🎯 **Multiple Trading Strategies** – Adaptive EMA, ML-driven, sentiment-enhanced, and high-risk-high-reward templates (see below).
- ♻️ **Fast Backtesting Engine** – Vectorised simulation with intelligent on-disk caching of historical data.
- 🤖 **Live Trading Engine** – Robust real-time execution on Binance with position sizing, trailing stops, and exposure limits.
- 💾 **Persistent Balance & Positions** – Never lose progress on restarts; automatic balance recovery and position restoration.
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

# Run database migration for persistent balance features
python scripts/migrate_database.py migrate
```

> Python 3.9+ is recommended. GPU acceleration is optional for model training.
> 
> **🔄 For existing users**: The database migration adds persistent balance tracking so your trading progress survives restarts.

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
# Balance and positions automatically recovered from last session
python scripts/run_live_trading.py adaptive --balance 1000

# The bot will display recovery information:
# 💾 Recovered balance from previous session: $1,250.00
# 🔄 Recovering 2 active positions...
# ✅ Recovered position: BTCUSDT long @ $45,000.00

# Switch to a different strategy on the fly
python live_trading_control.py switch ml_basic

# Manually adjust balance via dashboard at http://localhost:8080
# Or via API: POST /api/balance {"balance": 5000, "reason": "Added funds"}
```

> **💾 Persistent Progress**: Your balance and active positions are automatically saved and recovered on restart, so Railway deployments never lose your trading progress.

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

## Database Architecture

The trading bot features a **centralized database architecture** that allows multiple services (trading bot and dashboard) to share the same database while maintaining seamless local development capabilities.

### 🏗️ **Centralized Database Design**

#### **Local Development**
- **PostgreSQL Option**: Environment parity with production (recommended)
- **SQLite Fallback**: Zero setup for quick development
- **Flexible Configuration**: Easy switching between database types
- **Data Isolation**: Local data separate from production

#### **Production (Railway)**
- **PostgreSQL Database**: Shared across all services
- **Automatic Detection**: Services automatically use PostgreSQL when `DATABASE_URL` is available
- **Connection Pooling**: Efficient connection management
- **ACID Transactions**: Data integrity for financial operations

### 📊 **Database Schema**

The system uses a comprehensive schema designed for trading operations:

- **`trading_sessions`**: Track trading sessions with strategy configuration
- **`trades`**: Complete trade history with entry/exit prices and P&L
- **`positions`**: Active positions with real-time unrealized P&L
- **`account_history`**: Balance snapshots for performance tracking
- **`performance_metrics`**: Aggregated metrics (win rate, Sharpe ratio, drawdown)
- **`system_events`**: System logs and error tracking
- **`strategy_executions`**: Detailed strategy decision logs

### 🔧 **Database Configuration**

#### **Automatic Configuration**
```python
# The system automatically detects the database type:
if DATABASE_URL:  # PostgreSQL on Railway
    use PostgreSQL with connection pooling
else:  # Local development
    use SQLite with default settings
```

#### **Railway Setup**
1. Create PostgreSQL database service in Railway
2. Deploy services (no code changes needed)
3. Database connection configured automatically

#### **Local Development Options**

**Option 1: PostgreSQL (Recommended for Environment Parity)**
```bash
# Quick setup with guided wizard
python scripts/setup_local_development.py

# Manual setup
docker-compose up -d postgres
# Edit .env: DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
```

**Option 2: SQLite (Simple Development)**
- Uses SQLite database at `src/data/trading_bot.db`
- No configuration required (default when DATABASE_URL not set)
- Same commands work unchanged

### 🛠️ **Database Tools**

#### **Setup and Verification**
```bash
# Display Railway database setup instructions
python scripts/railway_database_setup.py

# Verify database connection and configuration
python scripts/railway_database_setup.py --verify

# Check if data migration is needed
python scripts/railway_database_setup.py --check-migration
```

#### **Data Migration (if needed)**
```bash
# Export existing SQLite data for migration
python scripts/export_sqlite_data.py

# Import data to PostgreSQL
python scripts/import_to_postgresql.py
```

#### **Connection Testing**
```bash
# Test database connection and basic operations
python scripts/verify_database_connection.py

# Setup local development environment (PostgreSQL or SQLite)
python scripts/setup_local_development.py
```

### 📈 **Benefits**

#### **For Multi-Service Architecture**
- **Shared Data**: Trading bot and dashboard access same database
- **Real-Time Updates**: Dashboard shows live trading data
- **Data Consistency**: ACID transactions ensure data integrity
- **Scalability**: Services can scale independently

#### **For Development**
- **Local SQLite**: Fast development without external dependencies
- **Production PostgreSQL**: Robust production database
- **Seamless Transition**: Same code works in both environments
- **Zero Configuration**: Automatic database type detection

#### **For Operations**
- **Connection Pooling**: Efficient database resource usage
- **Backup & Recovery**: Railway's built-in backup features
- **Monitoring**: Database performance metrics
- **Security**: Encrypted connections and private networking

### 🔍 **Database Choice: PostgreSQL vs Redis**

**PostgreSQL** was chosen over Redis for the centralized database:

✅ **PostgreSQL Advantages**:
- Perfect SQLAlchemy integration with existing models
- ACID transactions for financial data integrity
- Complex queries with joins and analytics
- Relational structure for foreign keys
- Built-in backup and recovery on Railway
- Cost-effective for persistent storage

❌ **Redis Limitations**:
- NoSQL nature would require complete model rewrite
- No built-in relationships or foreign keys
- Limited query capabilities (no SQL joins)
- Memory storage increases costs for large datasets
- No complex analytical queries

### 📊 **Performance Features**

- **Connection Pooling**: 5 connections with 10 overflow for PostgreSQL
- **SSL/TLS Encryption**: Secure connections in production
- **Query Optimization**: PostgreSQL's advanced query planner
- **Indexing**: Optimized indexes for trading queries
- **Private Networking**: Railway's internal network for service communication

### 🎯 **How It Works**

1. **Service Initialization**: Database manager detects environment
2. **Database Selection**: Uses `DATABASE_URL` if available, else SQLite
3. **Connection Setup**: Configures appropriate connection pooling
4. **Schema Creation**: Creates tables automatically if needed
5. **Service Ready**: Both trading bot and dashboard use same database

For detailed setup instructions, see:
- [Database Centralization Guide](docs/RAILWAY_DATABASE_CENTRALIZATION_GUIDE.md) - Railway PostgreSQL setup
- [Local PostgreSQL Setup](docs/LOCAL_POSTGRESQL_SETUP.md) - Local development with PostgreSQL

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

# Deploy – reads railway.json, starts the trading engine **and** monitoring dashboard
railway up
```

> The monitoring dashboard is served automatically at your Railway domain (root path).

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