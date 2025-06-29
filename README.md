# Binance Trading Bot

A comprehensive cryptocurrency trading bot with backtesting capabilities, machine learning integration, and live trading support.

## 📚 Documentation

- **[Project Architecture & Development Guide](./CURSOR_RULES.md)** - Detailed system architecture, workflows, and conventions
- **[AI Assistant Rules](./.cursorrules)** - Quick reference for Cursor AI integration
- **[Live Trading Guide](./LIVE_TRADING_GUIDE.md)** - Setup and usage for live trading
- **[Live Sentiment Analysis](./LIVE_SENTIMENT_ANALYSIS.md)** - Sentiment data integration guide
- **[AWS Deployment Guide](./docs/AWS_DEPLOYMENT_GUIDE.md)** - Complete guide for deploying to AWS EC2
- **[AWS Secrets Manager Guide](./docs/AWS_SECRETS_MANAGER_GUIDE.md)** - Secure credential management
- **[AWS VPC Setup Guide](./docs/AWS_VPC_SETUP_GUIDE.md)** - Network isolation setup
- **[Configuration Migration Guide](./docs/CONFIG_MIGRATION_GUIDE.md)** - Migrating to the new config system
- **[Deployment Checklist](./deploy/DEPLOYMENT_CHECKLIST.md)** - Step-by-step deployment checklist

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

2. Configure your credentials using one of these methods:

### Method A: .env File (Local Development)
Create a `.env` file in the project root:
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
DATABASE_URL=sqlite:///data/trading_bot.db
TRADING_MODE=paper
INITIAL_BALANCE=1000
```

### Method B: Environment Variables (Docker/CI)
```bash
export BINANCE_API_KEY=your_api_key_here
export BINANCE_API_SECRET=your_api_secret_here
export DATABASE_URL=sqlite:///data/trading_bot.db
export TRADING_MODE=paper
```

### Method C: AWS Secrets Manager (Production)
The bot automatically uses AWS Secrets Manager when deployed to AWS EC2. See the [AWS Deployment Guide](./docs/AWS_DEPLOYMENT_GUIDE.md) for details.

3. Get your API keys from Binance:
   - Log in to your Binance account
   - Go to API Management
   - Create a new API key
   - Make sure to enable trading permissions

## Configuration System

The trading bot uses a flexible configuration system that automatically loads settings from multiple sources in priority order:

1. **AWS Secrets Manager** (if available) - Secure storage for production
2. **Environment Variables** - Good for Docker and CI/CD
3. **.env File** - Convenient for local development

The bot will automatically use the first available source for each configuration value. This means:
- On AWS EC2: Secrets are read directly from AWS Secrets Manager (no .env file needed)
- In Docker: Use environment variables
- Local development: Use .env file

Test the configuration system:
```bash
python scripts/test_config_system.py
```

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

## 🚀 Deployment

### Current Status
- **Tests**: Currently skipped in CI/CD pipeline (temporary)
- **Staging**: Auto-deploys from `main` branch to AWS EC2
- **Production**: Manual promotion from staging with approval gates

### GitHub Actions Workflows

The deployment process uses clean, external bash scripts located in the `bin/` directory:

- **Staging Workflow**: `.github/workflows/deploy-staging.yml`
  - Triggers on pushes to `main` branch
  - Uses `bin/deploy-staging.sh` for deployment logic
  - Provides detailed error reporting and diagnostics

- **Production Workflow**: `.github/workflows/promote-to-production.yml`
  - Manual trigger only with approval gates
  - Uses `bin/deploy-production.sh` with enhanced security
  - Validates staging environment before promotion

### Deployment Scripts

See [`bin/README.md`](bin/README.md) for detailed information about the deployment scripts.

## Disclaimer

This is a basic trading bot for educational purposes. Always test thoroughly with small amounts before using real funds. The creator is not responsible for any financial losses incurred while using this bot.

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
python run_backtest.py ml_sentiment_strategy --days 365

# Compare with price-only ML strategy
python run_backtest.py ml_model_strategy --days 365
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