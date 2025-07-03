# Paper Trading Quick Start Guide

## Prerequisites Checklist ✓

Before you can run paper trading with live Binance data, ensure you have:

### 1. ✅ Database Logging (Completed!)
- Database infrastructure is set up and working
- All trades will be automatically logged

### 2. ⚠️ Binance API Credentials (Required)
Even for paper trading, you need Binance API credentials to fetch live market data.

#### Getting Binance API Keys:
1. Log in to your Binance account
2. Go to API Management: https://www.binance.com/en/my/settings/api-management
3. Create a new API key with a label (e.g., "AI-Trading-Bot Paper Trading")
4. **Important Security Settings:**
   - ❌ **Disable** "Enable Spot & Margin Trading" (for safety)
   - ❌ **Disable** "Enable Withdrawals"
   - ✅ **Enable** only "Enable Reading" (all we need for paper trading)
   - ✅ Restrict API access to your IP address if possible

### 3. 📝 Environment Configuration

Create a `.env` file in the project root:

```bash
# Create .env file
cat > .env << 'EOF'
# Binance API Credentials (read-only for paper trading)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Optional: Database URL (defaults to SQLite)
# DATABASE_URL=sqlite:///data/trading_bot.db
EOF
```

## Quick Start Commands

### 1. Test Connection First
```bash
# Test that Binance connection works
python -c "
from core.data_providers.binance_data_provider import BinanceDataProvider
provider = BinanceDataProvider()
price = provider.get_current_price('BTCUSDT')
print(f'✅ Connection successful! BTC Price: \${price:,.2f}')
"
```

### 2. Start Paper Trading (Safest Options)

#### Option A: Conservative Adaptive Strategy
```bash
python run_live_trading.py adaptive \
    --symbol BTCUSDT \
    --paper-trading \
    --balance 10000 \
    --max-position 0.05 \
    --check-interval 300
```

#### Option B: Enhanced Strategy with Small Positions
```bash
python run_live_trading.py enhanced \
    --symbol BTCUSDT \
    --paper-trading \
    --balance 10000 \
    --max-position 0.02 \
    --risk-per-trade 0.01
```

#### Option C: ML Strategy with Sentiment (if models are trained)
```bash
python run_live_trading.py ml_with_sentiment \
    --symbol BTCUSDT \
    --paper-trading \
    --use-sentiment \
    --balance 10000 \
    --max-position 0.05
```

## Monitoring Your Paper Trading

### Real-time Monitoring
While the bot is running, it will show:
- Current positions
- Recent trades
- Account balance
- Performance metrics

### Database Inspection
In another terminal, monitor your trades:
```bash
# View current status
python scripts/inspect_database.py

# Watch live (refresh every 30 seconds)
watch -n 30 python scripts/inspect_database.py
```

### Log Files
Check the daily log file:
```bash
tail -f live_trading_$(date +%Y%m%d).log
```

## Safety Features in Paper Trading Mode

1. **No Real Orders**: All trades are simulated
2. **Database Logging**: All activity is logged for analysis
3. **Risk Limits**: Position size and drawdown limits still apply
4. **Stop on Errors**: Bot stops if critical errors occur

## Recommended First Run Settings

For your first paper trading session:
- **Strategy**: `adaptive` (most conservative)
- **Symbol**: `BTCUSDT` (most liquid)
- **Balance**: `$10,000` (reasonable for testing)
- **Max Position**: `5%` (conservative)
- **Check Interval**: `300` seconds (5 minutes, reduces API calls)
- **Duration**: Run for at least 24 hours to see results

## Stop Trading

To stop the bot safely:
1. Press `Ctrl+C` in the terminal
2. The bot will close all positions and save final statistics
3. Check the database for results:
   ```bash
   python scripts/inspect_database.py
   ```

## Common Issues

### "Binance API credentials not found"
- Make sure `.env` file exists and contains valid credentials
- Ensure no extra spaces in the API keys

### "No market data received"
- Check internet connection
- Verify API keys are valid
- Check if Binance is accessible from your location

### Rate Limiting
- Increase `--check-interval` to 300 or more
- The bot automatically handles rate limits with retries

## Next Steps

After successful paper trading:
1. Analyze results in the database
2. Adjust strategy parameters based on performance
3. Try different strategies and compare results
4. Build monitoring dashboard (Week 2 of go-live plan)
5. Only consider live trading after consistent profitable paper trading

## Example Full Session

```bash
# 1. Check API connection
python -c "from core.data_providers.binance_data_provider import BinanceDataProvider; print('API OK')"

# 2. Start paper trading
python run_live_trading.py adaptive --symbol BTCUSDT --paper-trading

# 3. In another terminal, monitor progress
python scripts/inspect_database.py

# 4. Stop with Ctrl+C when ready
```

Remember: Paper trading is risk-free but uses real market data, making it perfect for testing strategies! 