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

# ⚠️ **Database Note**
#
# The trading bot now **requires PostgreSQL**.  Any previous references to a
# SQLite fallback have been removed.
#
# Optional: Database URL (defaults to the `DATABASE_URL` environment variable)
EOF
```

## Quick Start Commands

### 1. Test Connection First
```bash
# Test that Binance connection works
python -c "
from src.data_providers.binance_provider import BinanceProvider
provider = BinanceProvider()
price = provider.get_current_price('BTCUSDT')
print(f'✅ Connection successful! BTC Price: ${price:,.2f}')
"
```

### 2. Start Paper Trading (Safest Options)

#### Option A: Conservative ML Basic Strategy
```bash
atb live ml_basic \
    --symbol BTCUSDT \
    --paper-trading
```
