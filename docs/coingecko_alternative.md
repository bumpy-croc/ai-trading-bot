# CoinGecko Data Provider - Binance Alternative

## Overview

The **CoinGeckoProvider** is a fully-compatible alternative to the Binance data provider that works reliably from Claude Code web servers and other environments where Binance API access may be blocked.

## Why CoinGecko?

When running on Claude Code web servers or in certain geographic regions:
- ❌ Binance API returns 403 Forbidden
- ❌ Coinbase API returns 403 Forbidden
- ❌ Kraken API returns 403 Forbidden
- ✅ **CoinGecko API works perfectly!**

CoinGecko provides:
- Free tier with 30 API calls per minute
- Historical OHLCV data for 13M+ tokens
- No API key required for basic usage
- Reliable uptime and performance

## Quick Start

### Basic Usage

```python
from src.data_providers.coingecko_provider import CoinGeckoProvider
from datetime import datetime, timedelta, UTC

# Initialize provider (no credentials needed)
provider = CoinGeckoProvider()

# Get current price
price = provider.get_current_price("BTC-USD")
print(f"Current BTC price: ${price:,.2f}")

# Fetch historical data
start = datetime.now(UTC) - timedelta(days=7)
df = provider.get_historical_data(
    symbol="BTC-USD",
    timeframe="4h",
    start=start
)

print(f"Fetched {len(df)} candles")
print(df.head())
```

### Using with Backtesting

The CoinGecko provider is a **drop-in replacement** for Binance in backtesting:

```python
from src.data_providers.coingecko_provider import CoinGeckoProvider
from src.engines.backtest.backtester import Backtester
from src.strategies.ml_basic import MLBasicStrategy

# Use CoinGecko instead of Binance
provider = CoinGeckoProvider()

# Everything else works the same!
strategy = MLBasicStrategy()
backtester = Backtester(
    provider=provider,
    strategy=strategy,
    symbol="BTC-USD",
    timeframe="4h",
    initial_balance=10000
)

results = backtester.run(start_date, end_date)
```

### Supported Symbols

The provider automatically converts common symbol formats:

| Your Symbol | CoinGecko ID |
|------------|--------------|
| BTC-USD    | bitcoin      |
| BTCUSDT    | bitcoin      |
| BTC        | bitcoin      |
| ETH-USD    | ethereum     |
| ETHUSDT    | ethereum     |
| SOL-USD    | solana       |
| ADA-USD    | cardano      |
| DOGE-USD   | dogecoin     |

Add more symbols by extending `SYMBOL_MAPPING` in `coingecko_provider.py`.

## API Limits & Rate Limiting

### Free Tier (No API Key)
- **30 calls per minute** (1 call every 2 seconds)
- The provider automatically throttles requests to stay within limits
- Built-in retry logic with exponential backoff

### Pro Tier (With API Key)
```python
provider = CoinGeckoProvider(api_key="your_coingecko_api_key")
```
- Higher rate limits (depends on plan)
- Priority support
- Get API key at: https://www.coingecko.com/en/api/pricing

## Data Format

CoinGecko returns data in **identical format** to Binance:

```python
df.head()
#                              open     high      low    close        volume
# timestamp
# 2026-01-11 20:00:00+00:00  90745.0  91062.0  90601.0  90614.0  1.734140e+10
# 2026-01-12 00:00:00+00:00  90569.0  90819.0  90245.0  90819.0  1.987078e+10
```

- **Index**: DatetimeIndex (timezone-aware, UTC)
- **Columns**: open, high, low, close, volume
- **All values**: Numeric (float64)
- **No NaN values**

## Supported Timeframes

CoinGecko has different granularity than Binance:

| Timeframe | CoinGecko Granularity | Notes |
|-----------|----------------------|-------|
| 4h        | 4-hour candles       | ✅ Native support |
| 1d        | Daily candles        | ✅ Native support |
| 1h        | 4-hour candles       | ⚠️ Less granular than Binance |
| 30m       | 30-minute candles    | ⚠️ Only for recent data (1 day) |
| 1m, 5m, 15m | Not supported      | ❌ Use Binance if needed |

**Recommendation**: Use **4h** or **1d** timeframes for best compatibility.

## Environment Variables

No environment variables required for basic usage!

Optional (for Pro tier):
```bash
COINGECKO_API_KEY=your_api_key_here
```

## Troubleshooting

### Rate Limit Errors

If you see "Rate limited by CoinGecko" messages:

1. **Free tier**: Provider automatically waits 60s and retries
2. **Solution**: Reduce frequency of data fetches
3. **Upgrade**: Get a Pro API key for higher limits

### Missing Symbols

If you get "No price data for X" errors:

1. Check CoinGecko's coin list: https://api.coingecko.com/api/v3/coins/list
2. Add mapping to `SYMBOL_MAPPING` in `coingecko_provider.py`
3. Or use the exact CoinGecko coin ID as the symbol

### SSL Certificate Errors

If you see TLS/certificate errors:

```python
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
```

This is already handled in the provider - should not occur.

## Comparison: CoinGecko vs Binance

| Feature | CoinGecko | Binance |
|---------|-----------|---------|
| **Accessibility** | ✅ Works on Claude Code web | ❌ Blocked (403 Forbidden) |
| **API Key Required** | ❌ No (for basic) | ✅ Yes (for trading) |
| **Rate Limits** | 30/min (free) | 1200/min (varies) |
| **Supported Coins** | 13M+ tokens | ~600 pairs |
| **Timeframes** | 4h, 1d (best) | 1m, 5m, 15m, 1h, 4h, 1d |
| **Volume Data** | ✅ Yes | ✅ Yes |
| **Live Trading** | ❌ No | ✅ Yes |
| **Use Case** | Backtesting, Research | Live Trading + Backtesting |

## Best Practices

1. **Use 4h or 1d timeframes** for CoinGecko compatibility
2. **Cache data locally** to minimize API calls
3. **Test before production**: Run backtests to verify data quality
4. **Monitor rate limits**: Watch for 429 errors in logs
5. **Consider Pro tier**: If you need high-frequency backtesting

## Integration Examples

### Update Existing Strategy

```python
# Before (using Binance)
from src.data_providers.binance_provider import BinanceProvider
provider = BinanceProvider()

# After (using CoinGecko)
from src.data_providers.coingecko_provider import CoinGeckoProvider
provider = CoinGeckoProvider()

# Everything else stays the same!
```

### CLI Usage

Update your CLI commands to use CoinGecko:

```bash
# In cli/commands/backtest.py, add provider selection
atb backtest ml_basic --provider coingecko --symbol BTC-USD --timeframe 4h --days 30
```

## Testing

Run the CoinGecko provider tests:

```bash
# Unit tests (includes integration tests that hit real API)
pytest tests/unit/test_coingecko_provider.py -v

# Mark as integration to skip in CI
pytest tests/unit/test_coingecko_provider.py -m "not integration"
```

## Support & Resources

- **CoinGecko API Docs**: https://docs.coingecko.com/reference/introduction
- **Rate Limits**: https://www.coingecko.com/en/api/pricing
- **Coin List**: https://api.coingecko.com/api/v3/coins/list
- **Status Page**: https://status.coingecko.com/

## Future Enhancements

Potential improvements for the CoinGecko provider:

- [ ] Add support for more timeframes via aggregation
- [ ] Implement caching layer to reduce API calls
- [ ] Add support for CoinGecko Pro endpoints
- [ ] Add support for derivatives data
- [ ] Add support for DeFi protocols

## Conclusion

The CoinGecko provider is a **production-ready alternative** to Binance for backtesting and research on Claude Code web servers. It provides identical data format, automatic rate limiting, and supports all major cryptocurrencies.

**Use CoinGecko when**:
- Running on Claude Code web servers
- Binance API is blocked in your region
- You only need historical data (not live trading)
- You want to avoid API key management for research

**Use Binance when**:
- You need live trading capabilities
- You need minute-level granularity
- You need the latest tick data
- You have reliable access to Binance API
