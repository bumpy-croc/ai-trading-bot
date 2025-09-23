# Offline Cache Pre-loading Guide

This guide explains how to pre-load the Binance data cache for offline backtesting environments where internet access is restricted or unavailable.

## Problem Statement

When running the trading bot in environments without internet access (e.g., air-gapped systems, restricted networks), the Binance data provider automatically falls back to an "offline stub" that returns empty data. This causes:

- Cache misses for all data requests
- Backtests exiting early with zero trades
- Missing `annualized_return` fields in results
- Inability to perform historical analysis

## Solution Overview

The solution involves **pre-loading the cache** with historical Binance data in an environment with internet access, then copying the cache to the offline environment. The cached data provider will automatically use this pre-loaded data when the Binance API is unavailable.

## Quick Start

### 1. Pre-load Cache (Online Environment)

Run this command in an environment with internet access:

```bash
# Pre-load 10 years of data for top cryptocurrencies
atb data preload-offline

# Or customize the symbols, timeframes, and duration
atb data preload-offline \
    --symbols BTCUSDT ETHUSDT BNBUSDT \
    --timeframes 1h 4h 1d \
    --years-back 5 \
    --test-offline
```

### 2. Copy Cache to Offline Environment

Copy the cache directory to your offline environment:

```bash
# Source environment (with internet)
tar -czf trading_cache.tar.gz cache/

# Transfer to offline environment and extract
tar -xzf trading_cache.tar.gz
```

### 3. Run Backtests Normally

In the offline environment, run backtests as usual:

```bash
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 90
```

The system will automatically use cached data when Binance API is unavailable.

## Detailed Usage

### Pre-loading Command Options

The `atb data preload-offline` command supports the following options:

| Option | Default | Description |
|--------|---------|-------------|
| `--symbols` | Top 10 coins | Trading pairs to download |
| `--timeframes` | `1h 4h 1d` | Data timeframes to cache |
| `--years-back` | `10` | Number of years to download |
| `--cache-dir` | Auto-detected | Cache directory override |
| `--force-refresh` | `false` | Force refresh existing cache |
| `--test-offline` | `false` | Test offline access after pre-loading |

### Default Symbols

The default symbol list includes major cryptocurrencies:
- BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, SOLUSDT
- XRPUSDT, DOTUSDT, LINKUSDT, LTCUSDT, AVAXUSDT

### Cache Structure

The cache uses year-based storage with hashed filenames:

```
cache/market_data/
├── 9274aafe...6fb.pkl  # BTCUSDT_1h_2023
├── f51c07ae...619.pkl  # BTCUSDT_1h_2024
├── 6a1d28e2...e2f.pkl  # BTCUSDT_1h_2025
└── ...
```

Each file contains one year of OHLCV data for a specific symbol/timeframe combination.

## Advanced Usage

### Custom Symbol Lists

Pre-load data for specific trading pairs:

```bash
atb data preload-offline \
    --symbols BTCUSDT ETHUSDT SOLUSDT MATICUSDT \
    --timeframes 1h 4h \
    --years-back 3
```

### Incremental Updates

To update cache with recent data:

```bash
# Force refresh current year data
atb data preload-offline --force-refresh --years-back 1
```

### Cache Management

Use the cache manager to inspect and maintain the cache:

```bash
# Show cache information
atb data cache-manager info

# List all cache files
atb data cache-manager list

# Clear old cache files
atb data cache-manager clear-old --hours 720  # 30 days
```

### Verification

Test that offline mode works correctly:

```bash
# Test offline access after pre-loading
atb data preload-offline --test-offline

# Or run a quick backtest
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 7
```

## Production Deployment

### 1. Automated Pre-loading

Set up a scheduled job to keep cache updated:

```bash
#!/bin/bash
# update_cache.sh
cd /path/to/trading-bot
atb data preload-offline --force-refresh --years-back 1
tar -czf cache_$(date +%Y%m%d).tar.gz cache/
```

### 2. Cache Distribution

For multiple offline environments:

```bash
# Create cache archive with metadata
atb data preload-offline --years-back 10
echo "Cache created: $(date)" > cache/metadata.txt
tar -czf trading_cache_full.tar.gz cache/

# Deploy to offline environments
scp trading_cache_full.tar.gz offline-server:/tmp/
ssh offline-server 'cd /app/trading-bot && tar -xzf /tmp/trading_cache_full.tar.gz'
```

### 3. Monitoring Cache Health

Monitor cache effectiveness:

```bash
# Check cache hit rates in logs
grep "Cache hit" logs/backtest/*.log | wc -l
grep "Cache miss" logs/backtest/*.log | wc -l

# Verify cache size and age
ls -lah cache/market_data/ | head -10
```

## Troubleshooting

### Cache Not Working

If backtests still fail with cache misses:

1. **Verify cache location**:
   ```bash
   atb data cache-manager info
   ```

2. **Check cache contents**:
   ```bash
   ls -la cache/market_data/
   ```

3. **Test offline access**:
   ```bash
   atb data preload-offline --test-offline
   ```

4. **Cache TTL issues**: The `preload-offline` command uses a 10-year TTL to treat cached data as permanently valid, avoiding expiration issues in offline environments.

### Network Issues During Pre-loading

If pre-loading fails due to network issues:

1. **Check Binance connectivity**:
   ```bash
   curl -I https://api.binance.com/api/v3/ping
   ```

2. **Use existing cache command**:
   ```bash
   atb data prefill-cache --symbols BTCUSDT --timeframes 1h --years 3
   ```

3. **Manual data download**:
   ```bash
   atb data download BTCUSDT --timeframe 1h --start_date 2023-01-01
   ```

### Performance Issues

For large cache sizes:

1. **Reduce scope**:
   ```bash
   # Fewer symbols or shorter time periods
   atb data preload-offline --symbols BTCUSDT ETHUSDT --years-back 3
   ```

2. **Compress cache**:
   ```bash
   tar -czf cache.tar.gz cache/
   # Transfer compressed archive
   ```

3. **Selective caching**:
   ```bash
   # Only cache frequently used timeframes
   atb data preload-offline --timeframes 1h 1d
   ```

## Technical Details

### Cache Key Generation

Cache keys are generated using SHA-256 hashing:

```python
request_str = f"{symbol}_{timeframe}_{year}"
cache_key = hashlib.sha256(request_str.encode()).hexdigest()
```

### Offline Detection

The system automatically detects offline mode when:
- Binance API connection fails
- Network proxy errors occur
- SSL/TLS handshake failures happen

### Cache Validation

Cache files are considered valid if:
- File exists and is readable
- For historical years (< current year): Always valid
- For current year: Valid if within TTL
- **For offline preloading**: Uses extended TTL (10 years) to treat all cached data as permanently valid

### Data Format

Cached data is stored as pickled pandas DataFrames with:
- DatetimeIndex (timezone-aware)
- Columns: `open`, `high`, `low`, `close`, `volume`
- Sorted chronologically

## Best Practices

1. **Regular Updates**: Update cache monthly in production environments
2. **Monitoring**: Monitor cache hit rates and backtest success rates
3. **Backup**: Keep backup copies of cache for disaster recovery
4. **Testing**: Always test offline functionality after cache updates
5. **Documentation**: Document cache update procedures for your team

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
name: Update Trading Cache
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM

jobs:
  update-cache:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Update cache
        run: atb data preload-offline --years-back 2
      - name: Upload cache artifact
        uses: actions/upload-artifact@v3
        with:
          name: trading-cache
          path: cache/
```

This ensures your offline environments always have up-to-date market data for backtesting.