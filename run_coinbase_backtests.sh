#!/bin/bash
#
# Coinbase-Based Strategy Comparison
# 
# Since Binance is geo-restricted, use Coinbase for historical data
#

echo "🚀 Running Strategy Comparison with Coinbase Data"
echo "================================================="

SYMBOL="BTC-USD"     # Coinbase symbol format
TIMEFRAME="1h"
DAYS=1825           # 5 years
BALANCE=10000

echo "Using Coinbase as data provider (Binance geo-restricted)"
echo "Symbol: $SYMBOL"
echo "Timeframe: $TIMEFRAME"
echo "Period: $DAYS days (5 years)"
echo ""

# Function to run backtest with Coinbase
run_coinbase_backtest() {
    local strategy=$1
    echo "📊 Testing $strategy with Coinbase data..."
    python -m cli backtest $strategy \
        --symbol $SYMBOL \
        --timeframe $TIMEFRAME \
        --days $DAYS \
        --initial-balance $BALANCE \
        --provider coinbase \
        --no-db
    echo ""
}

echo "🔄 Running backtests with Coinbase data provider..."
echo ""

# Test each strategy
run_coinbase_backtest "regime_adaptive"
run_coinbase_backtest "momentum_leverage" 
run_coinbase_backtest "ml_basic"

echo "✅ Coinbase backtests completed!"
echo ""
echo "💡 Note: Use Coinbase data since Binance is geo-restricted"
echo "🔄 Next: Compare results and deploy best performing strategy"