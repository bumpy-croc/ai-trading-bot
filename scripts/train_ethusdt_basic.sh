#!/bin/bash
# Train ETHUSDT basic (price-only) model
# This enables ML Basic and ML Adaptive strategies to trade ETHUSDT
# Expected duration: 30-60 minutes depending on hardware

set -e

echo "🚀 Training ETHUSDT basic model..."
echo "Expected impact: Enable 2-symbol portfolio diversification"
echo ""

# Ensure dependencies are installed
if ! python3 -c "import tensorflow" 2>/dev/null; then
    echo "❌ TensorFlow not installed. Installing dependencies..."
    pip install tensorflow pandas numpy scikit-learn ta
fi

# Train ETHUSDT basic model with optimized parameters
atb train model ETHUSDT \
    --force-price-only \
    --start-date 2020-01-01 \
    --end-date 2025-11-21 \
    --timeframe 1h \
    --epochs 50 \
    --batch-size 64 \
    --sequence-length 120 \
    --skip-plots \
    --skip-robustness

echo "✅ ETHUSDT basic model training complete!"
echo "Model saved to: src/ml/models/ETHUSDT/basic/latest/"
echo ""
echo "Next steps:"
echo "1. Validate model: atb backtest ml_basic --symbol ETHUSDT --days 30"
echo "2. Compare with BTCUSDT: atb backtest ml_basic --symbol BTCUSDT --days 30"
echo "3. Test ensemble strategy with both symbols"
