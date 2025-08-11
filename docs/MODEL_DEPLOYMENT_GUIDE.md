# Model Deployment Guide

## Overview

The AI Trading Bot uses a simplified, production-safe model deployment architecture where:

- **Models are trained locally or in staging environments** (not in production)
- **Only validated models are deployed to production**
- **Temporary staging uses `/tmp` directory** (automatically cleaned up)
- **No complex hot-swapping in production** (restart for model updates)

## Architecture

### Local/Development Environment
```
1. Train models locally using train_model.py
2. Validate model performance with backtesting
3. Test models thoroughly in paper trading mode
4. Package validated models for deployment
```

### Staging Environment
```
1. Deploy and test models in staging environment
2. Run extended validation and performance tests
3. Verify model compatibility with live trading engine
4. Approve models for production deployment
```

### Production Environment
```
1. Deploy only pre-validated models
2. Simple model replacement (no hot-swapping)
3. Restart trading engine with new models
4. Monitor performance and rollback if needed
```

## Model Training Workflow

### 1. Local Training
```bash
# Train a new model locally (price-only)
python scripts/train_model.py BTCUSDT --force-price-only

# Validate with backtesting
python scripts/run_backtest.py ml_basic --days 90 --no-db

# Test in paper trading mode
python scripts/run_live_trading.py ml_basic --paper-trading
```

### 2. Model Packaging
```bash
# Models are automatically saved to src/ml/ directory:
# - btcusdt_sentiment.onnx (model file)
# - btcusdt_sentiment_metadata.json (training metadata)
# - BTCUSDT_sentiment_training.png (training visualization)
```

### 3. Deployment to Production
```bash
# 1. Upload model files to production server
scp src/ml/btcusdt_sentiment.* production-server:/opt/ai-trading-bot/src/ml/

# 2. Restart trading engine with new model
ssh production-server "sudo systemctl restart ai-trading-bot"

# 3. Monitor logs for successful startup
ssh production-server "sudo journalctl -u ai-trading-bot -f"
```

## Temporary Staging Directory

The system now uses `/tmp/ai-trading-bot-staging/` for any temporary model operations:

- **Automatic cleanup**: Files are automatically cleaned up on system restart
- **No persistence**: Temporary files don't clutter the project directory
- **Security**: Isolated from the main application directory

## Benefits of This Approach

### ✅ **Simplicity**
- No complex hot-swapping logic
- Clear separation between training and production
- Easier to debug and maintain

### ✅ **Safety**
- Models are thoroughly tested before production
- No training operations in production environment
- Explicit deployment steps with human oversight

### ✅ **Performance**
- No overhead from staging directories in production
- Faster startup times
- Cleaner file system

### ✅ **Reliability**
- Predictable deployment process
- Easy rollback (just deploy previous model)
- Clear audit trail of model changes

## Migration from Old Architecture

If you have existing code that relied on the old staging directory:

1. **Model training**: Move to local/staging environments
2. **Hot-swapping**: Replace with restart-based deployment
3. **Temporary files**: Will now use `/tmp` automatically

## Best Practices

### Model Training
- Always train on sufficient historical data (6+ months)
- Include multiple market conditions in training data
- Validate on out-of-sample data
- Test in paper trading before live deployment

### Model Deployment
- Deploy during low-activity periods
- Monitor performance closely after deployment
- Keep previous model files for quick rollback
- Document model changes and performance metrics

### Production Safety
- Never train models in production
- Always test new models in staging first
- Monitor system resources and performance
- Have rollback procedures ready

## Troubleshooting

### Model Loading Issues
```bash
# Check model file exists and is readable
ls -la src/ml/btcusdt_sentiment.*

# Verify model metadata
cat src/ml/btcusdt_sentiment_metadata.json | jq '.'

# Test model loading manually
python -c "import onnxruntime; print('Model loads successfully')"
```

### Deployment Issues
```bash
# Check service status
sudo systemctl status ai-trading-bot

# View recent logs
sudo journalctl -u ai-trading-bot -n 50

# Rollback to previous model
cp src/ml/btcusdt_sentiment.onnx.backup src/ml/btcusdt_sentiment.onnx
sudo systemctl restart ai-trading-bot
```

This simplified architecture makes the system more reliable, easier to maintain, and follows production best practices. 