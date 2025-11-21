# Maximize Risk-Adjusted Returns via Systematic Strategy Optimization

This ExecPlan is a living document maintained in accordance with `.agents/PLANS.md`. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` are updated continuously as work proceeds.

## Purpose / Big Picture

Traders and researchers using this system seek to maximize risk-adjusted returns (Sharpe ratio, Sortino ratio, max drawdown recovery) on BTCUSDT and ETHUSDT across both backtesting and live paper trading environments. Currently, the system has basic ML strategies (`ml_basic`, `ml_adaptive`) and trained models, but their parameters and architectures have not been systematically optimized for maximum risk-adjusted performance. After this work is complete, a novice will be able to run `atb backtest <optimized_strategy> --symbol BTCUSDT --timeframe 1h --days 365` and observe measurably superior Sharpe ratios (target: >1.5), lower maximum drawdowns (target: <20%), and higher win rates (target: >55%) compared to current baselines. The improved strategies will be validated through backtesting, deployed to paper trading, and monitored over a multi-day period to confirm real-world effectiveness.

## Scope and Non-Goals

This plan focuses on improving trading strategies, ML models, risk management parameters, and position sizing policies. In scope: strategy hyperparameter tuning, ML model feature engineering and architecture improvements, risk management policy refinement, backtesting validation, and paper trading deployment. Out of scope: production live trading with real capital, database schema changes, new exchange integrations, and UI/dashboard enhancements. The plan will use existing infrastructure (PostgreSQL, ONNX models, atb CLI, backtesting engine) without modifying core engine code unless critical bugs are discovered.

## Progress

- [x] (2025-11-21 22:56Z) Created ExecPlan structure and defined scope
- [ ] Establish baseline performance for BTCUSDT using ml_basic (365-day backtest)
- [ ] Establish baseline performance for ETHUSDT using ml_basic (365-day backtest)
- [ ] Establish baseline performance for BTCUSDT using ml_adaptive (365-day backtest)
- [ ] Establish baseline performance for ETHUSDT using ml_adaptive (365-day backtest)
- [ ] Document baseline metrics: Sharpe, Sortino, max drawdown, win rate, total return
- [ ] Identify top 3 optimization opportunities from baseline analysis
- [ ] Experiment Batch 1: Risk management parameter sweep (stop loss, take profit, position sizing)
- [ ] Experiment Batch 2: ML model feature engineering (add technical indicators)
- [ ] Experiment Batch 3: ML model architecture tuning (layers, dropout, sequence length)
- [ ] Experiment Batch 4: Regime detection improvements
- [ ] Experiment Batch 5: Entry/exit signal threshold optimization
- [ ] Select best-performing variant from each batch
- [ ] Validate top strategies on out-of-sample data (recent 90 days)
- [ ] Deploy best strategy to paper trading environment
- [ ] Monitor paper trading for 3-7 days and collect live metrics
- [ ] Document final results and create recommendations for next experiments

## Surprises & Discoveries

*This section will be updated as unexpected behaviors, optimizations, or insights are discovered during implementation.*

## Decision Log

**Decision**: Use Sharpe ratio as primary optimization metric, with max drawdown and win rate as secondary constraints.
**Rationale**: Sharpe ratio captures both returns and volatility in a single metric, making it ideal for comparing strategies. Max drawdown ensures downside protection, and win rate provides psychological comfort for traders.
**Date/Author**: 2025-11-21 / Claude Code

**Decision**: Run 365-day backtests for baseline measurement rather than shorter periods.
**Rationale**: Longer backtests capture multiple market regimes (bull, bear, sideways) and provide more statistically significant results. 365 days includes roughly 8760 hourly candles, sufficient for robust evaluation.
**Date/Author**: 2025-11-21 / Claude Code

**Decision**: Use existing ml_basic and ml_adaptive strategies as starting points rather than creating new strategies from scratch.
**Rationale**: These strategies already have component-based architecture, trained models, and proven integration with the backtesting engine. Incremental improvements are faster and lower-risk than greenfield development.
**Date/Author**: 2025-11-21 / Claude Code

**Decision**: Organize experiments into batches by domain (risk, features, architecture) and run them sequentially.
**Rationale**: Domain-specific batches allow focused iteration and clearer attribution of performance changes. Sequential execution prevents resource contention and makes it easier to track which changes caused which effects.
**Date/Author**: 2025-11-21 / Claude Code

## Outcomes & Retrospective

*To be completed at major milestones and plan completion.*

## Context and Orientation

The AI trading bot repository (`/home/user/ai-trading-bot`) is a modular Python system for cryptocurrency trading. Key directories:

- `src/strategies/` - Strategy implementations (ml_basic.py, ml_adaptive.py, etc.)
- `src/ml/models/` - Trained ML models organized by symbol/type/version
- `src/prediction/` - ONNX model registry and inference engine
- `src/risk/` - Risk management policies (RiskManager, RiskParameters)
- `src/backtesting/` - Vectorized backtesting engine
- `src/optimizer/` - Parameter optimization and strategy tuning tools
- `cli/commands/` - CLI commands for backtesting, training, live trading
- `tests/` - Unit and integration tests
- `docs/execplans/` - Execution plans for major features

**Strategy Architecture**: Strategies use component composition (SignalGenerator, RiskManager, PositionSizer, RegimeDetector). The `Strategy` class coordinates these components and produces `TradingDecision` objects.

**ML Models**: Trained models are stored in `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}/` with metadata.json, model.keras, and model.onnx files. The `latest` symlink points to the currently deployed version.

**Backtesting**: The `atb backtest` command runs vectorized simulations over historical data, producing metrics like Sharpe ratio, max drawdown, total return, and trade-by-trade logs.

**Current State**: Models exist for BTCUSDT and ETHUSDT. The most recent BTCUSDT basic model (2025-10-30_12h_v1) has test RMSE of 0.0665. Baseline performance metrics for strategies are not yet established.

## Plan of Work

### Phase 1: Baseline Measurement (Milestone 1)

Measure current strategy performance to establish comparison points for optimization. Run 365-day backtests for all strategy-symbol combinations and record comprehensive metrics.

**Steps**:
1. Verify database connection and ensure PostgreSQL is running
2. Check that trained models exist for BTCUSDT and ETHUSDT (symlinks to `latest`)
3. Run baseline backtests:
   - `atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 365`
   - `atb backtest ml_basic --symbol ETHUSDT --timeframe 1h --days 365`
   - `atb backtest ml_adaptive --symbol BTCUSDT --timeframe 1h --days 365`
   - `atb backtest ml_adaptive --symbol ETHUSDT --timeframe 1h --days 365`
4. Extract and record metrics from each run:
   - Sharpe ratio (risk-adjusted return metric)
   - Sortino ratio (downside-focused risk metric)
   - Maximum drawdown percentage (worst peak-to-trough decline)
   - Win rate (percentage of profitable trades)
   - Total return percentage (cumulative profit/loss)
   - Number of trades (activity level)
   - Average trade duration (holding period)
5. Create baseline metrics table in this ExecPlan
6. Identify weaknesses: Which metric needs most improvement? Which strategy performs worst?

**Expected Duration**: 2-4 hours (backtests may take 15-30 minutes each depending on hardware)

**Validation**: After this phase, the ExecPlan will contain a table showing baseline metrics for 4 strategy-symbol combinations. The table will clearly show which areas need improvement.

### Phase 2: Optimization Opportunity Analysis (Milestone 2)

Analyze baseline results to identify the highest-leverage opportunities for improvement.

**Steps**:
1. Compare BTCUSDT vs ETHUSDT performance - which symbol is harder to trade?
2. Compare ml_basic vs ml_adaptive - which architecture has more potential?
3. Analyze failure modes:
   - Low Sharpe → Improve signal quality or reduce trading frequency
   - High drawdown → Tighten stop losses or improve regime detection
   - Low win rate → Improve entry signals or exit timing
   - Too few/many trades → Adjust signal thresholds
4. Identify top 3 optimization levers based on analysis
5. Design experiment batches targeting each lever

**Expected Duration**: 1 hour

**Validation**: Clear prioritization of experiments with rationale for each batch.

### Phase 3: Systematic Experimentation (Milestones 3-7)

Run batches of experiments, each targeting a specific domain. After each batch, select the best-performing variant and carry it forward.

#### Experiment Batch 1: Risk Management Parameters

**Hypothesis**: Current 2% stop loss and 4% take profit may not be optimal for crypto volatility.

**Experiments**:
- Variant A: Stop loss 3%, take profit 6% (wider bands)
- Variant B: Stop loss 1%, take profit 3% (tighter bands)
- Variant C: Stop loss 2.5%, take profit 5% (moderate adjustment)
- Variant D: Asymmetric - stop loss 2%, take profit 8% (high reward/risk)

**Implementation**: Edit `ml_basic.py` to accept risk parameters, run backtests with each variant, compare Sharpe ratios.

**Success Criteria**: At least one variant improves Sharpe by >10% over baseline.

#### Experiment Batch 2: ML Feature Engineering

**Hypothesis**: Adding technical indicators to price-only features will improve prediction accuracy.

**Experiments**:
- Variant A: Add RSI(14) and RSI(28) features
- Variant B: Add MACD(12,26,9) features
- Variant C: Add Bollinger Bands (20, 2) features
- Variant D: Add volume-weighted features (VWAP, volume momentum)

**Implementation**: Modify `src/ml/training_pipeline/features.py` to compute additional features, retrain models with `atb train model`, run backtests.

**Success Criteria**: Test RMSE improves by >5%, or backtest Sharpe improves by >15%.

#### Experiment Batch 3: ML Architecture Tuning

**Hypothesis**: Current CNN+LSTM architecture may be suboptimal for sequence length and layer configuration.

**Experiments**:
- Variant A: Sequence length 60 (shorter context, faster adaptation)
- Variant B: Sequence length 240 (longer context, smoother signals)
- Variant C: Add dropout layers (0.2) to reduce overfitting
- Variant D: Increase LSTM units from default to 128/256

**Implementation**: Modify training config in `atb train model` command, retrain, backtest.

**Success Criteria**: Validation loss decreases without overfitting (train/test gap <10%).

#### Experiment Batch 4: Regime Detection Improvements

**Hypothesis**: Better regime detection allows adaptive position sizing and risk management.

**Experiments**:
- Variant A: Tune regime detector thresholds (volatility bands, trend strength)
- Variant B: Add regime-specific entry/exit rules (no longs in bear regime)
- Variant C: Scale position size by regime confidence (50% size in uncertain regimes)

**Implementation**: Modify `EnhancedRegimeDetector` parameters, add regime filters to signal generation.

**Success Criteria**: Max drawdown reduces by >20% while maintaining similar returns.

#### Experiment Batch 5: Signal Threshold Optimization

**Hypothesis**: Current ML prediction confidence thresholds (0.3 minimum) may be too loose or too tight.

**Experiments**:
- Variant A: Minimum confidence 0.5 (more selective)
- Variant B: Minimum confidence 0.2 (more aggressive)
- Variant C: Dynamic thresholds based on regime (0.5 in low-vol, 0.3 in high-vol)

**Implementation**: Modify `ConfidenceWeightedSizer` min_confidence parameter.

**Success Criteria**: Win rate improves by >5 percentage points.

### Phase 4: Validation and Deployment (Milestone 8)

**Steps**:
1. Take the best-performing variant from each batch
2. Combine improvements into a single optimized strategy
3. Run out-of-sample validation on recent 90 days (not used in optimization)
4. If validation Sharpe > baseline Sharpe, proceed to deployment
5. Deploy to paper trading: `atb live <optimized_strategy> --symbol BTCUSDT --paper-trading`
6. Monitor for 3-7 days, collecting live metrics
7. Compare live performance to backtest predictions
8. Document final results in Outcomes section

**Success Criteria**:
- Out-of-sample Sharpe ratio >1.5
- Max drawdown <20%
- Win rate >55%
- Live paper trading confirms backtest results (within 20% variance)

## Concrete Steps

### Environment Setup

```bash
# Ensure database is running
docker compose up -d postgres
export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot

# Verify database connection
atb db verify

# Check models exist
ls -la src/ml/models/BTCUSDT/basic/latest
ls -la src/ml/models/ETHUSDT/basic/latest
```

### Baseline Backtesting Commands

```bash
# BTCUSDT ml_basic baseline
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 365 > baseline_btc_basic.log 2>&1

# ETHUSDT ml_basic baseline
atb backtest ml_basic --symbol ETHUSDT --timeframe 1h --days 365 > baseline_eth_basic.log 2>&1

# BTCUSDT ml_adaptive baseline
atb backtest ml_adaptive --symbol BTCUSDT --timeframe 1h --days 365 > baseline_btc_adaptive.log 2>&1

# ETHUSDT ml_adaptive baseline
atb backtest ml_adaptive --symbol ETHUSDT --timeframe 1h --days 365 > baseline_eth_adaptive.log 2>&1
```

Expected output format (example):
```
=== Backtest Results ===
Symbol: BTCUSDT
Strategy: ml_basic
Timeframe: 1h
Period: 2024-11-21 to 2025-11-21 (365 days)

Performance Metrics:
- Total Return: 45.2%
- Sharpe Ratio: 1.23
- Sortino Ratio: 1.67
- Max Drawdown: 18.5%
- Win Rate: 52.3%
- Total Trades: 87
- Avg Trade Duration: 4.2 days
```

### Model Training Commands (for later experiments)

```bash
# Retrain BTCUSDT model with new features
atb train model BTCUSDT --timeframe 1h --start-date 2020-01-01 --end-date 2024-11-21 \
  --epochs 100 --batch-size 64 --sequence-length 120 --auto-deploy

# Retrain with different sequence length
atb train model BTCUSDT --timeframe 1h --start-date 2020-01-01 --end-date 2024-11-21 \
  --epochs 100 --batch-size 64 --sequence-length 240 --auto-deploy
```

## Validation and Acceptance

After completing all phases, the following must be true:

1. **Baseline Documentation**: ExecPlan contains complete baseline metrics table
2. **Improvement Verification**: At least one strategy shows >20% Sharpe improvement over baseline
3. **Out-of-Sample Validation**: Improved strategy maintains Sharpe >1.5 on recent 90-day period
4. **Test Passing**: All unit tests pass (`python tests/run_tests.py unit`)
5. **Paper Trading**: Strategy runs successfully in paper mode for 3+ days without errors
6. **Documentation**: All experiments documented with code changes, metrics, and rationale

To verify, run:
```bash
# Tests pass
python tests/run_tests.py unit

# Paper trading works
atb live <optimized_strategy> --symbol BTCUSDT --paper-trading

# Check logs for errors
tail -f logs/trading_*.log
```

## Idempotence and Recovery

- All backtests are read-only and can be re-run safely
- Model training produces versioned artifacts; never overwrites existing models
- Failed experiments can be abandoned without affecting baseline
- Database state is not modified by backtests
- Paper trading uses isolated test balance; real funds never at risk

To recover from failures:
- Backtest failures: Check logs, verify data availability, retry
- Training failures: Check GPU memory, reduce batch size, retry
- Paper trading issues: Stop with Ctrl+C, check database state, restart

## Artifacts and Notes

This section will contain:
- Baseline metrics table (to be populated)
- Experiment results (to be populated)
- Code diffs for key changes (to be populated)
- Final performance comparison charts (to be populated)

## Baseline Metrics Table

*To be populated after Phase 1 completion:*

| Strategy | Symbol | Sharpe | Sortino | Max DD | Win Rate | Total Return | Trades |
|----------|--------|--------|---------|--------|----------|--------------|--------|
| ml_basic | BTCUSDT | TBD | TBD | TBD | TBD | TBD | TBD |
| ml_basic | ETHUSDT | TBD | TBD | TBD | TBD | TBD | TBD |
| ml_adaptive | BTCUSDT | TBD | TBD | TBD | TBD | TBD | TBD |
| ml_adaptive | ETHUSDT | TBD | TBD | TBD | TBD | TBD | TBD |

## Experiment Results

*To be populated as experiments complete:*

### Batch 1: Risk Management Parameters
- TBD

### Batch 2: ML Feature Engineering
- TBD

### Batch 3: ML Architecture Tuning
- TBD

### Batch 4: Regime Detection
- TBD

### Batch 5: Signal Thresholds
- TBD

## Interfaces and Dependencies

**Key Components**:

- `src.strategies.ml_basic.create_ml_basic_strategy()` - Factory for ml_basic strategy
  - Parameters: name, sequence_length, model_name, model_type, timeframe
  - Returns: Strategy instance with configured components

- `src.strategies.components.Strategy` - Main strategy orchestrator
  - Methods: `process_candle(df, index, balance, positions) -> TradingDecision`
  - Components: SignalGenerator, RiskManager, PositionSizer, RegimeDetector

- `src.risk.risk_manager.RiskParameters` - Risk configuration
  - Fields: base_risk_per_trade, default_take_profit_pct, max_position_size

- `src.prediction.model_registry.ModelRegistry` - Model loading
  - Methods: `load_model(symbol, model_type) -> ONNXModel`

- CLI Commands:
  - `atb backtest <strategy> --symbol <SYM> --timeframe <TF> --days <N>`
  - `atb train model <symbol> --epochs <N> --batch-size <N> --sequence-length <N>`
  - `atb live <strategy> --symbol <SYM> --paper-trading`

## Commits

Commit frequently with clear messages as the plan executes:
- After baseline measurement: `feat: establish baseline metrics for strategy optimization`
- After each experiment batch: `experiment: test risk management variants for ml_basic`
- After improvements: `feat: optimize ml_basic strategy with improved risk params`
- After validation: `docs: document strategy optimization results and next steps`

---

**Revision History**:
- 2025-11-21: Initial plan created following PLANS.md specification
