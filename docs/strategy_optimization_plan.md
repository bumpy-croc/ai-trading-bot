# Trading Strategy Optimization Plan
**Date:** 2025-11-21
**Objective:** Improve trading strategy performance by 20% in Sharpe ratio OR 15% in total returns while maintaining max drawdown <25%

## Table of Contents
1. [Current State Analysis](#current-state-analysis)
2. [Identified Optimization Opportunities](#identified-optimization-opportunities)
3. [Detailed Action Plans](#detailed-action-plans)
4. [Experiment Matrix](#experiment-matrix)
5. [Implementation Checklist](#implementation-checklist)
6. [Validation & Testing Protocol](#validation--testing-protocol)

---

## Current State Analysis

### Existing Models - Quality Assessment

#### BTCUSDT Basic Models

**Best Model: `2025-10-30_12h_v1` (1h timeframe)**
- **Training Data:** 71,784 samples (8+ years: 2017-08-17 to 2025-10-30)
- **Performance Metrics:**
  - Train Loss: 0.0042, Test Loss: 0.0044 (excellent generalization!)
  - Train RMSE: 0.065, Test RMSE: 0.067
  - MAPE: 21,801,942 (very high - indicates normalization-based predictions)
- **Architecture:** 5 normalized features (OHLCV), sequence length: 120
- **Training:** Only 49 epochs, batch size: 256
- **Issues:**
  - ✅ Good generalization (test loss close to train loss)
  - ⚠️ Underfit - only 49 epochs suggests early stopping or insufficient training
  - ⚠️ Limited features - only basic OHLCV, no technical indicators
  - ⚠️ No evaluation on actual return prediction quality

**Other BTCUSDT Models:**
- `2025-10-27_14h_v1` (1d): 2,492 samples, 6 years data, missing evaluation metrics
- `2025-10-26_21h_v1` (1h): Similar to latest, only 38 epochs
- `2025-09-16_legacy` (1d): TERRIBLE - RMSE 29,798, MAPE 23.4%

#### BTCUSDT Sentiment Models
- `2025-09-16_legacy`: Marginal improvement over price-only (RMSE 28,650 vs 29,798)
- Sentiment degradation: Only 0.05% when removed - sentiment adds minimal value for BTC

#### ETHUSDT Sentiment Models
- `2025-09-16_legacy`: BEST PERFORMING overall
  - Test RMSE: 226.5, MAPE: 5.76% (excellent!)
  - **Sentiment Impact:** Removing sentiment causes 176% performance degradation
  - ✅ Sentiment is highly valuable for ETHUSDT

### Strategy Architecture Analysis

#### ml_basic Strategy Components

**Signal Generation (MLBasicSignalGenerator):**
```python
SHORT_ENTRY_THRESHOLD = -0.0005  # -0.05% predicted return
CONFIDENCE_MULTIPLIER = 12
```

**Position Sizing (ConfidenceWeightedSizer):**
```python
base_fraction = 0.2  # 20% base allocation
min_confidence = 0.3  # Minimum confidence threshold
```

**Risk Parameters:**
```python
base_risk_per_trade = 0.02  # 2%
stop_loss_pct = 0.02  # 2%
take_profit_pct = 0.04  # 4%
max_position_size = 0.1  # 10%
```

**Issues Identified:**
1. **Aggressive Position Sizing:** 20% base fraction is very aggressive, amplifies both gains and losses
2. **Short Entry Threshold:** -0.05% is very tight - may generate too many false short signals
3. **Fixed R:R Ratio:** 2:4 (1:2) is static - doesn't adapt to market volatility
4. **No ATR-Based Stops:** Stop loss doesn't account for asset volatility
5. **Limited Confidence Scaling:** Confidence multiplier of 12 may not be calibrated optimally

---

## Identified Optimization Opportunities

### 1. Model Training Improvements (Highest Impact)

**Priority: CRITICAL**

#### A. Increase Training Epochs
**Current:** 38-49 epochs (likely underfit)
**Proposed:**
- Test 100, 150, 200 epochs
- Implement early stopping with patience=20 epochs
- Monitor validation loss plateau

**Expected Impact:** 5-10% improvement in prediction accuracy

#### B. Expand Feature Set
**Current:** Only 5 basic features (OHLCV normalized)
**Proposed Additional Features:**
```python
Technical Indicators:
- RSI (14, 21, 28 periods)
- MACD (12, 26, 9)
- Bollinger Bands (20, 2std) + %B indicator
- ATR (14) for volatility
- ADX (14) for trend strength
- Stochastic RSI (14, 3, 3)
- OBV (On-Balance Volume)
- Volume SMA ratio (current_volume / SMA_20_volume)

Price Action:
- Price change % (1h, 4h, 1d, 1w)
- High-low range %
- Close position within range (close-low)/(high-low)
- Candlestick patterns (encoded)

Momentum:
- ROC (Rate of Change) - 10, 20 periods
- Williams %R - 14 periods
- CCI (Commodity Channel Index) - 20 periods
```

**Expected Impact:** 10-15% improvement in prediction accuracy

#### C. Sequence Length Optimization
**Current:** 120 candles (fixed)
**Proposed:** Test 60, 90, 120, 150, 180 candles

**Rationale:**
- Shorter sequences (60-90): Capture recent momentum, faster adaptation
- Longer sequences (150-180): Better trend identification, smoothing noise
- Test across timeframes: 1h may need different length than 1d

**Expected Impact:** 3-7% improvement

#### D. Training Period Optimization
**Current:** 8+ years for 1h model
**Proposed Test Matrix:**
```python
Training Periods:
- 1 year  (recent market behavior)
- 2 years (balanced recency and diversity)
- 3 years (includes major market cycle)
- 4+ years (maximum historical data)

Walk-Forward Testing:
- Train on 2017-2023, test on 2024
- Train on 2018-2023, test on 2024
- Train on 2019-2023, test on 2024
```

**Expected Impact:** 5-8% improvement by finding optimal training window

### 2. Signal Generation Tuning (High Impact)

**Priority: HIGH**

#### A. Confidence Threshold Optimization
**Current:** ml_basic uses confidence multiplier of 12
**Proposed Experiments:**

```python
Confidence Thresholds to Test:
thresholds = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.70]

For Each Threshold:
1. Backtest BTCUSDT 1h (365 days)
2. Measure:
   - Total Trades
   - Win Rate
   - Sharpe Ratio
   - Max Drawdown
   - Total Return
3. Find optimal trade-off between trade frequency and quality
```

**Expected Impact:** 8-12% improvement in Sharpe ratio

#### B. Short Entry Threshold Calibration
**Current:** -0.0005 (-0.05%)
**Issues:** May be too aggressive, generating false short signals

**Proposed:**
```python
Short Thresholds to Test:
thresholds = [-0.0003, -0.0005, -0.0007, -0.001, -0.0015]

Analysis:
- Measure short trade win rate vs long trade win rate
- If short_win_rate < long_win_rate - 10%, increase threshold
- Optimize for maximum total Sharpe ratio
```

**Expected Impact:** 5-10% improvement by reducing poor short entries

### 3. Position Sizing Optimization (High Impact)

**Priority: HIGH**

#### A. Reduce Base Position Size
**Current:** 20% base fraction (extremely aggressive)
**Proposed:**

```python
Base Fractions to Test:
fractions = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

Evaluation Criteria:
- Sharpe ratio (risk-adjusted return)
- Maximum drawdown
- Volatility of returns
- Recovery time from drawdowns

Hypothesis: Smaller positions (5-10%) will:
- Reduce max drawdown significantly
- Improve Sharpe ratio through lower volatility
- Allow more trades to be active simultaneously
```

**Expected Impact:** 15-25% improvement in Sharpe ratio, 30-40% reduction in max drawdown

#### B. Implement ATR-Based Dynamic Sizing
**Current:** Fixed position sizing regardless of volatility
**Proposed:**

```python
class ATRAdaptiveSizer(PositionSizer):
    def calculate_size(self, signal, balance, risk_amount, df, index):
        # Calculate ATR (14-period)
        atr = calculate_atr(df, period=14)
        current_atr = atr.iloc[index]
        atr_sma = atr.rolling(window=50).mean().iloc[index]

        # Volatility ratio
        vol_ratio = current_atr / atr_sma

        # Adjust position size inversely to volatility
        if vol_ratio > 1.5:  # High volatility
            size_mult = 0.6
        elif vol_ratio > 1.2:
            size_mult = 0.8
        elif vol_ratio < 0.8:  # Low volatility
            size_mult = 1.2
        elif vol_ratio < 0.6:
            size_mult = 1.4
        else:
            size_mult = 1.0

        base_size = balance * self.base_fraction
        return base_size * size_mult * signal.confidence
```

**Expected Impact:** 10-15% improvement in risk-adjusted returns

### 4. Risk Management Enhancements (Medium-High Impact)

**Priority: MEDIUM-HIGH**

#### A. Dynamic Stop Loss Based on ATR
**Current:** Fixed 2% stop loss
**Proposed:**

```python
def calculate_dynamic_stop_loss(df, index, atr_period=14, atr_multiplier=2.0):
    """
    Stop loss = entry_price ± (ATR * multiplier)

    Benefits:
    - Wider stops in volatile markets (fewer stop-outs)
    - Tighter stops in calm markets (better R:R)
    """
    atr = calculate_atr(df, period=atr_period)
    current_price = df['close'].iloc[index]
    current_atr = atr.iloc[index]

    stop_distance_pct = (current_atr * atr_multiplier) / current_price

    # Bounds: minimum 1%, maximum 4%
    return max(0.01, min(0.04, stop_distance_pct))
```

**ATR Multipliers to Test:** 1.5, 2.0, 2.5, 3.0

**Expected Impact:** 8-12% improvement in win rate and Sharpe ratio

#### B. Trailing Stop Implementation
**Current:** No trailing stop - exit only at fixed TP or SL
**Proposed:**

```python
Trailing Stop Parameters:
- Activation: When profit reaches 1.5% (75% of TP)
- Trail Distance: 1 * ATR
- Update Frequency: Every candle
- Minimum Lock-In: 0.5% profit

Test Configurations:
1. Conservative: Activate at 1%, trail at 1.5 ATR
2. Balanced: Activate at 1.5%, trail at 1 ATR
3. Aggressive: Activate at 2%, trail at 0.75 ATR
```

**Expected Impact:** 5-10% improvement in total returns by capturing extended moves

### 5. Regime-Aware Optimization (Medium Impact)

**Priority: MEDIUM**

#### A. Regime-Specific Position Sizing
**Current:** ml_basic doesn't leverage regime detector fully
**Proposed:** Use RegimeAdaptiveSizer with custom multipliers

```python
regime_multipliers = {
    "bull_low_vol": 1.4,    # Increase from 1.8 (still conservative)
    "bull_high_vol": 1.0,   # Reduce from 1.2 (volatility risk)
    "bear_low_vol": 0.3,    # Reduce from 0.4 (avoid bear market)
    "bear_high_vol": 0.1,   # Reduce from 0.2 (double risk)
    "range_low_vol": 0.7,   # Keep moderate
    "range_high_vol": 0.2,  # Reduce from 0.3 (choppy)
    "unknown": 0.4,         # Reduce from 0.5 (uncertainty)
}

Optimization Approach:
1. Backtest with default multipliers
2. Analyze regime classification accuracy
3. Adjust multipliers based on win rate per regime
4. Iterate 2-3 times
```

**Expected Impact:** 7-12% improvement in Sharpe ratio

#### B. Regime-Aware Confidence Thresholds
**Proposed:**

```python
confidence_thresholds = {
    "bull_low_vol": 0.50,   # Lower bar in favorable conditions
    "bull_high_vol": 0.55,
    "bear_low_vol": 0.65,   # Higher bar in unfavorable conditions
    "bear_high_vol": 0.70,
    "range_low_vol": 0.58,
    "range_high_vol": 0.68,
    "unknown": 0.60,
}
```

**Expected Impact:** 5-8% improvement by filtering low-quality signals in adverse regimes

### 6. Multi-Symbol Portfolio Approach (Medium Impact)

**Priority: MEDIUM**

#### Proposed Symbols
```python
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
timeframes = ["1h", "4h"]
allocation_per_symbol = 0.25  # 25% of portfolio each
```

#### Correlation Analysis
```python
1. Download 1-year price data for all symbols
2. Calculate daily returns
3. Compute correlation matrix
4. Target: portfolio with correlation < 0.7 between pairs

Expected Correlations:
- BTC/ETH: ~0.85 (high, but different use cases)
- BTC/SOL: ~0.75
- BTC/BNB: ~0.80
- ETH/SOL: ~0.70 (good diversification)
```

#### Risk Allocation
```python
# Kelly Criterion for multi-asset portfolio
def optimize_allocation(symbols, win_rates, sharpe_ratios, correlations):
    """
    Use Markowitz mean-variance optimization
    Constrained by:
    - Max 30% per asset
    - Min 15% per asset (if included)
    - Target Sharpe > 1.5
    """
    pass
```

**Expected Impact:** 10-15% improvement in Sharpe ratio through diversification

---

## Detailed Action Plans

### Phase 1: Model Retraining (Week 1-2)

#### Experiment Matrix: Feature Engineering

| Experiment | Features | Sequence Length | Epochs | Training Period | Priority |
|------------|----------|----------------|---------|-----------------|----------|
| BTCUSDT_v1 | OHLCV + RSI + MACD + BB | 120 | 100 | 2021-2024 (3yr) | HIGH |
| BTCUSDT_v2 | Full technical set (15 features) | 120 | 150 | 2021-2024 | HIGH |
| BTCUSDT_v3 | Full technical set | 90 | 150 | 2021-2024 | MEDIUM |
| BTCUSDT_v4 | Full technical set | 150 | 150 | 2021-2024 | MEDIUM |
| BTCUSDT_v5 | Full technical set | 120 | 200 | 2020-2024 (4yr) | MEDIUM |
| BTCUSDT_v6 | Full technical + sentiment | 120 | 150 | 2021-2024 | LOW |

**Execution Commands:**
```bash
# Train enhanced models
atb live-control train --symbol BTCUSDT --timeframe 1h \
  --start-date 2021-01-01 --end-date 2024-12-31 \
  --epochs 100 --sequence-length 120 \
  --features technical_enhanced \
  --auto-deploy

# Compare with current model via backtest
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h \
  --days 365 --model-version BTCUSDT_v1

# If improvement > 5%, deploy
atb live-control deploy-model --model-path BTCUSDT/basic/BTCUSDT_v1
```

### Phase 2: Signal & Position Sizing Optimization (Week 2-3)

#### Confidence Threshold Grid Search

**Script:** `scripts/optimize_confidence_threshold.py`
```python
import pandas as pd
from src.backtesting.backtester import Backtester
from src.strategies.ml_basic import create_ml_basic_strategy

thresholds = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.70]
results = []

for threshold in thresholds:
    # Create custom strategy with this threshold
    strategy = create_ml_basic_strategy(
        name=f"ml_basic_conf_{threshold}",
        confidence_threshold=threshold
    )

    # Backtest
    backtester = Backtester(
        strategy=strategy,
        symbol="BTCUSDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31"
    )

    metrics = backtester.run()
    results.append({
        'threshold': threshold,
        'sharpe': metrics['sharpe_ratio'],
        'total_return': metrics['total_return'],
        'max_drawdown': metrics['max_drawdown'],
        'win_rate': metrics['win_rate'],
        'total_trades': metrics['total_trades'],
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values('sharpe', ascending=False))
```

#### Position Size Grid Search

**Script:** `scripts/optimize_position_size.py`
```python
base_fractions = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
results = []

for fraction in base_fractions:
    strategy = create_ml_basic_strategy(
        name=f"ml_basic_size_{fraction}",
        base_fraction=fraction
    )

    metrics = run_backtest(strategy, "BTCUSDT", "1h", days=365)
    results.append({
        'base_fraction': fraction,
        'sharpe': metrics['sharpe_ratio'],
        'total_return': metrics['total_return'],
        'max_drawdown': metrics['max_drawdown'],
        'avg_position_pct': metrics['avg_position_size'] / 10000,  # Assuming 10k balance
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values('sharpe', ascending=False))

# Optimal fraction = highest Sharpe with max_dd < 0.25
optimal = results_df[(results_df['max_drawdown'] < 0.25)].sort_values('sharpe', ascending=False).iloc[0]
print(f"\nOptimal base_fraction: {optimal['base_fraction']}")
```

### Phase 3: Risk Management Enhancement (Week 3-4)

#### ATR-Based Stop Loss Implementation

**File:** `src/risk/atr_stop_loss.py`
```python
def calculate_atr_stop_loss(
    df: pd.DataFrame,
    index: int,
    direction: str,  # 'long' or 'short'
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
    min_stop_pct: float = 0.01,
    max_stop_pct: float = 0.04,
) -> float:
    """
    Calculate dynamic stop loss based on ATR

    Args:
        df: OHLCV DataFrame with ATR indicator
        index: Current candle index
        direction: 'long' or 'short'
        atr_period: ATR calculation period
        atr_multiplier: Multiplier for ATR (stop distance)
        min_stop_pct: Minimum stop loss percentage
        max_stop_pct: Maximum stop loss percentage

    Returns:
        Stop loss price
    """
    # Ensure ATR is calculated
    if 'atr' not in df.columns:
        df['atr'] = calculate_atr(df, period=atr_period)

    current_price = df['close'].iloc[index]
    current_atr = df['atr'].iloc[index]

    # Calculate stop distance as percentage
    stop_distance_pct = (current_atr * atr_multiplier) / current_price

    # Apply bounds
    stop_distance_pct = max(min_stop_pct, min(max_stop_pct, stop_distance_pct))

    # Calculate stop price
    if direction == 'long':
        stop_price = current_price * (1 - stop_distance_pct)
    else:  # short
        stop_price = current_price * (1 + stop_distance_pct)

    return stop_price
```

**Testing:**
```bash
# Compare fixed vs ATR-based stops
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 365 \
  --stop-loss-type fixed --stop-loss-pct 0.02

atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 365 \
  --stop-loss-type atr --atr-multiplier 2.0
```

### Phase 4: Comprehensive Testing & Validation (Week 4-5)

#### Multi-Configuration Backtest Suite

**Script:** `scripts/comprehensive_backtest.py`
```python
"""
Run comprehensive backtests across:
- 3 symbols (BTCUSDT, ETHUSDT, SOLUSDT)
- 2 timeframes (1h, 4h)
- 3 strategies (ml_basic, ml_adaptive, ml_sentiment)
- Multiple parameter sets

Total: 18 base configurations + optimized variants
"""

import itertools
from concurrent.futures import ProcessPoolExecutor

symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
timeframes = ["1h", "4h"]
strategies = ["ml_basic", "ml_adaptive", "ml_sentiment"]

# Optimized parameters from Phase 2-3
optimal_params = {
    'confidence_threshold': 0.56,
    'base_fraction': 0.10,
    'atr_multiplier': 2.0,
    'stop_loss_type': 'atr',
}

def run_single_backtest(config):
    symbol, timeframe, strategy_name = config
    strategy = create_strategy(strategy_name, **optimal_params)
    metrics = run_backtest(strategy, symbol, timeframe, days=365)
    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'strategy': strategy_name,
        **metrics
    }

# Run all combinations in parallel
configs = list(itertools.product(symbols, timeframes, strategies))

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_single_backtest, configs))

# Aggregate results
results_df = pd.DataFrame(results)
results_df.to_csv('backtest_results_comprehensive.csv')

# Generate summary report
print(results_df.groupby('strategy').agg({
    'sharpe_ratio': 'mean',
    'total_return': 'mean',
    'max_drawdown': 'mean',
    'win_rate': 'mean',
}))
```

---

## Experiment Matrix

### Model Training Experiments

| ID | Symbol | Timeframe | Features | Seq Len | Epochs | Training Period | Expected Improvement |
|----|--------|-----------|----------|---------|--------|----------------|---------------------|
| M1 | BTCUSDT | 1h | OHLCV + 5 tech | 120 | 100 | 3yr (2021-2024) | +5-8% |
| M2 | BTCUSDT | 1h | OHLCV + 10 tech | 120 | 150 | 3yr | +10-15% |
| M3 | BTCUSDT | 1h | OHLCV + 10 tech | 90 | 150 | 3yr | +3-5% |
| M4 | BTCUSDT | 1h | OHLCV + 10 tech | 150 | 150 | 3yr | +3-5% |
| M5 | BTCUSDT | 4h | OHLCV + 10 tech | 120 | 150 | 3yr | +10-15% |
| M6 | ETHUSDT | 1h | OHLCV + 10 tech + sentiment | 120 | 150 | 3yr | +15-20% |
| M7 | SOLUSDT | 1h | OHLCV + 10 tech | 120 | 150 | 2yr (2022-2024) | New model |

### Strategy Parameter Experiments

| ID | Parameter | Values to Test | Expected Optimal | Impact |
|----|-----------|---------------|-----------------|--------|
| S1 | Confidence Threshold | 0.50-0.70 (step 0.02) | 0.55-0.58 | High |
| S2 | Base Position Fraction | 0.05-0.20 (step 0.03) | 0.08-0.12 | Very High |
| S3 | Short Entry Threshold | -0.0003 to -0.0015 | -0.0007 | Medium |
| S4 | ATR Multiplier (stops) | 1.5-3.0 (step 0.5) | 2.0-2.5 | High |
| S5 | Min Confidence | 0.2-0.5 (step 0.1) | 0.3-0.4 | Medium |

### Regime-Aware Experiments

| ID | Regime | Current Multiplier | Proposed Range | Test Method |
|----|--------|-------------------|---------------|-------------|
| R1 | bull_low_vol | 1.8 | 1.2-1.6 | Grid search |
| R2 | bull_high_vol | 1.2 | 0.8-1.2 | Grid search |
| R3 | bear_low_vol | 0.4 | 0.2-0.4 | Grid search |
| R4 | bear_high_vol | 0.2 | 0.1-0.3 | Grid search |
| R5 | range_low_vol | 0.8 | 0.6-0.9 | Grid search |

---

## Implementation Checklist

### Pre-Implementation
- [ ] Review all existing models and document baseline performance
- [ ] Set up experiment tracking (MLflow or custom logging)
- [ ] Create data download scripts for all required symbols/timeframes
- [ ] Verify database is set up for backtest logging
- [ ] Create backup of current `latest` model symlinks

### Phase 1: Model Training (Priority 1)
- [ ] Implement enhanced feature extraction pipeline
- [ ] Add RSI, MACD, Bollinger Bands to feature set
- [ ] Add ATR, ADX, Stochastic RSI to feature set
- [ ] Add OBV and volume indicators
- [ ] Train BTCUSDT 1h model with enhanced features (100 epochs)
- [ ] Train BTCUSDT 1h model with enhanced features (150 epochs)
- [ ] Train BTCUSDT 4h model with enhanced features (150 epochs)
- [ ] Train ETHUSDT 1h model with sentiment (150 epochs)
- [ ] Train SOLUSDT 1h model with enhanced features (150 epochs)
- [ ] Validate all models with hold-out test set
- [ ] Compare new models vs existing via backtests
- [ ] Deploy best performing models

### Phase 2: Signal Optimization (Priority 1)
- [ ] Implement confidence threshold parameter in ml_basic
- [ ] Run grid search for confidence thresholds (0.50-0.70)
- [ ] Analyze trade frequency vs quality trade-off
- [ ] Run grid search for short entry thresholds
- [ ] Compare long-only vs long-short performance
- [ ] Select optimal thresholds based on Sharpe ratio
- [ ] Update strategy with optimal parameters

### Phase 3: Position Sizing (Priority 1)
- [ ] Run grid search for base position fractions (0.05-0.20)
- [ ] Analyze risk-return profile for each fraction
- [ ] Implement ATR-based position sizing
- [ ] Test ATR-based sizing vs fixed sizing
- [ ] Select optimal position sizing method
- [ ] Update ConfidenceWeightedSizer with optimal parameters

### Phase 4: Risk Management (Priority 2)
- [ ] Implement ATR-based dynamic stop loss
- [ ] Test ATR multipliers (1.5, 2.0, 2.5, 3.0)
- [ ] Implement trailing stop mechanism
- [ ] Test trailing stop activation and distance parameters
- [ ] Compare fixed vs dynamic stop loss performance
- [ ] Implement dynamic take profit based on ATR
- [ ] Select optimal risk management configuration

### Phase 5: Regime Optimization (Priority 2)
- [ ] Analyze regime classification accuracy
- [ ] Measure win rate per regime
- [ ] Optimize regime-specific position multipliers
- [ ] Implement regime-aware confidence thresholds
- [ ] Test RegimeAdaptiveSizer vs ConfidenceWeightedSizer
- [ ] Compare regime-aware vs regime-agnostic strategies

### Phase 6: Multi-Symbol Portfolio (Priority 3)
- [ ] Train models for SOLUSDT and BNBUSDT
- [ ] Calculate correlation matrix for all symbols
- [ ] Implement portfolio-level risk management
- [ ] Test equal-weight vs optimized allocation
- [ ] Run multi-symbol backtests
- [ ] Compare portfolio vs single-symbol performance

### Post-Implementation
- [ ] Run comprehensive validation across all configurations
- [ ] Document final performance metrics vs baseline
- [ ] Create strategy optimization results report
- [ ] Update production deployment configuration
- [ ] Set up monitoring for deployed strategies
- [ ] Create runbook for ongoing optimization

---

## Validation & Testing Protocol

### Success Criteria

**Primary Metrics:**
1. **Sharpe Ratio:** Improve by ≥20% (target: baseline × 1.20)
2. **Total Return:** Improve by ≥15% (target: baseline × 1.15)
3. **Max Drawdown:** Maintain ≤25% (hard constraint)

**Secondary Metrics:**
1. Win Rate: Improve by 5-10 percentage points
2. Profit Factor: Improve by ≥15%
3. Average Trade Duration: Optimize for strategy type
4. Recovery Time: Reduce by ≥20%

### Baseline Measurement

**Required Backtests (to establish baseline):**
```bash
# BTCUSDT - primary trading pair
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 365
atb backtest ml_basic --symbol BTCUSDT --timeframe 4h --days 365
atb backtest ml_adaptive --symbol BTCUSDT --timeframe 1h --days 365
atb backtest ml_sentiment --symbol BTCUSDT --timeframe 1h --days 365

# ETHUSDT - secondary pair
atb backtest ml_basic --symbol ETHUSDT --timeframe 1h --days 365
atb backtest ml_sentiment --symbol ETHUSDT --timeframe 1h --days 365

# Record all metrics to baseline_metrics.json
```

### Validation Methodology

**1. Walk-Forward Testing**
```python
# Train on rolling windows, test on out-of-sample
train_periods = [
    ('2021-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ('2022-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ('2022-06-01', '2023-12-31', '2024-01-01', '2024-12-31'),
]

for train_start, train_end, test_start, test_end in train_periods:
    model = train_model(start=train_start, end=train_end)
    metrics = backtest(model, start=test_start, end=test_end)
    record_metrics(metrics)

# Average performance across all windows
avg_sharpe = mean([m['sharpe'] for m in all_metrics])
```

**2. Monte Carlo Simulation**
```python
# Validate robustness via randomized start dates
num_simulations = 100
results = []

for i in range(num_simulations):
    # Random start date within last 2 years
    start_date = random_date(2023, 2024)
    end_date = start_date + timedelta(days=365)

    metrics = backtest(strategy, start=start_date, end=end_date)
    results.append(metrics)

# Statistical analysis
sharpe_mean = np.mean([r['sharpe'] for r in results])
sharpe_std = np.std([r['sharpe'] for r in results])
confidence_interval_95 = (sharpe_mean - 1.96*sharpe_std, sharpe_mean + 1.96*sharpe_std)

print(f"Sharpe: {sharpe_mean:.2f} ± {sharpe_std:.2f}")
print(f"95% CI: {confidence_interval_95}")
```

**3. Stress Testing**
```python
# Test performance in adverse conditions
stress_periods = [
    ('2022-05-01', '2022-07-31'),  # Terra Luna crash
    ('2022-11-01', '2022-12-31'),  # FTX collapse
    ('2020-03-01', '2020-04-30'),  # COVID crash
]

for start, end in stress_periods:
    metrics = backtest(strategy, start=start, end=end)
    print(f"{start} to {end}: Sharpe={metrics['sharpe']:.2f}, DD={metrics['max_dd']:.2%}")

# Strategy passes if:
# - Max DD < 40% in stress periods
# - No single period with Sharpe < -0.5
```

### Statistical Significance Testing

```python
# Compare new strategy vs baseline using paired t-test
from scipy.stats import ttest_rel

baseline_returns = run_baseline_backtest()  # Daily returns
optimized_returns = run_optimized_backtest()  # Daily returns

# Paired t-test (same time periods)
t_stat, p_value = ttest_rel(optimized_returns, baseline_returns)

if p_value < 0.05:
    print(f"Improvement is statistically significant (p={p_value:.4f})")
else:
    print(f"Improvement not significant (p={p_value:.4f})")

# Effect size (Cohen's d)
diff_mean = np.mean(optimized_returns - baseline_returns)
diff_std = np.std(optimized_returns - baseline_returns)
cohens_d = diff_mean / diff_std
print(f"Effect size (Cohen's d): {cohens_d:.2f}")
```

### Deployment Criteria

**Strategy approved for production if:**
1. ✅ Sharpe ratio improvement ≥20% OR total return improvement ≥15%
2. ✅ Max drawdown ≤25% in all backtests
3. ✅ P-value < 0.05 (statistically significant improvement)
4. ✅ Positive Sharpe in all stress test periods
5. ✅ Walk-forward validation shows consistent performance (Sharpe std < 0.3)
6. ✅ Monte Carlo simulation 95% CI for Sharpe > 0.5
7. ✅ Win rate ≥45% (for long-short) or ≥50% (for long-only)
8. ✅ Maximum consecutive losses ≤8

**Monitoring After Deployment:**
```python
# Real-time performance tracking
metrics_to_monitor = [
    'daily_sharpe_rolling_30d',
    'current_drawdown',
    'win_rate_rolling_50trades',
    'avg_trade_duration',
    'prediction_accuracy',
]

# Alert thresholds
alerts = {
    'daily_sharpe_rolling_30d': ('below', 0.3),
    'current_drawdown': ('above', 0.20),
    'win_rate_rolling_50trades': ('below', 0.40),
}
```

---

## Appendix: Code Snippets

### Enhanced Feature Extractor

**File:** `src/ml/training_pipeline/features_enhanced.py`

```python
import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive technical indicators to OHLCV data

    Features added:
    - RSI (14, 21, 28)
    - MACD (12, 26, 9) + histogram
    - Bollinger Bands (20, 2std) + %B
    - ATR (14)
    - ADX (14) + DI+/DI-
    - Stochastic RSI (14, 3, 3)
    - OBV
    - Volume SMA ratio
    - Price momentum (multiple periods)
    """
    df = df.copy()

    # RSI - multiple periods
    for period in [14, 21, 28]:
        df[f'rsi_{period}'] = calculate_rsi(df['close'], period)

    # MACD
    macd, signal, hist = calculate_macd(df['close'], 12, 26, 9)
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = hist

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'], 20, 2)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_pct_b'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle

    # ATR
    df['atr'] = calculate_atr(df, period=14)
    df['atr_pct'] = df['atr'] / df['close']

    # ADX
    df['adx'] = calculate_adx(df, period=14)
    df['di_plus'] = calculate_di_plus(df, period=14)
    df['di_minus'] = calculate_di_minus(df, period=14)

    # Stochastic RSI
    df['stoch_rsi'] = calculate_stochastic_rsi(df['close'], 14, 3, 3)

    # OBV
    df['obv'] = calculate_obv(df)
    df['obv_sma'] = df['obv'].rolling(window=20).mean()

    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # Price momentum
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)

    # Williams %R
    df['williams_r'] = calculate_williams_r(df, period=14)

    # CCI
    df['cci'] = calculate_cci(df, period=20)

    # Price position in range
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

    return df


def normalize_features(df: pd.DataFrame, method: str = 'minmax', window: int = 120) -> pd.DataFrame:
    """
    Normalize all features for model training

    Args:
        df: DataFrame with calculated features
        method: 'minmax' or 'zscore'
        window: Rolling window for normalization

    Returns:
        DataFrame with normalized features
    """
    df = df.copy()

    feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]

    for col in feature_cols:
        if method == 'minmax':
            rolling_min = df[col].rolling(window=window, min_periods=1).min()
            rolling_max = df[col].rolling(window=window, min_periods=1).max()
            df[f'{col}_normalized'] = (df[col] - rolling_min) / (rolling_max - rolling_min + 1e-8)

        elif method == 'zscore':
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window, min_periods=1).std()
            df[f'{col}_normalized'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

    return df
```

---

## Next Steps

1. **Immediate Actions (This Week):**
   - Download 3 years of historical data for BTCUSDT, ETHUSDT, SOLUSDT (1h and 4h)
   - Run baseline backtests to establish current performance metrics
   - Begin Phase 1: Model training with enhanced features

2. **Short Term (Weeks 2-3):**
   - Complete model retraining experiments
   - Run parameter optimization grid searches
   - Document improvement metrics for each experiment

3. **Medium Term (Weeks 4-5):**
   - Implement best-performing configurations
   - Run comprehensive validation testing
   - Prepare deployment plan

4. **Long Term (Week 6+):**
   - Deploy optimized strategies to production
   - Monitor performance with established metrics
   - Iterate based on live trading results

---

## Success Metrics Summary

| Metric | Baseline (Target) | Optimistic Target | Stretch Target |
|--------|------------------|-------------------|----------------|
| Sharpe Ratio | TBD | +20% | +30% |
| Total Return | TBD | +15% | +25% |
| Max Drawdown | TBD | <25% | <20% |
| Win Rate | TBD | +5pp | +10pp |
| Profit Factor | TBD | +15% | +25% |

**Final Deliverable:** `docs/strategy_optimization_results.md` with:
- Baseline performance metrics
- All experiment results
- Final optimized configuration
- Before/after comparison
- Deployment recommendations
- Ongoing monitoring plan

---

*Document Version: 1.0*
*Last Updated: 2025-11-21*
*Author: Claude (AI Trading Bot Optimization)*
