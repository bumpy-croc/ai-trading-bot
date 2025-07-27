# Trading Concepts & System Architecture Cheatsheet

A quick-reference guide to help you understand the **why** behind the code changes and design decisions in this trading bot.

---

## 1. The Trading Loop (30-second overview)
1. **Sense** – Collect market, sentiment, and on-chain data.
2. **Predict** – Estimate future price direction/volatility using indicators & models.
3. **Decide** – Apply a strategy + risk rules to pick trades.
4. **Execute** – Place orders, monitor fills & slippage.
5. **Evaluate** – Track P&L and metrics.
6. **Learn** – Retrain models, update parameters.

Think of it like **self-driving a car**: sensors (market data) → brain (model) → controls (orders) → feedback loop.

---

## 2. Prediction Engine

### 2.1 Raw Inputs
- **Price & volume candles** (open, high, low, close, volume).
- **Order-book snapshots** (depth, spread, liquidity).
- **Sentiment feeds** (Twitter, Reddit, news APIs).
- **Macro / on-chain metrics** (inflation, hash-rate, active wallets).

### 2.2 Technical Indicators (the "thermometers" of price action)
| Indicator | What it measures | Mental analogy |
|-----------|-----------------|----------------|
| SMA / EMA | Average price over N periods | Car’s speed averaged over last N seconds |
| RSI | Momentum & overbought/oversold | How "tired" the trend is |
| MACD | Trend vs. momentum crossover | Two moving averages racing each other |
| Bollinger Bands | Volatility envelope | Elastic band stretching & snapping back |
| ATR | Average true range (volatility) | Road bumpiness |
| OBV | Volume-weighted momentum | Crowd size pushing price |

*(The codebase implements many of these in* `src/indicators/technical.py`.)*

### 2.3 Machine / Deep Learning
- **Price models** – LSTM, CNN, or Transformer predicting next return.
- **Sentiment models** – Classify text → bullish/bearish score.
- **Feature engineering** – Combine indicators, lagged returns, sentiment momentum.
- **Metrics** – MSE for regression, accuracy/F1 for classification, **Information Ratio** & **Sharpe** for economic value.

### 2.4 Ensemble & Decision Logic
Blend multiple models/indicators (majority vote, weighted average) to reduce noise and improve robustness.

---

## 3. Risk Management (the "seatbelt")

| Tool | Purpose | Key metric |
|------|---------|-----------|
| Position sizing | Limit capital per trade | % of equity or Kelly fraction |
| Stop-loss | Cap single-trade loss | Distance in ATR or % |
| Take-profit | Lock gains | Risk-reward ≥ 1:2 |
| Max drawdown guard | Halt trading after X% equity drop | Peak-to-trough drawdown |
| Volatility targeting | Scale exposure when markets are wild | Realised vs. target vol |
| Value at Risk (VaR) | Worst-case loss over horizon | VaR / Expected Shortfall |
| Leverage rules | Avoid liquidation cascades | Max leverage × equity |

Good risk rules turn a **forecast** into a **business plan**.

---

## 4. Execution Layer & Order Types
- **Market order** – Immediate fill, possible slippage.
- **Limit order** – Price control, uncertain fill.
- **Stop order** – Triggers when price crosses level (used for stop-loss).
- **OCO** (One-Cancels-Other) – Paired TP/SL orders.
- **Iceberg / Post-Only** – Hide size or ensure maker fees.

Infrastructure matters: latency, retries, and exchange API quirks drive real P&L.

---

## 5. Portfolio & Money Management
- **Diversification** – Spread risk across symbols/timeframes.
- **Correlation analysis** – Avoid doubling exposure to same factor.
- **Rebalancing** – Periodically restore target weights.
- **Cash buffer** – Cover fees and margin calls.

---

## 6. Performance Measurement
| Metric | Why it matters |
|--------|----------------|
| Total Return & CAGR | Absolute growth |
| Sharpe Ratio | Return per unit of volatility |
| Sortino Ratio | Penalises downside only |
| Calmar Ratio | Return vs. max drawdown |
| Win Rate / Payoff Ratio | Edge & trade quality |
| Profit Factor | Gross win / gross loss |

These numbers guide optimisation and comparison between strategies.

---

## 7. Psychology & Operational Factors
- **Discipline automation** – Bots remove emotional bias.
- **Overfitting checks** – Walk-forward validation, out-of-sample testing.
- **Monitoring & Alerts** – Health checks, Telegram/SMS notifications.
- **Regulatory & tax** – KYC, reporting, wash-sale rules.

---

## 8. Quick Glossary
- **Alpha** – Edge over the market.
- **Beta** – Sensitivity to overall market moves.
- **Liquidity** – How easily you can enter/exit.
- **Slippage** – Difference between expected & executed price.
- **Drawdown** – Peak-to-trough equity decline.
- **Backtest** – Simulated historical run to gauge performance.

---

### Final Thoughts
Accurate **predictions** + solid **risk management** + reliable **execution** form the three legs of profitable trading. This guide should help you map each code module to its economic purpose and keep architecture decisions aligned with trading realities.