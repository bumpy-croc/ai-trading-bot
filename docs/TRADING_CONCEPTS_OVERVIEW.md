# Trading Concepts & System Architecture Cheat-Sheet

A short, plain-language reference for understanding the decisions made in this trading-bot codebase.

---

## 1. Core Trading Vocabulary

| Term | Plain Meaning | Analogy |
|------|---------------|---------|
| **Long / Short** | Betting price goes **up** (long) or **down** (short). | Like cheering for a team to win or lose. |
| **Market / Limit Order** | Buy/Sell immediately (**market**) or at a chosen price (**limit**). | Instant purchase vs. setting an alarm-clock price. |
| **Liquidity & Spread** | How easy it is to trade and the gap between buy/sell quotes. | Busy farmers’ market vs. remote roadside stall. |
| **Leverage** | Borrowing funds to amplify gains *and* losses. | Driving faster: you reach sooner, but crashes hurt more. |
| **Pip / Tick** | The smallest price movement. | Cents on a dollar. |
| **Drawdown** | Drop from a peak account value. | Hiking downhill before climbing higher. |
| **Volatility** | How quickly prices wiggle. | Calm lake vs. stormy sea. |

---

## 2. The Prediction Layer – Turning Data into Edge

Our bot ingests multiple data flavours; combining them tends to outperform any single source.

1. **Price Action (OHLCV)**  
   • Raw candles: Open, High, Low, Close, Volume.  
   • Feeds technical indicators & ML features.

2. **Technical Indicators** – mathematical summaries of price/volume patterns.
   - **Moving Averages (SMA, EMA)** – trend direction; like smoothing noisy ECG readings.
   - **MACD** – difference of two EMAs; highlights momentum shifts.
   - **RSI / Stochastic** – over-bought vs. over-sold meter.
   - **Bollinger Bands** – price “rails” ±2 std-dev; spot squeezes/breakouts.
   - **ATR / Std Dev** – pure volatility level.
   - **Volume-based**: OBV, VWAP – confirm strength.

3. **Sentiment Data** – crowd mood signals.
   - Source: SentiCrypt API (scores ‑1 → +1 every 2 h).  
   - Derived features: sentiment momentum, volatility, extremes, moving averages.

4. **Machine-Learning Models**
   - **Feature Engineering**: stack indicators + sentiment + macro signals.
   - **Algorithms**: Gradient Boosting, LSTM/CNN, ensembles.  
   - **ONNX exports** for fast inference inside `ml_adaptive`.
   - Goal: Predict *direction* or *probability* of price move > threshold.

5. **Meta-Signals**
   - **Correlation / Cointegration** with other assets.  
   - **Seasonality / Time-of-Day** patterns.

> Better accuracy emerges from diverse, de-correlated inputs (the “wisdom of data crowds”).

---

## 3. Strategy Archetypes in the Repo

| Strategy | Primary Edge | File |
|----------|--------------|------|
| **Trend-Following** | Ride sustained moves via moving averages | `strategies.adaptive` |
| **Mean-Reversion** | Fade extremes back to average | `strategies.enhanced` |
| **Breakout** | Enter when price escapes consolidation | `strategies.high_risk_high_reward` |
| **ML Adaptive** | Model-driven signals + dynamic risk | `strategies.ml_adaptive` |
| **ML Basic** | Simplified ML without adaptation | `strategies.ml_basic` |
| **Sentiment-Aware** | Blend price + sentiment | `strategies.ml_with_sentiment` |

Each strategy plugs into a common base class (`strategies.base`) and is orchestrated by `live.strategy_manager`.

---

## 4. Risk Management Toolkit

1. **Stop-Loss / Take-Profit** – predefined exit prices; programmatic seat-belts.
2. **Position Sizing** – risk ≤ x % of equity per trade (e.g., fixed-fractional, Kelly-fraction).
3. **Max Drawdown Guard** – pause trading if account falls ≥ y % from peak.
4. **Volatility Scaling** – trade size inversely proportional to ATR or std-dev.
5. **Risk-Reward Ratio** – ensure potential gain ≥ k × potential loss (e.g., 2:1).
6. **Value at Risk (VaR) / Expected Shortfall** – statistically probable worst-case.
7. **Diversification & Correlation Control** – avoid stacking highly correlated bets.

The bot’s `risk.risk_manager` enforces these rules before orders reach the exchange interface.

---

## 5. Performance & Health Metrics

| Metric | Why It Matters |
|--------|---------------|
| **Win Rate & Avg. R:R** | Together give expectancy. |
| **CAGR / Annualized Return** | Growth speed of equity curve. |
| **Sharpe / Sortino** | Return per unit of (downside) volatility. |
| **Max Drawdown** | Pain tolerance requirement. |
| **Equity Curve Stability** | Visual sanity check—smooth beats jagged. |
| **Latency & Slippage** | Execution quality; monitored in `monitoring.dashboard`. |

---

## 6. Other Critical Ingredients

1. **Backtesting** – Validate logic on historical data (`src.backtesting`).
2. **Paper-Trading** – Live-data dry-runs; surface hidden bugs.
3. **Execution Layer** – Manages order routing, retries, rate limits (`data_providers.exchange_interface`).
4. **Monitoring & Alerts** – Health checks, Telegram pings on anomalies (`utils.telegram_alert`).
5. **Database Logging** – Persistent trade & balance history for audits (`src.database`).
6. **Security & Secrets** – API keys via env providers; least-privilege.
7. **Deployment** – Docker-first; Railway cloud config in `docs/RAILWAY_*` guides.
8. **Psychology** – Code can’t fix fear/greed, but throttles & rules help.

---

## 7. Architect’s Quick-Reference

- **Data Ingestion → Feature Engineering → Prediction → Decision → Execution → Risk Check → Logging** – the pipeline.
- Separation of concerns in code mirrors this flow; each module swaps without ripple effects.
- Emphasize **testability**: unit tests for indicators, integration tests for account sync, backtest regression for strategies.
- **1-hour candles** are the default resolution for ML strategies (see project memory).

---

### Checklist Before Shipping a Change

1. Unit & integration tests pass: `python tests/run_tests.py all -q`.
2. Backtest new logic on ≥ 2 yrs data, compare Sharpe & Max DD.
3. Verify risk caps enforced by `risk_manager`.
4. Update docs if behaviour or parameters move.

---

*Happy building & safe trading!*