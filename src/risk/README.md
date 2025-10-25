# Risk Management

This package defines the baseline guardrails every trade must satisfy before it reaches
an exchange. It focuses on initial sizing, stop placement, and daily risk caps that all
strategies must respect.

## Role in the trading system

1. Strategies call `RiskManager` with market data to get a position size, stop-loss, and
   take-profit template.
2. The engine checks those values against live account usage (daily loss, max concurrent
   positions, drawdown) and blocks orders that violate the limits.
3. Position-management policies read the resulting `RiskParameters` so they can apply
   optional adjustments (partial exits, trailing logic, correlation constraints).

## Key pieces

- `RiskParameters` – declarative settings for firm-wide limits (risk per trade, max
  position fraction, correlation caps, partial-exit targets, trailing thresholds).
- `RiskManager` – enforces the settings by:
  - sizing trades via ATR or fraction-based methods,
  - producing ATR-based stop-losses and default take-profit levels,
  - tracking daily risk usage and drawdown,
  - providing context (`position_sizer`, correlation config) that downstream modules
    use.

## Usage

```python
from src.risk.risk_manager import RiskManager, RiskParameters

params = RiskParameters(base_risk_per_trade=0.02, max_position_size=0.25)
rm = RiskManager(params)

size = rm.calculate_position_size(price=50_000, atr=500, balance=10_000, regime="normal")
stop = rm.calculate_stop_loss(entry_price=50_000, atr=500, side="long")
```

## When to reach for it

- You need a single source of truth for account-level limits shared by every strategy.
- You want deterministic sizing that tests, backtests, and live trading can all reuse.
- You need safe defaults that position-management policies can extend without
  duplicating risk logic.
