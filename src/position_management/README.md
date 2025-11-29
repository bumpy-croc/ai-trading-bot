# Position Management

This package hosts the policies that refine open trades after the core `RiskManager`
has approved an order. Engines plug these tools in to shape portfolio behaviour without
rewriting strategy logic.

## Role in the trading system

1. A strategy queries `src.risk.RiskManager` to size the initial order and produce a
   base stop-loss.
2. Once the trade is live, position-management modules monitor performance, account
   balance, and correlation data to adjust exposure.
3. Live trading services choose which policies to enable per venue or account, keeping
   the stack modular.

## Modules

- `correlation_engine.py` – limits correlated exposure by cutting size when two assets
  move together.
- `dynamic_risk.py` – tightens risk after drawdowns or expands risk once balance
  recovers.
- `mfe_mae_analyzer.py` / `mfe_mae_tracker.py` – capture Maximum Favorable/Adverse
  Excursion so dashboards can show how trades behaved while open.
- `partial_manager.py` – automates scale-ins and partial exits based on profit targets
  configured in `RiskParameters`.
- `time_exits.py` – enforces holding-period limits, end-of-day/weekend flat rules, and
  session-aware scheduling.
- `trailing_stops.py` – moves stops to break-even and trails profits using percentage or
  ATR distance.

## Typical usage

```python
from src.database.manager import DatabaseManager
from src.risk.risk_manager import RiskManager
from src.position_management.dynamic_risk import DynamicRiskConfig, DynamicRiskManager

database_manager = DatabaseManager()
risk = RiskManager()
dyn = DynamicRiskManager(DynamicRiskConfig(enabled=True), database_manager)

size = risk.calculate_position_size(price=2_000, atr=25, balance=50_000)
adjustments = dyn.calculate_dynamic_risk_adjustments(
    current_balance=48_000,
    peak_balance=52_000,
)

effective_size = size * adjustments.position_size_factor
```

## When to reach for it

- You already have a compliant base size from `RiskManager` but need live controls such
  as trailing stops, correlation caps, or timed exits.
- You want reusable policies that every strategy can opt into without touching their
  entry logic.
- You need telemetry (MFE/MAE, policy decisions) for dashboards or auditors.

See [docs/live_trading.md](../../docs/live_trading.md#position-management-features) for
diagrams showing how these policies plug into the live toolchain.
