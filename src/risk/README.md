# Risk Management

Position sizing and basic risk controls used by strategies and engines.

## Modules
- `risk_manager.py`: `RiskParameters` and `RiskManager` for sizing, stop-loss, drawdown checks

## Usage
```python
from src.risk.risk_manager import RiskManager, RiskParameters

rm = RiskManager(RiskParameters(base_risk_per_trade=0.02))
size = rm.calculate_position_size(price=50000, atr=500, balance=10000, regime='normal')
```
