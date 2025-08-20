# Risk and Position Management Layer
**Combination Logic:**
Multiple adjustment factors are combined conservatively (most restrictive wins):
```python
final_position_factor = min(
    drawdown_adjustment.position_size_factor,
    performance_adjustment.position_size_factor,
    volatility_adjustment.position_size_factor
)
```

### Database Models
The following tables are implemented to support dynamic risk tracking:

**`dynamic_performance_metrics`**
- Rolling performance metrics (win rate, Sharpe ratio, drawdown)
- Volatility measurements and consecutive loss/win tracking
- Risk adjustment factors applied

**`risk_adjustments`**
- All risk parameter changes with timestamps and reasons
- Original vs adjusted values with adjustment factors
- Context information (drawdown, performance score, volatility)
- Effectiveness tracking (trades during adjustment, P&L impact)

## Default Behavior
- Position sizing uses `'fixed_fraction'` policy by default with `base_risk_per_trade`
- SL is ATR-based if no explicit `stop_loss_pct` override is provided
- TP uses `RiskParameters.default_take_profit_pct` if set, otherwise falls back to existing engine defaults (e.g., 4%)
- All sizes are clamped by `RiskParameters.max_position_size` and remaining `max_daily_risk`
- **Dynamic risk adjustments are applied automatically when enabled**