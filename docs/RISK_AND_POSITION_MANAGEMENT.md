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

## Per-Strategy Overrides
Strategies can override risk behavior by implementing `get_risk_overrides()`:

```python
class MyStrategy(BaseStrategy):
    def get_risk_overrides(self):
        return {
            'position_sizer': 'confidence_weighted',  # 'fixed_fraction' | 'confidence_weighted' | 'atr_risk'
            'base_fraction': 0.02,                    # 2% base allocation
            'min_fraction': 0.005,                    # 0.5% min allocation
            'max_fraction': 0.10,                     # 10% max per position
            'confidence_key': 'prediction_confidence',
            'stop_loss_pct': 0.02,                    # 2% SL; omit to use ATR-based SL
            'take_profit_pct': 0.04,                  # 4% TP; omit to use defaults
        }
```

- If `position_sizer` is `'confidence_weighted'`, the selected `confidence_key` is read from the indicators/columns for the current index; allocation scales with confidence in [0, 1].
- If `'atr_risk'` is chosen, size is derived from legacy ATR risk sizing and converted to a balance fraction.

## Using The Layer
- Live engine (`LiveTradingEngine`) and backtester (`Backtester`) both:
  - Ask strategy for overrides via `get_risk_overrides()`
  - Call `RiskManager.calculate_position_fraction(...)` to get the position size fraction
  - Call `RiskManager.compute_sl_tp(...)` to compute SL/TP
  - Fallback to prior TP defaults if TP not provided by overrides/params

## Multiple Strategies with Different Risk Profiles
Run different strategies simultaneously, each with its own overrides. The engines pass each strategy's overrides to the shared risk manager, which enforces global limits (`max_position_size`, `max_daily_risk`) while respecting per-strategy preferences.

Example:
- Strategy A (proven): `'fixed_fraction'` at 2%, ATR-based SL, TP 3%
- Strategy B (experimental): `'confidence_weighted'` with base 0.5%, min 0.1%, max 1%, wider SL/TP

## Backward Compatibility
- Strategies that do not implement `get_risk_overrides()` continue to work with defaults.
- Legacy methods `calculate_position_size(...)` and `calculate_stop_loss(...)` remain available and are used by the new layer for `'atr_risk'` sizing or ATR-based SL.
- Existing tests for drawdown and position limits continue to rely on `RiskParameters` and `RiskManager`.

## API Summary
- Risk sizing
  - `RiskManager.calculate_position_fraction(df, index, balance, price=None, indicators=None, strategy_overrides=None, regime='normal') -> float`
- Stops and targets
  - `RiskManager.compute_sl_tp(df, index, entry_price, side='long', strategy_overrides=None) -> Tuple[Optional[float], Optional[float]]`
- Strategy overrides (optional)
  - `BaseStrategy.get_risk_overrides() -> Optional[dict]`

## Notes
- `daily_risk_used` approximates risk as the sum of opened fraction sizes; adjust as needed for your brokerage/exchange semantics.
- For correlated exposure controls, extend `get_position_correlation_risk` to use actual correlation matrices across symbols/timeframes.
