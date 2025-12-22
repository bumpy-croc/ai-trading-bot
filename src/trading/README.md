# Trading Core

`src/trading` is a small package that keeps exchange-specific glue isolated from
strategies. Today it primarily exposes the shared `SymbolFactory`, which turns
human-friendly tickers into the formats required by each exchange adapter.

## Symbol utilities

- `symbols/factory.py` – Implements `SymbolFactory`, the single point of truth
  for mapping tickers such as `BTCUSDT`, `BTC-USD`, or `btc/usd` to whatever the
  underlying provider expects. The factory also performs light validation so CLI
  commands and engines can fail fast when a symbol is unknown.

### Example

```python
from src.trading.symbols.factory import SymbolFactory

# Normalize a user-supplied value before handing it to providers
symbol = SymbolFactory.to_exchange_symbol("btc-usd", "binance")
assert symbol == "BTCUSDT"

# Convert a Binance symbol back to a dash-separated variant for logging/UI
display = SymbolFactory.to_exchange_symbol("ETHUSDT", "coinbase")
assert display == "ETH-USD"
```

## When to use this package

- CLI commands need to accept flexible inputs (`BTC-USD`, `BTCUSDT`, etc.) while
  still calling provider methods with canonical names.
- Backtesting and live trading engines must log symbols consistently so metrics
  and dashboards can be merged across venues.
- New exchanges or quote formats should be added to `SymbolFactory` instead of
  scattering `replace("USDT", "-USD")` logic throughout the codebase.

## Related documentation

- [docs/data_pipeline.md](../../docs/data_pipeline.md) – explains how data
  providers rely on `SymbolFactory`.
- [docs/backtesting.md](../../docs/backtesting.md) and
  [docs/live_trading.md](../../docs/live_trading.md) – both describe how engines
  normalise symbols before running strategies.
