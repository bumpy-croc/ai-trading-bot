# Platform Runtime Helpers

Utilities that prepare the runtime environment for CLI commands, backtests, and
services:
- `paths.py` – discovers project roots and ensures `src/` is on `sys.path`.
- `geo.py` – geo-detection helpers for Binance routing.
- `cache.py` – cache TTL adjustments when providers run in offline mode.
- `secrets.py` – standardized secret lookup with development defaults.

These modules must be importable before heavy dependencies (NumPy, pandas)
initialise, so keep them lightweight.
