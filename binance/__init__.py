class Client:
    """Very small stub of `binance.client.Client` used only for attribute access in tests."""

    def __init__(self, *args, **kwargs):
        pass

    # Stub method used by tests (gets patched via unittest.mock)
    def get_historical_klines(self, *args, **kwargs):
        return []

    # Binance interval constants commonly used in the codebase
    KLINE_INTERVAL_1MINUTE = '1m'
    KLINE_INTERVAL_5MINUTE = '5m'
    KLINE_INTERVAL_15MINUTE = '15m'
    KLINE_INTERVAL_1HOUR = '1h'
    KLINE_INTERVAL_4HOUR = '4h'
    KLINE_INTERVAL_1DAY = '1d'

    # Alias for historical klines
    def get_klines(self, *args, **kwargs):
        return self.get_historical_klines(*args, **kwargs)

# Expose in submodule path so `from binance.client import Client` works
import types, sys as _sys
_client_mod = types.ModuleType('binance.client')
_client_mod.Client = Client
_sys.modules['binance.client'] = _client_mod