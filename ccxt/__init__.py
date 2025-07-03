# Minimal stub of the `ccxt` exchange library.

class Exchange:
    def fetch_ohlcv(self, *args, **kwargs):
        return []

# Provide a generic exchange class for attribute-style access: ccxt.binance()
import sys as _sys, types as _types

def _make_exchange(name):
    def _factory(*args, **kwargs):
        return Exchange()
    return _factory

for _ex in [
    'binance', 'coinbase', 'kraken', 'bitfinex', 'bitstamp'
]:
    _ex_mod = _types.ModuleType(f'ccxt.{_ex}')
    _ex_mod.__dict__.update({'Exchange': Exchange})
    setattr(_sys.modules[__name__], _ex, _make_exchange(_ex))