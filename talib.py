# Simple stub for the `talib` technical analysis library used in strategy modules.
# Only provides dummy functions returning lists / scalars so that tests can run.

import numpy as np

# Common indicator names referenced (add more if required)
def RSI(close, timeperiod=14):
    # Return array of mid-range values around 50
    return np.full_like(close, 50, dtype=float)

def ATR(high, low, close, timeperiod=14):
    return np.full_like(close, (high - low).mean() if len(high) else 1.0, dtype=float)

# Fallback for any other attribute access
class _GenericFunc:
    def __call__(self, *args, **kwargs):
        import numpy as _np
        length = len(args[0]) if args else 1
        return _np.zeros(length)

    def __getattr__(self, item):
        return _GenericFunc()

import sys as _sys
_sys.modules[__name__].__getattr__ = lambda name: _GenericFunc()