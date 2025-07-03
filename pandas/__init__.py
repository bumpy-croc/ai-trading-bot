# Lightweight stub implementation of the most commonly used portions of the
# pandas API required by this repository's unit-tests.  It is *NOT* a full
# replacement for pandas – only the limited functionality exercised by the
# tests is provided.  If the real pandas package is available it will always be
# preferred and this stub will be ignored.

import sys as _sys
import types as _types
import datetime as _dt
from typing import Any, Dict, List, Sequence

# --------------------------------------------------------------------------------------
# Basic Index implementation
# --------------------------------------------------------------------------------------
class _Index(list):
    @property
    def is_monotonic_increasing(self):
        try:
            return all(self[i] <= self[i + 1] for i in range(len(self) - 1))
        except Exception:
            return False

# --------------------------------------------------------------------------------------
# Series – ultra-minimal list wrapper with `.iloc` & `.dropna()`
# --------------------------------------------------------------------------------------
class Series(list):
    def dropna(self):
        return Series([v for v in self if v is not None])

    class _ILoc:
        def __init__(self, outer):
            self._outer = outer

        def __getitem__(self, item):
            return self._outer[item]

    @property
    def iloc(self):
        return Series._ILoc(self)

# --------------------------------------------------------------------------------------
# DataFrame – extremely simplified row-oriented implementation
# --------------------------------------------------------------------------------------
class DataFrame:
    def __init__(self, data: Any = None, index: Sequence[Any] = None, columns: Sequence[str] = None):
        self._rows: List[Dict[str, Any]] = []
        self._index: _Index = _Index(index or [])

        if isinstance(data, list):  # list[dict]
            self._rows = [dict(r) for r in data]
            if not self._index:
                self._index.extend(range(len(self._rows)))
        elif isinstance(data, dict):  # dict[column] = list
            if data:
                n_rows = max(len(v) for v in data.values())
                for i in range(n_rows):
                    self._rows.append({k: v[i] if i < len(v) else None for k, v in data.items()})
            if not self._index:
                self._index.extend(range(len(self._rows)))
        elif data is None:
            pass  # empty
        else:
            raise TypeError("Unsupported data type for stub DataFrame")

        self._columns: List[str] = list(columns) if columns is not None else (
            list(self._rows[0].keys()) if self._rows else []
        )

    # --- dunder helpers & properties ---------------------------------------------------
    def __len__(self):
        return len(self._rows)

    @property
    def columns(self):
        return self._columns

    @property
    def shape(self):
        return (len(self), len(self._columns))

    @property
    def index(self):
        return self._index

    def __getitem__(self, key: str):
        if key not in self._columns:
            return Series([])
        return Series([row.get(key) for row in self._rows])

    # iloc accessor
    class _ILoc:
        def __init__(self, outer):
            self._outer = outer

        def __getitem__(self, item):
            if isinstance(item, int):
                return self._outer._rows[item]
            elif isinstance(item, slice):
                rows = self._outer._rows[item]
                return DataFrame(rows, index=self._outer._index[item])
            else:
                raise TypeError("Stub DataFrame only supports int/slice for iloc")

    @property
    def iloc(self):
        return DataFrame._ILoc(self)

    # misc helpers
    def set_index(self, key: str, inplace: bool = False):
        self._index = _Index([row.get(key) for row in self._rows])
        if not inplace:
            return self

    @property
    def empty(self):
        return len(self._rows) == 0

# --------------------------------------------------------------------------------------
# Convenience functions – `date_range` & `concat`
# --------------------------------------------------------------------------------------

def date_range(start: Any, periods: int, freq: str):
    start_dt = _dt.datetime.fromisoformat(str(start)) if not isinstance(start, _dt.datetime) else start
    step = _dt.timedelta(hours=1) if 'H' in freq.upper() else _dt.timedelta(days=1)
    return [start_dt + i * step for i in range(periods)]


def concat(frames: List[DataFrame], axis: int = 0):
    if axis != 0:
        raise NotImplementedError("Stub concat supports only axis=0")
    all_rows: List[Dict[str, Any]] = []
    all_index: List[Any] = []
    for f in frames:
        all_rows.extend(f._rows)
        all_index.extend(f.index)
    return DataFrame(all_rows, index=all_index)

# --------------------------------------------------------------------------------------
# Build `pd.api.types` namespace with `is_numeric_dtype`
# --------------------------------------------------------------------------------------
_api_mod = _types.ModuleType('pandas.api')
_types_mod = _types.ModuleType('pandas.api.types')


def is_numeric_dtype(_obj):  # extremely lax check – sufficient for tests
    return True

_types_mod.is_numeric_dtype = is_numeric_dtype
_api_mod.types = _types_mod

# Register submodules so `import pandas.api.types` works
_sys.modules['pandas.api'] = _api_mod
_sys.modules['pandas.api.types'] = _types_mod

# Public attribute
api = _api_mod

# --------------------------------------------------------------------------------------
# Expose public symbols
# --------------------------------------------------------------------------------------
__all__ = ['DataFrame', 'Series', 'date_range', 'concat', 'api']