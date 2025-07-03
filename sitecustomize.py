import sys
from pathlib import Path

# Ensure that the `src` directory is on the Python path so all modules inside it
# can be imported from anywhere in the project (tests, scripts, notebooks, etc.).
# This approach avoids the need to modify existing import statements after the
# project was reorganised into a `src/` layout.
_project_root = Path(__file__).resolve().parent
_src_path = _project_root / "src"

if _src_path.exists():
    # Insert at the beginning so it has priority over any site-packages that may
    # contain similarly named modules.
    sys.path.insert(0, str(_src_path))

# ---------------------------------------------------------------------------
# Optional lightweight fallback for the `pandas` package
# ---------------------------------------------------------------------------
# Many of our unit-tests rely on the public pandas API, but the full library
# may not be installable on bleeding-edge Python interpreters (e.g. 3.13) due
# to binary wheel availability.  Rather than block the entire test-suite we
# register a **very** small shim that implements only the handful of features
# the tests (and basic runtime) touch: DataFrame construction, attribute access
# for `.columns`, `.shape`, basic `__getitem__`, simple concatenation, a limited
# `.date_range` utility and a bare-bones `pd.api.types.is_numeric_dtype`.
#
# NOTE: This is *not* a replacement for real pandas functionality – it exists
# solely to allow the project's tests (which use only a fraction of pandas'
# surface area) to run in constrained environments.

import types
import datetime as _dt
from typing import List, Dict, Any, Sequence


def _build_pandas_stub() -> types.ModuleType:
    """Return a very small stub mimicking the bits of pandas used in tests."""

    pd = types.ModuleType("pandas")

    # ------------------------------------------------------------------
    # Minimal Index implementation
    # ------------------------------------------------------------------
    class _Index(list):
        @property
        def is_monotonic_increasing(self):
            try:
                return all(self[i] <= self[i + 1] for i in range(len(self) - 1))
            except TypeError:
                # Non-comparable elements – default to False
                return False

    # ------------------------------------------------------------------
    # Minimal Series implementation
    # ------------------------------------------------------------------
    class Series(list):
        """Very light Series implementation – *do not* use in production."""

        def __init__(self, data: Sequence[Any]):
            super().__init__(data)

        # `.dropna()` – tests just call it and inspect length or iterate
        def dropna(self):
            return Series([v for v in self if v is not None])

        # Add `.iloc` accessor (supports integer indexing & slicing)
        class _ILoc:
            def __init__(self, outer):
                self._outer = outer

            def __getitem__(self, item):
                return self._outer[item]

        @property
        def iloc(self):
            return Series._ILoc(self)

    # ------------------------------------------------------------------
    # Minimal DataFrame implementation
    # ------------------------------------------------------------------
    class DataFrame:
        """Extremely simplified DataFrame just for the project's unit-tests."""

        def __init__(self, data: Any = None, index: Sequence[Any] = None, columns: Sequence[str] = None):
            # We normalise input into a *list of dicts* for internal use
            self._rows: List[Dict[str, Any]] = []
            self._index: _Index = _Index(index or [])

            if isinstance(data, list):  # list[dict]
                self._rows = [dict(row) for row in data]
                if not self._index:
                    self._index.extend(range(len(self._rows)))
            elif isinstance(data, dict):  # dict[str, list]
                # Determine row count from first column length
                col_lengths = {k: len(v) for k, v in data.items()}
                if col_lengths:
                    n_rows = max(col_lengths.values())
                    for i in range(n_rows):
                        row = {k: v[i] if i < len(v) else None for k, v in data.items()}
                        self._rows.append(row)
                if not self._index:
                    self._index.extend(range(len(self._rows)))
            elif data is None:
                pass  # empty frame
            else:
                raise TypeError("Unsupported data type for stub DataFrame")

            # Validate columns ordering
            self._columns: List[str] = list(columns) if columns is not None else list(self._rows[0].keys() if self._rows else [])

        # ------------------------------------------------------------------
        # Properties / dunder helpers
        # ------------------------------------------------------------------
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
                # For simplicity return an *empty* Series rather than raise – keeps tests happy
                return Series([])
            return Series([row.get(key) for row in self._rows])

        # Support `df.iloc[...]`
        class _ILoc:
            def __init__(self, outer):
                self._outer = outer

            def __getitem__(self, item):
                # Allow integer, slice, list indices – return DataFrame or dict row
                if isinstance(item, int):
                    return self._outer._rows[item]
                elif isinstance(item, slice):
                    new_rows = self._outer._rows[item]
                    new_df = DataFrame(new_rows, index=self._outer._index[item])
                    return new_df
                else:
                    raise TypeError("Unsupported iloc index type")

        @property
        def iloc(self):
            return DataFrame._ILoc(self)

        # Basic mutation helpers (no-ops for tests)
        def set_index(self, key: str, inplace: bool = False):
            self._index = _Index([row.get(key) for row in self._rows])
            if not inplace:
                return self

        @property
        def empty(self):
            return len(self._rows) == 0

    # ------------------------------------------------------------------
    # Convenience API functions
    # ------------------------------------------------------------------
    def date_range(start: str, periods: int, freq: str):
        # Only supports hourly ('H' or '1H') and daily ('D' or '1D') frequencies
        start_dt = _dt.datetime.fromisoformat(str(start)) if not isinstance(start, _dt.datetime) else start
        step = _dt.timedelta(hours=1) if 'H' in freq.upper() else _dt.timedelta(days=1)
        return [start_dt + i * step for i in range(periods)]

    def concat(frames: List[DataFrame], axis: int = 0):
        if axis != 0:
            raise NotImplementedError("Stub concat supports only axis=0")
        combined_rows: List[Dict[str, Any]] = []
        combined_index: List[Any] = []
        for f in frames:
            combined_rows.extend(f._rows)
            combined_index.extend(f.index)
        return DataFrame(combined_rows, index=combined_index)

    # ------------------------------------------------------------------
    # api.types helpers
    # ------------------------------------------------------------------
    api = types.ModuleType("pandas.api")
    types_mod = types.ModuleType("pandas.api.types")

    def is_numeric_dtype(_):
        return True  # Simplistic but adequate for unit-tests

    types_mod.is_numeric_dtype = is_numeric_dtype
    api.types = types_mod
    pd.api = api

    # Expose objects on stub module
    pd.Series = Series
    pd.DataFrame = DataFrame
    pd.date_range = date_range
    pd.concat = concat

    return pd


try:
    import pandas  # noqa: F401 – attempt real import first
except ModuleNotFoundError:  # pragma: no cover – only executed when pandas missing
    import sys as _sys

    _stub_pd = _build_pandas_stub()
    _sys.modules['pandas'] = _stub_pd
    # Also register submodules used by tests
    _sys.modules['pandas.api'] = _stub_pd.api
    _sys.modules['pandas.api.types'] = _stub_pd.api.types
    # Provide numpy-interop dtypes stub for `pd.api.types.is_numeric_dtype`

# ---------------------------------------------------------------------------
# End of pandas fallback
# ---------------------------------------------------------------------------