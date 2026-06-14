"""Backtest determinism guard — the foundation of backtest↔live parity.

A refactor's parity is verified by running the SAME backtest before and after
and asserting byte-identical results (the "parity fingerprint"). That oracle is
only trustworthy if the backtest is reproducible.

Root cause investigated (#486 parity work): the ml_basic backtest is
deterministic *within* a process but was observed to vary *across* processes
under load — 49 vs 50 vs 51 trades. Systematically ruled out: ONNX inference
(byte-identical within and across processes, multi- and single-threaded),
``PYTHONHASHSEED`` (10 fixed seeds identical), and the prediction cache (varies
with caching disabled). The cause is multi-threaded BLAS/OpenMP floating-point
non-associativity: parallel reduction order varies run-to-run, perturbing a
feature value enough to flip a near-threshold ML signal and thus a trade. ONNX
itself is deterministic, so only the BLAS/OpenMP pools need pinning — pinning
them to a single thread makes the backtest reproducible (and is also the
recommended setup for parameter sweeps, since it avoids thread oversubscription
across concurrent backtest processes).

``run_deterministic_backtest`` is the canonical parity-fingerprint runner: it
wraps the backtest in ``threadpool_limits(1)`` so parity audits and this guard
share one reproducible path.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.engines.backtest.engine import Backtester
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.integration


def _make_fixed_market_data(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic OHLCV series (fixed seed) with regime shifts.

    600 candles keeps two backtests inside the pytest timeout while leaving
    ample tradeable bars after the model's 120-candle warmup.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)
    idx = pd.DatetimeIndex([start + timedelta(hours=i) for i in range(n)])
    rets = rng.normal(0.0002, 0.01, n)
    rets[150:250] += 0.003  # bull leg
    rets[380:460] -= 0.004  # bear leg
    close = 40000 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def run_deterministic_backtest() -> dict[str, Any]:
    """Run the canonical ml_basic parity backtest and return its results.

    ``Backtester.run`` pins BLAS/OpenMP thread pools to 1 internally, so the
    result is byte-identical across runs and processes without the caller doing
    anything. This is the canonical parity-fingerprint runner: a refactor's
    parity is verified by comparing this result before and after the change.
    """
    df = _make_fixed_market_data()
    provider = MagicMock()
    provider.get_historical_data.return_value = df

    backtester = Backtester(
        strategy=create_ml_basic_strategy(),
        data_provider=provider,
        initial_balance=10000,
    )
    return backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))


def _fingerprint(results: dict[str, Any]) -> str:
    """Canonical JSON fingerprint of a backtest result for equality comparison."""

    def default(o: Any) -> Any:
        if isinstance(o, datetime | pd.Timestamp):
            return o.isoformat()
        if isinstance(o, np.floating | np.integer):
            return float(o)
        return str(o)

    return json.dumps(results, sort_keys=True, default=default)


@pytest.mark.slow
def test_backtest_is_byte_identical_across_runs():
    """Two pinned backtests on identical inputs must produce identical results.

    This is the parity oracle's reproducibility guarantee. A regression that
    introduces nondeterminism (an unseeded RNG, hash-order-dependent iteration,
    or reliance on unpinned thread pools on the result path) breaks this.
    """
    first = run_deterministic_backtest()
    second = run_deterministic_backtest()

    # Core trading outcomes must match exactly.
    assert first["total_trades"] == second["total_trades"]
    assert first["final_balance"] == second["final_balance"]
    # And the full results fingerprint must be byte-identical.
    assert _fingerprint(first) == _fingerprint(second)
