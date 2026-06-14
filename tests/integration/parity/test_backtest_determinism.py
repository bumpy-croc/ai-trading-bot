"""Backtest determinism guard — the foundation of backtest↔live parity.

A refactor's parity is verified by running the SAME backtest before and after
and asserting byte-identical results (the "parity fingerprint"). That oracle is
only trustworthy if the backtest is reproducible.

Root cause investigated (#486 parity work): the ml_basic backtest was
deterministic *within* a process but varied *across* processes under load —
49 vs 50 vs 51 trades on identical inputs. Systematically ruled out: ONNX
inference (byte-identical within and across processes, multi- and
single-threaded), ``PYTHONHASHSEED`` (10 fixed seeds identical), and the
prediction cache (varied with caching disabled). The cause is multi-threaded
BLAS/OpenMP floating-point non-associativity: parallel reduction order varies
run-to-run, perturbing a feature value enough to flip a near-threshold ML
signal and thus a trade. ``Backtester.run`` now pins the BLAS/OpenMP pools to 1
for the run, so the result is reproducible regardless of the ambient thread
count; ONNX keeps its own (deterministic) pool, so inference stays fast.

``run_deterministic_backtest`` is the canonical parity-fingerprint runner.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
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


def _fingerprint_hash(results: dict[str, Any]) -> str:
    return hashlib.sha256(_fingerprint(results).encode()).hexdigest()


def _run_in_subprocess() -> str:
    """Run the canonical backtest in a fresh process and return its fingerprint hash.

    A HIGH ambient BLAS/OpenMP thread count is forced so the run would be prone
    to multi-threaded reduction-order variance if the engine did not pin threads
    — i.e. this exercises the actual cross-process failure mode and proves the
    engine's runtime pin overrides the ambient thread environment.
    """
    project_root = Path(__file__).resolve().parents[3]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "PYTHONPATH": f"{project_root}:{project_root / 'src'}",
    }
    proc = subprocess.run(
        [sys.executable, __file__, "--emit-fingerprint"],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=True,
    )
    # The fingerprint hash is the last non-empty stdout line.
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return lines[-1]


def test_backtest_is_byte_identical_across_processes():
    """Two separate processes (under a high ambient thread count) must produce
    an identical backtest result.

    This is the parity oracle's reproducibility guarantee, exercised against the
    real failure mode: cross-process variance from multi-threaded BLAS. With the
    engine's thread pin it is deterministic; a regression that removed the pin —
    or introduced unseeded RNG / hash-order-dependent iteration on the result
    path — would break it.
    """
    first = _run_in_subprocess()
    second = _run_in_subprocess()
    assert first == second


def test_backtest_is_byte_identical_in_process():
    """Fast in-process guard against result-path nondeterminism (unseeded RNG,
    hash-order-dependent iteration)."""
    first = run_deterministic_backtest()
    second = run_deterministic_backtest()
    assert first["total_trades"] == second["total_trades"]
    assert first["final_balance"] == second["final_balance"]
    assert _fingerprint(first) == _fingerprint(second)


if __name__ == "__main__":
    # Subprocess entrypoint: print the canonical backtest fingerprint hash so the
    # cross-process determinism test can compare independent processes.
    if "--emit-fingerprint" in sys.argv:
        print(_fingerprint_hash(run_deterministic_backtest()))
