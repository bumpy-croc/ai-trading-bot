"""Unit tests for the signal-quality diagnostic (G5).

The diagnostic walks a strategy's signal generator bar-by-bar and reports
decision mix, predicted-return distribution, confidence distribution, and
direction-conditional hit rates. Tests here cover the math
(``DistributionStats``, hit-rate accounting), the degenerate-signal
detector, and the dependency-injection wiring so the module can be driven
without a real data provider.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.experiments.diagnostics import (
    DEFAULT_HORIZONS,
    DiagnosticReport,
    DistributionStats,
    HitRate,
    SignalDiagnostic,
)
from src.strategies.components.signal_generator import Signal, SignalDirection

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------
# DistributionStats — summary-statistic math.
# --------------------------------------------------------------------------


class TestDistributionStats:
    def test_empty_series_returns_zeros(self) -> None:
        s = DistributionStats.from_series([])
        assert s.n == 0
        assert s.mean == 0.0
        assert s.std == 0.0
        assert s.positive_fraction == 0.0

    def test_single_value_std_is_zero(self) -> None:
        s = DistributionStats.from_series([0.42])
        assert s.n == 1
        assert s.mean == 0.42
        assert s.std == 0.0

    def test_mean_and_std_are_sample_not_population(self) -> None:
        """Sample std divides by n-1, not n — matches numpy ddof=1."""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = DistributionStats.from_series(vals)
        assert pytest.approx(s.mean, rel=1e-12) == 3.0
        # Sample std: sqrt(sum((x - mean)^2) / (n - 1)) = sqrt(10/4) ≈ 1.5811
        assert pytest.approx(s.std, rel=1e-9) == float(np.std(vals, ddof=1))

    def test_positive_fraction_handles_negatives_and_zero(self) -> None:
        # Zero is NOT counted as positive.
        s = DistributionStats.from_series([-1.0, 0.0, 1.0, 2.0])
        assert pytest.approx(s.positive_fraction, rel=1e-9) == 0.5

    def test_min_and_max_handle_all_negative(self) -> None:
        s = DistributionStats.from_series([-1.0, -5.0, -0.5])
        assert s.min == -5.0
        assert s.max == -0.5


# --------------------------------------------------------------------------
# HitRate — dataclass serialization.
# --------------------------------------------------------------------------


def test_hitrate_to_dict_round_trip() -> None:
    hr = HitRate(horizon=12, buy_samples=10, buy_accuracy=0.6, sell_samples=8, sell_accuracy=0.4)
    assert hr.to_dict() == {
        "horizon": 12,
        "buy_samples": 10,
        "buy_accuracy": 0.6,
        "sell_samples": 8,
        "sell_accuracy": 0.4,
    }


# --------------------------------------------------------------------------
# Signal-generator walk — end-to-end with a stub generator + fake data.
# --------------------------------------------------------------------------


class _StubGenerator:
    """Minimal signal generator that emits a deterministic mix of signals.

    Uses index parity as a proxy for "decision": even → BUY, odd → SELL,
    multiples of 5 → HOLD. Predicted-return is a simple function of index
    so the distribution stats exercise real variance.
    """

    sequence_length = 1

    def __init__(self, name: str = "stub"):
        self.name = name

    def generate_signal(self, df: pd.DataFrame, index: int) -> Signal:
        if index % 5 == 0:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={"generator": self.name, "predicted_return": 0.0},
            )
        pr = 0.001 if index % 2 == 0 else -0.001
        return Signal(
            direction=SignalDirection.BUY if pr > 0 else SignalDirection.SELL,
            strength=0.5,
            confidence=0.1,
            metadata={"generator": self.name, "predicted_return": pr},
        )


def _df_trending_up(n_bars: int = 300, drift_per_bar: float = 0.001) -> pd.DataFrame:
    """Deterministic upward-drift OHLCV fixture."""
    ts = pd.date_range(start="2024-01-01", periods=n_bars, freq="1h", tz=UTC)
    close = 100.0 * (1.0 + drift_per_bar) ** np.arange(n_bars)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.ones(n_bars),
        }
    )


class _StubStrategy:
    """Strategy shim the runner returns — only exposes signal_generator."""

    def __init__(self, gen: Any):
        self.signal_generator = gen


class _StubProvider:
    """Data provider returning a pinned DataFrame regardless of args."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_historical_data(self, **_kwargs: Any) -> pd.DataFrame:
        return self._df


class _StubRunner:
    """Minimal ExperimentRunner shim for diagnostic dependency injection."""

    def __init__(self, strategy: _StubStrategy, df: pd.DataFrame):
        self._strategy = strategy
        self._df = df

    def _load_strategy(
        self,
        strategy_name: str,
        factory_kwargs: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> _StubStrategy:
        return self._strategy

    def _load_provider(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> _StubProvider:
        return _StubProvider(self._df)


def _run_diag(
    gen: Any,
    df: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> DiagnosticReport:
    diag = SignalDiagnostic(runner=_StubRunner(_StubStrategy(gen), df))
    end = datetime.now(UTC)
    start = end - timedelta(days=7)
    return diag.run(
        strategy_name="stub",
        symbol="BTCUSDT",
        timeframe="1h",
        start=start,
        end=end,
        provider="mock",
        horizons=horizons,
    )


def test_decision_mix_counts_match_bar_walk() -> None:
    df = _df_trending_up(n_bars=100)
    report = _run_diag(_StubGenerator(), df)
    total = report.buy_count + report.sell_count + report.hold_count
    assert total == report.bars_evaluated
    # Multiples of 5 (excluding index 0 since we start at sequence_length=1):
    # indices 5, 10, 15, ..., 95 → 19 HOLD bars.
    assert report.hold_count == 19
    # Remaining 80 bars split BUY/SELL by parity. Even-and-not-multiple-of-5
    # = 40 BUY; odd-and-not-multiple-of-5 = 40 SELL.
    assert report.buy_count == 40
    assert report.sell_count == 40


def test_predicted_return_distribution_captured() -> None:
    df = _df_trending_up(n_bars=100)
    report = _run_diag(_StubGenerator(), df)
    # Non-HOLD bars emit ±0.001; HOLD bars emit 0.0 → distribution has
    # real variance, not a constant.
    assert report.predicted_return.n == 99  # bars 1..99 (sequence_length=1)
    assert report.predicted_return.std > 0
    assert report.predicted_return.min == pytest.approx(-0.001)
    assert report.predicted_return.max == pytest.approx(0.001)


def test_hit_rate_on_monotone_up_trend_favors_buys() -> None:
    """With deterministic upward drift, BUY signals must be 100% correct
    at every horizon and SELL signals 0%."""
    df = _df_trending_up(n_bars=200)
    report = _run_diag(_StubGenerator(), df)
    for hr in report.hit_rates:
        if hr.buy_samples > 0:
            # Strict monotone: every BUY wins.
            assert hr.buy_accuracy == pytest.approx(1.0)
        if hr.sell_samples > 0:
            assert hr.sell_accuracy == pytest.approx(0.0)


def test_hit_rate_horizons_are_respected() -> None:
    df = _df_trending_up(n_bars=200)
    report = _run_diag(_StubGenerator(), df, horizons=(1, 24))
    horizons = [hr.horizon for hr in report.hit_rates]
    assert horizons == [1, 24]


# --------------------------------------------------------------------------
# Constant-signal detection — the key G5 contract.
# --------------------------------------------------------------------------


class _ConstantSellGenerator:
    """Reproduces the broken-sentiment-model fingerprint: predicted_return
    is exactly ``-1.0`` on every bar (``prediction = 0``)."""

    sequence_length = 1
    name = "constant_sell"

    def generate_signal(self, df: pd.DataFrame, index: int) -> Signal:  # noqa: ARG002
        return Signal(
            direction=SignalDirection.SELL,
            strength=1.0,
            confidence=1.0,
            metadata={"generator": self.name, "predicted_return": -1.0},
        )


def test_constant_predicted_return_triggers_warning() -> None:
    """A constant predicted_return is the fingerprint of a feature-
    pipeline / model-shape mismatch. Diagnostic must call it out by name."""
    df = _df_trending_up(n_bars=200)
    report = _run_diag(_ConstantSellGenerator(), df)
    assert report.constant_signal_warning is not None
    assert "constant" in report.constant_signal_warning.lower()
    # Decision mix is 100% SELL — another fingerprint, reflected correctly.
    assert report.buy_count == 0
    assert report.hold_count == 0
    assert report.sell_count > 0


def test_real_signal_no_warning() -> None:
    df = _df_trending_up(n_bars=200)
    report = _run_diag(_StubGenerator(), df)
    assert report.constant_signal_warning is None


def test_small_sample_doesnt_trigger_constant_warning_prematurely() -> None:
    """With n<50, we don't have enough bars to call std≈0 a fingerprint —
    a short backtest mustn't trigger the warning on every run."""
    df = _df_trending_up(n_bars=30)
    report = _run_diag(_ConstantSellGenerator(), df)
    # Either the warning is None (we aborted early) or it fires on actual
    # constant — the contract is "n >= 50 required". With 30 bars we
    # evaluate ~29 and skip the warning.
    if report.predicted_return.n < 50:
        assert report.constant_signal_warning is None


# --------------------------------------------------------------------------
# Error/guard paths.
# --------------------------------------------------------------------------


def test_empty_dataframe_raises() -> None:
    diag = SignalDiagnostic(runner=_StubRunner(_StubStrategy(_StubGenerator()), pd.DataFrame()))
    end = datetime.now(UTC)
    start = end - timedelta(days=1)
    with pytest.raises(ValueError, match="No historical data"):
        diag.run(
            strategy_name="stub",
            symbol="BTCUSDT",
            timeframe="1h",
            start=start,
            end=end,
            provider="mock",
        )


def test_strategy_without_signal_generator_raises() -> None:
    class _NoGenStrategy:
        pass

    class _NoGenRunner:
        def _load_strategy(
            self, _name: str, factory_kwargs: dict[str, Any] | None = None  # noqa: ARG002
        ) -> _NoGenStrategy:
            return _NoGenStrategy()

        def _load_provider(self, *_args: Any, **_kwargs: Any) -> _StubProvider:
            return _StubProvider(_df_trending_up())

    diag = SignalDiagnostic(runner=_NoGenRunner())  # type: ignore[arg-type]
    end = datetime.now(UTC)
    start = end - timedelta(days=1)
    with pytest.raises(ValueError, match="no signal_generator"):
        diag.run(
            strategy_name="broken",
            symbol="BTCUSDT",
            timeframe="1h",
            start=start,
            end=end,
            provider="mock",
        )


# --------------------------------------------------------------------------
# Render paths — json + text.
# --------------------------------------------------------------------------


def test_to_dict_round_trips_all_fields() -> None:
    df = _df_trending_up(n_bars=100)
    report = _run_diag(_StubGenerator(), df)
    d = report.to_dict()
    assert d["strategy"] == "stub"
    assert d["symbol"] == "BTCUSDT"
    assert d["decisions"]["buy"] == report.buy_count
    assert d["decisions"]["sell"] == report.sell_count
    assert d["decisions"]["hold"] == report.hold_count
    assert "predicted_return" in d and d["predicted_return"]["n"] == report.predicted_return.n
    assert len(d["hit_rates"]) == len(DEFAULT_HORIZONS)


def test_render_text_includes_decision_mix_and_hit_rates() -> None:
    df = _df_trending_up(n_bars=100)
    report = _run_diag(_StubGenerator(), df)
    text = report.render_text()
    assert "Decision mix" in text
    assert "Predicted return" in text
    assert "Direction-conditional hit rate" in text
    assert "BUY" in text and "SELL" in text and "HOLD" in text


def test_render_text_includes_warning_when_present() -> None:
    df = _df_trending_up(n_bars=200)
    report = _run_diag(_ConstantSellGenerator(), df)
    text = report.render_text()
    assert "WARNING" in text
