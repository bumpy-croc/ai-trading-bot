"""Regression test for regime-aware backtesting results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest
import numpy as np

from src.backtesting.engine import Backtester
from src.data_providers.data_provider import DataProvider
from src.strategies.base import BaseStrategy


class FixtureDataProvider(DataProvider):
    """Data provider that serves a deterministic OHLCV fixture."""

    def __init__(self, frame: pd.DataFrame):
        super().__init__()
        self._frame = frame

    def get_historical_data(
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None = None
    ) -> pd.DataFrame:
        return self._frame.copy()

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        return self._frame.tail(limit).copy()

    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self._frame.tail(1).copy()

    def get_current_price(self, symbol: str) -> float:
        return float(self._frame["close"].iloc[-1])


class DeterministicStrategy(BaseStrategy):
    """Simple deterministic strategy used for regression testing."""

    def __init__(self, name: str, entry_period: int, hold_period: int, size: float):
        super().__init__(name)
        self.entry_period = entry_period
        self.hold_period = hold_period
        self.size = size
        self._active_entry_index: int | None = None
        self.take_profit_pct = 0.02
        self.stop_loss_pct = 0.01
        self.entry_events: list[int] = []
        self.exit_events: list[int] = []
        self.entry_checks: list[int] = []

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = df.copy()
        self.indicator_na = enriched.isna().sum().to_dict()
        enriched["base_signal"] = (pd.Series(range(len(enriched)), index=enriched.index) % self.entry_period) == 0
        self.last_indicator_rows = len(enriched)
        return enriched

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        self.entry_checks.append(index)
        if self._active_entry_index is not None and index - self._active_entry_index >= self.hold_period:
            self._active_entry_index = None
        if self._active_entry_index is not None:
            return False
        if index == 0:
            return False
        if index % self.entry_period == 0:
            self._active_entry_index = index
            self.entry_events.append(index)
            return True
        return False

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        if self._active_entry_index is None:
            return False
        if index - self._active_entry_index >= self.hold_period:
            self._active_entry_index = None
            self.exit_events.append(index)
            return True
        return False

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        return self.size

    def calculate_stop_loss(self, df, index, price, side: str = "long") -> float:
        return float(price) * (1 - self.stop_loss_pct)

    def get_parameters(self) -> dict[str, Any]:
        return {
            "entry_period": self.entry_period,
            "hold_period": self.hold_period,
            "size": self.size,
        }


class StubStrategyManager:
    """Minimal stand-in for the live strategy manager."""

    def __init__(self, factory: Callable[[str], BaseStrategy]):
        self._factory = factory
        self.current_strategy: BaseStrategy | None = None

    def load_strategy(self, strategy_name: str) -> BaseStrategy:
        strategy = self._factory(strategy_name)
        self.current_strategy = strategy
        return strategy


class StubRegimeStrategySwitcher:
    """Deterministic strategy switcher used for regression validation."""

    def __init__(
        self,
        strategy_manager: StubStrategyManager,
        regime_config: Any | None = None,
        strategy_mapping: Any | None = None,
        switching_config: Any | None = None,
    ):
        self.strategy_manager = strategy_manager
        self.switching_config = type(
            "Cfg",
            (),
            {
                "min_regime_confidence": 0.5,
                "require_timeframe_agreement": 0.5,
                "min_regime_duration": 1,
                "switch_cooldown_minutes": 0,
                "emergency_strategy": "ml_basic",
            },
        )()
        self._regimes = [
            ("trend_up:low_vol", 0.9, 0.9),
            ("trend_down:high_vol", 0.85, 0.85),
            ("range:low_vol", 0.8, 0.8),
        ]
        self._analysis_calls = 0
        self._switches_executed = 0
        self.current_regime: str | None = None
        self.regime_start_time: datetime | None = None
        self.regime_start_candle_index: int | None = None
        self.regime_duration: int = 0
        self.last_switch_time: datetime | None = None

    def analyze_market_regime(self, price_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        idx = min(self._analysis_calls, len(self._regimes) - 1)
        label, confidence, agreement = self._regimes[idx]
        self._analysis_calls += 1
        consensus = {
            "regime_label": label,
            "confidence": confidence,
            "agreement_score": agreement,
        }
        return {
            "timeframe_regimes": {
                "1h": {
                    "regime_label": label,
                    "confidence": confidence,
                    "agreement": agreement,
                }
            },
            "consensus_regime": consensus,
            "analysis_timestamp": datetime(2022, 1, 1),
        }

    def should_switch_strategy(self, regime_analysis: dict[str, Any], current_candle_index: int | None = None) -> dict[str, Any]:
        consensus = regime_analysis["consensus_regime"]
        should_switch = self._analysis_calls >= 2 and self._switches_executed == 0
        optimal_strategy = "alternate" if should_switch else "ml_basic"
        decision = {
            "should_switch": should_switch,
            "reason": "deterministic-switch" if should_switch else "regime-stable",
            "new_regime": consensus["regime_label"],
            "optimal_strategy": optimal_strategy,
            "current_strategy": self.strategy_manager.current_strategy.name if self.strategy_manager.current_strategy else None,
            "confidence": consensus["confidence"],
            "agreement": consensus["agreement_score"],
        }
        if should_switch:
            self._switches_executed += 1
        return decision

    def execute_strategy_switch(self, decision: dict[str, Any]) -> bool:
        if decision.get("should_switch"):
            self.last_switch_time = datetime.now()
            return True
        return False


def _build_fixture_dataframe() -> pd.DataFrame:
    periods = 240
    index = pd.date_range("2022-01-01", periods=periods, freq="H")
    closes = []
    price = 100.0
    for i in range(periods):
        if i < 80:
            price += 0.6
        elif i < 160:
            price -= 0.4
        else:
            price += 0.2 if i % 2 == 0 else -0.1
        closes.append(round(price, 4))
    close_series = pd.Series(closes, index=index)
    open_series = close_series.shift(1).fillna(close_series.iloc[0])
    high_series = pd.concat([open_series, close_series], axis=1).max(axis=1) + 0.3
    low_series = pd.concat([open_series, close_series], axis=1).min(axis=1) - 0.3
    volume_values = 1_000 + (np.arange(periods) % 50) * 5
    volume = pd.Series(volume_values.astype(float), index=index)
    return pd.DataFrame(
        {
            "open": open_series.astype(float),
            "high": high_series.astype(float),
            "low": low_series.astype(float),
            "close": close_series.astype(float),
            "volume": volume,
        },
        index=index,
    )


def _sanitize_time_series(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for event in events:
        new_event = dict(event)
        timestamp = new_event.get("timestamp")
        if isinstance(timestamp, pd.Timestamp):
            new_event["timestamp"] = timestamp.isoformat()
        elif isinstance(timestamp, datetime):
            new_event["timestamp"] = timestamp.isoformat()
        for key in ("confidence", "agreement", "balance_at_switch"):
            if key in new_event and isinstance(new_event[key], float):
                new_event[key] = round(new_event[key], 10)
        if "agreement" not in new_event and "confidence" in new_event:
            new_event["agreement"] = round(float(new_event["confidence"]), 10)
        sanitized.append(new_event)
    return sanitized


SNAPSHOT_PATH = Path(__file__).with_name("regime_regression_snapshot.json")


@pytest.mark.integration
def test_regime_backtester_regression(monkeypatch):
    if not SNAPSHOT_PATH.exists():
        pytest.skip("Approved snapshot is missing")

    frame = _build_fixture_dataframe()
    provider = FixtureDataProvider(frame)

    primary_strategy = DeterministicStrategy(
        name="MlBasicStrategy", entry_period=30, hold_period=5, size=0.2
    )

    def strategy_factory(key: str) -> BaseStrategy:
        if key in {"ml_basic", "mlbasic"}:
            return DeterministicStrategy(
                name="MlBasicStrategy", entry_period=30, hold_period=5, size=0.2
            )
        if key == "alternate":
            return DeterministicStrategy(
                name="AlternateStrategy", entry_period=24, hold_period=4, size=0.15
            )
        return DeterministicStrategy(
            name="FallbackStrategy", entry_period=28, hold_period=4, size=0.18
        )

    strategy_manager = StubStrategyManager(strategy_factory)
    strategy_manager.current_strategy = primary_strategy

    monkeypatch.setenv("FEATURE_ENABLE_REGIME_DETECTION", "true")
    monkeypatch.setattr("src.live.strategy_manager.StrategyManager", lambda: strategy_manager)
    monkeypatch.setattr(
        "src.live.regime_strategy_switcher.RegimeStrategySwitcher",
        lambda strategy_manager, regime_config=None, strategy_mapping=None, switching_config=None: StubRegimeStrategySwitcher(
            strategy_manager, regime_config, strategy_mapping, switching_config
        ),
    )

    original_loader = Backtester._load_strategy_by_name

    def _patched_loader(self: Backtester, strategy_name: str) -> BaseStrategy | None:
        if strategy_name == "alternate":
            new_strategy = strategy_factory("alternate")
            strategy_manager.current_strategy = new_strategy
            return new_strategy
        if strategy_name in {"ml_basic", "mlbasic"}:
            new_strategy = strategy_factory("ml_basic")
            strategy_manager.current_strategy = new_strategy
            return new_strategy
        return original_loader(self, strategy_name)

    monkeypatch.setattr(Backtester, "_load_strategy_by_name", _patched_loader)

    backtester = Backtester(
        strategy=primary_strategy,
        data_provider=provider,
        initial_balance=10_000,
        enable_regime_switching=True,
        log_to_database=False,
        enable_dynamic_risk=False,
    )

    start = frame.index[0].to_pydatetime()
    end = frame.index[-1].to_pydatetime()
    results = backtester.run(symbol="TEST", timeframe="1h", start=start, end=end)

    sanitized_switches = _sanitize_time_series(results.get("strategy_switches", []))
    sanitized_regime_history = _sanitize_time_series(results.get("regime_history", []))

    observed = {
        "total_trades": results["total_trades"],
        "win_rate": results["win_rate"],
        "total_return": results["total_return"],
        "strategy_switches": sanitized_switches,
        "regime_history": sanitized_regime_history,
    }

    expected = json.loads(SNAPSHOT_PATH.read_text())

    assert observed["total_trades"] == expected["total_trades"]
    assert observed["win_rate"] == pytest.approx(expected["win_rate"], rel=1e-5, abs=1e-5)
    assert observed["total_return"] == pytest.approx(expected["total_return"], rel=1e-5, abs=1e-5)

    assert len(observed["strategy_switches"]) == len(expected["strategy_switches"])
    for obs, exp in zip(observed["strategy_switches"], expected["strategy_switches"]):
        assert obs["timestamp"] == exp["timestamp"]
        assert obs["old_strategy"] == exp["old_strategy"]
        assert obs["new_strategy"] == exp["new_strategy"]
        assert obs["regime"] == exp["regime"]
        assert obs["reason"] == exp["reason"]
        assert obs["confidence"] == pytest.approx(exp["confidence"], rel=1e-6, abs=1e-6)
        assert obs["agreement"] == pytest.approx(exp["agreement"], rel=1e-6, abs=1e-6)

    assert len(observed["regime_history"]) == len(expected["regime_history"])
    for obs, exp in zip(observed["regime_history"], expected["regime_history"]):
        assert obs["timestamp"] == exp["timestamp"]
        assert obs["regime"] == exp["regime"]
        assert obs["confidence"] == pytest.approx(exp["confidence"], rel=1e-6, abs=1e-6)
        assert obs["agreement"] == pytest.approx(exp["agreement"], rel=1e-6, abs=1e-6)

