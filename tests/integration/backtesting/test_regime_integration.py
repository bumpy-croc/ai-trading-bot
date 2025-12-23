import sys
from datetime import datetime, timedelta
from types import ModuleType, SimpleNamespace

import pandas as pd

from src.backtesting import engine as backtesting_engine
from src.backtesting.engine import Backtester
from src.data_providers.mock_data_provider import MockDataProvider
from src.regime.detector import RegimeConfig, RegimeDetector
from src.strategies.components.strategy import Strategy
from src.strategies.components.signal_generator import SignalGenerator, Signal, SignalDirection
from src.strategies.components.risk_manager import RiskManager
from src.strategies.components.position_sizer import PositionSizer


class PeriodicSignalGenerator(SignalGenerator):
    """Signal generator that signals BUY every 15 candles"""

    def __init__(self):
        super().__init__(name="periodic_signal")

    def generate_signal(self, df: pd.DataFrame, index: int, regime=None) -> Signal:
        if index % 15 == 0 and index > 0:
            return Signal(
                direction=SignalDirection.BUY,
                confidence=0.7,
                strength=1.0,
                metadata={"timestamp": df.index[index]},
            )
        return Signal(
            direction=SignalDirection.HOLD,
            confidence=0.0,
            strength=0.0,
            metadata={"timestamp": df.index[index]},
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        if index % 15 == 0 and index > 0:
            return 0.7
        return 0.0


class FixedRiskManager(RiskManager):
    """Risk manager with fixed risk"""

    def __init__(self):
        super().__init__(name="fixed_risk")

    def calculate_position_size(self, signal: Signal, balance: float, regime=None) -> float:
        return 0.05 * balance

    def should_exit(self, position, current_data, regime=None) -> bool:
        return False

    def get_stop_loss(self, entry_price: float, signal: Signal, regime=None) -> float:
        return entry_price * 0.99


class FixedPositionSizer(PositionSizer):
    """Position sizer with fixed fraction"""

    def __init__(self):
        super().__init__(name="fixed_sizer")

    def calculate_size(
        self, signal: Signal, balance: float, risk_amount: float, regime=None
    ) -> float:
        return 0.05


def create_dummy_strategy() -> Strategy:
    """Create a dummy component-based strategy for testing"""
    return Strategy(
        name="DummyStrategy",
        signal_generator=PeriodicSignalGenerator(),
        risk_manager=FixedRiskManager(),
        position_sizer=FixedPositionSizer(),
    )


def test_backtester_regime_annotation(monkeypatch):
    monkeypatch.setenv("FEATURE_ENABLE_REGIME_DETECTION", "true")
    strategy = create_dummy_strategy()
    provider = MockDataProvider(interval_seconds=1, num_candles=500)
    start = datetime.now() - timedelta(hours=400)
    end = datetime.now()
    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        initial_balance=1000,
        log_to_database=False,
    )

    # Inject a deterministic regime switcher so we can assert on regime metrics
    class StubRegimeSwitcher:
        def __init__(self):
            self._call_count = 0

        def analyze_market_regime(self, price_data):
            self._call_count += 1
            if self._call_count <= 4:
                regime_label = "bull:low_vol"
            else:
                regime_label = "bear:high_vol"
            return {
                "consensus_regime": {
                    "regime_label": regime_label,
                    "confidence": 0.75,
                    "agreement_score": 0.6,
                },
                "analysis_timestamp": datetime.now(),
            }

        def should_switch_strategy(self, regime_analysis, current_candle_index):
            return {
                "should_switch": False,
                "optimal_strategy": "dummy",
                "new_regime": regime_analysis["consensus_regime"]["regime_label"],
                "confidence": regime_analysis["consensus_regime"]["confidence"],
                "reason": "stubbed-no-switch",
            }

    backtester.enable_regime_switching = True

    # Manually initialize regime handler with stub switcher
    from src.backtesting.regime.regime_handler import RegimeHandler
    from src.live.strategy_manager import StrategyManager

    strategy_manager = StrategyManager()
    regime_switcher = StubRegimeSwitcher()
    backtester.regime_handler = RegimeHandler(
        regime_switcher=regime_switcher,
        strategy_manager=strategy_manager,
        initial_strategy_name="DummyStrategy",
    )

    result = backtester.run(symbol="BTCUSDT", timeframe="1h", start=start, end=end)
    # Detector should be initialized and run without error
    assert backtester.regime_detector is not None
    assert "total_trades" in result

    # Regime metadata should be populated when the feature flag is enabled
    assert result["regime_switching_enabled"] is True
    assert result["total_strategy_switches"] == 0
    assert len(result["regime_history"]) == 7
    assert result["regime_history"] == backtester.regime_handler.regime_history
    regime_indices = [entry["candle_index"] for entry in result["regime_history"]]
    assert regime_indices == [100, 150, 200, 250, 300, 350, 400]
    last_regime = result["regime_history"][-1]
    assert last_regime["regime"] == "bear:high_vol"
    assert last_regime["confidence"] == 0.75


def test_regime_switcher_respects_lookback(monkeypatch):
    monkeypatch.setenv("FEATURE_ENABLE_REGIME_DETECTION", "true")

    captured_slices: list[tuple[pd.Timestamp, pd.Timestamp, int, tuple[str, ...]]] = []

    class DummyStrategyManager:
        def __init__(self):
            self.current_strategy = SimpleNamespace(name="dummy")

        def load_strategy(self, strategy_key: str):
            strategy_obj = SimpleNamespace(name=strategy_key)
            self.current_strategy = strategy_obj
            return strategy_obj

    class DummySwitcher:
        def __init__(
            self, strategy_manager, regime_config=None, strategy_mapping=None, switching_config=None
        ):
            self.strategy_manager = strategy_manager
            self.regime_detector = RegimeDetector(
                RegimeConfig(slope_window=30, atr_percentile_lookback=80)
            )
            self.timeframe_detectors = {
                "1h": RegimeDetector(RegimeConfig(slope_window=35, atr_percentile_lookback=90)),
                "4h": RegimeDetector(RegimeConfig(slope_window=40, atr_percentile_lookback=110)),
            }
            self.switching_config = SimpleNamespace(
                enable_multi_timeframe=True,
                timeframes=["1h", "4h"],
                min_regime_confidence=0.0,
                require_timeframe_agreement=0.0,
                min_regime_duration=0,
                switch_cooldown_minutes=0,
            )

        def analyze_market_regime(self, price_data):
            df_slice = next(iter(price_data.values()))
            captured_slices.append(
                (
                    df_slice.index[0],
                    df_slice.index[-1],
                    len(df_slice),
                    tuple(sorted(price_data.keys())),
                )
            )
            return {
                "consensus_regime": {
                    "regime_label": "trend_up:low_vol",
                    "confidence": 0.9,
                    "agreement_score": 1.0,
                },
                "timeframe_regimes": {},
                "analysis_timestamp": datetime.now(),
            }

        @staticmethod
        def should_switch_strategy(regime_analysis, current_candle_index=None):
            return {
                "should_switch": False,
                "optimal_strategy": "dummy",
                "new_regime": regime_analysis["consensus_regime"]["regime_label"],
                "confidence": regime_analysis["consensus_regime"]["confidence"],
                "reason": "no-switch",
                "current_strategy": None,
                "agreement": regime_analysis["consensus_regime"]["agreement_score"],
            }

    strategy_manager_module = ModuleType("src.live.strategy_manager")
    strategy_manager_module.StrategyManager = DummyStrategyManager
    monkeypatch.setitem(sys.modules, "src.live.strategy_manager", strategy_manager_module)

    switcher_module = ModuleType("src.live.regime_strategy_switcher")
    switcher_module.RegimeStrategySwitcher = DummySwitcher
    monkeypatch.setitem(sys.modules, "src.live.regime_strategy_switcher", switcher_module)

    strategy = create_dummy_strategy()
    provider = MockDataProvider(interval_seconds=3600, num_candles=500, seed=123)
    start = datetime.now() - timedelta(hours=600)
    end = datetime.now()

    raw_df = provider.get_historical_data("BTCUSDT", "1h", start, end)
    prepared_df = raw_df.copy()
    prepared_df = prepared_df.dropna(subset=["open", "high", "low", "close", "volume"])

    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        initial_balance=1000,
        log_to_database=False,
        enable_regime_switching=True,
    )

    backtester.run(symbol="BTCUSDT", timeframe="1h", start=start, end=end)

    assert captured_slices, "Regime switcher should be invoked with price data"

    lookback = backtesting_engine._compute_regime_lookback(backtester.regime_switcher)
    assert lookback > 0

    for start_ts, end_ts, length, keys in captured_slices:
        # Ensure multi-timeframe inputs receive the shared slice
        assert keys
        assert {"1h", "4h"}.issuperset(set(keys))

        end_idx = prepared_df.index.get_loc(end_ts)
        expected_start_idx = max(0, (end_idx + 1) - lookback)
        actual_start_idx = prepared_df.index.get_loc(start_ts)
        assert actual_start_idx == expected_start_idx
        assert length == end_idx - actual_start_idx + 1
