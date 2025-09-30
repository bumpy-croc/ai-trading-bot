from datetime import datetime, timedelta

import pandas as pd

from src.backtesting.engine import Backtester
from src.data_providers.mock_data_provider import MockDataProvider
from src.strategies.base import BaseStrategy


class DummyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("DummyStrategy")
        self.take_profit_pct = 0.02
        self.stop_loss_pct = 0.01

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["onnx_pred"] = df["close"]
        df["prediction_confidence"] = 0.1
        return df

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        return index % 15 == 0 and index > 0

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        return False

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        return 0.05

    def calculate_stop_loss(self, df, index, price, side="long") -> float:
        return price * (1 - self.stop_loss_pct)

    def get_parameters(self) -> dict:
        return {}


def test_backtester_regime_annotation(monkeypatch):
    monkeypatch.setenv("FEATURE_ENABLE_REGIME_DETECTION", "true")
    strategy = DummyStrategy()
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
    backtester.regime_switcher = StubRegimeSwitcher()
    result = backtester.run(symbol="BTCUSDT", timeframe="1h", start=start, end=end)
    # Detector should be initialized and run without error
    assert backtester.regime_detector is not None
    assert "total_trades" in result

    # Regime metadata should be populated when the feature flag is enabled
    assert result["regime_switching_enabled"] is True
    assert result["total_strategy_switches"] == 0
    assert len(result["regime_history"]) == 7
    assert result["regime_history"] == backtester.regime_history
    regime_indices = [entry["candle_index"] for entry in result["regime_history"]]
    assert regime_indices == [100, 150, 200, 250, 300, 350, 400]
    last_regime = result["regime_history"][-1]
    assert last_regime["regime"] == "bear:high_vol"
    assert last_regime["confidence"] == 0.75
