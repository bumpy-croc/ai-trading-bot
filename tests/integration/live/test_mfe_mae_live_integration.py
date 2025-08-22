
import pandas as pd
import pytest

from src.data_providers.mock_data_provider import MockDataProvider
from src.live.trading_engine import LiveTradingEngine
from src.strategies.base import BaseStrategy


class QuickFlipStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("QuickFlipStrategy")
        self._opened = False

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        # Enter only once to keep test deterministic
        if not self._opened and index >= 1:
            self._opened = True
            return True
        return False

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        # Exit when price moves even slightly from entry
        if index < 1 or index >= len(df):
            return False
        current_price = float(df["close"].iloc[index])
        return current_price != float(entry_price)

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        return 0.1  # 10% of balance

    def calculate_stop_loss(self, df, index, price, side: str = "long") -> float:
        return float(price) * 0.98

    def get_parameters(self) -> dict:
        return {}


@pytest.mark.integration
@pytest.mark.mock_only
def test_live_engine_records_mfe_mae():
    # Mock data provider with fast updates
    provider = MockDataProvider(interval_seconds=1, num_candles=50)

    strategy = QuickFlipStrategy()

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=provider,
        check_interval=0.01,  # fast loop
        initial_balance=1000.0,
        enable_live_trading=False,
        log_trades=True,
        account_snapshot_interval=0,  # disable snapshots for speed
        database_url="sqlite:///:memory:",
        enable_dynamic_risk=False,
        enable_hot_swapping=False,
    )

    # Run a short session
    engine.start(symbol="BTCUSDT", timeframe="1h", max_steps=8)

    # Verify at least one trade with MFE/MAE recorded
    assert engine.trading_session_id is not None
    trades = engine.db_manager.get_recent_trades(limit=5, session_id=engine.trading_session_id)
    assert isinstance(trades, list)
    assert len(trades) >= 1

    t0 = trades[0]
    # MFE/MAE fields should be present and numeric (decimals returned by DB manager are acceptable)
    assert "mfe" in t0 and "mae" in t0
    mfe = float(t0.get("mfe") or 0.0)
    mae = float(t0.get("mae") or 0.0)
    # Expect sign consistency: MFE >= 0, MAE <= 0
    assert mfe >= 0.0
    assert mae <= 0.0