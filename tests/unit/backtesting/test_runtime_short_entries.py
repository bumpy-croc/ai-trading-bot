from datetime import datetime, timedelta

import pandas as pd

from src.backtesting.engine import Backtester
from src.data_providers.data_provider import DataProvider
from src.strategies.components.position_sizer import PositionSizer
from src.strategies.components.regime_context import RegimeContext
from src.strategies.components.risk_manager import MarketData, Position, RiskManager
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator
from src.strategies.components.strategy import Strategy


class _FrameProvider(DataProvider):
    """Simple data provider returning a prepared DataFrame."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def get_historical_data(self, symbol, timeframe, start, end=None):  # type: ignore[override]
        return self._frame.copy()

    def get_current_price(self, symbol):  # type: ignore[override]
        return float(self._frame["close"].iloc[-1])

    def get_live_data(self, symbol, timeframe, limit=500):  # type: ignore[override]
        return self._frame.tail(limit).copy()

    def update_live_data(self, symbol, timeframe):  # type: ignore[override]
        return self._frame.copy()


class _ShortSignalGenerator(SignalGenerator):
    def __init__(self, enable_short: bool | None) -> None:
        super().__init__("short_signal")
        self._enable_short = enable_short

    def generate_signal(self, df: pd.DataFrame, index: int, regime: RegimeContext | None = None) -> Signal:  # type: ignore[override]
        if index == 0:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=1.0,
                metadata={},
            )

        metadata: dict[str, bool] = {}
        if self._enable_short is not None:
            metadata["enter_short"] = self._enable_short

        return Signal(
            direction=SignalDirection.SELL,
            strength=1.0,
            confidence=0.9,
            metadata=metadata,
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:  # type: ignore[override]
        return 0.9


class _ShortRiskManager(RiskManager):
    def __init__(self) -> None:
        super().__init__("short_risk")

    def calculate_position_size(self, signal: Signal, balance: float, regime: RegimeContext | None = None) -> float:  # type: ignore[override]
        return balance * 0.1 if signal.direction != SignalDirection.HOLD else 0.0

    def should_exit(self, position: Position, current_data: MarketData, regime: RegimeContext | None = None) -> bool:  # type: ignore[override]
        if position.side == "short":
            return current_data.price <= position.entry_price * 0.9
        return current_data.price >= position.entry_price * 1.1

    def get_stop_loss(self, entry_price: float, signal: Signal, regime: RegimeContext | None = None) -> float:  # type: ignore[override]
        if signal.direction == SignalDirection.SELL:
            return entry_price * 1.05
        return entry_price * 0.95


class _PassThroughSizer(PositionSizer):
    def __init__(self) -> None:
        super().__init__("pass_through")

    def calculate_size(self, signal: Signal, balance: float, risk_amount: float, regime: RegimeContext | None = None) -> float:  # type: ignore[override]
        return risk_amount


def _build_dataset() -> pd.DataFrame:
    start = datetime(2024, 1, 1)
    dates = [start + timedelta(hours=i) for i in range(3)]
    closes = [100.0, 95.0, 85.0]
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": [price * 1.01 for price in closes],
            "low": [price * 0.99 for price in closes],
            "close": closes,
            "volume": [1_000 for _ in closes],
        },
        index=dates,
    )
    return frame


def _build_strategy(enable_short: bool | None) -> Strategy:
    return Strategy(
        name="short_strategy",
        signal_generator=_ShortSignalGenerator(enable_short),
        risk_manager=_ShortRiskManager(),
        position_sizer=_PassThroughSizer(),
    )


def test_runtime_short_entry_blocks_when_flag_false():
    frame = _build_dataset()
    provider = _FrameProvider(frame)
    strategy = _build_strategy(enable_short=False)

    backtester = Backtester(
        strategy,
        provider,
        log_to_database=False,
        enable_dynamic_risk=False,
        enable_engine_risk_exits=False,
        use_next_bar_execution=False,  # Disable for this test
    )

    result = backtester.run(
        symbol="TESTUSDT",
        timeframe="1h",
        start=frame.index[0],
        end=frame.index[-1],
    )

    assert result["total_trades"] == 0
    assert backtester.trades == []


def test_runtime_short_entry_honors_metadata():
    frame = _build_dataset()
    provider = _FrameProvider(frame)
    strategy = _build_strategy(enable_short=True)

    backtester = Backtester(
        strategy,
        provider,
        log_to_database=False,
        enable_dynamic_risk=False,
        enable_engine_risk_exits=False,
        use_next_bar_execution=False,  # Disable for this test
    )

    result = backtester.run(
        symbol="TESTUSDT",
        timeframe="1h",
        start=frame.index[0],
        end=frame.index[-1],
    )

    assert result["total_trades"] == 1
    assert backtester.trades
    assert backtester.trades[0].side == "short"


def test_runtime_short_entry_blocks_when_metadata_missing():
    frame = _build_dataset()
    provider = _FrameProvider(frame)
    strategy = _build_strategy(enable_short=None)

    backtester = Backtester(
        strategy,
        provider,
        log_to_database=False,
        enable_dynamic_risk=False,
        enable_engine_risk_exits=False,
        use_next_bar_execution=False,  # Disable for this test
    )

    result = backtester.run(
        symbol="TESTUSDT",
        timeframe="1h",
        start=frame.index[0],
        end=frame.index[-1],
    )

    assert result["total_trades"] == 0
    assert backtester.trades == []
