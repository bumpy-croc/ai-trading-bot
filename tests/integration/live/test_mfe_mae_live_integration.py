from datetime import datetime

import pandas as pd
import pytest

from src.data_providers.mock_data_provider import MockDataProvider
from src.database.models import TradeSource
from src.engines.live.trading_engine import LiveTradingEngine, PositionSide
from src.strategies.components import FixedFractionSizer, FixedRiskManager, Strategy
from src.strategies.components.regime_context import RegimeContext, TrendLabel, VolLabel
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator


class AlternatingSignalGenerator(SignalGenerator):
    """Deterministic signal generator that alternates BUY/SELL decisions."""

    def __init__(self):
        super().__init__("alternating_signal_generator")
        self._counter = 0

    def generate_signal(self, df: pd.DataFrame, index: int, regime=None) -> Signal:
        self.validate_inputs(df, index)
        self._counter += 1
        direction = SignalDirection.BUY if self._counter % 2 else SignalDirection.SELL
        return Signal(
            direction=direction,
            strength=1.0,
            confidence=1.0,
            metadata={"step": self._counter},
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        self.validate_inputs(df, index)
        return 1.0


class StaticRegimeDetector:
    """Lightweight regime detector returning a stable context."""

    name = "static_regime_detector"
    warmup_period = 0

    def detect_regime(self, df: pd.DataFrame, index: int) -> RegimeContext:
        return RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=index + 1,
            strength=1.0,
            timestamp=datetime.utcnow(),
        )

    def get_feature_generators(self):
        return []


def create_test_strategy() -> Strategy:
    """Strategy suitable for fast live-engine integration tests."""
    strategy = Strategy(
        name="integration_test_strategy",
        signal_generator=AlternatingSignalGenerator(),
        risk_manager=FixedRiskManager(risk_per_trade=0.02, stop_loss_pct=0.05),
        position_sizer=FixedFractionSizer(
            fraction=0.05, adjust_for_confidence=False, adjust_for_strength=False
        ),
        regime_detector=StaticRegimeDetector(),
        enable_logging=False,
    )
    strategy.set_warmup_period(0)
    return strategy


@pytest.mark.integration
@pytest.mark.mock_only
def test_live_engine_records_mfe_mae():
    provider = MockDataProvider(interval_seconds=1, num_candles=50)
    strategy = create_test_strategy()
    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=provider,
        check_interval=0.01,
        initial_balance=1000.0,
        enable_live_trading=False,
        log_trades=True,
        account_snapshot_interval=0,
        resume_from_last_balance=False,
        database_url=None,
        enable_dynamic_risk=False,
        enable_hot_swapping=False,
    )

    # Manually create a trading session to enable database logging
    session_id = engine.db_manager.create_trading_session(
        strategy_name=strategy.name,
        symbol="BTCUSDT",
        timeframe="1h",
        mode=TradeSource.PAPER,
        initial_balance=engine.current_balance,
        strategy_config={},
    )
    engine.trading_session_id = session_id
    engine.db_manager.update_balance(engine.current_balance, "session_start", "system", session_id)

    # Open a paper position
    engine._execute_entry(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=0.05,
        price=provider.get_current_price("BTCUSDT"),
        stop_loss=None,
        take_profit=None,
        signal_strength=0.0,
        signal_confidence=0.0,
    )

    order_id, position = next(iter(engine.live_position_tracker._positions.items()))

    # Simulate price moves to record MFE/MAE metrics
    entry_price = float(position.entry_price)
    engine.mfe_mae_tracker.update_position_metrics(
        position_key=order_id,
        entry_price=entry_price,
        current_price=entry_price * 1.03,  # +3% move
        side=position.side.value,
        position_fraction=float(position.size),
        current_time=datetime.utcnow(),
    )
    engine.mfe_mae_tracker.update_position_metrics(
        position_key=order_id,
        entry_price=entry_price,
        current_price=entry_price * 0.97,  # -3% move
        side=position.side.value,
        position_fraction=float(position.size),
        current_time=datetime.utcnow(),
    )

    # Close the position to trigger trade logging with MFE/MAE
    engine._execute_exit(
        position=position,
        reason="test_exit",
        limit_price=None,
        current_price=provider.get_current_price(position.symbol),
        candle_high=None,
        candle_low=None,
        candle=None,
    )

    trades = engine.db_manager.get_recent_trades(limit=5, session_id=session_id)
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
