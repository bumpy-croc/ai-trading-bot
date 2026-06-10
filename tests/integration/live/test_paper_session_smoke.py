"""End-to-end paper-trading session smoke (#486 refactor).

Drives the REAL LiveTradingEngine — real ml_basic strategy, real in-memory
DatabaseManager — through start -> trading loop -> paper entry -> monitoring
-> paper exit -> shutdown. Exercises the post-refactor wiring end to end:
handler construction (LiveStopLossManager, LiveAccountMonitor), the bounded
trading loop, entry/exit through the live handlers, and the shared
backtest-parity P&L formula on the recorded fills.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.database.manager import DatabaseManager
from src.engines.live.trading_engine import LiveTradingEngine, PositionSide
from src.performance.metrics import Side, pnl_percent
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.integration

SYMBOL = "BTCUSDT"


def _make_market_data(periods: int = 200) -> pd.DataFrame:
    idx = pd.date_range(
        datetime.now(UTC) - pd.Timedelta(hours=periods - 1), periods=periods, freq="1h"
    )
    base = [100.0 + 0.1 * i for i in range(periods)]
    return pd.DataFrame(
        {
            "open": base,
            "high": [p + 1.0 for p in base],
            "low": [p - 1.0 for p in base],
            "close": [p + 0.5 for p in base],
            "volume": [1000.0] * periods,
        },
        index=idx,
    )


@pytest.fixture
def paper_engine():
    """Real engine in paper mode with a real in-memory DB and mocked provider."""
    df = _make_market_data()
    provider = MagicMock()
    provider.get_live_data.return_value = df
    provider.get_historical_data.return_value = df
    provider.get_current_price.return_value = float(df["close"].iloc[-1])

    db = DatabaseManager("sqlite:///:memory:")
    with (
        patch("src.engines.live.trading_engine.DatabaseManager"),
        patch("src.engines.live.trading_engine.get_config", return_value={}),
    ):
        engine = LiveTradingEngine(
            strategy=create_ml_basic_strategy(),
            data_provider=provider,
            enable_live_trading=False,
            initial_balance=10_000.0,
            check_interval=0.01,
            log_trades=False,
        )
    # Rewire every handler to the real DB (construction used the patched class).
    engine.db_manager = db
    engine.live_position_tracker.db_manager = db
    engine.live_execution_engine.db_manager = db
    engine.event_logger.db_manager = db
    yield engine, df
    if engine.is_running:
        engine.stop()


def test_paper_session_start_entry_exit_shutdown(paper_engine):
    engine, df = paper_engine

    # --- start(): session creation, handler wiring, three real loop iterations.
    # Close-only mode keeps the real strategy from opening its own positions
    # during the loop (its signals depend on wall-clock-anchored data and would
    # make the scripted entry/exit below nondeterministic); exits still run.
    engine._close_only_mode = True
    engine.start(symbol=SYMBOL, timeframe="1h", max_steps=3)
    assert engine.trading_session_id is not None
    assert engine.stop_loss_manager is not None
    assert engine.account_monitor is not None
    assert engine._loop_crashed is False
    assert engine.live_position_tracker.position_count == 0

    baseline_positions = set(engine.live_position_tracker.positions)
    baseline_trades = len(engine.completed_trades)
    engine.resume_trading()

    # --- paper entry through the real entry path
    entry_price = float(df["close"].iloc[-1])
    engine._execute_entry(
        symbol=SYMBOL,
        side=PositionSide.LONG,
        size=0.05,
        price=entry_price,
        stop_loss=entry_price * 0.95,
        take_profit=entry_price * 1.10,
        signal_strength=0.9,
        signal_confidence=0.9,
    )
    new_position_ids = set(engine.live_position_tracker.positions) - baseline_positions
    assert len(new_position_ids) == 1

    # --- monitoring through LiveAccountMonitor
    engine._log_account_snapshot()
    engine._log_status(SYMBOL, entry_price)
    summary = engine.get_performance_summary()
    assert summary["current_balance"] == engine.current_balance
    assert summary["active_positions"] == engine.live_position_tracker.position_count

    # --- paper exit through the real exit path
    position = engine.live_position_tracker.positions[next(iter(new_position_ids))]
    entry_balance = float(position.entry_balance)  # P&L basis: balance at entry
    exit_price = entry_price * 1.05
    engine._execute_exit(
        position=position,
        reason="smoke_exit",
        limit_price=None,
        current_price=exit_price,
        candle_high=None,
        candle_low=None,
        candle=None,
    )
    assert next(iter(new_position_ids)) not in engine.live_position_tracker.positions
    assert len(engine.completed_trades) == baseline_trades + 1

    # Gross P&L must equal the shared pnl_percent formula on the RECORDED fill
    # prices (slippage included) times the entry balance — the same arithmetic
    # the backtest engine uses (backtest-live parity).
    trade = engine.completed_trades[-1]
    expected_gross = (
        pnl_percent(trade.entry_price, trade.exit_price, Side.LONG, 0.05) * entry_balance
    )
    assert trade.pnl == pytest.approx(expected_gross, abs=1e-9)
    # Default execution model applies 5bps slippage to both fills.
    assert trade.entry_price == pytest.approx(entry_price * 1.0005, abs=1e-9)
    assert trade.exit_price == pytest.approx(exit_price * 0.9995, abs=1e-9)

    # --- clean shutdown (final stats via LiveAccountMonitor)
    engine.stop()
    assert engine.is_running is False
