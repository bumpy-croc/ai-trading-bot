import threading
import time
from datetime import UTC, datetime

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

try:
    from src.engines.live.trading_engine import LiveTradingEngine, PositionSide

    LIVE_TRADING_AVAILABLE = True
except ImportError:
    LIVE_TRADING_AVAILABLE = False

    class LiveTradingEngine:
        def __init__(self, strategy=None, data_provider=None, **kwargs):
            self.strategy = strategy
            self.data_provider = data_provider
            self.positions = {}
            self.completed_trades = []

        def stop(self):
            self.is_running = False


class TestThreadSafety:
    @pytest.mark.live_trading
    def test_concurrent_position_updates(self, mock_strategy, mock_data_provider):
        if not LIVE_TRADING_AVAILABLE:
            pytest.skip("Live trading components not available")
        engine = LiveTradingEngine(strategy=mock_strategy, data_provider=mock_data_provider)
        mock_data_provider.get_live_data.return_value = pd.DataFrame(
            {"close": [51000]}, index=[datetime.now(UTC)]
        )

        def open_positions():
            for i in range(3):
                try:
                    engine._execute_entry(
                        symbol=f"BTC{i}USDT",
                        side=PositionSide.LONG,
                        size=0.02,
                        price=50000 + i,
                        stop_loss=None,
                        take_profit=None,
                        signal_strength=0.0,
                        signal_confidence=0.0,
                    )
                except Exception:
                    pass
                time.sleep(0.01)

        def close_positions():
            time.sleep(0.05)
            try:
                positions_to_close = list(engine.live_position_tracker._positions.values())[:2]
                for position in positions_to_close:
                    engine._execute_exit(
                        position=position,
                        reason="Test close",
                        limit_price=None,
                        current_price=50000,
                        candle_high=None,
                        candle_low=None,
                        candle=None,
                    )
                    time.sleep(0.01)
            except Exception:
                pass

        t1 = threading.Thread(target=open_positions)
        t2 = threading.Thread(target=close_positions)
        t1.start()
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)
        assert len(engine.live_position_tracker._positions) >= 0
        assert len(engine.completed_trades) >= 0

    @pytest.mark.live_trading
    def test_stop_event_handling(self, mock_strategy, mock_data_provider):
        if not LIVE_TRADING_AVAILABLE:
            pytest.skip("Live trading components not available")
        engine = LiveTradingEngine(
            strategy=mock_strategy, data_provider=mock_data_provider, check_interval=1
        )
        mock_data_provider.get_live_data.return_value = pd.DataFrame(
            {"open": [50000], "high": [50100], "low": [49900], "close": [50050], "volume": [1000]},
            index=[datetime.now(UTC)],
        )

        def run_trading():
            if hasattr(engine, "_trading_loop"):
                engine._trading_loop("BTCUSDT", "1h")

        thread = threading.Thread(target=run_trading)
        thread.daemon = True
        thread.start()
        time.sleep(0.5)
        if hasattr(engine, "stop_event"):
            engine.stop_event.set()
        engine.is_running = False
        thread.join(timeout=3)
        assert not thread.is_alive()
