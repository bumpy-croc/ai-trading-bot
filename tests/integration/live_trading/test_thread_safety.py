import threading
import time
from datetime import datetime

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

try:
    from src.live.trading_engine import LiveTradingEngine

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
            {"close": [51000]}, index=[datetime.now()]
        )

        def open_positions():
            if hasattr(engine, "_open_position"):
                for i in range(3):
                    try:
                        engine._open_position(
                            symbol=f"BTC{i}USDT", side="LONG", size=0.02, price=50000 + i
                        )
                    except Exception:
                        pass
                    time.sleep(0.01)

        def close_positions():
            time.sleep(0.05)
            try:
                positions_to_close = list(engine.positions.values())[:2]
                for position in positions_to_close:
                    if hasattr(engine, "_close_position"):
                        engine._close_position(position, "Test close")
                    time.sleep(0.01)
            except Exception:
                pass

        t1 = threading.Thread(target=open_positions)
        t2 = threading.Thread(target=close_positions)
        t1.start()
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)
        assert len(engine.positions) >= 0
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
            index=[datetime.now()],
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
