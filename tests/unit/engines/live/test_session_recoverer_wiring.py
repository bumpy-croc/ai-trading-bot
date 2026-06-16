"""Contract tests for the LiveSessionRecoverer extraction (#486 step 4).

Behavioral coverage of recovery semantics lives in test_session_recovery.py,
test_clean_restart_readoption.py, and test_reconciliation.py (exercised via
the engine wrappers). These tests pin the two contracts the extraction itself
introduced: state mutations through the RecoveryEngineState protocol must hit
the engine object, and the close-accounting helpers must remain importable
from trading_engine.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.engines.live import trade_close_accounting
from src.engines.live.recovery import LiveSessionRecoverer
from src.engines.live.trading_engine import (
    LiveTradingEngine,
    _close_entry_fee_usd,
    _close_position_portion,
    _closed_base_quantity,
)

pytestmark = pytest.mark.fast


def make_engine() -> LiveTradingEngine:
    with patch("src.engines.live.trading_engine.DatabaseManager"):
        return LiveTradingEngine(
            strategy=MagicMock(),
            data_provider=MagicMock(),
            initial_balance=1000.0,
        )


class TestHelperReExports:
    def test_trading_engine_reexports_are_the_moved_functions(self):
        assert _closed_base_quantity is trade_close_accounting._closed_base_quantity
        assert _close_entry_fee_usd is trade_close_accounting._close_entry_fee_usd
        assert _close_position_portion is trade_close_accounting._close_position_portion


class TestRecovererWiring:
    def test_engine_constructs_recoverer_bound_to_itself(self):
        engine = make_engine()
        assert isinstance(engine.session_recoverer, LiveSessionRecoverer)
        assert engine.session_recoverer._state is engine

    def test_session_id_mutation_reaches_engine(self):
        # Active-session crash recovery writes trading_session_id through the
        # protocol; the trading loop later reads it off the engine.
        engine = make_engine()
        engine.db_manager = MagicMock()
        engine.db_manager.get_active_session_id.return_value = 77
        engine.db_manager.recover_last_balance.return_value = 432.10

        recovered = engine._recover_existing_session()

        assert recovered == 432.10
        assert engine.trading_session_id == 77
        engine.db_manager.set_current_session.assert_called_once_with(77)
        assert engine.live_execution_engine.session_id == 77

    def test_inactive_session_marks_carry_forward_on_engine(self):
        engine = make_engine()
        engine.db_manager = MagicMock()
        engine.db_manager.get_active_session_id.return_value = None
        engine.db_manager.get_last_session_id.return_value = 55
        engine.db_manager.recover_last_balance.return_value = 100.0

        engine._recover_existing_session()

        assert engine._recovered_inactive_session_id == 55
        # Clean restart must NOT reuse the old session id.
        assert engine.trading_session_id is None

    def test_corrupt_balance_propagates_valueerror(self):
        # #681/#668: non-finite recovered balance must fail startup fast, not
        # fall back to the default balance.
        engine = make_engine()
        engine.db_manager = MagicMock()
        engine.db_manager.get_active_session_id.return_value = 9
        engine.db_manager.recover_last_balance.return_value = float("inf")

        with pytest.raises(ValueError, match="not finite"):
            engine._recover_existing_session()
