"""Regression: a balance recovered from the DB as a Decimal must be coerced to
float at the engine recovery boundary (discovered while building #668).

``DatabaseManager.recover_last_balance`` can return a ``decimal.Decimal``: when no
``account_balances`` row exists it falls back to ``TradingSession.initial_balance``
(a ``Numeric(18, 8)`` column) plus net trade PnL, and SQLAlchemy hands ``Numeric``
back as ``Decimal``. ``start()`` stored that straight into ``self.current_balance``,
so the shutdown banner's float arithmetic in ``_print_final_stats()`` —
``(self.current_balance - self.initial_balance) / self.initial_balance`` — raised
``TypeError: unsupported operand type(s) for -: 'decimal.Decimal' and 'float'``.

``start()`` must coerce the recovered balance to ``float`` so ``current_balance``
stays a float invariant (CODE.md "Arithmetic & Financial Calculations": coerce
``Numeric`` reads to ``float`` at the boundary, as ``get_active_positions`` does).

These drive ``LiveTradingEngine.start()`` in PAPER mode against a REAL in-memory DB
(so the Decimal-producing recovery transaction runs for real) with only the
trading loop / websocket I/O mocked, so ``start()`` returns promptly and runs
``stop()`` -> ``_print_final_stats()`` end to end. Paper mode skips all live-only
machinery (account sync, reconcilers, order tracker), keeping the repro minimal.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.database.manager import DatabaseManager
from src.database.models import TradeSource
from src.engines.live.trading_engine import LiveTradingEngine
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.fast

# Must equal create_ml_basic_strategy().name — get_last_session_id() filters the
# recovery candidate by strategy name.
STRATEGY_NAME = "MlBasic"
SYMBOL = "BTCUSDT"


def _seed_inactive_session(db: DatabaseManager) -> int:
    """Create an ENDED session with NO account_balances row.

    With no persisted balance row, recover_last_balance() takes its trades
    fallback and returns ``initial_balance`` (a Numeric column) as a Decimal —
    the exact value that used to corrupt the engine's current_balance.
    """
    session_id = db.create_trading_session(
        strategy_name=STRATEGY_NAME,
        symbol=SYMBOL,
        timeframe="1h",
        mode=TradeSource.PAPER,
        initial_balance=1000.0,
    )
    # Gracefully end the session (mirrors a clean shutdown): is_active -> False.
    db.end_trading_session(session_id=session_id, final_balance=1000.0)
    return session_id


def _log_closed_trade(db: DatabaseManager, session_id: int, *, pnl: float) -> None:
    """Log one CLOSED trade with a known net PnL and NO account_balances row.

    This forces recover_last_balance() down its trades fallback with a non-zero
    float ``total_pnl`` — the branch that did ``Decimal(initial_balance) + float``
    and raised TypeError.
    """
    now = datetime.now(UTC)
    db.log_trade(
        symbol=SYMBOL,
        side="long",
        entry_price=100.0,
        exit_price=100.0 + pnl,
        size=0.1,
        entry_time=now,
        exit_time=now,
        pnl=pnl,
        exit_reason="take_profit",
        strategy_name=STRATEGY_NAME,
        source=TradeSource.PAPER,
        session_id=session_id,
        quantity=1.0,
    )


def _make_paper_engine_with_real_db(db: DatabaseManager) -> LiveTradingEngine:
    """Build a paper-mode engine wired to a real in-memory DB.

    Paper mode needs no exchange provider, so the recovery transaction is the
    only thing touching the (real) database.
    """
    strategy = create_ml_basic_strategy()
    with (
        patch("src.engines.live.trading_engine.DatabaseManager"),
        patch("src.engines.live.trading_engine.get_config", return_value={}),
    ):
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=MagicMock(),
            initial_balance=1000.0,
            enable_live_trading=False,
            resume_from_last_balance=True,
        )

    # Swap in the real database holding the prior ended session.
    engine.db_manager = db
    return engine


def _drive_start(engine: LiveTradingEngine, *, mock_final_stats: bool) -> None:
    """Run start() to completion with runtime I/O neutralized.

    The trading loop is a no-op so its worker thread dies immediately and
    start()'s keep-alive loop exits and calls stop(). When ``mock_final_stats``
    is False the real ``_print_final_stats`` runs inside stop(), exercising the
    Decimal/float banner arithmetic end to end.
    """
    engine._run_trading_loop = MagicMock()
    engine._start_websocket_streams = MagicMock()
    engine._exit_if_loop_crashed = MagicMock()
    if mock_final_stats:
        engine._print_final_stats = MagicMock()
    engine.start(symbol=SYMBOL, timeframe="1h", max_steps=0)


class TestRecoveredDecimalBalanceCoercedToFloat:
    def test_current_balance_is_float_after_recovery(self):
        db = DatabaseManager("sqlite:///:memory:")
        _seed_inactive_session(db)
        engine = _make_paper_engine_with_real_db(db)

        # Mock the banner here so start() completes and we can inspect the stored
        # balance directly (the no-raise behaviour is covered by the test below).
        _drive_start(engine, mock_final_stats=True)

        # Recovered from a Numeric column via the trades fallback; the engine must
        # store it as float, not decimal.Decimal.
        assert isinstance(engine.current_balance, float)

    def test_print_final_stats_does_not_raise_on_recovered_balance(self):
        db = DatabaseManager("sqlite:///:memory:")
        _seed_inactive_session(db)
        engine = _make_paper_engine_with_real_db(db)

        # Real _print_final_stats runs inside stop(): it mixes current_balance
        # with the float initial_balance, so a Decimal current_balance would
        # raise "unsupported operand -: Decimal and float" here.
        _drive_start(engine, mock_final_stats=False)


class TestRecoverLastBalanceTradesFallback:
    """recover_last_balance's trades fallback adds net PnL (always float, via
    _trade_net_pnl) to TradingSession.initial_balance (Numeric -> Decimal).
    ``Decimal + float`` raises TypeError, which _recover_existing_session swallows
    -> None -> the balance is silently lost and the session resets. The DB read
    boundary must coerce to a finite float."""

    def test_recover_last_balance_with_trades_returns_finite_float(self):
        db = DatabaseManager("sqlite:///:memory:")
        session_id = db.create_trading_session(
            strategy_name=STRATEGY_NAME,
            symbol=SYMBOL,
            timeframe="1h",
            mode=TradeSource.PAPER,
            initial_balance=1000.0,
        )
        _log_closed_trade(db, session_id, pnl=250.0)
        db.end_trading_session(session_id=session_id, final_balance=1250.0)

        balance = db.recover_last_balance(session_id)

        assert isinstance(balance, float)
        assert math.isfinite(balance)
        assert balance == pytest.approx(1250.0)

    def test_engine_recovers_balance_with_trades_not_reset(self):
        db = DatabaseManager("sqlite:///:memory:")
        session_id = db.create_trading_session(
            strategy_name=STRATEGY_NAME,
            symbol=SYMBOL,
            timeframe="1h",
            mode=TradeSource.PAPER,
            initial_balance=1000.0,
        )
        _log_closed_trade(db, session_id, pnl=250.0)
        db.end_trading_session(session_id=session_id, final_balance=1250.0)
        engine = _make_paper_engine_with_real_db(db)

        _drive_start(engine, mock_final_stats=True)

        # Recovered as a float, and actually carried forward (initial 1000 + 250
        # PnL) rather than silently reset to the constructor default by a
        # swallowed TypeError.
        assert isinstance(engine.current_balance, float)
        assert engine.current_balance == pytest.approx(1250.0)


class TestNonFiniteRecoveredBalanceRejected:
    """A non-finite recovered balance (corrupt persisted state; float() happily
    turns Decimal('Infinity')/NaN into inf/nan) must never reach position sizing.
    _recover_existing_session() rejects it BEFORE its own `> 0` positivity filter —
    that filter would otherwise drop -inf/NaN to None (engine starts on the default
    balance) or raise on Decimal('NaN'). The engine fails fast instead (CODE.md
    "Arithmetic & Financial Calculations": math.isfinite() on trading inputs).

    The DB layer (recover_last_balance) is mocked so the REAL _recover_existing_session
    path — filter and guard — actually runs; mocking _recover_existing_session itself
    would bypass the very code under test.
    """

    @pytest.mark.parametrize(
        "bad_value",
        [
            float("inf"),
            float("-inf"),
            float("nan"),
            Decimal("Infinity"),
            Decimal("-Infinity"),
            Decimal("NaN"),
        ],
    )
    def test_non_finite_recovered_balance_refuses_to_start(self, bad_value):
        db = DatabaseManager("sqlite:///:memory:")
        engine = _make_paper_engine_with_real_db(db)
        # Drive the active (crash-recovery) branch deterministically, then have the
        # DB hand back a corrupt non-finite balance.
        engine.db_manager.get_active_session_id = MagicMock(return_value=42)
        engine.db_manager.recover_last_balance = MagicMock(return_value=bad_value)

        with pytest.raises(ValueError, match="Recovered balance is not finite"):
            _drive_start(engine, mock_final_stats=True)
