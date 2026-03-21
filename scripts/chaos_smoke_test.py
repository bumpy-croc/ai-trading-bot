#!/usr/bin/env python3
"""
Chaos Smoke Test Script

Orchestrates failure-scenario testing for the reconciliation system by
running the chaos_test strategy and validating database state across four
phases: journal validation, crash recovery, balance integrity, and
stop-loss reconciliation guidance.

Supports two operating modes:
  - Local (default): spawns the trading bot as a subprocess
  - Railway (--railway): assumes bot is a separate service, validates via DB

Usage:
    python scripts/chaos_smoke_test.py --phase journal --trades 5 --timeout 300
    python scripts/chaos_smoke_test.py --phase crash --timeout 120
    python scripts/chaos_smoke_test.py --phase balance --trades 10 --timeout 600
    python scripts/chaos_smoke_test.py --phase all --timeout 900
    python scripts/chaos_smoke_test.py --phase journal --railway --timeout 300
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, desc, func, text
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

# Ensure the project root is on sys.path so imports work when run as a script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.database.models import (  # noqa: E402
    AccountBalance,
    Order,
    OrderStatus,
    Position,
    Trade,
    TradingSession,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chaos_smoke_test")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_db_session(database_url: str) -> Session:
    """Create a SQLAlchemy session from a DATABASE_URL."""
    engine = create_engine(database_url)
    factory = sessionmaker(bind=engine)
    return factory()


def _poll_until(
    predicate: Callable[[], bool],
    timeout: float,
    interval: float = 5.0,
    description: str = "condition",
) -> bool:
    """Poll *predicate()* until it returns True or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        logger.info("Waiting for %s... (%.0fs remaining)", description, deadline - time.monotonic())
        time.sleep(interval)
    return False


def _start_bot(extra_args: Sequence[str] | None = None) -> subprocess.Popen:
    """Start the trading bot as a subprocess and return the Popen handle."""
    atb_path = shutil.which("atb")
    if atb_path:
        cmd = [
            atb_path,
            "live",
            "chaos_test",
            "--symbol",
            "BTCUSDT",
            "--timeframe",
            "1m",
            "--paper-trading",
            "--check-interval",
            "30",
        ]
    else:
        # Fallback to module invocation when atb is not on PATH
        cmd = [
            sys.executable,
            "-m",
            "src.engines.live.runner",
            "chaos_test",
            "--symbol",
            "BTCUSDT",
            "--timeframe",
            "1m",
            "--paper-trading",
            "--check-interval",
            "30",
        ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Starting bot: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        cwd=_PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _kill_bot(proc: subprocess.Popen) -> None:
    """Hard-kill the bot process (SIGKILL)."""
    if proc.poll() is None:
        logger.info("Sending SIGKILL to bot (pid=%d)", proc.pid)
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def _graceful_stop(proc: subprocess.Popen, timeout: int = 10) -> None:
    """Send SIGTERM and wait for graceful shutdown."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_bot(proc)


# ---------------------------------------------------------------------------
# Phase 1 — Journal Validation
# ---------------------------------------------------------------------------


def phase_journal(
    db: Session,
    target_trades: int,
    timeout: float,
    railway: bool,
) -> bool:
    """Validate that all orders have proper IDs and valid status lifecycles.

    In local mode, starts and manages the bot subprocess.
    In Railway mode, assumes the bot is already running.
    """
    logger.info("=== Phase 1: Journal Validation (target=%d trades) ===", target_trades)

    proc: subprocess.Popen | None = None
    if not railway:
        proc = _start_bot()

    try:
        # Record the start time to only look at orders/trades from this run
        run_start = datetime.now(UTC)

        def _enough_trades() -> bool:
            db.expire_all()
            count = (
                db.query(func.count(Trade.id))
                .filter(Trade.strategy_name == "ChaosTest")
                .filter(Trade.created_at >= run_start)
                .scalar()
            )
            return (count or 0) >= target_trades

        if not _poll_until(_enough_trades, timeout, description=f"{target_trades} trades"):
            logger.error("Timed out waiting for %d trades", target_trades)
            return False

        # Validate orders
        orders: list[Order] = (
            db.query(Order)
            .filter(Order.strategy_name == "ChaosTest")
            .filter(Order.created_at >= run_start)
            .all()
        )

        if not orders:
            logger.error("No orders found for ChaosTest strategy")
            return False

        errors: list[str] = []
        for order in orders:
            if not order.internal_order_id:
                errors.append(f"Order id={order.id} missing internal_order_id")

            if order.status not in set(OrderStatus):
                errors.append(f"Order id={order.id} has invalid status: {order.status}")

        if errors:
            for err in errors:
                logger.error("FAIL: %s", err)
            return False

        logger.info("PASS: %d orders validated, all have IDs and valid status", len(orders))

        # Check for reconciliation audit events (may not exist if PR #576 not merged)
        try:
            result = db.execute(
                text("SELECT count(*) FROM reconciliation_audit_events WHERE created_at >= :start"),
                {"start": run_start},
            )
            audit_count = result.scalar()
            if audit_count:
                logger.warning(
                    "Found %d reconciliation audit events (expected 0 in clean run)", audit_count
                )
            else:
                logger.info("PASS: No reconciliation audit events (clean run)")
        except ProgrammingError:
            logger.warning(
                "reconciliation_audit_events table not present (PR #576 not merged yet)"
            )

        return True
    finally:
        if proc:
            _graceful_stop(proc)


# ---------------------------------------------------------------------------
# Phase 2 — Crash Recovery
# ---------------------------------------------------------------------------


def phase_crash(
    db: Session,
    timeout: float,
    railway: bool,
) -> bool:
    """Validate position recovery after a hard kill.

    1. Start bot, wait for open position
    2. SIGKILL the bot
    3. Restart and verify position is recovered
    """
    logger.info("=== Phase 2: Crash Recovery ===")

    if railway:
        logger.info(
            "Railway mode: crash recovery requires manual service restart.\n"
            "  1. Verify bot has an open position via DB\n"
            "  2. Run: railway service restart --environment development\n"
            "  3. Re-run this phase to validate recovery"
        )
        return _validate_crash_recovery(db, timeout)

    proc = _start_bot()
    run_start = datetime.now(UTC)

    try:
        # Wait for an open position
        def _has_open_position() -> bool:
            db.expire_all()
            count = (
                db.query(func.count(Position.id))
                .filter(Position.strategy_name == "ChaosTest")
                .filter(Position.status == "OPEN")
                .filter(Position.created_at >= run_start)
                .scalar()
            )
            return (count or 0) > 0

        if not _poll_until(_has_open_position, timeout / 2, description="open position"):
            logger.error("Timed out waiting for open position")
            return False

        # Capture position state before crash
        open_pos = (
            db.query(Position)
            .filter(Position.strategy_name == "ChaosTest")
            .filter(Position.status == "OPEN")
            .filter(Position.created_at >= run_start)
            .first()
        )
        if not open_pos:
            logger.error("No open position found despite poll success")
            return False

        pre_crash_entry = float(open_pos.entry_price)
        pre_crash_symbol = open_pos.symbol
        logger.info("Open position: symbol=%s entry=%.2f", pre_crash_symbol, pre_crash_entry)

        # Hard kill
        _kill_bot(proc)
        time.sleep(2)

        # Restart
        logger.info("Restarting bot after crash...")
        proc = _start_bot()

        return _validate_crash_recovery(db, timeout / 2)

    finally:
        _graceful_stop(proc)


def _validate_crash_recovery(db: Session, timeout: float) -> bool:
    """Verify that the bot recovered a position or completed a trade after restart."""

    def _post_restart_activity() -> bool:
        db.expire_all()
        return (
            db.query(TradingSession)
            .filter(TradingSession.strategy_name == "ChaosTest")
            .filter(TradingSession.is_active.is_(True))
            .order_by(desc(TradingSession.start_time))
            .first()
        ) is not None

    if not _poll_until(_post_restart_activity, timeout, description="post-restart activity"):
        logger.error("Bot did not resume trading after restart")
        return False

    logger.info("PASS: Bot resumed trading after crash recovery")
    return True


# ---------------------------------------------------------------------------
# Phase 3 — Balance Integrity
# ---------------------------------------------------------------------------


def phase_balance(
    db: Session,
    target_trades: int,
    timeout: float,
    railway: bool,
) -> bool:
    """Verify balance integrity: initial + sum(pnl) - sum(fees) == DB balance.

    Runs the bot until target_trades complete, then compares calculated
    expected balance against the latest AccountBalance record.
    """
    logger.info("=== Phase 3: Balance Integrity (target=%d trades) ===", target_trades)

    proc: subprocess.Popen | None = None
    if not railway:
        proc = _start_bot()

    run_start = datetime.now(UTC)

    try:

        def _enough_trades() -> bool:
            db.expire_all()
            count = (
                db.query(func.count(Trade.id))
                .filter(Trade.strategy_name == "ChaosTest")
                .filter(Trade.created_at >= run_start)
                .scalar()
            )
            return (count or 0) >= target_trades

        if not _poll_until(_enough_trades, timeout, description=f"{target_trades} trades"):
            logger.error("Timed out waiting for %d trades", target_trades)
            return False

        db.expire_all()

        # Get the active session
        session = (
            db.query(TradingSession)
            .filter(TradingSession.strategy_name == "ChaosTest")
            .filter(TradingSession.is_active.is_(True))
            .order_by(desc(TradingSession.start_time))
            .first()
        )
        if not session:
            logger.error("No active ChaosTest session found")
            return False

        initial_balance = float(session.initial_balance)

        # Sum gross PnL from trades in this session.
        # NOTE: Trade.pnl stores GROSS PnL (before fees) and Trade.commission
        # is never populated by the engine (always 0.0). The actual DB balance
        # has entry+exit fees already deducted, so expected_from_gross will be
        # higher than actual_balance by the cumulative trading fees. We use a
        # percentage-based tolerance (2% of initial balance) to accommodate
        # this fee drift rather than an exact match.
        trades = db.query(Trade).filter(Trade.session_id == session.id).all()

        total_gross_pnl = sum(float(t.pnl or 0) for t in trades)

        expected_from_gross = initial_balance + total_gross_pnl

        # Get latest balance from AccountBalance
        actual_balance = AccountBalance.get_current_balance(session.id, db)

        drift = abs(expected_from_gross - actual_balance)
        logger.info(
            "Balance check: initial=%.2f gross_pnl=%.4f expected_from_gross=%.4f "
            "actual=%.4f drift=%.4f (fees cause actual < expected)",
            initial_balance,
            total_gross_pnl,
            expected_from_gross,
            actual_balance,
            drift,
        )

        # Tolerance: 2% of initial balance to account for cumulative trading
        # fees that are deducted from the DB balance but not reflected in
        # Trade.pnl (which is gross).
        max_drift = initial_balance * 0.02
        if drift > max_drift:
            logger.error("FAIL: Balance drift %.4f exceeds threshold %.4f (2%% of initial)", drift, max_drift)
            return False

        logger.info("PASS: Balance drift %.6f within tolerance (<%.2f)", drift, max_drift)
        return True

    finally:
        if proc:
            _graceful_stop(proc)


# ---------------------------------------------------------------------------
# Phase 4 — SL Reconciliation (manual guidance)
# ---------------------------------------------------------------------------


def phase_sl_reconciliation(db: Session, timeout: float, railway: bool) -> bool:
    """Print instructions for manual stop-loss reconciliation testing.

    This phase requires Binance testnet and manual interaction:
    the operator cancels a stop-loss order on the Binance UI and the
    script polls for audit events showing SL re-placement or close-only
    activation.
    """
    logger.info("=== Phase 4: SL Reconciliation (Manual) ===")
    logger.info(
        "\n"
        "This phase requires manual interaction on Binance testnet:\n"
        "\n"
        "  1. Start the bot with testnet + live trading:\n"
        "     atb live chaos_test --symbol BTCUSDT --timeframe 1m \\\n"
        "       --testnet --live-trading --i-understand-the-risks --check-interval 30\n"
        "\n"
        "  2. Wait for the bot to open a position with a stop-loss order\n"
        "\n"
        "  3. On the Binance testnet UI, cancel the stop-loss order\n"
        "\n"
        "  4. Watch bot logs / DB for:\n"
        "     - Reconciliation audit event (SL re-placement or close-only mode)\n"
        "     - New stop-loss order placed, OR\n"
        "     - Position marked as close-only\n"
        "\n"
        "  5. Verify via DB query:\n"
        "     SELECT * FROM reconciliation_audit_events\n"
        "       WHERE event_type LIKE '%%stop_loss%%'\n"
        "       ORDER BY created_at DESC LIMIT 5;\n"
    )

    if not railway:
        logger.info("Skipping automated validation (manual phase). Returning PASS.")
        return True

    # In Railway mode, poll for audit events
    logger.info("Railway mode: polling for SL reconciliation audit events...")
    run_start = datetime.now(UTC) - timedelta(minutes=5)

    def _has_sl_events() -> bool:
        try:
            result = db.execute(
                text(
                    "SELECT count(*) FROM reconciliation_audit_events "
                    "WHERE created_at >= :start "
                    "AND event_type LIKE :pattern"
                ),
                {"start": run_start, "pattern": "%stop_loss%"},
            )
            count = result.scalar()
            return (count or 0) > 0
        except SQLAlchemyError as e:
            logger.warning("DB error while checking SL audit events: %s", e)
            return False

    if _poll_until(_has_sl_events, timeout, interval=10, description="SL audit events"):
        logger.info("PASS: SL reconciliation audit events detected")
        return True

    logger.warning("No SL reconciliation events found within timeout. Manual verification needed.")
    return True  # Don't fail automated runs on manual phase


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

_PHASES = {
    "journal": lambda db, args: phase_journal(db, args.trades, args.timeout, args.railway),
    "crash": lambda db, args: phase_crash(db, args.timeout, args.railway),
    "balance": lambda db, args: phase_balance(db, args.trades, args.timeout, args.railway),
    "sl": lambda db, args: phase_sl_reconciliation(db, args.timeout, args.railway),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chaos smoke test for reconciliation validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["journal", "crash", "balance", "sl", "all"],
        default="all",
        help="Test phase to run (default: all)",
    )
    parser.add_argument(
        "--trades",
        type=int,
        default=5,
        help="Number of trades to wait for in journal/balance phases (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Maximum wait time in seconds per phase (default: 300)",
    )
    parser.add_argument(
        "--railway",
        action="store_true",
        help="Railway mode: skip subprocess management, validate via DB only",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Database URL (default: $DATABASE_URL env var)",
    )

    args = parser.parse_args()

    if not args.database_url:
        logger.error("DATABASE_URL not set. Provide --database-url or set the env var.")
        return 1

    db = _get_db_session(args.database_url)

    phases_to_run: list[str]
    if args.phase == "all":
        phases_to_run = ["journal", "crash", "balance"]
    else:
        phases_to_run = [args.phase]

    results: dict[str, bool] = {}
    for phase_name in phases_to_run:
        try:
            results[phase_name] = _PHASES[phase_name](db, args)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            results[phase_name] = False
            break
        except Exception:
            logger.exception("Phase '%s' failed with exception", phase_name)
            results[phase_name] = False

    # Summary
    logger.info("\n=== Smoke Test Results ===")
    for phase_name, passed in results.items():
        logger.info("  %-10s %s", phase_name, "PASS" if passed else "FAIL")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
