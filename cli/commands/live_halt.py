"""Manual kill-switch commands: `atb live-control halt` / `resume` (#922).

Writes the durable ``system_halt`` control flag in the TARGET environment's
database. The running engine polls the flag at the top of every trading-loop
iteration, so a halt takes effect within one iteration (typically seconds to a
few minutes, bounded by the engine's adaptive check interval) — no restart or
redeploy required. Semantics match FEATURE_ENTRY_PAUSE: new entries and
scale-ins are blocked; exits, stop-losses and reconciliation continue.

Fallback path (if the database write is impossible): flip
``FEATURE_ENTRY_PAUSE=true`` on the Railway service — note that a Railway
variable change triggers a redeploy, so that path has ~3 minutes of latency.

Both commands emit a ``system_events`` row (+ webhook alert when
``ALERT_WEBHOOK_URL`` is set) and print the resulting account state — open
positions and their protective stops — via read-only queries, in an
append-friendly one-line-per-fact format.
"""

from __future__ import annotations

import argparse
import getpass
import os
from datetime import UTC, datetime

from src.database.manager import DatabaseManager, SystemHaltStatus
from src.database.models import EventType

# Target environment -> env var holding its database URL. The RAILWAY_* vars
# are the same ones the heartbeat monitor uses (scripts/check_heartbeat.py);
# `development` targets the local DATABASE_URL for drills and tests.
ENV_DB_URL_VARS = {
    "production": "RAILWAY_PRODUCTION_DATABASE_URL",
    "staging": "RAILWAY_STAGING_DATABASE_URL",
    "development": "DATABASE_URL",
}


def halt_main(ns: argparse.Namespace) -> int:
    """Durably halt new risk in the target environment (entries + scale-ins)."""
    return _apply(ns, halt=True)


def resume_main(ns: argparse.Namespace) -> int:
    """Explicitly clear the manual halt in the target environment."""
    return _apply(ns, halt=False)


def _apply(ns: argparse.Namespace, *, halt: bool) -> int:
    """Shared halt/resume flow: resolve env -> flip flag -> announce -> print state."""
    action = "HALT" if halt else "RESUME"
    url_var = ENV_DB_URL_VARS[ns.env]
    url = os.environ.get(url_var)
    if not url:
        _line(f"ERROR: {url_var} is not set — cannot reach the {ns.env} database.")
        _line(
            "Fallback (requires restart, ~3 min): set FEATURE_ENTRY_PAUSE=true on the "
            f"{ns.env} Railway service (`railway variables --set FEATURE_ENTRY_PAUSE=true`)."
        )
        return 1

    try:
        db = DatabaseManager(url)
    except Exception as e:
        _line(f"ERROR: could not connect to the {ns.env} database: {e}")
        return 1

    current = db.get_system_halt()
    if current.active == halt:
        since = current.updated_at.isoformat() if current.updated_at else "unknown"
        state_word = "ACTIVE" if halt else "CLEAR"
        _line(
            f"system_halt already {state_word} (env={ns.env}, since={since}, "
            f"reason={current.reason or 'none'}, by={current.source or 'unknown'}) — no change"
        )
    else:
        source = f"cli:{getpass.getuser()}"
        status = db.set_system_halt(halt, reason=ns.reason, source=source)
        _line(
            f"system_halt {action} set (env={ns.env}, by={source}, "
            f"reason={ns.reason or 'none'})"
        )
        if halt:
            _line(
                "engine effect: entries + scale-ins BLOCKED within one trading-loop "
                "iteration; exits/stops/reconciliation UNAFFECTED"
            )
        else:
            _line("engine effect: entries + scale-ins re-enabled within one loop iteration")
        _announce(db, ns.env, halt=halt, status=status, source=source)

    _print_account_state(db, ns.env)
    _line(
        "verify enforcement: system_events error_code="
        + ("SYSTEM_HALT" if halt else "SYSTEM_HALT_CLEARED")
        + " (emitted by the engine when it honors the flag)"
    )
    return 0


def _announce(
    db: DatabaseManager,
    env: str,
    *,
    halt: bool,
    status: SystemHaltStatus,
    source: str,
) -> None:
    """Emit the CRITICAL system_event + webhook page for the command itself.

    The flag write above is the protective action; announcement failures are
    reported but never fail the command.
    """
    action = "HALT" if halt else "RESUME"
    message = (
        f"MANUAL {action} ISSUED for {env} by {source} "
        f"(reason: {status.reason or 'none'}). "
        + (
            "Entries and scale-ins will stop within one trading-loop iteration; "
            "exits and stop-losses remain active."
            if halt
            else "Entries and scale-ins will resume within one trading-loop iteration."
        )
    )
    delivered = _send_webhook_alert(message)
    if not delivered:
        _line(
            "WARNING: no alert webhook delivery (ALERT_WEBHOOK_URL unset or POST failed) "
            "— page the operator manually"
        )
    try:
        db.log_event(
            event_type=EventType.ALERT,
            message=message,
            severity="critical" if halt else "warning",
            component="ops",
            error_code="SYSTEM_HALT_COMMAND" if halt else "SYSTEM_RESUME_COMMAND",
            alert_sent=delivered,
            alert_method="webhook" if delivered else None,
        )
    except Exception as e:
        _line(f"WARNING: system_events write failed ({e}) — halt flag itself IS set")


def _send_webhook_alert(message: str) -> bool:
    """POST the operator page; same payload shape as the engine's _send_alert.

    Returns True only on a delivered (2xx) page so `alert_sent` reflects
    reality.
    """
    url = os.environ.get("ALERT_WEBHOOK_URL")
    if not url:
        return False
    try:
        import requests  # type: ignore[import-untyped]  # types-requests not installed

        payload = {
            "text": f"🤖 Trading Bot: {message}",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        _line(f"WARNING: alert webhook POST failed: {e}")
        return False


def _print_account_state(db: DatabaseManager, env: str) -> None:
    """Read-only snapshot of what the account holds and how it is protected."""
    try:
        session_id = db.get_active_session_id()
        _line(f"active session: {session_id if session_id is not None else 'none'} (env={env})")
        positions = db.get_active_positions()
    except Exception as e:
        _line(f"WARNING: could not read account state: {e}")
        return

    _line(f"open positions: {len(positions)}")
    for p in positions:
        stop_price = p.get("stop_loss")
        stop_order = p.get("stop_loss_order_id")
        if stop_price is None and not stop_order:
            protection = "NO STOP — position is UNPROTECTED, review immediately"
        else:
            protection = (
                f"sl={stop_price if stop_price is not None else 'unknown'}"
                f" (order {stop_order or 'not placed'})"
            )
        _line(
            f"  {p.get('symbol')} {p.get('side')} qty={p.get('quantity')} "
            f"entry={p.get('entry_price')} {protection} "
            f"tp={p.get('take_profit')} upnl={p.get('unrealized_pnl')}"
        )


def _line(text: str) -> None:
    """Append-friendly output: one UTC-timestamped fact per line."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] {text}")
