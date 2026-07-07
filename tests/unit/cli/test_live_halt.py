"""Tests for `atb live-control halt` / `resume` — the manual kill-switch (#922).

The commands write the durable ``system_halt`` DB flag for the TARGET
environment (resolved from RAILWAY_*_DATABASE_URL env vars), emit a CRITICAL
``system_events`` row + webhook alert, and print the resulting account state
(open positions + their stops) via read-only queries.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from cli.commands.live_halt import halt_main, resume_main
from src.database.manager import SystemHaltStatus

pytestmark = [pytest.mark.unit, pytest.mark.fast]

_STAGING_URL_VAR = "RAILWAY_STAGING_DATABASE_URL"
_PRODUCTION_URL_VAR = "RAILWAY_PRODUCTION_DATABASE_URL"


def _ns(control_cmd: str, env: str = "staging", reason: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(control_cmd=control_cmd, env=env, reason=reason)


def _halt_status(active: bool, reason: str | None = None) -> SystemHaltStatus:
    return SystemHaltStatus(
        active=active,
        reason=reason,
        source="cli:test",
        updated_at=datetime(2026, 7, 7, 12, 0, tzinfo=UTC),
    )


def _position(symbol="ETHUSDT", stop_loss=2400.0, stop_loss_order_id="sl-1"):
    return {
        "symbol": symbol,
        "side": "LONG",
        "quantity": 0.03,
        "entry_price": 2450.0,
        "stop_loss": stop_loss,
        "stop_loss_order_id": stop_loss_order_id,
        "take_profit": 2550.0,
        "unrealized_pnl": -1.2,
    }


@pytest.fixture
def db(monkeypatch):
    """Patched DatabaseManager wired to a healthy staging environment."""
    monkeypatch.setenv(_STAGING_URL_VAR, "postgresql://staging.example/db")
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    mock_db = MagicMock()
    mock_db.get_system_halt.return_value = _halt_status(False)
    mock_db.set_system_halt.return_value = _halt_status(True, "drill")
    mock_db.get_active_session_id.return_value = 7
    mock_db.get_active_positions.return_value = [_position()]
    with patch("cli.commands.live_halt.DatabaseManager", return_value=mock_db):
        yield mock_db


class TestEnvResolution:
    def test_missing_env_url_var_errors_without_touching_db(self, monkeypatch, capsys):
        monkeypatch.delenv(_PRODUCTION_URL_VAR, raising=False)
        with patch("cli.commands.live_halt.DatabaseManager") as manager_cls:
            result = halt_main(_ns("halt", env="production", reason="drill"))

        assert result == 1
        manager_cls.assert_not_called()
        assert _PRODUCTION_URL_VAR in capsys.readouterr().out

    def test_db_connection_failure_returns_error(self, monkeypatch, capsys):
        monkeypatch.setenv(_STAGING_URL_VAR, "postgresql://staging.example/db")
        with patch(
            "cli.commands.live_halt.DatabaseManager",
            side_effect=ConnectionError("unreachable"),
        ):
            result = halt_main(_ns("halt", reason="drill"))

        assert result == 1
        assert "unreachable" in capsys.readouterr().out


class TestHalt:
    def test_halt_sets_flag_and_emits_critical_event(self, db, capsys):
        result = halt_main(_ns("halt", reason="kill-switch drill"))

        assert result == 0
        args, kwargs = db.set_system_halt.call_args
        assert args == (True,)
        assert kwargs["reason"] == "kill-switch drill"
        assert kwargs["source"].startswith("cli:")
        event_kwargs = db.log_event.call_args.kwargs
        assert event_kwargs["severity"] == "critical"
        assert event_kwargs["error_code"] == "SYSTEM_HALT_COMMAND"
        out = capsys.readouterr().out
        assert "HALT" in out
        assert "kill-switch drill" in out

    def test_halt_prints_open_positions_and_stops(self, db, capsys):
        halt_main(_ns("halt", reason="drill"))

        out = capsys.readouterr().out
        assert "ETHUSDT" in out
        assert "2400" in out  # stop price
        assert "sl-1" in out  # stop-loss exchange order id

    def test_halt_flags_position_without_stop(self, db, capsys):
        db.get_active_positions.return_value = [_position(stop_loss=None, stop_loss_order_id=None)]

        halt_main(_ns("halt", reason="drill"))

        assert "NO STOP" in capsys.readouterr().out

    def test_halt_is_idempotent_when_already_active(self, db, capsys):
        db.get_system_halt.return_value = _halt_status(True, "earlier drill")

        result = halt_main(_ns("halt", reason="again"))

        assert result == 0
        db.set_system_halt.assert_not_called()
        db.log_event.assert_not_called()
        assert "already" in capsys.readouterr().out.lower()

    def test_halt_posts_webhook_when_configured(self, db, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example/alert")
        response = MagicMock()
        response.raise_for_status.return_value = None
        with patch("requests.post", return_value=response) as post:
            halt_main(_ns("halt", reason="drill"))

        post.assert_called_once()
        assert post.call_args.args[0] == "https://hooks.example/alert"
        assert db.log_event.call_args.kwargs["alert_sent"] is True

    def test_halt_without_webhook_records_undelivered_alert(self, db, capsys):
        halt_main(_ns("halt", reason="drill"))

        assert db.log_event.call_args.kwargs["alert_sent"] is False
        assert "no alert webhook" in capsys.readouterr().out.lower()

    def test_halt_survives_event_write_failure(self, db, capsys):
        """The flag write is the protective action; observability must not fail it."""
        db.log_event.side_effect = RuntimeError("events table locked")

        result = halt_main(_ns("halt", reason="drill"))

        assert result == 0
        db.set_system_halt.assert_called_once()

    def test_halt_echoes_masked_target_host_before_mutation(self, db, monkeypatch, capsys):
        """The resolved DB host is printed (credentials masked) BEFORE the flag
        write, so a mis-set RAILWAY_*_DATABASE_URL is visible to the operator."""
        monkeypatch.setenv(
            _STAGING_URL_VAR, "postgresql://bot:s3cretpw@db.staging.rlwy.net:5432/railway"
        )

        halt_main(_ns("halt", reason="drill"))

        out = capsys.readouterr().out
        assert "db.staging.rlwy.net:5432/railway" in out
        assert "s3cretpw" not in out
        assert "bot:" not in out
        target_line = next(i for i, line in enumerate(out.splitlines()) if "target:" in line)
        mutation_line = next(i for i, line in enumerate(out.splitlines()) if "HALT set" in line)
        assert target_line < mutation_line

    def test_halt_source_survives_getpass_failure(self, db, monkeypatch):
        """Operator attribution must never block the flag write (#929 review)."""
        monkeypatch.delenv("USER", raising=False)
        with patch("cli.commands.live_halt.getpass.getuser", side_effect=OSError("no passwd")):
            result = halt_main(_ns("halt", reason="drill"))

        assert result == 0
        assert db.set_system_halt.call_args.kwargs["source"] == "cli:unknown"

    def test_halt_source_falls_back_to_user_env(self, db, monkeypatch):
        monkeypatch.setenv("USER", "ops-oncall")
        with patch("cli.commands.live_halt.getpass.getuser", side_effect=KeyError("uid")):
            halt_main(_ns("halt", reason="drill"))

        assert db.set_system_halt.call_args.kwargs["source"] == "cli:ops-oncall"


class TestResume:
    def test_resume_clears_flag_and_emits_event(self, db, capsys):
        db.get_system_halt.return_value = _halt_status(True, "drill")
        db.set_system_halt.return_value = _halt_status(False, "drill complete")

        result = resume_main(_ns("resume", reason="drill complete"))

        assert result == 0
        args, kwargs = db.set_system_halt.call_args
        assert args == (False,)
        assert kwargs["reason"] == "drill complete"
        event_kwargs = db.log_event.call_args.kwargs
        assert event_kwargs["severity"] == "warning"
        assert event_kwargs["error_code"] == "SYSTEM_RESUME_COMMAND"
        assert "RESUME" in capsys.readouterr().out

    def test_resume_is_idempotent_when_not_halted(self, db, capsys):
        db.get_system_halt.return_value = _halt_status(False)

        result = resume_main(_ns("resume"))

        assert result == 0
        db.set_system_halt.assert_not_called()
        db.log_event.assert_not_called()
        assert "already" in capsys.readouterr().out.lower()

    def test_resume_prints_account_state(self, db, capsys):
        db.get_system_halt.return_value = _halt_status(True, "drill")
        db.set_system_halt.return_value = _halt_status(False)

        resume_main(_ns("resume"))

        assert "ETHUSDT" in capsys.readouterr().out
