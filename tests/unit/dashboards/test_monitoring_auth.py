"""Tests for the monitoring dashboard authentication guard.

These lock in the security fix that protects state-changing / data-leaking
endpoints (balance adjustment, fix-positions, config) on the monitoring
dashboard, which has no login system and may be bound to all interfaces.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from flask import Flask, jsonify

from src.dashboards.monitoring.dashboard import _dashboard_auth_required, _is_production_env


@pytest.fixture()
def client():
    app = Flask(__name__)

    @app.route("/protected", methods=["POST"])
    @_dashboard_auth_required
    def protected():
        return jsonify({"ok": True})

    return app.test_client()


@pytest.mark.fast
def test_production_without_token_is_disabled(client):
    """Fail closed: in production with no token, the endpoint returns 403."""
    with patch.dict(os.environ, {"ENV": "production"}, clear=False):
        os.environ.pop("MONITORING_DASHBOARD_TOKEN", None)
        resp = client.post("/protected")
        assert resp.status_code == 403
        assert resp.get_json()["success"] is False


@pytest.mark.fast
def test_token_required_when_configured(client):
    """When a token is configured, a missing/wrong token yields 401."""
    with patch.dict(os.environ, {"MONITORING_DASHBOARD_TOKEN": "s3cret"}, clear=False):
        assert client.post("/protected").status_code == 401
        assert client.post("/protected", headers={"X-Dashboard-Token": "wrong"}).status_code == 401


@pytest.mark.fast
def test_correct_token_allows_access(client):
    """A correct token (header or bearer) allows the request through."""
    with patch.dict(os.environ, {"MONITORING_DASHBOARD_TOKEN": "s3cret"}, clear=False):
        ok_header = client.post("/protected", headers={"X-Dashboard-Token": "s3cret"})
        assert ok_header.status_code == 200
        assert ok_header.get_json()["ok"] is True

        ok_bearer = client.post("/protected", headers={"Authorization": "Bearer s3cret"})
        assert ok_bearer.status_code == 200


@pytest.mark.fast
def test_dev_without_token_allowed(client):
    """In an explicit dev/test env, no token is required (warn-and-allow)."""
    with patch.dict(os.environ, {"ENV": "development"}, clear=False):
        os.environ.pop("MONITORING_DASHBOARD_TOKEN", None)
        assert client.post("/protected").status_code == 200


@pytest.mark.fast
def test_is_production_env_fails_closed():
    """Unset ENV/FLASK_ENV must be treated as production."""
    with patch.dict(os.environ, {}, clear=True):
        assert _is_production_env() is True
    with patch.dict(os.environ, {"ENV": "test"}, clear=True):
        assert _is_production_env() is False
