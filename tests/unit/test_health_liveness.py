"""Tests for the deep /health loop-liveness check (#627).

Before this, /health returned a hardcoded "healthy" with no reference to the
engine, so a zombie process (HTTP server up, trading loop dead) reported healthy
for 12 days during the 2026-05-19 outage. /health now reflects loop liveness.
"""

from unittest.mock import patch

import pytest

from cli.commands.live_health import _health_payload
from src.infrastructure import liveness


@pytest.fixture(autouse=True)
def _reset_liveness():
    """Isolate each test from the process-global heartbeat (order-independent)."""
    liveness.reset()
    yield
    liveness.reset()


@pytest.mark.fast
def test_liveness_beat_and_reset():
    liveness.reset()
    assert liveness.seconds_since_beat() is None
    liveness.beat()
    age = liveness.seconds_since_beat()
    assert age is not None and 0 <= age < 1.0
    liveness.reset()
    assert liveness.seconds_since_beat() is None


@pytest.mark.fast
def test_health_starting_when_loop_not_beaten():
    """Before the loop's first beat, /health is healthy+starting (let the deploy come up)."""
    liveness.reset()
    code, body = _health_payload()
    assert code == 200
    assert body["status"] == "healthy"
    assert body["loop"] == "starting"


@pytest.mark.fast
def test_health_ok_when_loop_fresh():
    liveness.beat()
    code, body = _health_payload()
    assert code == 200
    assert body["status"] == "healthy"
    assert body["loop_heartbeat_age_seconds"] < 5


@pytest.mark.fast
def test_health_unhealthy_when_loop_stale():
    """A stalled loop makes /health return 503 so the zombie is detectable."""
    with patch("src.infrastructure.liveness.seconds_since_beat", return_value=10_000.0):
        code, body = _health_payload()
    assert code == 503
    assert body["status"] == "unhealthy"
    assert "stalled" in body["reason"]
