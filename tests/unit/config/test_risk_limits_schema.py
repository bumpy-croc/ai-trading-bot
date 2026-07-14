"""CI schema gate: the committed `src/config/risk-limits.json` must load
through the real loader.

This is design §3.9.1 of the single-source risk-config proposal: any malformed
Board edit to the ratified limits file fails every PR before it can reach a
deploy. It deliberately uses the repo's actual file and the loader's actual
default-path resolution — no fixtures.
"""

from __future__ import annotations

import pytest

from src.config.paths import get_project_root
from src.config.risk_limits import RiskLimits, get_risk_limits, load_risk_limits

pytestmark = pytest.mark.fast

REAL_FILE = get_project_root() / "src" / "config" / "risk-limits.json"


def test_ratified_limits_file_exists():
    assert REAL_FILE.is_file(), (
        f"Board-ratified limits file missing at {REAL_FILE} — engines will "
        "fail closed at boot once consumers are wired (design §3.3)."
    )


def test_ratified_limits_file_passes_full_validation():
    limits = load_risk_limits(REAL_FILE)
    assert isinstance(limits, RiskLimits)
    assert limits.schema_version == "1"


def test_default_path_resolution_finds_the_ratified_file():
    """`get_risk_limits()` (the accessor all consumers will use) must resolve
    to the ratified file without an explicit path."""
    get_risk_limits.cache_clear()
    try:
        limits = get_risk_limits()
        assert limits == load_risk_limits(REAL_FILE)
    finally:
        get_risk_limits.cache_clear()


def test_ratified_values_spot_check():
    """Anchor the Board's signed numbers: a silent edit to a headline limit
    must show up as a failing diff in review, not just a green reload."""
    limits = load_risk_limits(REAL_FILE)
    assert limits.portfolio.max_drawdown_pct == 0.20
    assert limits.position.max_position_size_pct == 0.20
    assert limits.position.base_risk_per_trade_pct == 0.02
    assert limits.position.max_risk_per_trade_pct == 0.03
    assert limits.kill_switch.authorized_actors == ("human",)
