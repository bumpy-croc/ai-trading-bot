"""The 0014 migration's legacy mapping must stay in step with the Python one (#1115).

The view exists so historical rows can be read with a category without rewriting them.
If the two mappings drift, a SQL decomposition and a Python one disagree about the same
row — exactly the class of silent divergence the taxonomy was introduced to end.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.trading.exit_reason import LEGACY_EXIT_REASON_CATEGORIES, ExitReason

MIGRATION = (
    Path(__file__).resolve().parents[3]
    / "migrations"
    / "versions"
    / "0014_add_trade_exit_category.py"
)


@pytest.fixture(scope="module")
def view_sql() -> str:
    return MIGRATION.read_text()


@pytest.mark.fast
def test_migration_exists(view_sql: str) -> None:
    assert "CREATE OR REPLACE VIEW v_trades_exit_category" in view_sql
    assert "exit_category_inferred" in view_sql
    assert "exit_category_resolved" in view_sql


@pytest.mark.fast
def test_view_does_not_rewrite_history(view_sql: str) -> None:
    """The migration must never UPDATE trades — inference belongs at read time."""
    assert not re.search(r"\bUPDATE\s+trades\b", view_sql, re.IGNORECASE)


@pytest.mark.fast
@pytest.mark.parametrize(("reason", "category"), sorted(LEGACY_EXIT_REASON_CATEGORIES.items()))
def test_every_python_mapping_appears_in_the_view(
    view_sql: str, reason: str, category: ExitReason
) -> None:
    """Each legacy string and its target category are both present in the view SQL."""
    assert f"'{reason}'" in view_sql
    assert f"'{category.value}'" in view_sql


@pytest.mark.fast
def test_view_only_emits_known_categories(view_sql: str) -> None:
    """No THEN branch may produce a token that is not an ExitReason value."""
    emitted = set(re.findall(r"THEN\s+'([a-z_]+)'", view_sql))
    assert emitted
    assert emitted <= {member.value for member in ExitReason}
