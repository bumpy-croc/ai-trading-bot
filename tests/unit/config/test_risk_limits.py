"""Unit tests for the risk-limits loader (`src/config/risk_limits.py`).

Every validation rule has at least one failing case, mirroring the strictness
contract of the single-source risk-config design: pinned schema version,
unknown/missing keys rejected, typed range checks, and cross-field invariants.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from src.config.risk_limits import (
    EscalationPolicy,
    KillSwitchPolicy,
    OperationalLimits,
    PortfolioLimits,
    PositionLimits,
    RiskLimits,
    RiskLimitsError,
    StopLimits,
    get_risk_limits,
    load_risk_limits,
)

pytestmark = pytest.mark.fast


def make_limits_dict() -> dict[str, Any]:
    """A valid limits document matching the ratified schema (version 1)."""
    return {
        "$schema_version": "1",
        "$owner": "human_board",
        "$source_of_truth_note": "test fixture",
        "$last_reviewed": "2026-07-05",
        "$last_reviewer": "alexflorisca",
        "portfolio": {
            "max_drawdown_pct": 0.20,
            "max_daily_risk_pct": 0.06,
            "max_correlated_exposure_pct": 0.15,
            "dynamic_drawdown_thresholds_pct": [0.05, 0.10, 0.15],
            "dynamic_risk_reduction_factors": [0.8, 0.6, 0.4],
        },
        "position": {
            "max_position_size_pct": 0.20,
            "base_risk_per_trade_pct": 0.02,
            "max_risk_per_trade_pct": 0.03,
            "max_leverage": 3.0,
            "kelly_max_fraction": 0.20,
            "large_single_position_threshold_pct": 0.20,
        },
        "stops": {
            "default_stop_loss_pct": 0.05,
            "min_stop_loss_pct": 0.01,
            "max_stop_loss_pct": 0.20,
            "default_take_profit_pct": 0.04,
            "fallback_trailing_pct": 0.01,
        },
        "operational": {
            "max_consecutive_errors": 10,
            "max_holding_hours": 336,
            "max_filled_price_deviation_pct": 0.50,
            "balance_discrepancy_warning_pct": 0.01,
            "reconciliation_balance_critical_pct": 0.05,
        },
        "escalation": {
            "warning_at_pct_of_limit": 0.50,
            "critical_at_pct_of_limit": 0.80,
            "breach_action": "halt_new_entries_and_page_human",
        },
        "kill_switch": {
            "authorized_actors": ["human"],
            "auto_trigger_conditions": [
                "db_memory_divergence_detected",
                "duplicate_order_storm",
            ],
            "manual_trigger_command": "atb live-control halt",
        },
    }


def write_limits(tmp_path: Path, data: dict[str, Any] | str) -> Path:
    """Write a limits document (dict or raw text) to a temp file."""
    path = tmp_path / "risk-limits.json"
    text = data if isinstance(data, str) else json.dumps(data)
    path.write_text(text, encoding="utf-8")
    return path


def load_mutated(tmp_path: Path, mutate) -> RiskLimits:
    """Apply ``mutate`` to a valid document, write it, and load it."""
    data = make_limits_dict()
    mutate(data)
    return load_risk_limits(write_limits(tmp_path, data))


class TestValidDocument:
    """Loading a well-formed document produces typed, frozen values."""

    def test_loads_all_sections_with_exact_values(self, tmp_path):
        limits = load_risk_limits(write_limits(tmp_path, make_limits_dict()))

        assert limits.schema_version == "1"
        assert limits.portfolio == PortfolioLimits(
            max_drawdown_pct=0.20,
            max_daily_risk_pct=0.06,
            max_correlated_exposure_pct=0.15,
            dynamic_drawdown_thresholds_pct=(0.05, 0.10, 0.15),
            dynamic_risk_reduction_factors=(0.8, 0.6, 0.4),
        )
        assert limits.position == PositionLimits(
            max_position_size_pct=0.20,
            base_risk_per_trade_pct=0.02,
            max_risk_per_trade_pct=0.03,
            max_leverage=3.0,
            kelly_max_fraction=0.20,
            large_single_position_threshold_pct=0.20,
        )
        assert limits.stops == StopLimits(
            default_stop_loss_pct=0.05,
            min_stop_loss_pct=0.01,
            max_stop_loss_pct=0.20,
            default_take_profit_pct=0.04,
            fallback_trailing_pct=0.01,
        )
        assert limits.operational == OperationalLimits(
            max_consecutive_errors=10,
            max_holding_hours=336,
            max_filled_price_deviation_pct=0.50,
            balance_discrepancy_warning_pct=0.01,
            reconciliation_balance_critical_pct=0.05,
        )
        assert limits.escalation == EscalationPolicy(
            warning_at_pct_of_limit=0.50,
            critical_at_pct_of_limit=0.80,
            breach_action="halt_new_entries_and_page_human",
        )
        assert limits.kill_switch == KillSwitchPolicy(
            authorized_actors=("human",),
            auto_trigger_conditions=(
                "db_memory_divergence_detected",
                "duplicate_order_storm",
            ),
            manual_trigger_command="atb live-control halt",
        )

    def test_dataclasses_are_frozen(self, tmp_path):
        limits = load_risk_limits(write_limits(tmp_path, make_limits_dict()))

        with pytest.raises(dataclasses.FrozenInstanceError):
            limits.portfolio.max_drawdown_pct = 0.99  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            limits.position = None  # type: ignore[misc]

    def test_integer_valued_floats_are_accepted(self, tmp_path):
        """JSON integers are valid where floats are expected (e.g. leverage 3)."""

        def mutate(data):
            data["position"]["max_leverage"] = 3

        limits = load_mutated(tmp_path, mutate)
        assert limits.position.max_leverage == 3.0
        assert isinstance(limits.position.max_leverage, float)

    def test_empty_dynamic_arrays_are_accepted(self, tmp_path):
        """Empty tier arrays are valid policy (dynamic de-risking disabled)."""

        def mutate(data):
            data["portfolio"]["dynamic_drawdown_thresholds_pct"] = []
            data["portfolio"]["dynamic_risk_reduction_factors"] = []

        limits = load_mutated(tmp_path, mutate)
        assert limits.portfolio.dynamic_drawdown_thresholds_pct == ()
        assert limits.portfolio.dynamic_risk_reduction_factors == ()


class TestFileLevelFailures:
    """Missing/malformed files fail closed with RiskLimitsError."""

    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does-not-exist.json"
        with pytest.raises(RiskLimitsError, match="not found"):
            load_risk_limits(missing)

    def test_malformed_json_raises(self, tmp_path):
        path = write_limits(tmp_path, "{not valid json")
        with pytest.raises(RiskLimitsError, match="invalid JSON"):
            load_risk_limits(path)

    def test_non_object_top_level_raises(self, tmp_path):
        path = write_limits(tmp_path, "[1, 2, 3]")
        with pytest.raises(RiskLimitsError, match="JSON object"):
            load_risk_limits(path)


class TestSchemaVersion:
    def test_missing_schema_version_raises(self, tmp_path):
        with pytest.raises(RiskLimitsError, match=r"\$schema_version"):
            load_mutated(tmp_path, lambda d: d.pop("$schema_version"))

    def test_unsupported_schema_version_raises(self, tmp_path):
        def mutate(data):
            data["$schema_version"] = "2"

        with pytest.raises(RiskLimitsError, match=r"\$schema_version"):
            load_mutated(tmp_path, mutate)

    def test_non_string_schema_version_raises(self, tmp_path):
        def mutate(data):
            data["$schema_version"] = 1

        with pytest.raises(RiskLimitsError, match=r"\$schema_version"):
            load_mutated(tmp_path, mutate)


class TestKeyStrictness:
    """Unknown keys rejected; missing keys rejected — at every level."""

    def test_unknown_top_level_key_raises(self, tmp_path):
        def mutate(data):
            data["surprise_section"] = {}

        with pytest.raises(RiskLimitsError, match="surprise_section"):
            load_mutated(tmp_path, mutate)

    def test_unknown_metadata_key_raises(self, tmp_path):
        def mutate(data):
            data["$unexpected_note"] = "hello"

        with pytest.raises(RiskLimitsError, match=r"\$unexpected_note"):
            load_mutated(tmp_path, mutate)

    @pytest.mark.parametrize(
        "section",
        ["portfolio", "position", "stops", "operational", "escalation", "kill_switch"],
    )
    def test_missing_section_raises(self, tmp_path, section):
        with pytest.raises(RiskLimitsError, match=section):
            load_mutated(tmp_path, lambda d: d.pop(section))

    def test_section_must_be_object(self, tmp_path):
        def mutate(data):
            data["portfolio"] = [1, 2]

        with pytest.raises(RiskLimitsError, match="portfolio"):
            load_mutated(tmp_path, mutate)

    def test_unknown_key_in_section_raises(self, tmp_path):
        def mutate(data):
            data["position"]["max_positon_size_pct"] = 0.2  # typo'd key

        with pytest.raises(RiskLimitsError, match="max_positon_size_pct"):
            load_mutated(tmp_path, mutate)

    @pytest.mark.parametrize(
        ("section", "key"),
        [
            ("portfolio", "max_drawdown_pct"),
            ("portfolio", "dynamic_drawdown_thresholds_pct"),
            ("position", "max_position_size_pct"),
            ("stops", "min_stop_loss_pct"),
            ("operational", "max_holding_hours"),
            ("escalation", "breach_action"),
            ("kill_switch", "manual_trigger_command"),
        ],
    )
    def test_missing_key_in_section_raises(self, tmp_path, section, key):
        with pytest.raises(RiskLimitsError, match=key):
            load_mutated(tmp_path, lambda d: d[section].pop(key))


class TestTypeAndRangeChecks:
    @pytest.mark.parametrize("bad_value", ["0.2", None, True, [0.2], {"v": 0.2}])
    def test_fraction_must_be_number(self, tmp_path, bad_value):
        def mutate(data):
            data["position"]["max_position_size_pct"] = bad_value

        with pytest.raises(RiskLimitsError, match="max_position_size_pct"):
            load_mutated(tmp_path, mutate)

    @pytest.mark.parametrize("bad_value", [0, 0.0, -0.1, 1.0001, 5])
    def test_fraction_must_be_in_unit_interval(self, tmp_path, bad_value):
        def mutate(data):
            data["portfolio"]["max_daily_risk_pct"] = bad_value

        with pytest.raises(RiskLimitsError, match="max_daily_risk_pct"):
            load_mutated(tmp_path, mutate)

    def test_fraction_of_exactly_one_is_accepted(self, tmp_path):
        def mutate(data):
            data["operational"]["max_filled_price_deviation_pct"] = 1.0

        limits = load_mutated(tmp_path, mutate)
        assert limits.operational.max_filled_price_deviation_pct == 1.0

    def test_non_finite_number_raises(self, tmp_path):
        """Python's json module happily parses Infinity/NaN — reject them."""
        text = json.dumps(make_limits_dict()).replace("3.0", "Infinity")
        with pytest.raises(RiskLimitsError, match="max_leverage"):
            load_risk_limits(write_limits(tmp_path, text))

    @pytest.mark.parametrize("bad_value", [0, -3, 2.5, "10", True, None])
    def test_positive_int_fields_reject_bad_values(self, tmp_path, bad_value):
        def mutate(data):
            data["operational"]["max_consecutive_errors"] = bad_value

        with pytest.raises(RiskLimitsError, match="max_consecutive_errors"):
            load_mutated(tmp_path, mutate)

    @pytest.mark.parametrize("bad_value", [0, -1.0, "3", None])
    def test_max_leverage_must_be_positive_number(self, tmp_path, bad_value):
        def mutate(data):
            data["position"]["max_leverage"] = bad_value

        with pytest.raises(RiskLimitsError, match="max_leverage"):
            load_mutated(tmp_path, mutate)

    @pytest.mark.parametrize("bad_value", ["", None, 42, ["a"]])
    def test_breach_action_must_be_non_empty_string(self, tmp_path, bad_value):
        def mutate(data):
            data["escalation"]["breach_action"] = bad_value

        with pytest.raises(RiskLimitsError, match="breach_action"):
            load_mutated(tmp_path, mutate)

    @pytest.mark.parametrize("bad_value", [[], ["human", ""], ["human", 3], "human", None])
    def test_authorized_actors_must_be_non_empty_string_list(self, tmp_path, bad_value):
        def mutate(data):
            data["kill_switch"]["authorized_actors"] = bad_value

        with pytest.raises(RiskLimitsError, match="authorized_actors"):
            load_mutated(tmp_path, mutate)

    @pytest.mark.parametrize("bad_value", [[""], [1], "storm", None])
    def test_auto_trigger_conditions_must_be_string_list(self, tmp_path, bad_value):
        def mutate(data):
            data["kill_switch"]["auto_trigger_conditions"] = bad_value

        with pytest.raises(RiskLimitsError, match="auto_trigger_conditions"):
            load_mutated(tmp_path, mutate)

    @pytest.mark.parametrize("bad_value", [[0.05, "0.1"], [0.05, 0.0], [0.05, 1.5], 0.05, None])
    def test_dynamic_arrays_must_be_fraction_lists(self, tmp_path, bad_value):
        def mutate(data):
            data["portfolio"]["dynamic_drawdown_thresholds_pct"] = bad_value

        with pytest.raises(RiskLimitsError, match="dynamic_drawdown_thresholds_pct"):
            load_mutated(tmp_path, mutate)


class TestCrossFieldInvariants:
    def test_base_risk_above_max_risk_raises(self, tmp_path):
        def mutate(data):
            data["position"]["base_risk_per_trade_pct"] = 0.05
            data["position"]["max_risk_per_trade_pct"] = 0.03

        with pytest.raises(RiskLimitsError, match="base_risk_per_trade_pct"):
            load_mutated(tmp_path, mutate)

    def test_min_stop_above_default_stop_raises(self, tmp_path):
        def mutate(data):
            data["stops"]["min_stop_loss_pct"] = 0.06
            data["stops"]["default_stop_loss_pct"] = 0.05

        with pytest.raises(RiskLimitsError, match="min_stop_loss_pct"):
            load_mutated(tmp_path, mutate)

    def test_default_stop_above_max_stop_raises(self, tmp_path):
        def mutate(data):
            data["stops"]["default_stop_loss_pct"] = 0.25
            data["stops"]["max_stop_loss_pct"] = 0.20

        with pytest.raises(RiskLimitsError, match="default_stop_loss_pct"):
            load_mutated(tmp_path, mutate)

    def test_dynamic_arrays_length_mismatch_raises(self, tmp_path):
        def mutate(data):
            data["portfolio"]["dynamic_risk_reduction_factors"] = [0.8, 0.6]

        with pytest.raises(RiskLimitsError, match="equal length"):
            load_mutated(tmp_path, mutate)

    @pytest.mark.parametrize(
        "thresholds",
        [[0.10, 0.05, 0.15], [0.05, 0.05, 0.15]],
    )
    def test_thresholds_not_strictly_ascending_raises(self, tmp_path, thresholds):
        def mutate(data):
            data["portfolio"]["dynamic_drawdown_thresholds_pct"] = thresholds

        with pytest.raises(RiskLimitsError, match="ascending"):
            load_mutated(tmp_path, mutate)

    def test_threshold_at_or_above_max_drawdown_raises(self, tmp_path):
        """The dead-tier invariant: a tier >= max_drawdown_pct can never fire
        because the drawdown guard latches close-only first."""

        def mutate(data):
            data["portfolio"]["dynamic_drawdown_thresholds_pct"] = [0.05, 0.10, 0.20]

        with pytest.raises(RiskLimitsError, match="max_drawdown_pct"):
            load_mutated(tmp_path, mutate)

    def test_escalation_warning_above_critical_raises(self, tmp_path):
        def mutate(data):
            data["escalation"]["warning_at_pct_of_limit"] = 0.9
            data["escalation"]["critical_at_pct_of_limit"] = 0.8

        with pytest.raises(RiskLimitsError, match="warning_at_pct_of_limit"):
            load_mutated(tmp_path, mutate)

    def test_balance_warning_above_critical_raises(self, tmp_path):
        def mutate(data):
            data["operational"]["balance_discrepancy_warning_pct"] = 0.10
            data["operational"]["reconciliation_balance_critical_pct"] = 0.05

        with pytest.raises(RiskLimitsError, match="balance_discrepancy_warning_pct"):
            load_mutated(tmp_path, mutate)


class TestGetRiskLimitsCache:
    def test_get_risk_limits_is_cached(self, tmp_path, monkeypatch):
        # Point the default path at a temp copy so the test never depends on
        # (or mutates) the real repo file.
        import src.config.risk_limits as risk_limits_module

        path = write_limits(tmp_path, make_limits_dict())
        monkeypatch.setattr(risk_limits_module, "_default_limits_path", lambda: path)
        get_risk_limits.cache_clear()
        try:
            first = get_risk_limits()
            second = get_risk_limits()
            assert first is second
        finally:
            get_risk_limits.cache_clear()

    def test_error_message_names_offending_key_and_value(self, tmp_path):
        data = copy.deepcopy(make_limits_dict())
        data["position"]["max_position_size_pct"] = 1.5
        with pytest.raises(RiskLimitsError) as excinfo:
            load_risk_limits(write_limits(tmp_path, data))
        message = str(excinfo.value)
        assert "position.max_position_size_pct" in message
        assert "1.5" in message
