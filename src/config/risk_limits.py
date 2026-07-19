"""Runtime loader for the Board-ratified risk limits.

``src/config/risk-limits.json`` is the single ratified source of risk and
trading limits. The file is HUMAN-OWNED (``$owner: human_board``): agents and
automated tooling never edit its values; changes happen only through a Board
ratification sitting. This module loads the file and strictly validates it
into frozen dataclasses so every consumer reads exactly one typed accessor
per ratified key.

Validation is fail-closed: any missing file, malformed JSON, unknown or
missing key, type/range violation, or cross-field inconsistency raises
:class:`RiskLimitsError` naming the offending key and value. There is
deliberately no environment-variable path override — an env-redirectable
limits file would be a loosening vector.

Units convention: every ``*_pct`` key (and ``kelly_max_fraction``) is a
decimal fraction in ``(0, 1]`` — 0.20 means 20%.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config.paths import get_project_root

SUPPORTED_SCHEMA_VERSION = "1"

# Metadata keys are Board-owned annotations; they are accepted (and
# ``$schema_version`` enforced) but carry no runtime limit values.
_KNOWN_METADATA_KEYS = frozenset(
    {
        "$schema_version",
        "$owner",
        "$source_of_truth_note",
        "$last_reviewed",
        "$last_reviewer",
    }
)


class RiskLimitsError(Exception):
    """Raised when the risk-limits file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class PortfolioLimits:
    """Portfolio-wide exposure and drawdown limits."""

    max_drawdown_pct: float
    max_daily_risk_pct: float
    max_correlated_exposure_pct: float
    dynamic_drawdown_thresholds_pct: tuple[float, ...]
    dynamic_risk_reduction_factors: tuple[float, ...]


@dataclass(frozen=True)
class PositionLimits:
    """Per-position sizing and leverage limits."""

    max_position_size_pct: float
    base_risk_per_trade_pct: float
    max_risk_per_trade_pct: float
    max_leverage: float
    kelly_max_fraction: float
    large_single_position_threshold_pct: float


@dataclass(frozen=True)
class StopLimits:
    """Stop-loss / take-profit bounds and defaults."""

    default_stop_loss_pct: float
    min_stop_loss_pct: float
    max_stop_loss_pct: float
    default_take_profit_pct: float
    fallback_trailing_pct: float


@dataclass(frozen=True)
class OperationalLimits:
    """Operational guardrails (error budgets, holding time, discrepancies)."""

    max_consecutive_errors: int
    max_holding_hours: int
    max_filled_price_deviation_pct: float
    balance_discrepancy_warning_pct: float
    reconciliation_balance_critical_pct: float


@dataclass(frozen=True)
class EscalationPolicy:
    """Thresholds (as fractions of a limit) at which alerts escalate."""

    warning_at_pct_of_limit: float
    critical_at_pct_of_limit: float
    breach_action: str


@dataclass(frozen=True)
class KillSwitchPolicy:
    """Who may halt trading and what conditions trigger an automatic halt."""

    authorized_actors: tuple[str, ...]
    auto_trigger_conditions: tuple[str, ...]
    manual_trigger_command: str


@dataclass(frozen=True)
class RiskLimits:
    """The full ratified limit set, one attribute per JSON key."""

    schema_version: str
    portfolio: PortfolioLimits
    position: PositionLimits
    stops: StopLimits
    operational: OperationalLimits
    escalation: EscalationPolicy
    kill_switch: KillSwitchPolicy


def _default_limits_path() -> Path:
    """Resolve the ratified limits file relative to the project root."""
    return get_project_root() / "src" / "config" / "risk-limits.json"


def _fail(key: str, value: Any, requirement: str) -> RiskLimitsError:
    return RiskLimitsError(f"risk-limits key '{key}' must be {requirement}, got {value!r}")


def _is_number(value: Any) -> bool:
    # bool is an int subclass; True/False must not pass as limit values.
    return isinstance(value, int | float) and not isinstance(value, bool)


def _require_fraction(key: str, value: Any) -> float:
    """A decimal fraction in (0, 1] — the units convention for *_pct keys."""
    if not _is_number(value) or not math.isfinite(value):
        raise _fail(key, value, "a finite number")
    if not 0 < value <= 1:
        raise _fail(key, value, "a decimal fraction in (0, 1]")
    return float(value)


def _require_positive_number(key: str, value: Any) -> float:
    if not _is_number(value) or not math.isfinite(value):
        raise _fail(key, value, "a finite number")
    if value <= 0:
        raise _fail(key, value, "a positive number")
    return float(value)


def _require_positive_int(key: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _fail(key, value, "an integer")
    if value <= 0:
        raise _fail(key, value, "a positive integer")
    return value


def _require_non_empty_str(key: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(key, value, "a non-empty string")
    return value


def _require_str_tuple(key: str, value: Any, *, non_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _fail(key, value, "a list of non-empty strings")
    if non_empty and not value:
        raise _fail(key, value, "a non-empty list")
    for item in value:
        if not isinstance(item, str) or not item:
            raise _fail(key, value, "a list of non-empty strings")
    return tuple(value)


def _require_fraction_tuple(key: str, value: Any) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise _fail(key, value, "a list of decimal fractions")
    return tuple(_require_fraction(f"{key}[{i}]", item) for i, item in enumerate(value))


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise _fail(name, section, "a JSON object section")
    return section


def _check_exact_keys(section_name: str, section: dict[str, Any], expected: set[str]) -> None:
    unknown = sorted(section.keys() - expected)
    if unknown:
        raise RiskLimitsError(
            f"risk-limits section '{section_name}' has unknown key(s): {', '.join(unknown)}"
        )
    missing = sorted(expected - section.keys())
    if missing:
        raise RiskLimitsError(
            f"risk-limits section '{section_name}' is missing key(s): {', '.join(missing)}"
        )


def _parse_section(
    raw: dict[str, Any],
    name: str,
    field_validators: dict[str, Callable[[str, Any], Any]],
) -> dict[str, Any]:
    """Validate one section: exact key set, then per-key type/range checks."""
    section = _require_section(raw, name)
    _check_exact_keys(name, section, set(field_validators))
    return {
        key: validator(f"{name}.{key}", section[key]) for key, validator in field_validators.items()
    }


def _validate_schema_version(raw: dict[str, Any]) -> str:
    version = raw.get("$schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise _fail("$schema_version", version, f"the pinned string '{SUPPORTED_SCHEMA_VERSION}'")
    return version


def _validate_top_level_keys(raw: dict[str, Any], section_names: set[str]) -> None:
    unknown = sorted(raw.keys() - _KNOWN_METADATA_KEYS - section_names)
    if unknown:
        raise RiskLimitsError(f"risk-limits has unknown top-level key(s): {', '.join(unknown)}")
    missing = sorted(section_names - raw.keys())
    if missing:
        raise RiskLimitsError(f"risk-limits is missing section(s): {', '.join(missing)}")


def _validate_cross_field(
    portfolio: PortfolioLimits,
    position: PositionLimits,
    stops: StopLimits,
    operational: OperationalLimits,
    escalation: EscalationPolicy,
) -> None:
    """Invariants spanning multiple keys; each names the offending keys."""
    if position.base_risk_per_trade_pct > position.max_risk_per_trade_pct:
        raise RiskLimitsError(
            "position.base_risk_per_trade_pct "
            f"({position.base_risk_per_trade_pct}) must be <= "
            f"position.max_risk_per_trade_pct ({position.max_risk_per_trade_pct})"
        )
    if stops.min_stop_loss_pct > stops.default_stop_loss_pct:
        raise RiskLimitsError(
            f"stops.min_stop_loss_pct ({stops.min_stop_loss_pct}) must be <= "
            f"stops.default_stop_loss_pct ({stops.default_stop_loss_pct})"
        )
    if stops.default_stop_loss_pct > stops.max_stop_loss_pct:
        raise RiskLimitsError(
            f"stops.default_stop_loss_pct ({stops.default_stop_loss_pct}) must "
            f"be <= stops.max_stop_loss_pct ({stops.max_stop_loss_pct})"
        )

    thresholds = portfolio.dynamic_drawdown_thresholds_pct
    factors = portfolio.dynamic_risk_reduction_factors
    if len(thresholds) != len(factors):
        raise RiskLimitsError(
            "portfolio.dynamic_drawdown_thresholds_pct and "
            "portfolio.dynamic_risk_reduction_factors must have equal length, "
            f"got {len(thresholds)} and {len(factors)}"
        )
    if any(later <= earlier for earlier, later in zip(thresholds, thresholds[1:], strict=False)):
        raise RiskLimitsError(
            "portfolio.dynamic_drawdown_thresholds_pct must be strictly "
            f"ascending, got {list(thresholds)}"
        )
    # Dead-tier invariant: a tier at or beyond the hard drawdown cap can never
    # fire — the drawdown guard latches close-only first.
    if thresholds and max(thresholds) >= portfolio.max_drawdown_pct:
        raise RiskLimitsError(
            "every portfolio.dynamic_drawdown_thresholds_pct entry must be < "
            f"portfolio.max_drawdown_pct ({portfolio.max_drawdown_pct}), "
            f"got {list(thresholds)}"
        )

    if escalation.warning_at_pct_of_limit > escalation.critical_at_pct_of_limit:
        raise RiskLimitsError(
            "escalation.warning_at_pct_of_limit "
            f"({escalation.warning_at_pct_of_limit}) must be <= "
            f"escalation.critical_at_pct_of_limit "
            f"({escalation.critical_at_pct_of_limit})"
        )
    if (
        operational.balance_discrepancy_warning_pct
        > operational.reconciliation_balance_critical_pct
    ):
        raise RiskLimitsError(
            "operational.balance_discrepancy_warning_pct "
            f"({operational.balance_discrepancy_warning_pct}) must be <= "
            "operational.reconciliation_balance_critical_pct "
            f"({operational.reconciliation_balance_critical_pct})"
        )


_SECTION_VALIDATORS: dict[str, dict[str, Callable[[str, Any], Any]]] = {
    "portfolio": {
        "max_drawdown_pct": _require_fraction,
        "max_daily_risk_pct": _require_fraction,
        "max_correlated_exposure_pct": _require_fraction,
        "dynamic_drawdown_thresholds_pct": _require_fraction_tuple,
        "dynamic_risk_reduction_factors": _require_fraction_tuple,
    },
    "position": {
        "max_position_size_pct": _require_fraction,
        "base_risk_per_trade_pct": _require_fraction,
        "max_risk_per_trade_pct": _require_fraction,
        "max_leverage": _require_positive_number,
        "kelly_max_fraction": _require_fraction,
        "large_single_position_threshold_pct": _require_fraction,
    },
    "stops": {
        "default_stop_loss_pct": _require_fraction,
        "min_stop_loss_pct": _require_fraction,
        "max_stop_loss_pct": _require_fraction,
        "default_take_profit_pct": _require_fraction,
        "fallback_trailing_pct": _require_fraction,
    },
    "operational": {
        "max_consecutive_errors": _require_positive_int,
        "max_holding_hours": _require_positive_int,
        "max_filled_price_deviation_pct": _require_fraction,
        "balance_discrepancy_warning_pct": _require_fraction,
        "reconciliation_balance_critical_pct": _require_fraction,
    },
    "escalation": {
        "warning_at_pct_of_limit": _require_fraction,
        "critical_at_pct_of_limit": _require_fraction,
        "breach_action": _require_non_empty_str,
    },
    "kill_switch": {
        "authorized_actors": lambda key, value: _require_str_tuple(key, value, non_empty=True),
        "auto_trigger_conditions": lambda key, value: _require_str_tuple(
            key, value, non_empty=False
        ),
        "manual_trigger_command": _require_non_empty_str,
    },
}


def load_risk_limits(path: Path | None = None) -> RiskLimits:
    """Load and validate the ratified risk limits.

    Args:
        path: Explicit file path — for tests only. Production consumers use
            the default project-root resolution (no env-var override).

    Returns:
        The fully validated, immutable limit set.

    Raises:
        RiskLimitsError: If the file is missing, unreadable, malformed, or
            fails any schema/range/cross-field check.
    """
    resolved = path if path is not None else _default_limits_path()
    if not resolved.is_file():
        raise RiskLimitsError(f"risk-limits file not found: {resolved}")
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RiskLimitsError(f"risk-limits file unreadable: {resolved}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RiskLimitsError(f"risk-limits file is invalid JSON: {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RiskLimitsError(f"risk-limits file must contain a JSON object: {resolved}")

    schema_version = _validate_schema_version(raw)
    _validate_top_level_keys(raw, set(_SECTION_VALIDATORS))

    portfolio = PortfolioLimits(
        **_parse_section(raw, "portfolio", _SECTION_VALIDATORS["portfolio"])
    )
    position = PositionLimits(**_parse_section(raw, "position", _SECTION_VALIDATORS["position"]))
    stops = StopLimits(**_parse_section(raw, "stops", _SECTION_VALIDATORS["stops"]))
    operational = OperationalLimits(
        **_parse_section(raw, "operational", _SECTION_VALIDATORS["operational"])
    )
    escalation = EscalationPolicy(
        **_parse_section(raw, "escalation", _SECTION_VALIDATORS["escalation"])
    )
    kill_switch = KillSwitchPolicy(
        **_parse_section(raw, "kill_switch", _SECTION_VALIDATORS["kill_switch"])
    )

    _validate_cross_field(portfolio, position, stops, operational, escalation)

    return RiskLimits(
        schema_version=schema_version,
        portfolio=portfolio,
        position=position,
        stops=stops,
        operational=operational,
        escalation=escalation,
        kill_switch=kill_switch,
    )


@lru_cache(maxsize=1)
def get_risk_limits() -> RiskLimits:
    """Process-cached accessor for the ratified limits (default path)."""
    return load_risk_limits()


__all__ = [
    "EscalationPolicy",
    "KillSwitchPolicy",
    "OperationalLimits",
    "PortfolioLimits",
    "PositionLimits",
    "RiskLimits",
    "RiskLimitsError",
    "StopLimits",
    "get_risk_limits",
    "load_risk_limits",
]
