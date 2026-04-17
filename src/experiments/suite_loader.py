"""Load and validate suite definition YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.experiments.suite import (
    BacktestSettings,
    ComparisonSettings,
    SuiteConfig,
    VariantSpec,
)

_ALLOWED_TOP_LEVEL = {"id", "description", "backtest", "baseline", "variants", "comparison"}
_ALLOWED_BACKTEST = {
    "strategy",
    "symbol",
    "timeframe",
    "days",
    "initial_balance",
    "provider",
    "use_cache",
    "random_seed",
}
_ALLOWED_VARIANT = {"name", "overrides"}
_ALLOWED_COMPARISON = {
    "target_metric",
    "min_trades",
    "significance_level",
}
_ALLOWED_PROVIDERS = {"binance", "coinbase", "mock", "fixture"}


class SuiteValidationError(ValueError):
    """Raised when a YAML suite file does not match the schema."""


def load_suite(path: str | Path) -> SuiteConfig:
    """Parse and validate a suite YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"suite file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return parse_suite(raw, source=str(path))


def parse_suite(raw: Any, *, source: str = "<input>") -> SuiteConfig:
    """Parse an already-deserialized YAML payload into a :class:`SuiteConfig`."""
    if not isinstance(raw, dict):
        raise SuiteValidationError(f"{source}: top-level YAML must be a mapping")

    _reject_unknown_keys(raw, _ALLOWED_TOP_LEVEL, source)
    _require_keys(raw, {"id", "backtest", "baseline"}, source)

    suite_id = _require_str(raw, "id", source)
    description = raw.get("description", "") or ""
    if not isinstance(description, str):
        raise SuiteValidationError(f"{source}: description must be a string")

    backtest = _parse_backtest(raw["backtest"], source=f"{source}.backtest")
    baseline = _parse_variant(raw["baseline"], source=f"{source}.baseline", required_name=False)
    variants_raw = raw.get("variants") or []
    if not isinstance(variants_raw, list):
        raise SuiteValidationError(f"{source}.variants: must be a list")
    variants = [
        _parse_variant(v, source=f"{source}.variants[{i}]", required_name=True)
        for i, v in enumerate(variants_raw)
    ]

    comparison_raw = raw.get("comparison") or {}
    if not isinstance(comparison_raw, dict):
        raise SuiteValidationError(f"{source}.comparison: must be a mapping")
    _reject_unknown_keys(comparison_raw, _ALLOWED_COMPARISON, f"{source}.comparison")
    comparison = ComparisonSettings(**comparison_raw)

    return SuiteConfig(
        id=suite_id,
        description=description,
        backtest=backtest,
        baseline=baseline,
        variants=variants,
        comparison=comparison,
    )


def _parse_backtest(data: Any, *, source: str) -> BacktestSettings:
    if not isinstance(data, dict):
        raise SuiteValidationError(f"{source}: must be a mapping")
    _reject_unknown_keys(data, _ALLOWED_BACKTEST, source)
    _require_keys(data, {"strategy"}, source)

    provider = data.get("provider", "binance")
    if provider not in _ALLOWED_PROVIDERS:
        raise SuiteValidationError(
            f"{source}.provider: must be one of {sorted(_ALLOWED_PROVIDERS)}, got {provider!r}"
        )

    days = data.get("days", 30)
    if not isinstance(days, int) or days <= 0:
        raise SuiteValidationError(f"{source}.days: must be a positive int, got {days!r}")

    initial_balance = float(data.get("initial_balance", 1000.0))
    if initial_balance <= 0:
        raise SuiteValidationError(
            f"{source}.initial_balance: must be positive, got {initial_balance}"
        )

    return BacktestSettings(
        strategy=_require_str(data, "strategy", source),
        symbol=str(data.get("symbol", "BTCUSDT")),
        timeframe=str(data.get("timeframe", "1h")),
        days=days,
        initial_balance=initial_balance,
        provider=provider,
        use_cache=bool(data.get("use_cache", True)),
        random_seed=data.get("random_seed"),
    )


def _parse_variant(data: Any, *, source: str, required_name: bool) -> VariantSpec:
    if not isinstance(data, dict):
        raise SuiteValidationError(f"{source}: must be a mapping")
    _reject_unknown_keys(data, _ALLOWED_VARIANT, source)

    name = data.get("name")
    if required_name and not isinstance(name, str):
        raise SuiteValidationError(f"{source}.name: required string")
    if name is not None and not isinstance(name, str):
        raise SuiteValidationError(f"{source}.name: must be a string")

    overrides = data.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise SuiteValidationError(f"{source}.overrides: must be a mapping")
    for k in overrides:
        if not isinstance(k, str) or "." not in k:
            raise SuiteValidationError(
                f"{source}.overrides: keys must be dotted '<strategy>.<attr>' strings, got {k!r}"
            )

    return VariantSpec(name=name or "baseline", overrides=dict(overrides))


def _reject_unknown_keys(data: dict[str, Any], allowed: set[str], source: str) -> None:
    extras = set(data) - allowed
    if extras:
        raise SuiteValidationError(
            f"{source}: unknown keys {sorted(extras)}; allowed {sorted(allowed)}"
        )


def _require_keys(data: dict[str, Any], required: set[str], source: str) -> None:
    missing = required - set(data)
    if missing:
        raise SuiteValidationError(f"{source}: missing required keys {sorted(missing)}")


def _require_str(data: dict[str, Any], key: str, source: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise SuiteValidationError(f"{source}.{key}: required non-empty string")
    return value


__all__ = ["SuiteValidationError", "load_suite", "parse_suite"]
