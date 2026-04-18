"""Unit tests for the YAML suite loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.suite_loader import SuiteValidationError, load_suite, parse_suite

pytestmark = pytest.mark.fast
_MINIMAL_YAML = """
id: demo_suite
description: demo
backtest:
  strategy: ml_basic
  symbol: BTCUSDT
  timeframe: 1h
  days: 30
  initial_balance: 1000
  provider: mock
  random_seed: 42
baseline:
  name: current_defaults
  overrides: {}
variants:
  - name: long_thr_up
    overrides:
      ml_basic.long_entry_threshold: 0.0003
comparison:
  target_metric: sharpe_ratio
  min_trades: 30
  significance_level: 0.05
"""


def test_load_full_example(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(_MINIMAL_YAML)
    cfg = load_suite(path)

    assert cfg.id == "demo_suite"
    assert cfg.backtest.strategy == "ml_basic"
    assert cfg.baseline.name == "current_defaults"
    assert cfg.baseline.overrides == {}
    assert cfg.variants[0].name == "long_thr_up"
    assert cfg.variants[0].overrides["ml_basic.long_entry_threshold"] == 0.0003
    assert cfg.comparison.target_metric == "sharpe_ratio"
    assert cfg.comparison.min_trades == 30
    assert cfg.comparison.significance_level == 0.05


def test_missing_id_raises() -> None:
    with pytest.raises(SuiteValidationError, match="missing required keys"):
        parse_suite(
            {
                "backtest": {"strategy": "ml_basic"},
                "baseline": {"name": "x", "overrides": {}},
            }
        )


def test_unknown_top_level_key_rejected() -> None:
    with pytest.raises(SuiteValidationError, match="unknown keys"):
        parse_suite(
            {
                "id": "x",
                "backtest": {"strategy": "ml_basic"},
                "baseline": {"name": "b", "overrides": {}},
                "extra": 1,
            }
        )


def test_unknown_backtest_key_rejected() -> None:
    with pytest.raises(SuiteValidationError, match="unknown keys"):
        parse_suite(
            {
                "id": "x",
                "backtest": {"strategy": "ml_basic", "nonsense": 1},
                "baseline": {"name": "b", "overrides": {}},
            }
        )


def test_invalid_provider_rejected() -> None:
    with pytest.raises(SuiteValidationError, match="provider"):
        parse_suite(
            {
                "id": "x",
                "backtest": {"strategy": "ml_basic", "provider": "kraken"},
                "baseline": {"name": "b", "overrides": {}},
            }
        )


def test_override_keys_must_be_dotted() -> None:
    with pytest.raises(SuiteValidationError, match="dotted"):
        parse_suite(
            {
                "id": "x",
                "backtest": {"strategy": "ml_basic"},
                "baseline": {"name": "b", "overrides": {}},
                "variants": [{"name": "bad", "overrides": {"not_dotted": 1}}],
            }
        )


def test_zero_or_negative_days_rejected() -> None:
    with pytest.raises(SuiteValidationError, match="days"):
        parse_suite(
            {
                "id": "x",
                "backtest": {"strategy": "ml_basic", "days": 0},
                "baseline": {"name": "b", "overrides": {}},
            }
        )


def test_negative_initial_balance_rejected() -> None:
    with pytest.raises(SuiteValidationError, match="initial_balance"):
        parse_suite(
            {
                "id": "x",
                "backtest": {"strategy": "ml_basic", "initial_balance": -5},
                "baseline": {"name": "b", "overrides": {}},
            }
        )


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_suite("/tmp/does_not_exist_xyz123.yaml")


# --------------------------------------------------------------------------
# G1 — factory_kwargs YAML schema. The runner forwards these to the
# strategy factory at construction time (distinct from ``overrides`` which
# setattr post-construction).
# --------------------------------------------------------------------------


def test_factory_kwargs_parsed_as_dict_under_backtest() -> None:
    cfg = parse_suite(
        {
            "id": "fk",
            "backtest": {
                "strategy": "hyper_growth",
                "factory_kwargs": {"max_leverage": 2.0, "min_regime_bars": 5},
            },
            "baseline": {"name": "b", "overrides": {}},
        }
    )
    assert cfg.backtest.factory_kwargs == {"max_leverage": 2.0, "min_regime_bars": 5}


def test_factory_kwargs_default_is_empty_dict() -> None:
    cfg = parse_suite(
        {
            "id": "nofk",
            "backtest": {"strategy": "ml_basic"},
            "baseline": {"name": "b", "overrides": {}},
        }
    )
    assert cfg.backtest.factory_kwargs == {}


def test_factory_kwargs_non_mapping_rejected() -> None:
    with pytest.raises(SuiteValidationError, match="must be a mapping"):
        parse_suite(
            {
                "id": "bad",
                "backtest": {"strategy": "ml_basic", "factory_kwargs": ["not", "a", "map"]},
                "baseline": {"name": "b", "overrides": {}},
            }
        )


def test_factory_kwargs_non_identifier_key_rejected() -> None:
    """Factory kwargs map to Python keyword arguments; keys must be valid
    identifiers (no dots, no spaces)."""
    with pytest.raises(SuiteValidationError, match="valid Python identifiers"):
        parse_suite(
            {
                "id": "bad",
                "backtest": {
                    "strategy": "ml_basic",
                    "factory_kwargs": {"max.leverage": 2.0},
                },
                "baseline": {"name": "b", "overrides": {}},
            }
        )


def test_factory_kwargs_non_scalar_value_rejected() -> None:
    with pytest.raises(SuiteValidationError, match="must be a scalar"):
        parse_suite(
            {
                "id": "bad",
                "backtest": {
                    "strategy": "ml_basic",
                    "factory_kwargs": {"something": {"nested": 1}},
                },
                "baseline": {"name": "b", "overrides": {}},
            }
        )
