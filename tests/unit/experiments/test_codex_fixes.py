"""Regression tests for codex-review findings.

Covers path-traversal hardening, non-finite override rejection, CLI --days
validation, same-sign Calmar delta, and version-record timestamp uniqueness.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.experiments.reporter import (
    ExperimentReporter,
    Verdict,
    _metric_delta,
    _ranking_confidence,
)
from src.experiments.schemas import ExperimentConfig, ExperimentResult
from src.experiments.suite import (
    BacktestSettings,
    ComparisonSettings,
    SuiteConfig,
    SuiteResult,
    VariantSpec,
)
from src.experiments.suite_loader import SuiteValidationError, parse_suite

pytestmark = pytest.mark.fast
# ----------------------------------------------------------------------------
# P1: Path-traversal hardening for promotion output paths
# ----------------------------------------------------------------------------


def test_suite_loader_rejects_path_traversal_in_id() -> None:
    with pytest.raises(SuiteValidationError, match="slug"):
        parse_suite(
            {
                "id": "../escape",
                "backtest": {"strategy": "ml_basic"},
                "baseline": {"name": "baseline", "overrides": {}},
            }
        )


def test_suite_loader_rejects_slash_in_variant_name() -> None:
    with pytest.raises(SuiteValidationError, match="slug"):
        parse_suite(
            {
                "id": "ok",
                "backtest": {"strategy": "ml_basic"},
                "baseline": {"name": "baseline", "overrides": {}},
                "variants": [{"name": "evil/../name", "overrides": {}}],
            }
        )


def test_suite_loader_accepts_standard_slugs() -> None:
    cfg = parse_suite(
        {
            "id": "signal_thresholds_v1",
            "backtest": {"strategy": "ml_basic"},
            "baseline": {"name": "baseline_v1.0", "overrides": {}},
            "variants": [{"name": "variant-A_1", "overrides": {}}],
        }
    )
    assert cfg.id == "signal_thresholds_v1"


def test_safe_child_rejects_traversal(tmp_path: Path) -> None:
    from src.experiments.promotion import _safe_child

    with pytest.raises(ValueError, match="unsafe path segment"):
        _safe_child(tmp_path, "../outside.yaml")


def test_safe_child_rejects_null_and_slash(tmp_path: Path) -> None:
    from src.experiments.promotion import _safe_child

    with pytest.raises(ValueError):
        _safe_child(tmp_path, "nested/path.yaml")
    with pytest.raises(ValueError):
        _safe_child(tmp_path, "with\x00null.yaml")


def test_safe_child_accepts_legal_segments(tmp_path: Path) -> None:
    from src.experiments.promotion import _safe_child

    p = _safe_child(tmp_path, "suite_v1", "file-1.0.json")
    assert str(p).startswith(str(tmp_path.resolve()))


def test_safe_child_allows_generated_long_filenames(tmp_path: Path) -> None:
    """Composed filenames (strategy-variant-timestamp-uuid) exceed the slug
    length allowed for user-provided fields; the path check must still accept
    them as long as no separator/null/.. appears."""
    from src.experiments.promotion import _safe_child

    # e.g. ``<strategy>-<variant_128>-<timestamp_30>.json`` ≈ 170 chars
    long_name = f"{'x' * 128}-{'y' * 128}-20260101T000000_000000Z_abcdef12.json"
    p = _safe_child(tmp_path, long_name)
    assert str(p).startswith(str(tmp_path.resolve()))


def test_ledger_artifacts_dir_rejects_traversal(tmp_path: Path) -> None:
    from src.experiments.ledger import Ledger

    ledger = Ledger(root=tmp_path)
    with pytest.raises(ValueError, match="unsafe path segment"):
        ledger.artifacts_dir("../escape", "run1")
    with pytest.raises(ValueError, match="unsafe path segment"):
        ledger.artifacts_dir("ok_id", "..")


def test_ledger_artifacts_dir_accepts_legal_segments(tmp_path: Path) -> None:
    from src.experiments.ledger import Ledger

    ledger = Ledger(root=tmp_path)
    path = ledger.artifacts_dir("suite_v1", "run_20260417T000000_abcd1234")
    assert str(path).startswith(str(tmp_path.resolve()))


# ----------------------------------------------------------------------------
# P1: Non-finite override rejection at signal generator construction
# ----------------------------------------------------------------------------


def test_signal_generator_rejects_nan_threshold() -> None:
    from src.strategies.components.ml_signal_generator import MLBasicSignalGenerator

    with pytest.raises(ValueError, match="finite"):
        MLBasicSignalGenerator(long_entry_threshold=float("nan"))


def test_signal_generator_rejects_inf_multiplier() -> None:
    from src.strategies.components.ml_signal_generator import MLBasicSignalGenerator

    with pytest.raises(ValueError, match="finite"):
        MLBasicSignalGenerator(confidence_multiplier=float("inf"))


def test_signal_generator_rejects_non_positive_multiplier() -> None:
    from src.strategies.components.ml_signal_generator import MLBasicSignalGenerator

    with pytest.raises(ValueError, match="> 0"):
        MLBasicSignalGenerator(confidence_multiplier=0.0)
    with pytest.raises(ValueError, match="> 0"):
        MLBasicSignalGenerator(confidence_multiplier=-5.0)


def test_runner_catches_nan_override_via_setattr() -> None:
    """`setattr` bypasses __init__; post-override validator must catch NaN."""
    from src.experiments.runner import ExperimentRunner
    from src.experiments.schemas import ParameterSet

    runner = ExperimentRunner()
    strategy = runner._load_strategy("ml_basic")
    cfg = ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=datetime.now(UTC),
        end=datetime.now(UTC),
        initial_balance=1000.0,
        parameters=ParameterSet(
            name="nan_knob",
            values={"ml_basic.long_entry_threshold": float("nan")},
        ),
    )
    # The override path coerces via _coerce_value and calls setattr directly —
    # NaN passes through. The post-override validator must catch it.
    runner._apply_parameter_overrides(strategy, cfg)
    with pytest.raises(ValueError, match="finite"):
        runner._validate_post_override_invariants(strategy)


# ----------------------------------------------------------------------------
# P2: CLI --days zero/negative validation
# ----------------------------------------------------------------------------


def test_positive_int_argparse_validator_rejects_zero() -> None:
    import argparse

    from cli.commands.experiment import _positive_int

    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")


def test_positive_int_argparse_validator_rejects_negative() -> None:
    import argparse

    from cli.commands.experiment import _positive_int

    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("-5")


def test_positive_int_argparse_validator_accepts_one() -> None:
    from cli.commands.experiment import _positive_int

    assert _positive_int("1") == 1


# ----------------------------------------------------------------------------
# P2: Same-sign Calmar infinities produce 0.0 delta (tie), not +∞
# ----------------------------------------------------------------------------


def test_metric_delta_same_sign_infinities_yields_zero() -> None:
    assert _metric_delta(math.inf, math.inf) == 0.0
    assert _metric_delta(-math.inf, -math.inf) == 0.0


def test_metric_delta_opposite_sign_infinities_yields_signed_inf() -> None:
    assert _metric_delta(math.inf, -math.inf) == math.inf
    assert _metric_delta(-math.inf, math.inf) == -math.inf


def test_metric_delta_one_side_infinite() -> None:
    assert _metric_delta(math.inf, 3.0) == math.inf
    assert _metric_delta(-math.inf, 3.0) == -math.inf
    assert _metric_delta(3.0, math.inf) == -math.inf
    assert _metric_delta(3.0, -math.inf) == math.inf


def test_metric_delta_finite_is_plain_subtraction() -> None:
    assert _metric_delta(5.0, 3.0) == pytest.approx(2.0)


def _make_cfg() -> ExperimentConfig:
    return ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=datetime.now(UTC),
        end=datetime.now(UTC),
        initial_balance=1000.0,
    )


def _make_calmar_result(
    *, total_return: float, annualized: float, max_drawdown: float
) -> ExperimentResult:
    cfg = _make_cfg()
    return ExperimentResult(
        config=cfg,
        total_trades=200,
        win_rate=55.0,
        total_return=total_return,
        annualized_return=annualized,
        max_drawdown=max_drawdown,
        sharpe_ratio=0.0,
        final_balance=1000.0 + total_return,
    )


def test_reporter_same_sign_calmar_infinities_report_zero_delta() -> None:
    """Pre-fix bug: inf - inf = NaN was coerced to +∞ in render()."""
    suite = SuiteConfig(
        id="calmar_tie",
        description="",
        backtest=BacktestSettings(strategy="ml_basic"),
        baseline=VariantSpec(name="baseline"),
        variants=[VariantSpec(name="tied_variant")],
        comparison=ComparisonSettings(target_metric="calmar", min_trades=0),
    )
    # Both baseline and variant: zero drawdown + positive annualized → Calmar +∞
    baseline = _make_calmar_result(total_return=10.0, annualized=15.0, max_drawdown=0.0)
    variants = [
        _make_calmar_result(total_return=20.0, annualized=25.0, max_drawdown=0.0),
    ]
    suite_result = SuiteResult(
        suite_id=suite.id,
        config=suite,
        baseline=baseline,
        variants=variants,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )
    report = ExperimentReporter().render(suite_result)
    tied = next(r for r in report.rows if r.name == "tied_variant")
    # Delta should be a tie (0.0), not +∞
    assert tied.delta_vs_baseline == 0.0
    # Confidence should be low (same-sign inf → 0.0 in ranking_confidence)
    assert tied.ranking_confidence == 0.0
    # Verdict: HOLD (tie, not a strict improvement)
    assert tied.verdict == Verdict.HOLD


def test_ranking_confidence_same_sign_inf_is_zero() -> None:
    """Regression: ranking_confidence must treat same-sign inf as indistinguishable."""
    cfg_base = _make_calmar_result(total_return=1.0, annualized=1.0, max_drawdown=0.0)
    cfg_var = _make_calmar_result(total_return=5.0, annualized=5.0, max_drawdown=0.0)
    # Both are +∞ on Calmar
    assert _ranking_confidence(cfg_base, cfg_var, "calmar") == 0.0


# ----------------------------------------------------------------------------
# P3: Version-record timestamp uniqueness
# ----------------------------------------------------------------------------


def test_timestamp_is_unique_within_same_second() -> None:
    from src.experiments.promotion import _timestamp

    seen = {_timestamp() for _ in range(50)}
    assert len(seen) == 50


def test_metric_rejects_nan_sharpe() -> None:
    """NaN Sharpe would propagate silently through ranking and verdict."""
    from src.experiments.reporter import _metric

    cfg = _make_cfg()
    result = ExperimentResult(
        config=cfg,
        total_trades=100,
        win_rate=55.0,
        total_return=1.0,
        annualized_return=2.0,
        max_drawdown=5.0,
        sharpe_ratio=math.nan,
        final_balance=1010.0,
    )
    with pytest.raises(ValueError, match="NaN"):
        _metric(result, "sharpe_ratio")


def test_metric_rejects_nan_calmar_inputs() -> None:
    """Calmar with NaN annualized_return must not silently yield a number."""
    from src.experiments.reporter import _metric

    cfg = _make_cfg()
    result = ExperimentResult(
        config=cfg,
        total_trades=100,
        win_rate=55.0,
        total_return=1.0,
        annualized_return=math.nan,
        max_drawdown=5.0,
        sharpe_ratio=1.0,
        final_balance=1010.0,
    )
    with pytest.raises(ValueError, match="NaN"):
        _metric(result, "calmar")


def test_string_override_on_numeric_knob_is_caught_post_validation() -> None:
    """A YAML override `"abc"` on a numeric knob slips through _coerce_value;
    the post-override validator must reject it rather than deferring to signal gen."""
    from src.experiments.runner import ExperimentRunner
    from src.experiments.schemas import ParameterSet

    runner = ExperimentRunner()
    strategy = runner._load_strategy("ml_basic")
    cfg = ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=datetime.now(UTC),
        end=datetime.now(UTC),
        initial_balance=1000.0,
        parameters=ParameterSet(
            name="stringy",
            values={"ml_basic.long_entry_threshold": "abc"},
        ),
    )
    runner._apply_parameter_overrides(strategy, cfg)
    # Value is now a str on signal_generator.long_entry_threshold — post-validation must raise.
    with pytest.raises(ValueError, match="numeric"):
        runner._validate_post_override_invariants(strategy)


def test_check_numeric_bound_rejects_nan() -> None:
    """NaN silently passes < and > comparisons; explicit isfinite guard needed."""
    from src.experiments.runner import _check_numeric_bound

    class _Target:
        base_fraction = math.nan

    with pytest.raises(ValueError, match="finite"):
        _check_numeric_bound(_Target(), "base_fraction", 0.001, 0.5)


def test_check_numeric_bound_rejects_inf() -> None:
    from src.experiments.runner import _check_numeric_bound

    class _Target:
        base_fraction = math.inf

    with pytest.raises(ValueError, match="finite"):
        _check_numeric_bound(_Target(), "base_fraction", 0.001, 0.5)


def test_runner_rejects_nan_base_fraction_via_override() -> None:
    from src.experiments.runner import ExperimentRunner
    from src.experiments.schemas import ParameterSet

    runner = ExperimentRunner()
    strategy = runner._load_strategy("ml_basic")
    cfg = ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=datetime.now(UTC),
        end=datetime.now(UTC),
        initial_balance=1000.0,
        parameters=ParameterSet(
            name="nan_frac",
            values={"ml_basic.base_fraction": math.nan},
        ),
    )
    runner._apply_parameter_overrides(strategy, cfg)
    with pytest.raises(ValueError, match="finite"):
        runner._validate_post_override_invariants(strategy)


def test_suite_loader_rejects_nan_initial_balance() -> None:
    with pytest.raises(SuiteValidationError, match="finite"):
        parse_suite(
            {
                "id": "ok",
                "backtest": {"strategy": "ml_basic", "initial_balance": float("nan")},
                "baseline": {"name": "baseline", "overrides": {}},
            }
        )


def test_suite_loader_rejects_inf_initial_balance() -> None:
    with pytest.raises(SuiteValidationError, match="finite"):
        parse_suite(
            {
                "id": "ok",
                "backtest": {"strategy": "ml_basic", "initial_balance": float("inf")},
                "baseline": {"name": "baseline", "overrides": {}},
            }
        )


def test_confidence_weighted_sizer_raises_when_floor_exceeds_gate() -> None:
    """Constructor and override paths must agree: floor>gate is invalid."""
    from src.strategies.components.position_sizer import ConfidenceWeightedSizer

    with pytest.raises(ValueError, match="min_confidence_floor"):
        ConfidenceWeightedSizer(
            base_fraction=0.05,
            min_confidence=0.3,
            min_confidence_floor=0.5,
        )


def test_suite_loader_rejects_non_scalar_override_value() -> None:
    """Recursive YAML anchors / nested containers crash JSON serialization."""
    with pytest.raises(SuiteValidationError, match="scalar"):
        parse_suite(
            {
                "id": "ok",
                "backtest": {"strategy": "ml_basic"},
                "baseline": {"name": "baseline", "overrides": {}},
                "variants": [
                    {"name": "bad", "overrides": {"ml_basic.model_name": ["list", "not", "scalar"]}}
                ],
            }
        )


def test_suite_loader_accepts_scalar_override_values() -> None:
    cfg = parse_suite(
        {
            "id": "ok",
            "backtest": {"strategy": "ml_basic"},
            "baseline": {"name": "baseline", "overrides": {}},
            "variants": [
                {
                    "name": "good",
                    "overrides": {
                        "ml_basic.long_entry_threshold": 0.001,
                        "ml_basic.model_name": "basic",
                        "ml_basic.use_prediction_engine": True,
                    },
                }
            ],
        }
    )
    assert cfg.variants[0].name == "good"


def test_ledger_artifacts_dir_accepts_single_char_slug(tmp_path: Path) -> None:
    """Slug alphabet must match the loader: one-char ids are legal."""
    from src.experiments.ledger import Ledger

    ledger = Ledger(root=tmp_path)
    path = ledger.artifacts_dir("a", "1")
    assert str(path).startswith(str(tmp_path.resolve()))


def test_fixture_provider_accepts_utc_aware_bounds(tmp_path: Path) -> None:
    """FixtureProvider loaded a tz-naive feather; querying with UTC-aware
    datetimes must not raise ``TypeError`` (pandas refuses to compare a
    tz-naive index to a tz-aware bound)."""
    import pandas as pd

    from src.data_providers.offline import FixtureProvider

    feather_path = tmp_path / "fixture.feather"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h"),  # naive
            "open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "high": [1.1, 2.1, 3.1, 4.1, 5.1],
            "low": [0.9, 1.9, 2.9, 3.9, 4.9],
            "close": [1.05, 2.05, 3.05, 4.05, 5.05],
            "volume": [100.0] * 5,
        }
    )
    df.to_feather(feather_path)

    provider = FixtureProvider(feather_path)
    start = datetime(2024, 1, 1, 0, tzinfo=UTC)
    end = datetime(2024, 1, 1, 3, tzinfo=UTC)

    out = provider.get_historical_data("BTCUSDT", "1h", start=start, end=end)
    assert not out.empty
    assert len(out) == 4


def test_random_walk_provider_handles_utc_aware_bounds() -> None:
    from src.data_providers.offline import RandomWalkProvider

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    provider = RandomWalkProvider(start, end, timeframe="1h", seed=42)
    out = provider.get_historical_data("BTCUSDT", "1h", start=start, end=end)
    assert not out.empty


def test_timestamp_includes_microseconds_and_uuid_suffix() -> None:
    from src.experiments.promotion import _timestamp

    ts = _timestamp()
    # Format: YYYYMMDDTHHMMSS_microsZ_8-char-uuid
    assert ts.endswith(tuple("0123456789abcdef"))
    assert "_" in ts and "Z_" in ts
