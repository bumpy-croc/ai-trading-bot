"""ExperimentRunner risk seeding parity with the backtest CLI (GH #1021).

BEHAVIOR DELTA CAPTURED HERE: before this change the harness built a bare
``RiskParameters()`` and silently clamped any strategy-declared
``max_fraction`` (e.g. 0.25) down to the 0.10 default cap — while the backtest
CLI honored it. The harness now seeds ``max_position_size`` from the
strategy's ``get_risk_overrides()['max_fraction']`` exactly like the CLI, so
harness studies size like CLI backtests for strategies with overrides.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from src.experiments.runner import ExperimentRunner
from src.experiments.schemas import ExperimentConfig
from src.risk.risk_manager import RiskParameters

pytestmark = pytest.mark.fast


class _StubStrategy:
    """Strategy stub with configurable risk overrides."""

    name = "stub"

    def __init__(self, overrides=None):
        self._overrides = overrides

    def get_risk_overrides(self):
        return self._overrides


class _NoOverrideStrategy:
    """Strategy stub without a get_risk_overrides hook."""

    name = "no_override_stub"


_BACKTEST_RESULTS = {
    "total_trades": 0,
    "win_rate": 0.0,
    "total_return": 0.0,
    "annualized_return": 0.0,
    "max_drawdown": 0.0,
    "sharpe_ratio": 0.0,
    "final_balance": 1000.0,
    "session_id": None,
    "trade_pnl_pcts": [],
}


def _config(**overrides) -> ExperimentConfig:
    end = datetime(2026, 1, 10, tzinfo=UTC)
    kwargs = {
        "strategy_name": "ml_basic",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start": end - timedelta(days=3),
        "end": end,
        "initial_balance": 1000.0,
        "provider": "mock",
        "use_cache": False,
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def _run_and_capture_risk_params(strategy, config=None, results=None):
    """Run the harness with a stubbed strategy/backtester; return the
    RiskParameters handed to the Backtester and the ExperimentResult."""
    runner = ExperimentRunner()
    with (
        patch.object(ExperimentRunner, "_load_strategy", return_value=strategy),
        patch("src.experiments.runner.Backtester", autospec=True) as mock_backtester,
    ):
        mock_backtester.return_value.run.return_value = dict(results or _BACKTEST_RESULTS)
        result = runner.run(config or _config())
    risk_params = mock_backtester.call_args.kwargs["risk_parameters"]
    return risk_params, result


class TestRunnerSeedsStrategyMaxFraction:
    def test_strategy_max_fraction_seeds_max_position_size(self):
        """THE #1021 FIX: a strategy requesting max_fraction=0.25 now runs the
        harness at 0.25 (CLI parity), not silently clamped to the 0.10 bare
        default."""
        risk_params, _ = _run_and_capture_risk_params(_StubStrategy({"max_fraction": 0.25}))
        assert isinstance(risk_params, RiskParameters)
        assert risk_params.max_position_size == 0.25

    def test_explicit_config_value_wins_over_strategy_override(self):
        config = _config(risk_parameters={"max_position_size": 0.15})
        risk_params, _ = _run_and_capture_risk_params(
            _StubStrategy({"max_fraction": 0.25}), config=config
        )
        assert risk_params.max_position_size == 0.15

    def test_other_config_keys_pass_through_alongside_seeding(self):
        config = _config(risk_parameters={"max_drawdown": 0.3})
        risk_params, _ = _run_and_capture_risk_params(
            _StubStrategy({"max_fraction": 0.25}), config=config
        )
        assert risk_params.max_position_size == 0.25
        assert risk_params.max_drawdown == 0.3

    def test_strategy_without_hook_keeps_bare_default(self):
        risk_params, _ = _run_and_capture_risk_params(_NoOverrideStrategy())
        assert risk_params.max_position_size == RiskParameters().max_position_size

    def test_invalid_override_value_keeps_bare_default(self):
        risk_params, _ = _run_and_capture_risk_params(_StubStrategy({"max_fraction": 1.5}))
        assert risk_params.max_position_size == RiskParameters().max_position_size

    def test_non_dict_overrides_keep_bare_default(self):
        risk_params, _ = _run_and_capture_risk_params(_StubStrategy(None))
        assert risk_params.max_position_size == RiskParameters().max_position_size


class TestRunnerReportsEffectiveSizing:
    def test_effective_sizing_propagates_from_backtester_results(self):
        results = dict(_BACKTEST_RESULTS)
        results["effective_sizing"] = {
            "max_position_size": 0.25,
            "base_risk_per_trade": 0.02,
            "max_risk_per_trade": 0.03,
        }
        _, result = _run_and_capture_risk_params(
            _StubStrategy({"max_fraction": 0.25}), results=results
        )
        assert result.effective_sizing == {
            "max_position_size": 0.25,
            "base_risk_per_trade": 0.02,
            "max_risk_per_trade": 0.03,
        }

    def test_missing_effective_sizing_defaults_to_empty_dict(self):
        _, result = _run_and_capture_risk_params(_StubStrategy(None))
        assert result.effective_sizing == {}


class TestRunnerDrawdownCapModePlumbing:
    """#1102: ExperimentRunner must forward drawdown_cap_mode to the
    Backtester and surface the truncation markers on the result -- a
    preregistered research run must be able to opt into 'measure' and every
    result must say whether it was truncated, not just the ones that were.
    """

    def test_drawdown_cap_mode_defaults_to_enforce(self):
        """The harness default must match the live-representative/safe
        default -- omitting the field must never silently switch a study to
        the untruncated research mode."""
        runner = ExperimentRunner()
        with (
            patch.object(ExperimentRunner, "_load_strategy", return_value=_NoOverrideStrategy()),
            patch("src.experiments.runner.Backtester", autospec=True) as mock_backtester,
        ):
            mock_backtester.return_value.run.return_value = dict(_BACKTEST_RESULTS)
            runner.run(_config())
        assert mock_backtester.call_args.kwargs["drawdown_cap_mode"] == "enforce"

    def test_explicit_measure_mode_is_forwarded(self):
        config = _config(drawdown_cap_mode="measure")
        runner = ExperimentRunner()
        with (
            patch.object(ExperimentRunner, "_load_strategy", return_value=_NoOverrideStrategy()),
            patch("src.experiments.runner.Backtester", autospec=True) as mock_backtester,
        ):
            mock_backtester.return_value.run.return_value = dict(_BACKTEST_RESULTS)
            runner.run(config)
        assert mock_backtester.call_args.kwargs["drawdown_cap_mode"] == "measure"

    def test_truncated_backtester_result_marks_experiment_result(self):
        """A truncated Backtester result must produce an ExperimentResult
        that says so explicitly -- this is the actual defect in #1102: a
        preregistered study silently comparing a truncated arm against a
        complete one with no signal either result was partial."""
        results = dict(_BACKTEST_RESULTS)
        results.update(
            {
                "early_stopped": True,
                "drawdown_cap_mode": "enforce",
                "drawdown_cap_breached": True,
                "drawdown_cap_breach_date": datetime(2026, 1, 5, tzinfo=UTC),
                "drawdown_cap_threshold": 0.20,
            }
        )
        _, result = _run_and_capture_risk_params(_NoOverrideStrategy(), results=results)
        assert result.early_stopped is True
        assert result.drawdown_cap_mode == "enforce"
        assert result.drawdown_cap_breached is True
        assert result.drawdown_cap_breach_date == datetime(2026, 1, 5, tzinfo=UTC)
        assert result.drawdown_cap_threshold == 0.20

    def test_complete_backtester_result_defaults_result_markers_false(self):
        """A run that never breached the cap must report early_stopped=False
        explicitly, not merely omit the field -- the whole point of #1102 is
        that "no marker" and "definitely not truncated" must never look the
        same."""
        _, result = _run_and_capture_risk_params(_NoOverrideStrategy())
        assert result.early_stopped is False
        assert result.drawdown_cap_breached is False
