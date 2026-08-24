"""Boot-time wiring of the ratified risk limits into the three entry points.

Design §3.3 (fail-closed) and §3.5 (identical defaults for live, backtest and
harness). Each entry point must consult the ratified limits before touching a
provider or exchange, and each must resolve unset risk flags to ratified values
rather than to a drifted literal.
"""

from __future__ import annotations

import argparse

import pytest

from src.config.risk_limits import get_risk_limits

RATIFIED = get_risk_limits()


def _backtest_parser() -> argparse.ArgumentParser:
    from cli.commands import backtest as backtest_cmd

    parser = argparse.ArgumentParser()
    backtest_cmd.register(parser.add_subparsers(dest="command"))
    return parser


@pytest.mark.fast
class TestBacktestCliDefaultsResolveToRatified:
    """`atb backtest` with no risk flags must run at live-representative risk."""

    @pytest.mark.parametrize(
        ("dest", "expected"),
        [
            ("risk_per_trade", RATIFIED.position.base_risk_per_trade_pct),
            ("max_risk_per_trade", RATIFIED.position.max_risk_per_trade_pct),
            ("max_drawdown", RATIFIED.portfolio.max_drawdown_pct),
        ],
    )
    def test_unset_flag_resolves_to_ratified_value(self, dest: str, expected: float) -> None:
        from src.risk.risk_manager import RiskParameters

        ns = _backtest_parser().parse_args(["backtest", "ml_basic"])
        # Unset flags are sentinels, so RiskParameters supplies the ratified value.
        assert getattr(ns, dest) is None
        field = {"risk_per_trade": "base_risk_per_trade"}.get(dest, dest)
        assert getattr(RiskParameters(), field) == expected

    def test_explicit_flag_is_preserved(self) -> None:
        ns = _backtest_parser().parse_args(["backtest", "ml_basic", "--max-drawdown", "0.5"])
        assert ns.max_drawdown == 0.5


@pytest.mark.fast
class TestLiveRunnerDefaultsResolveToRatified:
    """The production start command passes no risk flags — these are its values."""

    def test_production_start_command_resolves_ratified_risk(self, monkeypatch) -> None:
        import sys

        from src.engines.live import runner
        from src.risk.risk_manager import RiskParameters

        # railway.json / Dockerfile CMD: `atb live-health hyper_growth --max-position 0.20`
        monkeypatch.setattr(sys, "argv", ["atb-live", "hyper_growth", "--max-position", "0.20"])
        args = runner.parse_args()
        assert args.risk_per_trade is None
        assert args.max_risk_per_trade is None
        assert args.max_drawdown is None

        overrides = {
            "base_risk_per_trade": args.risk_per_trade,
            "max_risk_per_trade": args.max_risk_per_trade,
            "max_drawdown": args.max_drawdown,
        }
        params = RiskParameters(**{k: v for k, v in overrides.items() if v is not None})

        # The live drawdown guard's cap. Confirmed against the deployed prod
        # boot log: "peak=$84.42, hard cap=20.0%".
        assert params.max_drawdown == RATIFIED.portfolio.max_drawdown_pct == 0.20
        assert params.base_risk_per_trade == 0.02
        assert params.max_risk_per_trade == 0.03

    def test_explicit_drawdown_flag_still_wins(self, monkeypatch) -> None:
        import sys

        from src.engines.live import runner

        monkeypatch.setattr(sys, "argv", ["atb-live", "hyper_growth", "--max-drawdown", "0.10"])
        assert runner.parse_args().max_drawdown == 0.10


@pytest.mark.fast
class TestEntryPointsFailClosed:
    """A missing/invalid limits file stops the engine before it reaches a venue."""

    def test_live_runner_main_aborts_before_strategy_load(self, monkeypatch) -> None:
        import sys

        from src.config.risk_limits import RiskLimitsError
        from src.engines.live import runner

        monkeypatch.setattr(sys, "argv", ["atb-live", "hyper_growth"])

        def _boom() -> None:
            raise RiskLimitsError("simulated invalid risk-limits.json")

        monkeypatch.setattr(runner, "get_risk_limits", _boom)

        # main() wraps its body in a broad except, so record the call instead of
        # raising inside it — an assertion there would be swallowed.
        loaded: list[object] = []
        monkeypatch.setattr(runner, "load_strategy", lambda *a, **k: loaded.append(a))
        monkeypatch.setattr(runner, "validate_configuration", lambda _args: True)

        with pytest.raises((RiskLimitsError, SystemExit)):
            runner.main()
        assert loaded == [], "strategy was loaded despite invalid risk limits"

    def test_experiment_runner_aborts_before_strategy_load(self, monkeypatch) -> None:
        from src.config.risk_limits import RiskLimitsError
        from src.experiments import runner as experiments_runner

        def _boom() -> None:
            raise RiskLimitsError("simulated invalid risk-limits.json")

        monkeypatch.setattr(experiments_runner, "get_risk_limits", _boom)

        harness = experiments_runner.ExperimentRunner.__new__(experiments_runner.ExperimentRunner)

        def _fail_load(*args, **kwargs):
            raise AssertionError("strategy loaded despite invalid risk limits")

        monkeypatch.setattr(
            experiments_runner.ExperimentRunner, "_load_strategy", _fail_load, raising=False
        )

        with pytest.raises(RiskLimitsError):
            experiments_runner.ExperimentRunner.run(harness, object())  # type: ignore[arg-type]
