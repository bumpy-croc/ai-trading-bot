"""Unit tests for the bear-market model-validation gate (#801)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from src.experiments.schemas import ExperimentConfig, ExperimentResult
from src.ml.validation.bear_validation import (
    BearValidationHarness,
    BearWindow,
    load_validation_windows,
)
from src.ml.validation.gate import (
    default_resolve_latest,
    promote_version_if_valid,
    validate_candidate,
    write_audit_record,
)

pytestmark = pytest.mark.unit


def _window(name: str = "w", max_dd: float = 30.0, min_trades: int = 1) -> BearWindow:
    return BearWindow(
        name=name,
        symbol="BTCUSDT",
        timeframe="1d",
        start=datetime(2022, 1, 1, tzinfo=UTC),
        end=datetime(2022, 6, 1, tzinfo=UTC),
        max_drawdown_pct=max_dd,
        min_trades=min_trades,
    )


class _StubRunner:
    """Runner double returning canned ExperimentResults keyed by call order."""

    def __init__(self, results: list[object]):
        self._results = list(results)
        self.calls: list[ExperimentConfig] = []

    def run(self, config: ExperimentConfig):
        self.calls.append(config)
        item = self._results[len(self.calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item


def _result(config: ExperimentConfig, *, max_dd: float, trades: int = 10) -> ExperimentResult:
    return ExperimentResult(
        config=config,
        total_trades=trades,
        win_rate=50.0,
        total_return=1.0,
        annualized_return=1.0,
        max_drawdown=max_dd,
        sharpe_ratio=0.5,
        final_balance=1010.0,
        trade_pnl_pcts=[0.01] * trades,
    )


# --- config loader ---------------------------------------------------------


def test_load_default_windows_config():
    windows = load_validation_windows()
    assert len(windows) == 3
    names = {w.name for w in windows}
    assert names == {"bear_2022", "crash_2025", "chop_2026"}
    for w in windows:
        assert w.end > w.start
        assert w.max_drawdown_pct > 0


def test_load_windows_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_validation_windows(tmp_path / "nope.json")


def test_load_windows_empty_list(tmp_path):
    cfg = tmp_path / "w.json"
    cfg.write_text(json.dumps({"windows": []}))
    with pytest.raises(ValueError, match="non-empty 'windows'"):
        load_validation_windows(cfg)


def test_load_windows_applies_defaults(tmp_path):
    cfg = tmp_path / "w.json"
    cfg.write_text(
        json.dumps(
            {
                "default_max_drawdown_pct": 33.0,
                "default_min_trades": 7,
                "windows": [
                    {"name": "x", "symbol": "BTCUSDT", "start": "2022-01-01", "end": "2022-02-01"}
                ],
            }
        )
    )
    (w,) = load_validation_windows(cfg)
    assert w.max_drawdown_pct == 33.0
    assert w.min_trades == 7


def test_bear_window_rejects_bad_range():
    with pytest.raises(ValueError, match="must be after start"):
        BearWindow(
            name="bad",
            symbol="BTCUSDT",
            timeframe="1d",
            start=datetime(2022, 6, 1, tzinfo=UTC),
            end=datetime(2022, 1, 1, tzinfo=UTC),
            max_drawdown_pct=30.0,
            min_trades=1,
        )


def test_bear_window_rejects_nonpositive_drawdown():
    with pytest.raises(ValueError, match="max_drawdown_pct"):
        _window(max_dd=0.0)


# --- harness ---------------------------------------------------------------


def test_harness_passes_when_all_windows_within_threshold():
    windows = [_window("a", max_dd=40.0), _window("b", max_dd=40.0)]
    captured: list[ExperimentConfig] = []

    def factory():
        runner = _StubRunner([])

        def run(config):
            captured.append(config)
            return _result(config, max_dd=20.0)

        runner.run = run  # type: ignore[method-assign]
        return runner

    harness = BearValidationHarness("BTCUSDT", "basic", provider="mock", runner_factory=factory)
    report = harness.run(windows)
    assert report.passed is True
    assert report.inconclusive is False
    assert all(s.passed for s in report.scores)


def test_harness_fails_on_excess_drawdown():
    windows = [_window("a", max_dd=25.0)]

    def factory():
        r = _StubRunner([])
        r.run = lambda config: _result(config, max_dd=60.0)  # type: ignore[method-assign]
        return r

    harness = BearValidationHarness("BTCUSDT", "basic", provider="mock", runner_factory=factory)
    report = harness.run(windows)
    assert report.passed is False
    assert report.scores[0].failures
    assert "drawdown" in report.scores[0].failures[0]


def test_harness_fails_on_too_few_trades():
    windows = [_window("a", max_dd=90.0, min_trades=5)]

    def factory():
        r = _StubRunner([])
        r.run = lambda config: _result(config, max_dd=5.0, trades=1)  # type: ignore[method-assign]
        return r

    harness = BearValidationHarness("BTCUSDT", "basic", provider="mock", runner_factory=factory)
    report = harness.run(windows)
    assert report.passed is False
    assert any("trades" in f for f in report.scores[0].failures)


def test_harness_inconclusive_when_backtest_raises():
    windows = [_window("a")]

    def factory():
        r = _StubRunner([])

        def run(config):
            raise RuntimeError("no data for window")

        r.run = run  # type: ignore[method-assign]
        return r

    harness = BearValidationHarness("BTCUSDT", "basic", provider="mock", runner_factory=factory)
    report = harness.run(windows)
    assert report.inconclusive is True
    assert report.passed is False
    assert report.scores[0].error is not None


# --- gate ------------------------------------------------------------------


class _PassingHarness:
    def run(self):
        from src.ml.validation.bear_validation import ValidationReport, WindowScore

        return ValidationReport(
            symbol="BTCUSDT",
            model_type="basic",
            strategy_name="ml_basic",
            provider="mock",
            created_at=datetime.now(UTC).isoformat(),
            scores=[WindowScore("a", 0.5, 10.0, 55.0, 10, 10.0, True, [])],
            passed=True,
            inconclusive=False,
        )


class _FailingHarness:
    def run(self):
        from src.ml.validation.bear_validation import ValidationReport, WindowScore

        return ValidationReport(
            symbol="BTCUSDT",
            model_type="basic",
            strategy_name="ml_basic",
            provider="mock",
            created_at=datetime.now(UTC).isoformat(),
            scores=[WindowScore("a", -1.0, 80.0, 20.0, 10, 10.0, False, ["max drawdown"])],
            passed=False,
            inconclusive=False,
        )


class _InconclusiveHarness:
    def run(self):
        from src.ml.validation.bear_validation import ValidationReport, WindowScore

        return ValidationReport(
            symbol="BTCUSDT",
            model_type="basic",
            strategy_name="ml_basic",
            provider="mock",
            created_at=datetime.now(UTC).isoformat(),
            scores=[WindowScore("a", 0.0, 0.0, 0.0, 0, None, False, ["backtest raised"], "boom")],
            passed=False,
            inconclusive=True,
        )


class _FakeRegistry:
    """Tracks the ``latest`` pointer so tests assert final state, not raw calls."""

    def __init__(self, initial=None):
        self.latest = initial
        self.repoints: list = []

    def repoint(self, version_dir):
        self.latest = version_dir
        self.repoints.append(version_dir)

    def resolve(self, _registry_dir):
        return self.latest


def _version_dir(tmp_path, name="2026-01-01_00h_v1"):
    d = tmp_path / "BTCUSDT" / "basic" / name
    d.mkdir(parents=True)
    return d


def _gate(version, harness, reg, **kw):
    return promote_version_if_valid(
        version,
        symbol="BTCUSDT",
        model_type="basic",
        repoint_fn=reg.repoint,
        resolve_latest_fn=reg.resolve,
        harness=harness,
        **kw,
    )


def test_gate_promotes_on_pass(tmp_path):
    version = _version_dir(tmp_path)
    old = _version_dir(tmp_path, "old_v0")
    reg = _FakeRegistry(initial=old)
    decision = _gate(version, _PassingHarness(), reg, require_validation=True)
    assert decision.promoted is True
    assert reg.latest == version  # candidate kept as latest
    audit = json.loads(decision.audit_path.read_text())
    assert audit["decision"]["promoted"] is True


def test_gate_blocks_and_rolls_back_on_fail(tmp_path):
    version = _version_dir(tmp_path)
    old = _version_dir(tmp_path, "old_v0")
    reg = _FakeRegistry(initial=old)
    decision = _gate(version, _FailingHarness(), reg, require_validation=True)
    assert decision.promoted is False
    assert reg.latest == old  # rolled back to the previously-live model
    assert decision.audit_path.exists()


def test_gate_soft_passes_inconclusive_when_not_required(tmp_path):
    version = _version_dir(tmp_path)
    reg = _FakeRegistry(initial=_version_dir(tmp_path, "old_v0"))
    decision = _gate(version, _InconclusiveHarness(), reg, require_validation=False)
    assert decision.promoted is True
    assert reg.latest == version


def test_gate_blocks_inconclusive_when_required(tmp_path):
    version = _version_dir(tmp_path)
    old = _version_dir(tmp_path, "old_v0")
    reg = _FakeRegistry(initial=old)
    decision = _gate(version, _InconclusiveHarness(), reg, require_validation=True)
    assert decision.promoted is False
    assert reg.latest == old


def test_gate_uses_known_prev_target_for_rollback(tmp_path):
    # Simulates the training path: training already flipped latest to the
    # candidate, so resolve() would return the candidate. known_prev_target
    # must drive the rollback instead.
    version = _version_dir(tmp_path)
    old = _version_dir(tmp_path, "old_v0")
    reg = _FakeRegistry(initial=version)  # training already promoted candidate
    decision = _gate(
        version, _FailingHarness(), reg, require_validation=True, known_prev_target=old
    )
    assert decision.promoted is False
    assert reg.latest == old  # rolled back to pre-training model, not candidate


def test_gate_force_promotes_without_validation(tmp_path):
    version = _version_dir(tmp_path)
    reg = _FakeRegistry()
    decision = promote_version_if_valid(
        version,
        symbol="BTCUSDT",
        model_type="basic",
        repoint_fn=reg.repoint,
        resolve_latest_fn=reg.resolve,
        force=True,
    )
    assert decision.promoted is True
    assert reg.latest == version
    audit = json.loads(decision.audit_path.read_text())
    assert audit["decision"]["forced"] is True


def test_validate_candidate_always_rolls_back(tmp_path):
    version = _version_dir(tmp_path)
    old = _version_dir(tmp_path, "old_v0")
    reg = _FakeRegistry(initial=old)
    report = validate_candidate(
        version,
        symbol="BTCUSDT",
        model_type="basic",
        repoint_fn=reg.repoint,
        resolve_latest_fn=reg.resolve,
        harness=_PassingHarness(),
    )
    assert report.passed is True
    assert reg.latest == old  # validate never leaves the candidate live


def test_default_resolve_latest_reads_symlink(tmp_path):
    type_dir = tmp_path / "BTCUSDT" / "basic"
    version = type_dir / "v1"
    version.mkdir(parents=True)
    (type_dir / "latest").symlink_to("v1")
    resolved = default_resolve_latest(type_dir)
    assert resolved is not None
    assert resolved.name == "v1"
    assert default_resolve_latest(tmp_path / "nonexistent") is None


def test_write_audit_record_atomic(tmp_path):
    version = _version_dir(tmp_path)
    path = write_audit_record(version, None, {"promoted": True, "reason": "x"})
    assert path.exists()
    assert json.loads(path.read_text())["decision"]["reason"] == "x"
