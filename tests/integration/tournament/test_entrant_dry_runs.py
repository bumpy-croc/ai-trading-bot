"""End-to-end acceptance tests for the TARGET-REDESIGN tournament (GH #933).

Phase 2b item 6: one dry-run per entrant, each proving the FULL path works,
not just its individual pieces in isolation: tiny synthetic OHLCV -> train
via the REAL CLI entrypoint (`atb train cloud --provider local`, 1-2
epochs) -> artifact with correct metadata -> exam backtest via the REAL
backtest CLI path (`atb backtest exam_<entrant>`) -> completes with >=1
trade and sane outputs.

Run in a genuinely separate PROCESS, not in-process via pytest function
calls -- two independent reasons force this:

1. tests/conftest.py unconditionally stubs the entire onnxruntime module
   for the whole pytest session (InferenceSession.run always returns a
   fixed [[[0.0]]] constant, installed once at collection time before any
   test runs). Under that stub, entrant (c)'s ternary output (3 values
   expected, the stub returns 1) always raises inside
   OnnxRunner._process_classification_output -> the exam signal generator's
   own try/except swallows it -> permanent HOLD -> zero trades. Entrant
   (d)'s regression path reads a constant predicted_return=0.0 -> never
   crosses either entry threshold -> permanent HOLD. Entrant (a)'s gate
   sees P(profitable)=0.0 always -> below any sane min_confidence ->
   permanent HOLD. Only a genuinely separate process, where the real
   onnxruntime (a declared, actually-installed dependency) loads, exercises
   real inference and can produce a real, non-degenerate trade count.
2. "train via the REAL CLI entrypoint" / "backtest via the REAL backtest
   CLI path" -- these tests invoke `python -m cli.__main__ train cloud ...`
   / `python -m cli.__main__ backtest ...` as actual subprocesses, not
   `_handle_cloud`/`_handle` called directly, so the tests exercise argparse
   parsing, env var handling, and process exit codes exactly as a human
   operator running the tournament for real would.

Why `python -m cli.__main__` and not the installed `atb` console script:
`atb`'s entry point resolves `cli.__main__` via whatever is importable from
sys.path, which for an editable install can point at a DIFFERENT checkout
entirely (verified empirically while building this file: `atb` in this
worktree's venv silently ran a stale `cli` package and rejected every new
flag these tests depend on). `python -m cli.__main__` with `cwd=REPO_ROOT`
guarantees this worktree's own `cli/` package is what actually runs.

Each test cleans up its own `src/ml/models/{SYMBOL}/` directory in
teardown -- these are real, if small, artifacts written into the actual
registry path (TrainingPaths.default() is not overridable via any CLI flag
today), so a distinctive ACCT<letter> symbol per entrant keeps them from
ever colliding with real BTCUSDT/ETHUSDT data.

Bugs found and fixed by actually running this suite against the wiring
(Phase 2b): a registry-namespace mismatch in the exam strategy factories
(exam_target_redesign.py, "registry_model_type" fix), a non-numeric-metric
crash in `atb train cloud`'s completion summary (train_cloud.py), a missing
top-level "timeframe" key in every bundle's metadata.json (pipeline.py --
made EVERY trained bundle unfindable by symbol/timeframe/model_type lookup,
not just target-redesign ones), and a risk_overrides["position_sizer"]
string collision that silently zeroed every trade regardless of signal
confidence or direction (exam_target_redesign.py, "fixed_fraction" fix).
None of these would have been caught by unit tests mocking the pieces in
isolation -- this is why the coordinator called these tests first-class.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_ROOT = REPO_ROOT / "src" / "ml" / "models"

TRAIN_TIMEOUT_SECONDS = 240
BACKTEST_TIMEOUT_SECONDS = 120
PROMOTE_TIMEOUT_SECONDS = 60

# Common to every entrant's training run: --provider local (in-process,
# synchronous, no real AWS), --mock-data (deterministic synthetic OHLCV, no
# network -- see ingestion.py::load_training_corpus's use_mock_data branch),
# --force-price-only (skips sentiment entirely, the simplest/fastest path;
# also what determines the "price" registry namespace these tests resolve
# artifacts from), --input-data-s3 (a placeholder that makes orchestrator.py
# skip its own real-S3 upload step entirely -- local provider never reads
# this URI, see providers/local.py).
_COMMON_TRAIN_ARGS = [
    "--provider",
    "local",
    "--mock-data",
    "--days",
    "20",
    "--sequence-length",
    "120",
    "--force-price-only",
    "--input-data-s3",
    "s3://acceptance-test-dummy/dummy.csv",
]


def _run_cli(
    args: list[str], *, extra_env: dict[str, str] | None = None, timeout: int
) -> subprocess.CompletedProcess[str]:
    """Invoke `python -m cli.__main__ <args>` as a real subprocess."""
    env = os.environ.copy()
    # CloudTrainingConfig.from_env requires this even for --provider local
    # (LocalProvider treats output_s3_path as a local path, never actually
    # touches S3) -- any non-empty value satisfies the check.
    env.setdefault("SAGEMAKER_S3_BUCKET", "acceptance-test-dummy-bucket")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "cli.__main__", *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _cleanup_symbol(symbol: str) -> None:
    path = MODELS_ROOT / symbol.upper()
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _latest_version_id(symbol: str, registry_model_type: str) -> str:
    latest = MODELS_ROOT / symbol.upper() / registry_model_type / "latest"
    assert latest.is_symlink(), (
        f"No 'latest' symlink at {latest} -- training did not produce a "
        f"registered artifact. Check the training subprocess's stdout/stderr."
    )
    return latest.resolve().name


def _read_metadata(symbol: str, registry_model_type: str, version_id: str) -> dict:
    metadata_path = (
        MODELS_ROOT / symbol.upper() / registry_model_type / version_id / "metadata.json"
    )
    assert metadata_path.exists(), f"metadata.json missing at {metadata_path}"
    return json.loads(metadata_path.read_text())


def _parse_total_trades(stdout: str) -> int:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Total Trades:"):
            return int(stripped.split(":", 1)[1].strip())
    raise AssertionError(f"Could not find 'Total Trades:' line in backtest output:\n{stdout}")


class TestEntrantBinaryDirectionDryRun:
    """Entrant (b): binary fixed-horizon direction classification (tft)."""

    SYMBOL = "ACCTB"

    def teardown_method(self) -> None:
        _cleanup_symbol(self.SYMBOL)

    def test_train_and_backtest_end_to_end(self) -> None:
        train = _run_cli(
            [
                "train",
                "cloud",
                self.SYMBOL,
                *_COMMON_TRAIN_ARGS,
                "--epochs",
                "2",
                "--model-type",
                "tft",
                "--target-type",
                "binary_direction",
                "--target-horizon",
                "1",
            ],
            timeout=TRAIN_TIMEOUT_SECONDS,
        )
        assert train.returncode == 0, train.stdout + train.stderr

        version_id = _latest_version_id(self.SYMBOL, "price")
        metadata = _read_metadata(self.SYMBOL, "price", version_id)
        assert metadata["task_type"] == "binary_classification"
        assert metadata["class_labels"] == [-1, 1]
        assert metadata["timeframe"] == "1h"

        backtest = _run_cli(
            [
                "backtest",
                "exam_binary_direction",
                "--symbol",
                self.SYMBOL,
                "--mock-data",
                "--days",
                "20",
                "--timeframe",
                "1h",
                "--no-cache",
            ],
            extra_env={"ATB_MODEL_VERSION_OVERRIDE": version_id},
            timeout=BACKTEST_TIMEOUT_SECONDS,
        )
        assert backtest.returncode == 0, backtest.stdout + backtest.stderr
        assert _parse_total_trades(backtest.stdout) >= 1


class TestEntrantTripleBarrierDryRun:
    """Entrant (c): triple-barrier ternary classification (tft_ternary)."""

    SYMBOL = "ACCTC"

    def teardown_method(self) -> None:
        _cleanup_symbol(self.SYMBOL)

    def test_train_and_backtest_end_to_end(self) -> None:
        train = _run_cli(
            [
                "train",
                "cloud",
                self.SYMBOL,
                *_COMMON_TRAIN_ARGS,
                "--epochs",
                "2",
                "--model-type",
                "tft_ternary",
                "--target-type",
                "triple_barrier",
            ],
            timeout=TRAIN_TIMEOUT_SECONDS,
        )
        assert train.returncode == 0, train.stdout + train.stderr

        version_id = _latest_version_id(self.SYMBOL, "price")
        metadata = _read_metadata(self.SYMBOL, "price", version_id)
        assert metadata["task_type"] == "ternary_classification"
        assert metadata["class_labels"] == [-1, 0, 1]
        assert metadata["timeframe"] == "1h"

        backtest = _run_cli(
            [
                "backtest",
                "exam_triple_barrier",
                "--symbol",
                self.SYMBOL,
                "--mock-data",
                "--days",
                "20",
                "--timeframe",
                "1h",
                "--no-cache",
            ],
            extra_env={"ATB_MODEL_VERSION_OVERRIDE": version_id},
            timeout=BACKTEST_TIMEOUT_SECONDS,
        )
        assert backtest.returncode == 0, backtest.stdout + backtest.stderr
        assert _parse_total_trades(backtest.stdout) >= 1


class TestEntrantSmoothedReturnDryRun:
    """Entrant (d): smoothed forward return (cnn_lstm -- the incumbent's own
    regression architecture, isolating the target reformulation)."""

    SYMBOL = "ACCTD"

    def teardown_method(self) -> None:
        _cleanup_symbol(self.SYMBOL)

    def test_train_and_backtest_end_to_end(self) -> None:
        train = _run_cli(
            [
                "train",
                "cloud",
                self.SYMBOL,
                *_COMMON_TRAIN_ARGS,
                "--epochs",
                "2",
                "--model-type",
                "cnn_lstm",
                "--target-type",
                "smoothed_return",
                "--target-horizon",
                "6",
            ],
            timeout=TRAIN_TIMEOUT_SECONDS,
        )
        assert train.returncode == 0, train.stdout + train.stderr

        version_id = _latest_version_id(self.SYMBOL, "price")
        metadata = _read_metadata(self.SYMBOL, "price", version_id)
        assert metadata["task_type"] == "regression"
        assert "target_distribution" in metadata

        backtest = _run_cli(
            [
                "backtest",
                "exam_smoothed_return",
                "--symbol",
                self.SYMBOL,
                "--mock-data",
                "--days",
                "20",
                "--timeframe",
                "1h",
                "--no-cache",
            ],
            extra_env={"ATB_MODEL_VERSION_OVERRIDE": version_id},
            timeout=BACKTEST_TIMEOUT_SECONDS,
        )
        assert backtest.returncode == 0, backtest.stdout + backtest.stderr
        assert _parse_total_trades(backtest.stdout) >= 1


@pytest.mark.timeout(420)
class TestEntrantMetaLabelDryRun:
    """Entrant (a): meta-labeling secondary classifier.

    Two-stage, unlike the other three: (1) train a PRIMARY regression model
    (lands under the "price" namespace, same as any other --force-price-only
    run), (2) promote it into the "basic" namespace via `atb train
    cloud-promote` (create_exam_meta_label_strategy's primary_model_type
    defaults to "basic", matching MLBasicSignalGenerator's own default --
    call_strategy_factory only threads `symbol` through _load_strategy, so
    there is no CLI way to override it for a backtest-CLI-driven test; the
    fix is to land the primary bundle in the default namespace rather than
    fight that), (3) train the meta_label classifier, which runs the
    promoted primary forward to find its fire points.

    Overrides pytest.ini's global 120s per-test timeout (pytest-timeout) --
    unlike the other three entrants' single train+backtest pair, this test
    makes THREE sequential subprocess calls (primary train, promote, meta
    train) plus the backtest, each paying its own ~10-20s Python/TF import
    overhead on top of actual work.
    """

    SYMBOL = "ACCTA"

    def teardown_method(self) -> None:
        _cleanup_symbol(self.SYMBOL)

    def test_train_and_backtest_end_to_end(self) -> None:
        primary_train = _run_cli(
            [
                "train",
                "cloud",
                self.SYMBOL,
                *_COMMON_TRAIN_ARGS,
                "--epochs",
                "2",
                "--model-type",
                "cnn_lstm",
                "--target-type",
                "regression",
            ],
            timeout=TRAIN_TIMEOUT_SECONDS,
        )
        assert primary_train.returncode == 0, primary_train.stdout + primary_train.stderr

        primary_version = _latest_version_id(self.SYMBOL, "price")

        promote = _run_cli(
            [
                "train",
                "cloud-promote",
                self.SYMBOL,
                primary_version,
                "--from",
                "price",
                "--to",
                "basic",
                "--set-latest",
            ],
            timeout=PROMOTE_TIMEOUT_SECONDS,
        )
        assert promote.returncode == 0, promote.stdout + promote.stderr

        # A wider window than the other three entrants' 20 days: the
        # primary's fires need to span enough of the (seed=42, deterministic
        # for every --mock-data run) synthetic walk's up/down phases for
        # BOTH profitable and unprofitable outcomes to appear -- with only
        # 20 days, every fire in this corpus happened to resolve the same
        # way, and _run_meta_label_pipeline correctly refuses to fit
        # LogisticRegression on a single-class corpus (see pipeline.py's
        # own guard, added in Phase 2b item 3).
        meta_train = _run_cli(
            [
                "train",
                "cloud",
                self.SYMBOL,
                "--provider",
                "local",
                "--mock-data",
                "--days",
                "60",
                "--sequence-length",
                "120",
                "--force-price-only",
                "--input-data-s3",
                "s3://acceptance-test-dummy/dummy.csv",
                "--epochs",
                "1",
                "--target-type",
                "meta_label",
                "--primary-model-type",
                "basic",
            ],
            timeout=TRAIN_TIMEOUT_SECONDS,
        )
        assert meta_train.returncode == 0, meta_train.stdout + meta_train.stderr

        meta_version = _latest_version_id(self.SYMBOL, "meta_label")
        metadata = _read_metadata(self.SYMBOL, "meta_label", meta_version)
        assert metadata["task_type"] == "binary_classification"
        assert metadata["class_labels"] == [0, 1]
        assert metadata["primary_model_type"] == "basic"
        assert metadata["timeframe"] == "1h"

        backtest = _run_cli(
            [
                "backtest",
                "exam_meta_label",
                "--symbol",
                self.SYMBOL,
                "--mock-data",
                "--days",
                "60",
                "--timeframe",
                "1h",
                "--no-cache",
            ],
            extra_env={"ATB_MODEL_VERSION_OVERRIDE": meta_version},
            timeout=BACKTEST_TIMEOUT_SECONDS,
        )
        assert backtest.returncode == 0, backtest.stdout + backtest.stderr
        assert _parse_total_trades(backtest.stdout) >= 1
