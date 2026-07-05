"""Model-promotion gate (#801).

Decides whether a candidate model version may keep the ``latest`` symlink,
based on the bear-market validation harness, and leaves an auditable JSON
record next to the model version.

Why flip-then-validate-then-rollback? The prediction registry resolves a
model purely by the ``latest`` symlink (``PredictionModelRegistry._scan_registry``
reads ``base/{symbol}/{type}/latest``) — there is no per-call "load this exact
version" override. So to score the *candidate* (not the model already live) the
gate atomically points ``latest`` at the candidate, runs the harness (each
backtest builds a fresh registry that picks up the new symlink), and rolls
``latest`` back to the previous target if the candidate fails. This is a
canary-with-rollback: bad models are reverted, so it is fail-safe.

Transient-exposure caveat: between the flip and a rollback, ``latest`` briefly
points at an unvalidated candidate. This is acceptable because (a) these are
operator actions (``deploy-model`` / ``train --auto-deploy``), and (b) the live
engine caches its loaded model and only reloads on an explicit swap/restart, so
it does not hot-pick a transient flip. The window is a single validation run.

The symlink flip is delegated to injected callables so this module stays
independent of the CLI and testable without a real registry.

Deviation from the plan noted for reviewers: the plan suggested routing through
``src.experiments.promotion.PromotionManager``. That manager is bound to
experiment *suites* and a ledger of strategy-parameter variants, not model
files, so this gate writes a purpose-built ``validation_audit.json`` next to the
model version instead — same intent (auditable, reproducible promotion
decision), correct granularity.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.config.config_manager import get_config
from src.config.constants import DEFAULT_VALIDATION_REQUIRED
from src.ml.validation.bear_validation import BearValidationHarness, ValidationReport

logger = logging.getLogger(__name__)

AUDIT_FILENAME = "validation_audit.json"

# Type aliases for the injected symlink operations.
RepointFn = Callable[[Path], None]
ResolveLatestFn = Callable[[Path], "Path | None"]


@dataclass
class GateDecision:
    """Outcome of a promotion-gate evaluation."""

    passed: bool
    promoted: bool
    reason: str
    report: ValidationReport | None
    audit_path: Path | None


def _validation_required() -> bool:
    """Whether an un-runnable validation blocks promotion.

    Env override ``VALIDATION_REQUIRED`` (truthy) forces the strict behavior in
    environments that have data and must not promote unvalidated models.
    """
    return get_config().get_bool("VALIDATION_REQUIRED", DEFAULT_VALIDATION_REQUIRED)


def default_resolve_latest(registry_dir: Path) -> Path | None:
    """Return the version dir the ``latest`` symlink currently targets, or None."""
    latest = Path(registry_dir) / "latest"
    if not (latest.exists() or latest.is_symlink()):
        return None
    try:
        target_name = Path(os.readlink(latest)).name
    except OSError:
        return None
    return Path(registry_dir) / target_name


def _remove_latest(registry_dir: Path) -> None:
    """Remove the ``latest`` symlink if present (used when there is no prior target)."""
    latest = Path(registry_dir) / "latest"
    if latest.exists() or latest.is_symlink():
        try:
            latest.unlink()
        except OSError as exc:  # pragma: no cover - filesystem edge
            logger.error("Failed to remove 'latest' symlink at %s: %s", latest, exc)


def write_audit_record(
    version_dir: Path,
    report: ValidationReport | None,
    decision: dict,
) -> Path:
    """Write an auditable JSON record of the promotion decision.

    Args:
        version_dir: The model version directory being considered.
        report: The validation report (may be ``None`` for forced runs).
        decision: A small dict describing the decision (promoted, reason, ...).

    Returns:
        Path to the written audit file.
    """
    version_dir = Path(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "version": version_dir.name,
        "decision": decision,
        "report": report.to_dict() if report is not None else None,
    }
    audit_path = version_dir / AUDIT_FILENAME
    # Atomic write so a crashed run never leaves a half-written audit file.
    tmp_path = version_dir / f".{AUDIT_FILENAME}.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, audit_path)
    return audit_path


def _score_candidate(
    version_dir: Path,
    *,
    symbol: str,
    model_type: str,
    provider: str,
    repoint_fn: RepointFn,
    resolve_latest_fn: ResolveLatestFn,
    harness: BearValidationHarness | None,
    known_prev_target: Path | None = None,
) -> tuple[ValidationReport, Path | None]:
    """Flip ``latest`` to the candidate and score it; return (report, prev_target).

    Does NOT roll back — the caller decides whether to keep the candidate or
    restore ``prev_target``. ``prev_target`` is the version ``latest`` pointed at
    before the flip (``None`` if there was none).

    ``known_prev_target`` overrides resolution of the previous target. Callers
    that flipped ``latest`` themselves *before* invoking the gate (e.g. the
    training path, where training promotes the freshly trained version) must
    pass the pre-training target here — otherwise resolving now would return the
    candidate itself and a rollback would be a no-op.
    """
    registry_dir = version_dir.parent
    prev_target = (
        known_prev_target if known_prev_target is not None else resolve_latest_fn(registry_dir)
    )
    # Point latest at the candidate so the harness's fresh registry loads it.
    repoint_fn(version_dir)
    harness = harness or BearValidationHarness(
        symbol=symbol, model_type=model_type, provider=provider
    )
    report = harness.run()
    logger.info("%s", report.summary())
    return report, prev_target


def _rollback(
    registry_dir: Path,
    prev_target: Path | None,
    repoint_fn: RepointFn,
) -> None:
    """Restore ``latest`` to ``prev_target`` (or remove it when there was none)."""
    if prev_target is not None:
        repoint_fn(prev_target)
    else:
        _remove_latest(registry_dir)


def promote_version_if_valid(
    version_dir: Path,
    *,
    symbol: str,
    model_type: str,
    repoint_fn: RepointFn,
    resolve_latest_fn: ResolveLatestFn = default_resolve_latest,
    provider: str = "binance",
    force: bool = False,
    require_validation: bool | None = None,
    harness: BearValidationHarness | None = None,
    known_prev_target: Path | None = None,
) -> GateDecision:
    """Validate a candidate model version, keeping ``latest`` on it only if it passes.

    Args:
        version_dir: Path to the candidate version directory.
        symbol: Trading symbol the model serves.
        model_type: Registry model type (``basic``/``sentiment``).
        repoint_fn: Callable that points ``latest`` at a given version dir.
        resolve_latest_fn: Callable returning the current ``latest`` target dir.
        provider: Data provider for validation backtests.
        force: Skip validation and promote regardless (human override). Still
            writes an audit record marking the override.
        require_validation: If True, an inconclusive/un-runnable validation
            blocks promotion (rollback). Defaults to ``VALIDATION_REQUIRED``.
        harness: Injectable harness (tests). Defaults to a real harness.

    Returns:
        A :class:`GateDecision` describing what happened.
    """
    version_dir = Path(version_dir)
    strict = _validation_required() if require_validation is None else require_validation

    if force:
        repoint_fn(version_dir)
        audit = write_audit_record(
            version_dir,
            None,
            {"promoted": True, "forced": True, "reason": "human --force override"},
        )
        logger.warning(
            "Model %s/%s promoted with --force (validation skipped): %s",
            symbol,
            model_type,
            version_dir.name,
        )
        return GateDecision(
            passed=False, promoted=True, reason="forced", report=None, audit_path=audit
        )

    report, prev_target = _score_candidate(
        version_dir,
        symbol=symbol,
        model_type=model_type,
        provider=provider,
        repoint_fn=repoint_fn,
        resolve_latest_fn=resolve_latest_fn,
        harness=harness,
        known_prev_target=known_prev_target,
    )

    if report.passed:
        reason = "passed all validation windows"
        promoted = True
    elif report.inconclusive and not strict:
        reason = "validation inconclusive (could not run); promoted because validation not required"
        promoted = True
        logger.warning(
            "Model %s/%s kept as latest despite INCONCLUSIVE validation "
            "(VALIDATION_REQUIRED is off): %s",
            symbol,
            model_type,
            version_dir.name,
        )
    else:
        reason = (
            "validation inconclusive and validation is required"
            if report.inconclusive
            else "failed one or more validation windows"
        )
        promoted = False

    if not promoted:
        # Fail-safe: revert latest to whatever it pointed at before the flip.
        _rollback(version_dir.parent, prev_target, repoint_fn)
        logger.error(
            "Model %s/%s BLOCKED; rolled 'latest' back to %s: %s\n%s",
            symbol,
            model_type,
            prev_target.name if prev_target else "(none)",
            reason,
            report.summary(),
        )

    audit = write_audit_record(
        version_dir,
        report,
        {"promoted": promoted, "forced": False, "reason": reason},
    )
    return GateDecision(
        passed=report.passed,
        promoted=promoted,
        reason=reason,
        report=report,
        audit_path=audit,
    )


def validate_candidate(
    version_dir: Path,
    *,
    symbol: str,
    model_type: str,
    repoint_fn: RepointFn,
    resolve_latest_fn: ResolveLatestFn = default_resolve_latest,
    provider: str = "binance",
    harness: BearValidationHarness | None = None,
) -> ValidationReport:
    """Score a candidate version WITHOUT deploying: flip, validate, always roll back.

    Used by ``validate-model`` so an operator can pre-check a version without
    changing what is live.
    """
    version_dir = Path(version_dir)
    report, prev_target = _score_candidate(
        version_dir,
        symbol=symbol,
        model_type=model_type,
        provider=provider,
        repoint_fn=repoint_fn,
        resolve_latest_fn=resolve_latest_fn,
        harness=harness,
    )
    _rollback(version_dir.parent, prev_target, repoint_fn)
    return report
