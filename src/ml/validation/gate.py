"""Model-promotion gate (#801).

Decides whether a candidate model version may take the ``latest`` symlink,
based on the bear-market validation harness, and leaves an auditable JSON
record next to the model version. The actual symlink flip is delegated to an
injected ``repoint_fn`` so this module stays independent of the CLI and of any
particular engine (avoids a circular import with ``cli/commands/live.py``).

Deviation from the plan noted for reviewers: the plan suggested routing through
``src.experiments.promotion.PromotionManager``. That manager is tightly bound
to experiment *suites* and a ledger of strategy-parameter variants, not model
files, so reusing it here would be a poor fit. Instead this gate writes a
purpose-built ``validation_audit.json`` next to the model version — same intent
(an auditable, reproducible promotion decision), correct granularity.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.config.config_manager import get_config
from src.config.constants import DEFAULT_VALIDATION_REQUIRED
from src.ml.validation.bear_validation import BearValidationHarness, ValidationReport

logger = logging.getLogger(__name__)

AUDIT_FILENAME = "validation_audit.json"


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


def write_audit_record(
    version_dir: Path,
    report: ValidationReport | None,
    decision: dict,
) -> Path:
    """Write an auditable JSON record of the promotion decision.

    Args:
        version_dir: The model version directory being considered.
        report: The validation report (may be ``None`` for forced/skipped runs).
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
    import os

    os.replace(tmp_path, audit_path)
    return audit_path


def promote_version_if_valid(
    version_dir: Path,
    *,
    symbol: str,
    model_type: str,
    repoint_fn: Callable[[Path], None],
    provider: str = "binance",
    force: bool = False,
    require_validation: bool | None = None,
    harness: BearValidationHarness | None = None,
) -> GateDecision:
    """Validate a candidate model version and flip ``latest`` only if it passes.

    Args:
        version_dir: Path to the candidate version directory.
        symbol: Trading symbol the model serves.
        model_type: Registry model type (``basic``/``sentiment``).
        repoint_fn: Callable that flips the ``latest`` symlink to ``version_dir``.
        provider: Data provider for validation backtests.
        force: Skip validation and promote regardless (human override). Still
            writes an audit record marking the override.
        require_validation: If True, an inconclusive/un-runnable validation
            blocks promotion. Defaults to the ``VALIDATION_REQUIRED`` config.
        harness: Injectable harness (tests). Defaults to a real harness pointed
            at this version.

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

    harness = harness or BearValidationHarness(
        symbol=symbol, model_type=model_type, provider=provider, model_path=version_dir
    )
    report = harness.run()
    logger.info("%s", report.summary())

    if report.passed:
        repoint_fn(version_dir)
        reason = "passed all validation windows"
        promoted = True
    elif report.inconclusive and not strict:
        repoint_fn(version_dir)
        reason = "validation inconclusive (could not run); promoted because validation not required"
        promoted = True
        logger.warning(
            "Model %s/%s promoted despite INCONCLUSIVE validation (VALIDATION_REQUIRED is off): %s",
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
        logger.error(
            "Model %s/%s BLOCKED from promotion: %s\n%s",
            symbol,
            model_type,
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
