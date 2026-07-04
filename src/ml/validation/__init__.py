"""Bear-market model-validation gate (#801).

Scores a candidate ML model on fixed historical bear/crash/chop windows and
blocks promotion of the ``latest`` symlink when it fails a configurable
drawdown threshold. See ``bear_validation`` for the harness and ``gate`` for
the promotion decision + audit trail.
"""

from src.ml.validation.bear_validation import (
    BearValidationHarness,
    BearWindow,
    ValidationReport,
    WindowScore,
    load_validation_windows,
)
from src.ml.validation.gate import (
    GateDecision,
    default_resolve_latest,
    promote_version_if_valid,
    validate_candidate,
    write_audit_record,
)

__all__ = [
    "BearValidationHarness",
    "BearWindow",
    "GateDecision",
    "ValidationReport",
    "WindowScore",
    "default_resolve_latest",
    "load_validation_windows",
    "promote_version_if_valid",
    "validate_candidate",
    "write_audit_record",
]
