from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cli.core.codex_workflow import run_auto_review


def _handle_auto_review(ns: argparse.Namespace) -> int:
    plan_path = Path(ns.plan_path).resolve() if ns.plan_path else None
    review_schema = Path(ns.review_schema).resolve()
    checks = list(ns.checks) if ns.checks else None

    return run_auto_review(
        plan_path=plan_path,
        checks=checks,
        max_iterations=ns.max_iterations,
        profile=ns.profile,
        review_schema_path=review_schema,
        cwd=Path.cwd(),
        dangerous_fix=ns.dangerous_fix,
        compare_branch=ns.compare_branch,
        python_bin=ns.python_bin,
    )


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("codex", help="Codex-assisted automation workflows")
    sub = parser.add_subparsers(dest="codex_cmd", required=True)

    auto = sub.add_parser(
        "auto-review",
        help=(
            "Run validations, request a structured Codex code review, "
            "and iterate fixes until the review reports no findings."
        ),
    )
    auto.add_argument(
        "--plan-path",
        help="Path to the ExecPlan Markdown file to feed into Codex for additional context.",
    )
    auto.add_argument(
        "--check",
        dest="checks",
        action="append",
        help=(
            "Validation command to execute before each review iteration. "
            "Provide multiple times to run several checks. Defaults to 'make test' "
            "and 'make code-quality' when omitted."
        ),
    )
    auto.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of review/fix cycles to attempt before exiting with failure.",
    )
    auto.add_argument(
        "--profile",
        help="Optional Codex CLI profile configured in ~/.codex/config.toml.",
    )
    auto.add_argument(
        "--review-schema",
        default=Path("cli/core/schemas/codex_review.schema.json"),
        help="JSON schema file used to validate the structured review output.",
    )
    auto.add_argument(
        "--dangerous-fix",
        action="store_true",
        help=(
            "Use '--dangerously-bypass-approvals-and-sandbox' during fix iterations. "
            "By default the workflow runs Codex in '--full-auto' mode."
        ),
    )
    auto.add_argument(
        "--compare-branch",
        default="develop",
        help=(
            "Name of the branch to diff against when gathering review context. "
            "Set to an empty string to disable diff injection."
        ),
    )
    auto.add_argument(
        "--python-bin",
        default=sys.executable or "python3",
        help="Python interpreter to inject as the PYTHON env var for make-based validation commands.",
    )
    auto.set_defaults(func=_handle_auto_review)
