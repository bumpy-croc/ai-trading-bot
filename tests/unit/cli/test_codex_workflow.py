from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cli.core import codex_workflow as cw


def test_truncate_adds_marker_when_needed() -> None:
    short = "abcd"
    assert cw._truncate(short, 10) == short

    long = "a" * 20
    truncated = cw._truncate(long, 5)
    assert truncated.startswith("a" * 5)
    assert truncated.endswith("...[truncated]...")


def test_slugify_command_generates_readable_tokens() -> None:
    assert cw._slugify_command("make test") == "make-test"
    assert cw._slugify_command("python -m cli codex auto-review").startswith("python-m-cli")


def test_summarise_validation_results_marks_failures(tmp_path: Path) -> None:
    log_ok = tmp_path / "ok.log"
    log_fail = tmp_path / "fail.log"
    log_ok.write_text("ok", encoding="utf-8")
    log_fail.write_text("fail", encoding="utf-8")

    results = [
        cw.CommandResult(
            command="echo pass",
            returncode=0,
            stdout="pass",
            stderr="",
            duration_seconds=0.1,
            log_path=log_ok,
        ),
        cw.CommandResult(
            command="echo fail",
            returncode=1,
            stdout="",
            stderr="boom",
            duration_seconds=0.2,
            log_path=log_fail,
        ),
    ]

    summary, ok = cw._summarise_validation_results(results)
    assert not ok
    assert "FAIL" in summary
    assert "echo fail" in summary
    assert str(log_fail) in summary


def test_render_findings_for_prompt_formats_output() -> None:
    findings = [
        {
            "file": "src/main.py",
            "line": 42,
            "severity": "major",
            "description": "Bug found",
            "recommendation": "Fix it",
        }
    ]
    rendered = cw._render_findings_for_prompt(findings)
    assert "[MAJOR]" in rendered
    assert "src/main.py:42" in rendered
    assert "Bug found" in rendered


def test_build_review_prompt_includes_diff() -> None:
    prompt = cw._build_review_prompt(
        plan_text="Plan text",
        validation_summary="All good",
        iteration=1,
        diff_context="diff content",
    )
    assert "diff content" in prompt
    assert "Plan text" in prompt


def test_gather_diff_context_without_branch(tmp_path: Path) -> None:
    result = cw._gather_diff_context(None, tmp_path)
    assert "No comparison branch" in result


def test_run_auto_review_zero_iterations_returns_success(tmp_path: Path) -> None:
    rc = cw.run_auto_review(
        plan_path=None,
        checks=["echo ok"],
        max_iterations=0,
        profile=None,
        review_schema_path=Path("cli/core/schemas/codex_review.schema.json"),
        cwd=tmp_path,
        python_bin=sys.executable,
    )
    assert rc == 0


def test_run_auto_review_negative_iterations_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        cw.run_auto_review(
            plan_path=None,
            checks=["echo ok"],
            max_iterations=-1,
            profile=None,
            review_schema_path=Path("cli/core/schemas/codex_review.schema.json"),
            cwd=tmp_path,
            python_bin=sys.executable,
        )


def test_python_shim_created_when_python_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = {"PATH": str(tmp_path)}
    shim = cw._ensure_python_on_path(sys.executable, env)
    try:
        assert shim is not None
        shim_python = Path(shim.name) / "python"
        assert shim_python.exists()
        assert env["PATH"].startswith(shim.name)
    finally:
        if shim is not None:
            shim.cleanup()
