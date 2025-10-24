from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

DEFAULT_CHECKS: list[str] = []
MAX_LOG_CHARS = 4000
PLAN_MAX_CHARS = 8000
DIFF_MAX_CHARS = 12000


@dataclass
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    log_path: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _slugify_command(command: str, limit: int = 40) -> str:
    slug = "".join(ch if ch.isalnum() else "-" for ch in command)
    slug = "-".join(filter(None, slug.split("-")))
    if not slug:
        slug = "command"
    if len(slug) > limit:
        slug = slug[:limit].rstrip("-")
    return slug


def _get_python_major_version(executable: str) -> int | None:
    """Return the major version of the supplied python executable or None on failure."""
    try:
        proc = subprocess.run(
            [executable, "-c", "import sys; print(sys.version_info[0])"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if proc.returncode != 0:
        return None

    output = (proc.stdout or proc.stderr or "").strip()
    try:
        return int(output.splitlines()[-1])
    except ValueError:
        return None


def _ensure_artifact_dir(base_dir: Path | None = None) -> Path:
    base = base_dir or Path(".codex/workflows")
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    path = base / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_log(artifact_dir: Path, filename: str, content: str) -> Path:
    path = artifact_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


def _load_plan_text(plan_path: Path | None) -> str:
    if plan_path is None:
        return "No ExecPlan provided."
    if not plan_path.exists():
        return f"ExecPlan not found at {plan_path}."
    text = plan_path.read_text(encoding="utf-8")
    return _truncate(text, PLAN_MAX_CHARS)


def _ensure_python_on_path(python_bin: str, env: dict[str, str]) -> TemporaryDirectory | None:
    existing_python = shutil.which("python", path=env.get("PATH"))
    python_path_str = shutil.which(python_bin) or python_bin
    python_path = Path(python_path_str).expanduser().resolve(strict=False)

    if not python_path.exists():
        raise FileNotFoundError(f"Specified python interpreter not found: {python_bin}")

    resolved_python = python_path

    need_shim = False
    if not existing_python:
        need_shim = True
    else:
        try:
            resolved_existing = Path(existing_python).resolve()
        except OSError:
            resolved_existing = None

        if resolved_existing != resolved_python:
            need_shim = True
        else:
            major = _get_python_major_version(existing_python)
            if major is not None and major < 3:
                need_shim = True

    if not need_shim:
        return None

    shim_dir = TemporaryDirectory(prefix="codex-python-shim-")
    shim_path = Path(shim_dir.name) / "python"
    shim_path.write_text(f'#!/bin/sh\n"{resolved_python}" "$@"\n', encoding="utf-8")
    os.chmod(shim_path, 0o755)
    env["PATH"] = f"{shim_dir.name}{os.pathsep}{env.get('PATH', '')}"
    return shim_dir


def _gather_diff_context(compare_branch: str | None, cwd: Path, limit: int = DIFF_MAX_CHARS) -> str:
    if not compare_branch:
        return "No comparison branch supplied; review current working tree as-is."

    def _run_git(args: list[str]) -> str:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("git command not found on PATH") from exc
        if proc.returncode != 0:
            return f"(git {' '.join(args)} failed with exit code {proc.returncode}: {proc.stderr.strip()})"
        return proc.stdout

    summary = _run_git(["diff", f"{compare_branch}...HEAD", "--stat"])
    patch = _run_git(["diff", f"{compare_branch}...HEAD", "--unified=0"])
    status = _run_git(["status", "--short"])

    if not summary.strip() and not patch.strip() and not status.strip():
        return f"Comparison branch: {compare_branch}\n\nNo differences detected. Working tree is clean."

    combined = textwrap.dedent(
        f"""\
        Comparison branch: {compare_branch}

        Diff summary (--stat):
        {summary.strip() or '(no changes)'}

        Unified diff (truncated):
        {patch.strip() or '(no changes)'}

        Working tree status:
        {status.strip() or '(no pending changes)'}
        """
    )
    return _truncate(combined, limit)


def _run_single_command(
    command: str,
    cwd: Path,
    artifact_dir: Path,
    index: int,
    env: dict[str, str],
) -> CommandResult:
    slug = _slugify_command(command)
    log_name = f"validation_{index:02d}_{slug}.log"
    print(f"→ Running validation command: {command}")
    start = time.perf_counter()
    proc = subprocess.run(
        ["bash", "-lc", command],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    duration = time.perf_counter() - start
    combined_output = textwrap.dedent(
        f"""\
        # Command
        {command}

        # Exit code
        {proc.returncode}

        # STDOUT
        {proc.stdout}

        # STDERR
        {proc.stderr}
        """
    )
    log_path = _write_log(artifact_dir, log_name, combined_output)
    return CommandResult(
        command=command,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_seconds=duration,
        log_path=log_path,
    )


def run_validation_commands(
    commands: Sequence[str],
    *,
    cwd: Path,
    artifact_dir: Path,
    env: dict[str, str],
) -> list[CommandResult]:
    results: list[CommandResult] = []
    for idx, command in enumerate(commands, start=1):
        results.append(
            _run_single_command(command, cwd=cwd, artifact_dir=artifact_dir, index=idx, env=env)
        )
    return results


def _summarise_validation_results(results: Sequence[CommandResult]) -> tuple[str, bool]:
    if not results:
        return "No validation commands configured.", True

    lines: list[str] = []
    all_ok = True
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        if not result.ok:
            all_ok = False
        lines.append(
            f"{status} | {result.command} | {result.duration_seconds:.1f}s | log: {result.log_path}"
        )
        if not result.ok:
            failure_excerpt = _truncate(
                (result.stderr or result.stdout or "").strip() or "(no output captured)",
                MAX_LOG_CHARS,
            )
            lines.append(textwrap.indent(failure_excerpt, prefix="    "))
    return "\n".join(lines), all_ok


def _render_findings_for_prompt(review_findings: Iterable[dict]) -> str:
    lines: list[str] = []
    for idx, finding in enumerate(review_findings, start=1):
        file_path = finding.get("file", "unknown")
        line = finding.get("line")
        location = f"{file_path}:{line}" if line else file_path
        severity = finding.get("severity", "unspecified").upper()
        description = finding.get("description", "").strip()
        recommendation = finding.get("recommendation", "").strip()
        lines.append(f"{idx}. [{severity}] {location}")
        if description:
            lines.append(f"   Issue: {description}")
        if recommendation:
            lines.append(f"   Recommendation: {recommendation}")
    return "\n".join(lines) if lines else "No previous findings."


def _build_review_prompt(
    *,
    plan_text: str,
    validation_summary: str,
    iteration: int,
    diff_context: str,
) -> str:
    return textwrap.dedent(
        f"""
        You are a senior reviewer for the ai-trading-bot repository. Assess the current working tree.

        ExecPlan context:
        {plan_text}

        Diff context versus target branch:
        {diff_context}

        Validation summary prior to review:
        {validation_summary}

        Review iteration: {iteration}

        Scope guardrails:
        - Focus on behaviours introduced or modified by the diff context above.
        - You may read other files for background, but only report findings caused or revealed by the diff.
        - If the diff does not introduce problems, respond with no findings.

        Produce a JSON object that strictly matches the provided schema. Only include actionable
        findings (correctness, security, performance, testing gaps). If the codebase looks good,
        return an empty findings list and set the summary to an affirmative statement.
        """
    ).strip()


def _build_fix_prompt(
    *,
    plan_text: str,
    validation_summary: str,
    findings: list[dict],
    commands: Sequence[str],
    iteration: int,
    diff_context: str,
) -> str:
    findings_section = _render_findings_for_prompt(findings)
    checks_text = "\n".join(f"- {command}" for command in commands) or "(no validation commands)"
    return textwrap.dedent(
        f"""
        You are acting as an autonomous fixer for the ai-trading-bot repository. Address every
        review finding and validation failure below.

        ExecPlan context:
        {plan_text}

        Diff context versus target branch:
        {diff_context}

        Outstanding review findings:
        {findings_section}

        Latest validation summary:
        {validation_summary}

        Validation commands you must keep green after your edits:
        {checks_text}

        Expectations:
        1. Apply code changes directly in the repository.
        2. Re-run the validation commands listed above to confirm success.
        3. Update the ExecPlan progress if you make meaningful milestones (optional but encouraged).
        4. Conclude with a concise summary of actions taken and test outcomes.
        5. Limit edits to files/lines from the diff unless resolving the diff's issues requires touching a closely-related dependency.

        You are entering fix iteration {iteration}. Stop once all findings are resolved or you have
        no further actionable changes.
        """
    ).strip()


def _invoke_codex_exec(
    prompt: str,
    *,
    cwd: Path,
    profile: str | None,
    output_schema: Path | None,
    additional_args: Sequence[str] | None,
) -> str:
    cmd: list[str] = ["codex", "exec", "--output-last-message"]
    with NamedTemporaryFile("r+", encoding="utf-8", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    cmd.append(str(temp_path))
    cmd.extend(["-C", str(cwd)])
    if profile:
        cmd.extend(["-p", profile])
    if output_schema:
        cmd.extend(["--output-schema", str(output_schema)])
    if additional_args:
        cmd.extend(additional_args)
    cmd.append("-")

    pretty_cmd = " ".join(arg for arg in cmd if arg != "-")
    print(f"→ Invoking Codex: {pretty_cmd} (prompt via stdin)")
    try:
        subprocess.run(
            cmd,
            input=prompt,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Codex execution failed with exit code {exc.returncode}") from exc
    try:
        output_text = temp_path.read_text(encoding="utf-8")
    finally:
        temp_path.unlink(missing_ok=True)
    return output_text.strip()


def _parse_review_output(raw_text: str, schema_path: Path) -> dict:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Codex review output is not valid JSON per {schema_path}: {exc}") from exc
    return data


def run_auto_review(
    *,
    plan_path: Path | None,
    checks: Sequence[str] | None,
    max_iterations: int,
    profile: str | None,
    review_schema_path: Path,
    cwd: Path | None = None,
    dangerous_fix: bool = False,
    compare_branch: str | None = None,
    python_bin: str | None = None,
) -> int:
    if max_iterations < 0:
        raise ValueError("max_iterations must be >= 0")

    if max_iterations == 0:
        print("Max iterations set to 0; exiting without invoking Codex.")
        return 0

    workdir = cwd or Path.cwd()
    artifact_dir = _ensure_artifact_dir()
    print(f"Artifacts for this run will be stored in {artifact_dir}")

    plan_text = _load_plan_text(plan_path)

    validation_commands = list(checks) if checks else list(DEFAULT_CHECKS)

    python_executable = python_bin or sys.executable or "python3"
    resolved_python = shutil.which(python_executable) or python_executable
    python_path = Path(resolved_python).expanduser().resolve(strict=False)
    if not python_path.exists():
        raise FileNotFoundError(f"Python interpreter not found: {python_executable}")

    env = os.environ.copy()
    env["PYTHON"] = str(python_path)

    shim_dir: TemporaryDirectory | None = None

    try:
        shim_dir = _ensure_python_on_path(str(python_path), env)
        last_review_payload: dict | None = None

        for iteration in range(1, max_iterations + 1):
            print(f"\n=== Automated Codex Loop :: Iteration {iteration}/{max_iterations} ===")
            diff_context = _gather_diff_context(compare_branch, workdir)
            if validation_commands:
                validation_results = run_validation_commands(
                    validation_commands, cwd=workdir, artifact_dir=artifact_dir, env=env
                )
                validation_summary, validations_ok = _summarise_validation_results(
                    validation_results
                )
                summary_log = _write_log(
                    artifact_dir,
                    f"validation_summary_{iteration:02d}.log",
                    validation_summary + "\n",
                )
                print(f"Validation summary recorded at {summary_log}")
            else:
                validation_results = []
                validation_summary = (
                    "Validation commands skipped; focusing on review/fix for current diff."
                )
                validations_ok = True
                summary_log = _write_log(
                    artifact_dir,
                    f"validation_summary_{iteration:02d}.log",
                    validation_summary + "\n",
                )
                print("Validation commands skipped (no --check provided).")

            review_prompt = _build_review_prompt(
                plan_text=plan_text,
                validation_summary=validation_summary,
                iteration=iteration,
                diff_context=diff_context,
            )
            raw_review_output = _invoke_codex_exec(
                review_prompt,
                cwd=workdir,
                profile=profile,
                output_schema=review_schema_path,
                additional_args=None,
            )

            review_payload = _parse_review_output(raw_review_output, review_schema_path)
            review_path = artifact_dir / f"review_{iteration:02d}.json"
            review_path.write_text(json.dumps(review_payload, indent=2), encoding="utf-8")
            print(f"Stored structured review at {review_path}")

            findings = review_payload.get("findings", [])
            summary = review_payload.get("summary", "")
            print(f"Review summary: {summary}")
            print(f"Review findings count: {len(findings)}")

            if validations_ok and not findings:
                print("✅ Validations pass and Codex reported no findings. Workflow complete.")
                return 0

            last_review_payload = review_payload
            next_iteration = iteration + 1 if iteration < max_iterations else iteration

            fix_prompt = _build_fix_prompt(
                plan_text=plan_text,
                validation_summary=validation_summary,
                findings=findings,
                commands=validation_commands,
                iteration=next_iteration,
                diff_context=diff_context,
            )

            fix_args: list[str] = ["--full-auto"]
            if dangerous_fix:
                fix_args.append("--dangerously-bypass-approvals-and-sandbox")

            raw_fix_output = _invoke_codex_exec(
                fix_prompt,
                cwd=workdir,
                profile=profile,
                output_schema=None,
                additional_args=fix_args,
            )
            fix_path = artifact_dir / f"fix_{iteration:02d}.log"
            fix_path.write_text(raw_fix_output + "\n", encoding="utf-8")
            print(f"Stored Codex fix transcript at {fix_path}")

            if iteration == max_iterations:
                print("Reached maximum iterations before achieving a clean review.")
                break

        if last_review_payload is not None:
            outstanding_path = artifact_dir / "outstanding_findings.json"
            outstanding_path.write_text(json.dumps(last_review_payload, indent=2), encoding="utf-8")
            print(f"Outstanding findings recorded at {outstanding_path}")
        print("❌ Automated Codex loop finished without clearing all findings.")
        return 1
    finally:
        if shim_dir is not None:
            shim_dir.cleanup()


__all__ = [
    "DEFAULT_CHECKS",
    "MAX_LOG_CHARS",
    "DIFF_MAX_CHARS",
    "CommandResult",
    "run_auto_review",
    "run_validation_commands",
]
