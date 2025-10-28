# Automated Codex Review Loop

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

PLANS.md reference: `.agents/PLANS.md`. This document must comply with the guidelines described there.

## Purpose / Big Picture

Ship a repeatable command-line workflow that pairs the local Codex CLI with our repository conventions so that contributors can run a fully automated loop: (1) execute fast validation checks, (2) request a structured Codex code review, (3) hand the review findings to Codex for fixes, and (4) iterate until Codex reports no issues. After this change, a developer can run one command that keeps invoking Codex until the work is clean, with artifacts (validation transcripts, review findings) saved for post-run inspection.

## Progress

- [x] (2025-10-23 17:10Z) Captured repository context and drafted this ExecPlan scaffold.
- [x] (2025-10-23 17:40Z) Finalised CLI shape, validation defaults, and prompt orchestration strategy.
- [x] (2025-10-23 18:35Z) Implemented CLI command, schema, helpers, unit tests, and developer docs.
- [x] (2025-10-23 18:55Z) Captured validation artefacts, updated documentation, and recorded retrospective notes.
- [x] (2025-10-23 19:20Z) Added compare-branch diff context and clarified optional ExecPlan usage.
- [x] (2025-10-23 19:45Z) Injected Python interpreter into validation env and hardened review schema expectations.

## Surprises & Discoveries

- Observation: Running the CLI with the system Python 3.9 interpreter crashes because the codebase assumes 3.10+ union syntax.
  Evidence: `python -m cli codex auto-review --help` raised `TypeError: unsupported operand type(s) for |`.
  Mitigation: Use the project’s Python 3.11 environment (`python3.11` or `.venv/bin/python`) when invoking the workflow.

## Decision Log

- Decision: Manage the workflow through a new `python -m cli codex auto-review` command instead of an ad-hoc shell script so it benefits from existing CLI ergonomics.
  Rationale: Keeps tooling discoverable, reuses logging, and aligns with how other automation lives under `cli/`.
  Date/Author: 2025-10-23, Codex agent.
- Decision: Default validation commands to `atb test unit` and `atb dev quality`, with `--check` extensions when contributors need a tighter loop.
  Rationale: Matches the repository's standard pre-flight checks while remaining overridable for fast paths.
  Date/Author: 2025-10-23, Codex agent.
- Decision: Persist all validation transcripts and review payloads under `.codex/workflows/<timestamp>/`.
  Rationale: Provides auditable artefacts per run without polluting the repository root and mirrors existing Codex conventions.
  Date/Author: 2025-10-23, Codex agent.
- Decision: Always include a diff against a configurable comparison branch (default `develop`) in review/fix prompts.
  Rationale: Mirrors `codex /review` behaviour by grounding the conversation in the current branch’s delta when no ExecPlan is provided.
  Date/Author: 2025-10-23, Codex agent.
- Decision: Set the `PYTHON` env var for validation commands using the caller’s interpreter and appease Codex schema constraints by requiring all declared keys.
  Rationale: Avoids Makefile failures on machines without a `python` shim and keeps structured reviews compatible with the Codex API.
  Date/Author: 2025-10-23, Codex agent.
- Decision: Default the workflow to skip validations and instruct Codex to focus reviews/fixes strictly on files present in the diff context.
  Rationale: Users can still opt into targeted checks with `--check`, but the common case now mirrors `codex /review` (diff-only) for faster iterations and prevents repo-wide edits.
  Date/Author: 2025-10-24, Codex agent.

## Outcomes & Retrospective

The automated loop now lives under `python -m cli codex auto-review`, storing artefacts per run and enforcing structured review output. Unit tests cover helper logic and guard-rails, while manual smoke checks confirm the CLI wiring and early-exit behaviour. Next improvement opportunity: once Codex credentials are available in CI, add an integration test (or recorded transcript) that demonstrates a full review/fix cycle on a seeded failing scenario.

## Context and Orientation

The Codex CLI is available via the `codex` binary configured under `.codex/`. Project automation commands live under `cli/commands/`, with shared helpers in `cli/core/`. There is no existing Codex-specific subcommand today. Documentation for developers resides in `docs/`, with agent-specific guidance in `AGENTS.md`. Tests run through `atb test unit` (parallel pytest) and linting through `atb dev quality`. Our implementation will introduce:

- A new subcommand module under `cli/commands/` to expose the workflow.
- Supporting logic under `cli/core/` to orchestrate validation, review, and fix loops.
- A JSON schema (likely under `cli/core/schemas/`) that forces Codex review responses into a machine-readable structure.
- Documentation in `docs/` (either augmenting `development.md` or a new page within `docs/execplans/`) describing how to run the workflow and tune validation commands.

## Plan of Work

Describe the edits in order with enough detail for a newcomer:

1. Create `cli/core/codex_workflow.py` housing orchestration utilities: optionally running validation commands, constructing prompts that emphasise diff-only scope, invoking `codex exec` with schemas, and parsing responses. Include facilities for truncating command logs and summarising results.
2. Add a JSON schema file (e.g., `cli/core/schemas/codex_review.schema.json`) defining the structure of review outcomes with `summary` plus a `findings` array capturing file, optional line, severity, description, and recommendation.
    3. Implement `cli/commands/codex.py` that registers a `codex auto-review` subcommand, accepts CLI options (`--plan-path`, `--check`, `--max-iterations`, `--profile`, `--review-schema`, `--compare-branch`, `--workspace`), and delegates to the core module.
4. Update `cli/__main__.py` to register the new command module.
5. Ensure the workflow stores intermediate artifacts (review JSON plus optional validation logs when checks are configured) under a predictable directory (e.g., `.codex/workflows/<timestamp>/`) to aid manual inspection.
    6. Author documentation (likely `docs/development.md` or a dedicated `docs/automation.md`) covering prerequisites, default behavior, optional ExecPlan usage, customization, and safety notes about Codex running without approvals.
7. Write unit-level tests (probably under `tests/unit/cli/test_codex_workflow.py`) for pure-Python helpers such as validation summarisation and findings parsing so we have coverage without hitting the real Codex service.
    8. Run formatting (`ruff`, `black`), targeted tests, and capture outputs for the Outcomes section.

## Concrete Steps

Commands run or queued during implementation (repository root unless noted):

- `black cli/core/codex_workflow.py cli/commands/codex.py cli/__main__.py tests/unit/cli/test_codex_workflow.py` → reformatted new helper module.
- `ruff check cli/core/codex_workflow.py cli/commands/codex.py cli/__main__.py tests/unit/cli/test_codex_workflow.py` → all checks passed after import sorting.
- `pytest tests/unit/cli/test_codex_workflow.py` → 10 passed (Python 3.11).
    - `python3.11 -m cli codex auto-review --help` → verified CLI wiring and help text without contacting Codex.
    - `python3.11 -m cli codex auto-review --max-iterations 0 --check "echo noop" --python-bin $(which python3.11)` → exercised early-exit path and confirmed env injection works.
- (Planned) `python3.11 -m cli codex auto-review --plan-path docs/execplans/codex_auto_review.md --max-iterations 2` for an end-to-end dry run (add `--check ...` only when specific validations are required).

## Validation and Acceptance

Current validation status:

1. `python3.11 -m cli codex auto-review --help` prints the expected options, confirming the new subcommand is wired.
2. `pytest tests/unit/cli/test_codex_workflow.py` exercises prompt helpers, diff injection, shim handling, and guard-rails (10 passed).
3. `python3.11 -m cli codex auto-review --max-iterations 0` validates the CLI path that exits before invoking Codex and confirms that skipping validations is supported.
4. A future integrated dry run (with Codex credentials available) should demonstrate the full review/fix loop; document results once executed.

## Idempotence and Recovery

The CLI should be safe to rerun: it recreates its artifact folder per timestamp and overwrites temporary files. If Codex leaves partial edits, rerunning the command continues from the current working tree. Failures in validations or Codex invocations should propagate non-zero exit codes, allowing developers to intervene manually.

## Artifacts and Notes

Plan to capture:

- Validation command transcripts saved under `.codex/workflows/<timestamp>/validation_<step>.log` whenever checks are configured.
- Structured review output stored as JSON under the same directory.
- Final summary echoed to stdout for quick scanning.

These artifacts make the run auditable.

## Interfaces and Dependencies

- `cli/core/codex_workflow.py` must expose a `run_auto_review()` function that accepts parsed argparse options and performs the loop.
- The module should rely on Python's `subprocess` for command execution, `json` for schema parsing, and `tempfile` for Codex output capture. No third-party packages are required.
- Interaction with the Codex CLI occurs via `codex exec` with `--output-last-message` and `--output-schema`. The script may optionally allow the caller to provide a Codex profile (matching entries in `.codex/config.toml`).

## Change Log
- 2025-10-23 17:10Z — Initial ExecPlan drafted by Codex agent.
- 2025-10-23 18:40Z — Updated progress, decisions, concrete steps, and validation notes after implementing CLI workflow.
- 2025-10-23 18:55Z — Recorded validation outcomes, surprises, and retrospective after smoke-testing the CLI.
- 2025-10-23 19:20Z — Integrated diff context defaults and clarified ExecPlan optionality.
- 2025-10-23 19:45Z — Injected Python interpreter handling, updated docs, and refined schema requirements to satisfy Codex.
- 2025-10-23 20:05Z — Added python shim helper plus tests to ensure Makefile commands locate the interpreter.
- 2025-10-24 10:30Z — Defaulted the workflow to diff-only review/fix cycles, tightened prompts to forbid repo-wide edits, and refreshed docs/tests accordingly.
