<!--
Sync Impact Report
Version: N/A -> 1.0.0
Modified principles:
- Template Principle 1 → Code Quality Is a Shipping Criterion
- Template Principle 2 → Testing Proves Every Change
- Template Principle 3 → Consistent User Experience
- Template Principle 4 → Performance Safeguards Decisions
Added sections:
- Quality Gate Requirements
- Delivery Workflow & Documentation
Removed sections:
- Principle 5 placeholder
Templates requiring updates:
- .specify/templates/plan-template.md ✅
- .specify/templates/spec-template.md ✅
- .specify/templates/tasks-template.md ✅
Follow-up TODOs:
- None
-->

# Crypto Trend-Following Trading Bot Constitution

## Core Principles

### Code Quality Is a Shipping Criterion
**Non-negotiable rules**
- Every change MUST keep `make code-quality` (Black, Ruff, MyPy, Bandit) green before review.
- All Python modules MUST be fully type hinted and document complex flows at module or function level.
- Domain boundaries in `src/` MUST remain acyclic; cross-domain imports require an architecture note in
  the relevant spec or plan justifying the dependency.
**Rationale**: This codebase powers trading decisions where silent regressions compound risk; disciplined
quality gates preserve maintainability and reviewer trust.

### Testing Proves Every Change
**Non-negotiable rules**
- `make test` (parallel pytest) MUST pass locally or in CI before merge; skipped tests require
  issue-linked justification.
- New behaviour MUST land with automated tests that fail without the change; choose unit, integration,
  and performance suites matching the impacted layer.
- `coverage.xml` MUST show non-decreasing overall line-rate (tolerance 0.5 percentage points); any
  regression requires a remediation plan captured in the PR description.
**Rationale**: Trading automation depends on deterministic evidence; tests provide the only defensible
proof that strategies and safeguards still behave as designed.

### Consistent User Experience
**Non-negotiable rules**
- CLI commands, dashboards, and docs MUST expose clear naming, flag semantics, and help text consistent
  with existing `atb` interfaces; new flows require README or docs updates before release.
- User-facing changes MUST capture success and error states via structured logging that references the
  command or dashboard module and includes remediation hints.
- Dashboards and CLIs MUST supply safe defaults (paper trading, dry runs, or preview outputs) unless the
  user explicitly opts into live execution.
**Rationale**: Consistency reduces operational load and enables traders and analysts to trust behaviour
across automation, dashboards, and manuals.

### Performance Safeguards Decisions
**Non-negotiable rules**
- Trading loops and backtests MUST maintain published baselines: p95 loop latency ≤500ms for
  single-symbol live trading and completion of `make backtest STRATEGY=ml_basic DAYS=30` within 5 minutes
  on reference hardware (8 vCPU / 16GB RAM). Deviations >10% require a documented mitigation timeline.
- Performance-sensitive code MUST emit metrics (structured logs or stats) that tie latency to strategy,
  exchange, and workload so regressions can be triaged.
- Any change touching `src/backtesting`, `src/live`, `src/performance`, or `tests/performance` MUST rerun
  the performance suite (`pytest tests/performance -q`) and attach results to the PR.
**Rationale**: Latency and throughput govern execution quality; explicit SLOs and instrumentation keep
strategies profitable and alerts actionable.

## Quality Gate Requirements
Teams MUST satisfy these checks before implementation work begins and again before requesting review:

- Align plan.md Constitution Check with all four principles, explicitly noting how code quality, testing,
  UX, and performance obligations are met or mitigated.
- Run `make code-quality`, `make test`, and relevant `pytest tests/performance` slices locally; CI logs
  MUST be linked in the PR.
- Update specs/tasks to include mandatory testing tasks, UX artefacts (docs, screenshots, CLI help), and
  performance validation steps; missing artefacts block merge.
- Document any exceptions (with expiry dates) in the PR checklist and track follow-up issues in `docs/`
  or `artifacts/` as appropriate.

## Delivery Workflow & Documentation
- Feature work MUST follow the `/specs/<feature>/` lifecycle: research → plan → spec → tasks, keeping
  user stories independently deliverable and testable.
- Plans MUST describe the affected domains under `src/` and how dependencies preserve the constitution's
  quality and performance guarantees.
- Specs MUST articulate measurable UX outcomes (CLI, dashboard, API) and include acceptance criteria that
  map to automated tests.
- Tasks MUST enumerate required test cases, documentation updates, and performance validation steps,
  enabling reviewers to correlate implementation progress with principle compliance.

## Governance
- Amendments require consensus from maintainers responsible for trading, ML, and platform domains plus a
  documented impact analysis; approve via PR referencing the analysis in `docs/`.
- Versioning follows semantic rules: MAJOR for removing or redefining principles, MINOR for adding new
  principles or clauses, PATCH for clarifications. Record version bumps in this file and PR titles.
- Compliance is reviewed during PR triage and quarterly audits; audits MUST sample merged work against
  each principle and track gaps in `docs/governance.md`.

**Version**: 1.0.0 | **Ratified**: 2025-10-14 | **Last Amended**: 2025-10-14
