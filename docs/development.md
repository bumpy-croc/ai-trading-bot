# Development workflow

> **Last Updated**: 2025-12-14

This project ships a command-line interface and Makefile targets that standardise local setup, quality checks, and diagnostics.

## Environment setup

**Requirements**: Python 3.11 (the repo relies on 3.11-only typing features and the dev tooling now enforces it)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
make install            # install the CLI in editable mode
make deps-dev           # install development dependencies (pytest, ruff, mypy, etc.)
```

Run `atb dev setup` to execute helper scripts (pre-commit hooks, git config) used by maintainers; the helper now looks up `python3.11` (or `PYTHON311`) and recreates `.venv` with that interpreter if needed, so you never end up running tests on the macOS 3.9 default.

## Key shared directories

- `src/infrastructure/logging`: centralized logging config, context propagation, structured event helpers, and database decision logging.
- `src/infrastructure/runtime`: process bootstrap helpers (project paths, geo detection, cache TTL overrides, secret resolution).
- `src/sentiment`: adapters that merge provider sentiment data into OHLCV frames.
- `src/trading/symbols`: utilities for converting symbols between exchanges (e.g., `BTCUSDT` ↔ `BTC-USD`).

**Related**: See [Configuration](configuration.md) for environment variable setup.

## Railway deployment quick start

1. Install the Railway CLI (`npm install -g @railway/cli`) and authenticate with `railway login`.
2. From the project root run `railway init` and select the target environment, then provision PostgreSQL with `railway add postgresql`.
3. Set required variables (`BINANCE_API_KEY`, `BINANCE_API_SECRET`, `TRADING_MODE`, `INITIAL_BALANCE`, `DATABASE_URL`) via `railway variables set <KEY>=<VALUE>`.
4. Deploy the service with `railway up`; the workflow builds the container and applies environment variables automatically.
5. After the deploy succeeds, verify connectivity from your workstation:
   - `railway run atb db setup-railway --verify`
   - `railway run atb db backup --env production --backup-dir ./backups --retention 7`
6. Monitor logs (`railway logs --environment production`) and dashboards exposed by `atb live-health` to confirm the strategy is processing market data.

## Tests and diagnostics

- `atb test unit` – run unit tests with parallelism.
- `atb test integration` – run integration tests.
- `atb test all` – run the entire unit/integration suite.
- `pytest -q` – run tests directly with pytest.
- `atb tests heartbeat` – insert a `SystemEvent` row for monitoring pipelines.
- `atb tests db` – verify database connectivity end-to-end.
- `atb tests download` – smoke test data downloads via CCXT.
- `atb tests parse-junit tests/reports/unit.xml --label "Unit Tests"` – parse a JUnit XML report and print condensed failure summaries in CI logs.

## Test reproducibility guidelines

Regression tests must produce identical results across environments (local macOS, CI Linux, etc.). Follow these principles:

### Dependency injection for test isolation

When testing components that import and instantiate classes at runtime, use dependency injection instead of monkeypatching:

```python
# ❌ Bad: Monkeypatching doesn't work for runtime imports
monkeypatch.setattr("src.module.ClassName", StubClass)

# ✅ Good: Inject test doubles via constructor parameters
backtester = Backtester(
    strategy=test_strategy,
    _regime_switcher_class=StubRegimeStrategySwitcher,  # Internal testing param
    _strategy_manager=stub_manager,  # Can be class or instance
)
```

**Why this matters**: Monkeypatching module-level attributes fails when the target code imports the class inside a method (after the patch is applied). Dependency injection ensures test doubles are actually used.

### Deterministic test fixtures

- Use deterministic data: fixed seeds, static arrays, or deterministic generators
- Avoid system time (`datetime.now()`) - use fixed timestamps
- Pin all randomness with explicit seeds
- Use lowercase frequency strings in pandas (`freq="h"` not `freq="H"`) to avoid deprecation warnings

### Verifying determinism

Run regression tests 10 times consecutively to confirm identical results:

```bash
for i in {1..10}; do
  pytest tests/integration/backtesting/test_regime_regression.py -v || exit 1
done
```

All assertions should pass with exact floating-point equality (`==`) or very tight tolerances (`rel=1e-5, abs=1e-5`).

### Example: Regime regression test

The regime regression test (`tests/integration/backtesting/test_regime_regression.py`) demonstrates these principles:

1. **Deterministic fixtures**: `_build_fixture_dataframe()` creates reproducible OHLCV data
2. **Stub components**: `StubRegimeStrategySwitcher` and `StubStrategyManager` provide controlled behavior
3. **Dependency injection**: Backtester accepts `_regime_switcher_class` and `_strategy_manager` for testing
4. **Snapshot validation**: Results are compared against a committed JSON snapshot

## Code quality

- `atb dev quality` – run Black formatting, Ruff linting, MyPy type checks, and Bandit security scans.
- `atb dev clean` – remove caches and build artifacts.
- `black .` and `ruff check . --fix` – apply formatting and lint fixes manually.
- `python bin/run_mypy.py` – strict type checking without formatting.
- `bandit -c pyproject.toml -r src` – security audit focusing on runtime code.

The repository enforces Ruff/Black style in CI, so commit formatted code to avoid failures.

## Git hooks

Hook **sources are tracked in `.githooks/`** and are installed by `make install` (also
`make deps-dev` / `make deps-server`, which depend on it). To install or repair them on their
own:

```bash
make hooks          # symlink .githooks/* into the active hooks directory
make hooks-check    # report drift; non-zero exit if not installed
```

The installer respects `core.hooksPath` when set and otherwise targets `$GIT_COMMON_DIR/hooks`,
which is **shared by every linked worktree** of a checkout — install once per clone, not once
per worktree. Hooks that this repo does not ship are left untouched.

Hooks are **copied, not symlinked**, and the source is read from the primary checkout when it
has a `.githooks/`. Both rules exist because the hooks directory is shared while `make install`
often runs inside an ephemeral agent worktree: a symlink into such a worktree dangles as soon
as it is pruned, and **git skips a dangling hook silently, exiting 0** — reintroducing GH #1077
through its own remedy. The cost of copying is drift, which `make hooks-check` detects — comparing **content and
the executable bit**, because git ignores a non-executable hook and lets the push through —
and `make install` repairs.

`make hooks-check` inspects the hooks installed on *your machine*, so it is a workstation
check, not a repo-content one: a fresh CI clone has no hooks installed by design and would
always report drift. Run it locally if a push ever seems not to be running tests.

`pre-push` runs the fast-marked unit tests (in parallel, `-n 4` — the same worker count `tests/run_tests.py` uses; ~48s) and blocks the push when
they fail. It distinguishes a genuine test failure from an environment failure — a usage or
collection error is reported as such, not as "the tests failed". It resolves the
repository root with `git rev-parse --show-toplevel` (never the cwd) and looks for an
interpreter that can import `pytest`, `numpy`, `pandas` and `xdist` — an `import pytest` probe
alone would accept another project's venv and then die in collection — in order:
`$ATB_PREPUSH_PYTHON`, this checkout's `.venv`, the primary checkout's `.venv` (linked
worktrees have none of their own), `$VIRTUAL_ENV`, then `python3`/`python` on `PATH`.
**If no such interpreter exists the push fails** — a check that
cannot run must not report success (GH #1077).

To skip deliberately, use git's own escape hatch:

```bash
git push --no-verify
```

## Append-only files and the merge driver

`.claude/state/log.md` and `.claude/skills/weekly-retro/AGENDA.md` are written by every agent
and by the PM. Because entries are appended at the end, two branches that each record something
collide at EOF — a conflict resolved the same way every time ("keep both, chronological"). That
toil was measured at ~8 hand-resolutions in one session, and each one is a chance to silently
drop an entry from a record whose whole point is that entries are never dropped (GH #1079,
#1090).

`.gitattributes` maps those paths to a custom driver, `tools/merge_append_only.py`:

- it splits each side into **entries** and runs git's own three-way merge over them, so
  nothing on either side is dropped and the common ancestor is never duplicated. An entry
  starts at a marker (an H2 heading in `log.md`, a top-level bullet in `AGENDA.md`) that
  *also* carries a date *and* is preceded by a blank line. All three conditions are needed:
  entries are encouraged to quote an earlier entry's header, and `log.md` bodies routinely
  carry column-0 markup, so splitting on the marker alone would cut an entry in half. A line
  that fails any condition stays inside its entry, which at worst makes that entry look
  edited — a conflict, never a silent split;
- concurrent **appends** are all kept, de-duplicated, sorted by the timestamp in each entry's
  first line when every entry in the region carries one;
- **deletions are honoured**: the retro clearing `AGENDA.md` still clears it, while an item a
  concurrent branch appended survives the clear;
- **edits still conflict.** Entries are identified by their first line, so the same entry with
  two different bodies is an edit, not two appends: it stops the merge with real markers, as
  does an edit racing a deletion. This is deliberately *not* git's built-in `union` driver,
  which cannot tell the two apart and would ship both halves of a rewrite.

  Because identity *is* the first line, editing that line would otherwise read as
  delete-old-add-new and slip past those checks. The driver therefore also pairs a vanished
  entry with an arrived one by **body** similarity and conflicts on the pair. **Residual
  carve-out, stated plainly:** an entry whose first line *and* body are both rewritten
  substantially, or a single-line entry whose only line is edited, is still indistinguishable
  from a delete plus an unrelated append and will merge as one. Bodies are compared rather
  than whole entries because whole-entry comparison cannot separate the cases — on real data
  unrelated entries score 0.66 against 0.89 for a genuine edit, while by body the same cases
  are 0.38 against 1.00.

### Which files qualify

Only files whose existing content is never edited or removed in ordinary work. Adding anything
else would union-merge changes that should have conflicted. `docs/changelog.md` is deliberately
**excluded**: new entries are prepended inside shared `### Added` / `### Fixed` sections and the
`[Unreleased]` section is rewritten at release time, so two branches genuinely edit the same
region. Incident and proposal files are excluded for the same reason — their `status:`
frontmatter changes in place.

The rule is enforced twice on purpose. A path must appear in `.gitattributes` **and** in
`APPEND_ONLY_PATHS` in `tools/merge_append_only.py`; the driver falls back to a normal conflict
for any path it does not recognise, so a stray `.gitattributes` line cannot quietly union-merge
a file.

### Installing it, and telling when it is not active

Half of a merge driver cannot be version-controlled: `merge.append-only.driver` is a local git
config key, and when it is missing git **ignores the `.gitattributes` entry without saying so**
— the same invisible-absence failure as the inert pre-push hook of GH #1077 above.
Registration therefore runs from the same place as the hooks and the worktree shim, and has
its own drift check alongside `make hooks-check`:

```bash
make install               # registers the driver (along with everything else)
make merge-drivers         # register / repair on its own
make merge-drivers-check   # non-zero exit if unregistered or stale
```

Git config lives in the shared common dir, so registering once per **clone** covers every
linked worktree. The registered command is *relative* (`python3 tools/merge_append_only.py`),
unlike the hook installer's primary-checkout rule: git runs a merge driver from the top of the
worktree doing the merge, so a relative path always matches the code being merged, whereas an
absolute one would pin every worktree to a single checkout — including the primary, which is
held on the production branch and does not carry this tool at all until it ships there.

Like `make hooks-check`, `make merge-drivers-check` inspects **this machine's** state, so it is
not a CI gate: a fresh clone would fail it every run and a post-`make install` clone would pass
it tautologically. What CI does assert is repository *content* — a unit test requires
`.gitattributes` and the driver's own allowlist to name the same files, since a path in only
one of them is silently inert.

The symptom of an unregistered driver is exactly the old behaviour — a conflict in `log.md` on
an ordinary append, with ordinary markers. If you get one, run `make merge-drivers-check`
before resolving by hand.

A driver that is registered but cannot *run* is more dangerous, and the registration is shaped
around it. When a merge driver exits non-zero git records a conflict and stages `UU`, but git
does **not** write the markers — that is the driver's job. So a command that never runs leaves
the working-tree file as **ours' content verbatim, with no markers**, which is indistinguishable
from a clean, complete merge; resolving it with `git add` would drop theirs' entry silently.
That is reachable precisely because the path is relative (a linked worktree on a branch
predating the driver has no such script). The registered command is therefore guarded — it
tests for the script and for `python3` and otherwise falls through to `git merge-file`, which
does write markers — and the script wraps itself so that an internal crash writes markers
before exiting. Both paths are covered by tests.

## Strategy versioning

Run `atb strategies version` after modifying any file in `src/strategies/`. The helper inspects staged changes, prompts for a
succinct changelog, bumps the semantic version, and auto-stages the updated manifests under `src/strategies/store/`. Add
`--yes` when scripting the workflow; the bundled pre-commit hook simply delegates to this command when the helper is available.

## Helpful shortcuts

- `atb backtest ml_basic --days 30` – quick simulations while iterating on strategies.
- `atb live ml_basic --paper-trading` – start the live runner in paper trading mode.
- `PORT=8000 atb live-health -- ml_basic --paper-trading` – start live trading with the embedded health endpoint (override with `PORT` or `HEALTH_CHECK_PORT`).
- `atb experiment run --config experiments/signal_thresholds.yaml` – run a declarative experiment suite.

Use these commands to mirror CI behaviour locally before opening pull requests.

## Codex auto-review workflow

The repository includes a Codex-driven loop that keeps running fast validations, requests a structured review, and lets Codex apply fixes until the review comes back clean.

```bash
python -m cli codex auto-review \
  --plan-path docs/execplans/codex_auto_review.md \
  --max-iterations 3
```

Key behaviour:

- Validation commands (`--check`) run before every review iteration. If you omit the flag, no tests or linters run and the workflow relies solely on Codex review/fix cycles. Add only the fast checks you actually need.
- The workflow automatically diffs your current branch against `develop` (override with `--compare-branch <name>` or `--compare-branch ""` to disable) so Codex focuses on the recent changes.
- Codex is explicitly told to focus on issues introduced by that diff; it can inspect other files for context but only reports regressions tied to the diff, so keep the diff scoped to the work you want touched.
- The review step enforces `cli/core/schemas/codex_review.schema.json`, so Codex replies with machine-readable findings. When the findings array is empty and validations pass, the command exits with status 0.
- Fix iterations run in `--full-auto` mode by default. Pass `--dangerous-fix` to let Codex bypass sandboxing/approvals entirely (recommended only in a disposable environment).
- Artifacts live under `.codex/workflows/<timestamp>/` and include validation logs, structured review JSON, and Codex fix transcripts for auditability.
- Run the command through the project’s Python 3.11 environment (`python3.11 -m cli ...` or `.venv/bin/python -m cli ...`) because the codebase relies on 3.10+ typing features. The loop also injects that interpreter as the `PYTHON` environment variable so Makefile targets like `make test` work even if `python` is not on your PATH (override with `--python-bin`).

You can point `--plan-path` to the ExecPlan that guided the change so Codex understands the intended milestones, but the flag is optional—leaving it out simply tells Codex to review the diff/validations. Use `--profile <name>` to select an alternate Codex configuration, or `--max-iterations 0` for a dry run that just prints the help/exit path without calling Codex.
