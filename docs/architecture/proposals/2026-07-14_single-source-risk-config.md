# Single-Source Risk Configuration

- **Status**: Accepted with Board amendments ([D-2026-07-14-04]): file moves to `src/config/risk-limits.json`; §3.7 prune-only (re-anchor rejected); §3.8 confirmed at 0.15
- **Date**: 2026-07-14
- **Authority**: Board directive [D-2026-07-14-02] (`.claude/state/log.md`) — backtest-live parity foremost; risk/trading variables defined in ONE place, read by ALL consumers; eliminate the risk-limits.json vs constants.py divergence *class*, not instances.
- **Refs**: GH #986 (items 2/3/4/5), GH #1021 (ExperimentRunner clamp drift), GH #835 (startCommand loosening incident), GH #1020 (allow_shorts, interaction noted)

## 1. Problem

The Board-ratified limits live in `.claude/state/risk-limits.json`, whose own header declares a manual mirror rule (`.claude/state/risk-limits.json:4`):

> `"$source_of_truth_note": "Must match src/config/constants.py. Any divergence is a P0."`

Manual mirrors fail. Verified divergences and drift instances as of `develop-current` (7e213d3a):

| # | Divergence | Evidence |
|---|---|---|
| 1 | `max_position_size_pct = 0.20` (json:17) vs `DEFAULT_MAX_POSITION_SIZE = 0.1` | `src/config/constants.py:135` (#986 item 2) |
| 2 | **Units** divergence: `operational.balance_discrepancy_warning_pct: 0.01` (fraction, json:37) vs `DEFAULT_BALANCE_DISCREPANCY_THRESHOLD_PCT = 1.0` (percent) | `src/config/constants.py:262`, consumed as percent at `src/engines/live/account_sync.py:224` — same intent, different units; previously uncatalogued, found during this design |
| 3 | Backtest CLI risk defaults are literals drifted from both json and constants: `--risk-per-trade default=0.01`, `--max-risk-per-trade default=0.02`, `--max-drawdown default=0.5` | `cli/commands/backtest.py:491,493,520` vs live's `DEFAULT_BASE_RISK_PER_TRADE=0.02`/`DEFAULT_MAX_RISK_PER_TRADE=0.03`/`DEFAULT_MAX_DRAWDOWN=0.20` (`src/engines/live/runner.py:136,142,148`). Default backtests run at **half** live risk and 2.5× live drawdown tolerance |
| 4 | Hidden sizer clamp `max_fraction=0.2` keyword default silently clamps HyperGrowth's 0.25 | `src/strategies/components/position_sizer.py:109-110` (base-class default), `:235` (FixedFractionSizer calls it without `max_fraction`), plus sibling literals `:474` (0.15), `:1062` (0.1), `:1260` (helper default 0.2) (#986 item 3) |
| 5 | Bare `RiskParameters()` in the experiment harness silently clamped a strategy's 0.25 override to 0.10 | `src/experiments/runner.py:481-483` + clamp at `src/risk/risk_manager.py:479-482` (GH #1021) |
| 6 | Prod ran `--max-position 0.5` for weeks via `railway.json` startCommand — an env-level **loosening** no code review saw | GH #835; today's pinned value at `railway.json:7` |
| 7 | Two constants for one concept: `DEFAULT_MAX_CORRELATED_RISK = 0.10` vs `DEFAULT_MAX_CORRELATED_EXPOSURE = 0.15`; json has only `max_correlated_exposure_pct: 0.15` (json:11) | `src/config/constants.py:139,349` (#986 item 5), plus a third hardcoded `0.10` fallback at `src/position_management/dynamic_risk.py:144` |
| 8 | HyperGrowth dynamic-risk tiers ≥ 0.20 are dead code | thresholds `[0.15, 0.30, 0.45]` at `src/strategies/hyper_growth.py:370-376`; the hard cap latches close-only at 0.20 first via `MaxDrawdownGuard(self.risk_manager.params.max_drawdown)` at `src/engines/live/trading_engine.py:1052-1058` (#986 item 4) |
| 9 | Backtest engine has its own `0.5` early-stop drawdown literal when no risk parameters are passed | `src/engines/backtest/engine.py:358-359` |

Every one of these is the same defect: a risk number defined somewhere other than the ratified source, kept aligned by convention.

## 2. Current-state inventory (consumer map)

**The ratified source** — `.claude/state/risk-limits.json` (git-tracked; `$owner: human_board`; sections `portfolio`, `position`, `stops`, `operational`, `escalation`, `kill_switch`). It already ships in the Docker image: `Dockerfile:26` is `COPY . .` and `.claude` is not in `.dockerignore`.

**The mirror** — `src/config/constants.py` risk entries: `:131-140` (stop/TP/position/risk/drawdown block), `:142-144` (escalation constants with their own "Must match .claude/state/risk-limits.json" comment), `:157` (max holding hours), `:262` (balance discrepancy, wrong units), `:270` (fallback trailing), `:275-276` (dynamic thresholds/factors), `:297` (large single position 0.20), `:349` (correlated exposure 0.15), `:412` (Kelly max fraction 0.20), `:428` (max leverage 3.0).

**Consumers of the mirror**:
- `RiskParameters` field defaults: `src/risk/risk_manager.py:138-173`; validation `__post_init__` `:175-198`; strategy-override clamp `:456-493` (the #1021 clamp is `:479-482`)
- Live engine: constructor default `src/engines/live/trading_engine.py:215`; drawdown guard wiring `:1052-1058`; correlation wiring `:604`
- Live runner CLI: `--max-position default=DEFAULT_MAX_POSITION_SIZE` `src/engines/live/runner.py:95-99`; risk args `:133-150`; `RiskParameters(...)` built from args `:268-272`, `max_position_size=args.max_position` into engine `:285`
- Backtest engine: `src/engines/backtest/engine.py:293-300` (backward-compat construction), `:358-359` (0.5 literal), `:558` (`DEFAULT_MAX_POSITION_SIZE` fallback)
- Backtest CLI: `cli/commands/backtest.py:314-333` (construction; `:325-332` honors strategy `max_fraction` — a seeding behavior ExperimentRunner does *not* share, which is exactly the #1021 drift), defaults `:491,493,520`
- ExperimentRunner: `src/experiments/runner.py:481-483` (bare `RiskParameters()`)
- Sizers: `src/strategies/components/position_sizer.py:109-110,235,474,1062,1260,1394`
- Dynamic risk: `src/position_management/dynamic_risk.py:132-144` (incl. hardcoded 0.10), `:563-565` (heuristic correlated cap)
- Correlation engine: `src/position_management/correlation_engine.py:28`
- Strategies constructing `RiskParameters`/overrides: `src/strategies/ml_basic.py:105-116`, `ml_sentiment.py:111-120`, `hyper_growth.py:295-395` (notably `:335` `_max_position_pct = 0.50`, `:361` `max_fraction = min(base_fraction * max_leverage, 0.50)`)
- Deploy: `railway.json:7` (`atb live-health hyper_growth --max-position 0.20`)
- Governance tooling: `.claude/skills/risk-ratification/SKILL.md:26,39,48,74` (three-way mirror-verification steps)

## 3. Design

### 3.1 The source: `.claude/state/risk-limits.json`, runtime-loaded

**Decision (as amended by [D-2026-07-14-04])**: the file MOVES to `src/config/risk-limits.json` (Board ruling, overriding the keep-in-place recommendation below) and is THE source, loaded and validated at boot by a new module `src/config/risk_limits.py`. It remains human-owned at the new location ($owner: human_board; agents never edit its values). The risk/sizing entries in `constants.py` are **deleted** — not converted to re-exports.

Why this file and not a new one: it is the file the Board already ratifies, with git history, `$last_reviewed` stamps, and every governance artifact (charter, board.md, risk-ratification skill) pointing at it. The mirror problem was two homes for one number; moving the file creates a third home during transition. It already deploys (Docker `COPY . .`; `.claude` not dockerignored). Should someone dockerignore `.claude/` in the future, fail-closed boot (3.3) turns that into a loud staging-boot failure, not silent defaults.

Why deletion and not re-export: a re-export keeps two names alive for one value, forces file I/O at `constants` import time for unrelated consumers, and makes the CI literal-guard unable to distinguish a live accessor from a stale import. Non-risk constants (prediction, feature-engineering, health, reconciliation cadence, etc.) stay in `constants.py` untouched.

### 3.2 Loader: `src/config/risk_limits.py`

- Frozen dataclasses mirroring the json sections: `PortfolioLimits`, `PositionLimits`, `StopLimits`, `OperationalLimits`, `EscalationPolicy`, `KillSwitchPolicy`, composed into `RiskLimits`. Each json key maps 1:1 to exactly one typed attribute (requirement for #986 item 5).
- `load_risk_limits(path: Path | None = None) -> RiskLimits` — resolves the default path via `get_project_root()` (same pattern as `src/config/feature_flags.py:25-36`). Explicit `path` is for tests only; there is deliberately **no** env-var path override in engine code paths (an env-redirectable limits file would be a loosening vector).
- `get_risk_limits() -> RiskLimits` — process-cached (`functools.lru_cache`) accessor all consumers use.
- **Strict validation** (house style: hand-rolled like `RiskParameters.__post_init__`, `src/risk/risk_manager.py:175-198`; no new pydantic/jsonschema dependency — neither is in `requirements.txt`/`pyproject.toml`): pinned `$schema_version`; unknown keys rejected; missing keys rejected; type and range checks (all `*_pct` keys are **decimal fractions** in (0,1] — this units convention resolves divergence #2 at the consumer, `account_sync.py`, during migration); cross-field invariants: `base_risk_per_trade_pct <= max_risk_per_trade_pct`, `min_stop_loss_pct <= default_stop_loss_pct <= max_stop_loss_pct`, dynamic threshold/factor arrays equal length, thresholds strictly ascending, and `max(dynamic_drawdown_thresholds_pct) < max_drawdown_pct` (the dead-tier invariant, see 3.7).
- Failure of any check raises `RiskLimitsError` with the offending key and value.

### 3.3 Fail-closed

- **Live**: `src/engines/live/runner.py:main()` calls `get_risk_limits()` (and the override gate, 3.4) before any provider/exchange construction. Missing/invalid file ⇒ non-zero exit with the schema error; the engine never reaches the exchange.
- **Backtest**: `cli/commands/backtest.py:_handle()` likewise, before data-provider setup.
- **ExperimentRunner**: `src/experiments/runner.py:run()` likewise.
- **Library-level backstop**: because `RiskParameters` hydrates its defaults from the loader (3.5), even a direct programmatic construction with no CLI in the loop fails closed. There is no fallback constant left in the tree to fall back to — the deletion makes the silent-default failure mode unwritable.

### 3.4 Override policy: tighten-only, fail-loud

New function in the loader module:

```python
def apply_overrides(limits: RiskLimits, overrides: dict[str, float], source: str) -> RiskLimits
```

`overrides` keys are dotted json paths (`"position.max_position_size_pct"`); `source` is a human string for the error (`"railway.json startCommand --max-position"`, `"env MAX_POSITION_SIZE"`). Comparison semantics per key class:

| Class | Keys | Valid override |
|---|---|---|
| Ceilings (lower = tighter) | `position.max_position_size_pct`, `position.base_risk_per_trade_pct`, `position.max_risk_per_trade_pct`, `position.max_leverage`, `position.kelly_max_fraction`, `position.large_single_position_threshold_pct`, `portfolio.max_drawdown_pct`, `portfolio.max_daily_risk_pct`, `portfolio.max_correlated_exposure_pct`, `stops.max_stop_loss_pct`, `operational.max_holding_hours`, `operational.max_consecutive_errors`, `operational.max_filled_price_deviation_pct`, `escalation.warning_at_pct_of_limit`, `escalation.critical_at_pct_of_limit` | `override <= ratified` |
| Floors (higher = tighter) | `stops.min_stop_loss_pct` | `override >= ratified` |
| Bounded defaults (not limits) | `stops.default_stop_loss_pct`, `stops.default_take_profit_pct`, `stops.fallback_trailing_pct` | must remain within ratified `[min_stop_loss_pct, max_stop_loss_pct]` |
| Non-overridable | `portfolio.dynamic_drawdown_thresholds_pct`, `portfolio.dynamic_risk_reduction_factors`, `escalation.breach_action`, all of `kill_switch` | any attempt is an error |

A violation raises `RiskLimitLoosenedError` naming key, ratified value, attempted value, and source, at boot. This retires the #835 class: a `startCommand --max-position 0.5` crashes the deploy before the first exchange call, permanently.

**CLI wiring**: risk flags on both CLIs change their argparse defaults to `None` = "no override" (removing today's drifted literals, divergence #3); only explicitly passed flags enter `apply_overrides`. Live/paper: no bypass exists. Backtest/harness: research legitimately explores outside ratified limits (e.g. trend-following at 0.95 allocation), so one explicit escape hatch, `--unratified-risk`, (a) skips the tighten-only check, (b) stamps `unratified_risk_overrides: {key: {ratified, used}}` into the results payload and logs a WARNING — a study can loosen, but never silently while appearing live-representative. ExperimentRunner accepts the equivalent config field, default off.

### 3.5 Identical defaults for live, backtest, and harness (kills #1021's drift dimension)

- `RiskParameters` risk-limit fields (`base_risk_per_trade`, `max_risk_per_trade`, `max_position_size`, `max_daily_risk`, `max_drawdown`, `max_correlated_exposure`, and the stop/trailing defaults currently seeded from constants) change to sentinel `None` defaults hydrated in `__post_init__` from `get_risk_limits()`. Every bare `RiskParameters()` — `src/experiments/runner.py:482`, `src/engines/backtest/engine.py:296`, `src/engines/live/strategy_runtime.py:143`, `src/engines/live/strategy_hot_swap.py:446`, `src/strategies/components/risk_adapter.py:156` — then yields ratified values by construction. Explicit constructor arguments still win (they are strategy/caller intent, clamped by the engine).
- **Strategy-override clamp semantics** (separate documented decision): strategies may *request* any `max_fraction` via `get_risk_overrides()`. Both engines clamp to `min(requested, effective_cap)` where `effective_cap` is the ratified cap after tighten-only overrides. Two additions: (1) the seeding logic currently unique to the backtest CLI (`cli/commands/backtest.py:325-332`) moves into one shared helper used by the backtest CLI **and** ExperimentRunner, so the harness matches CLI behavior and neither can drift again; (2) whenever the clamp binds, both engines emit a structured WARNING (`strategy requested X, effective cap Y, clamped`) and the effective per-arm sizing is auto-reported in backtest/experiment results — the visibility fix #1021 asked for, so a silent clamp can never invalidate a study again.
- Net behavior deltas at this step, stated for review: bare-default constructions move `max_position_size` 0.10→0.20 (the ratified number; live is unaffected — `railway.json:7` already pins 0.20 and now becomes a no-op tighten) and backtest CLI defaults move to ratified 0.02/0.03/0.20. Strategies that intentionally run tighter (e.g. `ml_basic.py:110-116` at 0.10) keep their explicit values — tightening is always allowed.

### 3.6 #986 item 3 — FixedFractionSizer's hidden clamp

`PositionSizer.apply_bounds_checking` (`src/strategies/components/position_sizer.py:109-110`) and `clamp_position_size` (`:1260`) lose their `max_fraction=0.2` keyword defaults; `max_fraction` becomes **required**. Every sizer passes an explicit cap held from construction: `FixedFractionSizer` gains a `max_fraction` constructor parameter; the HyperGrowth factory passes `min(base_fraction * max_leverage, get_risk_limits().position.max_position_size_pct)`; the stray literals at `:474` (0.15) and `:1062` (0.1) become explicit constructor-carried values reading the loader or documented tighter strategy intent. Resolution of the "which number is Board-approved" question: the ratified cap is 0.20 (json:17); HyperGrowth's 0.25 base (`hyper_growth.py:177-178`) remains an intentional over-request that is now clamped *visibly* (WARNING + reported effective sizing) instead of silently — raising the cap to 0.25 is a separate ratification decision the Board can take with real evidence in front of it.

### 3.7 #986 item 4 — HyperGrowth dead tiers

The tiers at 0.30/0.45 (`hyper_growth.py:373`) can never fire: `MaxDrawdownGuard` latches close-only at `max_drawdown = 0.20` first (`trading_engine.py:1052-1058`). **Board ruling [D-2026-07-14-04]: PRUNE-ONLY** (the re-anchor recommendation below was considered and rejected; kept for the record, available as a future separate proposal). Concretely: `drawdown_thresholds: [0.10, 0.15]`, `risk_reduction_factors: [0.8, 0.5]`. Justification: prune-only (`[0.15] → 0.8`) preserves the "one 0.8 nudge then a cliff" shape the audit criticized; re-anchoring restores a genuinely graduated ramp in which every configured tier is reachable, with de-risking arriving *earlier* (a tightening — no ratification blocker, but a behavior change requiring a decision-record and a preregistered backtest comparison per the experiment-preregister protocol before merge; prune-only is the pre-committed fallback if the tightened ramp measurably degrades the strategy). The *class* fix is the loader/engine invariant from 3.2 applied to strategy overrides too: any strategy dynamic-risk threshold `>= portfolio.max_drawdown_pct` fails validation at boot/config-merge (`src/engines/shared/risk_configuration.py:45-77` is the merge seam). Dead tiers become unrepresentable, not just removed.

### 3.8 #986 item 5 — correlated risk vs exposure

These are two enforcement mechanisms for one Board intent ("bounded exposure to a correlated basket"): the correlation-matrix cap (`correlation_engine.py:28,244`, wired from `RiskParameters.max_correlated_exposure` at `trading_engine.py:604`) and the position-count heuristic cap (`dynamic_risk.py:562-565`, reading `max_correlated_risk`). The json ratifies exactly one number: `max_correlated_exposure_pct: 0.15` (json:11).

**Decision**: one key → one accessor — `limits.portfolio.max_correlated_exposure_pct` — consumed by both mechanisms. `DEFAULT_MAX_CORRELATED_RISK` (`constants.py:139`), `RiskParameters.max_correlated_risk` (`risk_manager.py:144-146`), and the hardcoded `0.10` fallback (`dynamic_risk.py:144`) are deleted. Behavioral delta: the heuristic cap moves 0.10→0.15. Recommend adopting 0.15 — the json is the Board's signed number; 0.10 was never ratified. The alternative (ratify the key at 0.10, tightening the matrix cap too) is presented in the same sitting; either way, after this change the schema makes a second correlated-limit number unrepresentable.

### 3.9 CI enforcement

1. **Schema test** (`tests/unit/config/test_risk_limits_schema.py`): loads the actual `.claude/state/risk-limits.json` through the actual loader. Any malformed Board edit fails every PR before it can reach a deploy.
2. **Literal guard** (`tests/unit/config/test_no_risk_literals.py`): AST walk over `src/engines`, `src/risk`, `src/position_management`, `src/config` (excluding `risk_limits.py`), and `cli`, failing on (a) any reference to the deleted constant names, (b) any numeric keyword default on parameters matching `max_fraction|max_position_size|max_drawdown|base_risk|max_risk|max_daily|max_correlated|max_leverage|kelly_max|stop_loss_pct`, with a small committed allowlist file where each entry carries a written justification. `src/strategies` is exempt from (b): strategies may declare tighter intent as literals because the engine-side clamp binds; the boundary is documented in the test.
3. **Deploy-config guard** (`tests/unit/config/test_deploy_config_within_limits.py`): parses `railway.json` `startCommand` and the Dockerfile CMD, extracts risk flags, and asserts each passes `apply_overrides` — the #835 tripwire at CI time, in front of the boot-time one.
4. **Parity tripwire**: `RiskParameters()` equals loader values field-by-field (replaces the old mirror-check intent of `tests/unit/config/test_constants.py`).

### 3.10 Governance and the layer-1 change

`risk-limits.json` stays `$owner: human_board`; agents never edit it. The consolidation changes its self-description — this is a layer-1 edit packaged for Alex's own hand, verbatim:

```diff
--- a/.claude/state/risk-limits.json
+++ b/.claude/state/risk-limits.json
@@ -1,7 +1,7 @@
 {
   "$schema_version": "1",
   "$owner": "human_board",
-  "$source_of_truth_note": "Must match src/config/constants.py. Any divergence is a P0.",
+  "$source_of_truth_note": "RUNTIME SOURCE OF TRUTH: loaded and schema-validated at boot by src/config/risk_limits.py; engines fail closed if this file is missing or invalid. There is no code mirror. Env/CLI overrides may tighten but never loosen these values. Human-owned: agents never edit this file.",
   "$last_reviewed": "2026-07-05",
   "$last_reviewer": "alexflorisca",
```

(`$last_reviewed`/`$last_reviewer` shown as context; they get re-stamped in the sitting per the risk-ratification skill.) One sitting covers: (a) this note change, (b) correlated-exposure 0.15 adoption (3.8), (c) HyperGrowth tier re-anchor acknowledgment (3.7). The sitting is scheduled when migration steps 6–7 are ready to merge — the mirror rule dies in the same breath as the mirror. The risk-ratification skill is then amended (`.claude/skills/risk-ratification/SKILL.md:26,39,48,74`): the three-way `JSON ↔ constants.py ↔ as-deployed` check becomes two-way `JSON ↔ as-deployed`, and "constants.py mirror updates in the same diff" steps are deleted. The mirror comment at `constants.py:142-144` goes with the constants.

## 4. Migration plan

Each step independently shippable and testable; steps 3–6 are money-path and go through the full review gauntlet per delegation-protocol.

1. **Loader** — add `src/config/risk_limits.py` + unit tests + CI schema test (3.9.1). No consumers. Zero behavior change.
2. **Visibility first** — shared strategy-override seeding helper (backtest CLI + ExperimentRunner), clamp-binding WARNING in both engines, effective-sizing auto-report in results. Fixes #1021's *silent* aspect before any default changes value.
3. **Hydration** — `RiskParameters` sentinel-hydration from the loader; boot-time `get_risk_limits()` calls in live runner, backtest CLI, ExperimentRunner; parity tripwire (3.9.4). Documented deltas: bare-default `max_position_size` 0.10→0.20; backtest CLI defaults align to ratified.
4. **Tighten-only gate** — `apply_overrides` + CLI `None`-defaults + `--unratified-risk` + deploy-config CI guard (3.9.3). Kills the #835 class.
5. **Explicit sizer caps** — 3.6, plus deletion of the stray literals (`position_sizer.py:110/474/1062/1260`, `dynamic_risk.py:144`, `backtest/engine.py:358-359`), plus the `account_sync.py` units fix (divergence #2) with a regression test.
6. **Behavioral decisions** — correlated-exposure unification (3.8) + HyperGrowth tier re-anchor (3.7), each with a decision-record and the preregistered backtest check.
7. **Deletion + enforcement** — remove risk constants from `constants.py`; land the AST literal guard (3.9.2); update docs (`docs/configuration.md`, engine docs) and the risk-ratification skill; hold the ratification sitting applying the layer-1 diff.

**#1020 interaction — confirmed.** `allow_shorts` has no code in the tree yet (zero grep hits in `src/` and `feature_flags.json`); it ships independently against today's structure. Migrating it later, should the Board ratify short-permission as a limit rather than strategy config, is one schema key + one accessor + one read-site swap on a working loader — trivial, as the directive assumed. No reason to sequence #1020 behind this design.

## 5. Rejected alternatives

- **constants.py as source, json generated**: inverts ownership — the Board would be ratifying generated output while agents edit the true source. Violates the governance invariant.
- **Re-export shim in constants.py**: two live names per value, import-time file I/O for unrelated consumers, and the literal guard cannot separate live accessors from stale imports.
- **Move the file to `config/risk_limits.json`**: churns every governance reference for zero mechanical gain; the mirror pathology was two homes — do not mint a third. Docker already ships `.claude/state`; fail-closed boot covers future packaging mistakes.
- **pydantic/jsonschema validation**: neither is currently a dependency; hand-rolled frozen-dataclass validation matches the existing `RiskParameters.__post_init__` style and adds no supply-chain surface to the money path.
- **DB-stored limits**: runtime-mutable, no diffable ratification artifact; file + git history *is* the audit trail.

## 6. Risks

- **Behavior deltas in steps 3/6** are enumerated, tightening-or-ratified-only, and each carries its own test + decision-record; nothing changes value silently.
- **Loader as single point of failure**: intentional — that is what fail-closed means. Staging boots first per deploy-staging; a broken file stops staging, not prod capital.
- **Guard false positives**: the allowlist file with per-entry justification keeps the AST guard from ossifying legitimate non-limit numerics; strategy-layer exemption keeps strategy authors unblocked.
