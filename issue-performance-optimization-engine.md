### Epic: Closed-loop Performance Optimization System (Phases 2–3) — follow-up to PR #110

**Summary**
- **Goal**: Build a closed-loop performance optimization system that continuously learns from backtests and live trading, proposes safe improvements, validates them statistically, and applies small, reversible changes to steadily increase annual returns.
- **Context**: Phase 1 scaffolding was implemented in [PR #110](https://github.com/bumpy-croc/ai-trading-bot/pull/110). This epic tracks Phase 2–3 build-out and ongoing iterations, aligned with [Issue #42](https://github.com/bumpy-croc/ai-trading-bot/issues/42).

### Vision
- **Closed-loop optimization**: Analyze performance, propose bounded parameter/model/strategy tweaks, validate with robust stats, and roll out with canary + rollback.
- **Frequent, safe experimentation**: Small, high-confidence nudges with change budgets and cooldowns.
- **Compounding returns**: Systematic calibration, regime-aware adaptation, and continuous experiment logging.

### High-level Architecture
- **Data capture layer**
  - Log per-trade, per-decision, and per-prediction data from live and backtests.
  - Persist standardized metrics: returns, Sharpe, max drawdown, win rate, slippage, latency, and “position size” decisions.
  - Track prediction calibration and regime tags (volatility, trend, liquidity).
- **Performance warehouse**
  - Lightweight tables for optimization history and prediction performance (aggregated daily/hourly).
  - Maintain experiment runs with configs, seeds, results, and artifacts.
- **Backtest/experiment lab**
  - Deterministic, reproducible backtests over rolling windows and multiple market regimes.
  - Parameter search runners (grid/random/Bayesian) with safety bounds and time budgets.
  - Ensemble evaluators and strategy-switch simulations.
- **Optimization engine**
  - Analyzer: detects underperformance, drift, and regime changes.
  - Optimizer: proposes bounded adjustments to risk parameters, confidence thresholds, ensemble weights, and strategy selection.
  - Validator: performs A/B backtests and statistical checks (bootstrap, SPA/Reality Check) before applying.
- **Orchestrator**
  - Scheduled jobs (daily and 6-hour cadence) trigger analyze → propose → validate → (conditionally) apply.
  - Cooldowns and change budgets (≤20% parameter delta/cycle) with rollback support.
- **Change management**
  - Config changes emitted as PRs or feature-flag updates for transparency and review.
  - Shadow mode and canarying for live rollout; automatic rollback on degradation.
- **Monitoring + reporting**
  - Dashboard panels for optimization cycles, suggested/applied changes, and deltas vs. baseline.
  - Alerts on significant drift or performance regressions.

### Optimization Cycles
- **Every 6 hours**: Drift checks; small risk parameter nudges (position size caps, stop-loss/take-profit tweaks).
- **Daily**: Rolling-window backtests across recent regimes; refresh ensemble weights and confidence thresholds.
- **Weekly**: Deeper exploratory runs (wider parameter ranges, alternative strategies, model variants).

### Optimization Techniques
- **Risk parameters**: Adjust position size ceilings/floors, stop-loss/take-profit ratios, confidence scaling; max 10–20% delta per cycle.
- **Prediction calibration**: Refit confidence thresholds to match realized accuracy; recalibrate directional cutoffs.
- **Ensemble weighting**: Weight models/strategies by recent Information Coefficient, regime-conditional performance, and stability.
- **Regime-aware switching**: Detect bull/bear/sideways/high-vol regimes and suggest best-fit strategy/loadout per regime.
- **Statistical validation**: Walk-forward tests with bootstrapping; minimum effect sizes and confidence thresholds; SPA/Reality Check where practical.

### Guardrails and Safety
- **Change limits**: Cap per-cycle change (≤20%), minimum cool-down (≥6h), and require performance delta thresholds.
- **Shadow + canary**: Shadow backtests/live inference logging; roll out to a small capital fraction first.
- **Automatic rollback**: If live KPIs degrade beyond thresholds within a window, revert parameters.
- **Data quality checks**: Input completeness, leak detection, and distribution shift checks before optimizing.

### CI/CD and Scheduling
- **Nightly “Backtest Matrix”**
  - Rolling windows, regimes, and parameter samples. Publish artifacts and a summary report.
- **“Optimizer” workflow**
  - Consumes latest results, proposes bounded config changes, opens a PR with rationale and metrics deltas.
- **“Validation” workflow**
  - On PR label or schedule, re-run confirmatory backtests; auto-merge gated by thresholds.
- **“Deployment” workflow**
  - Apply config updates as constants/feature flags; optional canary gate.

### Data and Schema (high level)
- **Prediction performance**: `model_name`, `timestamp`, predicted vs. actual, direction correctness, confidence, calibration error, regime tags.
- **Optimization history**: change type, before/after KPIs, parameter deltas, confidence, reason, success flag, session linkage.
- **Experiment runs**: config hash, seeds, window(s), metrics, artifacts URI, environment info.

### Resources Needed
- **Compute**: CPU-optimized runners for frequent backtests; optional GPU for retraining; time budgets per workflow.
- **Storage**: Database for metrics/logs; object storage for artifacts (plots, large result sets).
- **Dependencies**: `scikit-learn`, `scipy`, `optuna` or `scikit-optimize`; `pandas`, `numpy`; optional `shap`.
- **Ops**: Scheduled CI/CD, secrets for storage, monitoring hooks.

### Product Constraints
- Use current configuration philosophy: constants in `src/config/constants.py`, feature flags in `feature_flags.json`.
- Use absolute imports and “position size” terminology.
- Keep the prediction engine off by default unless explicitly enabled.

### Phase Plan and Status
- **Phase 1 (done via PR #110)**
  - Instrumentation + warehouse tables; nightly rolling backtests; baseline analyzer producing suggestions; manual PRs for changes.
- **Phase 2 (this epic)**
  - Bounded optimizer with validation gates; canary rollout; dashboard panels; auto-PRs with rationale and metrics deltas.
- **Phase 3 (this epic, ongoing)**
  - Regime-aware switching, ensemble tuning, automated rollback, richer statistical testing.

### Deliverables and Acceptance Criteria
- **Phase 2**
  - Bounded optimizer proposes changes to: position size caps/floors, stop-loss/take-profit, confidence thresholds, ensemble weights.
  - Validator runs A/B backtests and bootstraps; enforces minimum effect size and confidence (e.g., ≥95% for risk tweaks).
  - Auto-PR content includes: before/after KPIs, parameter deltas (≤20%), rationale, artifacts links, and a rollback plan.
  - Canary deployment flag with configurable capital fraction; automatic rollback on KPI breach.
  - Dashboard panels: last N optimization cycles with suggested vs. applied changes and deltas vs. baseline.
- **Phase 3**
  - Regime detection deployed (bull/bear/sideways/high-vol) with switching logic and per-regime loadouts.
  - Ensemble weighting based on recent IC and regime-conditional performance; stability constraints.
  - SPA/Reality Check added to validator for model/strategy changes where feasible.
  - Automatic rollback wired into live monitoring with bounded reversion.

### Task Checklist
- [ ] Data capture: extend logging for per-decision, per-prediction with regimes and calibration fields.
- [ ] Warehouse: tables for `prediction_performance`, `optimization_history`, `experiment_runs`; ETL jobs.
- [ ] Backtest lab: deterministic runners; parameter search with safety bounds and budgets; artifacts publishing.
- [ ] Analyzer: drift/underperformance detectors; regime change detection; suggestion schema.
- [ ] Optimizer: bounded adjustments (≤20%/cycle); respect cooldowns and change budgets.
- [ ] Validator: A/B backtests + bootstrap; minimum effect sizes; SPA/Reality Check for higher-risk changes.
- [ ] Orchestrator: 6-hour and daily schedules; apply gating; shadow mode; canarying.
- [ ] Change management: config PR writer for `src/config/constants.py` and `feature_flags.json` with absolute imports.
- [ ] Monitoring: dashboards for cycles, changes, and KPI deltas; alerts on drift/regressions.
- [ ] CI/CD: workflows for Backtest Matrix, Optimizer (auto-PR), Validation (confirmatory), Deployment (canary).
- [ ] Documentation: playbooks for rollback, thresholds, and change budgets.

### Risks and Mitigations
- **Overfitting to recent regimes** → Use walk-forward validation, bootstrap, SPA tests; enforce minimum effect sizes and stability.
- **Cascading changes** → Enforce cooldowns and max deltas; limit simultaneous knobs per cycle.
- **Data quality** → Add completeness checks and distribution shift monitors before optimization runs.
- **Operational complexity** → Keep MVP lean; progressive hardening and observability-first approach.

### Definition of Done
- Auto-PR pipeline produces bounded, statistically validated suggestions with artifacts and clear rollback instructions.
- Canary deployments and automatic rollback wired and tested in staging/backtest mode.
- Dashboards show optimization cycles, applied changes, KPI deltas, and alerting works.
- Documentation/playbooks updated and discoverable.

### Labels and Ownership
- Suggested labels: `enhancement`, `automation`, `epic`, `area:optimizer` (or closest equivalents in this repo).
- Assignee: a background agent or team member responsible for Phases 2–3 delivery.

### References
- [PR #110](https://github.com/bumpy-croc/ai-trading-bot/pull/110) — MVP scaffolding
- [Issue #42](https://github.com/bumpy-croc/ai-trading-bot/issues/42) — Alignment and scope framing

---

If using GitHub CLI once authenticated:

```bash
gh issue create \
  --repo bumpy-croc/ai-trading-bot \
  --title "Epic: Closed-loop Performance Optimization System (Phases 2–3) — follow-up to PR #110" \
  --label "enhancement" --label "automation" --label "epic" --label "area:optimizer" \
  --body-file issue-performance-optimization-engine.md
```