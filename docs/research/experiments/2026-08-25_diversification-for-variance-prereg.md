# Experiment: Diversification-for-Variance Scoping (Phase 1, Vol-Regime Program #4)

**Author**: quant-researcher
**Status**: PREREGISTERED — locked before any statistic is computed. Do not edit thresholds after seeing results; corrections are new dated sections.
**Program**: Vol/regime program (this doc's sibling `agents/research/vol-regime-program.md`).
**Blocked on**: `#1106` for any promotion decision — see §10. Also gated on §12's capital-viability finding: this experiment's own conclusion may be "not viable at current account size," which is itself the deliverable.

---

## 0. Why this experiment, and what it is not

`docs/research/2026-07-12_returns-levers-synthesis.md` ranked diversification #2, with an explicit caveat: "this is a genuine, disclosed gap, not a result to cite... the first deliverable is a scoping exam (does BTCUSDT show the same ~51-53% pattern), not a capital-allocation decision." The model-free scan (#1105) has since answered that scoping question for direction — **BTCUSDT shows the identical zero-graduation pattern as ETHUSDT** (momentum accuracy 0.48/0.4798/0.4644/0.5034 across 15m/1h/4h/1d, all failing cost per `2026-08-13_econ_translation.csv`) — so the case for diversification cannot be "BTCUSDT has directional edge ETHUSDT lacks." It does not.

What the scan *does* establish, and what this experiment is actually about: **ARCH volatility clustering and regime persistence are strong and largely symbol-independent** (20/20 and 17/20 respectively, across all five symbols tested including BTCUSDT). This is the correct framing the July synthesis itself named but never quantified: **"diversification-for-variance rather than diversification-for-edge."** If ETHUSDT and BTCUSDT's volatility/regime states are not perfectly correlated, running the vol-targeted and regime-conditional sizing mechanisms from the two sibling experiments (`2026-08-25_volatility-target-sizing-prereg.md`, `2026-08-25_regime-conditional-exposure-prereg.md`) across two symbols simultaneously could smooth aggregate drawdown timing even though **neither symbol individually has positive directional expectancy** — this is explicitly a variance/drawdown-shape claim, not a return claim, and must never be reported as if it were the latter.

## 1. Hypotheses

**H0 (null)**: ETHUSDT and BTCUSDT's realized-volatility and regime states are sufficiently correlated (crypto beta effect) that running both simultaneously does not meaningfully smooth aggregate portfolio drawdown/variance relative to running ETHUSDT alone at proportionally larger size.

**H1 (diversification-for-variance hypothesis)**: BTCUSDT and ETHUSDT's regime/volatility states are correlated but not perfectly so (different market-maker structure, per the model-free scan's own symbol-selection rationale), and a two-symbol portfolio — each leg running the existing HyperGrowth strategy at half the capital allocation of a single-symbol deployment — produces a materially smoother equity curve (lower portfolio MaxDD, higher Sortino) than the single-symbol equivalent, even though **the pooled expected return is not claimed to improve** (both legs individually have the same net-negative-to-flat expectancy this whole research program has established; this is a shape claim, not an edge claim).

- *Mechanism if true*: imperfect correlation between symbols' regime timing means a drawdown episode concentrated in one symbol's regime state is partially offset by the other symbol not being in the same state at the same time, smoothing the combined equity curve the way any two-asset portfolio with correlation < 1 does — textbook, but never measured for this specific pair/strategy/cost structure.
- *Falsified if*: the two-symbol portfolio's MaxDD and Sortino are statistically indistinguishable from (or worse than) the single-symbol equivalent, i.e., correlation is high enough that diversification buys nothing net of the added complexity and fee drag from running two legs instead of one.

## 2. Metric

**Primary**: portfolio-level MaxDD and Sortino, two-symbol (ETHUSDT + BTCUSDT, each at 50% of the capital a single-symbol deployment would use) vs. single-symbol (ETHUSDT at full allocation) equivalent, on the frozen OOS exam window, fees/slippage on for both legs independently (diversification does not reduce cost drag — each leg pays its own round-trip cost; this must show up explicitly, not be netted away).

**Secondary**: pairwise realized correlation of each symbol's regime-bucket time series and of each symbol's daily-realized-vol series over the exam window (the actual number this hypothesis depends on — reported directly, not inferred from the portfolio-metric result alone), each symbol's standalone return/MaxDD/Sharpe (confirms neither leg is silently carrying the other), total round-trip cost as % of gross return for the combined portfolio.

Trade-count floor: ≥15 trades per symbol leg.

## 3. Success threshold (pre-committed, numeric)

The two-symbol portfolio **beats** the single-symbol equivalent if:
- Portfolio MaxDD is lower by **≥2 percentage points**, AND
- Portfolio Sortino is not worse (allowing the same 0.05 tolerance band used in the sibling sizing/exposure documents, for comparability), AND
- The measured pairwise regime/vol-state correlation is **below 0.8** (a correlation at or above 0.8 would mean any MaxDD improvement is likely a small-sample artifact of this specific window rather than a structural property worth building on — this threshold is set before the correlation is computed, per the anti-p-hacking rule, and is a legitimate, disclosed a-priori judgment call, not derived from a formula).

Statistical bar: paired bootstrap on the daily portfolio-vs-single-symbol equity-curve difference, Bonferroni-corrected across the 2 primary metrics -> alpha = 0.025 (confirmatory 2-arm test).

## 4. Arms (pre-committed)

- **Arm A (control)**: HyperGrowth/ETHUSDT/1h, single symbol, full capital allocation — the live-representative baseline.
- **Arm B (two-symbol)**: HyperGrowth/ETHUSDT/1h at 50% capital allocation + HyperGrowth/BTCUSDT/1h at 50% capital allocation, run as two independent backtest legs over the identical calendar window, combined post-hoc into a single equity curve (sum of the two legs' dollar P&L paths, each leg's own fees/slippage applied independently — this is the correct, conservative way to simulate a real two-symbol portfolio without needing a new multi-symbol portfolio-engine feature).
- **BTCUSDT model note**: `src/ml/models/BTCUSDT/basic/` already exists (confirmed in the July synthesis) — this experiment uses whatever model the `latest` symlink for BTCUSDT currently points to, recorded exactly (path + `metadata.json` training-window/version) in the results appendix, since a BTCUSDT-specific model quality claim is not this experiment's subject but must be disclosed as a dependency.

## 5. Data window / protocol

- Frozen exam window: identical to the sibling documents (2025-01-01 -> most recent complete UTC day), for direct cross-experiment comparability.
- Command (per symbol, per arm): `PYTHONPATH="$(pwd)" atb backtest hyper_growth --symbol <SYMBOL> --timeframe 1h --start 2025-01-01 --end <most recent complete UTC day> --risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20 --initial-balance <half-allocation for Arm B legs, full for Arm A> --drawdown-cap-mode measure --log-to-db`, long-only enforced (GH #1020), run strictly sequentially — **including across symbols**: BTCUSDT's leg is not run in parallel with ETHUSDT's leg, same thermal-discipline rule as every other experiment in this program. Explicit calendar dates and `PYTHONPATH`-forced, identity-verified worktree execution throughout, per GH #1070.
- Cache check: `atb data cache-manager info` for BTCUSDT/1h before running; `atb data prefill-cache --symbols BTCUSDT --timeframes 1h --years 2` if stale, noted explicitly if a prefill was needed (per the quant-researcher workflow's data-staleness rule).
- Determinism guard: Arm A/ETHUSDT leg run twice, must match, before combining.

## 6. Decision each outcome triggers

- **Supported**: recommend to pm as a "promising, not ready" diversification-for-variance candidate — explicitly **not** a return-improvement recommendation (this experiment cannot produce one; neither symbol has demonstrated positive expectancy). Frame precisely: "running two symbols at half size each may reduce drawdown severity/timing risk relative to concentrating in one, at the cost of added operational complexity, doubled fee drag in absolute terms, and the capital-viability constraint in §12." Any live-affecting recommendation additionally requires the capital-viability finding in §12 to show the split allocation is actually executable at the account's current balance.
- **Rejected**: full write-up. Correlation is too high for diversification to buy anything at this pair/window — closes the diversification-for-variance lever specifically for the ETHUSDT/BTCUSDT pair (does not preclude testing a lower-correlation pair, e.g. DOGEUSDT given its "low fundamentals-linkage" characterization in the model-free scan, as a separate future preregistration, not bundled here).
- **Correlation ≥0.8 specifically** (§3's third bar): reported as "structurally unlikely to help regardless of the MaxDD number" — even if MaxDD happens to improve on this one window, it is not treated as supported, since the mechanism this hypothesis depends on is absent by the pre-committed threshold.
- **Inconclusive**: BTCUSDT model/data quality issues (stale `latest` symlink, mismatched training window vs. the exam period) flagged and the experiment re-scoped rather than forcing a verdict on a compromised BTCUSDT leg.

## 7. Drawdown-truncation discipline (#1102)

`--drawdown-cap-mode measure` throughout; `early_stopped` checked per leg per arm. Portfolio-level MaxDD is computed from the combined (summed) equity curve, so a single leg's early-stop under `enforce` mode would not correctly represent portfolio behavior — `measure` mode is used for **both** legs of Arm B specifically so neither leg's drawdown accounting is truncated before being combined; this is disclosed as a deliberate protocol choice, not an oversight.

## 8. Sensitivity analysis

1. Capital split: 50/50 (primary) vs. 70/30 (ETHUSDT-weighted, since ETHUSDT is the live-traded symbol and a real deployment would plausibly not go fully 50/50 on an unproven second symbol) — tested on the training window only (2023-01-01 -> 2024-12-31), never touching the frozen exam.
2. Correlation window sensitivity: is the measured pairwise correlation stable across a rolling 90-day window within the exam period, or does it swing widely (a swinging correlation would itself be a finding worth naming, consistent with regime-persistence evidence suggesting correlation could itself be regime-dependent — untested territory, flagged for a possible future preregistration, not resolved here).

## 9. Risks of false positive

- **Two-leg combination is a backtest approximation, not a real multi-symbol portfolio engine feature.** Summing two independently-simulated equity curves assumes no cross-symbol interaction effects (e.g., a shared risk-limits.json portfolio-level drawdown cap or correlated-exposure cap that a real multi-symbol deployment would enforce jointly — `risk-limits.json`'s own `max_correlated_exposure_pct: 0.15` is a portfolio-level constraint this backtest approximation does not simulate). This is disclosed, not treated as equivalent to a genuine portfolio-level backtest.
- **Diversification is double-edged, exactly as the July synthesis warned**: "more instances of an edge that is currently net-negative after costs compounds losses exactly as readily as gains." A "supported" MaxDD-smoothing result on this specific window must not be read as evidence the strategy becomes profitable with more symbols — it explicitly does not test that claim and the write-up states this in the same paragraph as any positive finding, not in a buried caveat.
- **Single continuous exam window, same as every sibling document** — a correlation measured on one window may not hold in the next regime.
- **BTCUSDT model provenance**: whatever `latest` points to was trained under its own protocol, not audited here beyond recording its `metadata.json` — if BTCUSDT's model quality differs materially from ETHUSDT's (untested claim), that could drive the portfolio result independent of the diversification mechanism itself; flagged, not resolved, in this document.

## 10. Parity status

Both legs use the same closed-bar-entry HyperGrowth backtest as every sibling document; parity caveats are identical. Additionally: a real two-symbol live deployment would face the 43.2% entry-divergence issue **independently on each symbol**, which could either amplify or dampen the diversification benefit in ways this backtest-only approximation cannot characterize — flagged as an additional, diversification-specific parity risk beyond what the sibling documents carry.

## 11. What this experiment explicitly does not do

- Does not claim or test a return improvement from adding a second symbol.
- Does not build a real multi-symbol portfolio-engine feature (§9's disclosed approximation).
- Does not combine with the vol-targeted sizing or regime-conditional exposure mechanisms from the sibling documents — that combination is a natural, cheap follow-up once each piece is independently characterized, deliberately deferred per the same "no kitchen-sink arm" discipline as `2026-08-25_regime-conditional-exposure-prereg.md` §12.
- Does not propose a live-affecting change by itself.

## 12. Capital-viability check (required before any promotion recommendation, computed alongside the primary result)

`docs/research/experiments/2026-08-13_capital-sizing-knee.md` established that HyperGrowth's minimum-notional headroom is comfortable (1.8x+) down to $80 balance for a **single** symbol at its 11.5%-20% active-sizing band. A two-symbol 50/50 split at the **current live balance (~$87)** means each leg's entry notional would be computed on a ~$43.50 sub-balance, not $87 — re-running that document's exact table logic at $43.50 is required here (entry notional low end = $43.50 x 0.115 = $5.00, sitting essentially **at** the $5 `NOTIONAL` floor, not comfortably above it like the single-symbol case). This experiment states explicitly, before any backtest is run: **at the current live balance, a real two-symbol split is at or below the minimum-notional floor at the low end of HyperGrowth's sizing band, and is not viable today regardless of what the backtest shows.** The backtest result in this document characterizes the mechanism (does diversification-for-variance work in principle, at unconstrained size) — it is explicitly a research question, not a deployment plan, until capital crosses roughly **$150-$175** (2x the single-symbol $80 floor found in the capital-sizing-knee document, since each leg now needs to individually clear the same headroom margin). Any "supported" verdict in §6 is reported with this ceiling attached, not silently implied to be actionable at $87.

---

*Locked 2026-08-25, before any statistic is computed. Results appended below in a dated section per the anti-p-hacking rule.*
