# Weekly Strategy Review

Weekly deep-dive: are live strategies behaving the way their backtests said they would? Strategies decay silently — this is the scheduled check that catches it.

## Instructions

1. **Gather the data** (do this yourself, no subagent needed):
   - Last 7 days of trades per strategy from the `trades` table.
   - Corresponding `performance_metrics` rows.
   - Current `latest` model version per symbol (`atb live-control list-models`).

2. **Dispatch `quant-researcher`** with this prompt:
   > "Produce a backtest-vs-live divergence report for the last 7 days. For each strategy currently running live:
   > (a) Run a backtest over the same 7-day window with identical parameters.
   > (b) Compare on: total return, Sharpe, max drawdown, trade count, fee+slippage as % of gross.
   > (c) Flag any metric that diverges by more than 2σ of the strategy's historical week-to-week variance.
   > (d) For each divergence, form a hypothesis: parameter drift, regime shift, data issue, execution issue, or model decay.
   > Save the report under `docs/research/weekly-reviews/YYYY-WW.md`. Return a summary with the number of divergences found and the top 3."

3. **Dispatch `ml-engineer`** in parallel with:
   > "Produce a drift report for all models currently promoted to `latest`. Compare last-7-days live prediction distribution vs training distribution. Report feature drift, prediction drift, and realized performance decay separately. Save under `docs/research/model-drift/YYYY-WW.md`."

4. **Dispatch `risk-officer`** in parallel with:
   > "Weekly risk review: 7-day drawdown, concentration history, any threshold approaches. Has the risk profile shifted vs the prior week? Save under `docs/research/risk-reviews/YYYY-WW.md`."

5. **Wait for all three.** Synthesize findings into `docs/research/weekly-reviews/YYYY-WW_summary.md` using this structure:

```
# Weekly Review — Week WW, YYYY

## Verdict
[continue / investigate / reduce / halt — one-liner]

## Divergences (from quant)
- [strategy]: [metric] diverged by X, hypothesis: …

## Model drift (from ml-engineer)
- [model]: [signal strength]

## Risk shifts (from risk-officer)
- …

## Proposed actions
- [bulleted; each classified as autonomous-OK or board-required]

## Recommended experiments for next week
- …
```

6. **Update `docs/project_status.md`** with the outcome and next-week focus.

## When to run

- Weekly, ideally weekend (markets are 24/7 but human attention is not). Schedule via cron.
- Ad-hoc after any material live-config change.

## Guardrails

- Do not change model `latest` symlinks, strategy config, or position sizing as part of this review. Outputs are recommendations.
- If divergences are severe enough that `risk-officer` returns `reject` on the current live config, escalate the same day rather than waiting for the human to read the report.
- The review must cover the strategies *actually running live*, not the ones we wish were running. Verify via `atb` / config before dispatching.
