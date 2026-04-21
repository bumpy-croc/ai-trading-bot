# Drawdown Alert

Tripwire check: is current drawdown within configured limits? Runs the risk desk and escalates if breached. Designed to be invoked on a short schedule (e.g., every 15 minutes) or on demand.

## Instructions

1. **Quick check first** — cheap, no subagent call:
   - Read latest `account_history` row via a read-only query against the configured DB.
   - Compute rolling drawdown: `(peak_balance - current_balance) / peak_balance`.
   - Read the configured max-drawdown threshold from `src/config/` (or env).

2. **If drawdown < 50% of threshold**: write a one-line heartbeat to `docs/research/ops-snapshots/drawdown-heartbeat.log` and exit. No subagent call needed.

3. **If drawdown ≥ 50% of threshold**: dispatch `risk-officer` in live-monitor mode. Prompt: "Current drawdown is X% of balance (threshold Y%). Produce a risk snapshot and a recommended action. Be concrete: continue / reduce size / halt new entries / kill-switch."

4. **If drawdown ≥ threshold**: escalate IMMEDIATELY.
   - Dispatch `risk-officer` AND `live-ops` in parallel.
   - Write an incident record to `docs/research/incidents/YYYY-MM-DD_HHMM_drawdown.md` with: current balance, peak balance, positions open, last 20 trades, full risk verdict.
   - Print a red-flag summary to the user with a recommended action.
   - Do NOT execute kill-switch — that's a human decision.

## Output

- Heartbeat case: silent (file only).
- Warning case: one-paragraph summary + path to risk snapshot.
- Breach case: full red-flag incident report + file path + explicit "human decision required" line.

## Guardrails

- This command is read-only. It never modifies positions, config, or code.
- If the DB is unreachable, treat that as a P0 and escalate to `live-ops` — do not assume "no data" means "no drawdown".
- Never silently dismiss a breach. If the numbers look wrong, escalate; don't second-guess.
