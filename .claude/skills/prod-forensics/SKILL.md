---
name: prod-forensics
description: Read-only production forensics toolkit — decompose "where did the money go" from the account_balances ledger, locate fees, detect phantom positions/balances, and reconstruct history after logs are gone. Use for money-flow questions, balance discrepancies, position-truth disputes (DB vs exchange), or any historical incident investigation on prod/staging Postgres.
---

# Prod Forensics

Answering "what actually happened to the money/positions" with zero mutation risk. Every recipe
here was used on a real investigation (the −$16.71 two-week decomposition, the 2026-06-05
phantom position #12, the #913 latency-error hunt). Writes findings to layer 2 (a note under
`docs/research/notes/` + a log.md entry); never writes to prod. See
`docs/architecture/memory_system.md`.

## Read-only discipline (non-negotiable)

```bash
# Get the PUBLIC proxy URL (the bot's own DATABASE_URL is internal-only, unreachable locally):
railway variables -e production -s Postgres --json | jq -r .DATABASE_PUBLIC_URL
psql "$URL"
-- FIRST statement, before anything else:
SET default_transaction_read_only = on;
```

- `railway ssh` / `railway run` are policy-denied (LESSONS §3). The variables-read + local psql
  path is the allowed route; the prod-variables dump may still prompt — that's correct, get the
  human OK. Same recipe for staging via `-e staging`.
- Logs are NOT a historical source: `railway logs` serves only the current deployment and
  `--since` never reaches prior containers (#913 lesson). History lives in Postgres:
  `strategy_executions`, `system_events`, `reconciliation_audit_events`, `orders`, `trades`.

## Money flow: the `account_balances` ledger

`account_balances.update_reason` summed deltas = the authoritative "where money went":
`entry_fee_<SYM>`, `realized_pnl_<SYM>_<exit_reason>` (gross pnl net of exit fee),
`margin_equity_sync_correction`, `session_start`. Reference finding: the 2-week −$16.71 to
2026-06-08 was −$17.12 of `margin_equity_sync_correction` (phantom $100 → true $84 books
catching up), realized P&L +$0.53, fees ≈ $0.17 — NOT fees, NOT trading.

**THE LAG PITFALL:** computing `total_balance - lag(total_balance) OVER (ORDER BY last_updated,
id)` AFTER a `WHERE update_reason = …` filter makes `lag` span unrelated events → fake
cross-event deltas (produced phantom −$15.76 "entry fee" rows once). Compute `lag` over the
FULL ordered table in a CTE, THEN filter/group.

## Fees: three locations, two denominations

- `orders.actual_commission` — authoritative, **denominated in the RECEIVED asset**: base
  (ETH) on buys, quote (USDT) on sells. NEVER sum raw across both sides.
- `account_balances` `entry_fee_*` rows — USD, ledger-consistent.
- `trades.commission`/`quantity` — populated only since #731/#831; older rows are 0/NULL, so
  historical fee analysis must use the two sources above.

## Position truth: phantom detection

DB `positions` rows are claims; the exchange is truth. The 2026-06-05 read-only reconciliation
proved DB row #12 a phantom (2 OPEN rows vs 0.00378 ETH held and ONE live SL order): compare
per-symbol DB OPEN quantity vs exchange holdings, and each row's SL `order_id` vs
`get_open_orders`. On margin, split `free` vs `borrowed` vs `netAsset` before calling anything
a position — "dust" is usually an un-repaid borrow (LESSONS §1.5). Aggregate-balance checks
can't see phantoms (a phantom "borrows" the real position's holdings — the #679 adopt-all trap).

## Balance truth: the pinned-book-value check

Pre-2026-06-03 `account_history` equity is BOOK VALUE, not a live read (May 2026: ONE distinct
balance value across 451 hourly rows). Before trusting any peak/trough/drawdown:

```sql
SELECT date_trunc('month', timestamp) m, count(DISTINCT round(balance::numeric,4))
FROM account_history GROUP BY 1 ORDER BY 1;
```

Near-constant months = software-pinned; do not compute drawdowns across them. This check is what
withdrew the false "20.33% breach" claim (log.md 2026-07-04 13:55 correction). Adopted baseline:
peak = peak TRUE equity since the last reconciled reset (2026-06-05 / session 20).

## Sessions & liveness

Prod REUSES its active `trading_sessions` row across restarts — a missing "new session row" is
not an outage signal. Liveness ground truth: the hourly `account_history` heartbeat row + recent
`strategy_executions`. A gap in heartbeats brackets an outage window precisely (the 2026-05-19
zombie-bot class, where the deploy API said SUCCESS throughout).

## Decision forensics (what was the bot thinking)

`strategy_executions` carries per-decision signal/confidence/price rows — good enough to
reconstruct a 30-min entry decision trail (done for suspect position 22 in the #913 hunt, which
cleared prod of ever trading a latency-error result). Note `ml_predictions` is null before
#914/#917 landed. Anchor error-code greps in any log dump you do have (`code=51077`, not bare
digits — a nanosecond suffix once matched).

## Output

A dated note under `docs/research/notes/` (evidence: queries + row counts, not vibes) + a log.md
entry via `decision-record` if the finding is material. Corrections to earlier claims follow the
append-only correction pattern (memory_system.md discipline 1).
