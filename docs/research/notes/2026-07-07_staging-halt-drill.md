# Staging manual kill-switch drill — 2026-07-07

Post-merge live-fire validation of PR #929 (`atb live-control halt/resume`, closes #922) on the
staging environment (paper trading, no live capital). PM-authorized per the #929 validation plan;
within the chartered staging-drill autonomy envelope. Executed per
`.claude/skills/deploy-staging/SKILL.md` and `.claude/skills/kill-switch-drill/SKILL.md`.

All timestamps UTC. Operator: `cli:alex` (Claude agent on Alex's machine). The only mutations
performed: the staging branch sync PR and the staging `system_halt` flag (set, then cleared).
No production resources were touched; all DB reads ran with `default_transaction_read_only = on`.

## Verdict: PASS (6/6 drill steps), 1 pre-existing finding (#915, webhook parity)

| # | Step | Result |
|---|------|--------|
| 0 | Deploy: sync staging→develop, boot verification | PASS |
| 1 | Staging bot running, check interval noted | PASS (60s) |
| 2 | `halt` CLI: masked target echoed before mutation + account summary | PASS |
| 3 | Engine SYSTEM_HALT enforcement + entry-evaluation skip within one interval | PASS (26s) |
| 4 | Exits/stops ungated; webhook alert | PASS (ungated) / FINDING (no staging webhook — pre-existing #915) |
| 5 | `resume`: SYSTEM_HALT_CLEARED + entry evaluation resumes | PASS (15s) |
| 6 | Second `resume` is a clean no-op | PASS |

## Part 1 — deploy (staging sync)

- Sync PR **#934** (`develop` → `staging`, merge commit per skill): merged `2026-07-07T10:52:59Z`,
  merge commit `8ded4805`. Staging now carries develop @ `1f3ffcde` — includes **#929**
  (system_control_flags table, SystemHaltEnforcer, halt/resume CLI), **#925**, **#923**.
- Railway deployment `7e14ab77-0cbb-4bb9-8140-0a385b3163f6` (env `staging`, service "Trading Bot"):
  created `10:53:01Z`, **SUCCESS** `10:56:45Z`.
- Boot verification (per deploy-staging skill):
  - `10:56:42Z` `Check interval: 60s` (ETHUSDT 1h, HyperGrowth)
  - `10:56:44Z` `💾 Recovered balance $1001.17 from recent inactive session #18`
  - `10:56:47Z` `Created trading session #19: HyperGrowth_ETHUSDT_20260707_105644`
  - `10:56:51Z` `🔁 Carried 1 OPEN position(s) forward from inactive session #18 into new session #19` (#668 re-adoption)
  - `10:56:54Z` `Trading loop started`
  - `grep -cE "\[ERROR\]|UNVERIFIED"` over boot logs = **0** — no errors, and no
    `SYSTEM_HALT_UNVERIFIED` (the #929 fail-closed priming read succeeded on first try).
  - `system_control_flags` table auto-created on boot (`Database tables created/verified`;
    `DatabaseManager._create_tables` runs `create_all`), so no manual migration was needed.

## Part 2 — drill

### Step 1 — bot running, check interval

Confirmed above: loop started `10:56:54Z`, adaptive check interval base **60s**
(`DEFAULT_CHECK_INTERVAL`, bounds 30–300s). PASS.

### Step 2 — halt command

```
$ atb live-control halt --env staging --reason "post-merge validation drill (PR #929)"   # invoked 10:58:29Z
[2026-07-07T10:58:35Z] target: env=staging db=switchyard.proxy.rlwy.net:12631/railway (from $RAILWAY_STAGING_DATABASE_URL)
[2026-07-07T10:58:37Z] system_halt HALT set (env=staging, by=cli:alex, reason=post-merge validation drill (PR #929))
[2026-07-07T10:58:37Z] engine effect: entries + scale-ins BLOCKED within one trading-loop iteration; exits/stops/reconciliation UNAFFECTED
[2026-07-07T10:58:37Z] WARNING: no alert webhook delivery (ALERT_WEBHOOK_URL unset or POST failed) — page the operator manually
[2026-07-07T10:58:37Z] active session: 19 (env=staging)
[2026-07-07T10:58:38Z] open positions: 20
[2026-07-07T10:58:38Z]   ETHUSDT LONG qty=0.05560472 entry=1965.892455 sl=1768.419 (order not placed) tp=2554.383 upnl=-2.28968843
  … (20 rows, every one with sl= and tp= populated; none flagged NO STOP)
[2026-07-07T10:58:38Z] verify enforcement: system_events error_code=SYSTEM_HALT (emitted by the engine when it honors the flag)
exit=0
```

- Masked target (`host:port/dbname`, credentials stripped) echoed **before** the mutation. PASS.
- Account-state summary printed (active session, positions, protective stops). PASS.
- Note: the 20 "open positions" are all OPEN DB rows; the engine's tracker holds 1 (carried
  forward into session #19). The other 19 are stale rows from dead sessions — the known
  staging gotcha documented in the deploy-staging skill; the engine ignores them.

### Step 3 — engine enforcement (target: within one 60s interval)

Flag written `10:58:37Z` → enforced `10:59:03Z` = **26 seconds**. Exact log lines:

```
2026-07-07T10:59:05.230955543Z [ERRO] 🛑 MANUAL SYSTEM HALT ENFORCED (set by cli:alex, reason: post-merge validation drill (PR #929)). New entries and scale-ins are blocked; exits, stop-losses and reconciliation continue. Clear with 'atb live-control resume'. timestamp="2026-07-07 10:59:03,372" logger="atb.src.engines.live.monitoring.system_halt_enforcer"
2026-07-07T10:59:06.260933039Z [WARN] MANUAL SYSTEM HALT active (reason: post-merge validation drill (PR #929)) — skipping entry evaluation for ETHUSDT (exits, partial exits, stop-loss management and reconciliation continue) timestamp="2026-07-07 10:59:06,201" logger="atb.src.engines.live.execution.entry_pause"
```

Entry-evaluation skip observed in the same iteration. (Subsequent skips log at DEBUG — the WARN
is rate-limited to one per `ENTRY_PAUSE_WARNING_INTERVAL_SECONDS` = 300s by design.) PASS.

### Step 4 — exits/stops ungated; webhook alert

Loop iterations and position monitoring continued normally while halted (only entry evaluation
was skipped):

```
2026-07-07T10:59:05Z [INFO] Trading loop: current_index=499, last_candle_time=2026-07-07 10:00:00
2026-07-07T10:59:06Z [INFO] 📊 Status: ETHUSDT @ $1775.42 | Balance: $1001.17 | Positions: 1 | Unrealized: $4.33 | Trades: 0
2026-07-07T11:00:16Z [INFO] Trading loop: current_index=499, last_candle_time=2026-07-07 11:00:00
2026-07-07T11:00:16Z [INFO] 📊 Status: ETHUSDT @ $1774.65 | Balance: $1001.17 | Positions: 1 | Unrealized: $4.39 | Trades: 0
```

Code-path scope check: the halt state is consumed only by `EntryPauseGate` (entry + scale-in
paths); `LiveExitHandler.bind_system_halt` exists solely so scale-ins routed through the exit
handler can't bypass the gate — exit/stop-loss paths are not gated. Exits/stops ungated: **PASS**.

**Webhook: FINDING (pre-existing).** The staging "Trading Bot" service has **no
`ALERT_WEBHOOK_URL`** (verified via `railway variables -e staging`: no alert/webhook/slack keys),
so no Slack page fired for either the command or the enforcement events. The mechanism behaved
exactly as designed for an unset channel: the CLI printed the loud
`WARNING: no alert webhook delivery … page the operator manually` line, and every `system_events`
row honestly recorded `alert_sent=f` (2xx-only semantics). This is the staging webhook-parity gap
already tracked in **#915** (open, `priority:p3`) — a finding per the kill-switch-drill skill
("if staging has no webhook, that's a drill FINDING, not a skip"), not a #929 regression and not
a new issue. Alert delivery itself was NOT validated end-to-end by this drill; it will be once
#915 lands.

### Step 5 — resume

```
$ atb live-control resume --env staging --reason "drill complete — validation passed"   # invoked 11:00:46Z
[2026-07-07T11:00:52Z] target: env=staging db=switchyard.proxy.rlwy.net:12631/railway (from $RAILWAY_STAGING_DATABASE_URL)
[2026-07-07T11:00:54Z] system_halt RESUME set (env=staging, by=cli:alex, reason=drill complete — validation passed)
[2026-07-07T11:00:54Z] engine effect: entries + scale-ins re-enabled within one loop iteration
exit=0
```

Engine honored it in **15 seconds**:

```
2026-07-07T11:01:16.515312824Z [WARN] ✅ Manual system halt cleared — new entries and scale-ins are enabled again. timestamp="2026-07-07 11:01:09,374" logger="atb.src.engines.live.monitoring.system_halt_enforcer"
```

Subsequent iterations (`11:01:10Z`, `11:02:13Z`) ran with **no** skip lines — entry evaluation
resumed on the next cycle. PASS.

### Step 6 — idempotent second resume

```
$ atb live-control resume --env staging   # invoked 11:01:57Z
[2026-07-07T11:02:05Z] system_halt already CLEAR (env=staging, since=2026-07-07T11:00:53.890471, reason=drill complete — validation passed, by=cli:alex) — no change
exit=0
```

No flag mutation, no new `system_events` row (trail below ends at id 47), no engine transition.
Clean no-op. PASS.

## Database evidence (read-only)

`system_events` trail (staging DB, `SET default_transaction_read_only = on`):

| id | timestamp (UTC) | error_code | severity | alert_sent |
|----|-----------------|------------|----------|------------|
| 44 | 10:58:37.504 | SYSTEM_HALT_COMMAND | critical | f |
| 45 | 10:59:03.372 | SYSTEM_HALT | critical | f |
| 46 | 11:00:54.040 | SYSTEM_RESUME_COMMAND | warning | f |
| 47 | 11:01:09.375 | SYSTEM_HALT_CLEARED | warning | f |

Final flag state: `system_halt` `active=f`, reason `drill complete — validation passed`,
source `cli:alex`, updated `11:00:53.890Z`. No env flags were touched during the drill, so
there is nothing to restore; staging is back to its pre-drill operating state with #929 live.

## Timing summary

| Transition | Flag write → engine action | Bound |
|------------|---------------------------|-------|
| Halt | 10:58:37 → 10:59:03 = 26s | ≤ one 60s interval ✅ |
| Resume | 11:00:54 → 11:01:09 = 15s | ≤ one 60s interval ✅ |

## Follow-ups

- **#915** (open, p3): add `ALERT_WEBHOOK_URL` to the staging Trading Bot service so the next
  drill can validate end-to-end alert delivery (`alert_sent=true` + visible Slack message).
  No new issue filed — this drill's evidence (events 44–47 with `alert_sent=f`) is additional
  confirmation of the same gap.
- `risk-limits.json`'s `manual_trigger_command: "atb live-control halt"` is now real and
  staging-verified — resolves the kill-switch-drill skill's step-4 "vapor kill-switch" caveat
  (dated 2026-07-06) for the manual-trigger line.
