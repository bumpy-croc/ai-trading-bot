---
id: 2026-09-14T2052-P1-close-only-inventory-lock
opened_by: pm-session-6ee0f9
severity: P1
status: closed        # open | mitigated | closed
opened_at: 2026-09-14T20:52:00Z
mitigated_at: 2026-09-15T07:39:43Z
closed_at: 2026-09-15T23:53:28Z
human_paged: false
affected_components: [live-engine, execution, close-only-latch, stop-loss]
affected_symbols: [ETHUSDT]
---

## What happened

Production latched **close-only mode** at **2026-09-14 20:52 UTC** on position 29 (ETHUSDT) after
a recurring `CLOSE_INVENTORY_LOCKED` abort loop — the same symptom class as the 17-day #1121 halt
(2026-08-27 to 2026-09-14, closed via a promote+restart only hours earlier that same day, at
06:54:11 UTC). The stuck position exited via its exchange-side stop-loss **7 minutes later**
(**+$0.19 pnl**) — real exchange-side protection held, book flat again quickly, no capital at risk
— but the close-only latch itself, being a plain in-process bool with no remote-clear path (the
same structural gap named in GH #1127), stayed on for **~10h50m** of pure opportunity cost until
the next deliberate promote+restart.

## Detection

Caught mid-session by the PM daemon (session `6ee0f9`) while investigating the fresh
`CLOSE_INVENTORY_LOCKED` cycle directly from a pasted prod log excerpt, not by a scheduled standup
sighting — this incident never went 24h+ unactioned the way #1121 did.

## Root cause

**Proximate (this incident, #1165):** the close path cancels the resting stop-loss (#710) and then
immediately reads free base balance to size the close order. Binance's cross-margin wallet endpoint
(`/sapi/v1/margin/account`) is eventually consistent and can still report the just-cancelled stop's
`locked` amount for several seconds afterward, so the close sizes itself off a stale pre-cancel
snapshot and trips #1108's 98%-of-holdings cap gate on a position that is, in fact, fully sellable.
Abort → reprotect → the identical exit re-fires on the next ~66s cycle → three consecutive aborts
latch close-only, per the same threshold #1121 hit.

Full diagnosis: `agents/research/1165-recurring-close-inventory-lock.md`; narrative and fix detail:
`.claude/state/log.md` entry `[D-2026-09-15-01]`.

**Bypass root cause (#1166, fixed):** a separate read-only investigation opened alongside #1165
confirmed the trading loop itself never hung — only the latch (plus a stale trade-count-gated
status line, #1170) made it look dead. #1166 found that the partial-exit close path bypassed
#710's cancel-lock, a related but distinct gap in the same close/cancel/re-place surface; fixed in
PR #1183 (merged 2026-09-15T23:53:28Z).

**Structural (still open, #1127):** clearing the latch, once its trigger condition has resolved
and the book is flat, still requires a human-authorized restart — no scheduled agent in the fleet
has authority to clear it on its own. This incident cleared in ~10h50m only because a PM session
happened to run and promote a fix that same night; absent that, it would have re-latched the same
multi-day pattern as #1121.

## Impact

Live capital, ETHUSDT position 29. **No capital loss** — the position closed via its exchange-side
stop-loss within minutes of the latch, and the book stayed flat for the remainder of the window.
Pure opportunity cost: ~10h50m of blocked new entries (2026-09-14 20:52 UTC → 2026-09-15 07:39:43
UTC), well inside the charter's response envelope compared to #1121's 430x-SLA precedent, but the
underlying structural gap (#1127) is unchanged — this was fast only because a human-initiated
session happened to be running.

## Fix

PR #1165 (two independent architecture/code review rounds, each surfacing a real defect the other
caught): threads a `stop_just_cancelled` flag from the confirmed cancel down to a bounded free-base
balance-read retry (5 attempts, 300ms apart), padded by one lot step and capped at the position's
own intended quantity, so the retry's raw-balance verdict survives the caller's later
floor-to-step normalization without ever exceeding what is actually needed.

Promoted `develop @ 6f829d5f` → `main` (`cfc9a506`, 2026-09-15 07:39:43 UTC) — zero conflicts,
imports and the live model symlink verified, full fast suite green (2595 passed) pre-merge and in
the promote worktree. The restart that deployed this commit also cleared the already-latched
close-only state as a side effect, since the fix cannot clear a process that latched before it
shipped.

## Timeline

```
2026-09-14 06:54:11 UTC — unrelated: promote+restart (be451698) clears the prior #1121 latch
2026-09-14 20:52:00 UTC — new close-only latch sets on position 29 (ETHUSDT), CLOSE_INVENTORY_LOCKED
2026-09-14 ~21:00 UTC   — stuck position closes via exchange-side stop-loss, book flat
2026-09-14 21:53 UTC    — PM session confirms latch still active in the running process
                          (fix not yet deployed; a running process cannot self-clear)
2026-09-15 ~00:40 UTC   — PM session (6ee0f9) root-causes and fixes #1165, spins out 7 related
                          follow-up issues, merges PR #1165 to develop
2026-09-15 07:39:43 UTC — promote `cfc9a506` deployed; restart clears the latch as a side effect
```

## Recovery requirements

Same structural item as #1121, still open: **#1127** — build a safe auto-clear or remote-clear
path for the close-only latch once its trigger condition has resolved and the book has been flat
for N cycles, so a third occurrence does not again default to "wait for a session to run."

## Post-mortem

### Root cause
See above — proximate cause fixed (#1165, PR #1165, promoted `cfc9a506`); structural gap open (#1127).
### Contributing factors
Eventually-consistent exchange balance read used for a sizing decision immediately after a
cancel, with no retry/settle window — the same "trust the exchange snapshot too soon" bug shape as
prior precision/timing incidents in this repo (see `.claude/LESSONS.md`).
### What went well
Caught and root-caused within the same session it occurred, not by a delayed scheduled sweep;
fixed with two independent review rounds that each found real defects before merge; promoted and
verified live the same night.
### What went poorly
The close-only latch itself still required a human-initiated session to clear — the exact gap
#1127 names. This incident happened to resolve fast only because a PM session was already running;
absent that, it reproduces #1121's multi-day pattern.
### Action items
#1165 (fixed, PR #1165, merged, promoted `cfc9a506`), #1127 (open, structural — unchanged by this
incident), #1166/#1167/#1168/#1169/#1170/#1171/#1172/#1173/#1174/#1181/#1184 (spun-out follow-ups,
tracked separately per `.claude/state/log.md`'s `[D-2026-09-15-01]` entry).

Ref: `.claude/state/log.md` `[D-2026-09-15-01]` (full narrative), GH #1165 (closed via PR #1165),
#1127 (open), #1121 (closed, prior occurrence of the same failure shape).
