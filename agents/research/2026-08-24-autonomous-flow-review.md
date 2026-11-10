# Autonomous flow review — 2026-07-24 to 2026-08-24

Requested by Alex, 2026-08-24. Sources: `.claude/state/log.md` (full), `.claude/state/incidents/*.md`,
`.claude/skills/weekly-retro/AGENDA.md`, `.claude/LESSONS.md` diffs for the window, GitHub issues/PRs
(`gh`), and claude-mem session history filtered to this project. All issue numbers below were spot-checked
directly against GitHub, not just taken from subagent reports.

## Headline

Every category the review was asked about — credentials, stuck agents, missing safety gates — traces back
to **two root mechanisms**, not a long tail of unrelated bugs:

1. **No PM presence → no throughput.** The daemon proposes, reviews, and produces fixes fine. It cannot
   *land* them without Alex actively in a session to merge. [#1079](https://github.com/bumpy-croc/ai-trading-bot/issues/1079)
   (filed today) names this directly: two stalls of 27 and 11 days in the window, both invisible internally
   because scheduled tasks kept running and producing PRs while nothing shipped.
2. **Monitors assert absence-of-bad-signal instead of presence-of-expected-state.** At least six distinct
   checks (standup liveness, drawdown-guard grep, scheduled-task `lastRunAt`, risk-limits config load,
   pre-push hook, close-only detection) report "healthy" whenever they see no error — which is exactly the
   signature a broken component with no output also produces. Every P1 this month that went undetected for
   days did so through one of these checks, not despite them.

Everything else — lost worktrees, stranded PRs, the zero-size-decision symptom staying open for a month,
the shared-venv bug reaching a Board decision — is a downstream instance of one of those two.

---

## A. Credentials / access blockers

- **Railway CLI auth expiry, chronic and near-weekly** (Aug 4, 7, 9, 11, 12, 13). A session-token CLI login
  that keeps expiring, discovered reactively each time by a failed command rather than proactively. Already
  in project memory as a known recurring issue; still recurring through the whole window with no fix
  attempted.
- **Missing `timeout`/`gtimeout` (coreutils)** on the ops-checking environment (Aug 22) — blocked a
  safety-timeout-wrapped `railway logs` pull. A distinct, previously undocumented instance of the same
  "tooling gap disguised as a one-off" pattern.
- **[#944](https://github.com/bumpy-croc/ai-trading-bot/issues/944) — open since 2026-07-09, unresolved
  the entire window.** This is the one credential/permission item that's an actual *safety gate* gap, not
  friction: `settings.local.json`'s deny-list for Railway MCP tools is unenforced/inconsistent — the same
  class of hole that let `railway domain` create an unauthorized public prod domain in the 2026-07-08
  incident (#941). Six weeks later the only enforcement is still a prose rule in `live-ops.md`, not a
  technical control.

## B. Agents stuck, timed out, or lost work

- **Stranded work, recovered only by accident.** [#1026](https://github.com/bumpy-croc/ai-trading-bot/pull/1026)
  (a completed P1 post-mortem) was closed unmerged 2026-07-21; the incident record sat with an empty
  post-mortem section for **42 days** until a worktree/branch audit salvaged it today via PR #1107 — just
  before the source branch would have been deleted. A Board-directed predictability scan
  ([#1105](https://github.com/bumpy-croc/ai-trading-bot/pull/1105)) sat **uncommitted in a worktree for 11
  days** the same way. The very next weekly retro after the #1026 loss (PR #1047, Aug 10) repeated the exact
  anti-pattern the loss was supposed to teach — rebasing a distillate onto another still-open PR instead of
  `develop` — and #1047 itself then sat unmerged for **14 days**.
- **Worktree accumulation never resolves.** `prune-worktrees` fired repeatedly (Aug 17–24) while the orphan
  count went *up* (7→9). By Aug 22, all 7 active worktrees were dirty; two hadn't been touched in 48 and 39
  days. The pruner appears to refuse anything with uncommitted changes, which is correct caution, but there's
  no complementary "flag for a human decision" path — so genuinely abandoned work just accumulates silently.
- **PR lead times during the stall windows**: #1047 — 17 days; #1060/#1069/#1072/#1073/#1074 (all opened Aug
  13) — ~10.6–10.8 days each, landing in one catch-up burst; #1076 — ~6.9 days. Merge activity itself is
  bursty: 13 PRs in one day (Aug 13), 4 PRs in a 29-minute window (Aug 24), zero merges on most other days.
- **Scheduled-task silent-kill, four distinct mechanisms, same symptom.** Claude usage-quota exhaustion
  killed `daily-trading-standup` mid-run (Aug 15) and produced no session at all the next day, so
  `weekly-model-retrain` missed its only weekly slot; a stale model-provider selection killed runs on turn 1
  with no retry ([#1051](https://github.com/bumpy-croc/ai-trading-bot/issues/1051), cost a full week of
  retro+standup output Aug 3–5); a missed 2026-08-19 slot got no catch-up and 5 subsequent runs reported PASS
  anyway ([#1085](https://github.com/bumpy-croc/ai-trading-bot/issues/1085)); the shared-venv bug (below)
  silently executed some scheduled runs against the wrong checkout entirely.
- **[#1090](https://github.com/bumpy-croc/ai-trading-bot/issues/1090)** — a merge race where the weekly
  retro cleared `AGENDA.md` on `develop` while a PR still carrying an unactioned agenda item was open;
  the item was silently dropped from both sides.

## C. Safety gates missing, bypassed, or that failed to catch a real problem

- **The zero-size-decision symptom is the standout of the month.** First seen Jul 24, filed as
  [#1045](https://github.com/bumpy-croc/ai-trading-bot/issues/1045) Jul 27, independently rediscovered on
  staging *and* production roughly **ten separate times** through Aug 23 (most recently: 72.4% of 635
  sampled live decisions were `Size: 0.00` with no logged `gate_reason`). It is precisely the failure mode a
  bot exhibits while silently broken, and it's still open at P2 a full month after first discovery. Every
  rediscovery got filed or re-noted; none forced a priority bump or a fix.
- **Prod close-only latched silently for ~4 days**
  ([#1094](https://github.com/bumpy-croc/ai-trading-bot/issues/1094), 2026-08-20→08-24, P1). A swallowed
  Binance exception in `place_stop_loss_order` triggered close-only; `daily-trading-standup`'s liveness check
  greps for `Decision:` log lines emitted *before* the close-only early-return, so it reported NOMINAL all 4
  days. A Slack CRITICAL alert did fire and Alex did see it — the second gap
  ([#1096](https://github.com/bumpy-croc/ai-trading-bot/issues/1096)) was that nothing re-escalated an
  unacknowledged CRITICAL alert. Both fixed same-day via #1103 (positive re-announce-until-cleared pattern);
  root cause of the swallowed exception fixed via #1097.
- **Shared-venv stale-checkout execution**
  ([#1070](https://github.com/bumpy-croc/ai-trading-bot/issues/1070), P0, closed Aug 24 via #1080). The
  editable install pinned every `atb`/backtest invocation to whichever checkout it was first installed from,
  regardless of worktree — a 365-day backtest returned **+114.69% and -28.29% on consecutive runs**, and
  this had already fed a Board-level risk decision before being caught. It was originally filed at **P3**
  ("add a warning") on 2026-07-13 and sat 31 days before being correctly re-prioritized. Its sibling,
  relative reads resolving against the primary checkout after a cwd reset
  ([#1082](https://github.com/bumpy-croc/ai-trading-bot/issues/1082)), was fixed the same day via #1087 (the
  primary-checkout write-protection hook now in place).
- **Config that governs nothing.** `risk-limits.json` was found inert at runtime — zero `src/` consumers,
  the live value hardcoded in `constants.py:140` — meaning the Board had been ratifying changes to a file
  that didn't control anything. The remediation PR (#1073, hydrate `RiskParameters` from the ratified file)
  itself introduced new reporting regressions the same day
  ([#1088](https://github.com/bumpy-croc/ai-trading-bot/issues/1088)/[#1089](https://github.com/bumpy-croc/ai-trading-bot/issues/1089)/[#1102](https://github.com/bumpy-croc/ai-trading-bot/issues/1102)):
  backtests now silently truncate at 20% drawdown with no signal the result is partial, and the early-stop
  threshold falls back to 0.5 (not the ratified limit) when `risk_parameters` is `None`.
- **Cloud ML training pipeline silently produces garbage predictions**
  ([#1049](https://github.com/bumpy-croc/ai-trading-bot/issues/1049), P1, open ~2 weeks, same bug class as
  the pre-#838 partial-exit fabrication). SageMaker-trained bundles omit `price_normalization` metadata;
  both inference call sites silently skip denormalization instead of raising, so a promoted cloud-trained
  model would feed strategies raw [0,1]-range values as real prices. Still blocking the weekly retrain
  evaluation as of Aug 23.
- **`pre-push` hook was inert.** Piped exit-status masking plus a cwd-relative `.venv` path meant it printed
  "All fast tests passed" unconditionally, including when the interpreter was missing
  ([#1077](https://github.com/bumpy-croc/ai-trading-bot/issues/1077), closed Aug 24 via #1092).
- **One genuine save worth preserving as a pattern**: the pre-committed 14-day staging window for arming
  circuit breakers ([#986](https://github.com/bumpy-croc/ai-trading-bot/issues/986)) completed 2026-07-28,
  and when finally evaluated (16 days late — a queue-priority failure, not a charter ambiguity) risk-officer
  correctly rejected it: the window hadn't actually exercised a restart, and the one restart that did occur
  had a peak-seeder bug that never ran, meaning the "pass" was reading 0.020% drawdown against a true 1.374%
  fall. Arming would have shipped a control that was 69x blind. This is the review gate working exactly as
  designed and should be used as a training example, not filed away as a footnote.

## D. Escalation calibration

- **No instance found of over-escalation**, and no instance of a dispatched reviewer going silent on a
  genuine P0/P1 without eventually surfacing it.
- **One clear self-correction worth codifying**: a standup's first pass read "production bot inactive 21
  days" from a single table (`trades`) and, before escalating, cross-checked `account_history` and found the
  bot was actively heartbeating and holding a monitored position — it just hadn't *closed* a trade (selective,
  not broken). The corrected finding replaced the alarm in the same pass. Worth turning into a standing rule:
  **never escalate an anomaly read from a single table/log source without cross-checking at least one other
  durable source first.**
- **Under-escalation is the real risk, and it's specific**: #1045 and the risk-limits path mismatch were
  each correctly identified early, filed, and then *re-surfaced by the same automated check roughly ten
  times* without ever forcing a priority bump or a distinct human ping. Filing an issue is being treated as
  if it were resolution — this is already named in `.claude/LESSONS.md` §2.11 as an anti-pattern, but the
  backlog data below shows it's still happening at scale.
- **62 open issues, zero assignees, as of today.** #697 (short-entry block) — 76 days stale. #845 (the P1
  drawdown-breach incident itself, still `type:incident` and open) — 28 days stale despite three downstream
  fixes shipping in the meantime; it should have been explicitly closed or re-scoped. #1038 — 28 days stale.

## E. What's already been fixed

Credit where due — 2026-08-24 was the largest single-day remediation push of the window, closing #1070,
#1082, #1077, #1094/#1095/#1096 same-day. But the fixes are treating instances: #1085 (missed-slot detection
gap) and #1090 (AGENDA.md race) were both *found*, not fixed, on the same day the backlog got cleared,
confirming the underlying pattern (C's "absence read as health") is still generative.

---

## Recommendations

Ordered by leverage — highest-impact / already-diagnosed first.

### 1. Fix the PM-absence bottleneck (issue already filed: #1079)
This is the single highest-leverage change available: it's the reason a fixed post-mortem sat 42 days, a
completed research scan sat 11 days, and five reviewed PRs sat 10+ days. Recommend:
- **Auto-merge for a defined safe class**: PRs with CI green, required reviews passed, no
  `needs:human-approval` label, and `area:` outside the live money path (docs, retro distillates, infra
  hardening that already went through the review gauntlet) merge automatically. This matches the charter's
  existing high-autonomy stance (model promotion already doesn't require sign-off) — PR merge for
  already-reviewed, non-money-path work is a smaller ask than that.
- **A merge-sweep scheduled task** for everything else: applies the same review-gauntlet bar the PM would,
  merges what clears it, and escalates only genuinely ambiguous cases as a single consolidated ping — not
  the current pattern of scheduled tasks quietly producing PRs that nobody looks at.
- **A staleness alarm**: any green, unmerged, non-money-path PR older than ~48h should itself generate a
  `needs:human-input`-style signal, distinct from routine standup narrative.

### 2. Rewrite liveness/health checks to assert positive state, not absence of errors
This is the pattern behind #1045, #1085, #1075, #1077, the drawdown-guard grep, and the close-only blind
spot — at least six independently-discovered instances of the same design mistake. Recommend a dedicated
audit (not yet ticketed as its own item — the individual instances are fixed, but nothing catches the *next*
one): enumerate every scheduled-task/monitor check of the form "if grep/log/table read returns nothing →
PASS" or "if `lastRunAt` within N intervals → PASS," and replace with a check that asserts the expected
artifact/state positively exists and is fresh (e.g. a monotonic counter that must have incremented, a
positive record of "N candles processed this cycle" rather than "no error seen this cycle"). Suggest filing
this as its own tracked issue rather than letting it stay implicit — happy to file it if you want it tracked.

### 3. Prioritize #944 (Railway MCP permission enforcement)
Six weeks open, unresolved, and it's the one item in this review that's a genuine unclosed *security* gap
rather than friction — the same hole that caused the July 8 unauthorized-domain incident is still only
guarded by prose. Everything else in category A is annoying; this one is a live control gap.

### 4. Make worktree/PR salvage a tight, first-class cadence, not a side effect of retro
`prune-worktrees` firing without reducing the orphan count, and abandoned work sitting 39-48 days
untouched, both suggest the pruner needs a third state beyond "delete" / "leave alone": **flag dirty +
stale-beyond-N-days worktrees for an explicit human/PM decision** rather than silently skipping them forever.
Pair with a hard rule (add to `.claude/LESSONS.md` if not already there in updated form): a distillate or
research-output PR must target `develop` directly and never rebase onto another still-open PR — this is
exactly what caused the #1026 loss and its Aug 10 repeat.

### 5. Turn "filing an issue" alarms into forced follow-through for P0/P1
#1045 and the risk-limits mismatch were each rediscovered independently ~10 times without escalating harder.
Recommend: any P0/P1 issue re-triggered by an automated check 3+ times without a fix landing should
auto-escalate as a named, singular item to the human (not buried in routine standup narrative) — the
mechanism #1079 needs for PRs, applied to issues too.

### 6. Replace silent fallbacks with fail-loud errors on the money/prediction path
The cloud-ML `price_normalization` gap (#1049) and the backtest drawdown-truncation regression (#1089,
falling back to 0.5 instead of the ratified limit) are the same shape: a missing/None value silently
substitutes a default instead of raising. Given `.claude/LESSONS.md` already documents the pre-#838
partial-exit fabrication as the canonical instance of this bug class, recommend a standing rule (if not
already codified) that any code on the prediction or risk-limit path treats a missing expected field as a
hard error, never an implicit default.

### 7. Close the loop on stale backlog items explicitly
#845 (the incident this whole window traces back to) has been functionally superseded by three shipped
fixes but is still open 28 days later. #697 is 76 days stale with a human-input request nobody's answered.
Recommend a monthly sweep (or extend an existing retro/triage skill) that explicitly closes-or-reopens any
`type:incident` whose named downstream fixes have shipped, and re-surfaces `needs:human-input` items stale
beyond ~2 weeks as a single batched ask rather than leaving them silently open.

### 8. Preserve the #986 rejection as a template
The circuit-breaker rejection is the best example in the window of a risk gate working — evidence-based,
caught a 69x-blind control before it shipped. Worth writing up as a short case study (e.g. in
`agents/lessons/risk-review/`) for what "good enough evidence" looks like, since it's the positive control
against which the #1045/#1073 near-misses can be compared.

---

## What I did not do

I did not file new GitHub issues or dispatch fix agents for any of the above — everything with an existing
tracking issue is cited by number; #2 (the liveness-check pattern) and #8 (write-up) don't have tickets yet.
Say the word and I'll file/dispatch either.
