---
name: decision-record
description: How material decisions get made and recorded — when a decision record is required, the ΔP/ΔR/C/E rubric applied inline, the log.md entry format with [D-YYYY-MM-DD-NN] ids, and the append-only correction norm. Use whenever deciding on live-capital changes, promotions, halts, experiment verdicts, proposal approve/reject, or escalations — and whenever correcting an earlier recorded claim.
---

# Decision Record

If it changed what the system does with money, models, or risk — and it isn't in log.md — it
didn't happen. log.md is layer 2: append-only, corrections as new entries
(`docs/architecture/memory_system.md`). This skill is the write-path counterpart to
`pm-session-boot` (the read path).

## When a record is REQUIRED

- Live-capital changes (sizing, strategy activation/swap, symbol add, flag flips on prod).
- Model promotions at any gate (exam winner, staging trial start/verdict, prod promote).
- Halts and containment (entry-pause, close-only, kill-switch recommendation) and their lifts.
- Experiment verdicts that change direction (GO/NO-GO, keep-incumbent, category-killed).
- Proposal approve/reject/defer; escalations to the Board; stale-item closures.
- Ratifications (`risk-ratification`), drill results (`kill-switch-drill`), retro diffs
  (`weekly-retro`), material fleet actions (`agent-fleet-health`).

Routine mechanics (a green deploy poll, a clean monitor tick) do NOT get entries — the log
stays scannable because it records decisions, not activity.

## The rubric, applied inline

For anything scoreable, show the pm.md arithmetic in the entry — bare verdicts are unauditable:

- **ΔP** capital protection (×2 weight, the veto axis) · **ΔR** profitability · **C**
  confidence · **E** effort; priority = `((ΔP×2)+ΔR)×C/E`.
- **Confidence cap:** C ≥ 3 requires a cited artifact (file:line, backtest metric, issue/PR,
  incident id) — no artifact ⇒ C ≤ 2. This rule is what kept the fabricated "+16.67%" from
  becoming a live swap: the number had no verifiable artifact behind it.
- **Hard veto:** ΔP ≤ 2 on anything touching live trading / recon / risk / margin / orders →
  reject or demand a mitigation plan; don't score around it.
- Decisions above the charter's autonomy envelope additionally record WHO approved (the human,
  in which session/PR) — e.g. the #835 entry names the human approval and its option text.

## Entry format (state README base + decision ids)

```markdown
## [D-2026-07-06-01] 2026-07-06 14:00 · decision · daemon(PM)
<one-line summary of what was decided>
Rationale: <why — rubric arithmetic if scored; the losing options and why they lost>
Ref: <issues/#N, proposals/<file>, incidents/<file>, experiment file, PR>
```

- **Id:** `D-YYYY-MM-DD-NN`, NN per-day sequence from 01, assigned at append time, never
  reused. Cite ids from issues/PRs/skills so decisions are cross-referenceable. Pre-convention
  entries are cited by their `YYYY-MM-DD HH:MM` header.
- **Kind vocabulary** (extends README): `decision`, `escalation`, `proposal-open`,
  `incident-open/close`, `post-mortem`, `note`, `track-record`, `deploy-verify`, `correction`.
- Timestamps UTC always. Newest last. Refs are load-bearing — an entry whose evidence can't be
  followed is a claim, not a record.

## Corrections — the honest-correction norm

When later evidence contradicts a recorded claim, correct the record EXPLICITLY — new entry,
own id, kind `correction`, naming what it corrects:

```markdown
## [D-2026-07-06-02] 2026-07-06 15:00 · correction · risk-officer
Corrects [D-…]/2026-07-04 13:20: <withdrawn claim> is withdrawn because <evidence>.
Still standing: <the claims that survive>. Ref: <verification artifact>
```

The two reference specimens, both in log.md:
- **Phantom-peak** (2026-07-04 13:55): the 13:20 "20.33% cap breach off a $103.82 peak" was
  withdrawn same-day after the ledger showed the peak was pinned book value — the correction
  names exactly which claims fell AND which still stand (the 365d backtest breach, the four
  control failures). That partition is the craft: a correction that nukes everything is as
  lazy as no correction.
- **WS-triage** (2026-07-06 nightcap): "Correction to earlier triage: identical -2036
  signatures predate the IP rotation" — reclassified a fresh incident as a chronic defect,
  changing the fix's urgency and owner.

Never edit the wrong entry. The record's value is that it shows reasoning being corrected —
launder it once and every past entry becomes untrustworthy.

## Red flags

- A prod flag flip, merge-to-main, or model symlink move with no `[D-…]` entry.
- "Approved by human" with no pointer to where/how the human approved.
- A verdict entry with C ≥ 3 and no artifact ref.
- Fixing a typo'd number in place "because it's just a typo" — append the correction.
