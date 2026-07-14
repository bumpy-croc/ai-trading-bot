# Memory System

*Status: ADOPTED (2026-07-06, Board-approved alongside the operational skill suite). This is the
reference model every skill in `.claude/skills/` reads and writes against. When a skill and this
doc disagree about which layer owns a file, this doc wins and the skill gets amended.*

The daemon + agent fleet is memoryless between sessions except for what lands in files. Every
production incident this system has survived (phantom positions, phantom balances, wake-loss,
compaction) traces back to memory discipline — or its absence. Five layers, strict ownership.

## The five layers

| # | Layer | Files | Owner | Mutation rule |
|---|-------|-------|-------|---------------|
| 1 | **IDENTITY** | `.claude/state/charter.md`, `src/config/risk-limits.json` | human Board | **Read-only to all agents.** Changes only via a Board sitting (`risk-ratification` skill). Missing/TODO-laden identity blocks material decisions. |
| 2 | **RECORD** | `.claude/state/log.md`, `.claude/state/incidents/`, `.claude/state/proposals/`, `docs/research/experiments/`, `docs/research/notes/`, `docs/research/model-scoreboard.md` | writing agent | **Append-only.** Corrections are NEW entries referencing the old one — never edits. Proposal/incident lifecycle moves via `status:` frontmatter, body history stays. |
| 3 | **DISTILLATE** | `.claude/LESSONS.md`, the `.claude/skills/*/SKILL.md` files themselves (procedural memory) | daemon, via retro | **Curated and edited in place** — but routinely only by `weekly-retro` (the consolidation engine). Mid-week emergency lessons are allowed; the retro reconciles them. |
| 4 | **WORKING STATE** | scratchpad state JSONs, `.claude/state/handover.md` | owning session | **Overwritable ephemera. Never authoritative.** Always verify against ground truth (git/gh/ps/DB) before acting on it. Safe to delete once its lane completes. |
| 5 | **INDEX / RETRIEVAL** | one-line indexes (memory `MEMORY.md` index, scoreboard header rows), mem-search as the semantic layer | daemon | Kept to one line per entry; points into layers 1–3. An explicit mem-search is required before ranking work (`pm.md` gate k2) — auto-loaded context does not count. |

## Disciplines (each paid for)

1. **Append-only records, corrections as new entries.** The reference pattern is the phantom-peak
   correction: the 2026-07-04 13:20 incident-open claimed a 20.33% drawdown breach off a $103.82
   April peak; the 13:55 entry *withdrew* it ("CORRECTION to the 13:20 entry... ledger-verified")
   after the peak proved to be pinned book value, and stated exactly which claims stood and which
   fell. The wrong entry stays; the record shows the reasoning was corrected, not laundered.
   Same-day WS-triage correction (2026-07-06 nightcap, -2036 signatures predating the IP rotation)
   follows the same form. **Honest correction is a norm, not an embarrassment.**
2. **Pre-registration.** Hypothesis, metrics, thresholds, and the decision each outcome triggers
   are written to layer 2 *before* an experiment runs (`experiment-preregister` skill). Post-hoc
   threshold moves are how the fabricated "+16.67%" kelly result nearly reached live capital.
3. **One definition per process.** A durable *method* lives in exactly one skill; volatile
   *specifics* live in exactly one distillate section that the skill points at. Reference:
   `bot-monitor-live` (method) + LESSONS.md §5 (the evolving signature list). Never fork a second
   copy of either — divergent copies caused the `_extract_base_asset` USDC regression (LESSONS §2.2).
4. **Weekly retro is the consolidation engine.** Episodic memory (layer 2: what happened) is
   distilled into semantic memory (layer 3: what we now do differently) on a weekly cadence by
   `weekly-retro`. That's the only routine layer-3 write path; it keeps LESSONS.md curated instead
   of accreted.
5. **Working state is a hint, not a fact.** Handover files and scratchpad JSONs let a resumed or
   post-compaction session rebuild the picture fast — but they can be stale the moment they're
   written (the exit-sweep agent's background job *completed* while its wake-up link was lost,
   2026-07-04). Boot verifies layer 4 against the filesystem before trusting it.
6. **Identity gates everything.** If charter.md has unfilled TODOs or risk-limits.json is missing,
   the daemon refuses material decisions and pages the human (state README rule). No agent edits
   layer 1 — the daemon proposes, the human merges (`risk-ratification`).

## Decision IDs

Material decisions (defined in the `decision-record` skill) get a stable ID so issues, PRs, and
skills can cite them:

```
## [D-2026-07-06-01] 2026-07-06 14:00 · decision · daemon(PM)
```

`D-YYYY-MM-DD-NN`, NN = per-day sequence starting 01. IDs are assigned at append time and never
reused; a correcting entry gets its own ID and names the ID it corrects. Entries predating this
convention are cited by their `YYYY-MM-DD HH:MM` header.

## Which skill touches which layer

| Skill | Reads | Writes |
|---|---|---|
| `pm-session-boot` | 1, 2, 4, 5 | nothing (boot is read-only) |
| `decision-record` | 1, 2 | 2 (log.md, with decision ID) |
| `incident-response` | 1, 2 | 2 (incident file, log.md, GH issue); layer-3 lessons flow via retro |
| `prod-forensics` | 2 + prod DB (read-only) | 2 (notes/log) |
| `kill-switch-drill` | 1, 2 | 2 (drill record in log.md + issues for failures) |
| `experiment-preregister` | 2, 5 | 2 (experiments/, BEFORE the run) |
| `trade-review` | 2 + prod DB (read-only) | 2 (review note; hypotheses → preregister) |
| `capital-review` | 1, 2 | 2 (Board pack + log entry) |
| `agent-fleet-health` | 4 + ps/git/gh | 4 (state cleanup); 2 for material actions |
| `session-handover` | own session state | **4 only** (`handover.md`, overwrite-ok) |
| `delegation-protocol` | — (contract) | dispatches write 4; decisions write 2 |
| `weekly-retro` | 2, 4, 5 | **3 (the only routine editor)** + 2 (retro log entry) |
| `risk-ratification` | 1, 2 | assembles the diff; **human** writes 1; 2 (ratification log) |

## Anti-patterns

- Editing a past log.md entry "to fix a number" — that's how audit trails die. Append.
- Treating a handover.md or coordinator-relayed claim as ground truth. Verify first: a relayed
  "main checkout has ETHUSDT cache data" claim was fabricated (2026-07-05 ml-engineer log note);
  a relayed "zero callers" claim had grepped the wrong class (2026-07-04 Kelly evaluation).
- Writing lessons into a skill body when they're volatile specifics (they belong in LESSONS.md),
  or into LESSONS.md when they're a one-off episode (they belong in the log).
- A "temporary" divergence between risk-limits.json and `src/config/constants.py` — the JSON's
  own `$source_of_truth_note` declares divergence a P0 (`risk-ratification` owns the check).
