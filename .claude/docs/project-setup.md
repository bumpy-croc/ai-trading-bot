# Agent PM Board — Setup

The Project and custom fields are created. Views are **UI-only** in Projects v2 — this doc lists what to create and how.

Project: `https://github.com/users/bumpy-croc/projects/6` (number 6, "Agent PM Board").

## Views to create (UI → New view)

All views should be **Table** layout unless noted.

### 1. PM Queue (default view)
The daemon's primary view per tick.
- **Filter**: `state:researching,state:proposed,state:building,state:paper`
- **Group by**: Labels → `state`
- **Sort**: `Priority Score` descending, then `Priority` ascending
- Show fields: Title, Labels, Priority Score, Wake At, Owner

### 2. Awaiting You (Board layout)
What the human needs to action.
- **Filter**: `needs:human-approval OR needs:human-input`
- **Sort**: Last updated ascending (oldest first)
- Show fields: Title, Labels, Priority Score, updatedAt

### 3. In Paper
Observation windows in progress.
- **Filter**: `state:paper`
- **Sort**: `Wake At` ascending
- Show fields: Title, Wake At, Priority Score

### 4. Monitoring
Post-ship: did it actually help?
- **Filter**: `state:monitoring` updated in last 30 days
- **Sort**: updatedAt descending

### 5. Incidents (Board layout)
- **Filter**: `type:incident AND -state:closed`
- **Group by**: `Priority`

### 6. Idea Hopper
`/ideation` writes here.
- **Filter**: `state:idea`
- **Sort**: `Priority Score` descending

### 7. By Area — Strategy (clone per area as needed)
- **Filter**: `area:strategy AND -state:closed AND -state:icebox`
- **Group by**: `state`

## Bot identity — unresolved

v1 uses your PAT (simpler). If the daemon is ever hosted on Railway / elsewhere and runs unattended, switch to a **separate GitHub App or fine-grained PAT** so daemon comments / labels / PR actions are visually distinct from yours. Decision deferred until the daemon is actually running autonomously.

## Labels

All 50 labels are created in the repo. Source-of-truth list lives in `.claude/scripts/labels.tsv` (tab-separated: `name<TAB>color<TAB>description`). To re-sync after edits:

```bash
while IFS=$'\t' read -r name color desc; do
  gh label create "$name" --color "$color" --description "$desc" --force
done < .claude/scripts/labels.tsv
```

## Linking issues to the project

Issues in `bumpy-croc/ai-trading-bot` are not auto-added. Options:

1. Manual per issue: Project → Add item → search issue.
2. Project workflow: Project → Workflows → "Auto-add" → filter `repo:bumpy-croc/ai-trading-bot is:issue`. Recommended.
3. Per issue via CLI:
   ```bash
   gh project item-add 6 --owner bumpy-croc --url https://github.com/bumpy-croc/ai-trading-bot/issues/<N>
   ```

## Known residual work

- **Agent-team auth** — decide bot account vs PAT when the daemon goes autonomous.
- **Loss forensics automation** — the `source:forensics` label needs a triggering path (drawdown breach in `/heartbeat` → open issue). Wire up when the daemon starts running.
- **Weekly sweep** — a scheduled command that fills `outcome` for `· track-record · <agent>` entries whose horizon has elapsed. Defer until enough calls exist to grade.
- **PR ↔ issue auto-linking** — conventional `Fixes #N` in PR body is enough for now; formal automation later.
