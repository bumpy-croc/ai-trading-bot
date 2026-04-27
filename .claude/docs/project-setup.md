# Agent PM Board — Setup

The Project and custom fields are created. Views are **UI-only** in Projects v2 — this doc lists what to create and how.

Project: `https://github.com/orgs/bumpy-croc/projects/6` (number 6, "Agent PM Board"). Public, linked to `bumpy-croc/ai-trading-bot` so it appears in the repo's **Projects** tab.

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

### When you switch to a bot account (appendix)

Concrete steps — do these only when the daemon is about to run unattended.

**Choose one** — default recommendation is the bot account PAT (less ceremony for a single repo):

| | GitHub App | Fine-grained PAT under a bot account |
|---|---|---|
| Setup | Create an App, install on the repo | Create a 2nd account, generate a PAT |
| UI identity | `App-name[bot]` | `bot-username` |
| Best for | Multi-repo / org-wide | Single personal repo |

**Bot account PAT — steps:**

1. Sign out, create account `<your-handle>-bot`. Verify email.
2. Add it as a collaborator on `bumpy-croc/ai-trading-bot` with **Write** access.
3. Sign in as the bot, generate a fine-grained PAT scoped to:
   - **Repository access:** only `bumpy-croc/ai-trading-bot`
   - **Repository permissions:** `Issues: R/W`, `Pull requests: R/W`, `Contents: R/W`, `Metadata: R`, `Actions: R` (optional)
   - **User permissions:** `Profile: R`
4. Add the bot account to the Project (`https://github.com/orgs/bumpy-croc/projects/6` → Settings → Manage access → invite, role **Write**). Project access is separate from repo access.
5. Store the PAT where the daemon runs: `GH_BOT_TOKEN` env var on Railway / cron / wherever.

**Authenticate the daemon's `gh`:**

```bash
GH_TOKEN=$GH_BOT_TOKEN gh auth status   # confirms it's seeing the bot identity
```

For Railway / cron envs, set `GH_TOKEN` directly — `gh` picks it up.

**Verify:**

```bash
GH_TOKEN=$GH_BOT_TOKEN gh project view 6 --owner bumpy-croc   # bot can see Project
GH_TOKEN=$GH_BOT_TOKEN gh issue list -R bumpy-croc/ai-trading-bot --limit 1
```

**Troubleshooting:**

- `gh: project not found` — bot isn't on the Project. Add via Settings → Manage access.
- `403 Resource not accessible` — PAT scopes too narrow. Re-issue with full Issues + PRs + Contents.
- Bot comments still show as you — wrong `GH_TOKEN` exported in that shell.

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
