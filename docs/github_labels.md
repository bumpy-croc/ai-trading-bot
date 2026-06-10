# GitHub Label Taxonomy

This repo uses a **namespaced label taxonomy**. Every open issue should carry, at
minimum, one `type:`, one `state:`, and one `priority:` label. Areas, ownership,
blockers, and origin are added as relevant.

The canonical machine-readable definition lives in
[`scripts/sync_github_labels.sh`](../scripts/sync_github_labels.sh), which is
idempotent — run it to bring the repo's label set in line with this document.

## Dimensions

### `state:*` — lifecycle (exactly one)
| Label | Meaning |
|-------|---------|
| `state:idea` | In hopper; not yet researched. |
| `state:proposed` | Proposal written; awaiting reviews / approval. |
| `state:researching` | Active investigation / backtest. |
| `state:building` | Approved; implementation in flight. |
| `state:paper` | Running in paper mode; observation window. |
| `state:monitoring` | Post-ship observation: did the KPI move? |
| `state:shipped` | Merged / deployed. |
| `state:closed` | Done (success or acknowledged failure). |
| `state:icebox` | Archived without shipping. |

### `type:*` — kind of work (exactly one)
| Label | Meaning |
|-------|---------|
| `type:feature` | Net-new capability. |
| `type:fix` | Bug fix. |
| `type:chore` | Maintenance, refactor, cleanup, deps, tech-debt. |
| `type:docs` | Documentation only. |
| `type:strategy-change` | Modify an existing strategy. |
| `type:experiment` | Run-and-measure; backtest or paper. |
| `type:research` | Output is a document, not code. |
| `type:model-promotion` | ML model lifecycle change (train/eval/promote/retire). |
| `type:infra` | Ops / deployment / CI. |
| `type:incident` | Active or recent operational issue. |
| `type:post-mortem-action` | Action item from a post-mortem. |

### `priority:*` — urgency (exactly one)
| Label | Meaning |
|-------|---------|
| `priority:p0` | Drop everything; live capital or incident. |
| `priority:p1` | Ship this week. |
| `priority:p2` | Ship when bandwidth allows. |
| `priority:p3` | Someday / nice to have. |

### `area:*` — component (zero or more)
`area:backtest`, `area:data`, `area:infra`, `area:live-ops`, `area:ml-model`,
`area:risk`, `area:sentiment`, `area:strategy`.

### `owned-by:*` — responsible agent (zero or one)
`owned-by:pm`, `owned-by:quant-researcher`, `owned-by:risk-officer`,
`owned-by:ml-engineer`, `owned-by:live-ops`, `owned-by:market-analyst`,
`owned-by:code-reviewer`, `owned-by:architecture-reviewer`, `owned-by:human`.

### `needs:*` — blocked-on (zero or more)
`needs:code-review`, `needs:risk-review`, `needs:data`, `needs:human-approval`,
`needs:human-input`.

### `source:*` — origin (zero or one)
`source:forensics`, `source:ideation`, `source:incident`, `source:market-anomaly`,
`source:parity-gap`, `source:human`, `source:automation`.

## Orthogonal flags (kept from GitHub defaults)

`good first issue`, `help wanted`, `security`, `breaking-change`, `invalid`,
`wontfix`, `question`.

## Retired labels

The following legacy/duplicate/ad-hoc labels were removed in favour of the
namespaced taxonomy. Mapping for reference:

| Retired | Replaced by |
|---------|-------------|
| `critical` | `priority:p0` |
| `high`, `high-priority` | `priority:p1` |
| `medium-priority` | `priority:p2` |
| `low`, `low-priority` | `priority:p3` |
| `bug` | `type:fix` |
| `enhancement` | `type:feature` |
| `documentation` | `type:docs` |
| `tech debt`, `code maintenance`, `cleanup`, `refactor`, `refactoring`, `deprecation` | `type:chore` |
| `backtest` | `area:backtest` |
| `data` | `area:data` |
| `ml`, `training` | `area:ml-model` |
| `technical-indicators` | `area:strategy` |
| `trading`, `trading optimisation` | `area:strategy` / `area:live-ops` |
| `margin`, `reconciliation`, `race-condition`, `thread-safety`, `robustness` | `area:live-ops` |
| `testing`, `tests` | (dropped — covered by `area:*`) |
| `automated`, `automation`, `background-agent`, `codex`, `copilot`, `autofix` | `source:automation` |
| `ai & workflow`, `workflow` | (dropped) |
