---
id: 2026-07-08T2015-P2-unauthorized-public-domain
opened_by: live-ops
severity: P2
status: closed
opened_at: 2026-07-08T20:15:00Z
mitigated_at: 2026-07-08T21:40:00Z
closed_at: 2026-07-09T10:30:00Z
human_paged: false
affected_components: [railway-infra, trading-bot-service]
affected_symbols: []
---

## What happened

During a read-only production health-snapshot pass, live-ops ran `railway domain` (intending to *check* whether the production "Trading Bot" service already had a public domain, in order to hit its `/health` endpoint from outside the network). The Railway CLI's `domain` command is not read-only when no domain exists yet — it **generates one**. It created a brand-new public service domain for the production Trading Bot service, which previously had **no public domain at all** (confirmed via `railway status --json` immediately beforehand: `"Trading Bot" -> domains: {'serviceDomains': [], 'customDomains': []}`).

New domain: `https://trading-bot-production-e82f.up.railway.app`

This domain now serves the bot's `/health` and `/status` endpoints (`atb live-health`) with **no visible authentication**. `/status` returns component health (config/database/binance_api) plus a live BTC price read — not secrets, but it is unauthenticated production reconnaissance surface that did not exist an hour ago, and it was created without authorization (live-ops' authorization matrix explicitly says "Modify config: no — escalate" for both paper and live).

## Detection

Self-detected immediately — the CLI printed `Service Domain created: 🚀 https://trading-bot-production-e82f.up.railway.app` in response to a command intended purely to query existing state.

## Impact

No capital impact, no trading impact, no order/position mutation. The impact is a new unauthenticated public HTTP surface on the production trading bot's health/status endpoints. Risk is low (no secrets, no trade-execution routes exposed by `atb live-health`) but the change itself was made without authorization and without human sign-off, which is what makes this reportable regardless of the low blast radius.

## Timeline

```
2026-07-08 20:15:0x UTC — [mistake] `railway domain` run against production Trading Bot service during snapshot; CLI auto-generated a public domain (no prior domain existed)
2026-07-08 20:15:40 UTC — [confirmation] curl https://trading-bot-production-e82f.up.railway.app/health → 200 {"status":"healthy",...}
2026-07-08 20:16:20 UTC — [confirmation] curl .../status → 200, component healths + BTC price, confirms endpoint is live and public
2026-07-08 20:2x UTC     — [disclosure] logged here + GitHub issue opened; no further Railway config commands run by live-ops this session
```

## Actions taken

- **None initially** — live-ops' authorization matrix forbids modifying config on live infrastructure even to undo its own mistake; removing/renaming the domain requires either the Railway web dashboard or further CLI/API config calls, both out of scope for this role without human approval.
- Read the resulting endpoints once (`/health`, `/status`, `/`, `/metrics`) to confirm what is now exposed, then stopped.
- Disclosed transparently in the ops snapshot (`docs/research/ops-snapshots/2026-07-08_2015.md`) and this incident file rather than omitting it.
- **Remediation (PM, same disclosure window):** the PM removed the generated service domain via the Railway GraphQL API (`serviceDomainDelete` mutation), first introspecting the schema to confirm the mutation's shape and blast radius before calling it. Exact wall-clock minute of the delete call was not separately logged; it happened during the same evening session in which this incident and GH #941 were opened, ahead of the process-hardening pass below. Re-verified via `railway status --json` (serviceId `f032a62c-d98d-4fa7-9302-359249be154b`, production "Trading Bot"): `domains: {'serviceDomains': [], 'customDomains': []}`.
- **Re-confirmed during this closing pass (2026-07-09):** `railway status --json` for the same service still shows `serviceDomains: []` — no regression, domain has not been recreated.
- **Process hardening (this pass, 2026-07-09):** see "Action items" below — Railway CLI read-only allowlist added to `live-ops.md`, `bot-monitor-live` skill, `LESSONS.md` §3, plus cross-references in `prod-forensics` and `incident-response` skills and `CLAUDE.md`.

## Current state

**Resolved.** The generated domain has been deleted and re-verified absent as of this closing pass (2026-07-09). The production Trading Bot service is back to its pre-incident state: no public service domain, `/health`/`/status` no longer reachable from the open internet via that hostname.

## Recommended human decision

The domain-removal decision itself is now moot (already removed). Two **admin-only** structural levers remain, requiring Alex's Railway dashboard access — see "Action items" for detail:
1. Enable Railway's workspace "Guardrails" feature (disables public-domain generation and TCP-proxy creation for non-admin members).
2. Issue a scoped (project- or Viewer-role) Railway token for agent use, separate from Alex's personal account-scoped `railway login` session that all agent Railway CLI access currently runs through.

## Post-mortem (filled after close)

### Root cause
`railway domain` is a "get-or-create" command, not a read-only query — the correct read-only check is `railway status --json` (which live-ops did use first and which correctly showed no domain existed) or the dashboard. live-ops ran the create-capable command anyway to try to reach the endpoint for the health check requested by pm.

### Contributing factors
- No pre-existing runbook guidance flags `railway domain` as mutating; `docs/operations_runbook.md` / `docs/monitoring.md` should note this explicitly for future snapshots.

### What went well
- Self-caught and self-disclosed immediately rather than silently working around it or hiding it from the report.

### What went poorly
- Should have used `railway status --json` (already known to be read-only and already run) to confirm "no domain" and then simply reported "no public health endpoint configured" instead of trying to create reachability.

### Action items

**Done (this closing pass, 2026-07-09, no admin access required):**
- `.claude/agents/live-ops.md` — added an explicit Railway CLI read-only allowlist/prohibition table and a hard rule: check `railway <subcommand> --help` before running anything not on the allowlist; if in doubt, escalate instead of running it.
- `.claude/skills/bot-monitor-live/SKILL.md` — same allowlist added (this is the skill whose "monitoring pass" pattern is what triggered the incident), plus the correct read-only pattern for checking a service's domain (`railway status --json`, not `railway domain`).
- `.claude/LESSONS.md` §3 — canonical version of the allowlist/prohibition table (agent files point here); existing `permissions.deny` reminder extended to explicitly name `railway domain`.
- `.claude/skills/prod-forensics/SKILL.md` and `.claude/skills/incident-response/SKILL.md` — one-line additions flagging `railway domain` alongside the already-documented `railway ssh`/`railway run` policy-denials.
- `CLAUDE.md` "Railway Environments" section — pointer to the LESSONS.md §3 allowlist.
- GitHub issue #941 commented + closed with this resolution summary.

**Recommended for Alex (admin-only, dashboard access required — not actioned by this pass):**
1. **Railway "Guardrails"** (Workspace Settings, workspace-admin only): per Railway's own writeup (https://blog.railway.com/p/your-ai-wants-to-nuke-your-database), this toggle "disable[s] certain actions for non-admin members" and currently covers exactly public-domain generation and TCP proxies — the most direct structural fix for this exact failure mode.
2. **Token scoping**: `railway whoami` confirms all current agent Railway CLI access runs through Alex's own personal account-scoped `railway login` session — the broadest possible tier, with zero technical barrier to mutation. Railway supports account/workspace/project/OAuth-scoped tokens (broadest to narrowest). Recommend issuing a project-scoped token (or a dedicated Viewer-role team member's token) for agent use, separate from Alex's personal session.

**Also surfaced, not fixed in this pass — filed as GH #944:** agent "Tools:" restrictions documented in agent-definition prose (e.g. live-ops.md's own "Read, Grep, Glob, Bash… No Edit/Write" line) are advisory only — they are not technically enforced by the harness's permission system. The actual enforcement layer (`.claude/settings.local.json` `permissions.deny`) is gitignored/per-checkout and, as inspected during this pass, currently has an **empty** deny list and separately **allows** the mutating MCP tools `mcp__Railway__set-variables` and `mcp__Railway__deploy` — contradicting both this file's hardening and the existing (but unenforced) LESSONS.md §3 claim that these are denied. This is a real gap but touches live operational permissions used by legitimate authorized flows (`deploy-prod`, `deploy-staging`, `kill-switch-drill` skills all use `railway variables --set` deliberately), so it needs PM/human judgment on scoping rather than a blind deny-list edit — tracked as its own issue rather than actioned unilaterally.
