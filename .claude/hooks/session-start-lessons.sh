#!/bin/bash
# SessionStart hook — surface .claude/LESSONS.md into every session's context.
#
# Claude Code injects a SessionStart hook's stdout into the agent's context, so every
# session (local or remote, fresh / resumed / cleared / post-compact) starts already aware
# of the hard-won, incident-derived lessons for this live-capital trading bot.
#
# Single responsibility: this ONLY surfaces lessons. Environment bootstrap lives in
# session-start.sh. Keep them separate.
set -uo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
LESSONS="${PROJECT_DIR}/.claude/LESSONS.md"

# Graceful no-op if the file isn't present (e.g. a branch predating it).
[ -f "$LESSONS" ] || exit 0

cat <<'EOF'
========================================================================
📓 PROJECT LESSONS — .claude/LESSONS.md (auto-loaded every session)
Hard-won, incident-derived rules for this LIVE-CAPITAL trading bot. Read
before touching live-trading, margin, precision, prediction, reconciliation,
or deployment code, and treat each "trap → rule" as binding. When monitoring
production, §5 is the concrete list of log signatures to grep for.
========================================================================
EOF

cat "$LESSONS"
