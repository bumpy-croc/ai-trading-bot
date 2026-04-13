#!/usr/bin/env bash
# Launches Claude Code with the Telegram channel attached.
# Runs as the `ubuntu` user via the systemd unit.

set -euo pipefail

export HOME="${HOME:-/home/ubuntu}"
export PATH="$HOME/.bun/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

cd "$HOME/repos"

# --dangerously-skip-permissions is required for unattended operation.
# Access is gated by the Telegram allowlist, not interactive prompts.
exec claude \
  --channels plugin:telegram@claude-plugins-official \
  --dangerously-skip-permissions
