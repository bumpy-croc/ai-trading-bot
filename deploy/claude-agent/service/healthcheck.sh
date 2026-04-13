#!/usr/bin/env bash
# Restart the claude-code service if it's not active.
# Runs every 5 minutes via cron - see bootstrap/setup.sh.

set -euo pipefail

LOG="$HOME/healthcheck.log"

if ! systemctl is-active --quiet claude-code; then
  echo "$(date -Iseconds) claude-code not active, restarting" >> "$LOG"
  sudo systemctl restart claude-code
fi
