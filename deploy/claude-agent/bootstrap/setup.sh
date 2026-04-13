#!/usr/bin/env bash
# Interactive post-boot setup. Run as `ubuntu` user on the Lightsail instance.
#
# Handles:
#   1. GitHub auth via PAT
#   2. Cloning the repos listed in ~/repos.txt
#   3. Installing the systemd service that runs Claude Code
#   4. Guiding you through the Claude + Telegram interactive login
#
# Rerunnable - each step is idempotent.

set -euo pipefail

REPOS_FILE="$HOME/repos.txt"
SERVICE_SRC_DIR="$HOME/ai-trading-bot/deploy/claude-agent/service"

log() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; }

require_cloud_init() {
  if [[ ! -f "$HOME/.cloud-init-complete" ]] \
      || [[ "$(cat "$HOME/.cloud-init-complete")" != "done" ]]; then
    err "cloud-init hasn't finished yet. Wait a minute and retry."
    err "Tail progress with: sudo tail -f /var/log/cloud-init-output.log"
    exit 1
  fi
}

ensure_path() {
  # Bun installs to ~/.bun/bin; add to PATH for this script and future shells.
  export PATH="$HOME/.bun/bin:$HOME/.local/bin:$PATH"
  if ! grep -q '.bun/bin' "$HOME/.bashrc"; then
    echo 'export PATH="$HOME/.bun/bin:$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  fi
}

auth_github() {
  if gh auth status >/dev/null 2>&1; then
    log "GitHub CLI already authenticated."
    return
  fi
  log "Authenticate GitHub CLI. Paste a Personal Access Token when prompted."
  log "Required scopes: repo, read:org, workflow"
  gh auth login --hostname github.com --git-protocol https --web=false
}

clone_repos() {
  mkdir -p "$HOME/repos"
  cd "$HOME/repos"
  while IFS= read -r repo; do
    repo="$(echo "$repo" | tr -d '[:space:]')"
    [[ -z "$repo" ]] && continue
    local name="${repo##*/}"
    if [[ -d "$name/.git" ]]; then
      log "Repo already cloned: $repo"
    else
      log "Cloning $repo..."
      gh repo clone "$repo"
    fi
  done < "$REPOS_FILE"
}

install_service() {
  if [[ ! -d "$SERVICE_SRC_DIR" ]]; then
    err "Service files not found at $SERVICE_SRC_DIR"
    err "The ai-trading-bot repo must be cloned first."
    exit 1
  fi

  log "Installing start-claude.sh..."
  install -m 0755 "$SERVICE_SRC_DIR/start-claude.sh" "$HOME/start-claude.sh"

  log "Installing healthcheck.sh..."
  install -m 0755 "$SERVICE_SRC_DIR/healthcheck.sh" "$HOME/healthcheck.sh"

  log "Installing systemd unit..."
  sudo install -m 0644 "$SERVICE_SRC_DIR/claude-code.service" \
    /etc/systemd/system/claude-code.service
  sudo systemctl daemon-reload

  log "Installing logrotate config..."
  sudo install -m 0644 "$SERVICE_SRC_DIR/claude-code.logrotate" \
    /etc/logrotate.d/claude-code

  log "Installing sudoers drop-in (passwordless systemctl for healthcheck)..."
  sudo install -m 0440 -o root -g root "$SERVICE_SRC_DIR/claude-agent.sudoers" \
    /etc/sudoers.d/claude-agent

  # Health check every 5 min so a crashed service recovers on its own.
  if ! crontab -l 2>/dev/null | grep -q "healthcheck.sh"; then
    (crontab -l 2>/dev/null; echo "*/5 * * * * $HOME/healthcheck.sh") | crontab -
    log "Health check cron installed."
  fi
}

claude_first_login() {
  cat <<'EOF'

[manual step required]

Now run the Claude Code interactive login:

    claude

Follow the OAuth flow in the browser (you can complete it locally and paste
the code back). Once logged in, type /quit to exit.

Then install and configure the Telegram channel:

    claude
    > /plugin install telegram@claude-plugins-official
    > /telegram:configure <YOUR_BOT_TOKEN>
    > /quit

Finally pair your Telegram account:

    1. Send any message to your bot in Telegram
    2. Copy the pairing code it replies with
    3. Back in `claude`, run:
         /telegram:access pair <CODE>
         /telegram:access policy allowlist

Once that's done, enable and start the service:

    sudo systemctl enable claude-code
    sudo systemctl start claude-code
    sudo systemctl status claude-code

EOF
}

main() {
  require_cloud_init
  ensure_path
  auth_github
  clone_repos
  install_service
  claude_first_login
  log "Setup script complete."
}

main "$@"
