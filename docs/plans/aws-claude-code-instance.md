# Plan: Always-On Claude Code Instance on AWS with Channels

## Overview

Run a persistent Claude Code session on an AWS EC2 instance that:
- Is available 24/7 via Telegram (or Discord)
- Has access to all your repos (cloned locally on the instance)
- Can respond to messages, run code, manage PRs, deploy, and monitor

## Architecture

```
You (Telegram/Discord)
        │
        ▼
  Chat Platform API
        │
        ▼
┌─────────────────────────────┐
│  AWS EC2 Instance           │
│                             │
│  Claude Code Session        │
│  --channels plugin:telegram │
│                             │
│  ~/repos/                   │
│    ├── ai-trading-bot/      │
│    ├── other-repo-1/        │
│    └── other-repo-2/        │
│                             │
│  Tools: git, gh, node, etc. │
└─────────────────────────────┘
```

## Implementation Steps

### 1. Provision EC2 Instance

**Instance type**: `t3.medium` (2 vCPU, 4 GB RAM) — sufficient for Claude Code CLI
- Upgrade to `t3.large` if running heavy builds/tests locally

**AMI**: Ubuntu 24.04 LTS

**Storage**: 50 GB gp3 EBS (enough for multiple repos + deps)

**Security Group**: Outbound-only (no inbound ports needed — Telegram polling is outbound)

```bash
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \  # Ubuntu 24.04 in your region
  --instance-type t3.medium \
  --key-name your-key \
  --security-group-ids sg-xxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=claude-code-agent}]'
```

### 2. Install Dependencies on the Instance

```bash
# System packages
sudo apt update && sudo apt install -y git curl unzip python3.11 python3.11-venv nodejs npm

# Bun (required for channel plugins)
curl -fsSL https://bun.sh/install | bash

# GitHub CLI
sudo apt install -y gh
gh auth login  # authenticate with your GitHub token

# Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Authenticate Claude Code (one-time, interactive)
claude  # follow the oauth flow to log in via claude.ai
```

### 3. Clone All Repos

```bash
mkdir -p ~/repos && cd ~/repos

# Clone repos you want Claude to access
gh repo list YOUR_USERNAME --limit 100 --json nameWithOwner -q '.[].nameWithOwner' | \
  xargs -I {} gh repo clone {}

# Or clone specific repos
gh repo clone your-org/ai-trading-bot
gh repo clone your-org/other-repo
```

### 4. Set Up Telegram Bot

1. Open Telegram, message `@BotFather`
2. `/newbot` → name it (e.g., "Claude Code Agent")
3. Copy the bot token

### 5. Configure Claude Code Channel

```bash
cd ~/repos  # working directory with access to all repos

# Install the Telegram plugin
claude
# Inside session:
/plugin install telegram@claude-plugins-official
/telegram:configure YOUR_BOT_TOKEN
# Exit and restart with channels
exit
```

### 6. Create the Startup Script

```bash
cat > ~/start-claude.sh << 'SCRIPT'
#!/bin/bash
set -e

export PATH="$HOME/.bun/bin:$HOME/.local/bin:$PATH"

cd ~/repos

# Start Claude Code with Telegram channel
exec claude \
  --channels plugin:telegram@claude-plugins-official \
  --dangerously-skip-permissions \
  2>&1 | tee -a ~/claude-session.log
SCRIPT

chmod +x ~/start-claude.sh
```

### 7. Run as a systemd Service (Always-On)

```bash
sudo tee /etc/systemd/system/claude-code.service << 'EOF'
[Unit]
Description=Claude Code with Telegram Channel
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/repos
ExecStart=/home/ubuntu/start-claude.sh
Restart=always
RestartSec=30
Environment=HOME=/home/ubuntu
Environment=PATH=/home/ubuntu/.bun/bin:/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin

# Log rotation
StandardOutput=append:/home/ubuntu/claude-session.log
StandardError=append:/home/ubuntu/claude-session.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable claude-code
sudo systemctl start claude-code
```

### 8. Pair Your Telegram Account

1. Send any message to your bot in Telegram
2. Bot replies with a pairing code
3. SSH into the instance and run:
   ```bash
   # In the Claude session (attach via journalctl or log)
   /telegram:access pair <CODE>
   /telegram:access policy allowlist
   ```
4. Now only your Telegram account can send messages

### 9. Add a CLAUDE.md to ~/repos

Create a root-level `CLAUDE.md` so Claude knows about all repos:

```markdown
# Multi-Repo Agent

You have access to the following repositories in ~/repos/:
- ai-trading-bot/ - Crypto trading system
- other-repo/ - Description

When asked to work on a specific repo, `cd` into it first.
You have full git and gh CLI access. You can create branches,
commit, push, and create PRs.

## Common Tasks
- Review PRs: `gh pr list -R owner/repo` then `gh pr view N`
- Run tests: cd into repo, use its test commands
- Deploy: follow each repo's deploy process
```

### 10. Monitoring & Maintenance

**Log rotation** (prevent disk fill):
```bash
sudo tee /etc/logrotate.d/claude-code << 'EOF'
/home/ubuntu/claude-session.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
EOF
```

**Health check script** (cron every 5 min):
```bash
cat > ~/healthcheck.sh << 'SCRIPT'
#!/bin/bash
if ! systemctl is-active --quiet claude-code; then
    echo "$(date) Claude Code is down, restarting..." >> ~/healthcheck.log
    sudo systemctl restart claude-code
fi
SCRIPT
chmod +x ~/healthcheck.sh
crontab -l | { cat; echo "*/5 * * * * ~/healthcheck.sh"; } | crontab -
```

**Keep repos fresh** (cron daily):
```bash
cat > ~/sync-repos.sh << 'SCRIPT'
#!/bin/bash
for repo in ~/repos/*/; do
    cd "$repo"
    git fetch --all --prune 2>/dev/null
done
SCRIPT
chmod +x ~/sync-repos.sh
crontab -l | { cat; echo "0 6 * * * ~/sync-repos.sh"; } | crontab -
```

## Cost Estimate

| Component | Monthly Cost |
|-----------|-------------|
| EC2 t3.medium (on-demand, 24/7) | ~$30 |
| 50 GB gp3 EBS | ~$4 |
| Data transfer (minimal) | ~$1 |
| **Total** | **~$35/mo** |

**Cost optimization**: Use a Reserved Instance or Savings Plan to drop EC2 to ~$19/mo.

## Security Considerations

1. **`--dangerously-skip-permissions`** — required for unattended use but means Claude can run any command. Mitigate by:
   - Running as a non-root user with limited sudo
   - Using IAM instance roles with minimal AWS permissions
   - Keeping the instance in a private subnet (no inbound)
   - Telegram allowlist ensures only you can send commands

2. **Secrets management**:
   - Store API keys in `~/.env` on the instance (not in repos)
   - Use AWS Secrets Manager for sensitive credentials
   - Never commit credentials to repos

3. **SSH access**: Key-pair only, no password auth

## Limitations

- **No message queuing**: If Claude Code crashes, messages sent while it's down are lost
- **Single session**: One Claude Code session at a time per login
- **Auth renewal**: claude.ai OAuth tokens may need periodic refresh (SSH in and re-auth)
- **Context window**: Long-running sessions will compress older context — send clear, self-contained messages

## Usage Examples

Once running, message your Telegram bot:

```
"Check the latest PR on ai-trading-bot and review it"
"Run the unit tests on ai-trading-bot and tell me if anything fails"
"Create a branch and fix the typo in README.md of other-repo"
"What's the current status of the trading bot deployment?"
```
