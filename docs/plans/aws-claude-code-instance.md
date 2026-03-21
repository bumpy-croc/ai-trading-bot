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

## Deployment Options (Pick One)

### Option A: EC2 `t3.small` — Cheapest Always-On (~$13-17/mo)

Claude Code CLI is lightweight (Node.js process polling Telegram + calling the API).
It doesn't need much CPU/RAM — the heavy lifting happens on Anthropic's servers.

| Component | Monthly Cost |
|-----------|-------------|
| EC2 t3.small (2 vCPU, 2 GB RAM), 24/7 on-demand | ~$15 |
| 20 GB gp3 EBS | ~$1.60 |
| Data transfer | ~$1 |
| **Total (on-demand)** | **~$17/mo** |
| **Total (1yr Reserved Instance)** | **~$13/mo** |

```bash
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type t3.small \
  --key-name your-key \
  --security-group-ids sg-xxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=claude-code-agent}]'
```

**When to pick this**: You want simplicity and always-on availability. Best default choice.

### Option B: EC2 `t3.micro` — Bare Minimum (~$8-11/mo)

Even cheaper, but only 1 GB RAM. Works if you're only doing git ops / code review
and not running tests or builds on the instance itself.

| Component | Monthly Cost |
|-----------|-------------|
| EC2 t3.micro (2 vCPU, 1 GB RAM) | ~$7.50 |
| 15 GB gp3 EBS | ~$1.20 |
| Data transfer | ~$1 |
| **Total (on-demand)** | **~$10/mo** |
| **Total (1yr RI)** | **~$8/mo** |

**When to pick this**: You only need Claude to review PRs, write code, and push —
not run test suites or builds locally on the instance.

### Option C: ECS Fargate — Serverless Container (~$5-15/mo, pay-per-use)

No EC2 to manage. Run Claude Code in a container that stays up via ECS service.
Fargate pricing is per vCPU-hour + per GB-hour.

| Component | Monthly Cost (always-on) | Monthly Cost (12hr/day) |
|-----------|-------------------------|------------------------|
| Fargate 0.25 vCPU / 0.5 GB | ~$7 | ~$3.50 |
| Fargate 0.5 vCPU / 1 GB | ~$15 | ~$7.50 |
| EFS storage (optional, for repos) | ~$0.30/GB | ~$0.30/GB |
| **Total (light, always-on)** | **~$8/mo** | **~$4/mo** |

**Dockerfile**:
```dockerfile
FROM node:20-slim
RUN apt-get update && apt-get install -y git curl unzip python3 \
    && npm install -g @anthropic-ai/claude-code \
    && curl -fsSL https://bun.sh/install | bash
COPY start-claude.sh /start-claude.sh
CMD ["/start-claude.sh"]
```

**When to pick this**: You want hands-off infra management or want to shut it
down during off-hours to save money. Slightly more setup than EC2.

**Caveat**: Channels require a persistent session. If the container restarts,
messages during downtime are lost (same as EC2). ECS `desiredCount=1` with
restart policy keeps it running.

### Option D: Lightsail — Fixed Price, Simple (~$5-10/mo)

AWS Lightsail is a simpler, fixed-price alternative to EC2.

| Plan | Specs | Monthly Cost |
|------|-------|-------------|
| $5 plan | 1 vCPU, 1 GB RAM, 40 GB SSD | $5/mo |
| $10 plan | 1 vCPU, 2 GB RAM, 60 GB SSD | $10/mo |

```bash
aws lightsail create-instances \
  --instance-names claude-code-agent \
  --availability-zone us-east-1a \
  --blueprint-id ubuntu_24_04 \
  --bundle-id small_3_0
```

**When to pick this**: You want the simplest possible setup with predictable pricing.
No Reserved Instance complexity. Storage included.

---

### Recommendation

**Go with Option A (t3.small) or Option D (Lightsail $5)** depending on preference:
- Lightsail $5 if you want dead-simple and cheapest
- t3.small if you want standard EC2 tooling and may run light tests

## Implementation Steps

### 1. Provision the Instance

Pick your option above and provision it. All options need:
- **OS**: Ubuntu 24.04 LTS
- **Security Group**: Outbound-only (no inbound ports — Telegram polling is outbound)
- **Storage**: 15-20 GB is plenty for Claude Code + a few cloned repos

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

## Cost Summary

| Option | Always-On | 12hr/day | Complexity |
|--------|-----------|----------|------------|
| Lightsail $5 | **$5/mo** | N/A | Simplest |
| EC2 t3.micro | $10/mo | N/A | Simple |
| EC2 t3.small | $17/mo ($13 RI) | N/A | Simple |
| ECS Fargate (light) | $8/mo | **$4/mo** | Medium |

**Note**: These costs are for the compute infrastructure only. Claude Code API usage
(Anthropic billing) is separate and depends on how much you use it.

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
