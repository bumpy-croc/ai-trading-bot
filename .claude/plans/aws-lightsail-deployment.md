# AWS Lightsail Deployment for Claude Claw

> **Goal:** Deploy Claude Claw agent on AWS Lightsail for 24/7 operation

**Architecture:** Single Linux VM (Ubuntu 22.04) with Python, Claude CLI, and systemd for auto-start

**Tech Stack:** AWS Lightsail, Ubuntu 22.04, Python 3.11+, systemd, git

---

## Overview

Deploy Claude Claw (Claude Code agent) to AWS Lightsail for continuous 24/7 operation. The agent will have access to Python, Bash, git, GitHub APIs, and external services for constant development work.

### Recommended Plan

| Plan | Monthly | Specs | Free Tier |
|------|---------|-------|-----------|
| **Standard** | $7 | 1GB RAM, 2 vCPUs, 40GB SSD, 2TB transfer | 3 months free |

**Start with $7/month plan** - 1GB RAM is comfortable for Python agents, with 3 months free for testing.

---

## Step-by-Step Deployment Guide

### Step 1: Create AWS Account and Lightsail Instance

1. Log into [AWS Console](https://console.aws.amazon.com/)
2. Navigate to **Lightsail**
3. Click **Create instance**
4. Configure:
   - **Region:** Choose nearest to your users (us-east-1, eu-west-1, etc.)
   - **Platform:** Linux/Unix
   - **Blueprint:** Ubuntu 22.04 LTS
   - **Instance plan:** $7/month (1 GB RAM, 2 vCPUs, 40 GB SSD)
   - **Instance name:** `claude-claw-agent`
5. Click **Create instance**

### Step 2: Configure SSH Access

1. Wait for instance to be "Running"
2. Click **Connect** → **SSH** to verify access
3. **Set up SSH key locally** (recommended):

```bash
# On your local machine
ssh-keygen -t ed25519 -f ~/.ssh/lightsail_claude

# Copy the public key output
cat ~/.ssh/lightsail_claude.pub
```

4. In Lightsail console:
   - Go to **Networking** tab
   - Click **Download default key** (or upload your own)
   - Add your SSH public key to `~ubuntu/.ssh/authorized_keys`

### Step 3: Connect and Install Dependencies

```bash
# SSH into the instance
ssh -i ~/.ssh/lightsail_claude ubuntu@<your-instance-ip>

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+, git, and other essentials
sudo apt install -y python3.11 python3.11-venv python3-pip git curl wget vim

# Install Claude Code CLI
curl -fsSL https://cdn.jsdelivr.net/npm/@anthropic-ai/claude-code/install.sh | bash

# Verify installation
claude --version
```

### Step 4: Configure GitHub Access

```bash
# Configure git
git config --global user.name "Claude Claw Agent"
git config --global user.email "claude@agent"

# Set up GitHub credentials (use personal access token)
# Create token at: https://github.com/settings/tokens
gh auth login

# Or set token manually
export GITHUB_TOKEN=your_token_here
echo 'export GITHUB_TOKEN=your_token_here' >> ~/.bashrc
```

### Step 5: Set Up Workspace

```bash
# Create workspace directory
mkdir -p ~/workspace
cd ~/workspace

# Clone your repos (example)
git clone git@github.com:bumpy-croc/ai-trading-bot.git
```

### Step 6: Configure Environment Variables

```bash
# Create environment file
cat > ~/.env << EOF
# API Keys
ANTHROPIC_API_KEY=your_key_here
GITHUB_TOKEN=your_token_here

# Claude Claw Settings
CLAUDE_SESSION_PATH=~/.claude
WORKSPACE_PATH=~/workspace
EOF

# Source in shell
echo 'source ~/.env' >> ~/.bashrc
source ~/.bashrc
```

### Step 7: Create systemd Service for Auto-Start

```bash
# Create systemd service file
sudo tee /etc/systemd/system/claude-claw.service > /dev/null << EOF
[Unit]
Description=Claude Claw Agent
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/workspace
Environment="PATH=/home/ubuntu/.local/bin:/usr/bin:/bin"
EnvironmentFile=/home/ubuntu/.env
ExecStart=/home/ubuntu/.local/bin/claude start --daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable claude-claw
sudo systemctl start claude-claw

# Check status
sudo systemctl status claude-claw
```

### Step 8: Configure Monitoring and Logs

```bash
# Set up log rotation
sudo tee /etc/logrotate.d/claude-claw > /dev/null << EOF
/home/ubuntu/.claude/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
EOF

# View logs
journalctl -u claude-claw -f
```

### Step 9: Set Up Static IP (Optional but Recommended)

1. In Lightsail console, go to **Networking**
2. Create **Static IP** (free with instance)
3. Attach to your instance
4. Update DNS if needed

### Step 10: Firewall Configuration

```bash
# Allow SSH only from your IP (recommended)
sudo ufw allow from YOUR_IP_ADDRESS to any port 22
sudo ufw enable

# Or allow SSH from anywhere (less secure)
sudo ufw allow 22/tcp
sudo ufw enable
```

---

## Cost Optimization

1. **Use 3-month free tier** - New accounts get 3 months free on $7 plan
2. **Monitor usage** - Set up billing alarms
3. **Consider Spot instances** - Not available on Lightsail, but ECS Fargate Spot is cheaper if you switch later
4. **Right-size later** - Start with $7, upgrade only if needed

---

## Maintenance

### Updates

```bash
# SSH into instance
ssh ubuntu@<your-instance-ip>

# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Claude CLI
curl -fsSL https://cdn.jsdelivr.net/npm/@anthropic-ai/claude-code/install.sh | bash

# Restart service
sudo systemctl restart claude-claw
```

### Backups

```bash
# Create snapshot in Lightsail console
# Or automate via AWS CLI
aws lightsail create-instance-snapshot \
  --instance-name claude-claw-agent \
  --snapshot-name "claude-claw-backup-$(date +%Y%m%d)"
```

### Troubleshooting

```bash
# Check service status
sudo systemctl status claude-claw

# View logs
journalctl -u claude-claw -n 100

# Restart service
sudo systemctl restart claude-claw

# Check disk space
df -h

# Check memory
free -h
```

---

## Security Checklist

- [ ] SSH key authentication configured
- [ ] Firewall rules restrict SSH access
- [ ] API keys stored in environment files (not in code)
- [ ] GitHub token has minimal required permissions
- [ ] Automatic security updates enabled
- [ ] Regular snapshots configured
- [ ] Billing alerts configured

---

## Total Cost Summary

| Item | Monthly |
|------|---------|
| Instance ($7 plan) | $7 |
| Static IP | Free (included) |
| Data transfer | Free (up to 2TB) |
| **Total** | **$7/month** |

**First 3 months: Free** (new account promo)

---

## Next Steps

1. Create AWS account (if needed)
2. Launch Lightsail instance
3. Follow deployment steps above
4. Test agent with simple task
5. Configure monitoring and alerts
6. Set up backup routine

---

## Alternative: ECS Fargate (More Scalable)

If you need:
- Better cost control with Spot pricing
- Easier scaling
- More advanced networking

Consider ECS Fargate instead:
- **Setup complexity:** Higher
- **Monthly cost:** $8-15 (with Spot)
- **Scalability:** Excellent

See [ECS Deployment Plan](./ecs-fargate-deployment.md) (to be created)

---

**Document created:** 2026-03-14
**Author:** Claude Claw Agent
**Status:** Ready for implementation
