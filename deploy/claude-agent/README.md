# Always-On Claude Code Agent on AWS Lightsail

Terraform + shell scripts that provision a $5/mo AWS Lightsail instance running
Claude Code with a Telegram channel. Once deployed, you can message the bot
from anywhere and Claude will work on `ai-trading-bot` (or any other repos you
clone) on your behalf.

See `docs/plans/aws-claude-code-instance.md` for the full design and
cost-comparison of alternative deployment options.

---

## What Gets Created

- One Lightsail instance (`nano_3_0`, Ubuntu 24.04, 1 GB RAM, 40 GB SSD) in
  `eu-west-1`
- One static public IP attached to the instance
- One Lightsail key pair using your SSH public key
- Inbound TCP/22 only (restrictable to your IP via `ssh_ingress_cidr`)

Monthly cost: **~$5** (plus static IP is free while attached).

---

## Prerequisites

You need the following installed locally:

- [Terraform](https://developer.hashicorp.com/terraform/downloads) >= 1.5
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) v2
- An SSH keypair (create with `ssh-keygen -t ed25519 -f ~/.ssh/claude-agent`)
- A Telegram bot token (from [@BotFather](https://t.me/botfather))
- A GitHub Personal Access Token with `repo, read:org, workflow` scopes
  ([create one](https://github.com/settings/tokens/new))

---

## Step 1: Configure AWS Credentials

If you've never set up AWS CLI:

```bash
# Create an IAM user in the AWS Console with these managed policies:
#   - AmazonLightsailFullAccess
#
# Then create an access key for that user and run:
aws configure
# AWS Access Key ID: AKIA...
# AWS Secret Access Key: ...
# Default region name: eu-west-1
# Default output format: json

# Verify:
aws sts get-caller-identity
```

---

## Step 2: Generate an SSH Key (if you don't have one)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/claude-agent -C "claude-agent"
```

Leave the passphrase empty so Terraform / scripts can use it non-interactively.

---

## Step 3: Configure Terraform Variables

```bash
cd deploy/claude-agent/terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
aws_region        = "eu-west-1"
aws_profile       = "default"
availability_zone = "eu-west-1a"
instance_name     = "claude-code-agent"
bundle_id         = "nano_3_0"   # $5/mo

# Paste the output of: cat ~/.ssh/claude-agent.pub
ssh_public_key = "ssh-ed25519 AAAAC3Nza... claude-agent"

# Restrict to your IP (recommended). Find yours with:
#   curl -s https://checkip.amazonaws.com
ssh_ingress_cidr = "203.0.113.42/32"

repos_to_clone = [
  "bumpy-croc/ai-trading-bot",
]
```

---

## Step 4: Provision the Instance

```bash
cd deploy/claude-agent/terraform

terraform init
terraform plan    # review what will be created
terraform apply   # confirm with "yes"
```

When it finishes, Terraform prints the public IP:

```
Outputs:
public_ip   = "52.x.x.x"
ssh_command = "ssh -i ~/.ssh/claude-agent ubuntu@52.x.x.x"
```

Cloud-init takes ~3-5 min to finish installing packages on first boot.

---

## Step 5: Finish Setup on the Instance

SSH in:

```bash
ssh -i ~/.ssh/claude-agent ubuntu@<PUBLIC_IP>
```

Wait until cloud-init is done (check with `cat ~/.cloud-init-complete` — should
print `done`; tail `/var/log/cloud-init-output.log` if it's still running).

Then run the setup script from the cloned repo. The repo isn't there yet — you
need to clone it first to get the service files:

```bash
# Authenticate GitHub CLI first
gh auth login --hostname github.com --git-protocol https --web=false
# (paste your PAT)

# Clone the ai-trading-bot repo
mkdir -p ~/repos && cd ~/repos
gh repo clone bumpy-croc/ai-trading-bot

# Run the setup script
bash ~/repos/ai-trading-bot/deploy/claude-agent/bootstrap/setup.sh
```

The script is idempotent — rerun it any time. It will:
1. Verify GitHub auth
2. Clone any remaining repos from `~/repos.txt`
3. Install `start-claude.sh`, `healthcheck.sh`, the systemd unit, logrotate
   config, and sudoers drop-in
4. Install a cron entry that restarts the service if it crashes
5. Print the next (interactive) steps

---

## Step 6: Log In to Claude + Configure Telegram

Still SSH'd in:

```bash
claude
```

Follow the OAuth flow to sign in to claude.ai. You can complete the login in
your local browser and paste the code back. Once logged in:

```text
> /plugin install telegram@claude-plugins-official
> /telegram:configure <YOUR_BOT_TOKEN>
> /quit
```

Pair your Telegram account:

1. Open Telegram, send any message to your bot
2. The bot replies with a pairing code
3. Back in `claude`, pair and lock down:

```text
> /telegram:access pair <CODE>
> /telegram:access policy allowlist
> /quit
```

---

## Step 7: Start the Service

```bash
sudo systemctl enable claude-code
sudo systemctl start claude-code
sudo systemctl status claude-code
```

Tail the log to confirm it's running:

```bash
tail -f ~/claude-session.log
```

Now send a Telegram message like "list the repos you have access to" and you
should get a reply.

---

## Operations

### Check service status
```bash
sudo systemctl status claude-code
journalctl -u claude-code -n 100
```

### Restart the service
```bash
sudo systemctl restart claude-code
```

### Update the `ai-trading-bot` repo on the instance
```bash
cd ~/repos/ai-trading-bot && git pull
```

If any service files change in this directory, rerun `setup.sh` to reinstall
them.

### Add another repo
1. Add the repo to `repos_to_clone` in `terraform.tfvars`
2. `terraform apply` is NOT required — cloud-init only runs once.
   Instead, SSH in and:
   ```bash
   echo "new-owner/new-repo" >> ~/repos.txt
   cd ~/repos && gh repo clone new-owner/new-repo
   ```

### Destroy everything
```bash
cd deploy/claude-agent/terraform
terraform destroy
```

---

## Security Notes

- **Telegram allowlist** is the primary access control. Set policy to
  `allowlist` (Step 6) or anyone who finds the bot can send it commands.
- **`--dangerously-skip-permissions`** is intentional — the agent must run
  unattended. Do not store live exchange API keys on this instance unless you
  accept that Claude could access them.
- **SSH ingress** is wide open by default (`0.0.0.0/0`). Lock it to your IP
  via `ssh_ingress_cidr` in `terraform.tfvars`.
- **GitHub PAT** is stored in `~/.config/gh/` with `0600` perms. Use a
  fine-scoped token and rotate periodically.
- **Claude OAuth tokens** may expire periodically. If the service starts
  erroring out, SSH in, run `claude` to re-authenticate, then restart the
  service.

---

## File Layout

```
deploy/claude-agent/
├── README.md                         # This file
├── terraform/
│   ├── versions.tf
│   ├── variables.tf
│   ├── main.tf                       # Lightsail instance + static IP
│   ├── outputs.tf
│   ├── terraform.tfvars.example
│   └── .gitignore
├── bootstrap/
│   ├── cloud-init.yaml               # First-boot package install
│   └── setup.sh                      # Post-boot interactive setup
└── service/
    ├── start-claude.sh               # Launched by systemd
    ├── claude-code.service           # systemd unit
    ├── claude-code.logrotate         # log rotation
    ├── claude-agent.sudoers          # NOPASSWD for healthcheck
    └── healthcheck.sh                # cron-driven service restart
```
