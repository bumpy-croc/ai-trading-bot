# Upload the SSH public key to Lightsail so we can SSH in post-boot.
resource "aws_lightsail_key_pair" "agent" {
  name       = "${var.instance_name}-key"
  public_key = var.ssh_public_key
}

# Render the cloud-init bootstrap script with repo list injected.
locals {
  user_data = templatefile("${path.module}/../bootstrap/cloud-init.yaml", {
    repos_to_clone = var.repos_to_clone
  })
}

resource "aws_lightsail_instance" "agent" {
  name              = var.instance_name
  availability_zone = var.availability_zone
  blueprint_id      = "ubuntu_24_04"
  bundle_id         = var.bundle_id
  key_pair_name     = aws_lightsail_key_pair.agent.name
  user_data         = local.user_data

  tags = {
    Name    = var.instance_name
    Purpose = "claude-code-agent"
    Managed = "terraform"
  }
}

# Restrict inbound to SSH only. Outbound is unrestricted by default.
# Telegram polling is outbound so no inbound ports needed for the bot itself.
resource "aws_lightsail_instance_public_ports" "agent" {
  instance_name = aws_lightsail_instance.agent.name

  port_info {
    protocol  = "tcp"
    from_port = 22
    to_port   = 22
    cidrs     = [var.ssh_ingress_cidr]
  }
}

# Static IP so the address doesn't change on reboot.
resource "aws_lightsail_static_ip" "agent" {
  name = "${var.instance_name}-ip"
}

resource "aws_lightsail_static_ip_attachment" "agent" {
  static_ip_name = aws_lightsail_static_ip.agent.name
  instance_name  = aws_lightsail_instance.agent.name
}
