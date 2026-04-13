variable "aws_region" {
  description = "AWS region for the Lightsail instance"
  type        = string
  default     = "eu-west-1"
}

variable "aws_profile" {
  description = "AWS credentials profile to use"
  type        = string
  default     = "default"
}

variable "instance_name" {
  description = "Name of the Lightsail instance"
  type        = string
  default     = "claude-code-agent"
}

variable "availability_zone" {
  description = "Lightsail availability zone (must be in aws_region)"
  type        = string
  default     = "eu-west-1a"
}

variable "bundle_id" {
  description = "Lightsail bundle (plan). nano_3_0 = 1GB/2vCPU/40GB SSD/$5, micro_3_0 = 2GB/2vCPU/60GB/$10"
  type        = string
  default     = "nano_3_0"
}

variable "ssh_public_key" {
  description = "SSH public key content for access to the instance"
  type        = string
}

variable "ssh_ingress_cidr" {
  description = "CIDR allowed to SSH. Set to your public IP/32 to restrict."
  type        = string
  default     = "0.0.0.0/0"
}

variable "repos_to_clone" {
  description = "List of GitHub repos to clone (format: owner/repo)"
  type        = list(string)
  default     = ["bumpy-croc/ai-trading-bot"]
}
