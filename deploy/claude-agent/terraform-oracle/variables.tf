variable "tenancy_ocid" {
  type = string
}

variable "user_ocid" {
  type = string
}

variable "fingerprint" {
  type = string
}

variable "private_key_path" {
  type = string
}

variable "region" {
  type    = string
  default = "eu-stockholm-1"
}

variable "compartment_ocid" {
  type        = string
  description = "Compartment to create resources in. Defaults to tenancy (root)."
  default     = ""
}

variable "instance_name" {
  type    = string
  default = "claude-code-agent"
}

variable "ssh_public_key" {
  type = string
}

variable "ssh_ingress_cidr" {
  type    = string
  default = "0.0.0.0/0"
}

# Ampere A1 Always Free allowance: up to 4 OCPU / 24 GB across up to 4 VMs.
# Defaults below use the full allowance in a single VM.
variable "ocpus" {
  type    = number
  default = 4
}

variable "memory_in_gbs" {
  type    = number
  default = 24
}

variable "boot_volume_size_gb" {
  type        = number
  default     = 100
  description = "Free tier includes up to 200 GB total block storage."
}
