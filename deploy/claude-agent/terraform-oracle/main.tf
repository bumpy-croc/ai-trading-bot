locals {
  compartment_ocid = var.compartment_ocid != "" ? var.compartment_ocid : var.tenancy_ocid
}

# Pick the first availability domain in the region.
data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

# Find the latest Canonical Ubuntu 22.04 aarch64 image for this region/shape.
data "oci_core_images" "ubuntu_arm" {
  compartment_id           = local.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "22.04"
  shape                    = "VM.Standard.A1.Flex"
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

# VCN + subnet + internet gateway + route table + security list.
resource "oci_core_vcn" "agent" {
  compartment_id = local.compartment_ocid
  display_name   = "${var.instance_name}-vcn"
  cidr_blocks    = ["10.0.0.0/16"]
  dns_label      = "agentvcn"
}

resource "oci_core_internet_gateway" "agent" {
  compartment_id = local.compartment_ocid
  vcn_id         = oci_core_vcn.agent.id
  display_name   = "${var.instance_name}-igw"
  enabled        = true
}

resource "oci_core_route_table" "agent" {
  compartment_id = local.compartment_ocid
  vcn_id         = oci_core_vcn.agent.id
  display_name   = "${var.instance_name}-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.agent.id
  }
}

resource "oci_core_security_list" "agent" {
  compartment_id = local.compartment_ocid
  vcn_id         = oci_core_vcn.agent.id
  display_name   = "${var.instance_name}-sl"

  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
  }

  # SSH
  ingress_security_rules {
    protocol = "6" # TCP
    source   = var.ssh_ingress_cidr
    tcp_options {
      min = 22
      max = 22
    }
  }
}

resource "oci_core_subnet" "agent" {
  compartment_id    = local.compartment_ocid
  vcn_id            = oci_core_vcn.agent.id
  display_name      = "${var.instance_name}-subnet"
  cidr_block        = "10.0.1.0/24"
  route_table_id    = oci_core_route_table.agent.id
  security_list_ids = [oci_core_security_list.agent.id]
  dns_label         = "agent"
}

resource "oci_core_instance" "agent" {
  compartment_id      = local.compartment_ocid
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  display_name        = var.instance_name
  shape               = "VM.Standard.A1.Flex"

  shape_config {
    ocpus         = var.ocpus
    memory_in_gbs = var.memory_in_gbs
  }

  source_details {
    source_type             = "image"
    source_id               = data.oci_core_images.ubuntu_arm.images[0].id
    boot_volume_size_in_gbs = var.boot_volume_size_gb
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.agent.id
    assign_public_ip = true
    hostname_label   = "agent"
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
  }

  preserve_boot_volume = false
}
