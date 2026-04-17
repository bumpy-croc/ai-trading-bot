output "public_ip" {
  value = oci_core_instance.agent.public_ip
}

output "instance_name" {
  value = oci_core_instance.agent.display_name
}

output "ssh_command" {
  value = "ssh -i ~/.ssh/ai-trading-bot-lightsail ubuntu@${oci_core_instance.agent.public_ip}"
}
