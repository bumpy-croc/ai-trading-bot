output "public_ip" {
  description = "Static public IP of the Lightsail instance"
  value       = aws_lightsail_static_ip.agent.ip_address
}

output "ssh_command" {
  description = "Command to SSH into the instance"
  value       = "ssh -i ~/.ssh/claude-agent ubuntu@${aws_lightsail_static_ip.agent.ip_address}"
}

output "instance_name" {
  description = "Lightsail instance name"
  value       = aws_lightsail_instance.agent.name
}
