#!/bin/bash

echo "🔍 EC2 Environment Debug Script"
echo "================================="
echo "Timestamp: $(date)"
echo ""

echo "📋 System Information:"
echo "- OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "- Kernel: $(uname -r)"
echo "- Architecture: $(uname -m)"
echo "- User: $(whoami)"
echo "- Home: $HOME"
echo "- Working directory: $(pwd)"
echo ""

echo "📦 Package Manager:"
if command -v yum &> /dev/null; then
    echo "- Package manager: yum (Amazon Linux/RHEL/CentOS)"
elif command -v apt-get &> /dev/null; then
    echo "- Package manager: apt-get (Ubuntu/Debian)"
else
    echo "- Package manager: Unknown"
fi
echo ""

echo "🐍 Python Information:"
if command -v python3 &> /dev/null; then
    echo "- Python3: $(python3 --version)"
    echo "- Python3 path: $(which python3)"
else
    echo "- Python3: Not installed"
fi

if command -v pip3 &> /dev/null; then
    echo "- Pip3: $(pip3 --version)"
else
    echo "- Pip3: Not installed"
fi
echo ""

echo "☁️ AWS CLI:"
if command -v aws &> /dev/null; then
    echo "- AWS CLI: $(aws --version)"
    echo "- AWS Identity: $(aws sts get-caller-identity --query 'Arn' --output text 2>/dev/null || echo 'Not configured or no permissions')"
else
    echo "- AWS CLI: Not installed"
fi
echo ""

echo "📁 Directory Structure:"
echo "- /opt exists: $([ -d /opt ] && echo 'Yes' || echo 'No')"
echo "- /opt/ai-trading-bot exists: $([ -d /opt/ai-trading-bot ] && echo 'Yes' || echo 'No')"
if [ -d /opt/ai-trading-bot ]; then
    echo "- /opt/ai-trading-bot contents: $(ls -la /opt/ai-trading-bot 2>/dev/null | wc -l) items"
fi
echo ""

echo "🔐 Permissions:"
echo "- Can write to /tmp: $([ -w /tmp ] && echo 'Yes' || echo 'No')"
echo "- Can sudo: $(sudo -n true 2>/dev/null && echo 'Yes' || echo 'No (may require password)')"
echo ""

echo "⚙️ Services:"
if command -v systemctl &> /dev/null; then
    echo "- Systemd available: Yes"
    echo "- ai-trading-bot service exists: $(systemctl list-unit-files | grep ai-trading-bot &>/dev/null && echo 'Yes' || echo 'No')"
    if systemctl list-unit-files | grep ai-trading-bot &>/dev/null; then
        echo "- ai-trading-bot service status: $(systemctl is-active ai-trading-bot 2>/dev/null || echo 'inactive')"
    fi
else
    echo "- Systemd available: No"
fi
echo ""

echo "💾 Disk Space:"
df -h
echo ""

echo "🌐 Network:"
echo "- Can reach internet: $(ping -c 1 8.8.8.8 &>/dev/null && echo 'Yes' || echo 'No')"
echo "- Can reach S3: $(curl -s https://s3.amazonaws.com &>/dev/null && echo 'Yes' || echo 'No')"
echo ""

echo "📋 Environment Variables:"
env | grep -E "(AWS|ENVIRONMENT|PATH)" | sort
echo ""

echo "🔍 Debug script completed successfully!" 