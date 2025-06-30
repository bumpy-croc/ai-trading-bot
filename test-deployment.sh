#!/bin/bash

# Manual deployment testing script
# Usage: ./test-deployment.sh <instance-id>

INSTANCE_ID="$1"

if [ -z "$INSTANCE_ID" ]; then
    echo "❌ Usage: $0 <instance-id>"
    echo "Example: $0 i-1234567890abcdef0"
    exit 1
fi

echo "🔍 Testing deployment on instance: $INSTANCE_ID"

# Step 1: Upload and run debug script
echo "📤 Step 1: Uploading debug script..."
aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[\"curl -o /tmp/debug-deployment.sh https://raw.githubusercontent.com/$(git config remote.origin.url | sed 's/.*github.com[:/]\\([^.]*\\).*/\\1/')/main/debug-deployment.sh && chmod +x /tmp/debug-deployment.sh && /tmp/debug-deployment.sh\"]" \
    --output text \
    --query 'Command.CommandId'

echo ""
echo "📋 To see the debug results, run:"
echo "aws ssm get-command-invocation --command-id <command-id-from-above> --instance-id $INSTANCE_ID --query 'StandardOutputContent' --output text"

echo ""
echo "📤 Step 2: To test the fixed deployment script:"
echo "aws ssm send-command \\"
echo "    --instance-ids \"$INSTANCE_ID\" \\"
echo "    --document-name \"AWS-RunShellScript\" \\"
echo "    --parameters \"commands=[\\\"curl -o /tmp/deploy-staging-fixed.sh https://raw.githubusercontent.com/$(git config remote.origin.url | sed 's/.*github.com[:/]\\([^.]*\\).*/\\1/')/main/bin/deploy-staging-fixed.sh && chmod +x /tmp/deploy-staging-fixed.sh && /tmp/deploy-staging-fixed.sh <commit-sha> <s3-bucket>\\\"]\" \\"
echo "    --output text \\"
echo "    --query 'Command.CommandId'"

echo ""
echo "💡 Replace <commit-sha> and <s3-bucket> with actual values"
echo "💡 You can find recent commit SHA with: git rev-parse HEAD" 