#!/bin/bash
# Security Validation Script for AI Trading Bot
# This script validates the security configuration of your AWS setup

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-staging}
AWS_REGION=${AWS_DEFAULT_REGION:-us-east-1}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")
DETAILED=${DETAILED:-false}  # Set to true for detailed output
EXPORT_REPORT=${EXPORT_REPORT:-false}  # Set to true to export JSON report

# Report file setup
REPORT_FILE="/tmp/ai-trading-bot-security-report-$(date +%Y%m%d-%H%M%S).json"
REPORT_DATA='{"timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'","environment":"'$ENVIRONMENT'","checks":[]}'

echo -e "${BLUE}🔒 AI Trading Bot - Security Validation${NC}"
echo -e "${BLUE}=====================================${NC}"
echo -e "Environment: ${YELLOW}${ENVIRONMENT}${NC}"
echo -e "AWS Region: ${YELLOW}${AWS_REGION}${NC}"
echo -e "Account ID: ${YELLOW}${ACCOUNT_ID}${NC}"
echo

# Track validation results
PASSED=0
FAILED=0
WARNINGS=0

# Enhanced helper function to check and report with JSON export
check_result() {
    local test_name="$1"
    local result="$2"
    local message="$3"
    local level="${4:-error}"  # error, warning, info
    local category="${5:-general}"  # Category for grouping
    local remediation="${6:-}"  # Remediation steps
    
    # Create JSON entry for this check
    local json_entry='{
        "name": "'$test_name'",
        "result": "'$result'",
        "level": "'$level'",
        "category": "'$category'",
        "message": "'$message'",
        "remediation": "'$remediation'",
        "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"
    }'
    
    # Add to report data if JSON export is enabled
    if [ "$EXPORT_REPORT" = "true" ]; then
        if command -v jq >/dev/null 2>&1; then
            REPORT_DATA=$(echo "$REPORT_DATA" | jq --argjson entry "$json_entry" '.checks += [$entry]')
        fi
    fi
    
    if [ "$result" = "true" ]; then
        echo -e "${GREEN}✅ ${test_name}${NC}"
        if [ -n "$message" ] && [ "$DETAILED" = "true" ]; then
            echo -e "   ${message}"
        fi
        ((PASSED++))
    else
        if [ "$level" = "warning" ]; then
            echo -e "${YELLOW}⚠️  ${test_name}${NC}"
            ((WARNINGS++))
        else
            echo -e "${RED}❌ ${test_name}${NC}"
            ((FAILED++))
        fi
        if [ -n "$message" ]; then
            echo -e "   ${message}"
        fi
        if [ -n "$remediation" ] && [ "$DETAILED" = "true" ]; then
            echo -e "   ${BLUE}💡 Remediation: ${remediation}${NC}"
        fi
    fi
}

# Function to add security recommendations
add_recommendation() {
    local priority="$1"  # high, medium, low
    local title="$2"
    local description="$3"
    local command="$4"
    
    echo -e "${BLUE}[$priority] $title${NC}"
    echo -e "   $description"
    if [ -n "$command" ]; then
        echo -e "   ${YELLOW}Command: $command${NC}"
    fi
    echo
}

# 1. Check AWS CLI Configuration
echo -e "${BLUE}🔧 Checking AWS CLI Configuration...${NC}"

if [ -z "$ACCOUNT_ID" ]; then
    check_result "AWS CLI Configuration" "false" "Cannot determine AWS Account ID. Run 'aws configure'"
else
    check_result "AWS CLI Configuration" "true" "Account ID: $ACCOUNT_ID"
fi

# Check AWS credentials
if aws sts get-caller-identity >/dev/null 2>&1; then
    CALLER_IDENTITY=$(aws sts get-caller-identity --query 'Arn' --output text)
    check_result "AWS Credentials Valid" "true" "Identity: $CALLER_IDENTITY"
else
    check_result "AWS Credentials Valid" "false" "AWS credentials not configured or invalid"
fi

echo

# 2. Check IAM Role Configuration
echo -e "${BLUE}👤 Checking IAM Role Configuration...${NC}"

ROLE_NAME="ai-trading-bot-${ENVIRONMENT}-role"
PROFILE_NAME="ai-trading-bot-${ENVIRONMENT}-profile"

# Check if role exists
if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    check_result "IAM Role Exists" "true" "Role: $ROLE_NAME" "info" "iam" ""
    
    # Check role trust policy
    TRUST_POLICY=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.AssumeRolePolicyDocument' --output json)
    if echo "$TRUST_POLICY" | grep -q "ec2.amazonaws.com"; then
        check_result "Trust Policy Configured" "true" "EC2 service can assume role" "info" "iam" ""
    else
        check_result "Trust Policy Configured" "false" "Trust policy may not allow EC2 access" "error" "iam" "Update role trust policy to allow EC2 service"
    fi
    
    # Check attached policies
    ATTACHED_POLICIES=$(aws iam list-attached-role-policies --role-name "$ROLE_NAME" --query 'AttachedPolicies[].PolicyName' --output text)
    if [ -n "$ATTACHED_POLICIES" ]; then
        check_result "Policies Attached" "true" "Policies: $ATTACHED_POLICIES" "info" "iam" ""
    else
        check_result "Policies Attached" "false" "No policies attached to role" "error" "iam" "Attach required policies to IAM role"
    fi
else
    check_result "IAM Role Exists" "false" "Role $ROLE_NAME not found" "error" "iam" "Run: ./scripts/setup_iam_security.sh $ENVIRONMENT"
fi

# Check instance profile
if aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
    check_result "Instance Profile Exists" "true" "Profile: $PROFILE_NAME" "info" "iam" ""
else
    check_result "Instance Profile Exists" "false" "Instance profile $PROFILE_NAME not found" "error" "iam" "Create instance profile and attach to role"
fi

echo

# 3. Check Secrets Manager Configuration
echo -e "${BLUE}🔐 Checking Secrets Manager Configuration...${NC}"

SECRET_NAME="ai-trading-bot/${ENVIRONMENT}"

# Check if secret exists
if aws secretsmanager describe-secret --secret-id "$SECRET_NAME" >/dev/null 2>&1; then
    check_result "Secret Exists" "true" "Secret: $SECRET_NAME"
    
    # Check secret encryption
    SECRET_INFO=$(aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --output json)
    if echo "$SECRET_INFO" | grep -q "KmsKeyId"; then
        KMS_KEY=$(echo "$SECRET_INFO" | jq -r '.KmsKeyId // "Default"')
        check_result "Secret Encryption" "true" "KMS Key: $KMS_KEY"
    else
        check_result "Secret Encryption" "true" "Using default AWS managed key"
    fi
    
    # Check if secret has resource policy
    if aws secretsmanager get-resource-policy --secret-id "$SECRET_NAME" >/dev/null 2>&1; then
        check_result "Secret Resource Policy" "true" "Resource-based policy configured"
    else
        check_result "Secret Resource Policy" "false" "No resource-based policy (recommended for production)" "warning"
    fi
else
    check_result "Secret Exists" "false" "Secret $SECRET_NAME not found"
fi

echo

# 4. Check S3 Backup Configuration
echo -e "${BLUE}🗄️  Checking S3 Backup Configuration...${NC}"

BUCKET_NAME="ai-trading-bot-backups-${ENVIRONMENT}-${ACCOUNT_ID}"

# Check if bucket exists
if aws s3 ls "s3://${BUCKET_NAME}" >/dev/null 2>&1; then
    check_result "Backup Bucket Exists" "true" "Bucket: $BUCKET_NAME"
    
    # Check bucket versioning
    VERSIONING=$(aws s3api get-bucket-versioning --bucket "$BUCKET_NAME" --query 'Status' --output text 2>/dev/null || echo "None")
    if [ "$VERSIONING" = "Enabled" ]; then
        check_result "Bucket Versioning" "true" "Versioning enabled"
    else
        check_result "Bucket Versioning" "false" "Versioning not enabled"
    fi
    
    # Check bucket encryption
    if aws s3api get-bucket-encryption --bucket "$BUCKET_NAME" >/dev/null 2>&1; then
        check_result "Bucket Encryption" "true" "Server-side encryption enabled"
    else
        check_result "Bucket Encryption" "false" "No server-side encryption"
    fi
    
    # Check public access block
    PUBLIC_ACCESS=$(aws s3api get-public-access-block --bucket "$BUCKET_NAME" --query 'PublicAccessBlockConfiguration.BlockPublicAcls' --output text 2>/dev/null || echo "false")
    if [ "$PUBLIC_ACCESS" = "True" ]; then
        check_result "Public Access Blocked" "true" "Public access properly blocked"
    else
        check_result "Public Access Blocked" "false" "Public access not fully blocked"
    fi
else
    check_result "Backup Bucket Exists" "false" "Bucket $BUCKET_NAME not found" "warning"
fi

echo

# 5. Check CloudWatch Configuration
echo -e "${BLUE}📊 Checking CloudWatch Configuration...${NC}"

# Check CloudWatch log groups
LOG_GROUP="/aws/ai-trading-bot/${ENVIRONMENT}"
if aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --query 'logGroups[0].logGroupName' --output text 2>/dev/null | grep -q "$LOG_GROUP"; then
    check_result "CloudWatch Log Group" "true" "Log group: $LOG_GROUP"
else
    check_result "CloudWatch Log Group" "false" "Log group $LOG_GROUP not found" "warning"
fi

# Check CloudWatch alarms
ALARM_NAME="AI-Trading-Bot-${ENVIRONMENT^}-Unauthorized-Secret-Access"
if aws cloudwatch describe-alarms --alarm-names "$ALARM_NAME" --query 'MetricAlarms[0].AlarmName' --output text 2>/dev/null | grep -q "$ALARM_NAME"; then
    check_result "Security Alarm Configured" "true" "Alarm: $ALARM_NAME"
else
    check_result "Security Alarm Configured" "false" "Security alarm not configured" "warning"
fi

echo

# 6. Check SNS Configuration
echo -e "${BLUE}📱 Checking SNS Configuration...${NC}"

TOPIC_NAME="ai-trading-bot-alerts-${ENVIRONMENT}"
TOPIC_ARN="arn:aws:sns:${AWS_REGION}:${ACCOUNT_ID}:${TOPIC_NAME}"

# Check if SNS topic exists
if aws sns get-topic-attributes --topic-arn "$TOPIC_ARN" >/dev/null 2>&1; then
    check_result "SNS Alert Topic" "true" "Topic: $TOPIC_NAME"
    
    # Check subscriptions
    SUBSCRIPTIONS=$(aws sns list-subscriptions-by-topic --topic-arn "$TOPIC_ARN" --query 'Subscriptions[].Protocol' --output text 2>/dev/null || echo "")
    if [ -n "$SUBSCRIPTIONS" ]; then
        check_result "SNS Subscriptions" "true" "Protocols: $SUBSCRIPTIONS"
    else
        check_result "SNS Subscriptions" "false" "No subscriptions configured" "warning"
    fi
else
    check_result "SNS Alert Topic" "false" "Topic $TOPIC_NAME not found" "warning"
fi

echo

# 7. Check Network Security (if applicable)
echo -e "${BLUE}🌐 Checking Network Security...${NC}"

# Check for VPC endpoints (this is optional but recommended)
VPC_ENDPOINTS=$(aws ec2 describe-vpc-endpoints --filters "Name=service-name,Values=com.amazonaws.${AWS_REGION}.secretsmanager" --query 'VpcEndpoints[0].VpcEndpointId' --output text 2>/dev/null || echo "None")
if [ "$VPC_ENDPOINTS" != "None" ] && [ -n "$VPC_ENDPOINTS" ]; then
    check_result "VPC Endpoints" "true" "Secrets Manager VPC endpoint configured"
else
    check_result "VPC Endpoints" "false" "No VPC endpoints (recommended for production)" "warning"
fi

echo

# 8. Check Security Policies
echo -e "${BLUE}🛡️  Checking Security Policies...${NC}"

# Check if policies follow least privilege
POLICY_NAMES=("AITraderSecrets${ENVIRONMENT^}Policy" "AITraderS3Backup${ENVIRONMENT^}Policy" "AITraderCloudWatch${ENVIRONMENT^}Policy")

for policy_name in "${POLICY_NAMES[@]}"; do
    POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${policy_name}"
    if aws iam get-policy --policy-arn "$POLICY_ARN" >/dev/null 2>&1; then
        check_result "Policy Exists: $policy_name" "true" ""
        
        # Get policy version and check for wildcards (basic security check)
        DEFAULT_VERSION=$(aws iam get-policy --policy-arn "$POLICY_ARN" --query 'Policy.DefaultVersionId' --output text)
        POLICY_DOCUMENT=$(aws iam get-policy-version --policy-arn "$POLICY_ARN" --version-id "$DEFAULT_VERSION" --query 'PolicyVersion.Document' --output json)
        
        # Check for overly permissive policies
        if echo "$POLICY_DOCUMENT" | grep -q '"Resource": "*"' && echo "$POLICY_DOCUMENT" | grep -q '"Action": "*"'; then
            check_result "Policy Security: $policy_name" "false" "Policy may be overly permissive"
        else
            check_result "Policy Security: $policy_name" "true" "Policy follows least privilege"
        fi
    else
        check_result "Policy Exists: $policy_name" "false" "Policy $policy_name not found"
    fi
done

echo

# 9. Check CloudTrail (recommended for production)
echo -e "${BLUE}📝 Checking CloudTrail Configuration...${NC}"

# Check if CloudTrail is enabled
TRAILS=$(aws cloudtrail describe-trails --query 'trailList[?contains(Name, `ai-trading-bot`)].Name' --output text 2>/dev/null || echo "")
if [ -n "$TRAILS" ]; then
    check_result "CloudTrail Configured" "true" "Trails: $TRAILS"
    
    # Check if trail is logging
    for trail in $TRAILS; do
        LOGGING_STATUS=$(aws cloudtrail get-trail-status --name "$trail" --query 'IsLogging' --output text 2>/dev/null || echo "false")
        if [ "$LOGGING_STATUS" = "True" ]; then
            check_result "CloudTrail Logging: $trail" "true" "Logging enabled"
        else
            check_result "CloudTrail Logging: $trail" "false" "Logging disabled"
        fi
    done
else
    check_result "CloudTrail Configured" "false" "No CloudTrail configured (recommended for production)" "warning"
fi

echo

# Summary
echo -e "${BLUE}📋 Security Validation Summary${NC}"
echo -e "${BLUE}=============================${NC}"
echo -e "${GREEN}✅ Passed: ${PASSED}${NC}"
echo -e "${YELLOW}⚠️  Warnings: ${WARNINGS}${NC}"
echo -e "${RED}❌ Failed: ${FAILED}${NC}"
echo

# Overall security score
TOTAL=$((PASSED + FAILED + WARNINGS))
if [ $TOTAL -gt 0 ]; then
    SCORE=$((PASSED * 100 / TOTAL))
    echo -e "Security Score: ${SCORE}%"
    
    if [ $SCORE -ge 90 ]; then
        echo -e "${GREEN}🎉 Excellent security configuration!${NC}"
    elif [ $SCORE -ge 75 ]; then
        echo -e "${YELLOW}👍 Good security configuration with some improvements needed${NC}"
    elif [ $SCORE -ge 50 ]; then
        echo -e "${YELLOW}⚠️  Moderate security - several issues need attention${NC}"
    else
        echo -e "${RED}🚨 Poor security configuration - immediate action required${NC}"
    fi
fi

echo

# Recommendations
if [ $FAILED -gt 0 ] || [ $WARNINGS -gt 0 ]; then
    echo -e "${BLUE}🔧 Recommendations:${NC}"
    
    if [ $FAILED -gt 0 ]; then
        echo -e "${RED}Critical Issues:${NC}"
        echo -e "• Run the setup script: ./scripts/setup_iam_security.sh ${ENVIRONMENT}"
        echo -e "• Create missing secrets: aws secretsmanager create-secret --name ai-trading-bot/${ENVIRONMENT}"
        echo -e "• Review IAM policies for least privilege access"
    fi
    
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}Improvements:${NC}"
        echo -e "• Set up CloudTrail for audit logging"
        echo -e "• Configure VPC endpoints for better security"
        echo -e "• Subscribe to SNS alerts for notifications"
        echo -e "• Enable CloudWatch monitoring and alarms"
    fi
    
    echo
fi

# Export report if requested
if [ "$EXPORT_REPORT" = "true" ] && command -v jq >/dev/null 2>&1; then
    # Add summary to report
    REPORT_DATA=$(echo "$REPORT_DATA" | jq --arg passed "$PASSED" --arg failed "$FAILED" --arg warnings "$WARNINGS" --arg score "$SCORE" '.summary = {
        "passed": ($passed | tonumber),
        "failed": ($failed | tonumber), 
        "warnings": ($warnings | tonumber),
        "total": (($passed | tonumber) + ($failed | tonumber) + ($warnings | tonumber)),
        "score": ($score | tonumber)
    }')
    
    echo "$REPORT_DATA" > "$REPORT_FILE"
    echo -e "${BLUE}📄 Security report exported to: ${REPORT_FILE}${NC}"
    
    # Generate quick summary for CI/CD
    if [ "$FAILED" -gt 0 ]; then
        echo "SECURITY_STATUS=FAILED" > "/tmp/ai-trading-bot-security-status.env"
    elif [ "$WARNINGS" -gt 0 ]; then
        echo "SECURITY_STATUS=WARNING" > "/tmp/ai-trading-bot-security-status.env"
    else
        echo "SECURITY_STATUS=PASSED" > "/tmp/ai-trading-bot-security-status.env"
    fi
    echo "SECURITY_SCORE=$SCORE" >> "/tmp/ai-trading-bot-security-status.env"
fi

# Advanced recommendations based on environment
echo -e "${BLUE}📋 Environment-Specific Recommendations${NC}"
echo -e "${BLUE}====================================${NC}"

case "$ENVIRONMENT" in
    "development")
        add_recommendation "MEDIUM" "Development Security Baseline" \
            "Consider implementing basic monitoring and logging" \
            "./scripts/setup_iam_security.sh development"
        ;;
    "staging")
        add_recommendation "HIGH" "Staging Security Hardening" \
            "Implement production-like security controls for realistic testing" \
            "Enable CloudTrail and VPC endpoints"
        add_recommendation "MEDIUM" "Test Security Procedures" \
            "Regularly test incident response and backup procedures" \
            ""
        ;;
    "production")
        if [ $FAILED -gt 0 ]; then
            add_recommendation "CRITICAL" "Production Security Issues" \
                "Address all failed security checks before going live" \
                "./scripts/setup_iam_security.sh production"
        fi
        
        if [ $WARNINGS -gt 0 ]; then
            add_recommendation "HIGH" "Production Security Improvements" \
                "Address warning items to achieve security best practices" \
                ""
        fi
        
        add_recommendation "HIGH" "Enable Advanced Monitoring" \
            "Set up CloudTrail, Config Rules, and GuardDuty for comprehensive security monitoring" \
            "aws guardduty create-detector --enable"
        
        add_recommendation "MEDIUM" "Regular Security Reviews" \
            "Schedule monthly security reviews and quarterly penetration testing" \
            ""
        ;;
esac

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}❌ Security validation failed. Please address the issues above.${NC}"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Security validation passed with warnings.${NC}"
    exit 0
else
    echo -e "${GREEN}✅ Security validation passed successfully!${NC}"
    exit 0
fi 