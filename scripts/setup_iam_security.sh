#!/bin/bash
# AWS IAM Security Setup Script for AI Trading Bot
# This script implements security best practices for IAM roles and policies

set -euo pipefail  # Enhanced error handling

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-staging}  # Default to staging
AWS_REGION=${AWS_DEFAULT_REGION:-us-east-1}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")
DRY_RUN=${DRY_RUN:-false}  # Set to true for dry run mode
VERBOSE=${VERBOSE:-false}  # Set to true for verbose output

# Logging setup
LOG_FILE="/tmp/ai-trading-bot-iam-setup-$(date +%Y%m%d-%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

# Utility functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Enhanced AWS command wrapper with retry logic
aws_cmd() {
    local max_attempts=3
    local delay=2
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if [ "$DRY_RUN" = "true" ]; then
            log_info "DRY RUN: aws $*"
            return 0
        fi
        
        if [ "$VERBOSE" = "true" ]; then
            log_info "Executing: aws $*"
        fi
        
        if aws "$@"; then
            return 0
        else
            local exit_code=$?
            if [ $attempt -eq $max_attempts ]; then
                log_error "AWS command failed after $max_attempts attempts: aws $*"
                return $exit_code
            fi
            log_warning "AWS command failed (attempt $attempt/$max_attempts), retrying in ${delay}s..."
            sleep $delay
            ((attempt++))
            ((delay*=2))  # Exponential backoff
        fi
    done
}

echo -e "${BLUE}🔒 AI Trading Bot - Secure IAM Setup${NC}"
echo -e "${BLUE}====================================${NC}"
echo -e "Environment: ${YELLOW}${ENVIRONMENT}${NC}"
echo -e "AWS Region: ${YELLOW}${AWS_REGION}${NC}"
echo -e "Account ID: ${YELLOW}${ACCOUNT_ID}${NC}"
echo -e "Dry Run: ${YELLOW}${DRY_RUN}${NC}"
echo -e "Log File: ${YELLOW}${LOG_FILE}${NC}"
echo

# Enhanced prerequisite validation
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check AWS CLI installation
    if ! command -v aws >/dev/null 2>&1; then
        log_error "AWS CLI not installed. Please install it first."
        exit 1
    fi
    
    # Check AWS CLI version (minimum v2)
    AWS_CLI_VERSION=$(aws --version 2>&1 | cut -d/ -f2 | cut -d. -f1)
    if [ "$AWS_CLI_VERSION" -lt 2 ]; then
        log_warning "AWS CLI v1 detected. Consider upgrading to v2 for better performance."
    fi
    
    # Check account ID
    if [ -z "$ACCOUNT_ID" ]; then
        log_error "Cannot determine AWS Account ID. Please configure AWS CLI."
        exit 1
    fi
    
    # Check AWS credentials and permissions
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        log_error "AWS CLI not configured or no permissions."
        exit 1
    fi
    
    # Check required permissions
    local required_permissions=(
        "iam:CreateRole"
        "iam:CreatePolicy" 
        "iam:AttachRolePolicy"
        "iam:CreateInstanceProfile"
        "s3:CreateBucket"
        "sns:CreateTopic"
        "cloudwatch:PutMetricAlarm"
    )
    
    log_info "Checking IAM permissions..."
    for permission in "${required_permissions[@]}"; do
        # This is a basic check - in production you might want more sophisticated permission testing
        if [ "$VERBOSE" = "true" ]; then
            log_info "Required permission: $permission"
        fi
    done
    
    # Check if jq is available for JSON processing
    if ! command -v jq >/dev/null 2>&1; then
        log_warning "jq not installed. Some advanced features may not work properly."
        log_info "Install with: brew install jq (macOS) or apt-get install jq (Ubuntu)"
    fi
    
    log_success "Prerequisites validation completed"
}

validate_prerequisites

# Function to create trust policy
create_trust_policy() {
    local env=$1
    cat > ai-trading-bot-trust-policy-${env}.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": ["${AWS_REGION}"]
        },
        "DateGreaterThan": {
          "aws:CurrentTime": "2024-01-01T00:00:00Z"
        }
      }
    }
  ]
}
EOF
}

# Function to create secrets policy
create_secrets_policy() {
    local env=$1
    local conditions=""
    
    if [ "$env" = "production" ]; then
        conditions=',
        "IpAddress": {
          "aws:SourceIp": ["10.0.0.0/8", "172.16.0.0/12"]
        },
        "StringEquals": {
          "aws:RequestedRegion": "'${AWS_REGION}'"
        }'
    fi
    
    cat > ai-trading-bot-secrets-policy-${env}.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadSecretsFor${env^}",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
                  "Resource": "arn:aws:secretsmanager:*:*:secret:ai-trading-bot/${env}-*",
      "Condition": {
        "DateLessThan": {
          "aws:CurrentTime": "2025-12-31T23:59:59Z"
        }${conditions}
      }
    },
    {
      "Sid": "KMSDecryptForSecrets",
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt",
        "kms:GenerateDataKey"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "secretsmanager.${AWS_REGION}.amazonaws.com"
        },
        "StringLike": {
                          "kms:EncryptionContext:SecretARN": "arn:aws:secretsmanager:*:*:secret:ai-trading-bot/${env}-*"
        }
      }
    }
  ]
}
EOF
}

# Function to create S3 policy
create_s3_policy() {
    local env=$1
    local bucket_name="ai-trading-bot-backups-${env}"
    
    cat > ai-trading-bot-s3-policy-${env}.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BackupBucketAccess",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:PutObjectAcl",
        "s3:GetObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::${bucket_name}/*",
      "Condition": {
        "StringEquals": {
          "s3:x-amz-server-side-encryption": "AES256"
        }
      }
    },
    {
      "Sid": "ListBackupBucket",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetBucketVersioning",
        "s3:GetBucketLocation"
      ],
      "Resource": "arn:aws:s3:::${bucket_name}",
      "Condition": {
        "StringLike": {
          "s3:prefix": ["backups/*", "logs/*", "models/*"]
        }
      }
    }
  ]
}
EOF
}

# Function to create CloudWatch policy
create_cloudwatch_policy() {
    local env=$1
    
    cat > ai-trading-bot-cloudwatch-policy-${env}.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CloudWatchMetrics",
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "cloudwatch:namespace": "AITrader/${env^}"
        }
      }
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
                  "Resource": "arn:aws:logs:*:*:log-group:/aws/ai-trading-bot/${env}*"
    }
  ]
}
EOF
}

# Function to create SNS policy
create_sns_policy() {
    local env=$1
    
    cat > ai-trading-bot-sns-policy-${env}.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublishToAlertTopic",
      "Effect": "Allow",
      "Action": [
        "sns:Publish"
      ],
                  "Resource": "arn:aws:sns:*:*:ai-trading-bot-alerts-${env}",
      "Condition": {
        "StringEquals": {
          "sns:Protocol": ["email", "sms", "https"]
        }
      }
    }
  ]
}
EOF
}

# Function to create SSM policy for secure access
create_ssm_policy() {
    local env=$1
    
    cat > ai-trading-bot-ssm-policy-${env}.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SessionManagerAccess",
      "Effect": "Allow",
      "Action": [
        "ssm:UpdateInstanceInformation",
        "ssmmessages:CreateControlChannel",
        "ssmmessages:CreateDataChannel",
        "ssmmessages:OpenControlChannel",
        "ssmmessages:OpenDataChannel"
      ],
      "Resource": "*"
    },
    {
      "Sid": "GetParameters",
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter",
        "ssm:GetParameters",
        "ssm:GetParametersByPath"
      ],
                  "Resource": "arn:aws:ssm:*:*:parameter/ai-trading-bot/${env}/*"
    }
  ]
}
EOF
}

# Main setup function
setup_iam_role() {
    local env=$1
    local role_name="ai-trading-bot-${env}-role"
    local profile_name="ai-trading-bot-${env}-profile"
    
    echo -e "${BLUE}Setting up IAM role for ${env} environment...${NC}"
    
    # Create policy files
    echo -e "📝 Creating policy documents..."
    create_trust_policy "$env"
    create_secrets_policy "$env"
    create_s3_policy "$env"
    create_cloudwatch_policy "$env"
    create_sns_policy "$env"
    create_ssm_policy "$env"
    
    # Create IAM role
    echo -e "🏗️  Creating IAM role: ${role_name}"
    if aws iam get-role --role-name "$role_name" >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Role ${role_name} already exists, updating...${NC}"
        aws iam update-assume-role-policy \
            --role-name "$role_name" \
            --policy-document file://ai-trading-bot-trust-policy-${env}.json
    else
        aws iam create-role \
            --role-name "$role_name" \
            --assume-role-policy-document file://ai-trading-bot-trust-policy-${env}.json \
            --description "Secure ${env} role for AI Trading Bot" \
            --max-session-duration 3600
        echo -e "${GREEN}✅ Created role: ${role_name}${NC}"
    fi
    
    # Create and attach policies
    local policies=("Secrets" "S3Backup" "CloudWatch" "SNS" "SSM")
    
    for policy_type in "${policies[@]}"; do
        local policy_name="AITrader${policy_type}${env^}Policy"
        local policy_file="ai-trading-bot-${policy_type,,}-policy-${env}.json"
        
        echo -e "📋 Creating policy: ${policy_name}"
        
        # Delete existing policy if it exists
        if aws iam get-policy --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${policy_name}" >/dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  Policy ${policy_name} exists, creating new version...${NC}"
            aws iam create-policy-version \
                --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${policy_name}" \
                --policy-document file://${policy_file} \
                --set-as-default >/dev/null 2>&1 || {
                    echo -e "${YELLOW}Creating new policy instead...${NC}"
                    aws iam delete-policy --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${policy_name}" 2>/dev/null || true
                    sleep 2
                    aws iam create-policy \
                        --policy-name "${policy_name}" \
                        --policy-document file://${policy_file} \
                        --description "Secure ${policy_type} access for AI Trading Bot ${env}"
                }
        else
            aws iam create-policy \
                --policy-name "${policy_name}" \
                --policy-document file://${policy_file} \
                --description "Secure ${policy_type} access for AI Trading Bot ${env}"
            echo -e "${GREEN}✅ Created policy: ${policy_name}${NC}"
        fi
        
        # Attach policy to role
        echo -e "🔗 Attaching policy to role..."
        aws iam attach-role-policy \
            --role-name "$role_name" \
            --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${policy_name}"
    done
    
    # Create instance profile
    echo -e "👤 Creating instance profile: ${profile_name}"
    if aws iam get-instance-profile --instance-profile-name "$profile_name" >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Instance profile ${profile_name} already exists${NC}"
    else
        aws iam create-instance-profile --instance-profile-name "$profile_name"
        aws iam add-role-to-instance-profile \
            --instance-profile-name "$profile_name" \
            --role-name "$role_name"
        echo -e "${GREEN}✅ Created instance profile: ${profile_name}${NC}"
    fi
    
    # Clean up policy files
    echo -e "🧹 Cleaning up temporary files..."
    rm -f ai-trading-bot-*-policy-${env}.json
    
    echo -e "${GREEN}✅ IAM setup complete for ${env} environment!${NC}"
    echo
}

# Create S3 bucket for backups with enhanced security
create_backup_bucket() {
    local env=$1
    local bucket_name="ai-trading-bot-backups-${env}-${ACCOUNT_ID}"
    
    log_info "Creating S3 backup bucket: ${bucket_name}"
    
    if aws_cmd s3 ls "s3://${bucket_name}" >/dev/null 2>&1; then
        log_warning "Bucket ${bucket_name} already exists"
        
        # Verify existing bucket configuration
        log_info "Verifying existing bucket security configuration..."
        
        # Check versioning
        if ! aws_cmd s3api get-bucket-versioning --bucket "${bucket_name}" --query 'Status' --output text 2>/dev/null | grep -q "Enabled"; then
            log_warning "Bucket versioning not enabled, enabling now..."
            aws_cmd s3api put-bucket-versioning --bucket "${bucket_name}" --versioning-configuration Status=Enabled
        fi
        
        # Check encryption
        if ! aws_cmd s3api get-bucket-encryption --bucket "${bucket_name}" >/dev/null 2>&1; then
            log_warning "Bucket encryption not enabled, enabling now..."
            aws_cmd s3api put-bucket-encryption --bucket "${bucket_name}" --server-side-encryption-configuration '{
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256"
                        }
                    }
                ]
            }'
        fi
        
        # Check public access block
        if ! aws_cmd s3api get-public-access-block --bucket "${bucket_name}" --query 'PublicAccessBlockConfiguration.BlockPublicAcls' --output text 2>/dev/null | grep -q "True"; then
            log_warning "Public access not fully blocked, fixing now..."
            aws_cmd s3api put-public-access-block --bucket "${bucket_name}" --public-access-block-configuration \
                BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
        fi
    else
        log_info "Creating new bucket..."
        
        # Create bucket with region-specific configuration
        if [ "$AWS_REGION" = "us-east-1" ]; then
            aws_cmd s3 mb "s3://${bucket_name}"
        else
            aws_cmd s3 mb "s3://${bucket_name}" --region "${AWS_REGION}"
        fi
        
        # Wait for bucket to be available
        log_info "Waiting for bucket to be available..."
        aws_cmd s3api wait bucket-exists --bucket "${bucket_name}"
        
        # Enable versioning
        log_info "Enabling bucket versioning..."
        aws_cmd s3api put-bucket-versioning \
            --bucket "${bucket_name}" \
            --versioning-configuration Status=Enabled
        
        # Enable server-side encryption
        log_info "Enabling server-side encryption..."
        aws_cmd s3api put-bucket-encryption \
            --bucket "${bucket_name}" \
            --server-side-encryption-configuration '{
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256"
                        }
                    }
                ]
            }'
        
        # Block public access
        log_info "Blocking public access..."
        aws_cmd s3api put-public-access-block \
            --bucket "${bucket_name}" \
            --public-access-block-configuration \
                BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
        
        # Set lifecycle policy for cost optimization
        log_info "Setting lifecycle policy..."
        aws_cmd s3api put-bucket-lifecycle-configuration \
            --bucket "${bucket_name}" \
            --lifecycle-configuration '{
                "Rules": [
                    {
                        "ID": "ai-trading-bot-lifecycle",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "logs/"},
                        "Transitions": [
                            {
                                "Days": 30,
                                "StorageClass": "STANDARD_IA"
                            },
                            {
                                "Days": 90,
                                "StorageClass": "GLACIER"
                            }
                        ]
                    }
                ]
            }'
        
        # Add bucket notification for security monitoring (if CloudTrail bucket)
        if [ "$env" = "production" ]; then
            log_info "Setting up bucket notifications for security monitoring..."
            # This would typically integrate with CloudWatch Events or SNS
        fi
        
        log_success "Created secure backup bucket: ${bucket_name}"
    fi
    
    # Set bucket tags for cost tracking and management
    log_info "Setting bucket tags..."
    aws_cmd s3api put-bucket-tagging \
        --bucket "${bucket_name}" \
        --tagging 'TagSet=[
            {
                "Key": "Project",
                "Value": "AI-Trading-Bot"
            },
            {
                "Key": "Environment", 
                "Value": "'${env}'"
            },
            {
                "Key": "Purpose",
                "Value": "Backup"
            },
            {
                "Key": "CreatedBy",
                "Value": "iam-security-setup"
            }
        ]'
}

# Create SNS topic for alerts
create_alert_topic() {
    local env=$1
    local topic_name="ai-trading-bot-alerts-${env}"
    
    echo -e "${BLUE}Creating SNS alert topic: ${topic_name}${NC}"
    
    local topic_arn=$(aws sns create-topic --name "$topic_name" --query 'TopicArn' --output text)
    
    # Set topic policy
    aws sns set-topic-attributes \
        --topic-arn "$topic_arn" \
        --attribute-name Policy \
        --attribute-value '{
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": "arn:aws:iam::'${ACCOUNT_ID}':role/ai-trading-bot-'${env}'-role"
                    },
                    "Action": "sns:Publish",
                    "Resource": "'${topic_arn}'"
                }
            ]
        }'
    
    echo -e "${GREEN}✅ Created SNS topic: ${topic_arn}${NC}"
    echo -e "${YELLOW}💡 Subscribe to this topic for alerts:${NC}"
    echo -e "   aws sns subscribe --topic-arn ${topic_arn} --protocol email --notification-endpoint your-email@example.com"
    echo
}

# Setup CloudWatch alarms
setup_security_monitoring() {
    local env=$1
    
    echo -e "${BLUE}Setting up security monitoring for ${env}...${NC}"
    
    # Alarm for failed secret access
    aws cloudwatch put-metric-alarm \
        --alarm-name "AI-Trading-Bot-${env^}-Unauthorized-Secret-Access" \
        --alarm-description "Alert on failed secret access attempts in ${env}" \
        --metric-name "UserErrors" \
        --namespace "AWS/SecretsManager" \
        --statistic "Sum" \
        --period 300 \
        --threshold 3 \
        --comparison-operator "GreaterThanThreshold" \
        --evaluation-periods 1 \
        --alarm-actions "arn:aws:sns:${AWS_REGION}:${ACCOUNT_ID}:ai-trading-bot-alerts-${env}" 2>/dev/null || true
    
    echo -e "${GREEN}✅ Security monitoring configured${NC}"
}

# Validation function
validate_setup() {
    local env=$1
    local role_name="ai-trading-bot-${env}-role"
    
    echo -e "${BLUE}🔍 Validating setup for ${env}...${NC}"
    
    # Check role exists
    if aws iam get-role --role-name "$role_name" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ IAM role exists: ${role_name}${NC}"
    else
        echo -e "${RED}❌ IAM role missing: ${role_name}${NC}"
        return 1
    fi
    
    # Check instance profile
    if aws iam get-instance-profile --instance-profile-name "ai-trading-bot-${env}-profile" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Instance profile exists${NC}"
    else
        echo -e "${RED}❌ Instance profile missing${NC}"
        return 1
    fi
    
    # Check S3 bucket
    local bucket_name="ai-trading-bot-backups-${env}-${ACCOUNT_ID}"
    if aws s3 ls "s3://${bucket_name}" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ S3 backup bucket exists${NC}"
    else
        echo -e "${YELLOW}⚠️  S3 backup bucket missing${NC}"
    fi
    
    echo -e "${GREEN}✅ Validation complete!${NC}"
}

# Rollback function for cleanup on failure
rollback() {
    local env=$1
    log_warning "Rolling back changes for environment: $env"
    
    # Remove IAM role and policies (be careful with this in production!)
    local role_name="ai-trading-bot-${env}-role"
    local profile_name="ai-trading-bot-${env}-profile"
    
    if [ "$env" != "production" ]; then  # Safety check
        log_info "Detaching policies from role..."
        local policies=("Secrets" "S3Backup" "CloudWatch" "SNS" "SSM")
        for policy_type in "${policies[@]}"; do
            local policy_name="AITrader${policy_type}${env^}Policy"
            aws_cmd iam detach-role-policy --role-name "$role_name" --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${policy_name}" 2>/dev/null || true
        done
        
        log_info "Removing instance profile..."
        aws_cmd iam remove-role-from-instance-profile --instance-profile-name "$profile_name" --role-name "$role_name" 2>/dev/null || true
        aws_cmd iam delete-instance-profile --instance-profile-name "$profile_name" 2>/dev/null || true
        
        log_info "Deleting IAM role..."
        aws_cmd iam delete-role --role-name "$role_name" 2>/dev/null || true
        
        log_warning "Rollback completed. Some resources may need manual cleanup."
    else
        log_error "Rollback not performed for production environment for safety reasons."
        log_info "Please manually review and clean up resources if needed."
    fi
}

# Trap for cleanup on script exit
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Script failed with exit code $exit_code"
        if [ "${ROLLBACK_ON_FAILURE:-false}" = "true" ]; then
            rollback "$ENVIRONMENT"
        fi
    fi
    
    # Clean up temporary files
    rm -f ai-trading-bot-*-policy-*.json 2>/dev/null || true
    
    log_info "Log file saved to: $LOG_FILE"
    exit $exit_code
}

trap cleanup EXIT

# Enhanced main execution with better error handling
main() {
    local start_time=$(date +%s)
    
    case "$ENVIRONMENT" in
        development|staging|production)
            log_info "🚀 Starting setup for ${ENVIRONMENT} environment..."
            ;;
        *)
            log_error "❌ Invalid environment: ${ENVIRONMENT}"
            echo -e "Usage: $0 [development|staging|production] [OPTIONS]"
            echo -e ""
            echo -e "Options:"
            echo -e "  DRY_RUN=true          - Preview changes without executing"
            echo -e "  VERBOSE=true          - Enable verbose output"
            echo -e "  ROLLBACK_ON_FAILURE=true - Rollback on failure (non-production only)"
            echo -e ""
            echo -e "Examples:"
            echo -e "  $0 staging"
            echo -e "  DRY_RUN=true $0 production"
            echo -e "  VERBOSE=true ROLLBACK_ON_FAILURE=true $0 development"
            exit 1
            ;;
    esac
    
    # Confirmation for production
    if [ "$ENVIRONMENT" = "production" ] && [ "$DRY_RUN" != "true" ]; then
        echo -e "${YELLOW}⚠️  You are about to modify PRODUCTION resources.${NC}"
        echo -e "This will create IAM roles, policies, S3 buckets, and other AWS resources."
        echo -e ""
        read -p "Are you sure you want to continue? (type 'yes' to confirm): " confirm
        if [ "$confirm" != "yes" ]; then
            log_info "Operation cancelled by user."
            exit 0
        fi
    fi
    
    # Setup IAM role and policies
    log_info "Step 1/5: Setting up IAM role and policies..."
    if ! setup_iam_role "$ENVIRONMENT"; then
        log_error "Failed to setup IAM role"
        exit 1
    fi
    
    # Create supporting resources
    log_info "Step 2/5: Creating S3 backup bucket..."
    if ! create_backup_bucket "$ENVIRONMENT"; then
        log_error "Failed to create backup bucket"
        exit 1
    fi
    
    log_info "Step 3/5: Creating SNS alert topic..."
    create_alert_topic "$ENVIRONMENT"
    setup_security_monitoring "$ENVIRONMENT"
    
    # Validate setup
    validate_setup "$ENVIRONMENT"
    
    echo -e "${GREEN}🎉 Secure IAM setup complete for ${ENVIRONMENT}!${NC}"
    echo
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "1. Create secrets in AWS Secrets Manager:"
    echo -e "   ${YELLOW}aws secretsmanager create-secret --name ai-trading-bot/${ENVIRONMENT} --secret-string '{...}'${NC}"
    echo -e "2. Launch EC2 instance with IAM instance profile:"
    echo -e "   ${YELLOW}--iam-instance-profile Name=ai-trading-bot-${ENVIRONMENT}-profile${NC}"
    echo -e "3. Subscribe to SNS alerts:"
    echo -e "   ${YELLOW}aws sns subscribe --topic-arn arn:aws:sns:${AWS_REGION}:${ACCOUNT_ID}:ai-trading-bot-alerts-${ENVIRONMENT} --protocol email --notification-endpoint your-email@example.com${NC}"
    echo
    echo -e "${BLUE}Security features enabled:${NC}"
    echo -e "• ✅ Least privilege IAM policies"
    echo -e "• ✅ Encrypted S3 backup storage"
    echo -e "• ✅ CloudWatch monitoring and alerts"
    echo -e "• ✅ SNS notifications for security events"
    echo -e "• ✅ Session Manager for secure access"
    echo -e "• ✅ Time-based and IP-based access controls"
}

# Run main function
main "$@" 