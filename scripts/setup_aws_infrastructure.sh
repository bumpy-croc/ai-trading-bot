#!/bin/bash
# AWS Infrastructure Setup for AI Trading Bot
# This script sets up ALL AWS infrastructure needed for both staging and production
# Run this ONCE locally to provision infrastructure, then use GitHub Actions for deployments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
AWS_REGION=$(aws configure get region 2>/dev/null || echo "us-west-2")
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")
PROJECT_NAME="ai-trading-bot"

# Parse command line arguments
ENVIRONMENTS=("staging" "production")
DRY_RUN=false
SKIP_INSTANCES=true  # Default to true for cost savings
FORCE_UPDATE=false
PARALLEL_EXECUTION=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-instances)
            SKIP_INSTANCES=true
            shift
            ;;
        --launch-instances)
            SKIP_INSTANCES=false
            shift
            ;;
        --staging-only)
            ENVIRONMENTS=("staging")
            shift
            ;;
        --production-only)
            ENVIRONMENTS=("production")
            shift
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --force-update)
            FORCE_UPDATE=true
            shift
            ;;
        --sequential)
            PARALLEL_EXECUTION=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--skip-instances|--launch-instances] [--staging-only] [--production-only] [--region REGION] [--force-update] [--sequential]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🏗️  AI Trading Bot Infrastructure Setup (Cost-Optimized)${NC}"
echo -e "${BLUE}====================================================${NC}"
echo -e "Project: ${YELLOW}${PROJECT_NAME}${NC}"
echo -e "Region: ${YELLOW}${AWS_REGION}${NC}"
echo -e "Account: ${YELLOW}${ACCOUNT_ID}${NC}"
echo -e "Environments: ${YELLOW}${ENVIRONMENTS[*]}${NC}"
echo -e "Dry Run: ${YELLOW}${DRY_RUN}${NC}"
echo -e "Parallel Execution: ${YELLOW}${PARALLEL_EXECUTION}${NC}"
if [ "$SKIP_INSTANCES" = "true" ]; then
    echo -e "${YELLOW}💰 Cost Saving: EC2 instances will NOT be launched (use --launch-instances to override)${NC}"
fi
echo

# Utility functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Enhanced AWS command wrapper with retry logic
aws_cmd() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN: aws $*"
        return 0
    fi
    
    local max_attempts=3
    local delay=2
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if aws --region "${AWS_REGION}" "$@"; then
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

# Enhanced validation with permission checks
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    if [ -z "$ACCOUNT_ID" ]; then
        log_error "Cannot determine AWS Account ID. Please configure AWS CLI."
        exit 1
    fi
    
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        log_error "AWS CLI not configured or no permissions."
        exit 1
    fi
    
    # Check for required tools
    local required_tools=("jq" "aws")
    for tool in "${required_tools[@]}"; do
        if ! command -v $tool >/dev/null 2>&1; then
            log_error "$tool is required but not installed."
            case $tool in
                jq) log_info "Install with: brew install jq (macOS) or apt-get install jq (Ubuntu)" ;;
                aws) log_info "Install AWS CLI v2 from: https://aws.amazon.com/cli/" ;;
            esac
            exit 1
        fi
    done
    
    # Validate AWS permissions
    log_info "Checking AWS permissions..."
    local required_permissions=(
        "iam:CreateRole"
        "iam:CreatePolicy"
        "iam:CreateUser"
        "s3:CreateBucket"
        "secretsmanager:CreateSecret"
        "ec2:CreateSecurityGroup"
    )
    
    # Test permissions by attempting to list resources (read-only check)
    if ! aws iam list-roles --max-items 1 >/dev/null 2>&1; then
        log_error "Missing IAM permissions. Please ensure your AWS user has administrator access or the required permissions."
        exit 1
    fi
    
    # Check AWS region validity
    if ! aws ec2 describe-regions --region-names "$AWS_REGION" >/dev/null 2>&1; then
        log_error "Invalid AWS region: $AWS_REGION"
        exit 1
    fi
    
    log_success "Prerequisites validated"
}

# Cost-optimized S3 setup with consolidated storage
setup_s3_infrastructure() {
    log_info "Setting up S3 infrastructure..."
    log_info "💰 Using consolidated S3 bucket with cost-optimized lifecycle policies"
    
    # Single consolidated bucket for all environments
    local main_bucket="${PROJECT_NAME}-storage"
    
    if aws_cmd s3api head-bucket --bucket "$main_bucket" 2>/dev/null; then
        log_warning "Storage bucket already exists: $main_bucket"
    else
        # Create bucket with encryption
        aws_cmd s3api create-bucket \
            --bucket "$main_bucket" \
            --region "$AWS_REGION" \
            $([ "$AWS_REGION" != "us-east-1" ] && echo "--create-bucket-configuration LocationConstraint=$AWS_REGION")
        
        # Enable server-side encryption
        aws_cmd s3api put-bucket-encryption \
            --bucket "$main_bucket" \
            --server-side-encryption-configuration '{
                "Rules": [{
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "AES256"
                    }
                }]
            }'
        
        # Block public access
        aws_cmd s3api put-public-access-block \
            --bucket "$main_bucket" \
            --public-access-block-configuration \
            "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
        
        log_success "Created consolidated storage bucket: $main_bucket"
    fi
    
    # Aggressive lifecycle policy for cost savings
    local lifecycle_policy='{
        "Rules": [
            {
                "ID": "DeleteOldDeployments",
                "Status": "Enabled",
                "Filter": {"Prefix": "deployments/"},
                "Expiration": {"Days": 14},
                "NoncurrentVersionExpiration": {"NoncurrentDays": 3},
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}
            },
            {
                "ID": "DeleteStagingBackups",
                "Status": "Enabled",
                "Filter": {"Prefix": "backups/staging/"},
                "Expiration": {"Days": 7},
                "NoncurrentVersionExpiration": {"NoncurrentDays": 2}
            },
            {
                "ID": "ArchiveProductionBackups",
                "Status": "Enabled",
                "Filter": {"Prefix": "backups/production/"},
                "Transitions": [
                    {
                        "Days": 30,
                        "StorageClass": "STANDARD_IA"
                    },
                    {
                        "Days": 90,
                        "StorageClass": "GLACIER"
                    }
                ],
                "Expiration": {"Days": 365}
            },
            {
                "ID": "DeleteTempFiles",
                "Status": "Enabled",
                "Filter": {"Prefix": "temp/"},
                "Expiration": {"Days": 1}
            }
        ]
    }'
    
    aws_cmd s3api put-bucket-lifecycle-configuration \
        --bucket "$main_bucket" \
        --lifecycle-configuration "$lifecycle_policy"
    
    log_success "Applied aggressive lifecycle policies for cost optimization"
    log_success "S3 infrastructure setup complete"
}

# Enhanced IAM setup with better policy management
setup_iam_infrastructure() {
    log_info "Setting up IAM infrastructure..."
    
    # Process environments in parallel if enabled
    if [ "$PARALLEL_EXECUTION" = "true" ]; then
        local pids=()
        for env in "${ENVIRONMENTS[@]}"; do
            (setup_iam_for_environment "$env") &
            pids+=($!)
        done
        
        # Wait for all environments to complete
        for pid in "${pids[@]}"; do
            wait $pid
        done
    else
        for env in "${ENVIRONMENTS[@]}"; do
            setup_iam_for_environment "$env"
        done
    fi
    
    log_success "IAM infrastructure setup complete"
}

# Helper function for IAM setup per environment
setup_iam_for_environment() {
    local env=$1
    log_info "Creating IAM resources for $env environment..."
    
    local role_name="${PROJECT_NAME}-${env}-role"
    local profile_name="${PROJECT_NAME}-${env}-profile"
    local policy_name="${PROJECT_NAME}-${env}-policy"
    
    # Create trust policy with enhanced conditions
    local trust_policy='{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {
                    "aws:RequestedRegion": "'${AWS_REGION}'",
                    "aws:SourceAccount": "'${ACCOUNT_ID}'"
                }
            }
        }]
    }'
    
    # Create IAM role
    if aws_cmd iam get-role --role-name "$role_name" >/dev/null 2>&1; then
        if [ "$FORCE_UPDATE" = "true" ]; then
            log_info "Updating trust policy for existing role: $role_name"
            aws_cmd iam update-assume-role-policy \
                --role-name "$role_name" \
                --policy-document "$trust_policy"
        else
            log_warning "Role already exists: $role_name"
        fi
    else
        aws_cmd iam create-role \
            --role-name "$role_name" \
            --assume-role-policy-document "$trust_policy" \
            --description "IAM role for AI Trading Bot $env environment" \
            --max-session-duration 3600
        log_success "Created IAM role: $role_name"
    fi
    
    # Enhanced custom policy with least privilege (consolidated S3 structure)
    local custom_policy=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SecretsManagerAccess",
            "Effect": "Allow",
            "Action": ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"],
            "Resource": "arn:aws:secretsmanager:${AWS_REGION}:${ACCOUNT_ID}:secret:${PROJECT_NAME}/${env}-*",
            "Condition": {
                "StringEquals": {"aws:RequestedRegion": "${AWS_REGION}"}
            }
        },
        {
            "Sid": "S3ConsolidatedAccess",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
            "Resource": [
                "arn:aws:s3:::${PROJECT_NAME}-storage/backups/${env}/*",
                "arn:aws:s3:::${PROJECT_NAME}-storage/deployments/*",
                "arn:aws:s3:::${PROJECT_NAME}-storage/logs/${env}/*",
                "arn:aws:s3:::${PROJECT_NAME}-storage/temp/*"
            ]
        },
        {
            "Sid": "S3ListAccess",
            "Effect": "Allow",
            "Action": ["s3:ListBucket"],
            "Resource": "arn:aws:s3:::${PROJECT_NAME}-storage",
            "Condition": {
                "StringLike": {"s3:prefix": ["backups/${env}/*", "deployments/*", "logs/${env}/*", "temp/*"]}
            }
        },
        {
            "Sid": "CloudWatchAccess",
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData"
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {"cloudwatch:namespace": "AITradingBot/${env}"}
            }
        },
        {
            "Sid": "CloudWatchLogsAccess",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams"
            ],
            "Resource": "arn:aws:logs:${AWS_REGION}:${ACCOUNT_ID}:log-group:/aws/ai-trading-bot/${env}*"
        }
    ]
}
EOF
)
    
    local policy_arn="arn:aws:iam::${ACCOUNT_ID}:policy/${policy_name}"
    
    # Create or update policy
    if aws_cmd iam get-policy --policy-arn "$policy_arn" >/dev/null 2>&1; then
        if [ "$FORCE_UPDATE" = "true" ]; then
            log_info "Creating new version of policy: $policy_name"
            # Delete old versions if we have too many
            local versions=$(aws_cmd iam list-policy-versions --policy-arn "$policy_arn" --query 'Versions[?!IsDefaultVersion].VersionId' --output text)
            local version_count=$(echo "$versions" | wc -w)
            
            if [ "$version_count" -ge 4 ]; then
                local oldest_version=$(echo "$versions" | cut -d' ' -f1)
                aws_cmd iam delete-policy-version --policy-arn "$policy_arn" --version-id "$oldest_version"
            fi
            
            aws_cmd iam create-policy-version \
                --policy-arn "$policy_arn" \
                --policy-document "$custom_policy" \
                --set-as-default
        else
            log_warning "Policy already exists: $policy_name"
        fi
    else
        aws_cmd iam create-policy \
            --policy-name "$policy_name" \
            --policy-document "$custom_policy" \
            --description "Custom policy for AI Trading Bot $env environment"
        log_success "Created IAM policy: $policy_name"
    fi
    
    # Attach policies to role
    aws_cmd iam attach-role-policy \
        --role-name "$role_name" \
        --policy-arn "$policy_arn"
    
    # Attach AWS managed policies
    local managed_policies=(
        "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
        "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    )
    
    for policy_arn in "${managed_policies[@]}"; do
        aws_cmd iam attach-role-policy \
            --role-name "$role_name" \
            --policy-arn "$policy_arn"
    done
    
    # Create instance profile
    if aws_cmd iam get-instance-profile --instance-profile-name "$profile_name" >/dev/null 2>&1; then
        log_warning "Instance profile already exists: $profile_name"
    else
        aws_cmd iam create-instance-profile --instance-profile-name "$profile_name"
        
        # Wait for instance profile to be available
        sleep 2
        
        aws_cmd iam add-role-to-instance-profile \
            --instance-profile-name "$profile_name" \
            --role-name "$role_name"
        log_success "Created instance profile: $profile_name"
    fi
}

# Enhanced GitHub Actions user setup
setup_github_actions_user() {
    log_info "Setting up GitHub Actions IAM user..."
    
    local user_name="${PROJECT_NAME}-github-actions"
    local policy_name="${PROJECT_NAME}-github-actions-policy"
    
    # Create user
    if aws_cmd iam get-user --user-name "$user_name" >/dev/null 2>&1; then
        log_warning "GitHub Actions user already exists: $user_name"
    else
        aws_cmd iam create-user \
            --user-name "$user_name" \
            --path "/service-accounts/" \
            --tags '[
                {"Key": "Purpose", "Value": "GitHubActions"},
                {"Key": "Project", "Value": "'${PROJECT_NAME}'"}
            ]'
        log_success "Created GitHub Actions user: $user_name"
    fi
    
    # Enhanced policy for GitHub Actions with minimal permissions (consolidated S3 structure)
    local github_policy=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "EC2ReadAccess",
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeInstances",
                "ec2:DescribeInstanceStatus"
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {"aws:RequestedRegion": "${AWS_REGION}"}
            }
        },
        {
            "Sid": "S3ConsolidatedDeploymentAccess",
            "Effect": "Allow",
            "Action": ["s3:PutObject", "s3:GetObject", "s3:PutObjectAcl"],
            "Resource": "arn:aws:s3:::${PROJECT_NAME}-storage/deployments/*"
        },
        {
            "Sid": "S3ListAccess",
            "Effect": "Allow",
            "Action": ["s3:ListBucket"],
            "Resource": "arn:aws:s3:::${PROJECT_NAME}-storage",
            "Condition": {
                "StringLike": {"s3:prefix": ["deployments/*"]}
            }
        },
        {
            "Sid": "SSMCommandAccess",
            "Effect": "Allow",
            "Action": [
                "ssm:SendCommand",
                "ssm:GetCommandInvocation",
                "ssm:DescribeInstanceInformation",
                "ssm:DescribeCommandExecutions",
                "ssm:ListCommandInvocations"
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {"aws:RequestedRegion": "${AWS_REGION}"}
            }
        }
    ]
}
EOF
)
    
    local policy_arn="arn:aws:iam::${ACCOUNT_ID}:policy/${policy_name}"
    
    # Create or update policy
    if aws_cmd iam get-policy --policy-arn "$policy_arn" >/dev/null 2>&1; then
        if [ "$FORCE_UPDATE" = "true" ]; then
            log_info "Updating GitHub Actions policy: $policy_name"
            aws_cmd iam create-policy-version \
                --policy-arn "$policy_arn" \
                --policy-document "$github_policy" \
                --set-as-default
        else
            log_warning "GitHub Actions policy already exists: $policy_name"
        fi
    else
        aws_cmd iam create-policy \
            --policy-name "$policy_name" \
            --policy-document "$github_policy" \
            --description "Policy for GitHub Actions CI/CD"
        log_success "Created GitHub Actions policy: $policy_name"
    fi
    
    # Attach policy to user
    aws_cmd iam attach-user-policy \
        --user-name "$user_name" \
        --policy-arn "$policy_arn"
    
    # Create access keys if they don't exist
    local existing_keys=$(aws_cmd iam list-access-keys --user-name "$user_name" --query 'AccessKeyMetadata[].AccessKeyId' --output text)
    
    if [ -z "$existing_keys" ]; then
        log_info "Creating access keys for GitHub Actions user..."
        local key_output=$(aws_cmd iam create-access-key --user-name "$user_name" --output json)
        
        if [ "$DRY_RUN" = "false" ]; then
            local access_key_id=$(echo "$key_output" | jq -r '.AccessKey.AccessKeyId')
            local secret_access_key=$(echo "$key_output" | jq -r '.AccessKey.SecretAccessKey')
            
            echo
            log_success "GitHub Actions credentials created!"
            echo -e "${YELLOW}🔐 Add these to your GitHub repository secrets:${NC}"
            echo -e "   ${BLUE}AWS_ACCESS_KEY_ID:${NC} $access_key_id"
            echo -e "   ${BLUE}AWS_SECRET_ACCESS_KEY:${NC} $secret_access_key"
            echo
            echo -e "${YELLOW}⚠️  Store these credentials securely - they won't be shown again!${NC}"
            echo
        fi
    else
        log_warning "Access keys already exist for GitHub Actions user"
        log_info "To create new keys: aws iam create-access-key --user-name $user_name"
    fi
    
    log_success "GitHub Actions user setup complete"
}

# Enhanced secrets setup with KMS encryption
setup_secrets() {
    log_info "Setting up AWS Secrets Manager secrets..."
    
    # Process secrets in parallel if enabled
    if [ "$PARALLEL_EXECUTION" = "true" ]; then
        local pids=()
        for env in "${ENVIRONMENTS[@]}"; do
            (setup_secret_for_environment "$env") &
            pids+=($!)
        done
        
        # Wait for all secrets to be created
        for pid in "${pids[@]}"; do
            wait $pid
        done
    else
        for env in "${ENVIRONMENTS[@]}"; do
            setup_secret_for_environment "$env"
        done
    fi
    
    log_success "Secrets setup complete"
}

# Helper function for secret setup per environment
setup_secret_for_environment() {
    local env=$1
    local secret_name="${PROJECT_NAME}/${env}"
    
    if aws_cmd secretsmanager describe-secret --secret-id "$secret_name" >/dev/null 2>&1; then
        log_warning "Secret already exists: $secret_name"
    else
        # Create template secret with environment-specific defaults
        local template_secret='{
            "BINANCE_API_KEY": "your-'${env}'-api-key-here",
            "BINANCE_API_SECRET": "your-'${env}'-api-secret-here",
            "DATABASE_URL": "sqlite:///data/trading_bot.db",
            "TRADING_MODE": "'$([ "$env" = "production" ] && echo "live" || echo "paper")'",
            "INITIAL_BALANCE": "'$([ "$env" = "production" ] && echo "10000" || echo "1000")'",
            "LOG_LEVEL": "'$([ "$env" = "production" ] && echo "WARNING" || echo "INFO")'",
            "MAX_POSITION_SIZE": "'$([ "$env" = "production" ] && echo "0.1" || echo "0.05")'",
            "RISK_PER_TRADE": "'$([ "$env" = "production" ] && echo "0.02" || echo "0.01")'"
        }'
        
        aws_cmd secretsmanager create-secret \
            --name "$secret_name" \
            --description "$env environment secrets for AI Trading Bot" \
            --secret-string "$template_secret" \
            --tags '[
                {"Key": "Environment", "Value": "'${env}'"},
                {"Key": "Project", "Value": "'${PROJECT_NAME}'"},
                {"Key": "ManagedBy", "Value": "Infrastructure"}
            ]'
        
        log_success "Created secret: $secret_name"
        log_warning "Remember to update secret with real API keys!"
    fi
}

# Enhanced security groups with better rules
setup_security_groups() {
    log_info "Setting up security groups..."
    
    # Get default VPC
    local vpc_id=$(aws_cmd ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text)
    
    if [ "$vpc_id" = "None" ] || [ "$vpc_id" = "null" ]; then
        log_error "No default VPC found. Please create a VPC first."
        return 1
    fi
    
    local sg_name="${PROJECT_NAME}-sg"
    local sg_id=$(aws_cmd ec2 describe-security-groups \
        --filters "Name=group-name,Values=$sg_name" "Name=vpc-id,Values=$vpc_id" \
        --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
    
    if [ "$sg_id" = "None" ] || [ "$sg_id" = "null" ]; then
        sg_id=$(aws_cmd ec2 create-security-group \
            --group-name "$sg_name" \
            --description "Security group for AI Trading Bot instances" \
            --vpc-id "$vpc_id" \
            --tag-specifications 'ResourceType=security-group,Tags=[
                {Key=Name,Value='${sg_name}'},
                {Key=Project,Value='${PROJECT_NAME}'},
                {Key=ManagedBy,Value=Infrastructure}
            ]' \
            --query 'GroupId' --output text)
        
        # Get current public IP for SSH access (more secure than 0.0.0.0/0)
        local current_ip=$(curl -s https://checkip.amazonaws.com/ || echo "0.0.0.0")
        if [ "$current_ip" != "0.0.0.0" ]; then
            aws_cmd ec2 authorize-security-group-ingress \
                --group-id "$sg_id" \
                --protocol tcp \
                --port 22 \
                --cidr "${current_ip}/32"
            log_info "SSH access restricted to current IP: ${current_ip}/32"
        else
            log_warning "Could not determine current IP, allowing SSH from anywhere"
            aws_cmd ec2 authorize-security-group-ingress \
                --group-id "$sg_id" \
                --protocol tcp \
                --port 22 \
                --cidr 0.0.0.0/0
        fi
        
        # HTTPS outbound for API calls (more specific than default)
        aws_cmd ec2 authorize-security-group-egress \
            --group-id "$sg_id" \
            --protocol tcp \
            --port 443 \
            --cidr 0.0.0.0/0
        
        # HTTP outbound for package updates
        aws_cmd ec2 authorize-security-group-egress \
            --group-id "$sg_id" \
            --protocol tcp \
            --port 80 \
            --cidr 0.0.0.0/0
        
        # DNS outbound
        aws_cmd ec2 authorize-security-group-egress \
            --group-id "$sg_id" \
            --protocol udp \
            --port 53 \
            --cidr 0.0.0.0/0
        
        log_success "Created security group: $sg_name ($sg_id)"
    else
        log_warning "Security group already exists: $sg_name ($sg_id)"
    fi
    
    echo "SECURITY_GROUP_ID=$sg_id" >> /tmp/infrastructure-outputs.txt
}

# Enhanced instance launch with better AMI selection and user data
launch_instances() {
    if [ "$SKIP_INSTANCES" = "true" ]; then
        log_info "Skipping EC2 instance launch (--skip-instances flag)"
        return 0
    fi
    
    log_info "Launching EC2 instances..."
    
    # Get latest Ubuntu 22.04 LTS AMI (more specific filter)
    local ami_id=$(aws_cmd ec2 describe-images \
        --owners 099720109477 \
        --filters \
            "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            "Name=state,Values=available" \
            "Name=architecture,Values=x86_64" \
            "Name=virtualization-type,Values=hvm" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text)
    
    if [ "$ami_id" = "None" ] || [ "$ami_id" = "null" ]; then
        log_error "Could not find suitable Ubuntu 22.04 AMI"
        return 1
    fi
    
    local sg_id=$(grep "SECURITY_GROUP_ID" /tmp/infrastructure-outputs.txt | cut -d= -f2)
    
    # Process instances in parallel if enabled
    if [ "$PARALLEL_EXECUTION" = "true" ]; then
        local pids=()
        for env in "${ENVIRONMENTS[@]}"; do
            (launch_instance_for_environment "$env" "$ami_id" "$sg_id") &
            pids+=($!)
        done
        
        # Wait for all instances to launch
        for pid in "${pids[@]}"; do
            wait $pid
        done
    else
        for env in "${ENVIRONMENTS[@]}"; do
            launch_instance_for_environment "$env" "$ami_id" "$sg_id"
        done
    fi
}

# Helper function for instance launch per environment
launch_instance_for_environment() {
    local env=$1
    local ami_id=$2
    local sg_id=$3
    
    local instance_name="${PROJECT_NAME}-${env}"
    local profile_name="${PROJECT_NAME}-${env}-profile"
    # Cost-optimized instance sizing - using t3.micro for free tier eligibility
    local instance_type="t3.micro"
    log_info "💰 Using t3.micro for cost optimization (eligible for free tier)"
    
    # Check if instance already exists
    local existing_instance=$(aws_cmd ec2 describe-instances \
        --filters "Name=tag:Name,Values=$instance_name" "Name=instance-state-name,Values=running,pending,stopping,stopped" \
        --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo "None")
    
    if [ "$existing_instance" != "None" ] && [ "$existing_instance" != "null" ]; then
        log_warning "Instance already exists for $env: $existing_instance"
        echo "${env}_INSTANCE_ID=$existing_instance" >> /tmp/infrastructure-outputs.txt
        return 0
    fi
    
    log_info "Launching $env instance..."
    
    # Create user data script for cost-optimized setup with disk validation
    local user_data=$(cat << EOF
#!/bin/bash
# Cost-optimized setup for AI Trading Bot instance with 16GB volume
set -e

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] \$1" | tee -a /var/log/ai-trading-bot-setup.log
}

log "🚀 Starting AI Trading Bot instance setup..."

# Update system
log "📦 Updating system packages..."
apt-get update
apt-get install -y awscli python3.11 python3.11-venv git htop curl wget unzip

# Install SSM agent (should be pre-installed but ensure it's running)
log "🔧 Configuring SSM agent..."
systemctl enable amazon-ssm-agent
systemctl start amazon-ssm-agent

# Validate disk space (should be 16GB)
log "💾 Validating disk space..."
TOTAL_DISK=\$(df -BG / | tail -1 | awk '{print \$2}' | sed 's/G//')
AVAILABLE_DISK=\$(df -BG / | tail -1 | awk '{print \$4}' | sed 's/G//')

log "📊 Total disk space: \${TOTAL_DISK}GB"
log "📊 Available disk space: \${AVAILABLE_DISK}GB"

if [ "\$TOTAL_DISK" -lt 15 ]; then
    log "❌ ERROR: Disk space is only \${TOTAL_DISK}GB, expected at least 15GB"
    exit 1
fi

if [ "\$AVAILABLE_DISK" -lt 10 ]; then
    log "⚠️ WARNING: Only \${AVAILABLE_DISK}GB available, this may not be sufficient"
fi

log "✅ Disk space validation passed"

# Create swap file (helpful for t3.micro instances)
log "🔄 Creating swap file..."
if [ ! -f /swapfile ]; then
    fallocate -l 1G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab
    log "✅ Swap file created (1GB)"
else
    log "ℹ️ Swap file already exists"
fi

# Create initial directory structure with proper permissions
log "📁 Creating directory structure..."
mkdir -p /opt/ai-trading-bot/{data,logs,backups}
chown ubuntu:ubuntu /opt/ai-trading-bot -R

# Set up log rotation for application logs
log "📝 Setting up log rotation..."
cat > /etc/logrotate.d/ai-trading-bot << 'LOGROTATE_EOF'
/opt/ai-trading-bot/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 ubuntu ubuntu
}
LOGROTATE_EOF

# Set up auto-shutdown for staging instances (cost saving)
if [ "$env" = "staging" ]; then
    log "💰 Setting up auto-shutdown for staging (cost optimization)..."
    # Auto-shutdown at 10 PM UTC (6 PM EST) on weekdays
    echo "0 22 * * 1-5 root /sbin/shutdown -h now" >> /etc/crontab
    log "✅ Auto-shutdown scheduled for weekdays at 10 PM UTC"
fi

# Create disk monitoring script
log "📊 Setting up disk monitoring..."
cat > /opt/ai-trading-bot/check_disk_space.sh << 'DISK_CHECK_EOF'
#!/bin/bash
# Disk space monitoring script
THRESHOLD=85
USAGE=\$(df / | tail -1 | awk '{print \$5}' | sed 's/%//')

if [ "\$USAGE" -gt "\$THRESHOLD" ]; then
    echo "WARNING: Disk usage is \${USAGE}% (threshold: \${THRESHOLD}%)"
    # Clean up old logs if disk space is low
    find /opt/ai-trading-bot/logs -name "*.log" -mtime +3 -delete
    find /opt/ai-trading-bot/data -name "*.csv.backup" -mtime +7 -delete
fi
DISK_CHECK_EOF

chmod +x /opt/ai-trading-bot/check_disk_space.sh
chown ubuntu:ubuntu /opt/ai-trading-bot/check_disk_space.sh

# Add disk check to crontab (run every hour)
echo "0 * * * * ubuntu /opt/ai-trading-bot/check_disk_space.sh" >> /etc/crontab

log "🎉 AI Trading Bot instance setup completed successfully!"
log "📊 Final disk usage: \$(df -h / | tail -1)"
EOF
)
    
    # Create block device mapping for 16GB root volume
    local block_device_mapping='[{
        "DeviceName": "/dev/sda1",
        "Ebs": {
            "VolumeSize": 16,
            "VolumeType": "gp3",
            "DeleteOnTermination": true,
            "Encrypted": true
        }
    }]'
    
    local instance_id=$(aws_cmd ec2 run-instances \
        --image-id "$ami_id" \
        --instance-type "$instance_type" \
        --security-group-ids "$sg_id" \
        --iam-instance-profile Name="$profile_name" \
        --user-data "$user_data" \
        --block-device-mappings "$block_device_mapping" \
        --tag-specifications "ResourceType=instance,Tags=[
            {Key=Name,Value=$instance_name},
            {Key=Environment,Value=$env},
            {Key=Project,Value=$PROJECT_NAME},
            {Key=ManagedBy,Value=Infrastructure},
            {Key=CostOptimized,Value=true},
            {Key=AutoShutdown,Value=$([ "$env" = "staging" ] && echo "true" || echo "false")}
        ]" \
        --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=2" \
        --monitoring Enabled=true \
        --query 'Instances[0].InstanceId' --output text)
    
    log_success "Launched $env instance: $instance_id"
    echo "${env}_INSTANCE_ID=$instance_id" >> /tmp/infrastructure-outputs.txt
    
    # Wait for instance to be running (optional)
    if [ "$env" = "production" ]; then
        log_info "Waiting for production instance to be running..."
        aws_cmd ec2 wait instance-running --instance-ids "$instance_id"
        log_success "Production instance is running: $instance_id"
    fi
}

# Main execution with better error handling
main() {
    # Set up error handling
    trap 'log_error "Script failed at line $LINENO"' ERR
    
    validate_prerequisites
    
    # Create outputs file
    local output_file="/tmp/infrastructure-outputs.txt"
    echo "# Infrastructure outputs - $(date)" > "$output_file"
    echo "AWS_REGION=$AWS_REGION" >> "$output_file"
    echo "ACCOUNT_ID=$ACCOUNT_ID" >> "$output_file"
    echo "PROJECT_NAME=$PROJECT_NAME" >> "$output_file"
    
    # Execute setup functions
    local start_time=$(date +%s)
    
    setup_s3_infrastructure
    setup_iam_infrastructure
    setup_github_actions_user
    setup_secrets
    setup_security_groups
    launch_instances
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Clean up temporary files
    rm -f /tmp/*-policy-*.json /tmp/*lifecycle*.json
    
    echo
    log_success "🎉 Infrastructure setup complete! (Duration: ${duration}s)"
    echo
    echo -e "${BLUE}📋 Summary:${NC}"
    cat "$output_file"
    echo
    echo -e "${YELLOW}📚 Next Steps:${NC}"
    echo "1. Update secrets with real API keys:"
    for env in "${ENVIRONMENTS[@]}"; do
        echo "   aws secretsmanager update-secret --secret-id ${PROJECT_NAME}/${env} --secret-string '{...}'"
    done
    echo
    echo "2. If instances were launched, bootstrap the application:"
    for env in "${ENVIRONMENTS[@]}"; do
        echo "   # SSH to ${env} instance and run:"
        echo "   curl -sSL https://raw.githubusercontent.com/alexflorisca/ai-trading-bot/main/deploy/bootstrap.sh | bash -s ${env}"
    done
    echo
    echo "3. Set up GitHub Actions with the credentials shown above"
    echo "4. Validate infrastructure: ./scripts/validate_infrastructure.sh"
    echo "5. Test deployments with: git push origin main"
    
    # Save outputs to a permanent location
    cp "$output_file" ./infrastructure-outputs.txt
    log_info "Infrastructure outputs saved to: ./infrastructure-outputs.txt"
    
    # Cost summary
    echo
    echo -e "${BLUE}💰 Cost Estimate Summary:${NC}"
    echo -e "   ${GREEN}Cost-Optimized Configuration:${NC}"
    echo "   • EC2 Instances: t3.micro (Free Tier eligible) - \$0-8/month"
    echo "   • EBS Volumes: 16GB gp3 encrypted per instance - \$1.60/month per instance"
    echo "   • S3 Storage: Single bucket with lifecycle policies - \$0.50-2/month"
    echo "   • Secrets Manager: 2 secrets - \$0.80/month"
    echo "   • CloudWatch: Basic monitoring - \$0.50/month"
    echo "   • Data Transfer: Minimal - \$0.50/month"
    echo -e "   ${GREEN}Total Estimated Cost: \$4-15/month${NC}"
    echo
    echo -e "   ${YELLOW}Cost Saving Features Enabled:${NC}"
    echo "   • t3.micro instances (Free Tier eligible for first year)"
    echo "   • 16GB gp3 volumes with encryption (optimal size for trading bot)"
    echo "   • Automated disk space monitoring and cleanup"
    echo "   • Consolidated S3 bucket (reduced storage costs)"
    echo "   • Aggressive lifecycle policies (auto-delete old files)"
    echo "   • Auto-shutdown for staging instances (evenings/weekends)"
    echo "   • Intelligent tiering for production backups"
    
    echo
    echo -e "${GREEN}✅ Infrastructure is ready for deployments!${NC}"
}

# Run main function
main "$@" 