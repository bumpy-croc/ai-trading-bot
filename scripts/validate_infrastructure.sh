#!/bin/bash
# Infrastructure Validation Script for AI Trading Bot
# This script validates that all AWS infrastructure is properly set up
# Can be run locally or in GitHub Actions before deployments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_NAME="ai-trading-bot"
AWS_REGION=$(aws configure get region 2>/dev/null || echo "us-west-2")
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")
ENVIRONMENTS=("staging" "production")

# Parse arguments
ENVIRONMENT_FILTER=""
VERBOSE=false
FIX_ISSUES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --environment)
            ENVIRONMENT_FILTER="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --fix)
            FIX_ISSUES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--environment staging|production] [--verbose] [--fix]"
            exit 1
            ;;
    esac
done

# Filter environments if specified
if [ -n "$ENVIRONMENT_FILTER" ]; then
    ENVIRONMENTS=("$ENVIRONMENT_FILTER")
fi

echo -e "${BLUE}🔍 Infrastructure Validation for AI Trading Bot${NC}"
echo -e "${BLUE}=============================================${NC}"
echo -e "Project: ${YELLOW}${PROJECT_NAME}${NC}"
echo -e "Region: ${YELLOW}${AWS_REGION}${NC}"
echo -e "Account: ${YELLOW}${ACCOUNT_ID}${NC}"
echo -e "Environments: ${YELLOW}${ENVIRONMENTS[*]}${NC}"
echo

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0
ISSUES_FIXED=0

# Utility functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; ((CHECKS_PASSED++)); }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; ((CHECKS_FAILED++)); }
log_fix() { echo -e "${GREEN}[FIXED]${NC} $1"; ((ISSUES_FIXED++)); }

# Check if resource exists and optionally fix it
check_resource() {
    local resource_type="$1"
    local resource_name="$2"
    local check_command="$3"
    local fix_command="$4"
    local description="$5"
    
    if [ "$VERBOSE" = "true" ]; then
        log_info "Checking $resource_type: $resource_name"
    fi
    
    if eval "$check_command" >/dev/null 2>&1; then
        log_success "$description exists"
        return 0
    else
        if [ "$FIX_ISSUES" = "true" ] && [ -n "$fix_command" ]; then
            log_info "Attempting to fix: $description"
            if eval "$fix_command" >/dev/null 2>&1; then
                log_fix "$description created"
                return 0
            else
                log_error "$description creation failed"
                return 1
            fi
        else
            log_error "$description missing"
            if [ -n "$fix_command" ]; then
                log_info "To fix: $fix_command"
            fi
            return 1
        fi
    fi
}

# 1. Validate AWS CLI and credentials
validate_aws_setup() {
    log_info "Validating AWS setup..."
    
    if ! command -v aws >/dev/null 2>&1; then
        log_error "AWS CLI not installed"
        return 1
    fi
    
    if [ -z "$ACCOUNT_ID" ]; then
        log_error "AWS CLI not configured or no permissions"
        return 1
    fi
    
    log_success "AWS CLI configured (Account: $ACCOUNT_ID)"
    
    # Check if jq is available
    if ! command -v jq >/dev/null 2>&1; then
        log_warning "jq not installed - some features may not work"
    fi
}

# 2. Validate S3 buckets
validate_s3_buckets() {
    log_info "Validating S3 buckets..."
    
    # Deployment bucket (shared)
    local deployment_bucket="${PROJECT_NAME}-deployments"
    check_resource "S3 Bucket" "$deployment_bucket" \
        "aws s3 ls s3://$deployment_bucket" \
        "aws s3 mb s3://$deployment_bucket --region $AWS_REGION" \
        "Deployment bucket ($deployment_bucket)"
    
    # Environment-specific backup buckets
    for env in "${ENVIRONMENTS[@]}"; do
        local backup_bucket="${PROJECT_NAME}-backups-${env}"
        check_resource "S3 Bucket" "$backup_bucket" \
            "aws s3 ls s3://$backup_bucket" \
            "aws s3 mb s3://$backup_bucket --region $AWS_REGION && aws s3api put-bucket-versioning --bucket $backup_bucket --versioning-configuration Status=Enabled" \
            "Backup bucket for $env ($backup_bucket)"
    done
}

# 3. Validate IAM roles and policies
validate_iam_resources() {
    log_info "Validating IAM resources..."
    
    for env in "${ENVIRONMENTS[@]}"; do
        local role_name="${PROJECT_NAME}-${env}-role"
        local profile_name="${PROJECT_NAME}-${env}-profile"
        local policy_name="${PROJECT_NAME}-${env}-policy"
        
        # Check IAM role
        check_resource "IAM Role" "$role_name" \
            "aws iam get-role --role-name $role_name" \
            "" \
            "IAM role for $env ($role_name)"
        
        # Check instance profile
        check_resource "Instance Profile" "$profile_name" \
            "aws iam get-instance-profile --instance-profile-name $profile_name" \
            "" \
            "Instance profile for $env ($profile_name)"
        
        # Check custom policy
        local policy_arn="arn:aws:iam::${ACCOUNT_ID}:policy/${policy_name}"
        check_resource "IAM Policy" "$policy_name" \
            "aws iam get-policy --policy-arn $policy_arn" \
            "" \
            "Custom policy for $env ($policy_name)"
        
        # Check if role has required policies attached
        if aws iam get-role --role-name "$role_name" >/dev/null 2>&1; then
            local attached_policies=$(aws iam list-attached-role-policies --role-name "$role_name" --query 'AttachedPolicies[].PolicyName' --output text)
            
            # Check for AWS managed policies
            local required_managed_policies=("CloudWatchAgentServerPolicy" "AmazonSSMManagedInstanceCore")
            for policy in "${required_managed_policies[@]}"; do
                if echo "$attached_policies" | grep -q "$policy"; then
                    log_success "Required policy attached to $env role: $policy"
                else
                    log_error "Missing policy on $env role: $policy"
                fi
            done
        fi
    done
    
    # Check GitHub Actions user
    local github_user="${PROJECT_NAME}-github-actions"
    check_resource "IAM User" "$github_user" \
        "aws iam get-user --user-name $github_user" \
        "" \
        "GitHub Actions IAM user ($github_user)"
    
    # Check if GitHub Actions user has access keys
    if aws iam get-user --user-name "$github_user" >/dev/null 2>&1; then
        local access_keys=$(aws iam list-access-keys --user-name "$github_user" --query 'AccessKeyMetadata[].AccessKeyId' --output text)
        if [ -n "$access_keys" ]; then
            log_success "GitHub Actions user has access keys"
        else
            log_error "GitHub Actions user has no access keys"
        fi
    fi
}

# 4. Validate AWS Secrets Manager
validate_secrets() {
    log_info "Validating AWS Secrets Manager..."
    
    for env in "${ENVIRONMENTS[@]}"; do
        local secret_name="${PROJECT_NAME}/${env}"
        
        check_resource "Secret" "$secret_name" \
            "aws secretsmanager describe-secret --secret-id $secret_name" \
            "" \
            "Secrets for $env environment ($secret_name)"
        
        # Check if we can access the secret value
        if aws secretsmanager describe-secret --secret-id "$secret_name" >/dev/null 2>&1; then
            if aws secretsmanager get-secret-value --secret-id "$secret_name" >/dev/null 2>&1; then
                log_success "Can access $env secrets"
            else
                log_error "Cannot access $env secrets (check permissions)"
            fi
        fi
    done
}

# 5. Validate Security Groups
validate_security_groups() {
    log_info "Validating security groups..."
    
    local sg_name="${PROJECT_NAME}-sg"
    local vpc_id=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text 2>/dev/null || echo "None")
    
    if [ "$vpc_id" = "None" ] || [ "$vpc_id" = "null" ]; then
        log_error "No default VPC found"
        return 1
    fi
    
    local sg_id=$(aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=$sg_name" "Name=vpc-id,Values=$vpc_id" \
        --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
    
    if [ "$sg_id" = "None" ] || [ "$sg_id" = "null" ]; then
        log_error "Security group missing: $sg_name"
    else
        log_success "Security group exists: $sg_name ($sg_id)"
    fi
}

# 6. Validate EC2 instances
validate_ec2_instances() {
    log_info "Validating EC2 instances..."
    
    for env in "${ENVIRONMENTS[@]}"; do
        local instance_tag="${PROJECT_NAME}-${env}"
        local profile_name="${PROJECT_NAME}-${env}-profile"
        
        # Check if instance exists and is running
        local instance_info=$(aws ec2 describe-instances \
            --filters "Name=tag:Name,Values=$instance_tag" "Name=instance-state-name,Values=running,pending,stopping,stopped" \
            --query 'Reservations[0].Instances[0].[InstanceId,State.Name,IamInstanceProfile.Arn]' \
            --output text 2>/dev/null || echo "None None None")
        
        read -r instance_id instance_state instance_profile_arn <<< "$instance_info"
        
        if [ "$instance_id" = "None" ] || [ "$instance_id" = "null" ]; then
            log_warning "No EC2 instance found for $env environment"
            log_info "Expected tag: Name=$instance_tag"
        else
            if [ "$instance_state" = "running" ]; then
                log_success "$env instance running: $instance_id"
            else
                log_warning "$env instance exists but not running: $instance_id ($instance_state)"
            fi
            
            # Check if instance has correct IAM role
            if echo "$instance_profile_arn" | grep -q "$profile_name"; then
                log_success "$env instance has correct IAM profile"
            else
                log_error "$env instance missing correct IAM profile"
                log_info "Expected: $profile_name, Found: $instance_profile_arn"
            fi
            
            # Check if SSM agent is working (if instance is running)
            if [ "$instance_state" = "running" ]; then
                if aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$instance_id" --query 'InstanceInformationList[0].InstanceId' --output text >/dev/null 2>&1; then
                    log_success "$env instance SSM agent is working"
                else
                    log_error "$env instance SSM agent not responding"
                fi
            fi
        fi
    done
}

# 7. Validate GitHub Actions setup
validate_github_actions() {
    log_info "Validating GitHub Actions setup..."
    
    # Check if workflow files exist
    local staging_workflow=".github/workflows/deploy-staging.yml"
    local production_workflow=".github/workflows/promote-to-production.yml"
    
    if [ -f "$staging_workflow" ]; then
        log_success "Staging deployment workflow exists"
    else
        log_error "Staging deployment workflow missing: $staging_workflow"
    fi
    
    if [ -f "$production_workflow" ]; then
        log_success "Production promotion workflow exists"
    else
        log_error "Production promotion workflow missing: $production_workflow"
    fi
    
    # Check if we're in a git repository
    if [ -d ".git" ]; then
        log_success "In a git repository"
        
        # Check if GitHub remote exists
        if git remote get-url origin 2>/dev/null | grep -q "github.com"; then
            log_success "GitHub remote configured"
        else
            log_warning "No GitHub remote found"
        fi
    else
        log_warning "Not in a git repository"
    fi
}

# Main validation function
main() {
    validate_aws_setup
    validate_s3_buckets
    validate_iam_resources
    validate_secrets
    validate_security_groups
    validate_ec2_instances
    validate_github_actions
    
    echo
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}VALIDATION SUMMARY${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo -e "Checks passed: ${GREEN}$CHECKS_PASSED${NC}"
    echo -e "Checks failed: ${RED}$CHECKS_FAILED${NC}"
    
    if [ "$FIX_ISSUES" = "true" ]; then
        echo -e "Issues fixed: ${GREEN}$ISSUES_FIXED${NC}"
    fi
    
    echo
    
    if [ $CHECKS_FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 All infrastructure validation checks passed!${NC}"
        echo -e "${GREEN}Your environment is ready for deployments.${NC}"
        exit 0
    else
        echo -e "${RED}❌ Infrastructure validation failed.${NC}"
        echo
        echo -e "${YELLOW}Quick fixes:${NC}"
        echo "1. Run infrastructure setup: ./scripts/setup_aws_infrastructure.sh"
        echo "2. Check AWS permissions and configuration"
        echo "3. Ensure all required resources are created"
        
        if [ "$FIX_ISSUES" = "false" ]; then
            echo "4. Run with --fix flag to attempt automatic fixes"
        fi
        
        exit 1
    fi
}

# Run main function
main "$@" 