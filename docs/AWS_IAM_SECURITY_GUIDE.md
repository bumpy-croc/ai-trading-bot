# AWS IAM Security Configuration Guide for AI Trading Bot

## Table of Contents
1. [Security Principles](#security-principles)
2. [IAM Role Architecture](#iam-role-architecture)
3. [Step-by-Step Setup](#step-by-step-setup)
4. [Policy Templates](#policy-templates)
5. [Environment-Specific Configurations](#environment-specific-configurations)
6. [Security Monitoring](#security-monitoring)
7. [Incident Response](#incident-response)
8. [Best Practices Checklist](#best-practices-checklist)

## Security Principles

### 1. Least Privilege Access
- Grant only the minimum permissions required for functionality
- Separate roles for different environments (dev/staging/production)
- Use condition statements to restrict access by time, IP, or MFA

### 2. Defense in Depth
- Multiple layers of security controls
- Network isolation with VPC and Security Groups
- Application-level security with encrypted connections
- Monitoring and alerting at all layers

### 3. Zero Trust Architecture
- Never trust, always verify
- Assume breach mentality
- Continuous monitoring and validation

## IAM Role Architecture

### Role Hierarchy

```
ai-trading-bot-master-role (Production)
├── ai-trading-bot-staging-role (Staging)
├── ai-trading-bot-dev-role (Development)
└── ai-trading-bot-emergency-role (Break-glass access)
```

### Service Breakdown

| Service | Purpose | Risk Level | Access Type |
|---------|---------|------------|-------------|
| Secrets Manager | Store API keys | **HIGH** | Read-only with conditions |
| S3 | Backup storage | Medium | Read/Write to specific buckets |
| CloudWatch | Monitoring/Logs | Low | Write metrics/logs |
| Systems Manager | Configuration | Medium | Read parameters |
| SNS | Notifications | Low | Publish to specific topics |

## Step-by-Step Setup

### 1. Create Production IAM Role

First, create the trust policy that allows EC2 to assume the role:

```bash
# Create trust policy file
cat > ai-trading-bot-trust-policy.json << 'EOF'
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
          "aws:RequestedRegion": ["us-east-1", "us-west-2"]
        },
        "DateGreaterThan": {
          "aws:CurrentTime": "2024-01-01T00:00:00Z"
        }
      }
    }
  ]
}
EOF

# Create the role
aws iam create-role \
  --role-name ai-trading-bot-production-role \
--assume-role-policy-document file://ai-trading-bot-trust-policy.json \
  --description "Production role for AI Trading Bot with restricted access" \
  --max-session-duration 3600
```

### 2. Create Secrets Manager Policy (High Security)

```bash
cat > ai-trading-bot-secrets-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadProductionSecrets",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
                  "Resource": "arn:aws:secretsmanager:*:*:secret:ai-trading-bot/production-*",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "${aws:SourceVpce}"
        },
        "DateLessThan": {
          "aws:CurrentTime": "2025-12-31T23:59:59Z"
        },
        "IpAddress": {
          "aws:SourceIp": ["10.0.0.0/8", "172.16.0.0/12"]
        }
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
          "kms:ViaService": "secretsmanager.us-east-1.amazonaws.com"
        },
        "StringLike": {
                          "kms:EncryptionContext:SecretARN": "arn:aws:secretsmanager:*:*:secret:ai-trading-bot/production-*"
        }
      }
    }
  ]
}
EOF

aws iam create-policy \
  --policy-name AITraderSecretsProductionPolicy \
  --policy-document file://ai-trading-bot-secrets-policy.json \
  --description "Secure access to production secrets for AI Trading Bot"
```

### 3. Create S3 Backup Policy (Restricted)

```bash
cat > ai-trading-bot-s3-policy.json << 'EOF'
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
              "Resource": "arn:aws:s3:::ai-trading-bot-backups-production/*",
      "Condition": {
        "StringEquals": {
          "s3:x-amz-server-side-encryption": "AES256"
        },
        "StringLike": {
          "s3:x-amz-object-lock-mode": "GOVERNANCE"
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
              "Resource": "arn:aws:s3:::ai-trading-bot-backups-production",
      "Condition": {
        "StringEquals": {
          "s3:prefix": ["backups/", "logs/"]
        }
      }
    }
  ]
}
EOF

aws iam create-policy \
  --policy-name AITraderS3BackupPolicy \
  --policy-document file://ai-trading-bot-s3-policy.json
```

### 4. Create CloudWatch Policy (Monitoring)

```bash
cat > ai-trading-bot-cloudwatch-policy.json << 'EOF'
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
          "cloudwatch:namespace": "AITrader/Production"
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
              "Resource": "arn:aws:logs:*:*:log-group:/aws/ai-trading-bot/production*"
    }
  ]
}
EOF

aws iam create-policy \
  --policy-name AITraderCloudWatchPolicy \
  --policy-document file://ai-trading-bot-cloudwatch-policy.json
```

### 5. Create SNS Notification Policy

```bash
cat > ai-trading-bot-sns-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublishToAlertTopic",
      "Effect": "Allow",
      "Action": [
        "sns:Publish"
      ],
              "Resource": "arn:aws:sns:*:*:ai-trading-bot-alerts-production",
      "Condition": {
        "StringEquals": {
          "sns:Protocol": ["email", "sms"]
        }
      }
    }
  ]
}
EOF

aws iam create-policy \
  --policy-name AITraderSNSPolicy \
  --policy-document file://ai-trading-bot-sns-policy.json
```

### 6. Attach Policies to Role

```bash
# Get your AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Attach all policies
aws iam attach-role-policy \
  --role-name ai-trading-bot-production-role \
  --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/AITraderSecretsProductionPolicy

aws iam attach-role-policy \
  --role-name ai-trading-bot-production-role \
  --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/AITraderS3BackupPolicy

aws iam attach-role-policy \
  --role-name ai-trading-bot-production-role \
  --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/AITraderCloudWatchPolicy

aws iam attach-role-policy \
  --role-name ai-trading-bot-production-role \
  --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/AITraderSNSPolicy
```

### 7. Create Instance Profile

```bash
# Create instance profile
aws iam create-instance-profile \
  --instance-profile-name ai-trading-bot-production-profile

# Add role to instance profile
aws iam add-role-to-instance-profile \
  --instance-profile-name ai-trading-bot-production-profile \
  --role-name ai-trading-bot-production-role
```

## Environment-Specific Configurations

### Development Environment

```bash
# Create development role with broader access for testing
cat > ai-trading-bot-dev-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
              "Resource": "arn:aws:secretsmanager:*:*:secret:ai-trading-bot/development-*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:*"
      ],
      "Resource": [
        "arn:aws:s3:::ai-trading-bot-dev-*",
        "arn:aws:s3:::ai-trading-bot-dev-*/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:*",
        "logs:*"
      ],
      "Resource": "*"
    }
  ]
}
EOF
```

### Staging Environment

```bash
# Staging role - more restrictive than dev, less than production
cat > ai-trading-bot-staging-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
              "Resource": "arn:aws:secretsmanager:*:*:secret:ai-trading-bot/staging-*",
      "Condition": {
        "IpAddress": {
          "aws:SourceIp": ["10.0.0.0/8"]
        }
      }
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::ai-trading-bot-staging-*",
        "arn:aws:s3:::ai-trading-bot-staging-*/*"
      ]
    }
  ]
}
EOF
```

## Security Monitoring

### 1. CloudTrail Configuration

```bash
# Create CloudTrail for API auditing
aws cloudtrail create-trail \
  --name ai-trading-bot-audit-trail \
  --s3-bucket-name ai-trading-bot-audit-logs \
  --include-global-service-events \
  --is-multi-region-trail \
  --enable-log-file-validation

# Start logging
aws cloudtrail start-logging --name ai-trading-bot-audit-trail
```

### 2. CloudWatch Alarms

```bash
# Alarm for unauthorized secret access
aws cloudwatch put-metric-alarm \
  --alarm-name "AI-Trading-Bot-Unauthorized-Secret-Access" \
  --alarm-description "Alert on failed secret access attempts" \
  --metric-name "UserErrors" \
  --namespace "AWS/SecretsManager" \
  --statistic "Sum" \
  --period 300 \
  --threshold 3 \
  --comparison-operator "GreaterThanThreshold" \
  --evaluation-periods 1 \
  --alarm-actions "arn:aws:sns:us-east-1:${ACCOUNT_ID}:ai-trading-bot-security-alerts"

# Alarm for unusual API activity
aws cloudwatch put-metric-alarm \
  --alarm-name "AI-Trading-Bot-High-API-Usage" \
  --alarm-description "Alert on unusually high API usage" \
  --metric-name "CallCount" \
  --namespace "AWS/Usage" \
  --statistic "Sum" \
  --period 3600 \
  --threshold 1000 \
  --comparison-operator "GreaterThanThreshold" \
  --evaluation-periods 1
```

### 3. AWS Config Rules

```bash
# Monitor IAM role changes
aws configservice put-config-rule \
  --config-rule '{
    "ConfigRuleName": "ai-trading-bot-iam-role-changes",
    "Source": {
      "Owner": "AWS",
      "SourceIdentifier": "IAM_ROLE_MANAGED_POLICY_CHECK"
    },
    "InputParameters": "{\"managedPolicyArns\":\"arn:aws:iam::aws:policy/PowerUserAccess\"}"
  }'
```

## Advanced Security Features

### 1. Session Manager Integration

Replace SSH access with AWS Systems Manager Session Manager:

```bash
# Add Systems Manager policy
aws iam attach-role-policy \
  --role-name ai-trading-bot-production-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
```

### 2. VPC Endpoints for Security

```bash
# Create VPC endpoint for Secrets Manager
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-12345678 \
  --service-name com.amazonaws.us-east-1.secretsmanager \
  --vpc-endpoint-type Interface \
  --subnet-ids subnet-12345678 \
  --security-group-ids sg-12345678 \
  --policy-document '{
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": "*",
        "Action": "secretsmanager:GetSecretValue",
        "Resource": "arn:aws:secretsmanager:*:*:secret:ai-trading-bot/*"
      }
    ]
  }'
```

### 3. Resource-Based Policies

Add resource-based policies to secrets:

```bash
aws secretsmanager put-resource-policy \
  --secret-id ai-trading-bot/production \
  --resource-policy '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {
          "AWS": "arn:aws:iam::ACCOUNT-ID:role/ai-trading-bot-production-role"
        },
        "Action": "secretsmanager:GetSecretValue",
        "Resource": "*",
        "Condition": {
          "StringEquals": {
            "secretsmanager:ResourceTag/Environment": "Production"
          }
        }
      }
    ]
  }'
```

## Incident Response

### 1. Emergency Break-Glass Role

```bash
cat > emergency-break-glass-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "Bool": {
          "aws:MultiFactorAuthPresent": "true"
        },
        "NumericLessThan": {
          "aws:MultiFactorAuthAge": "3600"
        }
      }
    }
  ]
}
EOF

aws iam create-role \
  --role-name ai-trading-bot-emergency-access \
  --assume-role-policy-document file://emergency-trust-policy.json

aws iam create-policy \
  --policy-name EmergencyBreakGlassPolicy \
  --policy-document file://emergency-break-glass-policy.json
```

### 2. Automated Response

Create Lambda function for automated incident response:

```python
# Lambda function to disable trading on security alert
import boto3
import json

def lambda_handler(event, context):
    secrets_client = boto3.client('secretsmanager')
    
    # Disable trading by setting TRADING_MODE to 'disabled'
    try:
        secrets_client.update_secret(
            SecretId='ai-trading-bot/production',
            SecretString=json.dumps({
                'TRADING_MODE': 'disabled',
                'EMERGENCY_SHUTDOWN': 'true'
            })
        )
        
        # Send notification
        sns = boto3.client('sns')
        sns.publish(
            TopicArn='arn:aws:sns:us-east-1:ACCOUNT:ai-trading-bot-emergency',
            Message='EMERGENCY: Trading bot disabled due to security alert',
            Subject='AI Trader Emergency Shutdown'
        )
        
    except Exception as e:
        print(f"Error during emergency shutdown: {e}")
        raise
```

## Best Practices Checklist

### ✅ IAM Configuration
- [ ] Separate roles for each environment
- [ ] Least privilege access implemented
- [ ] Condition statements restrict access
- [ ] Regular policy reviews scheduled
- [ ] MFA required for sensitive operations

### ✅ Secrets Management
- [ ] API keys stored in Secrets Manager
- [ ] Resource-based policies applied
- [ ] Automatic rotation configured
- [ ] Access logging enabled
- [ ] VPC endpoints configured

### ✅ Monitoring & Alerting
- [ ] CloudTrail enabled for all regions
- [ ] CloudWatch alarms configured
- [ ] AWS Config rules deployed
- [ ] SNS notifications set up
- [ ] Log retention policies defined

### ✅ Network Security
- [ ] VPC with private subnets
- [ ] Security groups restrict access
- [ ] VPC endpoints for AWS services
- [ ] NAT Gateway for outbound traffic
- [ ] Network ACLs configured

### ✅ Incident Response
- [ ] Break-glass access procedures
- [ ] Automated response functions
- [ ] Emergency contact list
- [ ] Runbook documentation
- [ ] Regular drill exercises

## Security Validation Script

Create a script to validate your security configuration:

```bash
#!/bin/bash
# security-validation.sh

echo "🔒 AI Trader Security Validation"
echo "================================"

# Check IAM role exists
if aws iam get-role --role-name ai-trading-bot-production-role >/dev/null 2>&1; then
    echo "✅ Production IAM role exists"
else
    echo "❌ Production IAM role missing"
fi

# Check secrets access
if aws secretsmanager describe-secret --secret-id ai-trading-bot/production >/dev/null 2>&1; then
    echo "✅ Production secrets accessible"
else
    echo "❌ Cannot access production secrets"
fi

# Check CloudTrail
if aws cloudtrail describe-trails --trail-name-list ai-trading-bot-audit-trail >/dev/null 2>&1; then
    echo "✅ CloudTrail configured"
else
    echo "❌ CloudTrail not configured"
fi

# Check VPC endpoints
ENDPOINTS=$(aws ec2 describe-vpc-endpoints --filters "Name=service-name,Values=com.amazonaws.*.secretsmanager" --query 'VpcEndpoints[0].VpcEndpointId' --output text)
if [ "$ENDPOINTS" != "None" ]; then
    echo "✅ VPC endpoints configured"
else
    echo "⚠️  VPC endpoints not configured"
fi

echo "Security validation complete!"
```

## Maintenance Schedule

### Daily
- Monitor CloudWatch alarms
- Review access logs
- Check for failed authentication attempts

### Weekly
- Review IAM access reports
- Validate backup integrity
- Update security patches

### Monthly
- Rotate API keys
- Review and update policies
- Conduct security assessment
- Test incident response procedures

### Quarterly
- Full security audit
- Penetration testing
- Update emergency procedures
- Review and update documentation

---

**Remember**: Security is not a one-time setup but an ongoing process. Regularly review and update your security configuration as your application evolves and new threats emerge. 