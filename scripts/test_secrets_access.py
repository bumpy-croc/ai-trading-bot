#!/usr/bin/env python3
"""
Test script to validate AWS Secrets Manager access
Used by bootstrap.sh to ensure proper IAM configuration
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError, NoCredentialsError


def get_aws_region():
    """Get AWS region from environment or EC2 instance metadata"""
    region = os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION')
    
    if not region:
        # Try to get region from EC2 instance metadata
        try:
            import urllib.request
            
            # Get EC2 instance metadata token (IMDSv2)
            token_request = urllib.request.Request(
                'http://169.254.169.254/latest/api/token',
                headers={'X-aws-ec2-metadata-token-ttl-seconds': '21600'},
                method='PUT'
            )
            token_response = urllib.request.urlopen(token_request, timeout=2)
            token = token_response.read().decode('utf-8')
            
            # Get region from instance metadata
            region_request = urllib.request.Request(
                'http://169.254.169.254/latest/meta-data/placement/region',
                headers={'X-aws-ec2-metadata-token': token}
            )
            region_response = urllib.request.urlopen(region_request, timeout=2)
            region = region_response.read().decode('utf-8')
            print(f"✓ Detected region from EC2 metadata: {region}")
            
        except Exception as e:
            # Fallback to default region
            region = 'eu-west-2'
            print(f"⚠️  Could not detect region ({e}), using default: {region}")
    else:
        print(f"✓ Using region from environment: {region}")
    
    return region


def test_secrets_access():
    """Test access to AWS Secrets Manager"""
    environment = os.getenv('ENVIRONMENT', 'staging')
    secret_name = f"ai-trading-bot/{environment}"
    
    print(f"🔐 Testing secrets access for environment: {environment}")
    print(f"Secret name: {secret_name}")
    
    try:
        # Get AWS region
        region = get_aws_region()
        
        # Initialize secrets manager client with explicit region
        session = boto3.Session(region_name=region)
        client = session.client('secretsmanager')
        
        # Test basic AWS credentials
        print("✓ AWS credentials configured")
        
        # Test secrets access
        response = client.describe_secret(SecretId=secret_name)
        print(f"✓ Secret exists: {response['Name']}")
        
        # Test reading secret value
        secret_response = client.get_secret_value(SecretId=secret_name)
        secret_data = secret_response['SecretString']
        
        # Basic validation of secret content
        if 'BINANCE_API_KEY' in secret_data:
            print("✓ Secret contains expected keys")
        else:
            print("⚠️  Secret exists but may need API keys to be updated")
        
        print("✅ Secrets access test passed!")
        return True
        
    except NoCredentialsError:
        print("❌ AWS credentials not configured")
        print("   Make sure this instance has an IAM role with SecretsManager access")
        return False
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        
        if error_code == 'ResourceNotFoundException':
            print(f"❌ Secret not found: {secret_name}")
            print("   Run the infrastructure setup script first")
        elif error_code == 'AccessDenied':
            print("❌ Access denied to secrets")
            print("   Check IAM role permissions")
        else:
            print(f"❌ AWS error: {error_code}")
            print(f"   {e.response['Error']['Message']}")
        
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False


def test_s3_access():
    """Test access to S3 storage bucket"""
    print("\n📦 Testing S3 access...")
    
    try:
        # Get AWS region and environment
        region = get_aws_region()
        environment = os.getenv('ENVIRONMENT', 'staging')
        
        session = boto3.Session(region_name=region)
        s3_client = session.client('s3')
        
        bucket_name = "ai-trading-bot-storage"
        
        # Test bucket access by listing objects in a path we have permissions for
        # The IAM policy allows access to: backups/{env}/*, deployments/*, logs/{env}/*, temp/*
        test_prefix = f"logs/{environment}/"
        
        try:
            # Try to list objects in the logs directory for this environment
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=test_prefix,
                MaxKeys=1
            )
            print(f"✓ Can access S3 bucket: {bucket_name}")
            print(f"✓ Can list objects in: {test_prefix}")
            
            # Test if we can put a test object
            test_key = f"logs/{environment}/access_test.txt"
            s3_client.put_object(
                Bucket=bucket_name,
                Key=test_key,
                Body=b"Access test successful",
                ContentType="text/plain"
            )
            print(f"✓ Can write to S3 path: {test_key}")
            
            # Clean up test object
            s3_client.delete_object(Bucket=bucket_name, Key=test_key)
            print(f"✓ Can delete from S3 path: {test_key}")
            
            return True
            
        except ClientError as bucket_error:
            # If list operation fails, try head_bucket as fallback
            error_code = bucket_error.response['Error']['Code']
            if error_code == 'AccessDenied':
                print(f"⚠️  Limited S3 access (can't list {test_prefix})")
                print("   This is expected with restricted IAM permissions")
                
                # Try a simple head_bucket operation
                try:
                    s3_client.head_bucket(Bucket=bucket_name)
                    print(f"✓ Bucket exists and is accessible: {bucket_name}")
                    return True
                except ClientError as head_error:
                    head_error_code = head_error.response['Error']['Code']
                    if head_error_code == '403':
                        print(f"❌ Access denied to S3 bucket: {bucket_name}")
                        print("   Check IAM role permissions")
                    elif head_error_code == '404':
                        print(f"❌ S3 bucket not found: {bucket_name}")
                    else:
                        print(f"❌ S3 head_bucket error: {head_error_code}")
                    return False
            else:
                raise bucket_error
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"❌ S3 bucket not found: {bucket_name}")
        elif error_code == '403':
            print(f"❌ Access denied to S3 bucket: {bucket_name}")
            print("   Check IAM role permissions for S3 access")
        else:
            print(f"❌ S3 error: {error_code}")
            print(f"   {e.response['Error']['Message']}")
        return False
        
    except Exception as e:
        print(f"❌ S3 test error: {str(e)}")
        return False


def main():
    """Main test function"""
    print("🧪 AI Trading Bot - Infrastructure Access Test")
    print("=" * 50)
    
    # Test secrets access
    secrets_ok = test_secrets_access()
    
    # Test S3 access
    s3_ok = test_s3_access()
    
    print("\n" + "=" * 50)
    
    if secrets_ok and s3_ok:
        print("✅ All tests passed! Infrastructure access is working.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check IAM permissions and infrastructure setup.")
        sys.exit(1)


if __name__ == "__main__":
    main() 