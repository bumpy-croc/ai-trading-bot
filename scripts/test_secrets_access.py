#!/usr/bin/env python3
"""
Test script to validate AWS Secrets Manager access
Used by bootstrap.sh to ensure proper IAM configuration
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError, NoCredentialsError


def test_secrets_access():
    """Test access to AWS Secrets Manager"""
    environment = os.getenv('ENVIRONMENT', 'staging')
    secret_name = f"ai-trading-bot/{environment}"
    
    print(f"🔐 Testing secrets access for environment: {environment}")
    print(f"Secret name: {secret_name}")
    
    try:
        # Initialize secrets manager client
        session = boto3.Session()
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
        session = boto3.Session()
        s3_client = session.client('s3')
        
        bucket_name = "ai-trading-bot-storage"
        
        # Test bucket access
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✓ Can access S3 bucket: {bucket_name}")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"❌ S3 bucket not found: {bucket_name}")
        elif error_code == 'AccessDenied':
            print("❌ Access denied to S3 bucket")
        else:
            print(f"❌ S3 error: {error_code}")
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