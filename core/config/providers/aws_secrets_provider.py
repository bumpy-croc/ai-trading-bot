"""
AWS Secrets Manager configuration provider
"""
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from .base import ConfigProvider

try:
    import boto3
    from botocore.exceptions import ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


class AWSSecretsProvider(ConfigProvider):
    """Provider that reads configuration from AWS Secrets Manager"""
    
    def __init__(self, secret_name: Optional[str] = None, region: str = 'us-east-1', 
                 cache_ttl_minutes: int = 60):
        self.secret_name = secret_name or self._get_secret_name()
        self.region = region
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        self._client = None
        self._available = False
        
        if HAS_BOTO3 and self.secret_name:
            try:
                self._client = boto3.client('secretsmanager', region_name=self.region)
                # Test connection
                self._client.describe_secret(SecretId=self.secret_name)
                self._available = True
            except Exception:
                # Secrets Manager not available (likely running locally)
                pass
    
    def _get_secret_name(self) -> str:
        """Determine secret name from environment"""
        environment = os.environ.get('ENVIRONMENT', 'development')
        return f"ai-trading-bot/{environment}"
    
    def _fetch_secrets(self) -> Dict[str, Any]:
        """Fetch secrets from AWS Secrets Manager"""
        if not self._client or not self.secret_name:
            return {}
        
        try:
            response = self._client.get_secret_value(SecretId=self.secret_name)
            
            if 'SecretString' in response:
                return json.loads(response['SecretString'])
            else:
                # Binary secret not supported
                return {}
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                print(f"Secret {self.secret_name} not found")
            elif error_code == 'AccessDeniedException':
                print(f"Access denied to secret {self.secret_name}")
            else:
                print(f"Error accessing secret: {e}")
            return {}
        except Exception as e:
            print(f"Unexpected error accessing secrets: {e}")
            return {}
    
    def _refresh_cache_if_needed(self) -> None:
        """Refresh cache if it's expired"""
        now = datetime.now()
        if (not self._cache_time or 
            now - self._cache_time > self.cache_ttl):
            self._cache = self._fetch_secrets()
            self._cache_time = now
    
    def get(self, key: str) -> Optional[str]:
        """Get a configuration value from AWS Secrets Manager"""
        if not self._available:
            return None
        
        self._refresh_cache_if_needed()
        value = self._cache.get(key)
        return str(value) if value is not None else None
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values from AWS Secrets Manager"""
        if not self._available:
            return {}
        
        self._refresh_cache_if_needed()
        return self._cache.copy()
    
    def is_available(self) -> bool:
        """Check if AWS Secrets Manager is available"""
        return self._available
    
    @property
    def provider_name(self) -> str:
        return f"AWS Secrets Manager ({self.secret_name})"
    
    def refresh(self) -> None:
        """Force refresh of cached secrets"""
        self._cache_time = None
        self._refresh_cache_if_needed() 