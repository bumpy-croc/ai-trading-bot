"""
Railway configuration provider that reads from Railway environment variables
"""
import os
from typing import Optional, Dict, Any
from .base import ConfigProvider


class RailwayProvider(ConfigProvider):
    """
    Configuration provider for Railway deployments.
    
    Railway automatically injects environment variables like:
    - RAILWAY_ENVIRONMENT_NAME (staging/production)
    - RAILWAY_PROJECT_NAME
    - RAILWAY_SERVICE_NAME
    - Plus any custom variables set in Railway dashboard
    """
    
    def __init__(self):
        """Initialize Railway provider"""
        self._env_vars = {}
        self._load_env_vars()
    
    @property
    def provider_name(self) -> str:
        """Return the name of this provider"""
        return "Railway"
    
    def _load_env_vars(self) -> None:
        """Load all environment variables"""
        self._env_vars = dict(os.environ)
    
    def is_available(self) -> bool:
        """
        Check if running in Railway environment.
        
        Returns:
            True if Railway environment variables are detected
        """
        # Railway sets these environment variables automatically
        railway_indicators = [
            'RAILWAY_DEPLOYMENT_ID',
            'RAILWAY_PROJECT_ID',
            'RAILWAY_SERVICE_ID',
            'RAILWAY_ENVIRONMENT_ID'
        ]
        
        return any(key in self._env_vars for key in railway_indicators)
    
    def get(self, key: str) -> Optional[str]:
        """
        Get configuration value from Railway environment.
        
        Args:
            key: Configuration key to retrieve
            
        Returns:
            Configuration value or None if not found
        """
        if not self.is_available():
            return None
        
        # Try direct key first
        value = self._env_vars.get(key)
        if value is not None:
            return value
        
        # Try Railway-specific prefixes
        railway_key = f"RAILWAY_{key}"
        value = self._env_vars.get(railway_key)
        if value is not None:
            return value
        
        return None
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values from Railway environment.
        
        Returns:
            Dictionary of all environment variables
        """
        if not self.is_available():
            return {}
        
        return self._env_vars.copy()
    
    def refresh(self) -> None:
        """Refresh environment variables"""
        self._load_env_vars()
    
    def get_railway_info(self) -> Dict[str, Optional[str]]:
        """
        Get Railway-specific deployment information.
        
        Returns:
            Dictionary with Railway deployment details
        """
        return {
            'project_id': self.get('RAILWAY_PROJECT_ID'),
            'service_id': self.get('RAILWAY_SERVICE_ID'),
            'environment_id': self.get('RAILWAY_ENVIRONMENT_ID'),
            'deployment_id': self.get('RAILWAY_DEPLOYMENT_ID'),
            'replica_id': self.get('RAILWAY_REPLICA_ID'),
            'public_domain': self.get('RAILWAY_PUBLIC_DOMAIN'),
            'private_domain': self.get('RAILWAY_PRIVATE_DOMAIN')
        }