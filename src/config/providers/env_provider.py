"""
Environment variables configuration provider
"""
import os
from typing import Optional, Dict, Any
from .base import ConfigProvider


class EnvVarProvider(ConfigProvider):
    """Provider that reads configuration from environment variables"""
    
    def __init__(self):
        self._prefix = ""  # Optional prefix for env vars
    
    def get(self, key: str) -> Optional[str]:
        """Get a configuration value from environment variables"""
        env_key = f"{self._prefix}{key}" if self._prefix else key
        return os.environ.get(env_key)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all environment variables (optionally filtered by prefix)"""
        if self._prefix:
            return {
                k[len(self._prefix):]: v 
                for k, v in os.environ.items() 
                if k.startswith(self._prefix)
            }
        return dict(os.environ)
    
    def is_available(self) -> bool:
        """Environment variables are always available"""
        return True
    
    @property
    def provider_name(self) -> str:
        return "Environment Variables" 