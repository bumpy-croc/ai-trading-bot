"""
Configuration management system for AI Trading Bot
Provides abstraction for accessing configuration from multiple sources
"""

from .config_manager import ConfigManager, get_config
from .paths import (
    get_project_root, get_data_dir, get_cache_dir, 
    get_database_path, get_sentiment_data_path, 
    resolve_data_path, ensure_dir_exists
)

__all__ = [
    'ConfigManager', 'get_config',
    'get_project_root', 'get_data_dir', 'get_cache_dir',
    'get_database_path', 'get_sentiment_data_path', 
    'resolve_data_path', 'ensure_dir_exists'
] 