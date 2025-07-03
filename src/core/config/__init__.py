"""
Configuration management system for AI Trading Bot
Provides abstraction for accessing configuration from multiple sources
"""

from .config_manager import ConfigManager, get_config

__all__ = ['ConfigManager', 'get_config'] 