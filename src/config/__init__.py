"""
Configuration Management for Climate Data Processing

This module provides centralized configuration management:
- settings: Main configuration settings
- constants: System constants and defaults
"""

from .settings import ClimateProcessingConfig, get_default_config, get_production_config, get_development_config, get_testing_config
from .constants import *

__all__ = [
    'ClimateProcessingConfig', 
    'get_default_config',
    'get_production_config', 
    'get_development_config',
    'get_testing_config'
] + [name for name in dir() if name.isupper()] 