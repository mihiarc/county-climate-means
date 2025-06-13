"""
Configuration management system for climate data processing pipelines.

This module provides configuration loading, validation, and management
capabilities for configuration-driven pipeline orchestration.
"""

from .integration_config import (
    PipelineConfiguration,
    StageConfiguration,
    ProcessingProfile,
    ProcessingStage,
    TriggerType,
    DataFlowType,
    EnvironmentType,
    ResourceLimits,
    DataTransformation,
    DataFlowConfiguration,
)

from .config_loader import (
    ConfigurationLoader,
    ConfigurationManager,
    ConfigurationError,
)

__all__ = [
    # Configuration models
    "PipelineConfiguration",
    "StageConfiguration", 
    "ProcessingProfile",
    "ProcessingStage",
    "TriggerType",
    "DataFlowType",
    "EnvironmentType",
    "ResourceLimits",
    "DataTransformation",
    "DataFlowConfiguration",
    
    # Configuration management
    "ConfigurationLoader",
    "ConfigurationManager",
    "ConfigurationError",
]