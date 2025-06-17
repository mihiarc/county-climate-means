"""
Climate model handlers for flexible GCM support.

This module provides abstract base classes and implementations for handling
different Global Climate Models (GCMs) with their specific file structures,
naming conventions, and scenarios.
"""

from .base import ClimateModelHandler, ModelConfig, ScenarioConfig
from .noresm2 import NorESM2Handler
from .registry import ModelRegistry, get_model_handler, register_model, list_available_models

__all__ = [
    'ClimateModelHandler',
    'ModelConfig',
    'ScenarioConfig',
    'NorESM2Handler',
    'ModelRegistry',
    'get_model_handler',
    'register_model',
    'list_available_models',
]