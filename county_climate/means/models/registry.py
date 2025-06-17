#!/usr/bin/env python3
"""
Climate model registry for managing different GCM handlers.

Provides a centralized registry for registering and retrieving climate model handlers.
"""

import logging
from pathlib import Path
from typing import Dict, Type, Optional, List

from .base import ClimateModelHandler
from .noresm2 import NorESM2Handler

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for climate model handlers."""
    
    def __init__(self):
        """Initialize the model registry."""
        self._handlers: Dict[str, Type[ClimateModelHandler]] = {}
        self._instances: Dict[str, ClimateModelHandler] = {}
        
        # Register default handlers
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default climate model handlers."""
        self.register("NorESM2-LM", NorESM2Handler)
        self.register("NorESM2", NorESM2Handler)  # Alias
    
    def register(self, model_id: str, handler_class: Type[ClimateModelHandler]):
        """
        Register a climate model handler.
        
        Args:
            model_id: Model identifier (e.g., "NorESM2-LM", "GFDL-ESM4")
            handler_class: Handler class that extends ClimateModelHandler
        """
        if not issubclass(handler_class, ClimateModelHandler):
            raise TypeError(f"{handler_class} must be a subclass of ClimateModelHandler")
        
        self._handlers[model_id] = handler_class
        logger.info(f"Registered handler for {model_id}: {handler_class.__name__}")
    
    def get_handler(self, model_id: str, base_path: Path) -> ClimateModelHandler:
        """
        Get a climate model handler instance.
        
        Args:
            model_id: Model identifier
            base_path: Base path containing model data
            
        Returns:
            Initialized handler instance
            
        Raises:
            ValueError: If model_id is not registered
        """
        if model_id not in self._handlers:
            available = ", ".join(self._handlers.keys())
            raise ValueError(f"Unknown model: {model_id}. Available: {available}")
        
        # Create cache key
        cache_key = f"{model_id}:{base_path}"
        
        # Return cached instance if available
        if cache_key in self._instances:
            return self._instances[cache_key]
        
        # Create new instance
        handler_class = self._handlers[model_id]
        instance = handler_class(base_path)
        self._instances[cache_key] = instance
        
        return instance
    
    def list_models(self) -> List[str]:
        """Get list of registered model IDs."""
        return list(self._handlers.keys())
    
    def clear_cache(self):
        """Clear cached handler instances."""
        self._instances.clear()


# Global registry instance
_registry = ModelRegistry()


def get_model_handler(model_id: str, base_path: Path) -> ClimateModelHandler:
    """
    Get a climate model handler from the global registry.
    
    Args:
        model_id: Model identifier (e.g., "NorESM2-LM")
        base_path: Base path containing model data
        
    Returns:
        Initialized handler instance
    """
    return _registry.get_handler(model_id, base_path)


def register_model(model_id: str, handler_class: Type[ClimateModelHandler]):
    """
    Register a climate model handler in the global registry.
    
    Args:
        model_id: Model identifier
        handler_class: Handler class that extends ClimateModelHandler
    """
    _registry.register(model_id, handler_class)


def list_available_models() -> List[str]:
    """Get list of available model IDs."""
    return _registry.list_models()