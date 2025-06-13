"""
Integration module for bridging existing means processing to the configuration-driven orchestrator.

This module provides stage handlers that wrap the existing RegionalClimateProcessor
and other components to work with the new pipeline orchestration system.
"""

from .stage_handlers import (
    means_stage_handler,
    validation_stage_handler,
)

__all__ = [
    "means_stage_handler",
    "validation_stage_handler",
]