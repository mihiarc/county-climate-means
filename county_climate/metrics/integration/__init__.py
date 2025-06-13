"""
Integration module for bridging existing metrics processing to the configuration-driven orchestrator.

This module provides stage handlers that wrap the existing metrics processing
components to work with the new pipeline orchestration system.
"""

from .stage_handlers import (
    metrics_stage_handler,
)

__all__ = [
    "metrics_stage_handler",
]