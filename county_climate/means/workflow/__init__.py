"""
Workflow integration for climate means processing with downstream pipeline support.

This module provides high-level workflow functions that integrate climate means
processing with the new catalog system, output organization, and pipeline bridge
functionality.
"""

from .integrated_processor import IntegratedClimateProcessor, ProcessingWorkflow
from .pipeline_workflow import PipelineAwareWorkflow

__all__ = [
    "IntegratedClimateProcessor",
    "ProcessingWorkflow", 
    "PipelineAwareWorkflow"
]