"""
Pipeline integration for connecting climate means to downstream processing.

This module provides interfaces and utilities for seamlessly connecting
the climate means processing stage to downstream climate extremes and
other analysis pipelines.
"""

from .interface import PipelineInterface, DownstreamConfig
from .bridge import PipelineBridge, BridgeConfiguration

__all__ = [
    "PipelineInterface",
    "DownstreamConfig", 
    "PipelineBridge",
    "BridgeConfiguration"
]