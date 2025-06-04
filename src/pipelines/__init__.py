"""
Climate Data Processing Pipelines

This module provides different processing pipelines:
- sequential: Sequential processing pipeline
- parallel: Parallel/multiprocessing pipeline
"""

from .sequential import SequentialPipeline
from .parallel import ParallelPipeline

__all__ = ['SequentialPipeline', 'ParallelPipeline'] 