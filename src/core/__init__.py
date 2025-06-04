"""
Core Climate Processing Engines

This module contains the core climate data processing engines:
- ClimateEngine: Sequential processing with crash resistance
- MultiprocessingEngine: High-performance parallel processing
"""

from .climate_engine import ClimateEngine, compute_climate_normal, calculate_daily_climatology
from .multiprocessing_engine import ClimateMultiprocessor, ProcessingConfig, benchmark_multiprocessing_speedup

__all__ = [
    'ClimateEngine',
    'ClimateMultiprocessor', 
    'ProcessingConfig',
    'benchmark_multiprocessing_speedup',
    'compute_climate_normal',
    'calculate_daily_climatology'
] 