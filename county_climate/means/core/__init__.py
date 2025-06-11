"""
Core climate data processing modules.

This package contains the fundamental business logic and domain models
for climate data processing, including regional definitions, processing
engines, and multiprocessing capabilities.
"""

# Regional definitions and operations (core domain logic)
from .regions import REGION_BOUNDS, extract_region, validate_region_bounds

# Regional climate processing
from .regional_climate_processor import (
    RegionalProcessingConfig,
    RegionalClimateProcessor,
    create_regional_processor,
    process_region
)

# Multiprocessing engine
from .multiprocessing_engine import (
    MultiprocessingConfig,
    MultiprocessingEngine,
    create_multiprocessing_engine,
    process_climate_files_parallel
)

# Maximum performance processor
from .maximum_processor import (
    MaximumPerformanceProcessor,
    create_maximum_processor,
    run_maximum_processing
)

__all__ = [
    # Regional definitions
    'REGION_BOUNDS',
    'extract_region', 
    'validate_region_bounds',
    
    # Regional processing
    'RegionalProcessingConfig',
    'RegionalClimateProcessor',
    'create_regional_processor',
    'process_region',
    
    # Multiprocessing
    'MultiprocessingConfig',
    'MultiprocessingEngine', 
    'create_multiprocessing_engine',
    'process_climate_files_parallel',
    
    # Maximum performance processing
    'MaximumPerformanceProcessor',
    'create_maximum_processor',
    'run_maximum_processing',
]
