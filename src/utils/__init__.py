"""
Utility Modules for Climate Data Processing

This module contains utility functions for:
- I/O operations and NetCDF file handling
- Regional processing and coordinate conversion
- Time handling and period generation
- Performance optimization and benchmarking
"""

from .io import (
    open_dataset_safely, 
    open_dataset_optimized,
    NorESM2FileHandler,
    save_climate_result,
    SAFE_CHUNKS
)

from .regions import (
    REGION_BOUNDS,
    extract_region,
    validate_region_bounds,
    get_region_crs_info,
    convert_longitude_bounds
)

from .time_handling import (
    generate_climate_periods,
    handle_time_coordinates,
    extract_year_from_filename,
    reconstruct_time_dataarray,
    determine_climatology_type,
    add_time_metadata,
    get_time_decoding_params,
    try_time_engines
)

from .optimization import SafeOptimizer

__all__ = [
    # I/O utilities
    'open_dataset_safely',
    'open_dataset_optimized', 
    'NorESM2FileHandler',
    'save_climate_result',
    'SAFE_CHUNKS',
    
    # Regional utilities
    'REGION_BOUNDS',
    'extract_region',
    'validate_region_bounds',
    'get_region_crs_info',
    'convert_longitude_bounds',
    
    # Time utilities
    'generate_climate_periods',
    'handle_time_coordinates',
    'extract_year_from_filename',
    'reconstruct_time_dataarray',
    'determine_climatology_type',
    'add_time_metadata',
    'get_time_decoding_params',
    'try_time_engines',
    
    # Optimization utilities
    'SafeOptimizer'
] 