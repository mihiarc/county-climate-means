"""
Utility functions for county climate metrics processing.

This module provides shared utilities for coordinate transformations,
unit conversions, county handling, and other common operations.
"""

from .county_handler import (
    load_us_counties, 
    download_and_cache_counties, 
    filter_counties_by_region,
    validate_county_geometries,
    get_county_centroids
)
from .coordinates import (
    convert_indexed_to_geographic_coords,
    get_region_bounds,
    convert_longitude_360_to_180,
    convert_longitude_180_to_360,
    validate_geographic_coordinates
)
from .units import (
    convert_precipitation_units,
    calculate_annual_precipitation_from_daily,
    convert_temperature_units,
    get_climate_variable_info,
    standardize_variable_units
)

# Regional configuration and file discovery
from .region_config import (
    REGION_CONFIG,
    VARIABLE_CONFIG,
    SCENARIO_CONFIG,
    get_available_regions,
    get_region_info,
    get_variable_info,
    get_scenario_info,
    filter_counties_by_region,
    validate_region_variable_combination,
    get_valid_combinations,
    print_region_summary,
    print_variable_summary
)

from .file_discovery import (
    get_file_list,
    extract_year_from_filename,
    discover_available_data,
    validate_file_availability,
    get_scenario_for_year,
    get_years_for_scenario,
    build_file_path,
    print_data_availability_summary,
    get_missing_files
)

# Parallel processing framework
from .parallel_processor import (
    process_single_file,
    process_files_parallel,
    process_region_timeseries,
    get_available_metrics,
    get_optimal_worker_count,
    get_calculation_functions
)

# NetCDF loading utilities
from .netcdf_loader import load_and_prepare_netcdf

# Census data utilities
from .census_downloader import CensusCountyDownloader, download_us_counties

__all__ = [
    # County handling
    'load_us_counties',
    'download_and_cache_counties',
    'filter_counties_by_region', 
    'validate_county_geometries',
    'get_county_centroids',
    
    # Coordinate utilities
    'convert_indexed_to_geographic_coords',
    'get_region_bounds', 
    'convert_longitude_360_to_180',
    'convert_longitude_180_to_360',
    'validate_geographic_coordinates',
    
    # Unit conversion utilities
    'convert_precipitation_units',
    'calculate_annual_precipitation_from_daily',
    'convert_temperature_units',
    'get_climate_variable_info',
    'standardize_variable_units',
    
    # Regional configuration
    'REGION_CONFIG',
    'VARIABLE_CONFIG', 
    'SCENARIO_CONFIG',
    'get_available_regions',
    'get_region_info',
    'get_variable_info',
    'get_scenario_info',
    'filter_counties_by_region',
    'validate_region_variable_combination',
    'get_valid_combinations',
    'print_region_summary',
    'print_variable_summary',
    
    # File discovery
    'get_file_list',
    'extract_year_from_filename',
    'discover_available_data',
    'validate_file_availability',
    'get_scenario_for_year',
    'get_years_for_scenario',
    'build_file_path',
    'print_data_availability_summary',
    'get_missing_files',
    
    # Parallel processing
    'process_single_file',
    'process_files_parallel', 
    'process_region_timeseries',
    'get_available_metrics',
    'get_optimal_worker_count',
    'get_calculation_functions',
    
    # NetCDF loading
    'load_and_prepare_netcdf',
    
    # Census data utilities
    'CensusCountyDownloader',
    'download_us_counties'
]
