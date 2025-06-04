#!/usr/bin/env python3
"""
Constants and Default Values for Climate Data Processing

This module contains all system constants, default values, and configuration
parameters used throughout the climate data processing system.
"""

from pathlib import Path

# =============================================================================
# DATA PROCESSING CONSTANTS
# =============================================================================

# Default data directories
DEFAULT_INPUT_DATA_DIR = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
DEFAULT_OUTPUT_BASE_DIR = "output/rolling_30year_climate_normals"

# Climate variables
CLIMATE_VARIABLES = ['pr', 'tas', 'tasmax', 'tasmin']

# Geographic regions
REGIONS = ['CONUS']  # Can be extended to include AK, HI, PRVI, GU

# Climate scenarios  
CLIMATE_SCENARIOS = ['historical', 'ssp245', 'ssp585']

# Time period constants
MIN_YEARS_FOR_CLIMATE_NORMAL = 25  # Minimum years needed for a climate normal
STANDARD_CLIMATE_NORMAL_YEARS = 30  # Standard WMO climate normal period
SEASONAL_GROUPS = 4

# Year ranges
HISTORICAL_START_YEAR = 1980
HISTORICAL_END_YEAR = 2014
HYBRID_START_YEAR = 2015
HYBRID_END_YEAR = 2044
FUTURE_START_YEAR = 2045
MAX_FUTURE_YEAR = 2100

# =============================================================================
# PERFORMANCE AND MULTIPROCESSING CONSTANTS
# =============================================================================

# Multiprocessing configuration (based on optimization testing)
OPTIMAL_WORKERS = 6  # Optimal worker count (4.3x speedup, 72% efficiency)
MAX_CORES = 24  # Maximum cores to use
CORES_PER_VARIABLE = 6  # Cores per variable in parallel processing
BATCH_SIZE_YEARS = 2  # Small batch sizes for memory management
MAX_MEMORY_PER_PROCESS_GB = 4  # Conservative memory per process
MEMORY_CHECK_INTERVAL = 10  # Check memory every N files

# Processing timeouts and retries
TIMEOUT_PER_FILE = 300  # 5 minutes timeout per file
MAX_RETRIES = 2
PROGRESS_INTERVAL = 5  # Report progress every N files

# =============================================================================
# I/O AND CHUNKING CONSTANTS
# =============================================================================

# Conservative chunk sizes to prevent crashes
SAFE_CHUNKS = {
    'time': -1,     # Don't chunk time dimension - avoids groupby issues
    'lat': 50,      # Larger spatial chunks
    'lon': 50       # Larger spatial chunks
}

# NetCDF engines priority order
NETCDF_ENGINES = ['netcdf4', 'h5netcdf', 'scipy']

# File naming patterns
NORESM2_FILE_PATTERN = "{variable}_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc"
OUTPUT_FILE_PATTERN = "{variable}_{region}_{period_type}_{target_year}_30yr_normal.nc"
COMBINED_FILE_PATTERN = "{variable}_{region}_{period_type}_{year_range}_all_normals.nc"

# =============================================================================
# MONITORING AND LOGGING CONSTANTS
# =============================================================================

# Progress tracking files
PROGRESS_STATUS_FILE = "processing_progress.json"
PROGRESS_LOG_FILE = "processing_progress.log"
STATUS_UPDATE_INTERVAL = 30  # Update status every 30 seconds

# Processing targets (for progress tracking)
PROCESSING_TARGETS = {
    'pr': {'historical': 35, 'hybrid': 30, 'ssp245': 32, 'total': 97},
    'tas': {'historical': 35, 'hybrid': 30, 'ssp245': 85, 'total': 150},
    'tasmax': {'historical': 35, 'hybrid': 30, 'ssp245': 85, 'total': 150},
    'tasmin': {'historical': 35, 'hybrid': 30, 'ssp245': 85, 'total': 150}
}

TOTAL_PROCESSING_TARGET = 547  # Sum of all targets

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# =============================================================================
# MODEL AND DATA CONSTANTS
# =============================================================================

# NorESM2-LM model information
MODEL_NAME = "NorESM2-LM"
VARIANT_LABEL = "r1i1p1f1"
GRID_LABEL = "gn"

# Variable metadata
VARIABLE_METADATA = {
    'pr': {
        'long_name': 'Precipitation',
        'standard_name': 'precipitation_flux',
        'units': 'kg m-2 s-1',
        'description': 'Daily precipitation rate'
    },
    'tas': {
        'long_name': 'Near-Surface Air Temperature',
        'standard_name': 'air_temperature',
        'units': 'K',
        'description': 'Daily mean near-surface air temperature'
    },
    'tasmax': {
        'long_name': 'Daily Maximum Near-Surface Air Temperature',
        'standard_name': 'air_temperature',
        'units': 'K',
        'description': 'Daily maximum near-surface air temperature'
    },
    'tasmin': {
        'long_name': 'Daily Minimum Near-Surface Air Temperature',
        'standard_name': 'air_temperature', 
        'units': 'K',
        'description': 'Daily minimum near-surface air temperature'
    }
}

# Scenario metadata
SCENARIO_METADATA = {
    'historical': {
        'description': 'Historical climate simulation (1850-2014)',
        'period': '1850-2014'
    },
    'ssp245': {
        'description': 'Shared Socioeconomic Pathway 2-4.5 (2015-2100)',
        'period': '2015-2100',
        'radiative_forcing': '4.5 W/m2 by 2100'
    },
    'ssp585': {
        'description': 'Shared Socioeconomic Pathway 5-8.5 (2015-2100)', 
        'period': '2015-2100',
        'radiative_forcing': '8.5 W/m2 by 2100'
    }
}

# =============================================================================
# KNOWN ISSUES AND WORKAROUNDS
# =============================================================================

# Known corrupted files to skip
CORRUPTED_FILES = {
    'tasmin_day_NorESM2-LM_ssp245_r1i1p1f1_gn_2033.nc'
}

# Memory management settings
GARBAGE_COLLECTION_INTERVAL = 10  # Force GC every N files
MEMORY_WARNING_THRESHOLD_GB = 70  # Warn if system memory exceeds this
MEMORY_CRITICAL_THRESHOLD_GB = 85  # Critical memory threshold 