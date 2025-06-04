#!/usr/bin/env python3
"""
Time Handling Utilities for Climate Data Processing

This module contains all time-related functionality for climate data processing,
including period generation, time coordinate handling, climatology calculations,
year extraction from filenames, and time-aware chunking strategies.

Extracted from climate_means.py, io_util.py, and regions.py for better modularity.
"""

import logging
import numpy as np
import xarray as xr
import re
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# PERIOD GENERATION FUNCTIONS
# =============================================================================

def generate_climate_periods(scenario: str, data_availability: Dict) -> List[Tuple[int, int, int, str]]:
    """Generate climate periods based on scenario."""
    periods = []
    
    # Get available data range
    data_range = data_availability.get(scenario, {'start': 1850, 'end': 2014})
    data_start = data_range['start']
    data_end = data_range['end']
    
    if scenario == 'historical':
        # For historical, calculate 30-year periods for each year from 1980 to data_end
        for target_year in range(1980, min(2015, data_end + 1)):
            start_year = target_year - 29  # 30-year period ending in target_year
            # Check if we have data for the full period
            if start_year >= data_start:
                period_name = f"historical_{target_year}"
                periods.append((start_year, target_year, target_year, period_name))
            else:
                logger.warning(f"Insufficient data for 30-year period ending in {target_year}")
                logger.warning(f"Would need data from {start_year}, but data starts at {data_start}")
    else:
        # For projections, calculate 30-year periods for each year from 2015 to data_end
        for target_year in range(2015, min(2101, data_end + 1)):
            start_year = target_year - 29  # 30-year period ending in target_year
            period_name = f"{scenario}_{target_year}"
            periods.append((start_year, target_year, target_year, period_name))
    
    if not periods:
        logger.warning(f"No valid periods generated for {scenario}")
    else:
        logger.info(f"Generated {len(periods)} periods for {scenario}:")
        logger.info(f"First period: {periods[0][0]}-{periods[0][1]} (target: {periods[0][2]})")
        logger.info(f"Last period: {periods[-1][0]}-{periods[-1][1]} (target: {periods[-1][2]})")
    
    return periods


# =============================================================================
# TIME COORDINATE HANDLING
# =============================================================================

def handle_time_coordinates(ds: xr.Dataset, file_path: str) -> Tuple[xr.Dataset, str]:
    """Create day-of-year coordinates for daily climatology calculation."""
    if 'time' not in ds.coords:
        logger.warning(f"No time coordinate found in {file_path}")
        return ds, 'none'
    
    n_time = len(ds.time)
    
    # Create day-of-year coordinates for daily climatology
    # Use modulo to handle leap years safely
    day_of_year = (np.arange(n_time) % 365) + 1
    
    # Ensure we don't exceed 365 days
    day_of_year = np.clip(day_of_year, 1, 365)
    
    ds = ds.assign_coords(dayofyear=('time', day_of_year))
    logger.debug(f"Created day-of-year coordinate for {file_path} ({n_time} time steps)")
    return ds, 'daily'


# =============================================================================
# YEAR EXTRACTION AND FILENAME HANDLING
# =============================================================================

def extract_year_from_filename(file_path: str) -> Optional[int]:
    """Extract year from NorESM2-LM filename format."""
    try:
        # Expected format: {variable}_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc
        filename = Path(file_path).stem
        
        # Look for 4-digit year at the end of the filename
        year_match = re.search(r'_(\d{4})$', filename)
        if year_match:
            return int(year_match.group(1))
        
        # Fallback: look for any 4-digit year
        year_match = re.search(r'(\d{4})', filename)
        if year_match:
            return int(year_match.group(1))
        
        return None
    except Exception as e:
        logger.warning(f"Could not extract year from {file_path}: {e}")
        return None


def sort_files_by_year(files: List[str]) -> List[str]:
    """Sort files by year extracted from filename."""
    return sorted(files, key=lambda x: extract_year_from_filename(x) or 0)


def get_available_years_from_files(files: List[str]) -> Tuple[int, int]:
    """Get the available year range from a list of files."""
    years = []
    for file_path in files:
        year = extract_year_from_filename(file_path)
        if year is not None:
            years.append(year)
    
    if years:
        return min(years), max(years)
    else:
        return 0, 0


def filter_files_by_year_range(files: List[str], start_year: int, end_year: int) -> List[str]:
    """Filter files to only include those within the specified year range."""
    filtered_files = []
    for file_path in files:
        year = extract_year_from_filename(file_path)
        if year is not None and start_year <= year <= end_year:
            filtered_files.append(file_path)
    return filtered_files


# =============================================================================
# CLIMATE NORMAL COMPUTATIONS WITH TIME HANDLING
# =============================================================================

def reconstruct_time_dataarray(data: np.ndarray, batch_year: int) -> xr.DataArray:
    """Reconstruct DataArray with proper time coordinates based on dimensions."""
    if data.ndim == 1:
        # 1D data - could be daily (365), seasonal (4), or other
        if len(data) == 365:
            # Daily climatology
            coords = {'dayofyear': np.arange(1, 366)}
            dims = ['dayofyear']
        elif len(data) == 4:
            # Seasonal data
            coords = {'season': np.arange(4)}
            dims = ['season']
        else:
            # Generic 1D data
            coords = {'time': np.arange(len(data))}
            dims = ['time']
    elif data.ndim == 2:
        # 2D data (lat, lon) - overall mean
        coords = {'lat': np.arange(data.shape[0]), 'lon': np.arange(data.shape[1])}
        dims = ['lat', 'lon']
    elif data.ndim == 3:
        # 3D data (time/dayofyear/season, lat, lon)
        if data.shape[0] == 365:
            # Daily climatology
            coords = {
                'dayofyear': np.arange(1, 366),
                'lat': np.arange(data.shape[1]), 
                'lon': np.arange(data.shape[2])
            }
            dims = ['dayofyear', 'lat', 'lon']
        elif data.shape[0] == 4:
            # Seasonal climatology
            coords = {
                'season': np.arange(4),
                'lat': np.arange(data.shape[1]), 
                'lon': np.arange(data.shape[2])
            }
            dims = ['season', 'lat', 'lon']
        else:
            # Generic time dimension
            coords = {
                'time': np.arange(data.shape[0]),
                'lat': np.arange(data.shape[1]), 
                'lon': np.arange(data.shape[2])
            }
            dims = ['time', 'lat', 'lon']
    else:
        logger.warning(f"Unexpected data dimensions: {data.shape}")
        # Fallback to generic coordinates
        coords = {f'dim_{i}': np.arange(data.shape[i]) for i in range(data.ndim)}
        dims = [f'dim_{i}' for i in range(data.ndim)]
    
    da = xr.DataArray(data, coords=coords, dims=dims)
    
    # Add year coordinate for combining
    da = da.assign_coords(year=batch_year)
    return da


def determine_climatology_type(result: xr.DataArray) -> str:
    """Determine the climatology type for metadata based on dimensions."""
    if 'dayofyear' in result.dims:
        return f"daily ({len(result.dayofyear)} days)"
    elif 'season' in result.dims:
        return f"seasonal ({len(result.season)} seasons)"
    elif 'time' in result.dims:
        return f"temporal ({len(result.time)} time steps)"
    else:
        return "overall mean"


def add_time_metadata(result: xr.DataArray, years: List[int], target_year: int) -> xr.DataArray:
    """Add time-related metadata to climate normal result."""
    climatology_type = determine_climatology_type(result)
    
    result.attrs.update({
        'long_name': f'30-year climate normal for {target_year}',
        'description': f'Average of {climatology_type} climatologies from {min(years)} to {max(years)}',
        'target_year': target_year,
        'source_years': f"{min(years)}-{max(years)}",
        'number_of_years': len(years),
        'processing_method': 'batch_averaged_climatology',
        'climatology_type': climatology_type
    })
    
    return result


# =============================================================================
# TIME DECODING AND DATASET OPENING UTILITIES
# =============================================================================

def get_time_decoding_params(safe_mode: bool = False) -> Dict[str, Any]:
    """Get time decoding parameters for dataset opening."""
    if safe_mode:
        return {
            'decode_times': False,  # Avoid time decoding crashes
            'use_cftime': False,
            'mask_and_scale': True
        }
    else:
        return {
            'decode_times': True,
            'use_cftime': True,
            'mask_and_scale': True
        }


def try_time_engines() -> List[str]:
    """Get engine priority order for time-related dataset opening."""
    return ['netcdf4', 'h5netcdf', 'scipy']


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_year_range(start_year: int, end_year: int) -> bool:
    """Validate that year range is reasonable."""
    if start_year > end_year:
        logger.error(f"Start year ({start_year}) cannot be greater than end year ({end_year})")
        return False
    
    if start_year < 1850 or end_year > 2150:
        logger.warning(f"Year range {start_year}-{end_year} is outside typical climate data range (1850-2150)")
    
    return True


def get_years_in_range(start_year: int, end_year: int) -> List[int]:
    """Get list of years in range."""
    if not validate_year_range(start_year, end_year):
        return []
    return list(range(start_year, end_year + 1))


def is_leap_year(year: int) -> bool:
    """Check if a year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def get_days_in_year(year: int) -> int:
    """Get number of days in a year."""
    return 366 if is_leap_year(year) else 365


def standardize_to_365_days(day_of_year: np.ndarray, year: int) -> np.ndarray:
    """Standardize day-of-year to 365 days, handling leap years."""
    if is_leap_year(year) and len(day_of_year) == 366:
        # Remove February 29th (day 60 in leap years)
        mask = day_of_year != 60
        standardized = day_of_year[mask]
        # Adjust days after Feb 29th
        standardized[standardized > 60] -= 1
        return standardized
    else:
        # Ensure we don't exceed 365 days
        return np.clip(day_of_year, 1, 365) 