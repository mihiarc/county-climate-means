#!/usr/bin/env python3
"""
I/O functionality for climate data processing.

This module contains all file handling and data I/O operations for climate data processing,
including dataset opening, file path management, and the NorESM2FileHandler class.
"""

import logging
import numpy as np
import xarray as xr
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

# Import time handling functionality
from time_util import extract_year_from_filename, get_time_decoding_params, try_time_engines

logger = logging.getLogger(__name__)

# Conservative chunk sizes to prevent crashes
SAFE_CHUNKS = {
    'time': -1,     # Don't chunk time dimension - avoids groupby issues
    'lat': 50,      # Larger spatial chunks
    'lon': 50       # Larger spatial chunks
}


def open_dataset_safely(file_path: str) -> Optional[xr.Dataset]:
    """Open dataset with crash-resistant settings."""
    try:
        # Get safe time decoding parameters
        time_params = get_time_decoding_params(safe_mode=True)
        
        ds = xr.open_dataset(
            file_path,
            chunks=SAFE_CHUNKS,
            engine='netcdf4',
            cache=False,         # Don't cache to save memory
            lock=False,          # Avoid threading issues
            **time_params
        )
        return ds
    except Exception as e:
        logger.error(f"Failed to open {file_path}: {e}")
        return None


def open_dataset_optimized(file_path: str, chunks: Dict[str, int]) -> 'xr.Dataset':
    """Open dataset with optimization for climate data."""
    # Get engine priority order and time parameters
    engines = try_time_engines()
    time_params = get_time_decoding_params(safe_mode=False)
    
    for engine in engines:
        try:
            return xr.open_dataset(
                file_path,
                chunks=chunks,
                engine=engine,
                cache=False,
                parallel=True,  # Enable parallel loading where supported
                **time_params
            )
        except Exception as e:
            logger.warning(f"Failed to open {file_path} with {engine} engine: {e}")
            continue
    
    # If all engines fail, try without decode_times
    logger.warning(f"Trying to open {file_path} without time decoding")
    try:
        safe_time_params = get_time_decoding_params(safe_mode=True)
        return xr.open_dataset(
            file_path,
            chunks=chunks,
            cache=False,
            **safe_time_params
        )
    except Exception as e:
        logger.error(f"Failed to open {file_path} with all fallback methods: {e}")
        raise


class NorESM2FileHandler:
    """
    File handler specifically designed for NorESM2-LM data structure.
    
    Expected structure:
    /path/to/data/NorESM2-LM/
    ├── pr/
    │   ├── historical/
    │   ├── ssp245/
    │   └── ssp585/
    ├── tas/
    │   ├── historical/
    │   ├── ssp245/
    │   └── ssp585/
    ├── tasmax/
    └── tasmin/
    
    File naming: {variable}_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc
    """
    
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        self.model_name = "NorESM2-LM"
        self.variant_label = "r1i1p1f1"
        self.grid_label = "gn"
        
        # Validate data directory structure
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_directory}")
        
        # Check for expected variable directories
        expected_vars = ['pr', 'tas', 'tasmax', 'tasmin']
        missing_vars = []
        for var in expected_vars:
            var_path = self.data_directory / var
            if not var_path.exists():
                missing_vars.append(var)
        
        if missing_vars:
            logger.warning(f"Missing variable directories: {missing_vars}")
        
        logger.info(f"Initialized NorESM2 file handler for: {data_directory}")
    
    def extract_year_from_filename(self, file_path: str) -> Optional[int]:
        """Extract year from NorESM2-LM filename format."""
        # Use the extracted function from time_util
        return extract_year_from_filename(file_path)
    
    def get_files_for_period(self, variable_name: str, scenario: str, start_year: int, end_year: int) -> List[str]:
        """Get list of NorESM2-LM files for a given period."""
        files = []
        
        try:
            # Construct the directory path
            scenario_dir = self.data_directory / variable_name / scenario
            
            if not scenario_dir.exists():
                logger.warning(f"Scenario directory does not exist: {scenario_dir}")
                return []
            
            # Expected filename pattern
            pattern = f"{variable_name}_day_{self.model_name}_{scenario}_{self.variant_label}_{self.grid_label}_*.nc"
            
            logger.debug(f"Searching for files in {scenario_dir} with pattern: {pattern}")
            
            # Find all matching files
            for file_path in scenario_dir.glob(pattern):
                year = self.extract_year_from_filename(str(file_path))
                if year is not None and start_year <= year <= end_year:
                    files.append(str(file_path))
            
            # Sort files by year
            files.sort(key=lambda x: self.extract_year_from_filename(x) or 0)
            
            logger.debug(f"Found {len(files)} files for {variable_name} {scenario} {start_year}-{end_year}")
            
            return files
            
        except Exception as e:
            logger.error(f"Error getting files for {variable_name} {scenario} {start_year}-{end_year}: {e}")
            return []
    
    def get_available_years(self, variable_name: str, scenario: str) -> Tuple[int, int]:
        """Get the available year range for a variable and scenario."""
        try:
            scenario_dir = self.data_directory / variable_name / scenario
            
            if not scenario_dir.exists():
                return 0, 0
            
            years = []
            pattern = f"{variable_name}_day_{self.model_name}_{scenario}_{self.variant_label}_{self.grid_label}_*.nc"
            
            for file_path in scenario_dir.glob(pattern):
                year = self.extract_year_from_filename(str(file_path))
                if year is not None:
                    years.append(year)
            
            if years:
                return min(years), max(years)
            else:
                return 0, 0
                
        except Exception as e:
            logger.error(f"Error getting available years for {variable_name} {scenario}: {e}")
            return 0, 0
    
    def validate_data_availability(self) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """Validate and return data availability for all variables and scenarios."""
        availability = {}
        
        variables = ['pr', 'tas', 'tasmax', 'tasmin']
        scenarios = ['historical', 'ssp245', 'ssp585']
        
        for variable in variables:
            availability[variable] = {}
            for scenario in scenarios:
                start_year, end_year = self.get_available_years(variable, scenario)
                if start_year > 0 and end_year > 0:
                    availability[variable][scenario] = (start_year, end_year)
                    logger.info(f"{variable} {scenario}: {start_year}-{end_year}")
                else:
                    logger.warning(f"No data found for {variable} {scenario}")
        
        return availability

    def get_hybrid_files_for_period(self, variable_name: str, target_year: int, window_years: int = 30) -> Tuple[List[str], Dict[str, int]]:
        """
        Get files for a hybrid period that may span historical and SSP245 scenarios.
        
        Args:
            variable_name: Climate variable (e.g., 'pr', 'tas')
            target_year: The target year for the climate normal
            window_years: Number of years to include (default 30)
            
        Returns:
            Tuple of (file_paths, scenario_counts) where:
            - file_paths: List of file paths spanning both scenarios as needed
            - scenario_counts: Dict with counts like {'historical': 29, 'ssp245': 1}
        """
        start_year = target_year - window_years + 1
        end_year = target_year
        
        logger.debug(f"Getting hybrid files for {variable_name} target year {target_year} (period {start_year}-{end_year})")
        
        all_files = []
        scenario_counts = {'historical': 0, 'ssp245': 0, 'ssp585': 0}
        
        # Get data availability
        availability = self.validate_data_availability()
        
        if variable_name not in availability:
            logger.warning(f"No data found for variable {variable_name}")
            return [], scenario_counts
        
        var_availability = availability[variable_name]
        
        # Determine year ranges for each scenario
        hist_start = hist_end = None
        ssp245_start = ssp245_end = None
        
        if 'historical' in var_availability:
            hist_start, hist_end = var_availability['historical']
        if 'ssp245' in var_availability:
            ssp245_start, ssp245_end = var_availability['ssp245']
        
        # Collect files year by year to ensure proper ordering
        for year in range(start_year, end_year + 1):
            file_found = False
            
            # Try historical first (if available and year is in range)
            if (hist_start is not None and hist_end is not None and 
                hist_start <= year <= hist_end):
                
                hist_files = self._get_files_for_single_year(variable_name, 'historical', year)
                if hist_files:
                    all_files.extend(hist_files)
                    scenario_counts['historical'] += len(hist_files)
                    file_found = True
                    logger.debug(f"    Year {year}: found {len(hist_files)} historical file(s)")
            
            # Try SSP245 if not found in historical (or if historical not available)
            if (not file_found and ssp245_start is not None and ssp245_end is not None and 
                ssp245_start <= year <= ssp245_end):
                
                ssp245_files = self._get_files_for_single_year(variable_name, 'ssp245', year)
                if ssp245_files:
                    all_files.extend(ssp245_files)
                    scenario_counts['ssp245'] += len(ssp245_files)
                    file_found = True
                    logger.debug(f"    Year {year}: found {len(ssp245_files)} SSP245 file(s)")
            
            # Try SSP585 as fallback (if needed)
            if (not file_found and 'ssp585' in var_availability):
                ssp585_start, ssp585_end = var_availability['ssp585']
                if ssp585_start <= year <= ssp585_end:
                    ssp585_files = self._get_files_for_single_year(variable_name, 'ssp585', year)
                    if ssp585_files:
                        all_files.extend(ssp585_files)
                        scenario_counts['ssp585'] += len(ssp585_files)
                        file_found = True
                        logger.debug(f"    Year {year}: found {len(ssp585_files)} SSP585 file(s)")
            
            if not file_found:
                logger.warning(f"    Year {year}: no file found in any scenario")
        
        total_files = len(all_files)
        total_years = sum(scenario_counts.values())
        
        logger.info(f"Hybrid collection for target {target_year}: {total_files} files, {total_years} years")
        logger.info(f"  Scenario breakdown: {dict(scenario_counts)}")
        
        return all_files, scenario_counts
    
    def _get_files_for_single_year(self, variable_name: str, scenario: str, year: int) -> List[str]:
        """Get files for a single year from a specific scenario."""
        try:
            scenario_dir = self.data_directory / variable_name / scenario
            
            if not scenario_dir.exists():
                return []
            
            # Expected filename pattern for specific year
            pattern = f"{variable_name}_day_{self.model_name}_{scenario}_{self.variant_label}_{self.grid_label}_{year}*.nc"
            
            files = []
            for file_path in scenario_dir.glob(pattern):
                # Double-check the year matches exactly
                file_year = self.extract_year_from_filename(str(file_path))
                if file_year == year:
                    files.append(str(file_path))
            
            return sorted(files)
            
        except Exception as e:
            logger.error(f"Error getting files for {variable_name} {scenario} {year}: {e}")
            return []


def save_climate_result(result: xr.DataArray, output_path: Path, variable: str, 
                       region_key: str, period_name: str) -> bool:
    """
    Save climate processing result to NetCDF file.
    
    Args:
        result: The processed climate data array
        output_path: Output directory path
        variable: Climate variable name
        region_key: Region identifier
        period_name: Period name (e.g., 'historical_2014')
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        output_filename = f"{variable}_{region_key}_{period_name}_climate_normal.nc"
        output_file = output_path / output_filename
        
        result.to_netcdf(output_file)
        logger.info(f"Saved: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save {output_file}: {e}")
        return False 