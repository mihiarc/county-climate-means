#!/usr/bin/env python3
"""
Simple I/O utilities for climate data processing.

Provides essential file handling functionality for NorESM2-LM climate data
without unnecessary complexity or over-engineering.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import xarray as xr

logger = logging.getLogger(__name__)


def open_climate_dataset(file_path: str) -> xr.Dataset:
    """
    Open a climate dataset with sensible defaults.
    
    Simple, reliable dataset opening without complex fallback logic.
    """
    return xr.open_dataset(
        file_path,
        decode_times=False,  # Handle time manually to avoid issues
        chunks={'time': 365, 'lat': 50, 'lon': 50}  # Reasonable chunks
    )


def extract_year_from_filename(file_path: str) -> Optional[int]:
    """
    Extract year from NorESM2-LM filename.
    
    Expected formats: 
    - variable_day_NorESM2-LM_scenario_r1i1p1f1_gn_YYYY.nc
    - variable_day_NorESM2-LM_scenario_r1i1p1f1_gn_YYYY_v1.1.nc
    """
    try:
        filename = Path(file_path).name
        # Look for 4-digit year followed by optional version and .nc
        match = re.search(r'_(\d{4})(?:_v[\d.]+)?\.nc$', filename)
        if match:
            return int(match.group(1))
        return None
    except Exception:
        return None


class NorESM2FileHandler:
    """
    Simple file handler for NorESM2-LM data structure.
    
    Expected structure:
    data_dir/
    ├── pr/historical/
    ├── pr/ssp245/
    ├── tas/historical/
    └── tas/ssp245/
    """
    
    def __init__(self, data_directory: str):
        self.data_dir = Path(data_directory)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_directory}")
        
        logger.info(f"Initialized file handler: {data_directory}")
    
    def extract_year_from_filename(self, file_path: str) -> Optional[int]:
        """Extract year from filename."""
        return extract_year_from_filename(file_path)
    
    def get_files_for_period(self, variable: str, scenario: str, 
                           start_year: int, end_year: int) -> List[str]:
        """
        Get list of files for a variable, scenario, and year range.
        
        Args:
            variable: Climate variable (pr, tas, tasmax, tasmin)
            scenario: Climate scenario (historical, ssp245, ssp585)
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            
        Returns:
            List of file paths sorted by year
        """
        scenario_dir = self.data_dir / variable / scenario
        
        if not scenario_dir.exists():
            logger.warning(f"Directory not found: {scenario_dir}")
            return []
        
        files = []
        
        # Find all .nc files in the directory
        for file_path in scenario_dir.glob("*.nc"):
            year = self.extract_year_from_filename(str(file_path))
            if year and start_year <= year <= end_year:
                files.append(str(file_path))
        
        # Sort by year
        files.sort(key=lambda x: self.extract_year_from_filename(x) or 0)
        
        logger.debug(f"Found {len(files)} files for {variable} {scenario} {start_year}-{end_year}")
        return files
    
    def get_available_years(self, variable: str, scenario: str) -> Tuple[int, int]:
        """
        Get the available year range for a variable and scenario.
        
        Returns:
            Tuple of (start_year, end_year) or (0, 0) if no data found
        """
        scenario_dir = self.data_dir / variable / scenario
        
        if not scenario_dir.exists():
            return 0, 0
        
        years = []
        for file_path in scenario_dir.glob("*.nc"):
            year = self.extract_year_from_filename(str(file_path))
            if year:
                years.append(year)
        
        if years:
            return min(years), max(years)
        return 0, 0
    
    def validate_data_availability(self) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """
        Check data availability for all variables and scenarios.
        
        Returns:
            Dictionary with structure: {variable: {scenario: (start_year, end_year)}}
        """
        availability = {}
        variables = ['pr', 'tas', 'tasmax', 'tasmin']
        scenarios = ['historical', 'ssp245', 'ssp585']
        
        for variable in variables:
            availability[variable] = {}
            for scenario in scenarios:
                start_year, end_year = self.get_available_years(variable, scenario)
                if start_year > 0:
                    availability[variable][scenario] = (start_year, end_year)
                    logger.info(f"{variable} {scenario}: {start_year}-{end_year}")
        
        return availability
    
    def get_hybrid_files_for_period(self, variable: str, target_year: int, 
                                   window_years: int = 30) -> Tuple[List[str], Dict[str, int]]:
        """
        Get files for a period that may span historical and future scenarios.
        
        For a 30-year window ending at target_year, collect files from:
        - Historical data (if available)
        - SSP245 data (if needed)
        
        Args:
            variable: Climate variable
            target_year: End year of the period
            window_years: Length of the period (default 30)
            
        Returns:
            Tuple of (file_paths, scenario_counts)
        """
        start_year = target_year - window_years + 1
        all_files = []
        scenario_counts = {'historical': 0, 'ssp245': 0}
        
        # Get files year by year, preferring historical over ssp245
        for year in range(start_year, target_year + 1):
            # Try historical first
            hist_files = self.get_files_for_period(variable, 'historical', year, year)
            if hist_files:
                all_files.extend(hist_files)
                scenario_counts['historical'] += 1
            else:
                # Fall back to ssp245
                ssp245_files = self.get_files_for_period(variable, 'ssp245', year, year)
                if ssp245_files:
                    all_files.extend(ssp245_files)
                    scenario_counts['ssp245'] += 1
        
        logger.info(f"Hybrid files for {variable} target {target_year}: "
                   f"{scenario_counts['historical']} historical + {scenario_counts['ssp245']} ssp245")
        
        return all_files, scenario_counts


def save_climate_result(result: xr.DataArray, output_dir: Path, 
                       variable: str, region: str, period: str) -> bool:
    """
    Save climate result to NetCDF file.
    
    Args:
        result: Climate data array to save
        output_dir: Output directory
        variable: Variable name
        region: Region name
        period: Period identifier
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{variable}_{region}_{period}_climate_normal.nc"
        output_file = output_dir / filename
        
        result.to_netcdf(output_file)
        logger.info(f"Saved: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")
        return False 