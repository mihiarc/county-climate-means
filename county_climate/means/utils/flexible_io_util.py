#!/usr/bin/env python3
"""
Flexible I/O utilities for multiple climate models.

Provides model-agnostic file handling using the climate model handler system.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import xarray as xr

from county_climate.means.models import get_model_handler, ClimateModelHandler

logger = logging.getLogger(__name__)


def open_climate_dataset(file_path: Union[str, Path]) -> xr.Dataset:
    """
    Open a climate dataset with sensible defaults.
    
    Works with any climate model file format.
    """
    return xr.open_dataset(
        file_path,
        decode_times=False,  # Handle time manually to avoid issues
        chunks={'time': 365, 'lat': 50, 'lon': 50}  # Reasonable chunks
    )


class FlexibleClimateFileHandler:
    """
    Flexible file handler that supports multiple climate models.
    
    This replaces the hardcoded NorESM2FileHandler with a model-agnostic approach.
    """
    
    def __init__(self, data_directory: Union[str, Path], model_id: str = "NorESM2-LM"):
        """
        Initialize the flexible file handler.
        
        Args:
            data_directory: Base directory containing climate data
            model_id: Climate model identifier (e.g., "NorESM2-LM", "GFDL-ESM4")
        """
        self.data_dir = Path(data_directory)
        self.model_id = model_id
        
        # Get the appropriate model handler
        self.model_handler = get_model_handler(model_id, self.data_dir)
        
        logger.info(f"Initialized flexible file handler for {model_id}: {data_directory}")
    
    def extract_year_from_filename(self, file_path: str) -> Optional[int]:
        """Extract year from filename using model-specific logic."""
        filename = Path(file_path).name
        return self.model_handler.extract_year_from_filename(filename)
    
    def get_files_for_period(self, variable: str, scenario: str, 
                           start_year: int, end_year: int) -> List[str]:
        """
        Get list of files for a variable, scenario, and year range.
        
        Args:
            variable: Climate variable (pr, tas, tasmax, tasmin)
            scenario: Climate scenario (historical, ssp245, ssp585, etc.)
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            
        Returns:
            List of file paths as strings sorted by year
        """
        file_paths = self.model_handler.get_files_for_period(
            variable, scenario, start_year, end_year
        )
        return [str(path) for path in file_paths]
    
    def get_available_years(self, variable: str, scenario: str) -> Tuple[int, int]:
        """
        Get the available year range for a variable and scenario.
        
        Returns:
            Tuple of (start_year, end_year) or (0, 0) if no data found
        """
        return self.model_handler.get_available_years(variable, scenario)
    
    def validate_data_availability(self) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """
        Check data availability for all variables and scenarios.
        
        Returns:
            Dictionary with structure: {variable: {scenario: (start_year, end_year)}}
        """
        return self.model_handler.validate_data_availability()
    
    def get_hybrid_files_for_period(self, variable: str, target_year: int,
                                   projection_scenario: str = None,
                                   window_years: int = 30) -> Tuple[List[str], Dict[str, int]]:
        """
        Get files for a period that may span historical and future scenarios.
        
        Args:
            variable: Climate variable
            target_year: End year of the period
            projection_scenario: Which projection to use (if not specified, uses model default)
            window_years: Length of the period (default 30)
            
        Returns:
            Tuple of (file_paths, scenario_counts)
        """
        file_paths, counts = self.model_handler.get_hybrid_files_for_period(
            variable, target_year, projection_scenario, window_years
        )
        return [str(path) for path in file_paths], counts
    
    def get_supported_scenarios(self) -> List[str]:
        """Get list of scenarios supported by this model."""
        return self.model_handler.get_supported_scenarios()
    
    def get_model_config(self):
        """Get the model configuration."""
        return self.model_handler.config
    
    @property
    def model_name(self) -> str:
        """Get the full model name."""
        return self.model_handler.config.model_name
    
    @property
    def institution(self) -> str:
        """Get the institution that created this model."""
        return self.model_handler.config.institution


def save_climate_result(result: xr.DataArray, output_dir: Path, 
                       variable: str, region: str, period: str,
                       model_id: str = None) -> bool:
    """
    Save climate result to NetCDF file.
    
    Args:
        result: Climate data array to save
        output_dir: Output directory
        variable: Variable name
        region: Region name
        period: Period identifier
        model_id: Optional model identifier to include in filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Include model ID in filename if provided
        if model_id:
            filename = f"{variable}_{region}_{period}_{model_id}_climate_normal.nc"
        else:
            filename = f"{variable}_{region}_{period}_climate_normal.nc"
            
        output_file = output_dir / filename
        
        result.to_netcdf(output_file)
        logger.info(f"Saved: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")
        return False