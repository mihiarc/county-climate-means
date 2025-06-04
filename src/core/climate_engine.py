#!/usr/bin/env python3
"""
Core Climate Data Processing Engine

A comprehensive engine for climate data processing that provides:
- Regional boundary definitions and coordinate conversions
- Climate data calculations including 30-year climate normals
- Crash-resistant processing with memory management
- Time coordinate handling and climatology calculations

This is a refactored version of climate_means.py following OOP principles.
"""

import logging
import numpy as np
import xarray as xr
import gc
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

# Import utility modules
from ..utils.regions import REGION_BOUNDS, extract_region, validate_region_bounds
from ..utils.io import open_dataset_safely, NorESM2FileHandler, save_climate_result
from ..utils.time_handling import (
    generate_climate_periods, 
    handle_time_coordinates, 
    reconstruct_time_dataarray,
    determine_climatology_type,
    add_time_metadata
)

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
MIN_YEARS_FOR_CLIMATE_NORMAL = 30
SEASONAL_GROUPS = 4


class ClimateEngine:
    """
    Core climate data processing engine with sequential processing capabilities.
    
    Provides crash-resistant, memory-efficient processing of climate data
    for computing 30-year climate normals.
    """
    
    def __init__(self, data_directory: str, output_directory: str):
        """
        Initialize the climate processing engine.
        
        Args:
            data_directory: Path to input NetCDF files
            output_directory: Path for output files
        """
        self.data_directory = Path(data_directory)
        self.output_directory = Path(output_directory)
        self.file_handler = NorESM2FileHandler(str(data_directory))
        
        # Validate directories
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_directory}")
        
        # Create output directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Climate engine initialized: {data_directory} -> {output_directory}")
    
    def process_file(self, file_path: str, variable_name: str, region_key: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """
        Process a single file to extract daily climatology for a region.
        
        Args:
            file_path: Path to the NetCDF file
            variable_name: Name of the climate variable
            region_key: Region identifier
            
        Returns:
            Tuple of (year, processed_data) or (None, None) if processing fails
        """
        try:
            # Extract year from filename
            year = self.file_handler.extract_year_from_filename(file_path)
            if year is None:
                return None, None

            # Open dataset
            ds = open_dataset_safely(file_path)
            if ds is None:
                return None, None

            # Check if variable exists
            if variable_name not in ds.data_vars:
                return None, None

            # Handle time coordinates
            ds, time_method = handle_time_coordinates(ds, file_path)

            # Extract region
            region_ds = extract_region(ds, REGION_BOUNDS[region_key])
            var = region_ds[variable_name]

            # Calculate daily climatology
            climatology = calculate_daily_climatology(var, time_method, file_path)
            if climatology is None:
                return None, None

            # Compute result
            result = climatology.compute()
            result_array = result.values if hasattr(result, 'values') else np.array(result)

            # Close dataset and cleanup
            ds.close()
            del ds, region_ds, var, climatology
            gc.collect()

            return year, result_array

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None, None
    
    def process_period_region(self, period_info: Tuple[int, int, int, str], variable_name: str, 
                             region_key: str) -> Optional[xr.DataArray]:
        """
        Process a climate period for a specific region and variable.
        
        Args:
            period_info: Tuple of (start_year, end_year, target_year, period_name)
            variable_name: Climate variable name
            region_key: Region identifier
            
        Returns:
            Processed climate normal data array or None if failed
        """
        start_year, end_year, target_year, period_name = period_info
        
        try:
            # Get files for this period
            scenario = period_name.split('_')[0] if '_' in period_name else period_name
            files = self.file_handler.get_files_for_period(variable_name, scenario, start_year, end_year)
            
            if not files:
                logger.warning(f"No files found for {variable_name} {scenario} {start_year}-{end_year}")
                return None
            
            if len(files) < MIN_YEARS_FOR_CLIMATE_NORMAL:
                logger.warning(f"Insufficient files ({len(files)}) for climate normal (need {MIN_YEARS_FOR_CLIMATE_NORMAL})")
                return None
            
            logger.info(f"Processing {len(files)} files for {variable_name} {region_key} {period_name}")
            
            # Process files sequentially to get daily climatologies
            data_arrays = []
            years = []
            
            for i, file_path in enumerate(files):
                logger.debug(f"Processing file {i+1}/{len(files)}: {Path(file_path).name}")
                year, daily_clim = self.process_file(file_path, variable_name, region_key)
                if year is not None and daily_clim is not None:
                    data_arrays.append(daily_clim)
                    years.append(year)
                
                # Periodic garbage collection for memory management
                if (i + 1) % 10 == 0:
                    gc.collect()
            
            if len(data_arrays) < MIN_YEARS_FOR_CLIMATE_NORMAL:
                logger.warning(f"Only processed {len(data_arrays)} files successfully (need {MIN_YEARS_FOR_CLIMATE_NORMAL})")
                return None
            
            logger.info(f"Successfully processed {len(data_arrays)} files, computing climate normal")
            
            # Compute 30-year climate normal
            climate_normal = compute_climate_normal(data_arrays, years, target_year)
            
            if climate_normal is not None:
                # Add metadata
                climate_normal.attrs.update({
                    'variable': variable_name,
                    'region': REGION_BOUNDS[region_key]['name'],
                    'region_key': region_key,
                    'period': period_name,
                    'processing_method': 'Sequential processing without Dask',
                    'years_processed': len(years),
                    'start_year': min(years),
                    'end_year': max(years)
                })
                
                logger.info(f"Climate normal computed for {variable_name} {region_key} {period_name}")
                return climate_normal
            else:
                logger.error(f"Failed to compute climate normal for {variable_name} {region_key} {period_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing {variable_name} {period_name} in {region_key}: {e}")
            return None
    
    def process_workflow(self, variables: List[str], regions: List[str], 
                        scenarios: List[str], config: Dict[str, Any] = None) -> None:
        """
        Complete workflow for processing climate data using sequential processing.
        
        Args:
            variables: List of climate variables to process
            regions: List of region keys to process
            scenarios: List of climate scenarios to process
            config: Configuration dictionary (optional)
        """
        logger.info("Starting climate data processing workflow (sequential processing)")
        logger.info(f"Data directory: {self.data_directory}")
        logger.info(f"Output directory: {self.output_directory}")
        logger.info(f"Variables: {variables}")
        logger.info(f"Regions: {regions}")
        logger.info(f"Scenarios: {scenarios}")
        
        try:
            # Get data availability from file handler
            logger.info("Validating data availability...")
            data_availability_detailed = self.file_handler.validate_data_availability()
            
            # Simple conversion to format expected by generate_climate_periods
            data_availability = {}
            for scenario in scenarios:
                start_years = []
                end_years = []
                
                for variable in variables:
                    if variable in data_availability_detailed and scenario in data_availability_detailed[variable]:
                        start_year, end_year = data_availability_detailed[variable][scenario]
                        start_years.append(start_year)
                        end_years.append(end_year)
                
                if start_years and end_years:
                    data_availability[scenario] = {
                        'start': min(start_years),
                        'end': max(end_years)
                    }
            
            total_tasks = 0
            completed_tasks = 0
            
            # Count total tasks for progress tracking
            for scenario in scenarios:
                if scenario in data_availability:
                    periods = generate_climate_periods(scenario, data_availability)
                    for variable in variables:
                        if (variable in data_availability_detailed and 
                            scenario in data_availability_detailed[variable]):
                            for region_key in regions:
                                if validate_region_bounds(region_key):
                                    total_tasks += len(periods)
            
            logger.info(f"Total processing tasks: {total_tasks}")
            
            # Process each scenario sequentially
            for scenario in scenarios:
                if scenario not in data_availability:
                    logger.warning(f"Skipping scenario {scenario} - no data available")
                    continue
                    
                logger.info(f"Processing scenario: {scenario}")
                    
                # Generate climate periods for this scenario
                periods = generate_climate_periods(scenario, data_availability)
                
                if not periods:
                    logger.warning(f"No periods generated for scenario {scenario}")
                    continue
                
                logger.info(f"Generated {len(periods)} periods for scenario {scenario}")
                
                # Process each variable
                for variable in variables:
                    # Check if this variable is available for this scenario
                    if (variable not in data_availability_detailed or 
                        scenario not in data_availability_detailed[variable]):
                        logger.warning(f"Skipping {variable} for {scenario} - not available")
                        continue
                    
                    logger.info(f"Processing variable: {variable}")
                    
                    # Process each region
                    for region_key in regions:
                        if not validate_region_bounds(region_key):
                            logger.warning(f"Skipping invalid region: {region_key}")
                            continue
                        
                        logger.info(f"Processing region: {region_key}")
                        
                        # Process each period for this variable/region combination
                        for period_info in periods:
                            completed_tasks += 1
                            start_year, end_year, target_year, period_name = period_info
                            
                            logger.info(f"Task {completed_tasks}/{total_tasks}: Processing {variable} {region_key} {period_name}")
                            
                            # Process the period sequentially
                            result = self.process_period_region(
                                period_info, variable, region_key
                            )
                            
                            if result is not None:
                                # Save result
                                success = save_climate_result(result, self.output_directory, variable, region_key, period_name)
                                if success:
                                    logger.info(f"✓ Completed task {completed_tasks}/{total_tasks}")
                                else:
                                    logger.error(f"✗ Failed to save task {completed_tasks}/{total_tasks}")
                            else:
                                logger.error(f"✗ Failed to process task {completed_tasks}/{total_tasks}")
                            
                            # Cleanup memory after each task
                            if result is not None:
                                del result
                            gc.collect()
        
        except Exception as e:
            logger.error(f"Error in climate data workflow: {e}")
            raise
        finally:
            logger.info("Climate data processing workflow completed")


def compute_climate_normal(data_arrays: List[xr.DataArray], years: List[int], 
                          target_year: int) -> Optional[xr.DataArray]:
    """Simplified computation of climate normal from multiple data arrays."""
    logger.debug(f"Computing climate normal for target year {target_year} using {len(data_arrays)} years of data")
    
    if not data_arrays:
        logger.warning(f"No data arrays provided for target year {target_year}")
        return None
    
    try:
        # Convert numpy arrays to DataArrays if needed and align coordinates
        aligned_arrays = []
        for i, data in enumerate(data_arrays):
            if isinstance(data, np.ndarray):
                # Convert numpy array to DataArray
                da = reconstruct_time_dataarray(data, years[i])
            else:
                # Already a DataArray
                da = data.copy()
                # Add year coordinate for combining
                da = da.assign_coords(year=years[i])
            
            aligned_arrays.append(da)
        
        # Concatenate all arrays along a new 'year' dimension
        combined = xr.concat(aligned_arrays, dim='year')
        
        # Compute the mean across years
        climate_normal = combined.mean(dim='year')
        
        # Add basic metadata
        climate_normal = add_time_metadata(climate_normal, years, target_year)
        
        logger.debug(f"Successfully computed climate normal for {target_year}")
        return climate_normal
        
    except Exception as e:
        logger.error(f"Error computing climate normal for {target_year}: {e}")
        return None


def calculate_daily_climatology(var: xr.DataArray, time_method: str, file_path: str) -> Optional[xr.DataArray]:
    """Calculate daily climatology for input data."""
    try:
        # Check if we have dayofyear coordinate for daily climatology
        if 'dayofyear' in var.coords:
            # Calculate daily climatology - mean for each day of year
            climatology = var.groupby(var.dayofyear).mean(dim='time')
            logger.debug(f"Calculated daily climatology for {file_path}: {len(climatology.dayofyear)} days")
            return climatology
        else:
            # No daily climatology possible without dayofyear coordinate
            logger.error(f"Cannot calculate daily climatology for {file_path}: missing dayofyear coordinate")
            return None
        
    except Exception as e:
        logger.error(f"Daily climatology calculation failed for {file_path}: {e}")
        return None 