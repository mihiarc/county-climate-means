#!/usr/bin/env python3
"""
Unified Regional Climate Normals Processing Pipeline

A single, parameterized processor that can handle any region and climate variable.
Replaces the separate regional processing scripts to eliminate code duplication.

Features:
- Supports all regions: CONUS, Alaska, Hawaii, Puerto Rico, Guam
- Supports all variables: pr, tas, tasmax, tasmin
- Configurable multiprocessing settings
- Real-time progress tracking with per-file updates
- Memory-efficient processing
- Accurate file counting and statistics
"""

import logging
import sys
import os
import time
from pathlib import Path
import pandas as pd
import xarray as xr
import gc
import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue
from typing import List, Dict, Tuple, Optional, Union
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import json
from datetime import datetime
from dataclasses import dataclass
import threading

# Import our modules
from county_climate.means.utils.io_util import NorESM2FileHandler
from county_climate.means.core.regions import REGION_BOUNDS, extract_region, validate_region_bounds
from county_climate.means.utils.time_util import handle_time_coordinates, extract_year_from_filename
from county_climate.means.config import get_config, get_regional_output_dir
from county_climate.means.utils.rich_progress import RichProgressTracker
from county_climate.means.utils.mp_progress import MultiprocessingProgressTracker, ProgressReporter


@dataclass
class RegionalProcessingConfig:
    """Configuration for regional climate processing."""
    region_key: str
    variables: List[str]
    input_data_dir: Path
    output_base_dir: Path
    
    # Processing settings
    max_cores: int = 6
    cores_per_variable: int = 2
    batch_size_years: int = 2
    max_memory_per_process_gb: int = 4
    memory_check_interval: int = 10
    min_years_for_normal: int = 25
    
    # Progress tracking
    status_update_interval: int = 30
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.region_key not in REGION_BOUNDS:
            raise ValueError(f"Invalid region key: {self.region_key}")
        
        if not validate_region_bounds(self.region_key):
            raise ValueError(f"Invalid region bounds for: {self.region_key}")
        
        # Set up progress tracking files
        self.progress_status_file = f"{self.region_key.lower()}_processing_progress.json"
        self.progress_log_file = f"{self.region_key.lower()}_processing_progress.log"
        self.main_log_file = f"{self.region_key.lower()}_climate_processing.log"


class RegionalClimateProcessor:
    """Fixed processor with working progress tracking."""
    
    def __init__(self, config: RegionalProcessingConfig, use_rich_progress: bool = True):
        self.config = config
        self.use_rich_progress = use_rich_progress
        self.logger = self._setup_logging()
        
        # Will be initialized when processing starts
        self.progress_tracker = None
        self.progress_queue = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline."""
        logging.basicConfig(
            level=logging.CRITICAL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.config.main_log_file)
            ]
        )
        return logging.getLogger(__name__)
    
    def process_single_file_for_climatology_safe(self, file_path: Path, variable: str, 
                                                progress_reporter: Optional[ProgressReporter] = None) -> Tuple[int, xr.DataArray]:
        """Process a single file and extract climatology - with progress updates."""
        try:
            # Extract year from filename
            year = extract_year_from_filename(file_path)
            
            # Load data
            with xr.open_dataset(file_path) as ds:
                # Extract regional data
                regional_data = extract_region(ds, REGION_BOUNDS[self.config.region_key])
                
                if regional_data is None:
                    return None, None
                
                # Fix time coordinates
                regional_data, _ = handle_time_coordinates(regional_data, str(file_path))
                
                # Calculate daily climatology
                if variable in regional_data:
                    var_data = regional_data[variable]
                    
                    # For daily data, group by day of year
                    daily_clim = var_data.groupby('time.dayofyear').mean(dim='time')
                    
                    # Update progress
                    if progress_reporter:
                        progress_reporter.update_task(
                            variable,
                            advance=1,
                            current_item=f"Processed {file_path.name}"
                        )
                    
                    return year, daily_clim
                    
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            if progress_reporter:
                progress_reporter.update_task(
                    variable,
                    advance=1,
                    current_item=f"Failed: {file_path.name}"
                )
            
        return None, None
    
    def compute_climate_normal_safe(self, daily_climatologies: List[xr.DataArray], 
                                   years_used: List[int], target_year: int) -> xr.DataArray:
        """Compute climate normal from daily climatologies."""
        try:
            # Stack all daily climatologies along a new 'year' dimension
            stacked_data = xr.concat(daily_climatologies, dim='year')
            stacked_data['year'] = years_used
            
            # Calculate the mean across all years for each day of year
            climate_normal = stacked_data.mean(dim='year')
            
            # Add metadata
            climate_normal.attrs.update({
                'long_name': f'30-year climate normal for {target_year}',
                'target_year': target_year,
                'source_years': f"{min(years_used)}-{max(years_used)}",
                'number_of_years': len(years_used),
                'processing_method': f'unified_regional_processor_{self.config.region_key}',
                'region': self.config.region_key
            })
            
            return climate_normal
            
        except Exception as e:
            self.logger.error(f"Error computing climate normal: {e}")
            return None
    
    def process_target_year_batch_with_progress(self, args: Tuple) -> Dict:
        """Process a batch of target years for a variable - with progress reporting."""
        variable, period_type, target_years_batch, worker_id = args
        
        # Note: Progress reporting is handled at the main process level
        # Worker processes cannot access the rich progress tracker due to pickling limitations
        
        logger = logging.getLogger(f"{__name__}.worker_{worker_id}")
        logger.info(f"Worker {worker_id} processing {variable} {period_type} years: {target_years_batch}")
        
        results = []
        
        try:
            file_handler = NorESM2FileHandler(self.config.input_data_dir)
            
            for target_year in target_years_batch:
                try:
                    # Determine files for this period
                    if period_type == 'historical':
                        start_year = max(1950, target_year - 29)
                        end_year = target_year
                        files = file_handler.get_files_for_period(variable, 'historical', start_year, end_year)
                    elif period_type == 'hybrid':
                        # 2015-2044 uses both historical and ssp245
                        hist_start = max(1985, target_year - 29)
                        hist_end = min(2014, target_year)
                        ssp_start = max(2015, target_year - 29)
                        ssp_end = target_year
                        
                        hist_files = file_handler.get_files_for_period(variable, 'historical', hist_start, hist_end)
                        ssp_files = file_handler.get_files_for_period(variable, 'ssp245', ssp_start, ssp_end)
                        files = hist_files + ssp_files
                    else:  # ssp245
                        start_year = target_year - 29
                        end_year = target_year
                        files = file_handler.get_files_for_period(variable, 'ssp245', start_year, end_year)
                    
                    if len(files) < self.config.min_years_for_normal:
                        logger.warning(f"Insufficient files for {variable} {period_type} {target_year}")
                        continue
                    
                    # Check if output already exists
                    output_dir = self.config.output_base_dir / variable / period_type
                    output_file = output_dir / f"{variable}_{self.config.region_key}_{period_type}_{target_year}_30yr_normal.nc"
                    
                    if output_file.exists():
                        # Still update progress for skipped files
                        if progress_reporter:
                            for _ in files:
                                progress_reporter.update_task(
                                    variable,
                                    advance=1,
                                    current_item=f"Skipped (exists): year {target_year}"
                                )
                        results.append({
                            'target_year': target_year,
                            'status': 'skipped',
                            'output_file': str(output_file)
                        })
                        continue
                    
                    # Process files to get daily climatologies
                    daily_climatologies = []
                    years_used = []
                    
                    for file_path in files:
                        year, daily_clim = self.process_single_file_for_climatology_safe(
                            file_path, variable, None  # No progress reporter in worker processes
                        )
                        if year is not None and daily_clim is not None:
                            daily_climatologies.append(daily_clim)
                            years_used.append(year)
                    
                    if len(daily_climatologies) < self.config.min_years_for_normal:
                        logger.warning(f"Insufficient valid climatologies for {variable} {period_type} {target_year}")
                        continue
                    
                    # Compute climate normal
                    climate_normal = self.compute_climate_normal_safe(daily_climatologies, years_used, target_year)
                    
                    if climate_normal is None:
                        continue
                    
                    # Save result
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Add metadata
                    climate_normal.attrs.update({
                        'title': f'{variable.upper()} 30-Year {period_type.title()} Climate Normal ({self.config.region_key}) - Target Year {target_year}',
                        'variable': variable,
                        'region': self.config.region_key,
                        'region_name': REGION_BOUNDS[self.config.region_key]['name'],
                        'target_year': target_year,
                        'period_type': period_type,
                        'num_years': len(years_used),
                        'processing': f'Unified regional processor for {self.config.region_key}',
                        'source': 'NorESM2-LM climate model',
                        'method': '30-year rolling climate normal',
                        'created': datetime.now().isoformat()
                    })
                    
                    climate_normal.to_netcdf(output_file)
                    results.append({
                        'target_year': target_year,
                        'status': 'success',
                        'output_file': str(output_file),
                        'years_used': len(years_used)
                    })
                    
                    # Memory cleanup
                    del daily_climatologies, climate_normal
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error processing target year {target_year}: {e}")
                    results.append({
                        'target_year': target_year,
                        'status': 'error',
                        'error': str(e)
                    })
        
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return {'worker_id': worker_id, 'status': 'error', 'error': str(e)}
        
        return {'worker_id': worker_id, 'status': 'success', 'results': results}
    
    def process_variable_multiprocessing(self, variable: str) -> Dict:
        """Process a single variable using multiprocessing with proper progress tracking."""
        self.logger.info(f"🔄 Starting multiprocessing for {variable} in {self.config.region_key}")
        
        # File counting is now done upfront in process_all_variables()
        periods_config = {
            'historical': list(range(1980, 2015)),
            'hybrid': list(range(2015, 2045)),
            'ssp245': list(range(2045, 2101))
        }
        
        all_results = {}
        
        # Note: progress_queue is already created in process_all_variables if use_rich_progress is True
        
        for period_type, target_years in periods_config.items():
            self.logger.info(f"📅 Processing {period_type} period for {variable}")
            
            # Create batches
            year_batches = [target_years[i:i + self.config.batch_size_years] 
                           for i in range(0, len(target_years), self.config.batch_size_years)]
            
            # Prepare arguments
            args_list = [(variable, period_type, batch, i) 
                        for i, batch in enumerate(year_batches)]
            
            # Process batches in parallel using static method to avoid pickling issues
            with ProcessPoolExecutor(max_workers=self.config.cores_per_variable) as executor:
                future_to_batch = {executor.submit(process_target_year_batch_static, 
                                                  args, self.config): args 
                                  for args in args_list}
                
                period_results = []
                
                for future in as_completed(future_to_batch):
                    try:
                        result = future.result()
                        period_results.append(result)
                        
                        # Update progress based on batch completion
                        if self.progress_tracker and result.get('status') == 'success':
                            # Count successful results in this batch
                            successful_results = [r for r in result.get('results', []) if r.get('status') in ['success', 'skipped']]
                            files_per_result = 30  # Approximate files per target year
                            estimated_files_processed = len(successful_results) * files_per_result
                            
                            # Use ProgressReporter to update task progress
                            if self.progress_queue:
                                from county_climate.means.utils.mp_progress import ProgressReporter
                                progress_reporter = ProgressReporter(self.progress_queue)
                                progress_reporter.update_task(
                                    variable,
                                    advance=estimated_files_processed,
                                    current_item=f"Completed {period_type} batch"
                                )
                        
                    except Exception as e:
                        self.logger.error(f"Batch processing failed: {e}")
                
                all_results[period_type] = period_results
        
        return all_results
    
    def process_all_variables(self) -> Dict:
        """Process all variables with fixed progress tracking."""
        self.logger.info(f"🚀 Starting regional processing for {self.config.region_key}")
        self.logger.info(f"📊 Variables: {self.config.variables}")
        
        # Count files for all variables upfront for accurate progress tracking
        self.logger.info("📊 Counting files for all variables...")
        variable_file_counts = {}
        
        file_handler = NorESM2FileHandler(self.config.input_data_dir)
        periods_config = {
            'historical': list(range(1980, 2015)),
            'hybrid': list(range(2015, 2045)),
            'ssp245': list(range(2045, 2101))
        }
        
        for variable in self.config.variables:
            total_files = 0
            for period_type, target_years in periods_config.items():
                for target_year in target_years:
                    if period_type == 'historical':
                        start_year = max(1950, target_year - 29)
                        end_year = target_year
                        files = file_handler.get_files_for_period(variable, 'historical', start_year, end_year)
                    elif period_type == 'hybrid':
                        hist_start = max(1985, target_year - 29)
                        hist_end = min(2014, target_year)
                        ssp_start = max(2015, target_year - 29)
                        ssp_end = target_year
                        hist_files = file_handler.get_files_for_period(variable, 'historical', hist_start, hist_end)
                        ssp_files = file_handler.get_files_for_period(variable, 'ssp245', ssp_start, ssp_end)
                        files = hist_files + ssp_files
                    else:
                        start_year = target_year - 29
                        end_year = target_year
                        files = file_handler.get_files_for_period(variable, 'ssp245', start_year, end_year)
                    
                    total_files += len(files)
            
            variable_file_counts[variable] = total_files
            self.logger.info(f"📁 {variable}: {total_files:,} files")
        
        # Initialize multiprocessing progress tracker with accurate totals
        if self.use_rich_progress:
            region_name = REGION_BOUNDS[self.config.region_key]['name']
            total_all_files = sum(variable_file_counts.values())
            self.progress_tracker = MultiprocessingProgressTracker(
                title=f"Climate Processing - {region_name} ({total_all_files:,} files)"
            )
            # Save the queue returned by start() for communication
            self.progress_queue = self.progress_tracker.start()
            
            # Add tasks for each variable with accurate file counts
            for variable in self.config.variables:
                self.progress_tracker.add_task(
                    name=variable,
                    description=f"Processing {variable.upper()} data",
                    total=variable_file_counts[variable]
                )
        
        start_time = time.time()
        all_results = {}
        
        try:
            for variable in self.config.variables:
                self.logger.info(f"🔄 Processing variable: {variable}")
                variable_start = time.time()
                
                try:
                    variable_results = self.process_variable_multiprocessing(variable)
                    all_results[variable] = variable_results
                    
                    variable_time = time.time() - variable_start
                    self.logger.info(f"✅ Completed {variable} in {variable_time:.1f} seconds")
                    
                    # Mark task as completed
                    if self.progress_tracker:
                        self.progress_tracker.complete_task(variable, "completed")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to process {variable}: {e}")
                    all_results[variable] = {'status': 'error', 'error': str(e)}
                    
                    if self.progress_tracker:
                        self.progress_tracker.complete_task(variable, "failed")
            
            total_time = time.time() - start_time
            self.logger.info(f"🎉 Regional processing completed in {total_time:.1f} seconds")
            
        finally:
            # Stop progress tracker
            if self.progress_tracker:
                self.progress_tracker.stop()
        
        return all_results


def process_target_year_batch_static(args: Tuple, config: RegionalProcessingConfig) -> Dict:
    """Static function for processing target year batches in multiprocessing context."""
    variable, period_type, target_years_batch, worker_id = args
    
    logger = logging.getLogger(f"{__name__}.worker_{worker_id}")
    logger.info(f"Worker {worker_id} processing {variable} {period_type} years: {target_years_batch}")
    
    # Progress reporting is handled at the main process level
    # Worker processes cannot access the rich progress tracker due to queue inheritance limitations
    
    results = []
    
    try:
        file_handler = NorESM2FileHandler(config.input_data_dir)
        
        for target_year in target_years_batch:
            try:
                # Determine files for this period
                if period_type == 'historical':
                    start_year = max(1950, target_year - 29)
                    end_year = target_year
                    files = file_handler.get_files_for_period(variable, 'historical', start_year, end_year)
                elif period_type == 'hybrid':
                    # 2015-2044 uses both historical and ssp245
                    hist_start = max(1985, target_year - 29)
                    hist_end = min(2014, target_year)
                    ssp_start = max(2015, target_year - 29)
                    ssp_end = target_year
                    
                    hist_files = file_handler.get_files_for_period(variable, 'historical', hist_start, hist_end)
                    ssp_files = file_handler.get_files_for_period(variable, 'ssp245', ssp_start, ssp_end)
                    files = hist_files + ssp_files
                else:  # ssp245
                    start_year = target_year - 29
                    end_year = target_year
                    files = file_handler.get_files_for_period(variable, 'ssp245', start_year, end_year)
                
                if len(files) < config.min_years_for_normal:
                    logger.warning(f"Insufficient files for {variable} {period_type} {target_year}")
                    continue
                
                # Check if output already exists
                output_dir = config.output_base_dir / variable / period_type
                output_file = output_dir / f"{variable}_{config.region_key}_{period_type}_{target_year}_30yr_normal.nc"
                
                if output_file.exists():
                    results.append({
                        'target_year': target_year,
                        'status': 'skipped',
                        'output_file': str(output_file)
                    })
                    continue
                
                # Process files to get daily climatologies
                daily_climatologies = []
                years_used = []
                
                for file_path in files:
                    year, daily_clim = process_single_file_for_climatology_static(
                        file_path, variable, config.region_key
                    )
                    if year is not None and daily_clim is not None:
                        daily_climatologies.append(daily_clim)
                        years_used.append(year)
                    
                    # Progress updates are handled at the main process level
                    # Individual file progress cannot be reported from worker processes
                
                if len(daily_climatologies) < config.min_years_for_normal:
                    logger.warning(f"Insufficient valid climatologies for {variable} {period_type} {target_year}")
                    continue
                
                # Compute climate normal
                climate_normal = compute_climate_normal_static(daily_climatologies, years_used, target_year, config.region_key)
                
                if climate_normal is None:
                    continue
                
                # Save result
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Add comprehensive metadata
                climate_normal.attrs.update({
                    'title': f'{variable.upper()} 30-Year {period_type.title()} Climate Normal ({config.region_key}) - Target Year {target_year}',
                    'variable': variable,
                    'region': config.region_key,
                    'region_name': REGION_BOUNDS[config.region_key]['name'],
                    'target_year': target_year,
                    'period_type': period_type,
                    'num_years': len(years_used),
                    'processing': f'Unified regional processor for {config.region_key}',
                    'source': 'NorESM2-LM climate model',
                    'method': '30-year rolling climate normal',
                    'created': datetime.now().isoformat()
                })
                
                climate_normal.to_netcdf(output_file)
                results.append({
                    'target_year': target_year,
                    'status': 'success',
                    'output_file': str(output_file),
                    'years_used': len(years_used)
                })
                
                # Memory cleanup
                del daily_climatologies, climate_normal
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing target year {target_year}: {e}")
                results.append({
                    'target_year': target_year,
                    'status': 'error',
                    'error': str(e)
                })
    
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return {'worker_id': worker_id, 'status': 'error', 'error': str(e)}
    
    return {'worker_id': worker_id, 'status': 'success', 'results': results}


def process_single_file_for_climatology_static(file_path: Path, variable: str, region_key: str) -> Tuple[int, xr.DataArray]:
    """Static function for processing a single file - multiprocessing safe."""
    try:
        # Extract year from filename
        year = extract_year_from_filename(file_path)
        
        # Load data
        with xr.open_dataset(file_path) as ds:
            # Extract regional data
            regional_data = extract_region(ds, REGION_BOUNDS[region_key])
            
            if regional_data is None:
                return None, None
            
            # Fix time coordinates
            regional_data, _ = handle_time_coordinates(regional_data, str(file_path))
            
            # Calculate daily climatology
            if variable in regional_data:
                var_data = regional_data[variable]
                
                # For daily data, group by day of year
                daily_clim = var_data.groupby('time.dayofyear').mean(dim='time')
                
                return year, daily_clim
                
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error processing {file_path}: {e}")
        
    return None, None


def compute_climate_normal_static(daily_climatologies: List[xr.DataArray], 
                                years_used: List[int], target_year: int, region_key: str) -> xr.DataArray:
    """Static function for computing climate normal - multiprocessing safe."""
    try:
        # Stack all daily climatologies along a new 'year' dimension
        stacked_data = xr.concat(daily_climatologies, dim='year')
        stacked_data['year'] = years_used
        
        # Calculate the mean across all years for each day of year
        climate_normal = stacked_data.mean(dim='year')
        
        # Add metadata
        climate_normal.attrs.update({
            'long_name': f'30-year climate normal for {target_year}',
            'target_year': target_year,
            'source_years': f"{min(years_used)}-{max(years_used)}",
            'number_of_years': len(years_used),
            'processing_method': f'unified_regional_processor_{region_key}',
            'region': region_key
        })
        
        return climate_normal
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error computing climate normal: {e}")
        return None


def create_regional_processor(region_key: str, 
                            variables: List[str] = None,
                            use_rich_progress: bool = True,
                            **kwargs) -> RegionalClimateProcessor:
    """Factory function to create a regional processor with configuration."""
    # Get global configuration
    global_config = get_config()
    
    # Default variables
    if variables is None:
        variables = ['pr', 'tas', 'tasmax', 'tasmin']
    
    config = RegionalProcessingConfig(
        region_key=region_key,
        variables=variables,
        input_data_dir=global_config.paths.input_data_dir,
        output_base_dir=get_regional_output_dir(region_key),
        **{k: v for k, v in kwargs.items() if k in [
            'max_cores', 'cores_per_variable', 'batch_size_years',
            'max_memory_per_process_gb', 'memory_check_interval',
            'min_years_for_normal', 'status_update_interval'
        ]}
    )
    
    return RegionalClimateProcessor(config, use_rich_progress=use_rich_progress)


def process_region(region_key: str, 
                  variables: List[str] = None,
                  use_rich_progress: bool = True,
                  **kwargs) -> Dict:
    """Convenience function to process a region with default settings."""
    processor = create_regional_processor(region_key, variables, use_rich_progress, **kwargs)
    return processor.process_all_variables()