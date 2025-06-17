#!/usr/bin/env python3
"""
Flexible Regional Climate Normals Processing Pipeline

Refactored version that supports multiple climate models and flexible scenarios.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Union
import psutil
import traceback
import json
from datetime import datetime
from dataclasses import dataclass
import threading

# Import our modules
from county_climate.means.utils.flexible_io_util import FlexibleClimateFileHandler
from county_climate.means.core.regions import REGION_BOUNDS, extract_region, validate_region_bounds
from county_climate.means.utils.time_util import handle_time_coordinates, extract_year_from_filename
from county_climate.means.config import get_config, get_regional_output_dir
from county_climate.means.utils.rich_progress import RichProgressTracker
from county_climate.means.utils.mp_progress import MultiprocessingProgressTracker, ProgressReporter
from county_climate.means.core.flexible_config import (
    FlexibleRegionalProcessingConfig, 
    ScenarioProcessingConfig
)


class FlexibleRegionalClimateProcessor:
    """Regional climate processor with flexible scenario support."""
    
    def __init__(self, config: FlexibleRegionalProcessingConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize file handler with model support
        self.file_handler = FlexibleClimateFileHandler(
            self.config.input_data_dir,
            model_id=self.config.model_id
        )
        
        # Will be initialized when processing starts
        self.progress_tracker = None
        self.progress_queue = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.config.main_log_file)
            ]
        )
        return logging.getLogger(__name__)
    
    def process_single_file_for_climatology_safe(self, file_path: Path, variable: str, 
                                                progress_reporter: Optional[ProgressReporter] = None) -> Tuple[int, xr.DataArray]:
        """Process a single file and extract climatology."""
        try:
            # Extract year from filename
            year = self.file_handler.extract_year_from_filename(str(file_path))
            
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
                            current_item=f"Processed {Path(file_path).name}"
                        )
                    
                    return year, daily_clim
                    
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            if progress_reporter:
                progress_reporter.update_task(
                    variable,
                    advance=1,
                    current_item=f"Failed: {Path(file_path).name}"
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
                'processing_method': f'flexible_regional_processor_{self.config.region_key}',
                'region': self.config.region_key,
                'model': self.config.model_id
            })
            
            return climate_normal
            
        except Exception as e:
            self.logger.error(f"Error computing climate normal: {e}")
            return None
    
    def process_target_year_batch_with_progress(self, args: Tuple) -> Dict:
        """Process a batch of target years for a variable with flexible scenarios."""
        variable, scenario_config, target_years_batch, worker_id = args
        
        logger = logging.getLogger(f"{__name__}.worker_{worker_id}")
        logger.info(f"Worker {worker_id} processing {variable} {scenario_config.scenario_name} years: {target_years_batch}")
        
        results = []
        
        try:
            # Re-initialize file handler in worker process
            file_handler = FlexibleClimateFileHandler(
                self.config.input_data_dir,
                model_id=self.config.model_id
            )
            
            for target_year in target_years_batch:
                try:
                    # Determine output period type for directory structure
                    output_period_type = self.config.get_output_period_type(scenario_config)
                    
                    # Check if output already exists
                    output_dir = self.config.output_base_dir / variable / output_period_type
                    output_file = output_dir / f"{variable}_{self.config.region_key}_{output_period_type}_{target_year}_30yr_normal.nc"
                    
                    if output_file.exists():
                        results.append({
                            'target_year': target_year,
                            'status': 'skipped',
                            'output_file': str(output_file)
                        })
                        continue
                    
                    # Get files for this period
                    if scenario_config.process_as_hybrid:
                        # Hybrid period - use both historical and projection data
                        files, scenario_counts = file_handler.get_hybrid_files_for_period(
                            variable, 
                            target_year,
                            projection_scenario=scenario_config.scenario_name,
                            window_years=self.config.climate_normal_window
                        )
                    else:
                        # Single scenario period
                        start_year = target_year - self.config.climate_normal_window + 1
                        end_year = target_year
                        files = file_handler.get_files_for_period(
                            variable, 
                            scenario_config.scenario_name, 
                            start_year, 
                            end_year
                        )
                        scenario_counts = {scenario_config.scenario_name: len(files)}
                    
                    if len(files) < self.config.min_years_for_normal:
                        logger.warning(f"Insufficient files for {variable} {scenario_config.scenario_name} {target_year}: {len(files)}")
                        continue
                    
                    # Process files to get daily climatologies
                    daily_climatologies = []
                    years_used = []
                    
                    for file_path in files:
                        year, daily_clim = self.process_single_file_for_climatology_safe(
                            file_path, variable, None
                        )
                        if year is not None and daily_clim is not None:
                            daily_climatologies.append(daily_clim)
                            years_used.append(year)
                    
                    if len(daily_climatologies) < self.config.min_years_for_normal:
                        logger.warning(f"Insufficient valid climatologies for {variable} {scenario_config.scenario_name} {target_year}")
                        continue
                    
                    # Compute climate normal
                    climate_normal = self.compute_climate_normal_safe(daily_climatologies, years_used, target_year)
                    
                    if climate_normal is None:
                        continue
                    
                    # Save result
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Add comprehensive metadata
                    climate_normal.attrs.update({
                        'title': f'{variable.upper()} 30-Year Climate Normal ({self.config.region_key}) - Target Year {target_year}',
                        'variable': variable,
                        'region': self.config.region_key,
                        'region_name': REGION_BOUNDS[self.config.region_key]['name'],
                        'target_year': target_year,
                        'period_type': output_period_type,
                        'scenario': scenario_config.scenario_name,
                        'is_hybrid': int(scenario_config.process_as_hybrid),  # Convert bool to int for NetCDF
                        'scenario_counts': json.dumps(scenario_counts),
                        'num_years': len(years_used),
                        'model': self.config.model_id,
                        'processing': f'Flexible regional processor for {self.config.region_key}',
                        'source': f'{self.config.model_id} climate model',
                        'method': '30-year rolling climate normal',
                        'created': datetime.now().isoformat()
                    })
                    
                    climate_normal.to_netcdf(output_file)
                    results.append({
                        'target_year': target_year,
                        'status': 'success',
                        'output_file': str(output_file),
                        'years_used': len(years_used),
                        'scenario_counts': scenario_counts
                    })
                    
                    logger.info(f"Created: {output_file}")
                    
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
        """Process a single variable using multiprocessing with flexible scenarios."""
        self.logger.info(f"🔄 Starting multiprocessing for {variable} in {self.config.region_key}")
        
        all_results = {}
        
        # Process each configured scenario
        for scenario_config in self.config.scenarios_to_process:
            scenario_name = scenario_config.scenario_name
            target_years = scenario_config.get_years()
            
            self.logger.info(f"📅 Processing {scenario_name} ({len(target_years)} years) for {variable}")
            
            # Create batches
            year_batches = [target_years[i:i + self.config.batch_size_years] 
                           for i in range(0, len(target_years), self.config.batch_size_years)]
            
            # Prepare arguments
            args_list = [(variable, scenario_config, batch, i) 
                        for i, batch in enumerate(year_batches)]
            
            # Process batches in parallel
            with ProcessPoolExecutor(max_workers=self.config.cores_per_variable) as executor:
                future_to_batch = {
                    executor.submit(
                        process_target_year_batch_static_v3,
                        args, 
                        self.config
                    ): args for args in args_list
                }
                
                scenario_results = []
                
                for future in as_completed(future_to_batch):
                    try:
                        result = future.result()
                        scenario_results.append(result)
                        
                        # Update progress if available
                        if self.progress_tracker and result.get('status') == 'success':
                            successful_results = [r for r in result.get('results', []) 
                                                if r.get('status') in ['success', 'skipped']]
                            files_per_result = 30  # Approximate
                            estimated_files_processed = len(successful_results) * files_per_result
                            
                            if self.progress_queue:
                                from county_climate.means.utils.mp_progress import ProgressReporter
                                progress_reporter = ProgressReporter(self.progress_queue)
                                progress_reporter.update_task(
                                    variable,
                                    advance=estimated_files_processed,
                                    current_item=f"Completed {scenario_name} batch"
                                )
                        
                    except Exception as e:
                        self.logger.error(f"Batch processing failed: {e}")
                
                all_results[scenario_name] = scenario_results
        
        return all_results
    
    def process_all_variables(self) -> Dict:
        """Process all variables with flexible scenario support."""
        self.logger.info(f"🚀 Starting regional processing for {self.config.region_key}")
        self.logger.info(f"📊 Variables: {self.config.variables}")
        self.logger.info(f"🌍 Model: {self.config.model_id}")
        self.logger.info(f"📅 Scenarios: {[s.scenario_name for s in self.config.scenarios_to_process]}")
        
        # Count files for progress tracking
        self.logger.info("📊 Counting files for all variables...")
        variable_file_counts = self._count_files_for_progress()
        
        # Initialize progress tracking if enabled
        if self.config.enable_rich_progress:
            region_name = REGION_BOUNDS[self.config.region_key]['name']
            total_all_files = sum(variable_file_counts.values())
            self.progress_tracker = MultiprocessingProgressTracker(
                title=f"Climate Processing - {region_name} ({total_all_files:,} files)"
            )
            self.progress_queue = self.progress_tracker.start()
            
            # Add tasks for each variable
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
            if self.progress_tracker:
                self.progress_tracker.stop()
        
        return all_results
    
    def _count_files_for_progress(self) -> Dict[str, int]:
        """Count files for accurate progress tracking."""
        variable_file_counts = {}
        
        for variable in self.config.variables:
            total_files = 0
            
            for scenario_config in self.config.scenarios_to_process:
                target_years = scenario_config.get_years()
                
                for target_year in target_years:
                    if scenario_config.process_as_hybrid:
                        # Estimate files for hybrid period
                        files, _ = self.file_handler.get_hybrid_files_for_period(
                            variable, 
                            target_year,
                            projection_scenario=scenario_config.scenario_name,
                            window_years=self.config.climate_normal_window
                        )
                        total_files += len(files)
                    else:
                        # Count files for single scenario
                        start_year = target_year - self.config.climate_normal_window + 1
                        end_year = target_year
                        files = self.file_handler.get_files_for_period(
                            variable,
                            scenario_config.scenario_name,
                            start_year,
                            end_year
                        )
                        total_files += len(files)
            
            variable_file_counts[variable] = total_files
            self.logger.info(f"📁 {variable}: {total_files:,} files")
        
        return variable_file_counts


def process_target_year_batch_static_v3(args: Tuple, config: FlexibleRegionalProcessingConfig) -> Dict:
    """Static function for processing target year batches with flexible scenarios."""
    variable, scenario_config, target_years_batch, worker_id = args
    
    # Create processor instance in worker
    processor = FlexibleRegionalClimateProcessor(config)
    
    # Process batch
    return processor.process_target_year_batch_with_progress(
        (variable, scenario_config, target_years_batch, worker_id)
    )


def create_flexible_regional_processor(region_key: str, 
                                     variables: List[str] = None,
                                     scenario: str = None,
                                     model_id: str = "NorESM2-LM",
                                     **kwargs) -> FlexibleRegionalClimateProcessor:
    """
    Factory function to create a flexible regional processor.
    
    Args:
        region_key: Region to process
        variables: Variables to process (default: all)
        scenario: Scenario to process (e.g., 'ssp585', 'ssp245')
        model_id: Climate model ID
        **kwargs: Additional configuration parameters
    """
    # Get global configuration
    global_config = get_config()
    
    # Default variables
    if variables is None:
        variables = ['pr', 'tas', 'tasmax', 'tasmin']
    
    # Create flexible configuration
    if scenario:
        config = FlexibleRegionalProcessingConfig.for_scenario(
            region_key=region_key,
            variables=variables,
            input_data_dir=global_config.paths.input_data_dir,
            output_base_dir=get_regional_output_dir(region_key),
            scenario=scenario,
            model_id=model_id,
            **{k: v for k, v in kwargs.items() if k in [
                'max_cores', 'cores_per_variable', 'batch_size_years',
                'max_memory_per_process_gb', 'memory_check_interval',
                'min_years_for_normal', 'status_update_interval',
                'enable_rich_progress'
            ]}
        )
    else:
        # Use default configuration
        config = FlexibleRegionalProcessingConfig(
            region_key=region_key,
            variables=variables,
            input_data_dir=global_config.paths.input_data_dir,
            output_base_dir=get_regional_output_dir(region_key),
            model_id=model_id,
            **kwargs
        )
    
    return FlexibleRegionalClimateProcessor(config)