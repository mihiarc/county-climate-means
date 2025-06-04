#!/usr/bin/env python3
"""
Parallel Climate Data Processing Pipeline

High-performance multiprocessing implementation for climate data processing.
Designed to work with existing climate processing functions while utilizing
multiple CPU cores for significant speedup.

Key features:
- Optimal worker count based on system resources (6 workers optimal)
- Robust error handling and retry logic
- Memory monitoring and management
- Progress tracking and logging
- Process-based parallelism to avoid NetCDF threading issues
"""

import logging
import time
import psutil
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import xarray as xr
from dataclasses import dataclass

from ..core.multiprocessing_engine import ClimateMultiprocessor
from ..config import get_default_config
from ..utils.io import NorESM2FileHandler, open_dataset_safely
from ..utils.regions import REGION_BOUNDS, extract_region
from ..utils.time_handling import handle_time_coordinates, extract_year_from_filename
from ..core.climate_engine import ClimateEngine

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for multiprocessing climate data processing."""
    max_workers: int = 6  # Optimal worker count based on testing
    memory_per_worker_gb: float = 4.0
    timeout_per_file: int = 300  # 5 minutes
    max_retries: int = 2
    chunk_size: int = 10  # Files to process per batch
    progress_interval: int = 5  # Report progress every N files


class ParallelPipeline:
    """
    High-performance parallel processing pipeline for climate data.
    
    This pipeline uses multiple levels of parallelism:
    - Level 1: Process each variable in parallel
    - Level 2: Within each variable, process target years in parallel batches
    - Level 3: Each target year processing uses optimized I/O
    """
    
    def __init__(self, config=None):
        """Initialize the parallel pipeline."""
        self.config = config or get_default_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize multiprocessing engine
        self.multiprocessor = ClimateMultiprocessor(
            self.config.input_data_dir,
            self._create_processing_config()
        )
        
        self.logger.info(f"Initialized parallel pipeline with {self.config.max_workers} workers")
    
    def _create_processing_config(self) -> ProcessingConfig:
        """Create processing configuration from main config."""
        return ProcessingConfig(
            max_workers=self.config.max_workers,
            memory_per_worker_gb=self.config.max_memory_per_process_gb,
            timeout_per_file=self.config.timeout_per_file,
            max_retries=self.config.max_retries,
            progress_interval=self.config.progress_interval
        )
    
    def run(self, variables: Optional[List[str]] = None, 
            regions: Optional[List[str]] = None,
            scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the parallel processing pipeline.
        
        Args:
            variables: List of climate variables to process
            regions: List of regions to process
            scenarios: List of scenarios to process
            
        Returns:
            Dictionary containing processing results and statistics
        """
        variables = variables or self.config.variables
        regions = regions or self.config.regions  
        scenarios = scenarios or self.config.scenarios
        
        self.logger.info("🚀 Starting Parallel Climate Processing Pipeline")
        self.logger.info(f"Variables: {variables}")
        self.logger.info(f"Regions: {regions}")
        self.logger.info(f"Scenarios: {scenarios}")
        self.logger.info(f"Max workers: {self.config.max_workers}")
        
        start_time = time.time()
        
        # Process each variable in parallel
        variable_results = []
        
        with ProcessPoolExecutor(max_workers=len(variables)) as executor:
            # Submit variable processing jobs
            future_to_variable = {
                executor.submit(
                    self._process_variable_parallel,
                    variable, regions[0] if regions else 'CONUS'
                ): variable
                for variable in variables
            }
            
            # Collect results
            for future in as_completed(future_to_variable):
                variable = future_to_variable[future]
                try:
                    result = future.result()
                    variable_results.append(result)
                    self.logger.info(f"✅ {variable.upper()} processing completed!")
                    
                except Exception as e:
                    self.logger.error(f"❌ {variable.upper()} processing failed: {e}")
                    variable_results.append({
                        'variable': variable,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Compile final results
        results = {
            'status': 'completed',
            'duration_seconds': total_duration,
            'variables_processed': len([r for r in variable_results if r.get('status') == 'completed']),
            'variables_failed': len([r for r in variable_results if r.get('status') == 'failed']),
            'variable_results': variable_results
        }
        
        self.logger.info(f"🎉 Parallel pipeline completed in {total_duration:.1f} seconds")
        self.logger.info(f"Successful: {results['variables_processed']}/{len(variables)} variables")
        
        return results
    
    def _process_variable_parallel(self, variable: str, region_key: str) -> Dict[str, Any]:
        """Process a single variable using parallel processing."""
        self.logger.info(f"🔄 Starting parallel processing for {variable.upper()}")
        
        try:
            # Get file handler
            file_handler = NorESM2FileHandler(self.config.input_data_dir)
            
            # Process historical data
            historical_result = self._process_historical_parallel(
                variable, region_key, file_handler
            )
            
            # Process hybrid data  
            hybrid_result = self._process_hybrid_parallel(
                variable, region_key, file_handler
            )
            
            # Process future data
            future_result = self._process_future_parallel(
                variable, region_key, file_handler
            )
            
            return {
                'variable': variable,
                'status': 'completed',
                'historical': historical_result,
                'hybrid': hybrid_result,
                'future': future_result
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {variable}: {e}")
            return {
                'variable': variable,
                'status': 'failed',
                'error': str(e)
            }
    
    def _process_historical_parallel(self, variable: str, region_key: str, 
                                   file_handler) -> Dict[str, Any]:
        """Process historical data using parallel processing."""
        self.logger.info(f"Processing historical {variable} data in parallel")
        
        target_years = list(range(self.config.historical_start_year, 
                                self.config.historical_end_year + 1))
        
        return self._process_years_parallel(
            variable, region_key, file_handler, target_years, 'historical'
        )
    
    def _process_hybrid_parallel(self, variable: str, region_key: str,
                               file_handler) -> Dict[str, Any]:
        """Process hybrid data using parallel processing."""
        self.logger.info(f"Processing hybrid {variable} data in parallel")
        
        target_years = list(range(self.config.hybrid_start_year,
                                self.config.hybrid_end_year + 1))
        
        return self._process_years_parallel(
            variable, region_key, file_handler, target_years, 'hybrid'
        )
    
    def _process_future_parallel(self, variable: str, region_key: str,
                               file_handler) -> Dict[str, Any]:
        """Process future data using parallel processing."""
        self.logger.info(f"Processing future {variable} data in parallel")
        
        # Get available future years from data
        availability = file_handler.validate_data_availability()
        if variable not in availability or 'ssp245' not in availability[variable]:
            return {'status': 'skipped', 'reason': 'No future data available'}
        
        _, max_year = availability[variable]['ssp245']
        target_years = list(range(self.config.future_start_year, 
                                min(max_year + 1, self.config.max_future_year + 1)))
        
        return self._process_years_parallel(
            variable, region_key, file_handler, target_years, 'ssp245'
        )
    
    def _process_years_parallel(self, variable: str, region_key: str, 
                              file_handler, target_years: List[int],
                              period_type: str) -> Dict[str, Any]:
        """Process a list of target years in parallel batches."""
        if not target_years:
            return {'status': 'skipped', 'reason': 'No target years'}
        
        self.logger.info(f"Processing {len(target_years)} {period_type} years for {variable}")
        
        # Create work batches
        batch_size = self.config.batch_size_years
        work_batches = []
        
        for i in range(0, len(target_years), batch_size):
            batch_years = target_years[i:i + batch_size]
            work_batches.append((
                variable, region_key, period_type, batch_years,
                self.config.input_data_dir, self.config.output_base_dir
            ))
        
        self.logger.info(f"Created {len(work_batches)} work batches")
        
        # Process batches in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.config.cores_per_variable) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(process_year_batch_worker, batch): batch
                for batch in work_batches
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Batch completed: {len(result.get('processed', []))} years")
                except Exception as e:
                    self.logger.error(f"Batch failed: {e}")
                    results.append({'status': 'failed', 'error': str(e)})
        
        # Summarize results
        total_processed = sum(len(r.get('processed', [])) for r in results)
        total_failed = sum(len(r.get('failed', [])) for r in results)
        
        return {
            'status': 'completed',
            'period_type': period_type,
            'total_years': len(target_years),
            'processed': total_processed,
            'failed': total_failed,
            'batch_results': results
        }
    
    def process_files_parallel(self, file_list: List[str], output_path: str,
                             variable: str = 'pr', region: str = 'CONUS') -> Dict[str, Any]:
        """Process multiple files in parallel using the multiprocessor."""
        return self.multiprocessor.process_files_parallel(
            file_list, output_path, variable, region
        )
    
    def benchmark_speedup(self, num_files: int = 10) -> Dict[str, Any]:
        """Benchmark parallel vs sequential processing speedup."""
        return self.multiprocessor.benchmark_multiprocessing_speedup(
            self.config.input_data_dir, num_files
        )


def process_year_batch_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function to process a batch of target years.
    Must be at module level for multiprocessing.
    """
    (variable, region_key, period_type, target_years, 
     input_data_dir, output_base_dir) = args
    
    logger = logging.getLogger(f'worker_batch')
    logger.info(f"Processing batch: {variable} {period_type} years {target_years}")
    
    try:
        # Initialize components within worker
        file_handler = NorESM2FileHandler(input_data_dir)
        
        processed_years = []
        failed_years = []
        
        # Process each year in the batch
        for target_year in target_years:
            try:
                result = process_single_year_worker(
                    variable, region_key, period_type, target_year,
                    file_handler, input_data_dir, output_base_dir
                )
                
                if result:
                    processed_years.append(target_year)
                    logger.info(f"✅ Completed {variable} {period_type} {target_year}")
                else:
                    failed_years.append(target_year)
                    logger.warning(f"❌ Failed {variable} {period_type} {target_year}")
                    
            except Exception as e:
                failed_years.append(target_year)
                logger.error(f"Error processing {target_year}: {e}")
        
        return {
            'variable': variable,
            'period_type': period_type,
            'processed': processed_years,
            'failed': failed_years,
            'status': 'completed'
        }
        
    except Exception as e:
        logger.error(f"Fatal error in batch worker: {e}")
        return {
            'variable': variable,
            'period_type': period_type,
            'processed': [],
            'failed': target_years,
            'status': 'failed',
            'error': str(e)
        }


def process_single_year_worker(variable: str, region_key: str, period_type: str,
                             target_year: int, file_handler, input_data_dir: str,
                             output_base_dir: str) -> bool:
    """Process a single target year using working legacy patterns."""
    try:
        # Get files for this target year
        if period_type == 'historical':
            start_year = target_year - 29
            end_year = target_year
            files = file_handler.get_files_for_period(variable, 'historical', start_year, end_year)
        elif period_type == 'hybrid':
            files, _ = file_handler.get_hybrid_files_for_period(variable, target_year, 30)
        elif period_type == 'ssp245':
            start_year = target_year - 29
            end_year = target_year
            files = file_handler.get_files_for_period(variable, 'ssp245', start_year, end_year)
        else:
            return False
        
        if len(files) < 25:  # Minimum years for climate normal
            return False
        
        # Process files to get daily climatologies using working pattern
        daily_climatologies = []
        years_used = []
        
        for file_path in files:
            year, daily_clim = process_single_file_for_climatology_safe(
                file_path, variable, region_key
            )
            if year is not None and daily_clim is not None:
                daily_climatologies.append(daily_clim)
                years_used.append(year)
        
        if len(daily_climatologies) < 25:
            return False
        
        # Compute climate normal using working computation
        data_arrays = [clim.values for clim in daily_climatologies]
        climate_normal = compute_climate_normal_safe(data_arrays, years_used, target_year)
        
        if climate_normal is None:
            return False
        
        # Save result
        output_dir = Path(output_base_dir) / variable / period_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{variable}_{region_key}_{period_type}_{target_year}_30yr_normal.nc"
        
        # Add metadata
        climate_normal.attrs.update({
            'title': f'{variable.upper()} 30-Year {period_type.title()} Climate Normal ({region_key}) - Target Year {target_year}',
            'variable': variable,
            'region': region_key,
            'target_year': target_year,
            'period_type': period_type,
            'num_years': len(years_used),
            'processing': 'Parallel processing (process-based)',
            'source': 'NorESM2-LM climate model',
            'method': '30-year rolling climate normal'
        })
        
        climate_normal.to_netcdf(output_file)
        return True
        
    except Exception as e:
        logger.error(f"Error processing {variable} {period_type} {target_year}: {e}")
        return False


def process_single_file_for_climatology_safe(file_path: str, variable_name: str, 
                                           region_key: str) -> Tuple[Optional[int], Optional[xr.DataArray]]:
    """
    Process a single file to extract daily climatology - multiprocessing safe version
    Based on working legacy implementation.
    """
    try:
        # Extract year from filename
        year = extract_year_from_filename(file_path)
        if year is None:
            return None, None
        
        # Open dataset with conservative settings and blacklist checking
        ds = open_dataset_safely(file_path)
        
        if ds is None:
            # File was either blacklisted or failed to open
            return None, None
        
        # Check if variable exists
        if variable_name not in ds.data_vars:
            ds.close()
            return None, None
        
        # Handle time coordinates
        ds, time_method = handle_time_coordinates(ds, file_path)
        
        # Extract region
        region_bounds = REGION_BOUNDS[region_key]
        region_ds = extract_region(ds, region_bounds)
        var = region_ds[variable_name]
        
        # Calculate daily climatology
        if 'dayofyear' in var.coords:
            daily_clim = var.groupby(var.dayofyear).mean(dim='time')
            result = daily_clim.compute()
            
            # Cleanup
            ds.close()
            del ds, region_ds, var, daily_clim
            gc.collect()
            
            return year, result
        else:
            ds.close()
            return None, None
            
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        # Add to blacklist if it's a persistent error
        from ..utils.io import add_to_blacklist
        error_msg = str(e)
        if any(keyword in error_msg.lower() for keyword in ['netcdf', 'hdf', 'errno -101', 'corrupted']):
            add_to_blacklist(file_path, error_msg)
        return None, None


def compute_climate_normal_safe(data_arrays: List[np.ndarray], years: List[int], 
                              target_year: int) -> Optional[xr.DataArray]:
    """Compute climate normal from multiple data arrays - safe multiprocessing version."""
    logger.debug(f"Computing climate normal for target year {target_year} using {len(data_arrays)} years of data")
    
    if not data_arrays:
        logger.warning(f"No data arrays provided for target year {target_year}")
        return None
    
    try:
        # Assume all arrays have the same shape and are daily climatologies
        # Stack arrays and compute mean
        stacked_data = np.stack(data_arrays, axis=0)
        mean_data = np.mean(stacked_data, axis=0)
        
        # Create DataArray with proper coordinates
        if mean_data.ndim == 3:  # (dayofyear, lat, lon)
            coords = {
                'dayofyear': np.arange(1, mean_data.shape[0] + 1),
                'lat': np.arange(mean_data.shape[1]),
                'lon': np.arange(mean_data.shape[2])
            }
            dims = ['dayofyear', 'lat', 'lon']
        else:
            # Fallback for other dimensions
            coords = {f'dim_{i}': np.arange(mean_data.shape[i]) for i in range(mean_data.ndim)}
            dims = [f'dim_{i}' for i in range(mean_data.ndim)]
        
        climate_normal = xr.DataArray(
            mean_data,
            coords=coords,
            dims=dims,
            attrs={
                'long_name': f'30-year climate normal for {target_year}',
                'target_year': target_year,
                'source_years': f"{min(years)}-{max(years)}",
                'number_of_years': len(years),
                'processing_method': 'multiprocessing_safe'
            }
        )
        
        logger.debug(f"Successfully computed climate normal for {target_year}")
        return climate_normal
        
    except Exception as e:
        logger.error(f"Error computing climate normal for {target_year}: {e}")
        return None 