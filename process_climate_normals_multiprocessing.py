#!/usr/bin/env python3
"""
Multiprocessing Climate Normals Processing Pipeline

Optimized for high-performance processing using:
- 40 CPU cores 
- 80GB RAM available
- Process-based parallelism to avoid NetCDF threading issues
- Strategic work distribution across variables and target years

Strategy:
- Level 1: Process each variable in parallel (4 processes)
- Level 2: Within each variable, process target years in parallel batches
- Level 3: Each target year processing uses optimized sequential I/O

This avoids Dask's NetCDF threading issues while maximizing resource utilization.
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
from typing import List, Dict, Tuple, Optional
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import json
from datetime import datetime

# Import our modules
from io_util import NorESM2FileHandler
from regions import REGION_BOUNDS, extract_region
from time_util import handle_time_coordinates
from climate_means import compute_climate_normal, calculate_daily_climatology

# Configuration for high-performance processing
INPUT_DATA_DIR = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
OUTPUT_BASE_DIR = "output/rolling_30year_climate_normals"
VARIABLES = ['pr', 'tas', 'tasmax', 'tasmin']
REGIONS = ['CONUS']
MIN_YEARS_FOR_NORMAL = 25

# Performance configuration - Increased parallel processing (45 cores total)
MAX_CORES = 24
MAX_RAM_GB = 80 # Supposed to kill the process if it goes over 80GB
CORES_PER_VARIABLE = 6  # 4 variables × 12 cores = 48 cores total
BATCH_SIZE_YEARS = 2     # small batch sizes for memory management
MAX_MEMORY_PER_PROCESS_GB = 4  # Keep conservative memory per process
MEMORY_CHECK_INTERVAL = 10  # Check memory every 10 files (less frequent)

# Known corrupted files to skip
CORRUPTED_FILES = {
    'tasmin_day_NorESM2-LM_ssp245_r1i1p1f1_gn_2033.nc'
}

# Progress tracking configuration
PROGRESS_STATUS_FILE = "processing_progress.json"
PROGRESS_LOG_FILE = "processing_progress.log"
STATUS_UPDATE_INTERVAL = 30  # Update status every 30 seconds

class ProgressTracker:
    """Real-time progress tracking for multiprocessing pipeline."""
    
    def __init__(self, status_file: str, progress_log: str):
        self.status_file = status_file
        self.progress_log = progress_log
        self.start_time = time.time()
        self.last_update = 0
        
        # Initialize progress structure
        self.progress = {
            "pipeline_start": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "status": "initializing",
            "total_files_target": 547,
            "total_files_completed": 0,
            "completion_percentage": 0.0,
            "estimated_time_remaining": None,
            "variables": {
                "pr": {"completed": 0, "target": 97, "status": "pending"},
                "tas": {"completed": 0, "target": 150, "status": "pending"},
                "tasmax": {"completed": 0, "target": 150, "status": "pending"},
                "tasmin": {"completed": 0, "target": 150, "status": "pending"}
            },
            "periods": {
                "historical": {"completed": 0, "target": 140},
                "hybrid": {"completed": 0, "target": 120},
                "ssp245": {"completed": 0, "target": 287}
            },
            "performance": {
                "files_per_minute": 0.0,
                "workers_active": 0,
                "memory_usage_gb": 0.0,
                "cpu_usage_percent": 0.0
            },
            "current_work": {
                "active_batches": [],
                "recently_completed": []
            }
        }
        
        # Setup progress logging
        self.setup_progress_logging()
        self.update_status()
    
    def setup_progress_logging(self):
        """Setup dedicated progress logging."""
        progress_logger = logging.getLogger('progress')
        progress_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in progress_logger.handlers[:]:
            progress_logger.removeHandler(handler)
        
        # Add progress file handler
        handler = logging.FileHandler(self.progress_log)
        formatter = logging.Formatter('%(asctime)s - PROGRESS - %(message)s')
        handler.setFormatter(formatter)
        progress_logger.addHandler(handler)
        
        self.logger = progress_logger
    
    def scan_existing_files(self):
        """Scan for existing completed files to update baseline."""
        for variable in VARIABLES:
            count = 0
            for period in ['historical', 'hybrid', 'ssp245']:
                period_dir = Path(OUTPUT_BASE_DIR) / variable / period
                if period_dir.exists():
                    period_count = len(list(period_dir.glob("*.nc")))
                    count += period_count
                    self.progress["periods"][period]["completed"] += period_count
            
            self.progress["variables"][variable]["completed"] = count
            if count > 0:
                self.progress["variables"][variable]["status"] = "in_progress"
        
        # Update totals
        self.progress["total_files_completed"] = sum(
            var["completed"] for var in self.progress["variables"].values()
        )
        self.progress["completion_percentage"] = (
            self.progress["total_files_completed"] / self.progress["total_files_target"]
        ) * 100
        
        self.logger.info(f"Baseline scan: {self.progress['total_files_completed']} files already completed")
    
    def update_status(self, force=False):
        """Update progress status file and log."""
        current_time = time.time()
        
        if not force and (current_time - self.last_update) < STATUS_UPDATE_INTERVAL:
            return
        
        self.last_update = current_time
        self.progress["last_update"] = datetime.now().isoformat()
        
        # Update performance metrics
        try:
            # Get system stats
            memory = psutil.virtual_memory()
            self.progress["performance"]["memory_usage_gb"] = memory.used / (1024**3)
            self.progress["performance"]["cpu_usage_percent"] = psutil.cpu_percent()
            
            # Calculate processing rate
            elapsed_hours = (current_time - self.start_time) / 3600
            if elapsed_hours > 0:
                self.progress["performance"]["files_per_minute"] = (
                    self.progress["total_files_completed"] / (elapsed_hours * 60)
                )
                
                # Estimate time remaining
                remaining_files = (
                    self.progress["total_files_target"] - self.progress["total_files_completed"]
                )
                if self.progress["performance"]["files_per_minute"] > 0:
                    eta_minutes = remaining_files / self.progress["performance"]["files_per_minute"]
                    self.progress["estimated_time_remaining"] = f"{eta_minutes:.1f} minutes"
        except:
            pass
        
        # Write status file
        with open(self.status_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
        
        # Log progress summary
        self.logger.info(
            f"Progress: {self.progress['total_files_completed']}/{self.progress['total_files_target']} "
            f"({self.progress['completion_percentage']:.1f}%) - "
            f"Rate: {self.progress['performance']['files_per_minute']:.1f} files/min - "
            f"ETA: {self.progress.get('estimated_time_remaining', 'calculating...')}"
        )
    
    def report_batch_start(self, worker_id: int, variable: str, period: str, years: List[int]):
        """Report when a batch starts processing."""
        batch_info = {
            "worker_id": worker_id,
            "variable": variable,
            "period": period,
            "years": years,
            "start_time": datetime.now().isoformat()
        }
        
        self.progress["current_work"]["active_batches"].append(batch_info)
        self.progress["variables"][variable]["status"] = "processing"
        
        self.logger.info(f"Worker {worker_id} started: {variable} {period} years {years}")
        self.update_status()
    
    def report_batch_complete(self, worker_id: int, variable: str, completed_years: List[int]):
        """Report when a batch completes."""
        # Remove from active batches
        self.progress["current_work"]["active_batches"] = [
            batch for batch in self.progress["current_work"]["active_batches"]
            if batch["worker_id"] != worker_id
        ]
        
        # Add to recently completed
        completion_info = {
            "worker_id": worker_id,
            "variable": variable,
            "years": completed_years,
            "completion_time": datetime.now().isoformat()
        }
        
        self.progress["current_work"]["recently_completed"].append(completion_info)
        
        # Keep only last 10 completed batches
        if len(self.progress["current_work"]["recently_completed"]) > 10:
            self.progress["current_work"]["recently_completed"] = (
                self.progress["current_work"]["recently_completed"][-10:]
            )
        
        # Update completion counts
        self.progress["variables"][variable]["completed"] += len(completed_years)
        self.progress["total_files_completed"] += len(completed_years)
        self.progress["completion_percentage"] = (
            self.progress["total_files_completed"] / self.progress["total_files_target"]
        ) * 100
        
        self.logger.info(
            f"Worker {worker_id} completed: {variable} years {completed_years} "
            f"({len(completed_years)} files)"
        )
        self.update_status(force=True)
    
    def report_file_complete(self, variable: str, period: str, year: int):
        """Report individual file completion."""
        self.progress["periods"][period]["completed"] += 1
        # Force update every 5 files for more responsive tracking
        if self.progress["total_files_completed"] % 5 == 0:
            self.update_status(force=True)
    
    def finalize(self):
        """Finalize progress tracking."""
        self.progress["status"] = "completed"
        self.progress["pipeline_end"] = datetime.now().isoformat()
        
        total_time = time.time() - self.start_time
        self.progress["total_duration_minutes"] = total_time / 60
        
        self.logger.info(f"Pipeline completed in {total_time/60:.1f} minutes")
        self.update_status(force=True)

# Global progress tracker
progress_tracker = None

# Setup logging for multiprocessing
def setup_logging(log_level=logging.INFO):
    """Setup logging configuration for multiprocessing."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('processing_multiprocessing_v4_resume.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_single_file_for_climatology_mp(file_path: str, variable_name: str, 
                                          region_key: str, input_data_dir: str) -> Tuple[int, Optional[xr.DataArray]]:
    """
    Multiprocessing-safe version: process a single file to extract daily climatology.
    Implements aggressive memory management to prevent accumulation.
    """
    ds = None
    region_ds = None
    var = None
    climatology = None
    result = None
    
    try:
        # Initialize file handler within the process (avoid sharing across processes)
        file_handler = NorESM2FileHandler(input_data_dir)
        
        # Extract year from filename
        year = file_handler.extract_year_from_filename(file_path)
        if year is None:
            return None, None
        
        # Open dataset with very conservative memory settings and smaller chunks
        ds = xr.open_dataset(file_path, decode_times=False, chunks={'time': 30})  # Smaller chunks
        
        # Check if variable exists
        if variable_name not in ds.data_vars:
            return None, None
        
        # Handle time coordinates
        ds, time_method = handle_time_coordinates(ds, file_path)
        
        # Extract region
        region_bounds = REGION_BOUNDS[region_key]
        region_ds = extract_region(ds, region_bounds)
        var = region_ds[variable_name]
        
        # Calculate daily climatology
        climatology = calculate_daily_climatology(var, time_method, file_path)
        if climatology is None:
            return None, None
        
        # Compute result immediately with limited memory usage
        with xr.set_options(keep_attrs=True):
            result = climatology.compute()
            # Ensure result is a copy, not a view
            result = result.copy(deep=True)
        
        return year, result
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None
    
    finally:
        # Aggressive cleanup - explicitly close and delete everything
        try:
            if climatology is not None:
                del climatology
            if var is not None:
                del var  
            if region_ds is not None:
                del region_ds
            if ds is not None:
                ds.close()
                del ds
            # Force garbage collection
            gc.collect()
        except:
            pass

def process_target_year_batch(args: Tuple) -> Dict:
    """
    Process a batch of target years for a specific variable and period type.
    This function runs in a separate process.
    """
    (variable, region_key, period_type, target_years, input_data_dir, 
     output_base_dir, min_years, worker_id) = args
    
    # Setup logging for this worker
    logger = logging.getLogger(f'worker_{worker_id}')
    
    results = {
        'worker_id': worker_id,
        'variable': variable,
        'period_type': period_type,
        'target_years': target_years,
        'processed': [],
        'failed': [],
        'start_time': time.time()
    }
    
    try:
        logger.info(f"Worker {worker_id}: Processing {len(target_years)} {period_type} years for {variable}")
        
        # Report batch start to progress tracker
        # Note: We can't directly access the global tracker from worker processes,
        # so we'll track in the main process instead
        
        # Log initial memory usage
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024 * 1024)
            logger.info(f"Worker {worker_id}: Initial memory usage: {initial_memory:.1f}GB")
        except:
            pass
        
        # Initialize file handler within this process
        file_handler = NorESM2FileHandler(input_data_dir)
        
        # Output directory
        output_dir = Path(output_base_dir) / variable / period_type
        output_dir.mkdir(parents=True, exist_ok=True)

        for target_year in target_years:
            try:
                # Skip if already processed
                if file_already_processed(variable, region_key, period_type, target_year, output_base_dir):
                    logger.info(f"Worker {worker_id}: Skipping {variable} {period_type} {target_year} (already processed)")
                    results['processed'].append(target_year)
                    continue
                
                logger.info(f"Worker {worker_id}: Processing {variable} {period_type} {target_year}")
                
                # Get files based on period type
                if period_type == 'historical':
                    start_year = target_year - 29
                    end_year = target_year
                    files = file_handler.get_files_for_period(variable, 'historical', start_year, end_year)
                    scenario_info = {'historical': len(files)}
                    
                elif period_type == 'hybrid':
                    files, scenario_counts = file_handler.get_hybrid_files_for_period(variable, target_year, 30)
                    scenario_info = scenario_counts
                    
                elif period_type == 'ssp245':
                    start_year = target_year - 29
                    end_year = target_year
                    files = file_handler.get_files_for_period(variable, 'ssp245', start_year, end_year)
                    scenario_info = {'ssp245': len(files)}
                else:
                    logger.error(f"Unknown period type: {period_type}")
                    continue
                
                # Filter out corrupted files
                valid_files = [f for f in files if not is_corrupted_file(f)]
                if len(valid_files) < len(files):
                    logger.warning(f"Worker {worker_id}: Skipped {len(files) - len(valid_files)} corrupted files for {target_year}")
                    files = valid_files
                
                if len(files) < min_years:
                    logger.warning(f"Worker {worker_id}: Insufficient files ({len(files)}) for {target_year} after filtering")
                    results['failed'].append((target_year, f"Insufficient files: {len(files)}"))
                    continue
                
                logger.info(f"Worker {worker_id}: Processing {len(files)} files for {target_year}")
                
                # Process files to get daily climatologies with progress tracking
                daily_climatologies = []
                years_used = []
                scenarios_used = []
                
                file_count = 0
                for file_path in files:
                    file_count += 1
                    
                    # Periodic memory check
                    if file_count % MEMORY_CHECK_INTERVAL == 0:
                        if not check_memory_usage(worker_id, logger):
                            logger.error(f"Worker {worker_id}: Memory limit exceeded, aborting target year {target_year}")
                            results['failed'].append((target_year, "Memory limit exceeded"))
                            # Aggressive cleanup before breaking
                            del daily_climatologies, years_used, scenarios_used
                            gc.collect()
                            break
                    
                    year, daily_clim = process_single_file_for_climatology_mp(
                        file_path, variable, region_key, input_data_dir
                    )
                    if year is not None and daily_clim is not None:
                        daily_climatologies.append(daily_clim)
                        years_used.append(year)
                        
                        # Determine scenario from file path
                        if 'historical' in str(file_path):
                            scenarios_used.append('historical')
                        elif 'ssp245' in str(file_path):
                            scenarios_used.append('ssp245')
                        elif 'ssp585' in str(file_path):
                            scenarios_used.append('ssp585')
                        else:
                            scenarios_used.append('unknown')
                        
                        # Periodic cleanup during processing
                        if file_count % MEMORY_CHECK_INTERVAL == 0:
                            gc.collect()
                
                # Check if we have enough data and didn't abort due to memory issues
                if len(daily_climatologies) >= min_years:
                    logger.info(f"Worker {worker_id}: Computing normal from {len(daily_climatologies)} years")
                    
                    # Compute 30-year climate normal
                    data_arrays = [clim.values for clim in daily_climatologies]
                    climate_normal = compute_climate_normal(data_arrays, years_used, target_year)
                    
                    if climate_normal is not None:
                        # Add metadata
                        metadata = {
                            'title': f'{variable.upper()} 30-Year {period_type.title()} Climate Normal ({region_key}) - Target Year {target_year}',
                            'variable': variable,
                            'region': region_key,
                            'target_year': target_year,
                            'period_type': period_type,
                            'period_used': f"{min(years_used)}-{max(years_used)}",
                            'num_years': len(years_used),
                            'creation_date': str(pd.Timestamp.now()),
                            'source': 'NorESM2-LM climate model',
                            'method': f'30-year {period_type} rolling climate normal',
                            'processing': 'Multiprocessing (process-based)',
                            'worker_id': worker_id
                        }
                        
                        # Add scenario-specific metadata
                        if period_type == 'hybrid':
                            metadata.update({
                                'historical_years': scenarios_used.count('historical'),
                                'ssp245_years': scenarios_used.count('ssp245'),
                                'scenarios': 'historical + ssp245'
                            })
                        elif period_type == 'historical':
                            metadata['scenario'] = 'historical'
                        elif period_type == 'ssp245':
                            metadata['scenario'] = 'ssp245'
                        
                        climate_normal.attrs.update(metadata)
                        
                        # Save individual file
                        output_file = output_dir / f"{variable}_{region_key}_{period_type}_{target_year}_30yr_normal.nc"
                        climate_normal.to_netcdf(output_file)
                        
                        logger.info(f"Worker {worker_id}: ✅ Saved {period_type} normal for {target_year}")
                        results['processed'].append(target_year)
                        
                        # Log progress milestone
                        logger.info(f"PROGRESS: {variable}_{period_type}_{target_year} COMPLETED")
                        
                        # Cleanup
                        del climate_normal, daily_climatologies
                        gc.collect()
                    else:
                        logger.error(f"Worker {worker_id}: ✗ Failed to compute normal for {target_year}")
                        results['failed'].append((target_year, "Failed to compute normal"))
                else:
                    logger.warning(f"Worker {worker_id}: Insufficient processed files ({len(daily_climatologies)}) for {target_year}")
                    results['failed'].append((target_year, f"Insufficient processed files: {len(daily_climatologies)}"))
                
                # Aggressive cleanup after each target year regardless of success/failure
                try:
                    if 'daily_climatologies' in locals():
                        del daily_climatologies
                    if 'years_used' in locals():
                        del years_used
                    if 'scenarios_used' in locals():
                        del scenarios_used
                    if 'climate_normal' in locals():
                        del climate_normal
                    gc.collect()
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"Worker {worker_id}: Error processing {target_year}: {e}")
                results['failed'].append((target_year, str(e)))
                continue
        
        results['end_time'] = time.time()
        results['duration'] = results['end_time'] - results['start_time']
        
        # Log final memory usage
        try:
            process = psutil.Process()
            final_memory = process.memory_info().rss / (1024 * 1024 * 1024)
            logger.info(f"Worker {worker_id}: Final memory usage: {final_memory:.1f}GB")
        except:
            pass
            
        logger.info(f"Worker {worker_id}: Completed batch in {results['duration']:.1f}s")
        
    except Exception as e:
        logger.error(f"Worker {worker_id}: Fatal error: {e}")
        results['fatal_error'] = str(e)
    
    return results

def process_variable_multiprocessing(variable: str, region_key: str, 
                                   input_data_dir: str, output_base_dir: str,
                                   cores_per_variable: int, logger: logging.Logger, progress_tracker: ProgressTracker) -> Dict:
    """
    Process a single variable using multiple processes for different target years.
    """
    logger.info(f"🔄 Starting multiprocessing for {variable.upper()}")
    
    # Initialize file handler to get data availability
    file_handler = NorESM2FileHandler(input_data_dir)
    availability = file_handler.validate_data_availability()
    
    if variable not in availability:
        logger.error(f"Variable {variable} not available in data")
        return {'variable': variable, 'status': 'failed', 'reason': 'Variable not available'}
    
    # Define target years for each period type
    historical_years = list(range(1980, 2015))  # 1980-2014
    hybrid_years = list(range(2015, 2045))      # 2015-2044
    
    # Future years depend on data availability
    future_years = []
    if 'ssp245' in availability[variable]:
        _, max_year = availability[variable]['ssp245']
        future_years = list(range(2045, min(max_year + 1, 2101)))
    
    # Create work batches
    work_batches = []
    worker_id = 0
    
    # Historical batches
    for i in range(0, len(historical_years), BATCH_SIZE_YEARS):
        batch_years = historical_years[i:i + BATCH_SIZE_YEARS]
        work_batches.append((
            variable, region_key, 'historical', batch_years, input_data_dir,
            output_base_dir, MIN_YEARS_FOR_NORMAL, worker_id
        ))
        worker_id += 1
    
    # Hybrid batches
    for i in range(0, len(hybrid_years), BATCH_SIZE_YEARS):
        batch_years = hybrid_years[i:i + BATCH_SIZE_YEARS]
        work_batches.append((
            variable, region_key, 'hybrid', batch_years, input_data_dir,
            output_base_dir, MIN_YEARS_FOR_NORMAL, worker_id
        ))
        worker_id += 1
    
    # Future batches
    if future_years:
        for i in range(0, len(future_years), BATCH_SIZE_YEARS):
            batch_years = future_years[i:i + BATCH_SIZE_YEARS]
            work_batches.append((
                variable, region_key, 'ssp245', batch_years, input_data_dir,
                output_base_dir, MIN_YEARS_FOR_NORMAL, worker_id
            ))
            worker_id += 1
    
    logger.info(f"{variable}: Created {len(work_batches)} work batches using {cores_per_variable} cores")
    
    # Process batches using process pool
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=cores_per_variable) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(process_target_year_batch, batch): batch 
            for batch in work_batches
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"{variable}: Batch completed - Worker {result['worker_id']}: "
                          f"{len(result['processed'])} processed, {len(result['failed'])} failed")
            except Exception as e:
                logger.error(f"{variable}: Batch failed: {e}")
                results.append({
                    'worker_id': batch[-1],
                    'variable': variable,
                    'status': 'failed',
                    'error': str(e)
                })
    
    end_time = time.time()
    
    # Summarize results
    total_processed = sum(len(r.get('processed', [])) for r in results)
    total_failed = sum(len(r.get('failed', [])) for r in results)
    
    summary = {
        'variable': variable,
        'status': 'completed',
        'duration': end_time - start_time,
        'total_batches': len(work_batches),
        'total_processed': total_processed,
        'total_failed': total_failed,
        'batch_results': results
    }
    
    logger.info(f"✅ {variable.upper()} completed in {summary['duration']:.1f}s: "
              f"{total_processed} processed, {total_failed} failed")
    
    return summary

def create_combined_datasets(variable: str, output_base_dir: str, logger: logging.Logger):
    """Create combined datasets from individual files."""
    logger.info(f"Creating combined datasets for {variable}")
    
    base_path = Path(output_base_dir) / variable
    
    for period_type in ['historical', 'hybrid', 'ssp245']:
        period_dir = base_path / period_type
        if not period_dir.exists():
            continue
        
        # Find all individual files
        pattern = f"{variable}_CONUS_{period_type}_*_30yr_normal.nc"
        files = list(period_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No files found for {variable} {period_type}")
            continue
        
        logger.info(f"Combining {len(files)} files for {variable} {period_type}")
        
        try:
            # Load all files and extract target years
            datasets = []
            target_years = []
            
            for file_path in sorted(files):
                ds = xr.open_dataset(file_path)
                target_year = int(ds.attrs['target_year'])
                # Ensure the dataset has proper coordinates
                ds = ds.expand_dims('target_year')
                ds = ds.assign_coords(target_year=[target_year])
                datasets.append(ds)
                target_years.append(target_year)
            
            # Combine along target_year dimension
            combined = xr.concat(datasets, dim='target_year')
            
            # Ensure target_year coordinate is properly set
            combined = combined.assign_coords(target_year=target_years)
            
            # Add global attributes
            combined.attrs.update({
                'title': f'{variable.upper()} {period_type.title()} 30-Year Climate Normals (CONUS)',
                'description': f'{period_type.title()} 30-year rolling climate normals for {variable}',
                'variable': variable,
                'region': 'CONUS',
                'period_type': period_type,
                'target_years': f"{min(target_years)}-{max(target_years)}",
                'num_normals': len(target_years),
                'creation_date': str(pd.Timestamp.now()),
                'source': 'NorESM2-LM climate model',
                'method': f'{period_type.title()} multiprocessing rolling 30-year climate normals',
                'processing': 'Multiprocessing (process-based)'
            })
            
            # Save combined file
            if period_type == 'hybrid':
                combined_file = period_dir / f"{variable}_CONUS_hybrid_2015-2044_all_normals.nc"
            elif period_type == 'historical':
                combined_file = period_dir / f"{variable}_CONUS_historical_1980-2014_all_normals.nc"
            else:  # ssp245
                combined_file = period_dir / f"{variable}_CONUS_ssp245_{min(target_years)}-{max(target_years)}_all_normals.nc"
            
            combined.to_netcdf(combined_file)
            logger.info(f"✓ Created combined file: {combined_file}")
            
            # Close datasets
            for ds in datasets:
                ds.close()
            combined.close()
            
        except Exception as e:
            logger.error(f"Error creating combined dataset for {variable} {period_type}: {e}")

def check_memory_usage(worker_id: int, logger: logging.Logger) -> bool:
    """Check if current process is using too much memory."""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        memory_gb = memory_mb / 1024
        
        if memory_gb > MAX_MEMORY_PER_PROCESS_GB:
            logger.warning(f"Worker {worker_id}: Memory usage {memory_gb:.1f}GB exceeds limit {MAX_MEMORY_PER_PROCESS_GB}GB")
            return False
        
        logger.debug(f"Worker {worker_id}: Memory usage: {memory_gb:.1f}GB")
        return True
    except:
        return True

def file_already_processed(variable: str, region_key: str, period_type: str, target_year: int, output_base_dir: str) -> bool:
    """Check if a target year has already been processed."""
    output_dir = Path(output_base_dir) / variable / period_type
    output_file = output_dir / f"{variable}_{region_key}_{period_type}_{target_year}_30yr_normal.nc"
    return output_file.exists()

def is_corrupted_file(file_path: str) -> bool:
    """Check if a file is in the known corrupted files list."""
    return Path(file_path).name in CORRUPTED_FILES

def main():
    """Main multiprocessing function."""
    global progress_tracker
    
    # Setup logging
    logger = setup_logging()
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(PROGRESS_STATUS_FILE, PROGRESS_LOG_FILE)
    progress_tracker.scan_existing_files()
    progress_tracker.progress["status"] = "running"
    progress_tracker.update_status(force=True)
    
    logger.info("🚀 Starting Multiprocessing Climate Normals Processing Pipeline")
    logger.info(f"Available cores: {mp.cpu_count()}")
    logger.info(f"Available RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"Configuration: {MAX_CORES} max cores, {CORES_PER_VARIABLE} cores per variable")
    logger.info(f"Input directory: {INPUT_DATA_DIR}")
    logger.info(f"Output directory: {OUTPUT_BASE_DIR}")
    logger.info(f"Variables: {VARIABLES}")
    logger.info(f"Regions: {REGIONS}")
    logger.info(f"📊 Progress tracking: {PROGRESS_STATUS_FILE}")
    logger.info(f"📈 Progress log: {PROGRESS_LOG_FILE}")
    
    start_time = time.time()
    
    # Process each variable in parallel
    logger.info(f"\n{'='*80}")
    logger.info("🔥 Starting parallel variable processing")
    logger.info(f"{'='*80}")
    
    variable_results = []
    
    # Use ProcessPoolExecutor to process variables in parallel
    with ProcessPoolExecutor(max_workers=len(VARIABLES)) as executor:
        # Submit variable processing jobs
        future_to_variable = {
            executor.submit(
                process_variable_multiprocessing,
                variable, REGIONS[0], INPUT_DATA_DIR, OUTPUT_BASE_DIR,
                CORES_PER_VARIABLE, logger, progress_tracker
            ): variable
            for variable in VARIABLES
        }
        
        # Collect results
        for future in as_completed(future_to_variable):
            variable = future_to_variable[future]
            try:
                result = future.result()
                variable_results.append(result)
                logger.info(f"🎉 {variable.upper()} processing completed!")
                
                # Update progress tracker
                progress_tracker.progress["variables"][variable]["status"] = "completed"
                progress_tracker.update_status(force=True)
                
            except Exception as e:
                logger.error(f"❌ {variable.upper()} processing failed: {e}")
                variable_results.append({
                    'variable': variable,
                    'status': 'failed',
                    'error': str(e)
                })
                progress_tracker.progress["variables"][variable]["status"] = "failed"
    
    # Create combined datasets
    logger.info(f"\n{'='*80}")
    logger.info("📦 Creating combined datasets")
    logger.info(f"{'='*80}")
    
    for variable in VARIABLES:
        create_combined_datasets(variable, OUTPUT_BASE_DIR, logger)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("🎊 MULTIPROCESSING PIPELINE COMPLETED!")
    logger.info(f"{'='*80}")
    logger.info(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    
    successful_vars = [r for r in variable_results if r.get('status') == 'completed']
    failed_vars = [r for r in variable_results if r.get('status') == 'failed']
    
    logger.info(f"✅ Successful variables: {len(successful_vars)}/{len(VARIABLES)}")
    for result in successful_vars:
        logger.info(f"  {result['variable'].upper()}: {result['total_processed']} normals, "
                   f"{result['duration']:.1f}s")
    
    if failed_vars:
        logger.info(f"❌ Failed variables: {len(failed_vars)}")
        for result in failed_vars:
            logger.info(f"  {result['variable'].upper()}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"📁 Results saved to: {OUTPUT_BASE_DIR}/")
    
    # Finalize progress tracking
    progress_tracker.finalize()
    
    return variable_results

if __name__ == "__main__":
    # Ensure proper multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1) 