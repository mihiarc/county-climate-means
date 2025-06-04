#!/usr/bin/env python3
"""
Guam & Northern Mariana Islands Climate Normals Processing Pipeline - Multiprocessing Version

Specialized script for processing climate variables (pr, tas, tasmax, tasmin) for GU region.
Processes all four variables simultaneously with parallel processing.
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
sys.path.append('src')
from utils.io import NorESM2FileHandler
from utils.regions import REGION_BOUNDS, extract_region
from utils.time_handling import handle_time_coordinates, extract_year_from_filename

# Configuration for Guam climate processing
INPUT_DATA_DIR = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
OUTPUT_BASE_DIR = "output/guam_normals"
VARIABLES = ['pr', 'tas', 'tasmax', 'tasmin']  # All climate variables
REGION = 'GU'  # Guam and Northern Mariana Islands
MIN_YEARS_FOR_NORMAL = 25

# Performance configuration
MAX_CORES = 8
CORES_PER_VARIABLE = 2   # 2 cores per variable (4 variables = 8 total)
BATCH_SIZE_YEARS = 2     # Small batch sizes
MAX_MEMORY_PER_PROCESS_GB = 4
MEMORY_CHECK_INTERVAL = 10

# Progress tracking
PROGRESS_STATUS_FILE = "guam_processing_progress.json"
PROGRESS_LOG_FILE = "guam_processing_progress.log"
STATUS_UPDATE_INTERVAL = 30

def setup_logging():
    """Setup logging for the pipeline with CRITICAL level only."""
    logging.basicConfig(
        level=logging.CRITICAL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('guam_climate_processing.log')
        ]
    )
    return logging.getLogger(__name__)

def extract_year_from_path(file_path: str) -> Optional[int]:
    """Extract year from file path."""
    return extract_year_from_filename(file_path)

def process_single_file_for_climatology_safe(file_path: str, variable_name: str, 
                                           region_key: str) -> Tuple[Optional[int], Optional[xr.DataArray]]:
    """
    Process a single file to extract daily climatology - safe multiprocessing version.
    Based on the working legacy implementation.
    """
    try:
        # Extract year from filename
        year = extract_year_from_path(file_path)
        if year is None:
            return None, None
        
        # Open dataset with conservative settings (no chunking in multiprocessing)
        ds = xr.open_dataset(file_path, decode_times=False, cache=False)
        
        # Check if variable exists
        if variable_name not in ds.data_vars:
            ds.close()
            return None, None
        
        # Handle time coordinates
        ds, time_method = handle_time_coordinates(ds, file_path)
        
        # Extract Guam region
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
        logging.error(f"Error processing {Path(file_path).name}: {e}")
        return None, None

def compute_climate_normal_safe(data_arrays: List, years: List[int], target_year: int) -> Optional[xr.DataArray]:
    """Compute climate normal from multiple data arrays - safe multiprocessing version."""
    if not data_arrays:
        return None
    
    try:
        # Stack arrays and compute mean
        stacked_data = xr.concat(data_arrays, dim='year')
        mean_data = stacked_data.mean(dim='year')
        
        # Add metadata
        mean_data.attrs.update({
            'long_name': f'30-year climate normal for {target_year}',
            'target_year': target_year,
            'source_years': f"{min(years)}-{max(years)}",
            'number_of_years': len(years),
            'processing_method': 'multiprocessing_safe_guam'
        })
        
        return mean_data
        
    except Exception as e:
        logging.error(f"Error computing climate normal for {target_year}: {e}")
        return None

def process_target_year_batch(args: Tuple) -> Dict:
    """Process a batch of target years for a specific variable and period."""
    (variable, region_key, period_type, target_years, 
     input_data_dir, output_base_dir, worker_id) = args
    
    logger = logging.getLogger(f'worker_{worker_id}')
    results = []
    
    try:
        # Initialize file handler
        file_handler = NorESM2FileHandler(input_data_dir)
        
        for target_year in target_years:
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
                    continue
                
                if len(files) < MIN_YEARS_FOR_NORMAL:
                    continue
                
                # Process files to get daily climatologies
                daily_climatologies = []
                years_used = []
                
                for file_path in files:
                    year, daily_clim = process_single_file_for_climatology_safe(
                        file_path, variable, region_key
                    )
                    if year is not None and daily_clim is not None:
                        daily_climatologies.append(daily_clim)
                        years_used.append(year)
                
                if len(daily_climatologies) < MIN_YEARS_FOR_NORMAL:
                    continue
                
                # Compute climate normal
                climate_normal = compute_climate_normal_safe(daily_climatologies, years_used, target_year)
                
                if climate_normal is None:
                    continue
                
                # Save result
                output_dir = Path(output_base_dir) / variable / period_type
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = output_dir / f"{variable}_{region_key}_{period_type}_{target_year}_30yr_normal.nc"
                
                # RESTART FUNCTIONALITY: Check if file already exists and skip
                if output_file.exists():
                    results.append({
                        'target_year': target_year,
                        'status': 'skipped',
                        'output_file': str(output_file),
                        'reason': 'file_already_exists'
                    })
                    continue
                
                # Add comprehensive metadata
                climate_normal.attrs.update({
                    'title': f'{variable.upper()} 30-Year {period_type.title()} Climate Normal ({region_key}) - Target Year {target_year}',
                    'variable': variable,
                    'region': region_key,
                    'target_year': target_year,
                    'period_type': period_type,
                    'num_years': len(years_used),
                    'processing': 'Guam multiprocessing pipeline',
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
                results.append({
                    'target_year': target_year,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return {
            'worker_id': worker_id,
            'variable': variable,
            'period_type': period_type,
            'results': results,
            'status': 'completed'
        }
        
    except Exception as e:
        return {
            'worker_id': worker_id,
            'variable': variable,
            'period_type': period_type,
            'results': [],
            'status': 'failed',
            'error': str(e)
        }

def process_variable_multiprocessing(variable: str, region_key: str, 
                                   input_data_dir: str, output_base_dir: str,
                                   cores_per_variable: int, logger: logging.Logger) -> Dict:
    """Process a single variable using multiprocessing for different periods."""
    
    try:
        # Get file handler
        file_handler = NorESM2FileHandler(input_data_dir)
        
        # Define target years for each period
        periods_config = {
            'historical': list(range(1980, 2015)),  # 1980-2014
            'hybrid': list(range(2015, 2045)),      # 2015-2044
            'ssp245': list(range(2045, 2101))       # 2045-2100
        }
        
        all_work_items = []
        worker_id = 0
        
        # Create work batches for all periods
        for period_type, target_years in periods_config.items():
            # Split years into batches
            for i in range(0, len(target_years), BATCH_SIZE_YEARS):
                batch_years = target_years[i:i + BATCH_SIZE_YEARS]
                work_item = (
                    variable, region_key, period_type, batch_years,
                    input_data_dir, output_base_dir, worker_id
                )
                all_work_items.append(work_item)
                worker_id += 1
        
        # Process batches in parallel
        results = []
        successful_batches = 0
        failed_batches = 0
        
        with ProcessPoolExecutor(max_workers=cores_per_variable) as executor:
            # Submit all work items
            future_to_work = {
                executor.submit(process_target_year_batch, work_item): work_item
                for work_item in all_work_items
            }
            
            # Collect results
            for future in as_completed(future_to_work):
                work_item = future_to_work[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'completed':
                        successful_batches += 1
                    else:
                        failed_batches += 1
                        
                except Exception as e:
                    failed_batches += 1
        
        # Summarize results
        total_successful_years = sum(
            len([r for r in result['results'] if r['status'] == 'success'])
            for result in results if result['status'] == 'completed'
        )
        
        return {
            'variable': variable,
            'status': 'completed',
            'total_years_processed': total_successful_years,
            'successful_batches': successful_batches,
            'failed_batches': failed_batches,
            'batch_results': results
        }
        
    except Exception as e:
        return {
            'variable': variable,
            'status': 'failed',
            'error': str(e)
        }

def main():
    """Main processing function for Guam climate normals."""
    start_time = time.time()
    logger = setup_logging()
    
    print("🏖️  Starting Guam Climate Normals Processing Pipeline")
    print("📍 Guam & Northern Mariana Islands")
    print(f"Variables: {', '.join(VARIABLES)}")
    print(f"Region: {REGION}")
    print(f"Cores per variable: {CORES_PER_VARIABLE}")
    print(f"Total cores: {len(VARIABLES) * CORES_PER_VARIABLE}")
    print(f"Region bounds: 144.58°-146.12°E, 13.18°-20.61°N")
    print("🌊 Processing the smallest U.S. territory - should be very fast!")
    
    # Create output directory
    Path(OUTPUT_BASE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Process all variables in parallel
    variable_results = []
    
    with ProcessPoolExecutor(max_workers=len(VARIABLES)) as main_executor:
        # Submit all variables for processing
        future_to_variable = {
            main_executor.submit(
                process_variable_multiprocessing,
                variable, REGION, INPUT_DATA_DIR, OUTPUT_BASE_DIR,
                CORES_PER_VARIABLE, logger
            ): variable for variable in VARIABLES
        }
        
        # Collect results
        for future in as_completed(future_to_variable):
            variable = future_to_variable[future]
            try:
                result = future.result()
                variable_results.append(result)
                print(f"✅ {variable}: {result.get('total_years_processed', 0)} years processed")
            except Exception as e:
                print(f"❌ {variable}: Failed with error {e}")
                variable_results.append({
                    'variable': variable,
                    'status': 'failed',
                    'error': str(e)
                })
    
    # Final summary
    end_time = time.time()
    total_duration = end_time - start_time
    total_years_processed = sum(
        result.get('total_years_processed', 0) 
        for result in variable_results
    )
    
    print(f"\n🎉 Guam Climate Processing Complete!")
    print(f"Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Total years processed: {total_years_processed}")
    
    # Detailed results per variable
    for result in variable_results:
        variable = result['variable']
        if result['status'] == 'completed':
            years = result['total_years_processed']
            batches = result['successful_batches']
            print(f"  {variable}: {years} years in {batches} batches")
        else:
            print(f"  {variable}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Save final results
    summary = {
        'processing_complete': datetime.now().isoformat(),
        'duration_seconds': total_duration,
        'region': REGION,
        'region_name': 'Guam and Northern Mariana Islands',
        'variables': VARIABLES,
        'total_years_processed': total_years_processed,
        'variable_results': variable_results
    }
    
    with open('guam_processing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: guam_processing_summary.json")
    print("🏁 All U.S. regions complete when this finishes!")

if __name__ == "__main__":
    main() 