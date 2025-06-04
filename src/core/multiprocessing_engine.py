#!/usr/bin/env python3
"""
Multiprocessing Climate Data Processing Engine

High-performance multiprocessing implementation for climate data processing.
Designed to work with existing climate processing functions while utilizing
multiple CPU cores for significant speedup.

This is a refactored version of climate_multiprocessing.py following OOP principles.
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

# Import utility modules
from ..utils.io import NorESM2FileHandler
from ..utils.regions import REGION_BOUNDS, extract_region  
from ..utils.time_handling import handle_time_coordinates

# Setup logging
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


class ClimateMultiprocessor:
    """High-performance multiprocessing climate data processor."""
    
    def __init__(self, data_directory: str, config: Optional[ProcessingConfig] = None):
        """
        Initialize the multiprocessing climate data processor.
        
        Args:
            data_directory: Path to input NetCDF files
            config: Processing configuration (auto-configured if None)
        """
        self.data_directory = data_directory
        self.file_handler = NorESM2FileHandler(data_directory)
        self.config = config or self._auto_configure()
        
        logger.info(f"Initialized multiprocessor with {self.config.max_workers} workers")
        logger.info(f"Memory allocation: {self.config.memory_per_worker_gb:.1f}GB per worker")
    
    def _auto_configure(self) -> ProcessingConfig:
        """Automatically configure based on system resources."""
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1e9
        
        # Optimal worker allocation based on testing
        # Testing showed 6 workers is optimal for this system (4.3x speedup, 72% efficiency)
        max_workers_by_memory = max(1, int(memory_gb / 4))  # 4GB per worker
        max_workers_by_cpu = max(1, cpu_count - 2)  # Leave 2 CPUs free
        
        # Use 6 workers as optimal (from optimization testing)
        # Cap at 6 unless system constraints require fewer
        optimal_workers = min(6, max_workers_by_memory, max_workers_by_cpu)
        
        config = ProcessingConfig(
            max_workers=optimal_workers,
            memory_per_worker_gb=min(4.0, memory_gb / optimal_workers * 0.8)
        )
        
        logger.info(f"Auto-configuration: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
        logger.info(f"Selected {optimal_workers} workers (optimal: 6, memory limited: {max_workers_by_memory}, CPU limited: {max_workers_by_cpu})")
        
        return config
    
    def process_files_parallel(
        self, 
        file_list: List[str], 
        output_path: str,
        variable: str = 'pr',
        region: str = 'CONUS'
    ) -> Dict[str, Any]:
        """Process multiple files in parallel and combine results."""
        
        logger.info(f"Starting parallel processing of {len(file_list)} files")
        logger.info(f"Using {self.config.max_workers} workers")
        
        start_time = time.time()
        results = []
        failed_files = []
        processing_times = []
        
        # Monitor system resources
        initial_memory = psutil.virtual_memory().percent
        
        try:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(
                        process_single_file_worker, 
                        file_path, 
                        variable, 
                        region,
                        self.config.timeout_per_file
                    ): file_path 
                    for file_path in file_list
                }
                
                # Collect results with progress tracking
                completed_count = 0
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    completed_count += 1
                    
                    try:
                        result_data, exec_time, success = future.result()
                        processing_times.append(exec_time)
                        
                        if success and result_data is not None:
                            # Reconstruct xarray from serialized data
                            data, coords, attrs, dims = result_data
                            daily_clim = xr.DataArray(
                                data, 
                                coords=coords, 
                                dims=dims,
                                attrs=attrs
                            )
                            results.append(daily_clim)
                            
                        else:
                            failed_files.append(file_path)
                            logger.warning(f"Failed to process {Path(file_path).name}")
                            
                    except Exception as e:
                        failed_files.append(file_path)
                        logger.error(f"Exception processing {Path(file_path).name}: {e}")
                    
                    # Progress reporting
                    if completed_count % self.config.progress_interval == 0:
                        current_memory = psutil.virtual_memory().percent
                        avg_time = np.mean(processing_times[-self.config.progress_interval:])
                        logger.info(f"Progress: {completed_count}/{len(file_list)} files "
                                  f"(avg: {avg_time:.1f}s/file, memory: {current_memory:.1f}%)")
        
        except Exception as e:
            logger.error(f"Critical error in parallel processing: {e}")
            raise
        
        # Calculate final statistics
        total_time = time.time() - start_time
        final_memory = psutil.virtual_memory().percent
        memory_delta = final_memory - initial_memory
        
        # Combine results into climate normal
        if results:
            logger.info(f"Combining {len(results)} successful results...")
            
            # Stack all years and compute mean
            combined_data = xr.concat(results, dim='year')
            climate_normal = combined_data.mean(dim='year', keep_attrs=True)
            
            # Add metadata
            climate_normal.attrs.update({
                'description': f'{len(results)}-year climate normal',
                'years_processed': len(results),
                'failed_files': len(failed_files),
                'processing_time_minutes': total_time / 60,
                'region': region,
                'variable': variable
            })
            
            # Save result
            climate_normal.to_netcdf(output_path)
            logger.info(f"Climate normal saved to {output_path}")
            
        else:
            climate_normal = None
            logger.error("No files processed successfully!")
        
        # Summary statistics
        stats = {
            'total_files': len(file_list),
            'successful_files': len(results),
            'failed_files': len(failed_files),
            'total_time_minutes': total_time / 60,
            'avg_time_per_file': np.mean(processing_times) if processing_times else 0,
            'speedup_estimate': len(file_list) * np.mean(processing_times) / total_time if processing_times else 0,
            'memory_delta_percent': memory_delta,
            'workers_used': self.config.max_workers
        }
        
        # Log summary
        logger.info(f"\n=== Processing Summary ===")
        logger.info(f"Processed: {stats['successful_files']}/{stats['total_files']} files")
        logger.info(f"Total time: {stats['total_time_minutes']:.1f} minutes")
        logger.info(f"Average per file: {stats['avg_time_per_file']:.1f} seconds")
        logger.info(f"Estimated speedup: {stats['speedup_estimate']:.1f}x")
        logger.info(f"Memory change: {stats['memory_delta_percent']:+.1f}%")
        
        if failed_files:
            logger.warning(f"Failed files: {[Path(f).name for f in failed_files[:5]]}")
        
        return {
            'climate_normal': climate_normal,
            'statistics': stats,
            'failed_files': failed_files
        }
    
    def process_historical_precipitation_parallel(self, start_year: int = 1950, end_year: int = 2014) -> str:
        """Process historical precipitation data using multiprocessing."""
        
        logger.info(f"Processing historical precipitation {start_year}-{end_year} with multiprocessing")
        
        # Get all files for the period
        file_list = self.file_handler.get_files_for_period('pr', 'historical', start_year, end_year)
        
        if not file_list:
            raise ValueError(f"No files found for historical period {start_year}-{end_year}")
        
        logger.info(f"Found {len(file_list)} files to process")
        
        # Output path
        output_dir = Path("output/multiprocessing_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"pr_CONUS_historical_{start_year}-{end_year}_climate_normal_mp.nc"
        
        # Process in parallel
        result = self.process_files_parallel(file_list, str(output_path), 'pr', 'CONUS')
        
        return str(output_path)
    
    def process_hybrid_normals_parallel(self, start_year: int = 2015, end_year: int = 2044) -> str:
        """Process hybrid climate normals using multiprocessing."""
        
        logger.info(f"Processing hybrid 30-year normals {start_year}-{end_year} with multiprocessing")
        
        # Get hybrid files (combining historical and SSP245)
        all_files = []
        target_years = list(range(start_year, end_year + 1))
        
        for target_year in target_years:
            period_start = target_year - 29  # 30-year period
            
            hybrid_files = self.file_handler.get_hybrid_files_for_period(
                'pr', period_start, target_year
            )
            
            if len(hybrid_files) >= 25:  # At least 25 years for a reasonable normal
                all_files.extend(hybrid_files)
            else:
                logger.warning(f"Insufficient files for target year {target_year}: {len(hybrid_files)} files")
        
        if not all_files:
            raise ValueError(f"No valid files found for hybrid normals {start_year}-{end_year}")
        
        logger.info(f"Found {len(all_files)} files for hybrid processing")
        
        # Output path
        output_dir = Path("output/multiprocessing_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"pr_CONUS_hybrid_{start_year}-{end_year}_climate_normal_mp.nc"
        
        # Process in parallel
        result = self.process_files_parallel(all_files, str(output_path), 'pr', 'CONUS')
        
        return str(output_path)


def process_single_file_worker(
    file_path: str, 
    variable: str = 'pr', 
    region: str = 'CONUS',
    timeout: int = 300
) -> Tuple[Optional[Tuple], float, bool]:
    """Worker function for processing a single file - must be at module level."""
    
    start_time = time.time()
    
    try:
        # Use the optimal approach (baseline, no chunking)
        ds = xr.open_dataset(file_path, decode_times=False, cache=False)
        ds, _ = handle_time_coordinates(ds, file_path)
        
        # Extract region
        region_bounds = REGION_BOUNDS[region]
        region_ds = extract_region(ds, region_bounds)
        
        # Get the variable
        var_data = getattr(region_ds, variable)
        
        if 'dayofyear' in var_data.coords:
            # Calculate daily climatology
            daily_clim = var_data.groupby(var_data.dayofyear).mean(dim='time')
            
            # Serialize for multiprocessing (avoid xarray serialization issues)
            result_data = daily_clim.values
            result_coords = {
                'dayofyear': daily_clim.dayofyear.values,
                'lat': daily_clim.lat.values,
                'lon': daily_clim.lon.values
            }
            result_attrs = dict(daily_clim.attrs)
            result_dims = list(daily_clim.dims)
            
            # Cleanup
            ds.close()
            region_ds.close()
            del ds, region_ds, var_data, daily_clim
            gc.collect()
            
            execution_time = time.time() - start_time
            return (result_data, result_coords, result_attrs, result_dims), execution_time, True
        
        else:
            logger.warning(f"No dayofyear coordinate in {file_path}")
            ds.close()
            execution_time = time.time() - start_time
            return None, execution_time, False
    
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Worker error processing {Path(file_path).name}: {e}")
        return None, execution_time, False


def benchmark_multiprocessing_speedup(data_directory: str, num_files: int = 10):
    """Benchmark sequential vs multiprocessing performance."""
    
    logger.info(f"Benchmarking multiprocessing speedup with {num_files} files")
    
    processor = ClimateMultiprocessor(data_directory)
    
    # Get test files
    test_files = processor.file_handler.get_files_for_period('pr', 'historical', 2010, 2014)[:num_files]
    
    if len(test_files) < num_files:
        logger.warning(f"Only found {len(test_files)} files, using those")
        test_files = test_files[:len(test_files)]
    
    results = {}
    
    # Test sequential processing (1 worker)
    logger.info("Testing sequential processing...")
    processor.config.max_workers = 1
    
    start_time = time.time()
    seq_result = processor.process_files_parallel(
        test_files, 
        "output/multiprocessing_test/benchmark_sequential.nc", 
        'pr', 'CONUS'
    )
    sequential_time = time.time() - start_time
    results['sequential'] = {
        'time': sequential_time,
        'files': seq_result['statistics']['successful_files']
    }
    
    # Test multiprocessing (auto-configured workers)
    logger.info("Testing multiprocessing...")
    processor.config = processor._auto_configure()  # Reset to optimal config
    
    start_time = time.time()
    mp_result = processor.process_files_parallel(
        test_files,
        "output/multiprocessing_test/benchmark_parallel.nc",
        'pr', 'CONUS'
    )
    parallel_time = time.time() - start_time
    results['parallel'] = {
        'time': parallel_time,
        'files': mp_result['statistics']['successful_files'],
        'workers': processor.config.max_workers
    }
    
    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    
    logger.info(f"\n=== Speedup Benchmark Results ===")
    logger.info(f"Sequential: {sequential_time:.1f}s ({results['sequential']['files']} files)")
    logger.info(f"Parallel ({results['parallel']['workers']} workers): {parallel_time:.1f}s ({results['parallel']['files']} files)")
    logger.info(f"Speedup: {speedup:.1f}x")
    
    return results


if __name__ == "__main__":
    # Add any additional main function logic here
    pass 