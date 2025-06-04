#!/usr/bin/env python3
"""
Safe Optimization Exploration for Climate Data Processing

This script focuses on safe, proven optimization strategies that don't cause
threading issues with NetCDF/HDF5 libraries.

Key optimization areas:
1. Chunking strategies
2. Memory management
3. Algorithm optimization
4. I/O patterns
5. Simple profiling
"""

import logging
import time
import psutil
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import xarray as xr
from typing import List, Dict, Any, Optional, Tuple
import cProfile
import pstats
from functools import wraps
import tracemalloc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from src.io_util import NorESM2FileHandler
from src.regions import REGION_BOUNDS, extract_region
from src.time_util import handle_time_coordinates


class SafeOptimizer:
    """Safe optimization strategies that avoid threading issues."""
    
    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.file_handler = NorESM2FileHandler(data_directory)
    
    def time_function(self, func, *args, **kwargs):
        """Time a function execution."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function took {execution_time:.2f} seconds")
        return result, execution_time
    
    def monitor_memory(self):
        """Monitor current memory usage."""
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'used_gb': memory.used / 1e9,
            'available_gb': memory.available / 1e9
        }
    
    def process_file_baseline(self, file_path: str) -> Optional[xr.DataArray]:
        """Baseline file processing (original method)."""
        try:
            ds = xr.open_dataset(file_path, decode_times=False)
            ds, _ = handle_time_coordinates(ds, file_path)
            
            conus_bounds = REGION_BOUNDS['CONUS']
            region_ds = extract_region(ds, conus_bounds)
            
            pr = region_ds.pr
            if 'dayofyear' in pr.coords:
                daily_clim = pr.groupby(pr.dayofyear).mean(dim='time')
                ds.close()
                return daily_clim
            
            ds.close()
            return None
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def process_file_optimized_chunks(self, file_path: str, chunk_strategy: Dict[str, int]) -> Optional[xr.DataArray]:
        """Optimized file processing with custom chunking."""
        try:
            ds = xr.open_dataset(
                file_path, 
                chunks=chunk_strategy,
                decode_times=False,
                cache=False
            )
            
            ds, _ = handle_time_coordinates(ds, file_path)
            
            conus_bounds = REGION_BOUNDS['CONUS']
            region_ds = extract_region(ds, conus_bounds)
            
            pr = region_ds.pr
            if 'dayofyear' in pr.coords:
                daily_clim = pr.groupby(pr.dayofyear).mean(dim='time', keep_attrs=True)
                daily_clim = daily_clim.compute()  # Force computation
                ds.close()
                return daily_clim
            
            ds.close()
            return None
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def process_file_memory_efficient(self, file_path: str) -> Optional[xr.DataArray]:
        """Memory-efficient processing with aggressive cleanup."""
        try:
            # Use very conservative chunks
            chunks = {'time': 180, 'lat': 25, 'lon': 25}
            
            ds = xr.open_dataset(file_path, chunks=chunks, decode_times=False, cache=False)
            ds, _ = handle_time_coordinates(ds, file_path)
            
            # Extract region first to reduce data size
            conus_bounds = REGION_BOUNDS['CONUS']
            region_ds = extract_region(ds, conus_bounds)
            
            # Close original dataset immediately
            ds.close()
            del ds
            
            pr = region_ds.pr
            if 'dayofyear' in pr.coords:
                # Process in smaller chunks if dataset is large
                if pr.nbytes > 500e6:  # If larger than 500MB
                    logger.info("Large dataset detected, processing in temporal chunks")
                    # Split by seasons and process separately
                    seasonal_results = []
                    for season_start in range(1, 366, 90):  # 4 seasons
                        season_end = min(season_start + 89, 365)
                        season_mask = (pr.dayofyear >= season_start) & (pr.dayofyear <= season_end)
                        if season_mask.any():
                            season_pr = pr.where(season_mask, drop=True)
                            season_clim = season_pr.groupby(season_pr.dayofyear).mean(dim='time')
                            seasonal_results.append(season_clim.compute())
                            del season_pr, season_clim
                            gc.collect()
                    
                    # Combine seasonal results
                    daily_clim = xr.concat(seasonal_results, dim='dayofyear').sortby('dayofyear')
                    del seasonal_results
                else:
                    daily_clim = pr.groupby(pr.dayofyear).mean(dim='time', keep_attrs=True)
                    daily_clim = daily_clim.compute()
                
                region_ds.close()
                del region_ds, pr
                gc.collect()
                
                return daily_clim
            
            region_ds.close()
            return None
            
        except Exception as e:
            logger.error(f"Error in memory-efficient processing {file_path}: {e}")
            return None
    
    def benchmark_chunking_strategies(self, test_file: str):
        """Test different chunking strategies."""
        logger.info("=== Benchmarking Chunking Strategies ===")
        
        chunk_strategies = {
            "baseline": None,  # No chunking
            "time_heavy": {'time': -1, 'lat': 20, 'lon': 20},
            "spatial_heavy": {'time': 100, 'lat': -1, 'lon': -1},
            "balanced": {'time': 365, 'lat': 50, 'lon': 50},
            "memory_conservative": {'time': 180, 'lat': 25, 'lon': 25},
            "very_conservative": {'time': 90, 'lat': 15, 'lon': 15}
        }
        
        results = {}
        
        for strategy_name, chunks in chunk_strategies.items():
            logger.info(f"Testing {strategy_name} strategy...")
            
            # Monitor memory before
            mem_before = self.monitor_memory()
            
            try:
                if strategy_name == "baseline":
                    result, exec_time = self.time_function(self.process_file_baseline, test_file)
                else:
                    result, exec_time = self.time_function(
                        self.process_file_optimized_chunks, test_file, chunks
                    )
                
                # Monitor memory after
                mem_after = self.monitor_memory()
                mem_delta = mem_after['used_gb'] - mem_before['used_gb']
                
                if result is not None:
                    results[strategy_name] = {
                        'time': exec_time,
                        'memory_delta': mem_delta,
                        'shape': result.shape,
                        'success': True
                    }
                    logger.info(f"  ✓ {exec_time:.2f}s, {mem_delta:+.1f}GB memory, shape: {result.shape}")
                else:
                    results[strategy_name] = {'success': False}
                    logger.info(f"  ✗ Failed")
                
                # Cleanup
                del result
                gc.collect()
                
            except Exception as e:
                results[strategy_name] = {'success': False, 'error': str(e)}
                logger.error(f"  ✗ Failed: {e}")
        
        # Summary
        logger.info("\n=== Chunking Strategy Results ===")
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}
        
        if successful_results:
            # Find fastest
            fastest = min(successful_results.items(), key=lambda x: x[1]['time'])
            logger.info(f"Fastest: {fastest[0]} ({fastest[1]['time']:.2f}s)")
            
            # Find most memory efficient
            most_efficient = min(successful_results.items(), key=lambda x: x[1]['memory_delta'])
            logger.info(f"Most memory efficient: {most_efficient[0]} ({most_efficient[1]['memory_delta']:+.1f}GB)")
        
        return results
    
    def benchmark_memory_processing(self, test_files: List[str]):
        """Test memory-efficient processing on multiple files."""
        logger.info("=== Benchmarking Memory-Efficient Processing ===")
        
        # Test baseline vs memory-efficient
        methods = {
            'baseline': self.process_file_baseline,
            'memory_efficient': self.process_file_memory_efficient
        }
        
        for method_name, method_func in methods.items():
            logger.info(f"Testing {method_name} method...")
            
            mem_initial = self.monitor_memory()
            start_time = time.time()
            
            processed_count = 0
            for i, file_path in enumerate(test_files):
                logger.info(f"  Processing file {i+1}/{len(test_files)}")
                
                result = method_func(file_path)
                if result is not None:
                    processed_count += 1
                
                # Monitor memory every 2 files
                if i % 2 == 1:
                    mem_current = self.monitor_memory()
                    mem_delta = mem_current['used_gb'] - mem_initial['used_gb']
                    logger.info(f"    Memory delta: {mem_delta:+.1f}GB")
                
                del result
                gc.collect()
            
            total_time = time.time() - start_time
            mem_final = self.monitor_memory()
            final_delta = mem_final['used_gb'] - mem_initial['used_gb']
            
            logger.info(f"{method_name} results:")
            logger.info(f"  Processed: {processed_count}/{len(test_files)} files")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Avg time per file: {total_time/len(test_files):.2f}s")
            logger.info(f"  Final memory delta: {final_delta:+.1f}GB")
    
    def benchmark_multiprocessing(self, test_files: List[str]):
        """Test multiprocessing vs sequential processing."""
        logger.info("=== Benchmarking Multiprocessing vs Sequential ===")
        
        # Determine number of workers based on system
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1e9
        
        # Conservative worker count (each file needs ~3GB RAM)
        max_workers_by_memory = max(1, int(memory_gb / 4))  # 4GB per worker for safety
        max_workers_by_cpu = max(1, cpu_count - 1)  # Leave one CPU free
        
        suggested_workers = min(max_workers_by_memory, max_workers_by_cpu, 4)  # Cap at 4
        
        logger.info(f"System: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
        logger.info(f"Suggested workers: {suggested_workers} (memory limited: {max_workers_by_memory}, CPU limited: {max_workers_by_cpu})")
        
        # Test different worker counts
        worker_counts = [1, 2, suggested_workers] if suggested_workers > 2 else [1, 2]
        worker_counts = sorted(set(worker_counts))  # Remove duplicates
        
        results = {}
        
        # Sequential baseline
        logger.info("Testing sequential processing...")
        mem_before = self.monitor_memory()
        start_time = time.time()
        
        sequential_results = []
        for file_path in test_files:
            result = self.process_file_baseline(file_path)
            if result is not None:
                sequential_results.append(result)
        
        sequential_time = time.time() - start_time
        mem_after = self.monitor_memory()
        sequential_memory_delta = mem_after['used_gb'] - mem_before['used_gb']
        
        results['sequential'] = {
            'time': sequential_time,
            'files_processed': len(sequential_results),
            'memory_delta': sequential_memory_delta,
            'avg_time_per_file': sequential_time / len(test_files)
        }
        
        logger.info(f"Sequential: {len(sequential_results)}/{len(test_files)} files in {sequential_time:.2f}s")
        logger.info(f"  Avg per file: {sequential_time/len(test_files):.2f}s, Memory: {sequential_memory_delta:+.1f}GB")
        
        # Clean up
        del sequential_results
        gc.collect()
        
        # Test multiprocessing with different worker counts
        for num_workers in worker_counts:
            if num_workers == 1:
                continue  # Skip 1 worker (same as sequential)
                
            logger.info(f"Testing multiprocessing with {num_workers} workers...")
            mem_before = self.monitor_memory()
            start_time = time.time()
            
            try:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all tasks
                    futures = [executor.submit(process_single_file_worker, file_path) for file_path in test_files]
                    
                    # Collect results
                    multiprocessing_results = []
                    individual_times = []
                    
                    for future in futures:
                        try:
                            file_path, result_data, exec_time = future.result(timeout=300)  # 5 min timeout
                            individual_times.append(exec_time)
                            
                            if result_data is not None:
                                # Reconstruct xarray from serialized data
                                data, coords, attrs = result_data
                                result = xr.DataArray(
                                    data, 
                                    coords=coords, 
                                    dims=['dayofyear', 'lat', 'lon'],
                                    attrs=attrs
                                )
                                multiprocessing_results.append(result)
                                
                        except Exception as e:
                            logger.error(f"Failed to get result: {e}")
                
                total_time = time.time() - start_time
                mem_after = self.monitor_memory()
                memory_delta = mem_after['used_gb'] - mem_before['used_gb']
                
                # Calculate statistics
                avg_worker_time = np.mean(individual_times) if individual_times else 0
                max_worker_time = np.max(individual_times) if individual_times else 0
                
                results[f'multiprocessing_{num_workers}'] = {
                    'time': total_time,
                    'files_processed': len(multiprocessing_results),
                    'memory_delta': memory_delta,
                    'avg_time_per_file': total_time / len(test_files),
                    'avg_worker_time': avg_worker_time,
                    'max_worker_time': max_worker_time
                }
                
                speedup = sequential_time / total_time if total_time > 0 else 0
                
                logger.info(f"Multiprocessing ({num_workers} workers): {len(multiprocessing_results)}/{len(test_files)} files in {total_time:.2f}s")
                logger.info(f"  Speedup: {speedup:.2f}x, Memory: {memory_delta:+.1f}GB")
                logger.info(f"  Avg worker time: {avg_worker_time:.2f}s, Max worker time: {max_worker_time:.2f}s")
                
                # Clean up
                del multiprocessing_results
                gc.collect()
                
            except Exception as e:
                logger.error(f"Multiprocessing with {num_workers} workers failed: {e}")
                results[f'multiprocessing_{num_workers}'] = {'failed': True, 'error': str(e)}
        
        # Summary
        logger.info("\n=== Multiprocessing Results Summary ===")
        
        if 'sequential' in results:
            seq_time = results['sequential']['time']
            logger.info(f"Sequential baseline: {seq_time:.2f}s")
            
            for key, result in results.items():
                if key.startswith('multiprocessing_') and not result.get('failed', False):
                    speedup = seq_time / result['time']
                    workers = key.split('_')[1]
                    logger.info(f"  {workers} workers: {result['time']:.2f}s (speedup: {speedup:.2f}x)")
        
        return results
    
    def run_optimization_analysis(self):
        """Run comprehensive optimization analysis."""
        logger.info("Starting optimization analysis...")
        
        # Get test files
        test_files = self.file_handler.get_files_for_period('pr', 'historical', 2012, 2014)[:6]
        
        if not test_files:
            logger.error("No test files found!")
            return
        
        logger.info(f"Using {len(test_files)} test files")
        
        # Run benchmarks
        chunking_results = self.benchmark_chunking_strategies(test_files[0])
        self.benchmark_memory_processing(test_files[:4])
        self.benchmark_multiprocessing(test_files[:4])
        
        # Recommendations
        logger.info("\n=== Optimization Recommendations ===")
        
        successful_chunking = {k: v for k, v in chunking_results.items() if v.get('success', False)}
        if successful_chunking:
            fastest = min(successful_chunking.items(), key=lambda x: x[1]['time'])
            most_efficient = min(successful_chunking.items(), key=lambda x: x[1]['memory_delta'])
            
            logger.info(f"1. For speed: Use {fastest[0]} chunking strategy")
            logger.info(f"2. For memory efficiency: Use {most_efficient[0]} chunking strategy")
        
        logger.info("3. Always use cache=False to avoid memory buildup")
        logger.info("4. Force garbage collection between files")
        logger.info("5. Close datasets immediately after processing")
        logger.info("6. For large datasets, consider seasonal processing")
        logger.info("7. Avoid threading with NetCDF files (use sequential processing)")


def process_single_file_worker(file_path: str) -> Tuple[str, Optional[xr.DataArray], float]:
    """Worker function for multiprocessing - must be at module level."""
    start_time = time.time()
    
    try:
        # Use baseline approach (no chunking) since it was fastest
        ds = xr.open_dataset(file_path, decode_times=False, cache=False)
        ds, _ = handle_time_coordinates(ds, file_path)
        
        conus_bounds = REGION_BOUNDS['CONUS']
        region_ds = extract_region(ds, conus_bounds)
        
        pr = region_ds.pr
        if 'dayofyear' in pr.coords:
            daily_clim = pr.groupby(pr.dayofyear).mean(dim='time')
            
            # Convert to plain numpy arrays to avoid serialization issues
            result_data = daily_clim.values
            result_coords = {
                'dayofyear': daily_clim.dayofyear.values,
                'lat': daily_clim.lat.values, 
                'lon': daily_clim.lon.values
            }
            result_attrs = dict(daily_clim.attrs)
            
            ds.close()
            region_ds.close()
            
            execution_time = time.time() - start_time
            return file_path, (result_data, result_coords, result_attrs), execution_time
        
        ds.close()
        execution_time = time.time() - start_time
        return file_path, None, execution_time
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Worker error processing {file_path}: {e}")
        return file_path, None, execution_time


def main():
    """Main optimization function."""
    data_directory = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
    
    if not Path(data_directory).exists():
        logger.error(f"Data directory not found: {data_directory}")
        return
    
    optimizer = SafeOptimizer(data_directory)
    optimizer.run_optimization_analysis()


if __name__ == "__main__":
    main() 