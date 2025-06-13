#!/usr/bin/env python3
"""
Maximum Performance Climate Processor

A high-performance climate data processor designed for high-end systems with
extensive RAM and CPU resources. Integrated into the means package architecture.

Originally from maximum_processing.py, now properly integrated with:
- Consistent with means package patterns
- Uses existing configuration system
- Integrates with rich progress tracking
- Follows package import structure
"""

import sys
import time
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import xarray as xr
import numpy as np
from datetime import datetime
import logging
import psutil
from typing import List, Dict, Any, Optional, Tuple

# Import our modules
from county_climate.means.utils.io_util import NorESM2FileHandler
from county_climate.means.core.regions import REGION_BOUNDS, extract_region
from county_climate.means.config import get_config
from county_climate.means.utils.rich_progress import RichProgressTracker

logger = logging.getLogger(__name__)


class MaximumPerformanceProcessor:
    """
    Maximum performance climate processor designed for high-end systems.
    
    Integrated version of the original maximum_processing.py script,
    now properly integrated with the means package architecture.
    """
    
    def __init__(self, max_workers=48, memory_limit_gb=80, use_rich_progress=True):
        self.max_workers = max_workers
        self.memory_limit_gb = memory_limit_gb
        self.config = get_config()
        self.file_handler = NorESM2FileHandler(self.config.paths.input_data_dir)
        self.use_rich_progress = use_rich_progress
        
        # System info
        self.total_ram = psutil.virtual_memory().total / (1024**3)
        self.cpu_count = psutil.cpu_count()
        
        # Rich progress tracker
        self.rich_tracker = None
        if self.use_rich_progress:
            self.rich_tracker = RichProgressTracker(
                title="Maximum Performance Climate Processing",
                show_system_stats=True,
                update_interval=0.5
            )
        
        logger.info(f"🚀 Maximum Performance Processor Initialized")
        logger.info(f"💾 System RAM: {self.total_ram:.1f} GB")
        logger.info(f"🔧 CPU Cores: {self.cpu_count}")
        logger.info(f"⚡ Max Workers: {self.max_workers}")
        logger.info(f"🎯 Memory Limit: {self.memory_limit_gb} GB")
    
    def process_single_file_chunk(self, args):
        """Process a single file for a specific region and variable."""
        file_path, variable, region_key, output_dir = args
        
        try:
            # Load data
            ds = xr.open_dataset(file_path)
            
            # Extract region
            region_bounds = REGION_BOUNDS[region_key]
            region_ds = extract_region(ds, region_bounds)
            
            # Calculate daily climatology
            if variable in region_ds.data_vars:
                var_data = region_ds[variable]
                
                # Add dayofyear coordinate if missing
                if 'dayofyear' not in var_data.coords:
                    var_data = var_data.assign_coords(dayofyear=var_data.time.dt.dayofyear)
                
                # Calculate climatology
                daily_clim = var_data.groupby('dayofyear').mean(dim='time')
                
                # Save output
                output_file = output_dir / f"{variable}_{region_key}_{Path(file_path).stem}_climatology.nc"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                daily_clim.to_netcdf(output_file)
                
                # Cleanup
                ds.close()
                del ds, region_ds, var_data, daily_clim
                
                return {
                    'status': 'success',
                    'file': str(output_file),
                    'variable': variable,
                    'region': region_key,
                    'shape': daily_clim.shape if 'daily_clim' in locals() else None
                }
            else:
                ds.close()
                return {
                    'status': 'error',
                    'message': f'Variable {variable} not found in {file_path}',
                    'variable': variable,
                    'region': region_key
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'file': file_path,
                'variable': variable,
                'region': region_key
            }
    
    def get_all_processing_tasks(self, variables, regions, years_range=(1950, 2100)):
        """Generate all processing tasks for maximum throughput."""
        tasks = []
        
        for variable in variables:
            for region_key in regions:
                # Get all available files for this variable
                try:
                    # Historical files
                    hist_files = []
                    for year in range(years_range[0], min(2015, years_range[1] + 1)):
                        year_files = self.file_handler.get_files_for_period(
                            variable, 'historical', year, year
                        )
                        hist_files.extend(year_files)
                    
                    # Future scenario files (if available)
                    future_files = []
                    if years_range[1] > 2014:
                        for scenario in ['ssp245', 'ssp585']:
                            try:
                                for year in range(max(2015, years_range[0]), years_range[1] + 1):
                                    year_files = self.file_handler.get_files_for_period(
                                        variable, scenario, year, year
                                    )
                                    future_files.extend(year_files)
                            except:
                                continue
                    
                    all_files = hist_files + future_files
                    
                    # Create output directory using package configuration
                    output_base = Path(self.config.paths.output_base_dir) / "data"
                    output_dir = output_base / region_key / variable
                    
                    # Add tasks
                    for file_path in all_files:
                        tasks.append((file_path, variable, region_key, output_dir))
                        
                except Exception as e:
                    logger.warning(f"Could not get files for {variable} in {region_key}: {e}")
                    continue
        
        return tasks
    
    def run_maximum_processing(self, variables=None, regions=None, years_range=(1950, 2100)):
        """Run maximum performance processing across all available data."""
        
        if variables is None:
            variables = ['pr', 'tas', 'tasmax', 'tasmin']
        
        if regions is None:
            regions = ['CONUS', 'AK', 'HI', 'PRVI', 'GU']
        
        logger.info(f"🎯 Starting Maximum Performance Processing")
        logger.info(f"📊 Variables: {variables}")
        logger.info(f"🗺️  Regions: {regions}")
        logger.info(f"📅 Years: {years_range[0]}-{years_range[1]}")
        
        # Generate all tasks
        logger.info("📋 Generating processing tasks...")
        tasks = self.get_all_processing_tasks(variables, regions, years_range)
        
        logger.info(f"🚀 Total tasks to process: {len(tasks)}")
        logger.info(f"⚡ Using {self.max_workers} parallel workers")
        
        if not tasks:
            logger.error("❌ No tasks generated! Check data availability.")
            return None
        
        # Initialize rich progress tracking
        if self.rich_tracker:
            self.rich_tracker.start()
            
            # Add progress tasks by variable
            for variable in variables:
                var_count = sum(1 for _, v, _, _ in tasks if v == variable)
                if var_count > 0:
                    self.rich_tracker.add_task(
                        name=variable,
                        description=f"Processing {variable.upper()} files",
                        total=var_count
                    )
        
        # Process with maximum parallelism
        start_time = time.time()
        completed = 0
        failed = 0
        
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(self.process_single_file_chunk, task): task for task in tasks}
                
                # Process results as they complete
                for future in as_completed(future_to_task):
                    result = future.result()
                    task = future_to_task[future]
                    variable = task[1]
                    
                    if result['status'] == 'success':
                        completed += 1
                        
                        # Update rich progress
                        if self.rich_tracker:
                            self.rich_tracker.update_task(
                                variable,
                                advance=1,
                                current_item=f"Processed {Path(result['file']).name}"
                            )
                        
                        if completed % 50 == 0:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed
                            logger.info(f"✅ Completed {completed}/{len(tasks)} ({rate:.1f}/s)")
                    else:
                        failed += 1
                        
                        # Update rich progress for failed tasks
                        if self.rich_tracker:
                            self.rich_tracker.update_task(
                                variable,
                                advance=1,
                                current_item=f"Failed: {Path(task[0]).name}",
                                failed=True
                            )
                        
                        if failed % 10 == 0:
                            logger.warning(f"❌ Failed {failed} tasks")
        
        finally:
            # Complete all variable tasks and stop progress tracking
            if self.rich_tracker:
                for variable in variables:
                    self.rich_tracker.complete_task(variable, "completed")
                self.rich_tracker.stop()
        
        # Final statistics
        end_time = time.time()
        total_duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("🎊 MAXIMUM PROCESSING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        logger.info(f"📁 Total Tasks: {len(tasks)}")
        logger.info(f"✅ Completed: {completed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📈 Success Rate: {completed/len(tasks)*100:.1f}%")
        logger.info(f"🚀 Throughput: {completed/total_duration:.1f} tasks/second")
        logger.info(f"💾 Memory Usage: {psutil.virtual_memory().percent:.1f}%")
        logger.info(f"🔧 CPU Usage: {psutil.cpu_percent():.1f}%")
        
        return {
            'total_tasks': len(tasks),
            'completed': completed,
            'failed': failed,
            'duration': total_duration,
            'throughput': completed/total_duration if total_duration > 0 else 0,
            'success_rate': completed/len(tasks)*100 if len(tasks) > 0 else 0
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_maximum_processor(max_workers: int = 48,
                           memory_limit_gb: int = 80,
                           use_rich_progress: bool = True) -> MaximumPerformanceProcessor:
    """
    Create a maximum performance processor with specified settings.
    
    Args:
        max_workers: Maximum number of parallel workers
        memory_limit_gb: Memory limit in GB
        use_rich_progress: Whether to use rich progress tracking
        
    Returns:
        Configured MaximumPerformanceProcessor instance
    """
    return MaximumPerformanceProcessor(
        max_workers=max_workers,
        memory_limit_gb=memory_limit_gb,
        use_rich_progress=use_rich_progress
    )


def run_maximum_processing(variables: Optional[List[str]] = None,
                         regions: Optional[List[str]] = None,
                         years_range: Tuple[int, int] = (1950, 2100),
                         max_workers: int = 48,
                         memory_limit_gb: int = 80,
                         use_rich_progress: bool = True) -> Optional[Dict[str, Any]]:
    """
    Convenience function to run maximum performance processing.
    
    Args:
        variables: List of variables to process
        regions: List of regions to process
        years_range: Year range tuple (start, end)
        max_workers: Maximum number of workers
        memory_limit_gb: Memory limit in GB
        use_rich_progress: Whether to use rich progress tracking
        
    Returns:
        Processing results dictionary or None if failed
    """
    processor = create_maximum_processor(
        max_workers=max_workers,
        memory_limit_gb=memory_limit_gb,
        use_rich_progress=use_rich_progress
    )
    
    return processor.run_maximum_processing(
        variables=variables,
        regions=regions,
        years_range=years_range
    )


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Maximum Performance Climate Processing (Integrated)')
    parser.add_argument('--max-workers', type=int, default=48,
                       help='Maximum number of workers (default: 48)')
    parser.add_argument('--memory-limit', type=int, default=80,
                       help='Memory limit in GB (default: 80)')
    parser.add_argument('--variables', nargs='+', 
                       choices=['pr', 'tas', 'tasmax', 'tasmin'],
                       default=['pr', 'tas', 'tasmax', 'tasmin'],
                       help='Variables to process')
    parser.add_argument('--regions', nargs='+',
                       choices=['CONUS', 'AK', 'HI', 'PRVI', 'GU'],
                       default=['CONUS', 'AK', 'HI', 'PRVI', 'GU'],
                       help='Regions to process')
    parser.add_argument('--start-year', type=int, default=1950,
                       help='Start year for processing')
    parser.add_argument('--end-year', type=int, default=2100,
                       help='End year for processing')
    parser.add_argument('--no-rich-progress', action='store_true',
                       help='Disable rich progress tracking')
    
    args = parser.parse_args()
    
    print("🚀 MAXIMUM PERFORMANCE CLIMATE PROCESSING (INTEGRATED)")
    print("=" * 80)
    print("💪 Designed for high-end systems with means package integration")
    print(f"⚡ Variables: {args.variables}")
    print(f"🗺️  Regions: {args.regions}")
    print(f"📅 Years: {args.start_year}-{args.end_year}")
    print(f"🔧 Max workers: {args.max_workers}")
    print(f"💾 Memory limit: {args.memory_limit} GB")
    print("=" * 80)
    
    # Run processing
    results = run_maximum_processing(
        variables=args.variables,
        regions=args.regions,
        years_range=(args.start_year, args.end_year),
        max_workers=args.max_workers,
        memory_limit_gb=args.memory_limit,
        use_rich_progress=not args.no_rich_progress
    )
    
    if results and results['completed'] > 0:
        print(f"\n🎉 SUCCESS! Processed {results['completed']}/{results['total_tasks']} files")
        print(f"🚀 Throughput: {results['throughput']:.1f} files/second")
        print(f"📈 Success rate: {results['success_rate']:.1f}%")
        print(f"⏱️  Total time: {results['duration']:.1f} seconds ({results['duration']/60:.1f} minutes)")
        return 0
    else:
        print("\n❌ FAILED: Processing completed but no files were successfully processed.")
        print("💡 This might indicate data access issues or configuration problems.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main()) 