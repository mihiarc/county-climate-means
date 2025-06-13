#!/usr/bin/env python3
"""
Parallel Variables Climate Processor

Coordinates parallel processing of multiple climate variables, with each variable
processing its years sequentially to avoid NetCDF file conflicts.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from county_climate.means.core.single_variable_processor import process_single_variable
from county_climate.means.utils.io_util import NorESM2FileHandler
from county_climate.means.utils.mp_progress import MultiprocessingProgressTracker, ProgressReporter
from county_climate.means.core.regions import REGION_BOUNDS


class ParallelVariablesProcessor:
    """Processes multiple climate variables in parallel for a region/scenario."""
    
    def __init__(
        self,
        region_key: str,
        variables: List[str],
        scenario: str,
        input_data_dir: Path,
        output_base_dir: Path,
        max_workers: Optional[int] = None,
        min_years_for_normal: int = 25,
        use_progress_tracking: bool = True
    ):
        """Initialize the parallel variables processor.
        
        Args:
            region_key: Region identifier (CONUS, AK, HI, PRVI, GU)
            variables: List of climate variables to process
            scenario: Climate scenario (historical, ssp245, ssp585)
            input_data_dir: Base directory for input climate data
            output_base_dir: Base directory for output files
            max_workers: Maximum number of parallel workers (default: num_variables)
            min_years_for_normal: Minimum years required for climate normal
            use_progress_tracking: Whether to use Rich progress display
        """
        self.region_key = region_key
        self.variables = variables
        self.scenario = scenario
        self.input_data_dir = input_data_dir
        self.output_base_dir = output_base_dir
        self.max_workers = max_workers or len(variables)
        self.min_years_for_normal = min_years_for_normal
        self.use_progress_tracking = use_progress_tracking
        
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.{region_key}_{scenario}")
        
        # Progress tracking
        self.progress_tracker = None
        self.progress_queue = None
        
    def process_all_variables(self) -> Dict:
        """Process all variables in parallel.
        
        Returns:
            Dictionary with results for all variables
        """
        self.logger.info(f"Starting parallel processing for {len(self.variables)} variables in {self.region_key}/{self.scenario}")
        start_time = time.time()
        
        # Count total files for progress tracking
        if self.use_progress_tracking:
            file_counts = self._count_files_per_variable()
            total_files = sum(file_counts.values())
            
            # Initialize progress tracker
            region_name = REGION_BOUNDS[self.region_key]['name']
            self.progress_tracker = MultiprocessingProgressTracker(
                title=f"Climate Processing - {region_name} / {self.scenario} ({total_files:,} files)"
            )
            
            # Create a Manager for the progress queue to share across processes
            manager = mp.Manager()
            self.progress_queue = manager.Queue()
            
            # Start progress display process
            progress_process = mp.Process(
                target=self.progress_tracker._run_progress_display,
                args=(self.progress_queue, self.progress_tracker._shared_stats, self.progress_tracker.title)
            )
            progress_process.start()
            
            # Add tasks for each variable
            progress_reporter = ProgressReporter(self.progress_queue)
            for variable in self.variables:
                progress_reporter.create_task(
                    variable,
                    f"Processing {variable.upper()} data",
                    file_counts[variable]
                )
        
        # Prepare arguments for each variable processor
        processor_args = []
        for variable in self.variables:
            args = {
                'variable': variable,
                'region_key': self.region_key,
                'scenario': self.scenario,
                'input_data_dir': str(self.input_data_dir),
                'output_base_dir': str(self.output_base_dir),
                'min_years_for_normal': self.min_years_for_normal,
                'progress_queue': self.progress_queue if self.use_progress_tracking else None
            }
            processor_args.append(args)
        
        # Process variables in parallel
        results = {}
        
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all variables for processing
                future_to_variable = {
                    executor.submit(process_single_variable, args): args['variable']
                    for args in processor_args
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_variable):
                    variable = future_to_variable[future]
                    try:
                        result = future.result()
                        results[variable] = result
                        
                        if result['status'] == 'success':
                            self.logger.info(f"✅ Completed {variable} - processed {len(result['target_years_processed'])} years")
                        else:
                            self.logger.error(f"❌ Failed {variable} - {result.get('errors', 'Unknown error')}")
                            
                        # Mark task as complete
                        if self.use_progress_tracking:
                            status = "completed" if result['status'] == 'success' else "failed"
                            progress_reporter.complete_task(variable)
                            
                    except Exception as e:
                        self.logger.error(f"Exception processing {variable}: {e}")
                        results[variable] = {
                            'status': 'error',
                            'error': str(e)
                        }
                        
                        if self.use_progress_tracking:
                            progress_reporter.fail_task(variable, str(e))
        
        finally:
            # Stop progress tracking
            if self.use_progress_tracking and self.progress_queue:
                # Send stop signal
                from county_climate.means.utils.mp_progress import ProgressUpdate
                self.progress_queue.put(ProgressUpdate("__STOP__", "stop"))
                
                # Wait for progress process to finish
                progress_process.join(timeout=5)
                if progress_process.is_alive():
                    progress_process.terminate()
        
        # Calculate summary statistics
        total_time = time.time() - start_time
        successful_vars = [v for v, r in results.items() if r.get('status') == 'success']
        failed_vars = [v for v, r in results.items() if r.get('status') != 'success']
        
        summary = {
            'region': self.region_key,
            'scenario': self.scenario,
            'total_variables': len(self.variables),
            'successful_variables': len(successful_vars),
            'failed_variables': len(failed_vars),
            'processing_time_seconds': total_time,
            'processing_time_minutes': total_time / 60,
            'variables': results
        }
        
        # Log summary
        self.logger.info(f"🎉 Completed {self.region_key}/{self.scenario} in {total_time/60:.1f} minutes")
        self.logger.info(f"✅ Success: {successful_vars}")
        if failed_vars:
            self.logger.error(f"❌ Failed: {failed_vars}")
            
        return summary
    
    def _count_files_per_variable(self) -> Dict[str, int]:
        """Count total files to process for each variable.
        
        Returns:
            Dictionary mapping variable name to file count
        """
        file_counts = {}
        file_handler = NorESM2FileHandler(self.input_data_dir)
        
        # Determine target years based on scenario
        if self.scenario == 'historical':
            target_years = list(range(1980, 2015))
        else:
            target_years = list(range(2015, 2101))
        
        for variable in self.variables:
            total_files = 0
            
            for target_year in target_years:
                # Calculate 30-year window
                start_year = target_year - 29
                end_year = target_year
                
                # Get files for this window
                if self.scenario == 'historical' and target_year >= 2015:
                    # Hybrid period
                    hist_files = file_handler.get_files_for_period(
                        variable, 'historical', max(1985, start_year), min(2014, end_year)
                    )
                    ssp_files = file_handler.get_files_for_period(
                        variable, 'ssp245', max(2015, start_year), end_year
                    )
                    files = hist_files + ssp_files
                else:
                    # Single scenario period
                    files = file_handler.get_files_for_period(
                        variable, self.scenario, start_year, end_year
                    )
                
                total_files += len(files)
            
            file_counts[variable] = total_files
            self.logger.info(f"📁 {variable}: {total_files:,} files to process")
            
        return file_counts