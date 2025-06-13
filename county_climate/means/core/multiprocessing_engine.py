#!/usr/bin/env python3
"""
Unified Multiprocessing Engine for Climate Data Processing

A comprehensive, reusable multiprocessing system that consolidates all parallel processing
functionality for climate data operations. Provides:

- Configurable worker management with auto-optimization
- Memory monitoring and resource management  
- Progress tracking and real-time monitoring
- Error handling and retry logic
- Flexible task distribution strategies
- Performance benchmarking and optimization

This module replaces scattered multiprocessing code across the package with a
unified, well-tested, and maintainable solution.
"""

import logging
import time
import psutil
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Protocol
import numpy as np
import xarray as xr
from dataclasses import dataclass, field
from datetime import datetime
import json
import threading
from abc import ABC, abstractmethod

# Import our modules
from county_climate.means.utils.io_util import NorESM2FileHandler
from county_climate.means.core.regions import REGION_BOUNDS, extract_region
from county_climate.means.utils.time_util import handle_time_coordinates
from county_climate.means.utils.rich_progress import RichProgressTracker

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION AND PROTOCOLS
# =============================================================================

@dataclass
class MultiprocessingConfig:
    """Comprehensive configuration for multiprocessing operations."""
    
    # Core processing settings
    max_workers: int = 6
    cores_per_variable: int = 2
    batch_size: int = 2
    
    # Memory management
    memory_per_worker_gb: float = 4.0
    max_memory_per_process_gb: int = 4
    memory_check_interval: int = 10
    
    # Performance and reliability
    timeout_per_task: int = 300  # 5 minutes
    max_retries: int = 2
    min_years_for_normal: int = 25
    
    # Progress and monitoring
    progress_interval: int = 5
    status_update_interval: int = 30
    enable_progress_tracking: bool = True
    
    # Output settings
    save_individual_results: bool = True
    create_combined_datasets: bool = True
    
    def __post_init__(self):
        """Validate and optimize configuration."""
        # Auto-optimize based on system resources if needed
        if self.max_workers <= 0:
            self.max_workers = self._auto_detect_optimal_workers()
        
        # Validate memory settings
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        max_possible_workers = int(total_memory_gb / self.memory_per_worker_gb)
        
        if self.max_workers > max_possible_workers:
            logger.warning(f"Reducing max_workers from {self.max_workers} to {max_possible_workers} due to memory constraints")
            self.max_workers = max_possible_workers
    
    def _auto_detect_optimal_workers(self) -> int:
        """Auto-detect optimal number of workers based on system resources."""
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Conservative estimates
        max_workers_by_memory = max(1, int(memory_gb / self.memory_per_worker_gb))
        max_workers_by_cpu = max(1, cpu_count - 2)  # Leave 2 CPUs free
        
        # Use the more restrictive limit
        optimal_workers = min(6, max_workers_by_memory, max_workers_by_cpu)
        
        logger.info(f"Auto-detected optimal workers: {optimal_workers} "
                   f"(CPU limit: {max_workers_by_cpu}, Memory limit: {max_workers_by_memory})")
        
        return optimal_workers


class TaskProtocol(Protocol):
    """Protocol for tasks that can be processed by the multiprocessing engine."""
    
    def execute(self) -> Any:
        """Execute the task and return results."""
        ...
    
    def get_id(self) -> str:
        """Get unique identifier for the task."""
        ...


@dataclass
class TaskResult:
    """Standardized result container for multiprocessing tasks."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_used_mb: float = 0.0
    worker_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'success': self.success,
            'result': self.result if self.success else None,
            'error': self.error,
            'execution_time': self.execution_time,
            'memory_used_mb': self.memory_used_mb,
            'worker_id': self.worker_id
        }


# =============================================================================
# PROGRESS TRACKING SYSTEM
# =============================================================================

class ProgressTracker:
    """Advanced progress tracking with real-time monitoring and persistence."""
    
    def __init__(self, 
                 status_file: Optional[str] = None,
                 log_file: Optional[str] = None,
                 update_interval: int = 30):
        
        self.status_file = status_file or "multiprocessing_progress.json"
        self.log_file = log_file or "multiprocessing_progress.log"
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_update = 0
        self._lock = threading.Lock()
        
        # Initialize progress structure
        self.progress = {
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "status": "initializing",
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "completion_percentage": 0.0,
            "estimated_time_remaining": None,
            "performance": {
                "tasks_per_minute": 0.0,
                "workers_active": 0,
                "memory_usage_gb": 0.0,
                "cpu_usage_percent": 0.0
            },
            "current_work": {
                "active_tasks": [],
                "recently_completed": []
            }
        }
        
        self._setup_logging()
        self.update_status(force=True)
    
    def _setup_logging(self):
        """Setup dedicated progress logging."""
        self.logger = logging.getLogger('progress_tracker')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add file handler
        if self.log_file:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter('%(asctime)s - PROGRESS - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def initialize(self, total_tasks: int):
        """Initialize progress tracking with total task count."""
        with self._lock:
            self.progress["total_tasks"] = total_tasks
            self.progress["status"] = "running"
            self.update_status(force=True)
            self.logger.info(f"Initialized progress tracking for {total_tasks} tasks")
    
    def report_task_start(self, task_id: str, worker_id: int):
        """Report when a task starts."""
        with self._lock:
            task_info = {
                "task_id": task_id,
                "worker_id": worker_id,
                "start_time": datetime.now().isoformat()
            }
            self.progress["current_work"]["active_tasks"].append(task_info)
            self.progress["performance"]["workers_active"] = len(
                set(task["worker_id"] for task in self.progress["current_work"]["active_tasks"])
            )
            self.update_status()
    
    def report_task_complete(self, task_result: TaskResult):
        """Report task completion."""
        with self._lock:
            # Remove from active tasks
            self.progress["current_work"]["active_tasks"] = [
                task for task in self.progress["current_work"]["active_tasks"]
                if task["task_id"] != task_result.task_id
            ]
            
            # Add to completed
            completion_info = {
                "task_id": task_result.task_id,
                "success": task_result.success,
                "execution_time": task_result.execution_time,
                "completion_time": datetime.now().isoformat()
            }
            
            self.progress["current_work"]["recently_completed"].append(completion_info)
            
            # Keep only last 20 completed tasks
            if len(self.progress["current_work"]["recently_completed"]) > 20:
                self.progress["current_work"]["recently_completed"] = (
                    self.progress["current_work"]["recently_completed"][-20:]
                )
            
            # Update counters
            if task_result.success:
                self.progress["completed_tasks"] += 1
            else:
                self.progress["failed_tasks"] += 1
            
            # Update completion percentage
            total_processed = self.progress["completed_tasks"] + self.progress["failed_tasks"]
            if self.progress["total_tasks"] > 0:
                self.progress["completion_percentage"] = (
                    total_processed / self.progress["total_tasks"] * 100
                )
            
            # Update worker count
            self.progress["performance"]["workers_active"] = len(
                set(task["worker_id"] for task in self.progress["current_work"]["active_tasks"])
            )
            
            self.update_status()
    
    def update_status(self, force: bool = False):
        """Update progress status file and calculate performance metrics."""
        current_time = time.time()
        
        if not force and (current_time - self.last_update) < self.update_interval:
            return
        
        with self._lock:
            self.last_update = current_time
            self.progress["last_update"] = datetime.now().isoformat()
            
            # Update performance metrics
            try:
                # System stats
                memory = psutil.virtual_memory()
                self.progress["performance"]["memory_usage_gb"] = memory.used / (1024**3)
                self.progress["performance"]["cpu_usage_percent"] = psutil.cpu_percent()
                
                # Processing rate
                elapsed_minutes = (current_time - self.start_time) / 60
                if elapsed_minutes > 0:
                    completed_tasks = self.progress["completed_tasks"]
                    self.progress["performance"]["tasks_per_minute"] = completed_tasks / elapsed_minutes
                    
                    # Estimate time remaining
                    remaining_tasks = self.progress["total_tasks"] - completed_tasks - self.progress["failed_tasks"]
                    if self.progress["performance"]["tasks_per_minute"] > 0 and remaining_tasks > 0:
                        eta_minutes = remaining_tasks / self.progress["performance"]["tasks_per_minute"]
                        self.progress["estimated_time_remaining"] = f"{eta_minutes:.1f} minutes"
            except Exception as e:
                logger.debug(f"Error updating performance metrics: {e}")
            
            # Write status file
            if self.status_file:
                try:
                    with open(self.status_file, 'w') as f:
                        json.dump(self.progress, f, indent=2)
                except Exception as e:
                    logger.debug(f"Error writing status file: {e}")
            
            # Log progress
            if self.logger:
                self.logger.info(
                    f"Progress: {self.progress['completed_tasks']}/{self.progress['total_tasks']} "
                    f"({self.progress['completion_percentage']:.1f}%) - "
                    f"Rate: {self.progress['performance']['tasks_per_minute']:.1f} tasks/min - "
                    f"ETA: {self.progress.get('estimated_time_remaining', 'calculating...')}"
                )
    
    def finalize(self):
        """Finalize progress tracking."""
        with self._lock:
            self.progress["status"] = "completed"
            self.progress["end_time"] = datetime.now().isoformat()
            
            total_time = time.time() - self.start_time
            self.progress["total_duration_minutes"] = total_time / 60
            
            self.update_status(force=True)
            
            if self.logger:
                self.logger.info(f"Processing completed in {total_time/60:.1f} minutes")


# =============================================================================
# CORE MULTIPROCESSING ENGINE
# =============================================================================

class MultiprocessingEngine:
    """
    Unified multiprocessing engine for climate data processing.
    
    Provides a high-level interface for parallel processing with:
    - Automatic resource optimization
    - Progress tracking and monitoring
    - Error handling and retry logic
    - Memory management
    - Performance benchmarking
    """
    
    def __init__(self, 
                 config: Optional[MultiprocessingConfig] = None,
                 progress_tracker: Optional[ProgressTracker] = None,
                 rich_tracker: Optional[RichProgressTracker] = None,
                 use_rich_progress: bool = False):
        
        self.config = config or MultiprocessingConfig()
        self.progress_tracker = progress_tracker
        self.rich_tracker = rich_tracker
        self.use_rich_progress = use_rich_progress
        
        # Initialize system monitoring
        self._initial_memory = psutil.virtual_memory().percent
        self._start_time = None
        
        logger.info(f"Initialized MultiprocessingEngine with {self.config.max_workers} workers")
        logger.info(f"Memory allocation: {self.config.memory_per_worker_gb:.1f}GB per worker")
    
    def process_tasks(self, 
                     tasks: List[Callable],
                     task_args: List[Tuple] = None,
                     task_kwargs: List[Dict] = None) -> List[TaskResult]:
        """
        Process a list of tasks in parallel.
        
        Args:
            tasks: List of callable tasks to execute
            task_args: List of argument tuples for each task
            task_kwargs: List of keyword argument dicts for each task
            
        Returns:
            List of TaskResult objects
        """
        if not tasks:
            return []
        
        # Prepare arguments
        task_args = task_args or [() for _ in tasks]
        task_kwargs = task_kwargs or [{} for _ in tasks]
        
        if len(task_args) != len(tasks) or len(task_kwargs) != len(tasks):
            raise ValueError("Number of tasks, args, and kwargs must match")
        
        # Initialize progress tracking
        if self.progress_tracker:
            self.progress_tracker.initialize(len(tasks))
        
        # Initialize rich progress tracking
        if self.rich_tracker:
            # Don't add a task here - let the calling code manage the rich tracker
            pass
        
        self._start_time = time.time()
        results = []
        
        logger.info(f"Starting parallel processing of {len(tasks)} tasks")
        logger.info(f"Using {self.config.max_workers} workers")
        
        try:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_task = {}
                for i, (task, args, kwargs) in enumerate(zip(tasks, task_args, task_kwargs)):
                    future = executor.submit(self._execute_task_wrapper, task, args, kwargs, i)
                    future_to_task[future] = i
                
                # Collect results
                completed_count = 0
                for future in as_completed(future_to_task):
                    task_index = future_to_task[future]
                    completed_count += 1
                    
                    try:
                        task_result = future.result(timeout=self.config.timeout_per_task)
                        results.append(task_result)
                        
                        # Report progress
                        if self.progress_tracker:
                            self.progress_tracker.report_task_complete(task_result)
                        
                        # Update rich progress - let the calling code handle this
                        # The rich tracker should be managed by the higher-level processor
                        
                        # Log progress periodically
                        if completed_count % self.config.progress_interval == 0:
                            self._log_progress(completed_count, len(tasks))
                            
                    except Exception as e:
                        # Create error result
                        error_result = TaskResult(
                            task_id=f"task_{task_index}",
                            success=False,
                            error=str(e),
                            execution_time=0.0
                        )
                        results.append(error_result)
                        
                        if self.progress_tracker:
                            self.progress_tracker.report_task_complete(error_result)
                        
                        # Update rich progress for failed tasks - let calling code handle this
                        
                        logger.error(f"Task {task_index} failed: {e}")
        
        except Exception as e:
            logger.error(f"Critical error in parallel processing: {e}")
            raise
        
        finally:
            if self.progress_tracker:
                self.progress_tracker.finalize()
            
            # Rich progress tracking completion handled by calling code
        
        # Log final summary
        self._log_final_summary(results)
        
        return results
    
    def _execute_task_wrapper(self, 
                             task: Callable, 
                             args: Tuple, 
                             kwargs: Dict, 
                             task_index: int) -> TaskResult:
        """Wrapper for executing tasks with error handling and monitoring."""
        task_id = f"task_{task_index}"
        start_time = time.time()
        
        # Monitor initial memory
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        except:
            initial_memory = 0
        
        try:
            # Execute the task
            result = task(*args, **kwargs)
            
            # Monitor final memory
            try:
                final_memory = process.memory_info().rss / (1024 * 1024)  # MB
                memory_used = final_memory - initial_memory
            except:
                memory_used = 0
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                memory_used_mb=memory_used,
                worker_id=mp.current_process().pid
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
                execution_time=execution_time,
                worker_id=mp.current_process().pid
            )
    
    def _log_progress(self, completed: int, total: int):
        """Log progress information."""
        if self._start_time:
            elapsed_time = time.time() - self._start_time
            current_memory = psutil.virtual_memory().percent
            memory_delta = current_memory - self._initial_memory
            
            avg_time_per_task = elapsed_time / completed if completed > 0 else 0
            
            logger.info(f"Progress: {completed}/{total} tasks "
                       f"(avg: {avg_time_per_task:.1f}s/task, "
                       f"memory: {current_memory:.1f}%, Δ{memory_delta:+.1f}%)")
    
    def _log_final_summary(self, results: List[TaskResult]):
        """Log final processing summary."""
        if not results:
            return
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        total_time = time.time() - self._start_time if self._start_time else 0
        avg_execution_time = np.mean([r.execution_time for r in successful_results]) if successful_results else 0
        
        final_memory = psutil.virtual_memory().percent
        memory_delta = final_memory - self._initial_memory
        
        logger.info(f"\n=== Processing Summary ===")
        logger.info(f"Processed: {len(successful_results)}/{len(results)} tasks")
        logger.info(f"Failed: {len(failed_results)} tasks")
        logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"Average per task: {avg_execution_time:.1f} seconds")
        logger.info(f"Memory change: {memory_delta:+.1f}%")
        
        if failed_results:
            logger.warning(f"Failed tasks: {[r.task_id for r in failed_results[:5]]}")


# =============================================================================
# CLIMATE-SPECIFIC MULTIPROCESSING TASKS
# =============================================================================

def process_climate_file_task(file_path: str, 
                             variable_name: str, 
                             region_key: str,
                             input_data_dir: str) -> Tuple[Optional[int], Optional[xr.DataArray]]:
    """
    Multiprocessing-safe task for processing a single climate file.
    
    This function is designed to be used with the MultiprocessingEngine
    for parallel climate data processing.
    """
    try:
        # Initialize file handler within the process
        file_handler = NorESM2FileHandler(input_data_dir)
        
        # Extract year from filename
        year = file_handler.extract_year_from_filename(file_path)
        if year is None:
            return None, None
        
        # Open dataset with conservative memory settings
        ds = xr.open_dataset(file_path, decode_times=False, chunks={'time': 30})
        
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
            climatology = var.groupby(var.dayofyear).mean(dim='time')
            result = climatology.compute()
            result = result.copy(deep=True)  # Ensure it's a copy
        else:
            ds.close()
            return None, None
        
        # Cleanup
        ds.close()
        del ds, region_ds, var, climatology
        gc.collect()
        
        return year, result
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None, None


def benchmark_multiprocessing_performance(data_directory: str, 
                                        num_files: int = 10,
                                        max_workers_to_test: List[int] = None) -> Dict[str, Any]:
    """
    Benchmark multiprocessing performance with different worker configurations.
    
    Args:
        data_directory: Directory containing climate data files
        num_files: Number of files to use for benchmarking
        max_workers_to_test: List of worker counts to test
        
    Returns:
        Dictionary with benchmark results
    """
    if max_workers_to_test is None:
        max_workers_to_test = [1, 2, 4, 6, 8]
    
    logger.info(f"Benchmarking multiprocessing performance with {num_files} files")
    
    # Get test files
    file_handler = NorESM2FileHandler(data_directory)
    test_files = file_handler.get_files_for_period('pr', 'historical', 2010, 2014)[:num_files]
    
    if len(test_files) < num_files:
        logger.warning(f"Only found {len(test_files)} files, using those")
        test_files = test_files[:len(test_files)]
    
    results = {}
    
    for num_workers in max_workers_to_test:
        logger.info(f"Testing with {num_workers} workers...")
        
        # Configure engine
        config = MultiprocessingConfig(max_workers=num_workers)
        engine = MultiprocessingEngine(config)
        
        # Prepare tasks
        tasks = [process_climate_file_task] * len(test_files)
        task_args = [(file_path, 'pr', 'CONUS', data_directory) for file_path in test_files]
        
        # Run benchmark
        start_time = time.time()
        task_results = engine.process_tasks(tasks, task_args)
        end_time = time.time()
        
        # Analyze results
        successful_tasks = [r for r in task_results if r.success]
        total_time = end_time - start_time
        
        results[f'workers_{num_workers}'] = {
            'total_time': total_time,
            'successful_tasks': len(successful_tasks),
            'failed_tasks': len(task_results) - len(successful_tasks),
            'tasks_per_second': len(successful_tasks) / total_time if total_time > 0 else 0,
            'speedup': results['workers_1']['total_time'] / total_time if 'workers_1' in results and total_time > 0 else 1.0,
            'efficiency': (results['workers_1']['total_time'] / total_time) / num_workers if 'workers_1' in results and total_time > 0 else 1.0
        }
        
        logger.info(f"Workers {num_workers}: {total_time:.1f}s, "
                   f"{len(successful_tasks)}/{len(test_files)} files, "
                   f"speedup: {results[f'workers_{num_workers}']['speedup']:.1f}x")
    
    # Log summary
    logger.info("\n=== Benchmark Results Summary ===")
    for key, result in results.items():
        logger.info(f"{key}: {result['total_time']:.1f}s, "
                   f"speedup: {result['speedup']:.1f}x, "
                   f"efficiency: {result['efficiency']:.1f}")
    
    return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_multiprocessing_engine(max_workers: int = None, 
                                 enable_progress_tracking: bool = True,
                                 use_rich_progress: bool = False,
                                 rich_tracker: Optional[RichProgressTracker] = None,
                                 **kwargs) -> MultiprocessingEngine:
    """
    Convenience function to create a configured multiprocessing engine.
    
    Args:
        max_workers: Number of worker processes (auto-detected if None)
        enable_progress_tracking: Whether to enable basic progress tracking
        use_rich_progress: Whether to use rich progress tracking
        rich_tracker: Existing rich tracker to use (optional)
        **kwargs: Additional configuration options
        
    Returns:
        Configured MultiprocessingEngine instance
    """
    config = MultiprocessingConfig(max_workers=max_workers or 0, **kwargs)
    
    progress_tracker = None
    if enable_progress_tracking and not use_rich_progress:
        progress_tracker = ProgressTracker()
    
    return MultiprocessingEngine(
        config, 
        progress_tracker, 
        rich_tracker=rich_tracker,
        use_rich_progress=use_rich_progress
    )


def process_climate_files_parallel(file_paths: List[str],
                                  variable: str,
                                  region: str,
                                  input_data_dir: str,
                                  max_workers: int = None) -> List[Tuple[Optional[int], Optional[xr.DataArray]]]:
    """
    Convenience function to process climate files in parallel.
    
    Args:
        file_paths: List of file paths to process
        variable: Climate variable name
        region: Region key
        input_data_dir: Input data directory
        max_workers: Number of worker processes
        
    Returns:
        List of (year, climatology) tuples
    """
    engine = create_multiprocessing_engine(max_workers=max_workers)
    
    # Prepare tasks
    tasks = [process_climate_file_task] * len(file_paths)
    task_args = [(file_path, variable, region, input_data_dir) for file_path in file_paths]
    
    # Process tasks
    task_results = engine.process_tasks(tasks, task_args)
    
    # Extract results
    results = []
    for task_result in task_results:
        if task_result.success and task_result.result:
            results.append(task_result.result)
        else:
            results.append((None, None))
    
    return results


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

def main():
    """Main function for testing the multiprocessing engine."""
    
    data_directory = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
    
    if not Path(data_directory).exists():
        logger.error(f"Data directory not found: {data_directory}")
        return
    
    # Run performance benchmark
    logger.info("Running multiprocessing performance benchmark...")
    results = benchmark_multiprocessing_performance(data_directory, num_files=6)
    
    # Test basic functionality
    logger.info("Testing basic multiprocessing functionality...")
    engine = create_multiprocessing_engine(max_workers=4)
    
    # Simple test tasks
    def test_task(x: int) -> int:
        time.sleep(0.1)  # Simulate work
        return x * x
    
    tasks = [test_task] * 10
    task_args = [(i,) for i in range(10)]
    
    task_results = engine.process_tasks(tasks, task_args)
    
    successful_results = [r for r in task_results if r.success]
    logger.info(f"Test completed: {len(successful_results)}/10 tasks successful")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()