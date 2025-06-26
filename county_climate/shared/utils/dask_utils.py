"""
Dask utilities for distributed processing of climate data.

This module provides utilities for setting up and managing Dask clusters,
optimizing chunk sizes, and monitoring distributed computations.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any
import psutil
import warnings

import dask
import dask.array as da
from dask.distributed import Client, LocalCluster, performance_report
from dask.diagnostics import ProgressBar
import numpy as np

logger = logging.getLogger(__name__)


class DaskConfig:
    """Configuration for Dask distributed computing."""
    
    # Memory settings
    MEMORY_LIMIT_PER_WORKER = "8GB"
    MEMORY_TARGET_FRACTION = 0.8
    MEMORY_SPILL_FRACTION = 0.9
    MEMORY_PAUSE_FRACTION = 0.95
    
    # Worker settings
    DEFAULT_THREADS_PER_WORKER = 2
    DEFAULT_N_WORKERS = None  # Auto-detect
    
    # Chunk size targets
    TARGET_CHUNK_SIZE_MB = 128
    MAX_CHUNK_SIZE_MB = 256
    MIN_CHUNK_SIZE_MB = 64


def create_climate_cluster(
    n_workers: Optional[int] = None,
    threads_per_worker: int = DaskConfig.DEFAULT_THREADS_PER_WORKER,
    memory_limit: str = DaskConfig.MEMORY_LIMIT_PER_WORKER,
    dashboard_address: str = ':8787',
    local_directory: Optional[str] = None
) -> LocalCluster:
    """
    Create a Dask LocalCluster optimized for climate data processing.
    
    Args:
        n_workers: Number of workers (None for auto-detect based on memory)
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        dashboard_address: Address for Dask dashboard
        local_directory: Directory for temporary files
        
    Returns:
        Configured LocalCluster instance
    """
    # Auto-detect optimal worker count if not specified
    if n_workers is None:
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        memory_per_worker_gb = float(memory_limit.rstrip('GB'))
        n_workers = min(
            psutil.cpu_count() // threads_per_worker,
            int(total_memory_gb * 0.8 / memory_per_worker_gb)
        )
        logger.info(f"Auto-detected optimal worker count: {n_workers}")
    
    # Create cluster
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        dashboard_address=dashboard_address,
        local_directory=local_directory,
        silence_logs=logging.WARNING
    )
    
    # Configure Dask settings
    dask.config.set({
        'distributed.worker.memory.target': DaskConfig.MEMORY_TARGET_FRACTION,
        'distributed.worker.memory.spill': DaskConfig.MEMORY_SPILL_FRACTION,
        'distributed.worker.memory.pause': DaskConfig.MEMORY_PAUSE_FRACTION,
        'distributed.scheduler.work-stealing': True,
        'distributed.scheduler.bandwidth': '100 MB/s',
        'array.chunk-size': f"{DaskConfig.TARGET_CHUNK_SIZE_MB}MiB"
    })
    
    logger.info(f"Created cluster with {n_workers} workers, dashboard at {cluster.dashboard_link}")
    
    return cluster


def optimize_chunk_size_3d(
    shape: tuple,
    dtype: np.dtype,
    target_size_mb: float = DaskConfig.TARGET_CHUNK_SIZE_MB,
    time_chunk_days: int = 365,
    prefer_whole_arrays: bool = True
) -> Dict[str, int]:
    """
    Optimize chunk sizes for 3D climate data (time, lat, lon).
    
    Args:
        shape: Data shape (time, lat, lon)
        dtype: Data type
        target_size_mb: Target chunk size in MB
        time_chunk_days: Preferred time chunk size in days
        prefer_whole_arrays: Whether to prefer whole lat/lon arrays
        
    Returns:
        Dictionary with optimal chunk sizes
    """
    time_size, lat_size, lon_size = shape
    dtype_size = np.dtype(dtype).itemsize
    
    # Target number of elements
    target_bytes = target_size_mb * 1024 * 1024
    target_elements = target_bytes / dtype_size
    
    # Start with preferred time chunking
    time_chunk = min(time_chunk_days, time_size)
    
    # Calculate spatial elements for target size
    spatial_elements = target_elements / time_chunk
    spatial_size = lat_size * lon_size
    
    if prefer_whole_arrays and spatial_size <= spatial_elements * 1.5:
        # Use whole spatial arrays if they fit reasonably
        lat_chunk = lat_size
        lon_chunk = lon_size
    else:
        # Calculate balanced spatial chunks
        aspect_ratio = lon_size / lat_size
        lat_chunk = int(np.sqrt(spatial_elements / aspect_ratio))
        lon_chunk = int(lat_chunk * aspect_ratio)
        
        # Ensure chunks aren't too small
        lat_chunk = max(50, min(lat_chunk, lat_size))
        lon_chunk = max(50, min(lon_chunk, lon_size))
    
    chunks = {
        'time': time_chunk,
        'lat': lat_chunk,
        'lon': lon_chunk
    }
    
    # Validate chunk size
    chunk_elements = time_chunk * lat_chunk * lon_chunk
    chunk_mb = chunk_elements * dtype_size / (1024 * 1024)
    
    logger.info(
        f"Optimized chunks: {chunks} "
        f"({chunk_mb:.1f} MB per chunk)"
    )
    
    return chunks


def create_distributed_client(
    cluster: Optional[LocalCluster] = None,
    **kwargs
) -> Client:
    """
    Create a Dask distributed client with monitoring.
    
    Args:
        cluster: Existing cluster (creates new if None)
        **kwargs: Additional arguments for Client/Cluster
        
    Returns:
        Configured Client instance
    """
    if cluster is None:
        cluster = create_climate_cluster(**kwargs)
    
    client = Client(cluster)
    
    # Set up plugins for monitoring
    from dask.distributed import WorkerPlugin
    
    class MemoryMonitor(WorkerPlugin):
        def __init__(self):
            self.name = "memory_monitor"
        
        def setup(self, worker):
            import gc
            # Force garbage collection on setup
            gc.collect()
    
    client.register_worker_plugin(MemoryMonitor())
    
    logger.info(f"Connected to cluster: {client.dashboard_link}")
    
    return client


def rechunk_climate_data(
    data: Union[da.Array, 'xr.DataArray'],
    target_chunks: Dict[str, int],
    max_mem: str = '2GB',
    temp_store: Optional[str] = None
) -> Union[da.Array, 'xr.DataArray']:
    """
    Rechunk climate data efficiently using rechunker patterns.
    
    Args:
        data: Input dask array or xarray DataArray
        target_chunks: Target chunk sizes
        max_mem: Maximum memory to use
        temp_store: Temporary storage location
        
    Returns:
        Rechunked array
    """
    import xarray as xr
    
    is_xarray = hasattr(data, 'chunks')
    
    if is_xarray:
        # Handle xarray DataArray
        arr = data.data
        dims = data.dims
    else:
        # Handle dask array
        arr = data
        dims = ['dim_%d' % i for i in range(arr.ndim)]
    
    # Convert target chunks to tuple
    if isinstance(target_chunks, dict) and is_xarray:
        chunks_tuple = tuple(target_chunks.get(dim, arr.chunks[i][0]) 
                           for i, dim in enumerate(dims))
    else:
        chunks_tuple = target_chunks
    
    # Rechunk
    logger.info(f"Rechunking from {arr.chunks} to {chunks_tuple}")
    
    if temp_store:
        # Use two-pass rechunking for very large arrays
        intermediate_chunks = tuple(
            min(c * 2, s) for c, s in zip(chunks_tuple, arr.shape)
        )
        arr = arr.rechunk(intermediate_chunks)
        arr = arr.persist()
    
    arr_rechunked = arr.rechunk(chunks_tuple)
    
    if is_xarray:
        # Recreate xarray DataArray
        return data.copy(data=arr_rechunked)
    else:
        return arr_rechunked


def monitor_computation(
    computation: Any,
    client: Client,
    task_name: str = "Computation",
    log_interval: int = 10
) -> Any:
    """
    Monitor a Dask computation with progress and performance tracking.
    
    Args:
        computation: Dask computation to run
        client: Dask client
        task_name: Name for logging
        log_interval: Seconds between progress logs
        
    Returns:
        Computation result
    """
    import time
    from dask.distributed import as_completed
    
    logger.info(f"Starting {task_name}")
    
    # Submit computation
    future = client.compute(computation)
    
    # Monitor progress
    start_time = time.time()
    last_log = start_time
    
    while not future.done():
        current_time = time.time()
        if current_time - last_log > log_interval:
            # Get progress info
            progress = client.scheduler_info()['tasks']
            n_complete = sum(1 for t in progress.values() if t == 'memory')
            n_total = len(progress)
            
            elapsed = current_time - start_time
            percent = (n_complete / n_total * 100) if n_total > 0 else 0
            
            logger.info(
                f"{task_name}: {percent:.1f}% "
                f"({n_complete}/{n_total} tasks) "
                f"Elapsed: {elapsed:.1f}s"
            )
            
            last_log = current_time
        
        time.sleep(1)
    
    # Get result
    result = future.result()
    total_time = time.time() - start_time
    
    logger.info(f"{task_name} completed in {total_time:.1f}s")
    
    return result


def adaptive_rechunking(
    data: 'xr.Dataset',
    operation: str = 'time_mean',
    memory_limit_gb: float = 8.0
) -> Dict[str, int]:
    """
    Determine optimal chunking based on the operation to be performed.
    
    Args:
        data: Input dataset
        operation: Type of operation ('time_mean', 'spatial_mean', 'rolling')
        memory_limit_gb: Memory limit in GB
        
    Returns:
        Optimal chunk configuration
    """
    # Get data variable (assume first one)
    var_name = list(data.data_vars)[0]
    var = data[var_name]
    
    shape = var.shape
    dtype_size = var.dtype.itemsize
    total_size_gb = np.prod(shape) * dtype_size / (1024**3)
    
    logger.info(f"Data shape: {shape}, Total size: {total_size_gb:.1f} GB")
    
    if operation == 'time_mean':
        # Optimize for time operations - larger time chunks
        time_chunk = min(shape[0], 365 * 5)  # 5 years
        spatial_elements = (memory_limit_gb * 1024**3) / (time_chunk * dtype_size)
        lat_chunk = min(shape[1], int(np.sqrt(spatial_elements)))
        lon_chunk = min(shape[2], int(spatial_elements / lat_chunk))
        
    elif operation == 'spatial_mean':
        # Optimize for spatial operations - full spatial arrays
        lat_chunk = shape[1]
        lon_chunk = shape[2]
        time_elements = (memory_limit_gb * 1024**3) / (lat_chunk * lon_chunk * dtype_size)
        time_chunk = min(shape[0], int(time_elements))
        
    elif operation == 'rolling':
        # Balanced chunking for rolling operations
        chunks = optimize_chunk_size_3d(shape, var.dtype)
        return chunks
        
    else:
        # Default balanced chunking
        chunks = optimize_chunk_size_3d(shape, var.dtype)
        return chunks
    
    return {
        'time': time_chunk,
        'lat': lat_chunk,
        'lon': lon_chunk
    }


def batch_process_zarr_stores(
    store_paths: List[str],
    process_func: callable,
    client: Client,
    max_concurrent: int = 4,
    **kwargs
) -> List[Any]:
    """
    Process multiple Zarr stores in parallel with controlled concurrency.
    
    Args:
        store_paths: List of Zarr store paths
        process_func: Function to apply to each store
        client: Dask client
        max_concurrent: Maximum concurrent processes
        **kwargs: Additional arguments for process_func
        
    Returns:
        List of results
    """
    from dask.distributed import as_completed
    
    logger.info(f"Processing {len(store_paths)} stores with max concurrency {max_concurrent}")
    
    # Submit tasks with controlled concurrency
    futures = []
    results = [None] * len(store_paths)
    
    for i in range(0, len(store_paths), max_concurrent):
        batch = store_paths[i:i + max_concurrent]
        batch_futures = []
        
        for j, path in enumerate(batch):
            future = client.submit(process_func, path, **kwargs)
            batch_futures.append((i + j, future))
        
        # Wait for batch to complete
        for idx, future in batch_futures:
            results[idx] = future.result()
            logger.info(f"Completed {idx + 1}/{len(store_paths)}")
    
    return results