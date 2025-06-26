"""
Zarr utilities for climate data processing.

This module provides functions for converting NetCDF files to Zarr format,
managing Zarr stores, and optimizing data access patterns for climate analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import warnings

import numpy as np
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc

logger = logging.getLogger(__name__)


class ZarrConfig:
    """Configuration for Zarr store creation and management."""
    
    DEFAULT_CHUNKS = {
        'time': 365,
        'lat': 100,
        'lon': 100
    }
    
    DEFAULT_COMPRESSOR = Blosc(
        cname='zstd',
        clevel=3,
        shuffle=Blosc.BITSHUFFLE
    )
    
    CLIMATE_VARIABLE_ENCODINGS = {
        'pr': {
            'dtype': 'float32',
            'scale_factor': 0.01,
            'add_offset': 0.0,
            '_FillValue': -9999.0
        },
        'tas': {
            'dtype': 'float32',
            'scale_factor': 0.01,
            'add_offset': 273.15,
            '_FillValue': -9999.0
        },
        'tasmax': {
            'dtype': 'float32',
            'scale_factor': 0.01,
            'add_offset': 273.15,
            '_FillValue': -9999.0
        },
        'tasmin': {
            'dtype': 'float32',
            'scale_factor': 0.01,
            'add_offset': 273.15,
            '_FillValue': -9999.0
        }
    }


def optimize_chunks(
    ds: xr.Dataset,
    target_chunk_size_mb: float = 128,
    dim_priorities: Optional[Dict[str, int]] = None
) -> Dict[str, int]:
    """
    Calculate optimal chunk sizes for a dataset based on target chunk size.
    
    Args:
        ds: xarray Dataset to optimize
        target_chunk_size_mb: Target chunk size in megabytes
        dim_priorities: Priority for each dimension (higher = prefer larger chunks)
        
    Returns:
        Dictionary of optimal chunk sizes for each dimension
    """
    if dim_priorities is None:
        dim_priorities = {'time': 3, 'lat': 2, 'lon': 1}
    
    # Get dataset info
    var_name = list(ds.data_vars)[0]
    var = ds[var_name]
    dtype_size = var.dtype.itemsize
    
    # Calculate total elements for target size
    target_bytes = target_chunk_size_mb * 1024 * 1024
    target_elements = target_bytes // dtype_size
    
    # Start with full dimensions
    chunks = {dim: size for dim, size in var.sizes.items()}
    
    # Iteratively reduce chunks based on priority
    while np.prod(list(chunks.values())) > target_elements:
        # Find dimension to reduce (lowest priority, largest chunk)
        reduce_dim = min(
            chunks.keys(),
            key=lambda d: (dim_priorities.get(d, 0), -chunks[d])
        )
        chunks[reduce_dim] = max(1, chunks[reduce_dim] // 2)
    
    return chunks


def netcdf_to_zarr(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    chunks: Optional[Dict[str, int]] = None,
    variable: Optional[str] = None,
    compressor: Optional[Any] = None,
    consolidated: bool = True,
    overwrite: bool = False,
    progress: bool = True
) -> zarr.hierarchy.Group:
    """
    Convert a NetCDF file to Zarr format with optimizations.
    
    Args:
        input_path: Path to input NetCDF file
        output_path: Path to output Zarr store
        chunks: Chunk sizes for each dimension
        variable: Specific variable to extract (None for all)
        compressor: Numcodecs compressor (default: Blosc)
        consolidated: Whether to consolidate metadata
        overwrite: Whether to overwrite existing store
        progress: Show progress bar
        
    Returns:
        Zarr group object
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Zarr store already exists: {output_path}")
    
    logger.info(f"Converting {input_path} to Zarr format")
    
    # Open dataset
    ds = xr.open_dataset(input_path, decode_times=False)
    
    # Select variable if specified
    if variable:
        if variable not in ds.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset")
        ds = ds[[variable]]
    
    # Determine chunks
    if chunks is None:
        chunks = optimize_chunks(ds)
        logger.info(f"Using optimized chunks: {chunks}")
    
    # Rechunk if needed
    if chunks:
        ds = ds.chunk(chunks)
    
    # Set up encoding
    encoding = {}
    if compressor is None:
        compressor = ZarrConfig.DEFAULT_COMPRESSOR
    
    for var in ds.data_vars:
        encoding[var] = {
            'compressor': compressor,
            'chunks': tuple(chunks.get(dim, size) for dim, size in ds[var].sizes.items())
        }
        
        # Add climate-specific encoding if available
        if var in ZarrConfig.CLIMATE_VARIABLE_ENCODINGS:
            encoding[var].update(ZarrConfig.CLIMATE_VARIABLE_ENCODINGS[var])
    
    # Write to Zarr
    mode = 'w' if overwrite else 'w-'
    
    if progress:
        with ProgressBar():
            ds.to_zarr(
                output_path,
                mode=mode,
                encoding=encoding,
                consolidated=consolidated,
                compute=True
            )
    else:
        ds.to_zarr(
            output_path,
            mode=mode,
            encoding=encoding,
            consolidated=consolidated,
            compute=True
        )
    
    logger.info(f"Successfully created Zarr store at {output_path}")
    
    # Return the Zarr group
    return zarr.open_group(output_path, mode='r')


def create_multiscale_zarr(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    scales: List[int] = [1, 2, 4, 8],
    spatial_dims: List[str] = ['lat', 'lon'],
    **kwargs
) -> None:
    """
    Create a multiscale Zarr store for efficient visualization at different resolutions.
    
    Args:
        input_path: Path to input NetCDF file
        output_path: Path to output Zarr store
        scales: Downsampling factors for each scale
        spatial_dims: Spatial dimensions to downsample
        **kwargs: Additional arguments for netcdf_to_zarr
    """
    output_path = Path(output_path)
    
    # Create base resolution
    base_path = output_path / "0"
    netcdf_to_zarr(input_path, base_path, **kwargs)
    
    # Open base dataset
    ds_base = xr.open_zarr(base_path)
    
    # Create downsampled versions
    for i, scale in enumerate(scales[1:], 1):
        scale_path = output_path / str(i)
        logger.info(f"Creating scale {i} (factor: {scale})")
        
        # Downsample
        ds_scaled = ds_base.coarsen(
            dim={dim: scale for dim in spatial_dims if dim in ds_base.dims},
            boundary='trim'
        ).mean()
        
        # Save
        ds_scaled.to_zarr(scale_path, mode='w', consolidated=True)
    
    # Create metadata
    import json
    metadata = {
        'multiscales': [{
            'version': '0.1',
            'name': 'climate_data',
            'datasets': [
                {'path': str(i)} for i in range(len(scales))
            ],
            'axes': spatial_dims,
            'scales': scales
        }]
    }
    
    with open(output_path / '.zattrs', 'w') as f:
        json.dump(metadata, f, indent=2)


def append_to_zarr(
    ds: xr.Dataset,
    zarr_path: Union[str, Path],
    append_dim: str = 'time',
    region: Optional[Dict[str, slice]] = None
) -> None:
    """
    Append data to an existing Zarr store along a dimension.
    
    Args:
        ds: Dataset to append
        zarr_path: Path to existing Zarr store
        append_dim: Dimension to append along
        region: Region specification for writing
    """
    zarr_path = Path(zarr_path)
    
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")
    
    # Determine region if not specified
    if region is None:
        existing = xr.open_zarr(zarr_path)
        if append_dim in existing.dims:
            start = existing.dims[append_dim]
            stop = start + ds.dims[append_dim]
            region = {append_dim: slice(start, stop)}
        else:
            region = {}
    
    # Append
    ds.to_zarr(
        zarr_path,
        mode='a',
        region=region,
        consolidated=True
    )
    
    logger.info(f"Appended data to {zarr_path} at region {region}")


def consolidate_zarr_metadata(zarr_path: Union[str, Path]) -> None:
    """
    Consolidate metadata for faster Zarr store opening.
    
    Args:
        zarr_path: Path to Zarr store
    """
    zarr_path = Path(zarr_path)
    
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")
    
    zarr.consolidate_metadata(str(zarr_path))
    logger.info(f"Consolidated metadata for {zarr_path}")


def validate_zarr_store(zarr_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate a Zarr store and return information about its contents.
    
    Args:
        zarr_path: Path to Zarr store
        
    Returns:
        Dictionary with store information
    """
    zarr_path = Path(zarr_path)
    
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")
    
    # Open store
    store = zarr.open_group(str(zarr_path), mode='r')
    
    info = {
        'path': str(zarr_path),
        'arrays': {},
        'groups': list(store.groups()),
        'attrs': dict(store.attrs)
    }
    
    # Get array info
    for name, array in store.arrays():
        info['arrays'][name] = {
            'shape': array.shape,
            'chunks': array.chunks,
            'dtype': str(array.dtype),
            'compressor': str(array.compressor),
            'size_mb': array.nbytes / 1024 / 1024,
            'compressed_size_mb': array.nbytes_stored / 1024 / 1024
        }
    
    return info