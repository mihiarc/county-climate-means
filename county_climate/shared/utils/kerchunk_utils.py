"""
Kerchunk utilities for creating virtual Zarr references from NetCDF files.

This module enables accessing NetCDF files as if they were Zarr stores without
duplicating data, significantly reducing storage requirements and improving
data access performance.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import concurrent.futures
from tqdm import tqdm

import fsspec
import ujson
import xarray as xr
from kerchunk.hdf import SingleHdf5ToZarr
from kerchunk.combine import MultiZarrToZarr

logger = logging.getLogger(__name__)


class KerchunkConfig:
    """Configuration for Kerchunk reference generation."""
    
    DEFAULT_INLINE_THRESHOLD = 100
    DEFAULT_CONCAT_DIMS = ['time']
    DEFAULT_IDENTICAL_DIMS = ['lat', 'lon']
    

def create_single_file_reference(
    netcdf_path: Union[str, Path],
    reference_path: Optional[Union[str, Path]] = None,
    inline_threshold: int = KerchunkConfig.DEFAULT_INLINE_THRESHOLD,
    storage_options: Optional[Dict] = None
) -> Dict:
    """
    Create Kerchunk reference for a single NetCDF file.
    
    Args:
        netcdf_path: Path to NetCDF file
        reference_path: Path to save reference JSON (optional)
        inline_threshold: Threshold for inlining small data chunks
        storage_options: Storage options for fsspec
        
    Returns:
        Reference dictionary
    """
    netcdf_path = Path(netcdf_path)
    
    if not netcdf_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {netcdf_path}")
    
    logger.info(f"Creating Kerchunk reference for {netcdf_path}")
    
    # Create reference
    h5chunks = SingleHdf5ToZarr(
        str(netcdf_path),
        inline_threshold=inline_threshold,
        storage_options=storage_options or {}
    )
    
    refs = h5chunks.translate()
    
    # Save if path provided
    if reference_path:
        reference_path = Path(reference_path)
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(reference_path, 'wb') as f:
            f.write(ujson.dumps(refs).encode())
        
        logger.info(f"Saved reference to {reference_path}")
    
    return refs


def create_multi_file_reference(
    netcdf_paths: List[Union[str, Path]],
    reference_path: Union[str, Path],
    concat_dims: Optional[List[str]] = None,
    identical_dims: Optional[List[str]] = None,
    preprocess: Optional[callable] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> Dict:
    """
    Create combined Kerchunk reference for multiple NetCDF files.
    
    Args:
        netcdf_paths: List of NetCDF file paths
        reference_path: Path to save combined reference
        concat_dims: Dimensions to concatenate along
        identical_dims: Dimensions that are identical across files
        preprocess: Function to preprocess individual references
        parallel: Whether to process files in parallel
        max_workers: Maximum number of parallel workers
        
    Returns:
        Combined reference dictionary
    """
    reference_path = Path(reference_path)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    
    if concat_dims is None:
        concat_dims = KerchunkConfig.DEFAULT_CONCAT_DIMS
    if identical_dims is None:
        identical_dims = KerchunkConfig.DEFAULT_IDENTICAL_DIMS
    
    # Convert paths
    netcdf_paths = [Path(p) for p in netcdf_paths]
    
    # Verify files exist
    missing = [p for p in netcdf_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"NetCDF files not found: {missing}")
    
    logger.info(f"Creating combined reference for {len(netcdf_paths)} files")
    
    # Generate individual references
    temp_refs_dir = reference_path.parent / f".temp_refs_{reference_path.stem}"
    temp_refs_dir.mkdir(exist_ok=True)
    
    def process_file(idx_path):
        idx, path = idx_path
        ref_path = temp_refs_dir / f"ref_{idx:04d}.json"
        if not ref_path.exists():
            refs = create_single_file_reference(path)
            if preprocess:
                refs = preprocess(refs)
            with open(ref_path, 'wb') as f:
                f.write(ujson.dumps(refs).encode())
        return str(ref_path)
    
    # Process files
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            ref_paths = list(tqdm(
                executor.map(process_file, enumerate(netcdf_paths)),
                total=len(netcdf_paths),
                desc="Creating references"
            ))
    else:
        ref_paths = []
        for idx, path in tqdm(enumerate(netcdf_paths), desc="Creating references"):
            ref_paths.append(process_file((idx, path)))
    
    # Combine references
    logger.info("Combining references...")
    
    mzz = MultiZarrToZarr(
        ref_paths,
        concat_dims=concat_dims,
        identical_dims=identical_dims,
        storage_options={'anon': True}
    )
    
    combined_refs = mzz.translate()
    
    # Save combined reference
    with open(reference_path, 'wb') as f:
        f.write(ujson.dumps(combined_refs).encode())
    
    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_refs_dir)
    
    logger.info(f"Saved combined reference to {reference_path}")
    
    return combined_refs


def open_kerchunk_dataset(
    reference_path: Union[str, Path],
    storage_options: Optional[Dict] = None,
    chunks: Optional[Dict[str, int]] = None
) -> xr.Dataset:
    """
    Open a dataset using Kerchunk references.
    
    Args:
        reference_path: Path to Kerchunk reference JSON
        storage_options: Storage options for fsspec
        chunks: Chunk sizes for dask arrays
        
    Returns:
        xarray Dataset backed by virtual Zarr
    """
    reference_path = Path(reference_path)
    
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    
    # Load references
    with open(reference_path, 'rb') as f:
        refs = ujson.loads(f.read())
    
    # Create filesystem
    fs = fsspec.filesystem(
        'reference',
        fo=refs,
        remote_protocol='file',
        remote_options=storage_options or {}
    )
    
    # Open as Zarr
    mapper = fs.get_mapper('')
    
    # Open with xarray
    ds = xr.open_zarr(mapper, chunks=chunks)
    
    return ds


def create_climate_data_catalog(
    base_path: Union[str, Path],
    output_path: Union[str, Path],
    variables: List[str] = ['pr', 'tas', 'tasmax', 'tasmin'],
    scenarios: List[str] = ['historical', 'ssp245', 'ssp585'],
    regions: List[str] = ['CONUS', 'Alaska', 'Hawaii', 'PRVI', 'Guam']
) -> Dict:
    """
    Create a catalog of Kerchunk references for climate data.
    
    Args:
        base_path: Base path to climate data
        output_path: Path to save catalog
        variables: Climate variables to include
        scenarios: Climate scenarios to include
        regions: Regions to include
        
    Returns:
        Catalog dictionary
    """
    base_path = Path(base_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    catalog = {
        'version': '1.0',
        'description': 'Climate data Kerchunk reference catalog',
        'datasets': {}
    }
    
    refs_dir = output_path.parent / 'references'
    refs_dir.mkdir(exist_ok=True)
    
    for variable in variables:
        for scenario in scenarios:
            for region in regions:
                # Find NetCDF files
                pattern = f"{variable}*{scenario}*{region}*.nc"
                files = sorted(base_path.glob(f"**/{pattern}"))
                
                if not files:
                    logger.warning(f"No files found for {variable}/{scenario}/{region}")
                    continue
                
                # Create reference
                ref_name = f"{variable}_{scenario}_{region}.json"
                ref_path = refs_dir / ref_name
                
                if len(files) == 1:
                    create_single_file_reference(files[0], ref_path)
                else:
                    create_multi_file_reference(files, ref_path)
                
                # Add to catalog
                key = f"{variable}/{scenario}/{region}"
                catalog['datasets'][key] = {
                    'reference': str(ref_path.relative_to(output_path.parent)),
                    'files': len(files),
                    'variable': variable,
                    'scenario': scenario,
                    'region': region
                }
    
    # Save catalog
    with open(output_path, 'w') as f:
        json.dump(catalog, f, indent=2)
    
    logger.info(f"Created catalog with {len(catalog['datasets'])} datasets")
    
    return catalog


def benchmark_kerchunk_performance(
    netcdf_path: Union[str, Path],
    reference_path: Union[str, Path],
    test_operations: Optional[List[callable]] = None
) -> Dict[str, float]:
    """
    Benchmark performance of Kerchunk references vs direct NetCDF access.
    
    Args:
        netcdf_path: Path to NetCDF file
        reference_path: Path to Kerchunk reference
        test_operations: List of operations to benchmark
        
    Returns:
        Dictionary of timing results
    """
    import time
    
    netcdf_path = Path(netcdf_path)
    reference_path = Path(reference_path)
    
    if test_operations is None:
        test_operations = [
            lambda ds: ds.mean(),
            lambda ds: ds.sel(time=slice('2000', '2010')).mean(),
            lambda ds: ds.groupby('time.year').mean()
        ]
    
    results = {}
    
    # Benchmark NetCDF
    logger.info("Benchmarking NetCDF access...")
    for i, op in enumerate(test_operations):
        start = time.time()
        with xr.open_dataset(netcdf_path, chunks={'time': 365}) as ds:
            result = op(ds).compute()
        results[f'netcdf_op_{i}'] = time.time() - start
    
    # Benchmark Kerchunk
    logger.info("Benchmarking Kerchunk access...")
    for i, op in enumerate(test_operations):
        start = time.time()
        ds = open_kerchunk_dataset(reference_path, chunks={'time': 365})
        result = op(ds).compute()
        results[f'kerchunk_op_{i}'] = time.time() - start
    
    # Calculate speedups
    for i in range(len(test_operations)):
        nc_time = results[f'netcdf_op_{i}']
        kc_time = results[f'kerchunk_op_{i}']
        results[f'speedup_op_{i}'] = nc_time / kc_time
    
    return results