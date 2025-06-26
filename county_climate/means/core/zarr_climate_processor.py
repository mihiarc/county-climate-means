"""
Zarr-based climate processor for efficient 30-year normal calculations.

This processor leverages Zarr and Dask for distributed processing of climate data,
offering significant performance improvements over file-based processing.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from dask.distributed import Client, as_completed, progress
from tqdm.auto import tqdm

from county_climate.means.core.base import BaseClimateProcessor
from county_climate.shared.utils.zarr_utils import (
    ZarrConfig, consolidate_zarr_metadata, validate_zarr_store
)
from county_climate.shared.utils.kerchunk_utils import open_kerchunk_dataset

logger = logging.getLogger(__name__)


class ZarrClimateProcessor(BaseClimateProcessor):
    """
    Zarr-based processor for calculating 30-year climate normals.
    
    This processor uses Zarr stores (either native or virtual via Kerchunk)
    for efficient data access and Dask for distributed computation.
    """
    
    def __init__(
        self,
        variable: str,
        scenario: str,
        region: str,
        model: str = 'NorESM2-LM',
        use_kerchunk: bool = False,
        dask_client: Optional[Client] = None,
        chunk_strategy: str = 'auto'
    ):
        """
        Initialize the Zarr climate processor.
        
        Args:
            variable: Climate variable (pr, tas, tasmax, tasmin)
            scenario: Climate scenario (historical, ssp245, ssp585)
            region: Geographic region
            model: Climate model name
            use_kerchunk: Whether to use Kerchunk virtual references
            dask_client: Dask distributed client (creates local if None)
            chunk_strategy: Chunking strategy ('auto', 'time', 'space')
        """
        super().__init__()
        
        self.variable = variable
        self.scenario = scenario
        self.region = region
        self.model = model
        self.use_kerchunk = use_kerchunk
        self.chunk_strategy = chunk_strategy
        
        # Initialize or use provided Dask client
        if dask_client is None:
            self.client = Client(
                n_workers=4,
                threads_per_worker=2,
                memory_limit='8GB',
                silence_logs=logging.WARNING
            )
            self.owns_client = True
        else:
            self.client = dask_client
            self.owns_client = False
        
        logger.info(f"Initialized ZarrClimateProcessor for {variable}/{scenario}/{region}")
        logger.info(f"Dask dashboard: {self.client.dashboard_link}")
    
    def __del__(self):
        """Clean up Dask client if we created it."""
        if hasattr(self, 'owns_client') and self.owns_client and hasattr(self, 'client'):
            self.client.close()
    
    def _get_optimal_chunks(self, ds: xr.Dataset) -> Dict[str, int]:
        """Determine optimal chunk sizes based on dataset and strategy."""
        if self.chunk_strategy == 'auto':
            # Auto strategy based on data characteristics
            time_size = ds.dims['time']
            lat_size = ds.dims.get('lat', ds.dims.get('latitude', 100))
            lon_size = ds.dims.get('lon', ds.dims.get('longitude', 100))
            
            # Aim for ~128MB chunks
            if time_size > 10000:  # Long time series
                chunks = {'time': 365, 'lat': 50, 'lon': 50}
            else:
                chunks = {'time': -1, 'lat': 100, 'lon': 100}
        
        elif self.chunk_strategy == 'time':
            # Optimize for time-series operations
            chunks = {'time': 365 * 5, 'lat': -1, 'lon': -1}
        
        elif self.chunk_strategy == 'space':
            # Optimize for spatial operations
            chunks = {'time': -1, 'lat': 100, 'lon': 100}
        
        else:
            chunks = ZarrConfig.DEFAULT_CHUNKS
        
        # Adjust chunk names to match dataset dimensions
        actual_chunks = {}
        for dim, size in chunks.items():
            if dim in ds.dims:
                actual_chunks[dim] = size
            elif dim == 'lat' and 'latitude' in ds.dims:
                actual_chunks['latitude'] = size
            elif dim == 'lon' and 'longitude' in ds.dims:
                actual_chunks['longitude'] = size
        
        return actual_chunks
    
    def open_zarr_dataset(
        self,
        path: Union[str, Path],
        chunks: Optional[Dict[str, int]] = None
    ) -> xr.Dataset:
        """Open a Zarr dataset with appropriate method."""
        path = Path(path)
        
        if self.use_kerchunk and path.suffix == '.json':
            # Open using Kerchunk reference
            ds = open_kerchunk_dataset(path, chunks=chunks)
        else:
            # Open native Zarr store
            if not path.exists():
                raise FileNotFoundError(f"Zarr store not found: {path}")
            
            ds = xr.open_zarr(path, chunks=chunks, consolidated=True)
        
        # Apply optimal chunking if not specified
        if chunks is None:
            chunks = self._get_optimal_chunks(ds)
            ds = ds.chunk(chunks)
        
        return ds
    
    def calculate_rolling_normals_zarr(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        start_year: int,
        end_year: int,
        window_size: int = 30,
        min_years: int = 25
    ) -> xr.Dataset:
        """
        Calculate rolling 30-year normals using Zarr-backed computation.
        
        Args:
            input_path: Path to input Zarr store or Kerchunk reference
            output_path: Path to save output Zarr store
            start_year: First year of the period
            end_year: Last year of the period
            window_size: Size of rolling window (default: 30)
            min_years: Minimum years required for valid normal
            
        Returns:
            Dataset with calculated normals
        """
        logger.info(f"Calculating rolling normals for {start_year}-{end_year}")
        
        # Open dataset
        ds = self.open_zarr_dataset(input_path)
        
        # Select time period
        time_slice = slice(f"{start_year}-01-01", f"{end_year}-12-31")
        ds = ds.sel(time=time_slice)
        
        # Ensure we have the variable
        if self.variable not in ds.data_vars:
            raise ValueError(f"Variable '{self.variable}' not found in dataset")
        
        # Get data array
        da_var = ds[self.variable]
        
        # Convert time to pandas for easier manipulation
        time_index = pd.to_datetime(da_var.time.values)
        da_var = da_var.assign_coords(time=time_index)
        
        # Calculate daily climatology for each rolling window
        normals_list = []
        years = list(range(start_year, end_year - window_size + 2))
        
        # Create tasks for parallel processing
        tasks = []
        for window_start in years:
            window_end = window_start + window_size - 1
            task = dask.delayed(self._calculate_window_normal)(
                da_var, window_start, window_end, min_years
            )
            tasks.append(task)
        
        # Process windows in parallel
        logger.info(f"Processing {len(tasks)} rolling windows...")
        
        futures = self.client.compute(tasks)
        results = []
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing normals"):
            results.append(future.result())
        
        # Combine results
        normals = xr.concat(results, dim='year')
        normals = normals.sortby('year')
        
        # Create output dataset
        ds_out = xr.Dataset({
            f"{self.variable}_normal": normals,
            f"{self.variable}_count": normals.count(dim='dayofyear')
        })
        
        # Add metadata
        ds_out.attrs.update({
            'title': f'30-year rolling climate normals for {self.variable}',
            'institution': 'County Climate Project',
            'source': f'Calculated from {self.model} {self.scenario}',
            'history': f'Created on {datetime.now().isoformat()}',
            'variable': self.variable,
            'scenario': self.scenario,
            'region': self.region,
            'window_size': window_size,
            'min_years': min_years
        })
        
        # Save to Zarr
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        encoding = {
            f"{self.variable}_normal": {
                'compressor': ZarrConfig.DEFAULT_COMPRESSOR,
                'chunks': (1, 366, *normals.shape[2:])
            }
        }
        
        ds_out.to_zarr(
            output_path,
            mode='w',
            encoding=encoding,
            consolidated=True
        )
        
        logger.info(f"Saved rolling normals to {output_path}")
        
        return ds_out
    
    def _calculate_window_normal(
        self,
        da: xr.DataArray,
        start_year: int,
        end_year: int,
        min_years: int
    ) -> xr.DataArray:
        """Calculate normal for a single window."""
        # Select window
        window = da.sel(time=slice(f"{start_year}", f"{end_year}"))
        
        # Group by day of year and calculate mean
        normal = window.groupby('time.dayofyear').mean(dim='time')
        
        # Check if we have enough years
        years_available = len(np.unique(window.time.dt.year))
        if years_available < min_years:
            logger.warning(
                f"Window {start_year}-{end_year} has only {years_available} years "
                f"(minimum: {min_years})"
            )
            normal = normal.where(False)  # Set to NaN
        
        # Add year coordinate
        normal = normal.expand_dims(year=[start_year])
        
        return normal
    
    def process_hybrid_period(
        self,
        historical_path: Union[str, Path],
        scenario_path: Union[str, Path],
        output_path: Union[str, Path],
        overlap_years: Tuple[int, int] = (2015, 2015)
    ) -> xr.Dataset:
        """
        Process hybrid period combining historical and scenario data.
        
        Args:
            historical_path: Path to historical Zarr store
            scenario_path: Path to scenario Zarr store
            output_path: Path to save output
            overlap_years: Years that exist in both datasets
            
        Returns:
            Combined dataset
        """
        logger.info("Processing hybrid period")
        
        # Open datasets
        ds_hist = self.open_zarr_dataset(historical_path)
        ds_scen = self.open_zarr_dataset(scenario_path)
        
        # Remove overlap from scenario
        ds_scen = ds_scen.sel(
            time=ds_scen.time.dt.year > overlap_years[1]
        )
        
        # Concatenate
        ds_combined = xr.concat([ds_hist, ds_scen], dim='time')
        
        # Process combined dataset
        return self.calculate_rolling_normals_zarr(
            ds_combined,
            output_path,
            2015,
            2044
        )
    
    def create_multiscale_normals(
        self,
        input_path: Union[str, Path],
        output_base: Union[str, Path],
        scales: List[int] = [1, 2, 4, 8]
    ) -> None:
        """
        Create multiscale versions of climate normals for visualization.
        
        Args:
            input_path: Path to full-resolution normals
            output_base: Base path for multiscale outputs
            scales: Downsampling factors
        """
        logger.info(f"Creating multiscale normals with scales: {scales}")
        
        output_base = Path(output_base)
        output_base.mkdir(parents=True, exist_ok=True)
        
        # Open full resolution
        ds = self.open_zarr_dataset(input_path)
        
        for scale in scales:
            scale_path = output_base / f"scale_{scale}"
            logger.info(f"Creating scale {scale} at {scale_path}")
            
            if scale == 1:
                # Just copy
                ds.to_zarr(scale_path, mode='w', consolidated=True)
            else:
                # Downsample spatial dimensions
                ds_scaled = ds.coarsen(
                    lat=scale,
                    lon=scale,
                    boundary='trim'
                ).mean()
                
                ds_scaled.to_zarr(scale_path, mode='w', consolidated=True)
    
    def validate_output(self, output_path: Union[str, Path]) -> Dict:
        """Validate the output Zarr store."""
        return validate_zarr_store(output_path)