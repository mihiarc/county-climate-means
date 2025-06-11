"""
NetCDF loading utilities for climate data processing.

Provides functions for loading and preparing NetCDF files with proper
coordinate transformations.
"""

import xarray as xr
from .coordinates import convert_indexed_to_geographic_coords, validate_geographic_coordinates


def load_and_prepare_netcdf(netcdf_path: str, region: str = "CONUS") -> xr.Dataset:
    """
    Load NetCDF file and prepare with proper coordinates.
    
    This is a convenience function that combines loading and coordinate conversion.
    
    Args:
        netcdf_path (str): Path to NetCDF file
        region (str): Geographic region for coordinate conversion
        
    Returns:
        xarray.Dataset: Loaded dataset with proper geographic coordinates
    """
    ds = xr.open_dataset(netcdf_path)
    ds = convert_indexed_to_geographic_coords(ds, region=region)
    validate_geographic_coordinates(ds)
    return ds 