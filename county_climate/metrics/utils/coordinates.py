"""
Coordinate transformation utilities for climate data processing.

This module handles the conversion from indexed NetCDF coordinates to actual
geographic coordinates (lat/lon) for different climate datasets and regions.
"""

import numpy as np
import xarray as xr
from typing import Tuple, Optional


def convert_indexed_to_geographic_coords(ds: xr.Dataset, 
                                       region: str = "CONUS",
                                       lat_dim: str = "lat",
                                       lon_dim: str = "lon") -> xr.Dataset:
    """
    Convert indexed coordinates to actual geographic coordinates.
    
    Many climate NetCDF files use indexed coordinates (0, 1, 2, ...) instead of
    actual lat/lon values. This function converts them to proper geographic coordinates.
    
    Args:
        ds (xarray.Dataset): Dataset with indexed coordinates
        region (str): Geographic region ("CONUS", "AK", "HI", "PRVI", "GU")
        lat_dim (str): Name of latitude dimension
        lon_dim (str): Name of longitude dimension
        
    Returns:
        xarray.Dataset: Dataset with actual geographic coordinates
        
    Example:
        >>> ds = xr.open_dataset("climate_data.nc")
        >>> ds_geo = convert_indexed_to_geographic_coords(ds, region="CONUS")
        >>> print(f"Lat range: {ds_geo.lat.min():.2f} to {ds_geo.lat.max():.2f}")
    """
    print("Converting indexed coordinates to geographic coordinates...")
    
    # Get the indexed coordinate arrays
    lat_indices = ds[lat_dim].values
    lon_indices = ds[lon_dim].values
    
    # Get coordinate bounds for the region
    lat_min, lat_max, lon_min_360, lon_max_360 = get_region_bounds(region)
    
    # Create actual coordinate arrays
    actual_lats = np.linspace(lat_min, lat_max, len(lat_indices))
    actual_lons_360 = np.linspace(lon_min_360, lon_max_360, len(lon_indices))
    
    # Convert from 0-360 to -180 to 180 longitude system
    actual_lons = convert_longitude_360_to_180(actual_lons_360)
    
    # Update the dataset coordinates
    ds = ds.assign_coords({
        lat_dim: actual_lats,
        lon_dim: actual_lons
    })
    
    # Set spatial dimensions as coordinates if they aren't already
    if lat_dim not in ds.coords:
        ds = ds.set_coords([lat_dim, lon_dim])
    
    # Set CRS (WGS84 for lat/lon data)
    ds.rio.write_crs("EPSG:4326", inplace=True)
    
    print(f"Converted coordinates for {region}:")
    print(f"  Latitude range: {actual_lats.min():.2f}° to {actual_lats.max():.2f}°")
    print(f"  Longitude range: {actual_lons.min():.2f}° to {actual_lons.max():.2f}°")
    
    return ds


def get_region_bounds(region: str) -> Tuple[float, float, float, float]:
    """
    Get coordinate bounds for different US regions.
    
    Args:
        region (str): Region code ("CONUS", "AK", "HI", "PRVI", "GU")
        
    Returns:
        Tuple[float, float, float, float]: (lat_min, lat_max, lon_min_360, lon_max_360)
        
    Note:
        Longitude bounds are returned in 0-360 system and need conversion to -180/180
    """
    region_bounds = {
        "CONUS": {
            "lat_range": (20.0, 50.0),      # Continental US: ~20°N to 50°N
            "lon_range_360": (235.0, 295.0), # ~125°W to 65°W in 360 system
            "description": "Continental United States"
        },
        "AK": {
            "lat_range": (54.0, 72.0),      # Alaska: ~54°N to 72°N  
            "lon_range_360": (180.0, 235.0), # ~180°W to 125°W in 360 system
            "description": "Alaska"
        },
        "HI": {
            "lat_range": (18.0, 23.0),      # Hawaii: ~18°N to 23°N
            "lon_range_360": (200.0, 210.0), # ~160°W to 150°W in 360 system
            "description": "Hawaii"
        },
        "PRVI": {
            "lat_range": (17.0, 19.0),      # Puerto Rico/Virgin Islands: ~17°N to 19°N
            "lon_range_360": (292.0, 298.0), # ~68°W to 62°W in 360 system
            "description": "Puerto Rico and Virgin Islands"
        },
        "GU": {
            "lat_range": (13.0, 14.0),      # Guam: ~13°N to 14°N
            "lon_range_360": (144.0, 145.0), # ~144°E to 145°E in 360 system
            "description": "Guam"
        }
    }
    
    if region not in region_bounds:
        raise ValueError(f"Unknown region '{region}'. Available regions: {list(region_bounds.keys())}")
    
    bounds = region_bounds[region]
    lat_min, lat_max = bounds["lat_range"]
    lon_min_360, lon_max_360 = bounds["lon_range_360"]
    
    print(f"Using bounds for {bounds['description']} ({region})")
    
    return lat_min, lat_max, lon_min_360, lon_max_360


def convert_longitude_360_to_180(lon_360: np.ndarray) -> np.ndarray:
    """
    Convert longitude from 0-360 system to -180 to 180 system.
    
    Args:
        lon_360 (np.ndarray): Longitude values in 0-360 system
        
    Returns:
        np.ndarray: Longitude values in -180 to 180 system
        
    Example:
        >>> lon_360 = np.array([10, 180, 270, 350])
        >>> lon_180 = convert_longitude_360_to_180(lon_360)
        >>> print(lon_180)  # [10, 180, -90, -10]
    """
    return np.where(lon_360 > 180, lon_360 - 360, lon_360)


def convert_longitude_180_to_360(lon_180: np.ndarray) -> np.ndarray:
    """
    Convert longitude from -180 to 180 system to 0-360 system.
    
    Args:
        lon_180 (np.ndarray): Longitude values in -180 to 180 system
        
    Returns:
        np.ndarray: Longitude values in 0-360 system
        
    Example:
        >>> lon_180 = np.array([10, 180, -90, -10])
        >>> lon_360 = convert_longitude_180_to_360(lon_180)
        >>> print(lon_360)  # [10, 180, 270, 350]
    """
    return np.where(lon_180 < 0, lon_180 + 360, lon_180)


def validate_geographic_coordinates(ds: xr.Dataset, 
                                  lat_dim: str = "lat", 
                                  lon_dim: str = "lon") -> bool:
    """
    Validate that coordinates are in proper geographic ranges.
    
    Args:
        ds (xarray.Dataset): Dataset to validate
        lat_dim (str): Name of latitude dimension  
        lon_dim (str): Name of longitude dimension
        
    Returns:
        bool: True if coordinates are valid, False otherwise
        
    Raises:
        ValueError: If coordinates are invalid with detailed message
    """
    lats = ds[lat_dim].values
    lons = ds[lon_dim].values
    
    # Check latitude bounds
    if not (-90 <= lats.min() <= lats.max() <= 90):
        raise ValueError(f"Invalid latitude range: {lats.min():.2f} to {lats.max():.2f}. Must be between -90 and 90.")
    
    # Check longitude bounds (allow both -180/180 and 0/360 systems)
    if not ((-180 <= lons.min() <= lons.max() <= 180) or (0 <= lons.min() <= lons.max() <= 360)):
        raise ValueError(f"Invalid longitude range: {lons.min():.2f} to {lons.max():.2f}. Must be between -180/180 or 0/360.")
    
    # Check for reasonable coordinate spacing
    lat_spacing = np.diff(lats).mean()
    lon_spacing = np.diff(lons).mean()
    
    if lat_spacing <= 0:
        raise ValueError("Latitude coordinates are not in ascending order")
    if lon_spacing <= 0:
        raise ValueError("Longitude coordinates are not in ascending order")
    
    print(f"✅ Coordinates validated:")
    print(f"  Latitude: {lats.min():.2f}° to {lats.max():.2f}° (spacing: {lat_spacing:.3f}°)")
    print(f"  Longitude: {lons.min():.2f}° to {lons.max():.2f}° (spacing: {lon_spacing:.3f}°)")
    
    return True 