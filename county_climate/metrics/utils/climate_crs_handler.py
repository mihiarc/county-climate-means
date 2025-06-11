#!/usr/bin/env python3
"""
Comprehensive CRS (Coordinate Reference System) Handler for Climate Data Processing

This module implements proper coordinate system detection and transformation
for climate data, following the methodology documented in CRS_HANDLING_DOCUMENTATION.md

Key features:
- Automatic detection of indexed vs geographic coordinates
- Handling of 0-360° vs -180/180° longitude conventions  
- Regional boundary definitions for CONUS and other U.S. territories
- Validation and error checking
- Consistent coordinate transformations across all processing scripts
"""

import xarray as xr
import numpy as np
import logging
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Regional boundary definitions (0-360° longitude system)
REGION_BOUNDS = {
    'CONUS': {
        'name': 'Continental United States',
        'lat_min': 24.0,   # Extended south to fully cover Florida
        'lat_max': 50.0,   # Extended north to ensure full coverage
        'lon_min': 234.0,  # 234°E in 0-360 system (-126°W in -180/180)
        'lon_max': 294.0,  # 294°E in 0-360 system (-66°W in -180/180)
        'convert_longitudes': True
    },
    'AK': {
        'name': 'Alaska',
        'lat_min': 50.0,
        'lat_max': 72.0,
        'lon_min': 170.0,   # Crosses dateline
        'lon_max': 235.0,
        'convert_longitudes': True
    },
    'HI': {
        'name': 'Hawaii',
        'lat_min': 18.0,
        'lat_max': 29.0, 
        'lon_min': 182.0,   # 182°E = -178°W
        'lon_max': 205.0,   # 205°E = -155°W
        'convert_longitudes': True
    },
    'PRVI': {
        'name': 'Puerto Rico and Virgin Islands',
        'lat_min': 17.0,
        'lat_max': 19.0,
        'lon_min': 292.0,   # 292°E = -68°W
        'lon_max': 296.0,   # 296°E = -64°W
        'convert_longitudes': True
    },
    'GU': {
        'name': 'Guam and Northern Mariana Islands',
        'lat_min': 13.0,
        'lat_max': 21.0,
        'lon_min': 144.0,   # Eastern Pacific, no conversion needed
        'lon_max': 147.0,
        'convert_longitudes': False
    }
}

def detect_coordinate_system(ds: xr.Dataset) -> Dict[str, Any]:
    """
    Detect if NetCDF uses indexed or geographic coordinates and longitude convention.
    
    Args:
        ds: xarray Dataset to analyze
        
    Returns:
        Dictionary with coordinate system information:
        - is_indexed: True if coordinates are 0-based indices
        - is_0_360: True if longitude uses 0-360° convention
        - lon_name: Name of longitude coordinate ('lon' or 'x')
        - lat_name: Name of latitude coordinate ('lat' or 'y')
        - bounds: Coordinate ranges
    """
    
    # Check coordinate names
    lon_name = 'lon' if 'lon' in ds.coords else 'x'
    lat_name = 'lat' if 'lat' in ds.coords else 'y'
    
    if lon_name not in ds.coords or lat_name not in ds.coords:
        raise ValueError(f"Could not find longitude/latitude coordinates. Available coords: {list(ds.coords.keys())}")
    
    # Get coordinate ranges
    lon_min = float(ds[lon_name].min().item())
    lon_max = float(ds[lon_name].max().item())
    lat_min = float(ds[lat_name].min().item())
    lat_max = float(ds[lat_name].max().item())
    
    # Determine if coordinates are indexed (0-based) or geographic
    is_indexed = (
        (lon_min == 0 and lon_max < 500) and  # Longitude looks like indices
        (lat_min == 0 and lat_max < 200)      # Latitude looks like indices
    )
    
    # For geographic coordinates, determine longitude convention
    is_0_360 = False
    if not is_indexed:
        is_0_360 = lon_min >= 0 and lon_max > 180
    
    logger.info(f"Coordinate system detection:")
    logger.info(f"  Coordinate names: {lon_name}, {lat_name}")
    logger.info(f"  Longitude range: {lon_min} to {lon_max}")
    logger.info(f"  Latitude range: {lat_min} to {lat_max}")
    logger.info(f"  Is indexed: {is_indexed}")
    logger.info(f"  Is 0-360 longitude: {is_0_360}")
    
    return {
        'is_indexed': is_indexed,
        'is_0_360': is_0_360,
        'lon_name': lon_name,
        'lat_name': lat_name,
        'bounds': {
            'lon_min': lon_min,
            'lon_max': lon_max,
            'lat_min': lat_min,
            'lat_max': lat_max
        }
    }

def convert_longitude_bounds(lon_min: float, lon_max: float, to_0_360: bool = False) -> Tuple[float, float]:
    """
    Convert longitude bounds between 0-360 and -180/180 coordinate systems.
    
    Args:
        lon_min: Minimum longitude
        lon_max: Maximum longitude  
        to_0_360: If True, convert to 0-360; if False, convert to -180/180
        
    Returns:
        Tuple of (converted_lon_min, converted_lon_max)
    """
    
    if to_0_360:
        # Convert from -180/180 to 0-360
        converted_min = lon_min + 360 if lon_min < 0 else lon_min
        converted_max = lon_max + 360 if lon_max < 0 else lon_max
    else:
        # Convert from 0-360 to -180/180
        converted_min = lon_min - 360 if lon_min > 180 else lon_min
        converted_max = lon_max - 360 if lon_max > 180 else lon_max
    
    return converted_min, converted_max

def transform_indexed_to_geographic(ds: xr.Dataset, region: str = 'CONUS') -> xr.Dataset:
    """
    Transform indexed coordinates to proper geographic coordinates.
    
    Args:
        ds: Dataset with indexed coordinates
        region: Regional bounds to use ('CONUS', 'AK', 'HI', 'PRVI', 'GU')
        
    Returns:
        Dataset with geographic coordinates
    """
    
    coord_info = detect_coordinate_system(ds)
    
    if not coord_info['is_indexed']:
        logger.info("Dataset already uses geographic coordinates")
        
        # Check if longitude conversion is needed for geographic coordinates
        if region in REGION_BOUNDS:
            region_bounds = REGION_BOUNDS[region]
            if coord_info['is_0_360'] and region_bounds['convert_longitudes']:
                logger.info("Converting geographic coordinates from 0-360 to -180/180 longitude system")
                
                # Convert longitude coordinates
                lon_name = coord_info['lon_name']
                lon_coords = ds[lon_name].values
                
                # Convert longitudes > 180 to negative values
                converted_lons = np.where(lon_coords > 180, lon_coords - 360, lon_coords)
                
                # Update dataset with converted coordinates
                ds_converted = ds.assign_coords({lon_name: converted_lons})
                
                logger.info(f"  Original longitude range: {lon_coords.min():.3f}° to {lon_coords.max():.3f}°")
                logger.info(f"  Converted longitude range: {converted_lons.min():.3f}° to {converted_lons.max():.3f}°")
                
                return ds_converted
        
        return ds
    
    if region not in REGION_BOUNDS:
        raise ValueError(f"Unsupported region: {region}. Supported regions: {list(REGION_BOUNDS.keys())}")
    
    # Get the indexed coordinate arrays
    lat_indices = ds[coord_info['lat_name']].values
    lon_indices = ds[coord_info['lon_name']].values
    
    # Get regional bounds
    region_bounds = REGION_BOUNDS[region]
    
    logger.info(f"Transforming indexed coordinates for region: {region_bounds['name']}")
    logger.info(f"  Grid dimensions: {len(lat_indices)} lat x {len(lon_indices)} lon")
    logger.info(f"  Target bounds: {region_bounds['lat_min']}°N to {region_bounds['lat_max']}°N")
    logger.info(f"  Target bounds: {region_bounds['lon_min']}°E to {region_bounds['lon_max']}°E (0-360 system)")
    
    # Calculate geographic coordinates from indices
    # Assume linear mapping from indices to geographic bounds
    actual_lats = np.linspace(
        region_bounds['lat_min'],
        region_bounds['lat_max'],
        len(lat_indices)
    )
    
    actual_lons_360 = np.linspace(
        region_bounds['lon_min'],
        region_bounds['lon_max'],
        len(lon_indices)
    )
    
    # Convert to -180/180 longitude system if requested
    if region_bounds['convert_longitudes']:
        actual_lons = np.where(actual_lons_360 > 180, actual_lons_360 - 360, actual_lons_360)
        logger.info(f"  Converted to -180/180 system: {actual_lons.min():.3f}°W to {actual_lons.max():.3f}°W")
    else:
        actual_lons = actual_lons_360
        logger.info(f"  Keeping 0-360 system: {actual_lons.min():.3f}°E to {actual_lons.max():.3f}°E")
    
    # Update dataset coordinates
    ds_transformed = ds.assign_coords({
        coord_info['lat_name']: actual_lats,
        coord_info['lon_name']: actual_lons
    })
    
    logger.info(f"Coordinate transformation complete:")
    logger.info(f"  Final latitude range: {actual_lats.min():.3f}°N to {actual_lats.max():.3f}°N")
    logger.info(f"  Final longitude range: {actual_lons.min():.3f}° to {actual_lons.max():.3f}°")
    
    return ds_transformed

def validate_geographic_bounds(ds: xr.Dataset, region: str = 'CONUS') -> bool:
    """
    Validate that geographic coordinates are reasonable for the specified region.
    
    Args:
        ds: Dataset with geographic coordinates
        region: Expected region
        
    Returns:
        True if bounds are reasonable, False otherwise
    """
    
    coord_info = detect_coordinate_system(ds)
    
    if coord_info['is_indexed']:
        logger.error("Cannot validate bounds - dataset still uses indexed coordinates")
        return False
    
    bounds = coord_info['bounds']
    
    # Define expected bounds for validation (with tolerance)
    if region == 'CONUS':
        expected_lat_min, expected_lat_max = 20.0, 55.0
        expected_lon_min, expected_lon_max = -135.0, -60.0
    elif region == 'AK':
        expected_lat_min, expected_lat_max = 45.0, 75.0
        expected_lon_min, expected_lon_max = -180.0, -125.0
    elif region == 'HI':
        expected_lat_min, expected_lat_max = 15.0, 32.0
        expected_lon_min, expected_lon_max = -180.0, -150.0
    elif region == 'PRVI':
        expected_lat_min, expected_lat_max = 15.0, 22.0
        expected_lon_min, expected_lon_max = -70.0, -60.0
    elif region == 'GU':
        expected_lat_min, expected_lat_max = 10.0, 25.0
        expected_lon_min, expected_lon_max = 140.0, 150.0
    else:
        logger.warning(f"No validation bounds defined for region: {region}")
        return True
    
    # Check latitude bounds
    lat_valid = (expected_lat_min <= bounds['lat_min'] < bounds['lat_max'] <= expected_lat_max)
    
    # Check longitude bounds
    lon_valid = (expected_lon_min <= bounds['lon_min'] < bounds['lon_max'] <= expected_lon_max)
    
    if not lat_valid:
        logger.warning(f"Latitude bounds {bounds['lat_min']:.1f}° to {bounds['lat_max']:.1f}° "
                      f"outside expected range {expected_lat_min}° to {expected_lat_max}°")
    
    if not lon_valid:
        logger.warning(f"Longitude bounds {bounds['lon_min']:.1f}° to {bounds['lon_max']:.1f}° "
                      f"outside expected range {expected_lon_min}° to {expected_lon_max}°")
    
    return lat_valid and lon_valid

def load_climate_data_with_proper_crs(file_path: str, region: str = 'CONUS') -> xr.Dataset:
    """
    Load climate data with proper CRS handling and coordinate transformation.
    
    This is the main function that should be used by all county processing scripts
    to ensure consistent coordinate handling.
    
    Args:
        file_path: Path to NetCDF climate data file
        region: Regional bounds to use for coordinate transformation
        
    Returns:
        Dataset with properly transformed geographic coordinates
    """
    
    logger.info(f"Loading climate data: {file_path}")
    logger.info(f"Target region: {region}")
    
    # Load the dataset
    ds = xr.open_dataset(file_path)
    
    # Transform coordinates if needed
    ds_geo = transform_indexed_to_geographic(ds, region=region)
    
    # Set proper CRS metadata
    try:
        ds_geo.rio.write_crs("EPSG:4326", inplace=True)
        logger.info("Set CRS metadata: EPSG:4326 (WGS84)")
    except Exception as e:
        logger.warning(f"Could not set CRS metadata: {e}")
    
    # Validate the transformation
    coord_info = detect_coordinate_system(ds_geo)
    if coord_info['is_indexed']:
        ds.close()
        ds_geo.close()
        raise ValueError("Coordinate transformation failed - still using indexed coordinates")
    
    # Validate bounds are reasonable
    bounds_valid = validate_geographic_bounds(ds_geo, region)
    if not bounds_valid:
        logger.warning("Geographic bounds validation failed - results may be inaccurate")
    
    # Close original dataset to free memory
    ds.close()
    
    logger.info("Climate data loaded successfully with proper CRS handling")
    return ds_geo

def extract_region_bounds(ds: xr.Dataset, region: str) -> xr.Dataset:
    """
    Extract data within regional bounds with CRS-aware filtering.
    
    Args:
        ds: Dataset with geographic coordinates
        region: Region key for bounds lookup
        
    Returns:
        Dataset filtered to regional bounds
    """
    
    if region not in REGION_BOUNDS:
        raise ValueError(f"Unknown region: {region}")
    
    coord_info = detect_coordinate_system(ds)
    
    if coord_info['is_indexed']:
        raise ValueError("Cannot extract region bounds - dataset uses indexed coordinates")
    
    region_bounds = REGION_BOUNDS[region]
    lon_name = coord_info['lon_name']
    lat_name = coord_info['lat_name']
    
    # Convert region bounds to match dataset longitude convention
    if coord_info['is_0_360'] and region_bounds['convert_longitudes']:
        # Dataset uses 0-360, but region bounds are defined for -180/180
        target_lon_min, target_lon_max = convert_longitude_bounds(
            region_bounds['lon_min'] - 360,  # Convert from 0-360 definition to -180/180
            region_bounds['lon_max'] - 360,
            to_0_360=True
        )
    elif not coord_info['is_0_360'] and not region_bounds['convert_longitudes']:
        # Dataset uses -180/180, but region bounds are defined for 0-360
        target_lon_min, target_lon_max = convert_longitude_bounds(
            region_bounds['lon_min'],
            region_bounds['lon_max'],
            to_0_360=False
        )
    else:
        # No conversion needed
        target_lon_min = region_bounds['lon_min']
        target_lon_max = region_bounds['lon_max']
        if region_bounds['convert_longitudes']:
            target_lon_min, target_lon_max = convert_longitude_bounds(
                target_lon_min, target_lon_max, to_0_360=False
            )
    
    # Extract the region
    if target_lon_min > target_lon_max:
        # Handle dateline crossing
        region_ds = ds.where(
            ((ds[lon_name] >= target_lon_min) | (ds[lon_name] <= target_lon_max)) &
            (ds[lat_name] >= region_bounds['lat_min']) &
            (ds[lat_name] <= region_bounds['lat_max']),
            drop=True
        )
    else:
        # Standard rectangular region
        region_ds = ds.where(
            (ds[lon_name] >= target_lon_min) &
            (ds[lon_name] <= target_lon_max) &
            (ds[lat_name] >= region_bounds['lat_min']) &
            (ds[lat_name] <= region_bounds['lat_max']),
            drop=True
        )
    
    logger.info(f"Extracted region {region}: "
               f"{region_ds[lat_name].size} lat x {region_ds[lon_name].size} lon points")
    
    return region_ds

# Convenience function for existing scripts
def load_and_prepare_netcdf(netcdf_path: str, region: str = 'CONUS') -> xr.Dataset:
    """
    Convenience function that matches the signature used in existing scripts.
    
    This allows easy drop-in replacement of existing coordinate transformation code.
    """
    return load_climate_data_with_proper_crs(netcdf_path, region) 