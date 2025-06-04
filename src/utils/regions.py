#!/usr/bin/env python3
"""
Regional Definitions and Operations for Climate Data Processing

This module contains regional boundary definitions, coordinate conversions,
and operations for extracting specific geographic regions from climate datasets.

Regional definitions cover:
- Continental United States (CONUS)
- Alaska (AK)
- Hawaii (HI)
- Puerto Rico and Virgin Islands (PRVI)
- Guam and Northern Mariana Islands (GU)
"""

import logging
import numpy as np
import xarray as xr
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# REGIONAL DEFINITIONS AND BOUNDS
# =============================================================================

# Define regional bounds (for 0-360 longitude system)
REGION_BOUNDS = {
    'CONUS': {
        'name': 'CONUS',
        'lon_min': 234,   # 234°E in 0-360 system (-126°E in -180/180)
        'lon_max': 294,   # 294°E in 0-360 system (-66°E in -180/180)
        'lat_min': 24.0,  # Extended south to fully cover Florida
        'lat_max': 50.0,  # Extended north to ensure coverage
        'convert_longitudes': True
    },
    'AK': {
        'name': 'Alaska',
        'lon_min': 170,   # 170°E in 0-360 system
        'lon_max': 235,   # 235°E in 0-360 system - extended to include SE Alaska
        'lat_min': 50.0,
        'lat_max': 72.0,
        'convert_longitudes': True
    },
    'HI': {
        'name': 'Hawaii and Islands',
        'lon_min': 181.63,   # 181.63 in 0-360 system
        'lon_max': 205.20,   # 205.20 in 0-360 system
        'lat_min': 18.92,
        'lat_max': 28.45,
        'convert_longitudes': True
    },
    'PRVI': {
        'name': 'Puerto Rico and U.S. Virgin Islands',
        'lon_min': 292.03,   # -67.97 in 0-360 system
        'lon_max': 295.49,   # -64.51 in 0-360 system
        'lat_min': 17.62,
        'lat_max': 18.57,
        'convert_longitudes': True
    },
    'GU': {
        'name': 'Guam and Northern Mariana Islands',
        'lon_min': 144.58,   # 144.58°E in 0-360 system
        'lon_max': 146.12,   # 146.12°E in 0-360 system
        'lat_min': 13.18,
        'lat_max': 20.61,
        'convert_longitudes': True
    }
}


def get_region_crs_info(region_key: str) -> Dict[str, Any]:
    """Get coordinate reference system information for a specific region."""
    region_crs = {
        'CONUS': {
            'crs_type': 'epsg',
            'crs_value': 5070,  # NAD83 / Conus Albers
            'central_longitude': -96,
            'central_latitude': 37.5,
            'extent': [-125, -65, 25, 50]  # West, East, South, North
        },
        'AK': {
            'crs_type': 'epsg',
            'crs_value': 3338,  # NAD83 / Alaska Albers
            'central_longitude': -154,
            'central_latitude': 50,
            'extent': [-170, -130, 50, 72]
        },
        'HI': {
            'crs_type': 'proj4',
            'crs_value': "+proj=aea +lat_1=8 +lat_2=18 +lat_0=13 +lon_0=157 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs",
            'central_longitude': -157,
            'central_latitude': 20,
            'extent': [-178, -155, 18, 29]
        },
        'PRVI': {
            'crs_type': 'epsg',
            'crs_value': 6566,  # NAD83(2011) / Puerto Rico and Virgin Islands
            'central_longitude': -66,
            'central_latitude': 18,
            'extent': [-68, -64, 17, 19]
        },
        'GU': {
            'crs_type': 'epsg',
            'crs_value': 32655,  # WGS 84 / UTM zone 55N
            'central_longitude': 147,
            'central_latitude': 13.5,
            'extent': [144, 147, 13, 21]
        }
    }
    
    return region_crs.get(region_key, {
        'crs_type': 'epsg',
        'crs_value': 5070,
        'central_longitude': -96,
        'central_latitude': 37.5,
        'extent': [-125, -65, 25, 50]
    })


def convert_longitude_bounds(lon_min: float, lon_max: float, is_0_360: bool) -> Dict[str, float]:
    """Convert longitude bounds between 0-360 and -180-180 coordinate systems."""
    if is_0_360:
        return {'lon_min': lon_min, 'lon_max': lon_max}
    else:
        converted_min = lon_min - 360 if lon_min > 180 else lon_min
        converted_max = lon_max - 360 if lon_max > 180 else lon_max
        return {'lon_min': converted_min, 'lon_max': converted_max}


def extract_region(ds: xr.Dataset, region_bounds: Dict) -> xr.Dataset:
    """Extract a specific region from the dataset with improved coordinate handling."""
    # Check coordinate names
    lon_name = 'lon' if 'lon' in ds.coords else 'x'
    lat_name = 'lat' if 'lat' in ds.coords else 'y'
    
    # Check longitude range (0-360 or -180-180)
    lon_min = ds[lon_name].min().item()
    lon_max = ds[lon_name].max().item()
    
    # Determine if we're using 0-360 or -180-180 coordinate system
    is_0_360 = lon_min >= 0 and lon_max > 180
    
    # Convert region bounds to match the dataset's coordinate system
    lon_bounds = convert_longitude_bounds(
        region_bounds['lon_min'], 
        region_bounds['lon_max'], 
        is_0_360
    )
    
    # Handle the case where we cross the 0/360 or -180/180 boundary
    if lon_bounds['lon_min'] > lon_bounds['lon_max']:
        region_ds = ds.where(
            ((ds[lon_name] >= lon_bounds['lon_min']) | 
             (ds[lon_name] <= lon_bounds['lon_max'])) & 
            (ds[lat_name] >= region_bounds['lat_min']) & 
            (ds[lat_name] <= region_bounds['lat_max']), 
            drop=True
        )
    else:
        region_ds = ds.where(
            (ds[lon_name] >= lon_bounds['lon_min']) & 
            (ds[lon_name] <= lon_bounds['lon_max']) & 
            (ds[lat_name] >= region_bounds['lat_min']) & 
            (ds[lat_name] <= region_bounds['lat_max']), 
            drop=True
        )
    
    # Check if we have data
    if region_ds[lon_name].size == 0 or region_ds[lat_name].size == 0:
        logger.warning(f"No data found within region bounds after filtering.")
        logger.warning(f"Dataset longitude range: {lon_min} to {lon_max}")
        logger.warning(f"Region bounds: {region_bounds['lon_min']} to {region_bounds['lon_max']} (original)")
        logger.warning(f"Converted bounds: {lon_bounds['lon_min']} to {lon_bounds['lon_max']}")
    
    return region_ds


def validate_region_bounds(region_key: str) -> bool:
    """Validate that a region key exists and has proper bounds."""
    if region_key not in REGION_BOUNDS:
        logger.error(f"Unknown region key: {region_key}")
        logger.info(f"Available regions: {list(REGION_BOUNDS.keys())}")
        return False
    
    region = REGION_BOUNDS[region_key]
    
    # Check required fields
    required_fields = ['name', 'lon_min', 'lon_max', 'lat_min', 'lat_max']
    missing_fields = [field for field in required_fields if field not in region]
    
    if missing_fields:
        logger.error(f"Region {region_key} missing required fields: {missing_fields}")
        return False
    
    # Check bounds validity
    if region['lon_min'] >= region['lon_max']:
        # Allow crossing dateline for specific cases
        if not (region['lon_min'] > 180 and region['lon_max'] < 180):
            logger.warning(f"Region {region_key} has invalid longitude bounds")
    
    if region['lat_min'] >= region['lat_max']:
        logger.error(f"Region {region_key} has invalid latitude bounds")
        return False
    
    if region['lat_min'] < -90 or region['lat_max'] > 90:
        logger.error(f"Region {region_key} has latitude bounds outside valid range")
        return False
    
    return True 