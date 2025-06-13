#!/usr/bin/env python3
"""
County Handler Module

Handles loading, processing, and validation of US county boundary data
for climate analysis applications, using the centralized CountyBoundariesManager.
"""

import os
import geopandas as gpd
import pandas as pd
import warnings
from pathlib import Path
from typing import Optional, Union, List
import logging

from county_climate.shared.data import CountyBoundariesManager


def load_us_counties(shapefile_path: Optional[Union[str, Path]] = None, 
                    regions: Optional[List[str]] = None,
                    auto_download: bool = True,
                    target_crs: str = 'EPSG:4326',
                    validate: bool = True,
                    use_modern_format: bool = True,
                    simplified: bool = True) -> gpd.GeoDataFrame:
    """
    Load and prepare US county boundaries for climate analysis.
    
    This function loads county boundaries using the modern GeoParquet format by default,
    validates required columns, formats GEOIDs consistently, and ensures proper 
    coordinate reference system.
    
    Args:
        shapefile_path (str, Path, optional): Path to county shapefile (legacy). 
            If None and use_modern_format is True, uses CountyBoundariesManager.
        regions (List[str], optional): List of regions to include 
            (CONUS, AK, HI, PRVI, GU). If None, loads all regions.
        auto_download (bool): Legacy parameter for backwards compatibility.
        target_crs (str): Target coordinate reference system. 
            Default is 'EPSG:4326' (WGS84) to match NetCDF climate data.
        validate (bool): Whether to validate the loaded data. Default True.
        use_modern_format (bool): Whether to use modern GeoParquet format. Default True.
        simplified (bool): Whether to use simplified geometries for performance. Default True.
    
    Returns:
        geopandas.GeoDataFrame: County boundaries with standardized GEOID format
            and specified CRS.
    
    Raises:
        ValueError: If GEOID column is missing from shapefile or validation fails.
        FileNotFoundError: If shapefile path doesn't exist.
    
    Example:
        >>> # Load using modern format (recommended)
        >>> counties = load_us_counties()
        >>> print(f"Loaded {len(counties)} counties")
        
        >>> # Load specific regions only
        >>> counties = load_us_counties(regions=['CONUS', 'AK'])
        
        >>> # Load from existing shapefile (legacy)
        >>> counties = load_us_counties("my_counties.shp", use_modern_format=False)
    """
    logger = logging.getLogger(__name__)
    
    # Use modern format if requested
    if use_modern_format and shapefile_path is None:
        logger.info("Loading counties using modern GeoParquet format")
        
        # Initialize CountyBoundariesManager
        manager = CountyBoundariesManager()
        
        # Ensure GeoParquet exists
        manager.ensure_geoparquet_exists()
        
        # Load counties with optional filtering
        counties = manager.load_counties(
            regions=regions,
            simplified=simplified
        )
        
        # CRS is already WGS84 in the GeoParquet file
        if str(counties.crs) != target_crs:
            counties = counties.to_crs(target_crs)
            
        print(f"Loaded {len(counties)} counties using modern format")
        print(f"Example GEOID format: {counties['GEOID'].iloc[0]} (should be 5 characters)")
        print(f"CRS: {counties.crs}")
        
        return counties
    
    # Legacy shapefile loading (for backwards compatibility)
    logger.info("Loading counties using legacy shapefile format")
    
    # Set default shapefile path if not provided
    if shapefile_path is None:
        shapefile_path = "tl_2024_us_county/tl_2024_us_county.shp"
    
    shapefile_path = Path(shapefile_path)
    
    # If no shapefile exists and auto_download is enabled, download it
    if not shapefile_path.exists() and auto_download:
        logger.info("County shapefile not found, attempting to download from Census Bureau...")
        try:
            from .census_downloader import download_us_counties
            
            # Create output directory
            output_dir = shapefile_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Download counties
            downloaded_path = download_us_counties(
                regions=regions,
                output_dir=str(output_dir),
                force_refresh=False
            )
            
            # Update shapefile_path to the downloaded file
            shapefile_path = Path(downloaded_path)
            logger.info(f"Downloaded county boundaries to: {shapefile_path}")
            
        except ImportError as e:
            raise ImportError(
                "Auto-download requires additional dependencies. "
                "Install with: pip install geopandas requests"
            ) from e
        except Exception as e:
            raise FileNotFoundError(
                f"County shapefile not found: {shapefile_path}, "
                f"and auto-download failed: {e}"
            ) from e
    
    # Check if shapefile exists
    if not shapefile_path.exists():
        if auto_download:
            raise FileNotFoundError(
                f"County shapefile not found: {shapefile_path}, "
                "and auto-download failed to create it."
            )
        else:
            raise FileNotFoundError(
                f"County shapefile not found: {shapefile_path}. "
                "Set auto_download=True to download automatically."
            )
    
    print(f"Loading county boundaries from: {shapefile_path}")
    
    try:
        counties = gpd.read_file(shapefile_path)
    except Exception as e:
        raise FileNotFoundError(f"Could not load county shapefile: {shapefile_path}: {e}")
    
    if validate:
        # Validate required columns
        if 'GEOID' not in counties.columns:
            raise ValueError("Column 'GEOID' not found in county shapefile")
        
        # Check for null geometries
        if counties.geometry.isnull().any():
            null_count = counties.geometry.isnull().sum()
            warnings.warn(f"{null_count} counties have null geometry")
    
    # Standardize GEOID format (5-character string with leading zeros)
    counties['GEOID'] = counties['GEOID'].astype(str).str.zfill(5)
    
    # Handle CRS validation and reprojection
    original_crs = counties.crs
    if original_crs is None:
        warnings.warn("No CRS defined in shapefile, assuming EPSG:4326 (WGS84)")
        counties = counties.set_crs('EPSG:4326')
    elif str(original_crs) != target_crs:
        if str(original_crs) == 'EPSG:4269':
            # Expected NAD83, this is normal for Census shapefiles
            print(f"Reprojecting from NAD83 (EPSG:4269) to {target_crs}")
        else:
            warnings.warn(f"Unexpected county CRS: {original_crs}, reprojecting to {target_crs}")
        
        counties = counties.to_crs(target_crs)
    
    # Filter by regions if specified
    if regions:
        counties = filter_counties_by_regions(counties, regions)
    
    print(f"Loaded {len(counties)} counties")
    print(f"Example GEOID format: {counties['GEOID'].iloc[0]} (should be 5 characters)")
    print(f"CRS: {counties.crs}")
    
    return counties


def download_and_cache_counties(regions: Optional[List[str]] = None,
                               cache_dir: str = "cache/county_boundaries",
                               force_refresh: bool = False) -> str:
    """
    Download and cache county boundaries for reuse.
    
    Args:
        regions (List[str], optional): List of regions to download (CONUS, AK, HI, PRVI, GU).
            If None, downloads all regions.
        cache_dir (str): Directory to cache the downloaded boundaries.
        force_refresh (bool): Force re-download even if cached version exists.
        
    Returns:
        str: Path to the cached shapefile.
        
    Example:
        # Download and cache all regions
        shapefile_path = download_and_cache_counties()
        
        # Download only specific regions
        shapefile_path = download_and_cache_counties(['CONUS', 'AK'])
    """
    from .census_downloader import download_us_counties
    
    return download_us_counties(
        regions=regions,
        output_dir=cache_dir,
        force_refresh=force_refresh
    )


def filter_counties_by_region(counties: gpd.GeoDataFrame, region: str) -> gpd.GeoDataFrame:
    """
    Filter counties by climate analysis region.
    
    Args:
        counties (gpd.GeoDataFrame): County boundaries dataframe
        region (str): Region code (CONUS, AK, HI, PRVI, GU)
        
    Returns:
        gpd.GeoDataFrame: Filtered counties for the specified region
        
    Raises:
        ValueError: If invalid region specified
    """
    # Regional FIPS code mappings
    region_fips = {
        'CONUS': [str(i).zfill(2) for i in range(1, 57) if i not in [2, 15, 60, 66, 69, 72, 78]],
        'AK': ['02'],      # Alaska
        'HI': ['15'],      # Hawaii  
        'PRVI': ['72', '78'],  # Puerto Rico and Virgin Islands
        'GU': ['66', '69'],    # Guam and Northern Mariana Islands
    }
    
    if region not in region_fips:
        raise ValueError(f"Invalid region: {region}. Valid regions: {list(region_fips.keys())}")
    
    # Get state FIPS codes for the region
    target_fips = region_fips[region]
    
    # Filter counties by state FIPS (first 2 characters of GEOID)
    state_fips = counties['GEOID'].str[:2]
    filtered_counties = counties[state_fips.isin(target_fips)].copy()
    
    print(f"Filtered to {len(filtered_counties)} counties for region: {region}")
    
    return filtered_counties


def filter_counties_by_regions(counties: gpd.GeoDataFrame, regions: List[str]) -> gpd.GeoDataFrame:
    """
    Filter counties by multiple regions.
    
    Args:
        counties (gpd.GeoDataFrame): County boundaries dataframe
        regions (List[str]): List of region codes
        
    Returns:
        gpd.GeoDataFrame: Filtered counties for all specified regions
    """
    all_filtered = []
    
    for region in regions:
        filtered = filter_counties_by_region(counties, region)
        all_filtered.append(filtered)
    
    # Combine all regions
    combined = gpd.GeoDataFrame(pd.concat(all_filtered, ignore_index=True))
    
    # Remove any duplicates (shouldn't be any, but just in case)
    combined = combined.drop_duplicates(subset=['GEOID'])
    
    return combined


def validate_county_geometries(counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Validate and fix county geometries.
    
    Args:
        counties (gpd.GeoDataFrame): County boundaries dataframe
        
    Returns:
        gpd.GeoDataFrame: Counties with validated/fixed geometries
    """
    print("Validating county geometries...")
    
    # Check for invalid geometries
    invalid_mask = ~counties.geometry.is_valid
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        print(f"Found {invalid_count} invalid geometries, attempting to fix...")
        
        # Try to fix invalid geometries
        counties.loc[invalid_mask, 'geometry'] = counties.loc[invalid_mask, 'geometry'].buffer(0)
        
        # Check again
        still_invalid = (~counties.geometry.is_valid).sum()
        if still_invalid > 0:
            warnings.warn(f"{still_invalid} geometries could not be fixed")
        else:
            print("All geometries fixed successfully")
    
    # Check for null geometries
    null_count = counties.geometry.isnull().sum()
    if null_count > 0:
        warnings.warn(f"{null_count} counties have null geometry")
    
    print("Geometry validation complete")
    
    return counties


# Additional utility functions
def get_county_info(geoid: str) -> Optional[dict]:
    """
    Get information about a specific county using the modern format.
    
    Args:
        geoid: County GEOID
        
    Returns:
        Dictionary with county information or None if not found
    """
    manager = CountyBoundariesManager()
    return manager.get_county_info(geoid)


def get_county_centroids(counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate centroids for county geometries.
    
    Args:
        counties (gpd.GeoDataFrame): County boundaries dataframe
        
    Returns:
        gpd.GeoDataFrame: Counties with centroid coordinates added
    """
    # Make a copy to avoid modifying the original
    counties_with_centroids = counties.copy()
    
    # Calculate centroids
    centroids = counties.geometry.centroid
    
    # Add centroid coordinates as columns
    counties_with_centroids['centroid_lon'] = centroids.x
    counties_with_centroids['centroid_lat'] = centroids.y
    
    # Also add the centroid geometry as a separate column if needed
    counties_with_centroids['centroid'] = centroids
    
    return counties_with_centroids