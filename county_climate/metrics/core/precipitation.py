"""
Precipitation (pr) variable processing for county-level climate metrics.

This module provides functions to process precipitation data and calculate
county-level metrics including annual totals and extreme precipitation days.
"""

import os
import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from typing import Optional
from ..utils import load_us_counties, calculate_annual_precipitation_from_daily
from ..utils.netcdf_loader import load_and_prepare_netcdf


def _detect_precipitation_variable(ds: xr.Dataset) -> str:
    """
    Detect the correct precipitation variable name in the dataset.
    
    Args:
        ds (xarray.Dataset): Dataset to check
        
    Returns:
        str: Name of the precipitation variable
        
    Raises:
        ValueError: If no precipitation variable is found
    """
    # Check for standard climate variable names first
    standard_precip_vars = ['pr', 'precip', 'precipitation', 'rain', 'prcp']
    for var in standard_precip_vars:
        if var in ds.data_vars:
            return var
    
    # Check for the generic xarray variable name
    if '__xarray_dataarray_variable__' in ds.data_vars:
        return '__xarray_dataarray_variable__'
    
    # If none found, use the first data variable
    data_vars = list(ds.data_vars.keys())
    if data_vars:
        return data_vars[0]
    
    raise ValueError(f"No precipitation variable found in dataset. Available variables: {list(ds.data_vars.keys())}")


def calculate_annual_precipitation(ds: xr.Dataset, counties: gpd.GeoDataFrame, region: str = "CONUS") -> pd.DataFrame:
    """
    Calculate annual precipitation for each county from daily precipitation data.
    
    Combines daily-to-annual calculation with county-level extraction to return
    a DataFrame ready for CSV export.
    
    Args:
        ds (xarray.Dataset): Dataset with daily precipitation data
        counties (geopandas.GeoDataFrame): County boundaries  
        region (str): Geographic region for coordinate handling
        
    Returns:
        pandas.DataFrame: County-level annual precipitation statistics with columns:
            - GEOID: County identifier
            - annual_precipitation_mm: Mean annual precipitation
            - min_precipitation_mm: Min annual precipitation within county
            - max_precipitation_mm: Max annual precipitation within county  
            - pixel_count: Number of grid cells used for calculation
    """
    print("Calculating annual precipitation totals...")
    
    # Step 1: Daily-to-Annual calculation with variable detection
    precip_var_name = _detect_precipitation_variable(ds)
    
    # Convert from kg m^(-2) s^(-1) to mm/day
    pr_daily_mm = ds[precip_var_name] * 86400
    
    # Calculate annual total precipitation
    annual_precip = pr_daily_mm.sum(dim='dayofyear')
    annual_precip.attrs['units'] = 'mm/year'
    annual_precip.attrs['long_name'] = 'Total Annual Precipitation'
    
    print(f"Annual precipitation range: {float(annual_precip.min()):.1f} - {float(annual_precip.max()):.1f} mm/year")
    
    # Step 2: County-level extraction  
    print("Extracting precipitation by county...")
    
    # Get raster properties
    height, width = annual_precip.shape
    lat_min, lat_max = float(annual_precip.lat.min()), float(annual_precip.lat.max())
    lon_min, lon_max = float(annual_precip.lon.min()), float(annual_precip.lon.max())
    
    print(f"  Data grid bounds: {lat_min:.3f}°N to {lat_max:.3f}°N, {lon_min:.3f}° to {lon_max:.3f}°")
    
    # Create transform for rasterization
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    
    # Get coordinate arrays for nearest neighbor lookup
    lats = annual_precip.lat.values
    lons = annual_precip.lon.values
    
    results = []
    counties_fixed = 0
    counties_processed = 0
    
    for idx, county in counties.iterrows():
        geoid = county['GEOID']
        geometry = county['geometry']
        counties_processed += 1
        
        try:
            # Create a mask for this county
            mask = rasterize(
                [(geometry, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype='uint8'
            )
            
            # Apply mask to precipitation data
            masked_precip = np.where(mask == 1, annual_precip.values, np.nan)
            
            # Calculate statistics
            valid_pixels = ~np.isnan(masked_precip)
            if valid_pixels.sum() > 0:
                # Standard case: county intersects with grid cells
                mean_precip = np.nanmean(masked_precip)
                min_precip = np.nanmin(masked_precip)
                max_precip = np.nanmax(masked_precip)
                pixel_count = valid_pixels.sum()
            else:
                # Fallback: use nearest neighbor for small counties
                centroid = geometry.centroid
                centroid_lon, centroid_lat = centroid.x, centroid.y
                
                # Check if centroid is within grid bounds
                if (lon_min <= centroid_lon <= lon_max and 
                    lat_min <= centroid_lat <= lat_max):
                    
                    # Find nearest grid cell
                    lon_diff = np.abs(lons - centroid_lon)
                    lat_diff = np.abs(lats - centroid_lat)
                    nearest_lon_idx = np.argmin(lon_diff)
                    nearest_lat_idx = np.argmin(lat_diff)
                    
                    # Calculate distances and check if it's a direct neighbor
                    min_lon_distance = lon_diff[nearest_lon_idx]
                    min_lat_distance = lat_diff[nearest_lat_idx]
                    lon_spacing = (lon_max - lon_min) / (len(lons) - 1)
                    lat_spacing = (lat_max - lat_min) / (len(lats) - 1)
                    
                    if (min_lon_distance <= 0.75 * lon_spacing and 
                        min_lat_distance <= 0.75 * lat_spacing):
                        
                        # Get precipitation value from nearest grid cell
                        nearest_precip = annual_precip.isel(
                            lon=nearest_lon_idx, 
                            lat=nearest_lat_idx
                        ).values
                        
                        if not np.isnan(nearest_precip):
                            mean_precip = float(nearest_precip)
                            min_precip = float(nearest_precip)
                            max_precip = float(nearest_precip)
                            pixel_count = 1
                            counties_fixed += 1
                        else:
                            mean_precip = np.nan
                            min_precip = np.nan
                            max_precip = np.nan
                            pixel_count = 0
                    else:
                        mean_precip = np.nan
                        min_precip = np.nan
                        max_precip = np.nan
                        pixel_count = 0
                else:
                    mean_precip = np.nan
                    min_precip = np.nan
                    max_precip = np.nan
                    pixel_count = 0
            
            results.append({
                'GEOID': geoid,
                'annual_precipitation_mm': mean_precip,
                'min_precipitation_mm': min_precip,
                'max_precipitation_mm': max_precip,
                'pixel_count': pixel_count
            })
            
        except Exception as e:
            print(f"Warning: Error processing county {geoid}: {e}")
            results.append({
                'GEOID': geoid,
                'annual_precipitation_mm': np.nan,
                'min_precipitation_mm': np.nan,
                'max_precipitation_mm': np.nan,
                'pixel_count': 0
            })
    
    print(f"  Processed {counties_processed} counties, fixed {counties_fixed} small counties")
    
    if counties_fixed > 0:
        print(f"✅ Fixed {counties_fixed} small counties using nearest neighbor assignment")
    
    return pd.DataFrame(results)


def calculate_high_precip_days_95th(ds: xr.Dataset, counties: gpd.GeoDataFrame, region: str = "CONUS") -> pd.DataFrame:
    """
    Calculate high precipitation days (95th percentile threshold) for each county.
    
    Uses adaptive thresholds based on the 95th percentile of daily precipitation
    values, making it appropriate for climate normals data.
    
    Args:
        ds (xarray.Dataset): Dataset with daily precipitation data
        counties (geopandas.GeoDataFrame): County boundaries  
        region (str): Geographic region for coordinate handling
        
    Returns:
        pandas.DataFrame: County-level high precipitation days statistics with columns:
            - GEOID: County identifier
            - high_precip_days_95th: Mean days above 95th percentile per year
            - percentile_threshold_mm: The 95th percentile threshold value (mm/day)
            - pixel_count: Number of grid cells used for calculation
    """
    print("Calculating high precipitation days (95th percentile threshold)...")
    
    # Step 1: Daily-to-Annual calculation with percentile threshold
    precip_var_name = _detect_precipitation_variable(ds)
    
    # Convert from kg m^(-2) s^(-1) to mm/day
    pr_daily_mm = ds[precip_var_name] * 86400
    
    # Calculate 95th percentile threshold for the entire dataset
    percentile_95 = float(pr_daily_mm.quantile(0.95))
    
    # Calculate high precipitation days using the 95th percentile threshold
    high_precip_days = (pr_daily_mm > percentile_95).sum(dim='dayofyear')
    high_precip_days.attrs['units'] = 'days/year'
    high_precip_days.attrs['long_name'] = f'High Precipitation Days (>{percentile_95:.1f} mm/day, 95th percentile)'
    
    print(f"95th percentile threshold: {percentile_95:.1f} mm/day")
    print(f"High precipitation days range: {int(high_precip_days.min())} - {int(high_precip_days.max())} days/year")
    print("Note: Using percentile-based threshold appropriate for climate normals data.")
    
    # Step 2: County-level extraction
    print("Extracting high precipitation days by county...")
    
    # Get raster properties
    height, width = high_precip_days.shape
    lat_min, lat_max = float(high_precip_days.lat.min()), float(high_precip_days.lat.max())
    lon_min, lon_max = float(high_precip_days.lon.min()), float(high_precip_days.lon.max())
    
    print(f"  Data grid bounds: {lat_min:.3f}°N to {lat_max:.3f}°N, {lon_min:.3f}° to {lon_max:.3f}°")
    
    # Create transform for rasterization
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    
    # Get coordinate arrays for nearest neighbor lookup
    lats = high_precip_days.lat.values
    lons = high_precip_days.lon.values
    
    results = []
    counties_fixed = 0
    counties_processed = 0
    
    for idx, county in counties.iterrows():
        geoid = county['GEOID']
        geometry = county['geometry']
        counties_processed += 1
        
        try:
            # Create a mask for this county
            mask = rasterize(
                [(geometry, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype='uint8'
            )
            
            # Apply mask to high precipitation days data
            masked_high_precip_days = np.where(mask == 1, high_precip_days.values, np.nan)
            
            # Calculate statistics
            valid_pixels = ~np.isnan(masked_high_precip_days)
            if valid_pixels.sum() > 0:
                # Standard case: county intersects with grid cells
                mean_high_precip_days = np.nanmean(masked_high_precip_days)
                pixel_count = valid_pixels.sum()
            else:
                # Fallback: use nearest neighbor for small counties
                centroid = geometry.centroid
                centroid_lon, centroid_lat = centroid.x, centroid.y
                
                # Check if centroid is within grid bounds
                if (lon_min <= centroid_lon <= lon_max and 
                    lat_min <= centroid_lat <= lat_max):
                    
                    # Find nearest grid cell
                    lon_diff = np.abs(lons - centroid_lon)
                    lat_diff = np.abs(lats - centroid_lat)
                    nearest_lon_idx = np.argmin(lon_diff)
                    nearest_lat_idx = np.argmin(lat_diff)
                    
                    # Calculate distances and check if it's a direct neighbor
                    min_lon_distance = lon_diff[nearest_lon_idx]
                    min_lat_distance = lat_diff[nearest_lat_idx]
                    lon_spacing = (lon_max - lon_min) / (len(lons) - 1)
                    lat_spacing = (lat_max - lat_min) / (len(lats) - 1)
                    
                    if (min_lon_distance <= 0.75 * lon_spacing and 
                        min_lat_distance <= 0.75 * lat_spacing):
                        
                        # Get high precipitation days value from nearest grid cell
                        nearest_high_precip_days = high_precip_days.isel(
                            lon=nearest_lon_idx,
                            lat=nearest_lat_idx
                        ).values
                        
                        if not np.isnan(nearest_high_precip_days):
                            mean_high_precip_days = float(nearest_high_precip_days)
                            pixel_count = 1
                            counties_fixed += 1
                        else:
                            mean_high_precip_days = np.nan
                            pixel_count = 0
                    else:
                        mean_high_precip_days = np.nan
                        pixel_count = 0
                else:
                    mean_high_precip_days = np.nan
                    pixel_count = 0
            
            results.append({
                'GEOID': geoid,
                'high_precip_days_95th': mean_high_precip_days,
                'percentile_threshold_mm': percentile_95,
                'pixel_count': pixel_count
            })
            
        except Exception as e:
            print(f"Warning: Error processing county {geoid}: {e}")
            results.append({
                'GEOID': geoid,
                'high_precip_days_95th': np.nan,
                'percentile_threshold_mm': percentile_95,
                'pixel_count': 0
            })
    
    print(f"  Processed {counties_processed} counties, fixed {counties_fixed} small counties")
    
    if counties_fixed > 0:
        print(f"✅ Fixed {counties_fixed} small counties using nearest neighbor assignment")
    
    return pd.DataFrame(results)


def process_pr(scenario: str = "historical", 
               region: str = "CONUS",
               output_dir: str = "output",
               include_extremes: bool = True,
               counties_shapefile: Optional[str] = None) -> str:
    """
    Process precipitation data for a specific scenario and region.
    
    Calculates county-level precipitation metrics across the full time series
    (1980-2100) including annual totals and extreme precipitation days.
    
    Args:
        scenario (str): Climate scenario ("historical", "hybrid", "ssp245")
        region (str): Geographic region ("CONUS", "AK", "HI", "PRVI", "GU")
        output_dir (str): Directory for output files
        include_extremes (bool): Whether to include extreme precipitation metrics
        counties_shapefile (str, optional): Path to custom county shapefile
        
    Returns:
        str: Path to the output CSV file
        
    Example:
        >>> output_file = process_pr(scenario="ssp245", region="AK")
        >>> print(f"Results saved to {output_file}")
    """
    print(f"=== PROCESSING PRECIPITATION (pr) ===")
    print(f"Scenario: {scenario}")
    print(f"Region: {region}")
    print(f"Include extremes: {include_extremes}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load county boundaries
    print("Loading county boundaries...")
    counties = load_us_counties(counties_shapefile)
    
    # Determine which processing function to use based on extremes flag
    if include_extremes:
        from .county_pr_total_with_heavy_rain import main as process_precip_with_extremes
        
        # For now, this is a placeholder - we'll need to modify the existing
        # functions to work with the new API structure
        print("Processing precipitation with extreme metrics...")
        
        # Generate output filename
        output_file = os.path.join(
            output_dir, 
            f"county_pr_timeseries_{region}_{scenario}_1980_2100.csv"
        )
        
        # TODO: This needs to be updated to work with the batch processing
        # For now, return a placeholder
        print(f"TODO: Implement full time series processing")
        print(f"Would save to: {output_file}")
        
        return output_file
    else:
        from .county_pr_total import main as process_precip_basic
        
        print("Processing basic precipitation metrics...")
        
        # Generate output filename  
        output_file = os.path.join(
            output_dir,
            f"county_pr_basic_{region}_{scenario}_1980_2100.csv"
        )
        
        print(f"TODO: Implement full time series processing")
        print(f"Would save to: {output_file}")
        
        return output_file 