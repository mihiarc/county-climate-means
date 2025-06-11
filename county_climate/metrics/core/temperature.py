"""
Temperature variable processing for county-level climate metrics.

This module provides functions to process temperature data and calculate
county-level metrics for different temperature variables:
- tas: Near-surface air temperature (mean)
- tasmin: Minimum near-surface air temperature 
- tasmax: Maximum near-surface air temperature
"""

import os
import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from typing import Optional
from ..utils import load_us_counties
from ..utils.netcdf_loader import load_and_prepare_netcdf


def _detect_temperature_variable(ds: xr.Dataset) -> str:
    """
    Detect the correct temperature variable name in the dataset.
    
    Args:
        ds (xarray.Dataset): Dataset to check
        
    Returns:
        str: Name of the temperature variable
        
    Raises:
        ValueError: If no temperature variable is found
    """
    # Check for standard climate variable names first
    standard_temp_vars = ['tas', 'tasmin', 'tasmax', 'temp', 'temperature', 'air_temperature']
    for var in standard_temp_vars:
        if var in ds.data_vars:
            return var
    
    # Check for the generic xarray variable name
    if '__xarray_dataarray_variable__' in ds.data_vars:
        return '__xarray_dataarray_variable__'
    
    # If none found, use the first data variable
    data_vars = list(ds.data_vars.keys())
    if data_vars:
        return data_vars[0]
    
    raise ValueError(f"No temperature variable found in dataset. Available variables: {list(ds.data_vars.keys())}")


def calculate_annual_mean_temperature(ds: xr.Dataset, counties: gpd.GeoDataFrame, region: str = "CONUS") -> pd.DataFrame:
    """
    Calculate annual mean temperature for each county from daily temperature data.
    
    Args:
        ds (xarray.Dataset): Dataset with daily temperature data (tas variable)
        counties (geopandas.GeoDataFrame): County boundaries  
        region (str): Geographic region for coordinate handling
        
    Returns:
        pandas.DataFrame: County-level annual mean temperature with columns:
            - GEOID: County identifier
            - annual_mean_temp_c: Mean annual temperature (°C)
            - pixel_count: Number of grid cells used for calculation
    """
    print("Calculating annual mean temperature...")
    
    # Step 1: Daily-to-Annual calculation with variable detection
    temp_var_name = _detect_temperature_variable(ds)
    
    # Temperature data is in Kelvin, convert to Celsius
    temp_daily_c = ds[temp_var_name] - 273.15
    
    # Calculate annual mean temperature
    annual_mean_temp = temp_daily_c.mean(dim='dayofyear')
    annual_mean_temp.attrs['units'] = 'degrees_C'
    annual_mean_temp.attrs['long_name'] = 'Annual Mean Temperature'
    
    print(f"Annual mean temperature range: {float(annual_mean_temp.min()):.1f} - {float(annual_mean_temp.max()):.1f} °C")
    
    # Step 2: County-level extraction
    print("Extracting mean temperature by county...")
    
    # Get raster properties
    height, width = annual_mean_temp.shape
    lat_min, lat_max = float(annual_mean_temp.lat.min()), float(annual_mean_temp.lat.max())
    lon_min, lon_max = float(annual_mean_temp.lon.min()), float(annual_mean_temp.lon.max())
    
    print(f"  Data grid bounds: {lat_min:.3f}°N to {lat_max:.3f}°N, {lon_min:.3f}° to {lon_max:.3f}°")
    
    # Create transform for rasterization
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    
    # Get coordinate arrays for nearest neighbor lookup
    lats = annual_mean_temp.lat.values
    lons = annual_mean_temp.lon.values
    
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
            
            # Apply mask to temperature data
            masked_temp = np.where(mask == 1, annual_mean_temp.values, np.nan)
            
            # Calculate statistics
            valid_pixels = ~np.isnan(masked_temp)
            if valid_pixels.sum() > 0:
                # Standard case: county intersects with grid cells
                mean_temp = np.nanmean(masked_temp)
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
                        
                        # Get temperature value from nearest grid cell
                        nearest_temp = annual_mean_temp.isel(
                            lon=nearest_lon_idx, 
                            lat=nearest_lat_idx
                        ).values
                        
                        if not np.isnan(nearest_temp):
                            mean_temp = float(nearest_temp)
                            pixel_count = 1
                            counties_fixed += 1
                        else:
                            mean_temp = np.nan
                            pixel_count = 0
                    else:
                        mean_temp = np.nan
                        pixel_count = 0
                else:
                    mean_temp = np.nan
                    pixel_count = 0
            
            results.append({
                'GEOID': geoid,
                'annual_mean_temp_c': mean_temp,
                'pixel_count': pixel_count
            })
            
        except Exception as e:
            print(f"Warning: Error processing county {geoid}: {e}")
            results.append({
                'GEOID': geoid,
                'annual_mean_temp_c': np.nan,
                'pixel_count': 0
            })
    
    print(f"  Processed {counties_processed} counties, fixed {counties_fixed} small counties")
    
    if counties_fixed > 0:
        print(f"✅ Fixed {counties_fixed} small counties using nearest neighbor assignment")
    
    return pd.DataFrame(results)


def calculate_growing_degree_days(ds: xr.Dataset, counties: gpd.GeoDataFrame, region: str = "CONUS") -> pd.DataFrame:
    """
    Calculate growing degree days (base 10°C) for each county from daily temperature data.
    
    Args:
        ds (xarray.Dataset): Dataset with daily temperature data (tas variable)
        counties (geopandas.GeoDataFrame): County boundaries  
        region (str): Geographic region for coordinate handling
        
    Returns:
        pandas.DataFrame: County-level growing degree days with columns:
            - GEOID: County identifier
            - growing_degree_days: Annual cumulative degree-days above 10°C
            - pixel_count: Number of grid cells used for calculation
    """
    print("Calculating growing degree days (base 10°C)...")
    
    # Step 1: Daily-to-Annual calculation
    temp_var_name = _detect_temperature_variable(ds)
    
    # Temperature data is in Kelvin, convert to Celsius
    temp_daily_c = ds[temp_var_name] - 273.15
    
    # Calculate growing degree days (base 10°C / 50°F)
    growing_degree_days = np.maximum(temp_daily_c - 10.0, 0.0).sum(dim='dayofyear')
    growing_degree_days.attrs['units'] = 'degree_days'
    growing_degree_days.attrs['long_name'] = 'Growing Degree Days (base 10°C)'
    
    print(f"Growing degree days range: {float(growing_degree_days.min()):.0f} - {float(growing_degree_days.max()):.0f} degree-days")
    
    # Step 2: County-level extraction
    print("Extracting growing degree days by county...")
    
    # Get raster properties
    height, width = growing_degree_days.shape
    lat_min, lat_max = float(growing_degree_days.lat.min()), float(growing_degree_days.lat.max())
    lon_min, lon_max = float(growing_degree_days.lon.min()), float(growing_degree_days.lon.max())
    
    print(f"  Data grid bounds: {lat_min:.3f}°N to {lat_max:.3f}°N, {lon_min:.3f}° to {lon_max:.3f}°")
    
    # Create transform for rasterization
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    
    # Get coordinate arrays for nearest neighbor lookup
    lats = growing_degree_days.lat.values
    lons = growing_degree_days.lon.values
    
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
            
            # Apply mask to degree days data
            masked_gdd = np.where(mask == 1, growing_degree_days.values, np.nan)
            
            # Calculate statistics
            valid_pixels = ~np.isnan(masked_gdd)
            if valid_pixels.sum() > 0:
                # Standard case: county intersects with grid cells
                mean_gdd = np.nanmean(masked_gdd)
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
                        
                        # Get degree days value from nearest grid cell
                        nearest_gdd = growing_degree_days.isel(
                            lon=nearest_lon_idx, 
                            lat=nearest_lat_idx
                        ).values
                        
                        if not np.isnan(nearest_gdd):
                            mean_gdd = float(nearest_gdd)
                            pixel_count = 1
                            counties_fixed += 1
                        else:
                            mean_gdd = np.nan
                            pixel_count = 0
                    else:
                        mean_gdd = np.nan
                        pixel_count = 0
                else:
                    mean_gdd = np.nan
                    pixel_count = 0
            
            results.append({
                'GEOID': geoid,
                'growing_degree_days': mean_gdd,
                'pixel_count': pixel_count
            })
            
        except Exception as e:
            print(f"Warning: Error processing county {geoid}: {e}")
            results.append({
                'GEOID': geoid,
                'growing_degree_days': np.nan,
                'pixel_count': 0
            })
    
    print(f"  Processed {counties_processed} counties, fixed {counties_fixed} small counties")
    
    if counties_fixed > 0:
        print(f"✅ Fixed {counties_fixed} small counties using nearest neighbor assignment")
    
    return pd.DataFrame(results)


def calculate_high_temperature_days_90th(ds: xr.Dataset, counties: gpd.GeoDataFrame, region: str = "CONUS") -> pd.DataFrame:
    """
    Calculate high temperature days (90th percentile threshold) for each county.
    
    For tasmax data - uses adaptive thresholds based on the 90th percentile.
    
    Args:
        ds (xarray.Dataset): Dataset with daily temperature data (tasmax variable)
        counties (geopandas.GeoDataFrame): County boundaries  
        region (str): Geographic region for coordinate handling
        
    Returns:
        pandas.DataFrame: County-level high temperature days with columns:
            - GEOID: County identifier
            - high_temp_days_90th: Days above 90th percentile per year
            - percentile_threshold_c: The 90th percentile threshold value (°C)
            - pixel_count: Number of grid cells used for calculation
    """
    print("Calculating high temperature days (90th percentile threshold)...")
    
    # Step 1: Daily-to-Annual calculation with percentile threshold
    temp_var_name = _detect_temperature_variable(ds)
    
    # Temperature data is in Kelvin, convert to Celsius
    temp_daily_c = ds[temp_var_name] - 273.15
    
    # Calculate 90th percentile threshold for the entire dataset
    percentile_90 = float(temp_daily_c.quantile(0.90))
    
    # Calculate high temperature days using the 90th percentile threshold
    high_temp_days = (temp_daily_c > percentile_90).sum(dim='dayofyear')
    high_temp_days.attrs['units'] = 'days/year'
    high_temp_days.attrs['long_name'] = f'High Temperature Days (>{percentile_90:.1f}°C, 90th percentile)'
    
    print(f"90th percentile threshold: {percentile_90:.1f}°C")
    print(f"High temperature days range: {int(high_temp_days.min())} - {int(high_temp_days.max())} days/year")
    print("Note: Using percentile-based threshold appropriate for climate normals data.")
    
    # Step 2: County-level extraction
    print("Extracting high temperature days by county...")
    
    # Get raster properties
    height, width = high_temp_days.shape
    lat_min, lat_max = float(high_temp_days.lat.min()), float(high_temp_days.lat.max())
    lon_min, lon_max = float(high_temp_days.lon.min()), float(high_temp_days.lon.max())
    
    print(f"  Data grid bounds: {lat_min:.3f}°N to {lat_max:.3f}°N, {lon_min:.3f}° to {lon_max:.3f}°")
    
    # Create transform for rasterization
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    
    # Get coordinate arrays for nearest neighbor lookup
    lats = high_temp_days.lat.values
    lons = high_temp_days.lon.values
    
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
            
            # Apply mask to high temperature days data
            masked_high_temp_days = np.where(mask == 1, high_temp_days.values, np.nan)
            
            # Calculate statistics
            valid_pixels = ~np.isnan(masked_high_temp_days)
            if valid_pixels.sum() > 0:
                # Standard case: county intersects with grid cells
                mean_high_temp_days = np.nanmean(masked_high_temp_days)
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
                        
                        # Get high temperature days value from nearest grid cell
                        nearest_high_temp_days = high_temp_days.isel(
                            lon=nearest_lon_idx,
                            lat=nearest_lat_idx
                        ).values
                        
                        if not np.isnan(nearest_high_temp_days):
                            mean_high_temp_days = float(nearest_high_temp_days)
                            pixel_count = 1
                            counties_fixed += 1
                        else:
                            mean_high_temp_days = np.nan
                            pixel_count = 0
                    else:
                        mean_high_temp_days = np.nan
                        pixel_count = 0
                else:
                    mean_high_temp_days = np.nan
                    pixel_count = 0
            
            results.append({
                'GEOID': geoid,
                'high_temp_days_90th': mean_high_temp_days,
                'percentile_threshold_c': percentile_90,
                'pixel_count': pixel_count
            })
            
        except Exception as e:
            print(f"Warning: Error processing county {geoid}: {e}")
            results.append({
                'GEOID': geoid,
                'high_temp_days_90th': np.nan,
                'percentile_threshold_c': percentile_90,
                'pixel_count': 0
            })
    
    print(f"  Processed {counties_processed} counties, fixed {counties_fixed} small counties")
    
    if counties_fixed > 0:
        print(f"✅ Fixed {counties_fixed} small counties using nearest neighbor assignment")
    
    return pd.DataFrame(results)


def calculate_low_temperature_days_10th(ds: xr.Dataset, counties: gpd.GeoDataFrame, region: str = "CONUS") -> pd.DataFrame:
    """
    Calculate low temperature days (10th percentile threshold) for each county.
    
    For tasmin data - uses adaptive thresholds based on the 10th percentile.
    
    Args:
        ds (xarray.Dataset): Dataset with daily temperature data (tasmin variable)
        counties (geopandas.GeoDataFrame): County boundaries  
        region (str): Geographic region for coordinate handling
        
    Returns:
        pandas.DataFrame: County-level low temperature days with columns:
            - GEOID: County identifier
            - low_temp_days_10th: Days below 10th percentile per year
            - percentile_threshold_c: The 10th percentile threshold value (°C)
            - pixel_count: Number of grid cells used for calculation
    """
    print("Calculating low temperature days (10th percentile threshold)...")
    
    # Step 1: Daily-to-Annual calculation with percentile threshold
    temp_var_name = _detect_temperature_variable(ds)
    
    # Temperature data is in Kelvin, convert to Celsius
    temp_daily_c = ds[temp_var_name] - 273.15
    
    # Calculate 10th percentile threshold for the entire dataset
    percentile_10 = float(temp_daily_c.quantile(0.10))
    
    # Calculate low temperature days using the 10th percentile threshold
    low_temp_days = (temp_daily_c < percentile_10).sum(dim='dayofyear')
    low_temp_days.attrs['units'] = 'days/year'
    low_temp_days.attrs['long_name'] = f'Low Temperature Days (<{percentile_10:.1f}°C, 10th percentile)'
    
    print(f"10th percentile threshold: {percentile_10:.1f}°C")
    print(f"Low temperature days range: {int(low_temp_days.min())} - {int(low_temp_days.max())} days/year")
    print("Note: Using percentile-based threshold appropriate for climate normals data.")
    
    # Step 2: County-level extraction
    print("Extracting low temperature days by county...")
    
    # Get raster properties
    height, width = low_temp_days.shape
    lat_min, lat_max = float(low_temp_days.lat.min()), float(low_temp_days.lat.max())
    lon_min, lon_max = float(low_temp_days.lon.min()), float(low_temp_days.lon.max())
    
    print(f"  Data grid bounds: {lat_min:.3f}°N to {lat_max:.3f}°N, {lon_min:.3f}° to {lon_max:.3f}°")
    
    # Create transform for rasterization
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    
    # Get coordinate arrays for nearest neighbor lookup
    lats = low_temp_days.lat.values
    lons = low_temp_days.lon.values
    
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
            
            # Apply mask to low temperature days data
            masked_low_temp_days = np.where(mask == 1, low_temp_days.values, np.nan)
            
            # Calculate statistics
            valid_pixels = ~np.isnan(masked_low_temp_days)
            if valid_pixels.sum() > 0:
                # Standard case: county intersects with grid cells
                mean_low_temp_days = np.nanmean(masked_low_temp_days)
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
                        
                        # Get low temperature days value from nearest grid cell
                        nearest_low_temp_days = low_temp_days.isel(
                            lon=nearest_lon_idx,
                            lat=nearest_lat_idx
                        ).values
                        
                        if not np.isnan(nearest_low_temp_days):
                            mean_low_temp_days = float(nearest_low_temp_days)
                            pixel_count = 1
                            counties_fixed += 1
                        else:
                            mean_low_temp_days = np.nan
                            pixel_count = 0
                    else:
                        mean_low_temp_days = np.nan
                        pixel_count = 0
                else:
                    mean_low_temp_days = np.nan
                    pixel_count = 0
            
            results.append({
                'GEOID': geoid,
                'low_temp_days_10th': mean_low_temp_days,
                'percentile_threshold_c': percentile_10,
                'pixel_count': pixel_count
            })
            
        except Exception as e:
            print(f"Warning: Error processing county {geoid}: {e}")
            results.append({
                'GEOID': geoid,
                'low_temp_days_10th': np.nan,
                'percentile_threshold_c': percentile_10,
                'pixel_count': 0
            })
    
    print(f"  Processed {counties_processed} counties, fixed {counties_fixed} small counties")
    
    if counties_fixed > 0:
        print(f"✅ Fixed {counties_fixed} small counties using nearest neighbor assignment")
    
    return pd.DataFrame(results)


def calculate_annual_min_temperature(ds: xr.Dataset, counties: gpd.GeoDataFrame, region: str = "CONUS") -> pd.DataFrame:
    """
    Calculate annual minimum temperature for each county from daily temperature data.
    
    Args:
        ds (xarray.Dataset): Dataset with daily temperature data (tasmin variable)
        counties (geopandas.GeoDataFrame): County boundaries  
        region (str): Geographic region for coordinate handling
        
    Returns:
        pandas.DataFrame: County-level annual minimum temperature with columns:
            - GEOID: County identifier
            - annual_min_temp_c: Annual minimum temperature (°C)
            - pixel_count: Number of grid cells used for calculation
    """
    print("Calculating annual minimum temperature...")
    
    # Step 1: Daily-to-Annual calculation with variable detection
    temp_var_name = _detect_temperature_variable(ds)
    
    # Temperature data is in Kelvin, convert to Celsius
    temp_daily_c = ds[temp_var_name] - 273.15
    
    # Calculate annual minimum temperature
    annual_min_temp = temp_daily_c.min(dim='dayofyear')
    annual_min_temp.attrs['units'] = 'degrees_C'
    annual_min_temp.attrs['long_name'] = 'Annual Minimum Temperature'
    
    print(f"Annual minimum temperature range: {float(annual_min_temp.min()):.1f} - {float(annual_min_temp.max()):.1f} °C")
    
    # Step 2: County-level extraction
    print("Extracting minimum temperature by county...")
    
    # Get raster properties
    height, width = annual_min_temp.shape
    lat_min, lat_max = float(annual_min_temp.lat.min()), float(annual_min_temp.lat.max())
    lon_min, lon_max = float(annual_min_temp.lon.min()), float(annual_min_temp.lon.max())
    
    print(f"  Data grid bounds: {lat_min:.3f}°N to {lat_max:.3f}°N, {lon_min:.3f}° to {lon_max:.3f}°")
    
    # Create transform for rasterization
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    
    # Get coordinate arrays for nearest neighbor lookup
    lats = annual_min_temp.lat.values
    lons = annual_min_temp.lon.values
    
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
            
            # Apply mask to temperature data
            masked_temp = np.where(mask == 1, annual_min_temp.values, np.nan)
            
            # Calculate statistics
            valid_pixels = ~np.isnan(masked_temp)
            if valid_pixels.sum() > 0:
                # Standard case: county intersects with grid cells
                mean_temp = np.nanmean(masked_temp)
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
                        
                        # Get temperature value from nearest grid cell
                        nearest_temp = annual_min_temp.isel(
                            lon=nearest_lon_idx, 
                            lat=nearest_lat_idx
                        ).values
                        
                        if not np.isnan(nearest_temp):
                            mean_temp = float(nearest_temp)
                            pixel_count = 1
                            counties_fixed += 1
                        else:
                            mean_temp = np.nan
                            pixel_count = 0
                    else:
                        mean_temp = np.nan
                        pixel_count = 0
                else:
                    mean_temp = np.nan
                    pixel_count = 0
            
            results.append({
                'GEOID': geoid,
                'annual_min_temp_c': mean_temp,
                'pixel_count': pixel_count
            })
            
        except Exception as e:
            print(f"Warning: Error processing county {geoid}: {e}")
            results.append({
                'GEOID': geoid,
                'annual_min_temp_c': np.nan,
                'pixel_count': 0
            })
    
    print(f"  Processed {counties_processed} counties, fixed {counties_fixed} small counties")
    
    if counties_fixed > 0:
        print(f"✅ Fixed {counties_fixed} small counties using nearest neighbor assignment")
    
    return pd.DataFrame(results)


def calculate_annual_max_temperature(ds: xr.Dataset, counties: gpd.GeoDataFrame, region: str = "CONUS") -> pd.DataFrame:
    """
    Calculate annual maximum temperature for each county from daily temperature data.
    
    Args:
        ds (xarray.Dataset): Dataset with daily temperature data (tasmax variable)
        counties (geopandas.GeoDataFrame): County boundaries  
        region (str): Geographic region for coordinate handling
        
    Returns:
        pandas.DataFrame: County-level annual maximum temperature with columns:
            - GEOID: County identifier
            - annual_max_temp_c: Annual maximum temperature (°C)
            - pixel_count: Number of grid cells used for calculation
    """
    print("Calculating annual maximum temperature...")
    
    # Step 1: Daily-to-Annual calculation with variable detection
    temp_var_name = _detect_temperature_variable(ds)
    
    # Temperature data is in Kelvin, convert to Celsius
    temp_daily_c = ds[temp_var_name] - 273.15
    
    # Calculate annual maximum temperature
    annual_max_temp = temp_daily_c.max(dim='dayofyear')
    annual_max_temp.attrs['units'] = 'degrees_C'
    annual_max_temp.attrs['long_name'] = 'Annual Maximum Temperature'
    
    print(f"Annual maximum temperature range: {float(annual_max_temp.min()):.1f} - {float(annual_max_temp.max()):.1f} °C")
    
    # Step 2: County-level extraction
    print("Extracting maximum temperature by county...")
    
    # Get raster properties
    height, width = annual_max_temp.shape
    lat_min, lat_max = float(annual_max_temp.lat.min()), float(annual_max_temp.lat.max())
    lon_min, lon_max = float(annual_max_temp.lon.min()), float(annual_max_temp.lon.max())
    
    print(f"  Data grid bounds: {lat_min:.3f}°N to {lat_max:.3f}°N, {lon_min:.3f}° to {lon_max:.3f}°")
    
    # Create transform for rasterization
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    
    # Get coordinate arrays for nearest neighbor lookup
    lats = annual_max_temp.lat.values
    lons = annual_max_temp.lon.values
    
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
            
            # Apply mask to temperature data
            masked_temp = np.where(mask == 1, annual_max_temp.values, np.nan)
            
            # Calculate statistics
            valid_pixels = ~np.isnan(masked_temp)
            if valid_pixels.sum() > 0:
                # Standard case: county intersects with grid cells
                mean_temp = np.nanmean(masked_temp)
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
                        
                        # Get temperature value from nearest grid cell
                        nearest_temp = annual_max_temp.isel(
                            lon=nearest_lon_idx, 
                            lat=nearest_lat_idx
                        ).values
                        
                        if not np.isnan(nearest_temp):
                            mean_temp = float(nearest_temp)
                            pixel_count = 1
                            counties_fixed += 1
                        else:
                            mean_temp = np.nan
                            pixel_count = 0
                    else:
                        mean_temp = np.nan
                        pixel_count = 0
                else:
                    mean_temp = np.nan
                    pixel_count = 0
            
            results.append({
                'GEOID': geoid,
                'annual_max_temp_c': mean_temp,
                'pixel_count': pixel_count
            })
            
        except Exception as e:
            print(f"Warning: Error processing county {geoid}: {e}")
            results.append({
                'GEOID': geoid,
                'annual_max_temp_c': np.nan,
                'pixel_count': 0
            })
    
    print(f"  Processed {counties_processed} counties, fixed {counties_fixed} small counties")
    
    if counties_fixed > 0:
        print(f"✅ Fixed {counties_fixed} small counties using nearest neighbor assignment")
    
    return pd.DataFrame(results)


def calculate_temp_days_95th_percentile(ds: xr.Dataset, counties: gpd.GeoDataFrame, region: str = "CONUS") -> pd.DataFrame:
    """
    Calculate temperature days above 95th percentile for each county.
    
    For tasmax data - uses adaptive thresholds based on the 95th percentile.
    
    Args:
        ds (xarray.Dataset): Dataset with daily temperature data (tasmax variable)
        counties (geopandas.GeoDataFrame): County boundaries  
        region (str): Geographic region for coordinate handling
        
    Returns:
        pandas.DataFrame: County-level temperature days with columns:
            - GEOID: County identifier
            - temp_days_95th: Days above 95th percentile per year
            - percentile_threshold_c: The 95th percentile threshold value (°C)
            - pixel_count: Number of grid cells used for calculation
    """
    print("Calculating temperature days above 95th percentile...")
    
    # Step 1: Daily-to-Annual calculation with percentile threshold
    temp_var_name = _detect_temperature_variable(ds)
    
    # Temperature data is in Kelvin, convert to Celsius
    temp_daily_c = ds[temp_var_name] - 273.15
    
    # Calculate 95th percentile threshold for the entire dataset
    percentile_95 = float(temp_daily_c.quantile(0.95))
    
    # Calculate temperature days using the 95th percentile threshold
    temp_days_95th = (temp_daily_c > percentile_95).sum(dim='dayofyear')
    temp_days_95th.attrs['units'] = 'days/year'
    temp_days_95th.attrs['long_name'] = f'Temperature Days (>{percentile_95:.1f}°C, 95th percentile)'
    
    print(f"95th percentile threshold: {percentile_95:.1f}°C")
    print(f"Temperature days range: {int(temp_days_95th.min())} - {int(temp_days_95th.max())} days/year")
    print("Note: Using percentile-based threshold appropriate for climate normals data.")
    
    # Step 2: County-level extraction
    print("Extracting temperature days by county...")
    
    # Get raster properties
    height, width = temp_days_95th.shape
    lat_min, lat_max = float(temp_days_95th.lat.min()), float(temp_days_95th.lat.max())
    lon_min, lon_max = float(temp_days_95th.lon.min()), float(temp_days_95th.lon.max())
    
    print(f"  Data grid bounds: {lat_min:.3f}°N to {lat_max:.3f}°N, {lon_min:.3f}° to {lon_max:.3f}°")
    
    # Create transform for rasterization
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    
    # Get coordinate arrays for nearest neighbor lookup
    lats = temp_days_95th.lat.values
    lons = temp_days_95th.lon.values
    
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
            
            # Apply mask to temperature days data
            masked_temp_days = np.where(mask == 1, temp_days_95th.values, np.nan)
            
            # Calculate statistics
            valid_pixels = ~np.isnan(masked_temp_days)
            if valid_pixels.sum() > 0:
                # Standard case: county intersects with grid cells
                mean_temp_days = np.nanmean(masked_temp_days)
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
                        
                        # Get temperature days value from nearest grid cell
                        nearest_temp_days = temp_days_95th.isel(
                            lon=nearest_lon_idx,
                            lat=nearest_lat_idx
                        ).values
                        
                        if not np.isnan(nearest_temp_days):
                            mean_temp_days = float(nearest_temp_days)
                            pixel_count = 1
                            counties_fixed += 1
                        else:
                            mean_temp_days = np.nan
                            pixel_count = 0
                    else:
                        mean_temp_days = np.nan
                        pixel_count = 0
                else:
                    mean_temp_days = np.nan
                    pixel_count = 0
            
            results.append({
                'GEOID': geoid,
                'temp_days_95th': mean_temp_days,
                'percentile_threshold_c': percentile_95,
                'pixel_count': pixel_count
            })
            
        except Exception as e:
            print(f"Warning: Error processing county {geoid}: {e}")
            results.append({
                'GEOID': geoid,
                'temp_days_95th': np.nan,
                'percentile_threshold_c': percentile_95,
                'pixel_count': 0
            })
    
    print(f"  Processed {counties_processed} counties, fixed {counties_fixed} small counties")
    
    if counties_fixed > 0:
        print(f"✅ Fixed {counties_fixed} small counties using nearest neighbor assignment")
    
    return pd.DataFrame(results)


def calculate_temp_days_5th_percentile(ds: xr.Dataset, counties: gpd.GeoDataFrame, region: str = "CONUS") -> pd.DataFrame:
    """
    Calculate temperature days below 5th percentile for each county.
    
    For tasmin data - uses adaptive thresholds based on the 5th percentile.
    
    Args:
        ds (xarray.Dataset): Dataset with daily temperature data (tasmin variable)
        counties (geopandas.GeoDataFrame): County boundaries  
        region (str): Geographic region for coordinate handling
        
    Returns:
        pandas.DataFrame: County-level temperature days with columns:
            - GEOID: County identifier
            - temp_days_5th: Days below 5th percentile per year
            - percentile_threshold_c: The 5th percentile threshold value (°C)
            - pixel_count: Number of grid cells used for calculation
    """
    print("Calculating temperature days below 5th percentile...")
    
    # Step 1: Daily-to-Annual calculation with percentile threshold
    temp_var_name = _detect_temperature_variable(ds)
    
    # Temperature data is in Kelvin, convert to Celsius
    temp_daily_c = ds[temp_var_name] - 273.15
    
    # Calculate 5th percentile threshold for the entire dataset
    percentile_5 = float(temp_daily_c.quantile(0.05))
    
    # Calculate temperature days using the 5th percentile threshold
    temp_days_5th = (temp_daily_c < percentile_5).sum(dim='dayofyear')
    temp_days_5th.attrs['units'] = 'days/year'
    temp_days_5th.attrs['long_name'] = f'Temperature Days (<{percentile_5:.1f}°C, 5th percentile)'
    
    print(f"5th percentile threshold: {percentile_5:.1f}°C")
    print(f"Temperature days range: {int(temp_days_5th.min())} - {int(temp_days_5th.max())} days/year")
    print("Note: Using percentile-based threshold appropriate for climate normals data.")
    
    # Step 2: County-level extraction
    print("Extracting temperature days by county...")
    
    # Get raster properties
    height, width = temp_days_5th.shape
    lat_min, lat_max = float(temp_days_5th.lat.min()), float(temp_days_5th.lat.max())
    lon_min, lon_max = float(temp_days_5th.lon.min()), float(temp_days_5th.lon.max())
    
    print(f"  Data grid bounds: {lat_min:.3f}°N to {lat_max:.3f}°N, {lon_min:.3f}° to {lon_max:.3f}°")
    
    # Create transform for rasterization
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    
    # Get coordinate arrays for nearest neighbor lookup
    lats = temp_days_5th.lat.values
    lons = temp_days_5th.lon.values
    
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
            
            # Apply mask to temperature days data
            masked_temp_days = np.where(mask == 1, temp_days_5th.values, np.nan)
            
            # Calculate statistics
            valid_pixels = ~np.isnan(masked_temp_days)
            if valid_pixels.sum() > 0:
                # Standard case: county intersects with grid cells
                mean_temp_days = np.nanmean(masked_temp_days)
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
                        
                        # Get temperature days value from nearest grid cell
                        nearest_temp_days = temp_days_5th.isel(
                            lon=nearest_lon_idx,
                            lat=nearest_lat_idx
                        ).values
                        
                        if not np.isnan(nearest_temp_days):
                            mean_temp_days = float(nearest_temp_days)
                            pixel_count = 1
                            counties_fixed += 1
                        else:
                            mean_temp_days = np.nan
                            pixel_count = 0
                    else:
                        mean_temp_days = np.nan
                        pixel_count = 0
                else:
                    mean_temp_days = np.nan
                    pixel_count = 0
            
            results.append({
                'GEOID': geoid,
                'temp_days_5th': mean_temp_days,
                'percentile_threshold_c': percentile_5,
                'pixel_count': pixel_count
            })
            
        except Exception as e:
            print(f"Warning: Error processing county {geoid}: {e}")
            results.append({
                'GEOID': geoid,
                'temp_days_5th': np.nan,
                'percentile_threshold_c': percentile_5,
                'pixel_count': 0
            })
    
    print(f"  Processed {counties_processed} counties, fixed {counties_fixed} small counties")
    
    if counties_fixed > 0:
        print(f"✅ Fixed {counties_fixed} small counties using nearest neighbor assignment")
    
    return pd.DataFrame(results)


def calculate_temp_days_1st_percentile(ds: xr.Dataset, counties: gpd.GeoDataFrame, region: str = "CONUS") -> pd.DataFrame:
    """
    Calculate temperature days below 1st percentile for each county.
    
    For tasmin data - uses adaptive thresholds based on the 1st percentile.
    
    Args:
        ds (xarray.Dataset): Dataset with daily temperature data (tasmin variable)
        counties (geopandas.GeoDataFrame): County boundaries  
        region (str): Geographic region for coordinate handling
        
    Returns:
        pandas.DataFrame: County-level temperature days with columns:
            - GEOID: County identifier
            - temp_days_1st: Days below 1st percentile per year
            - percentile_threshold_c: The 1st percentile threshold value (°C)
            - pixel_count: Number of grid cells used for calculation
    """
    print("Calculating temperature days below 1st percentile...")
    
    # Step 1: Daily-to-Annual calculation with percentile threshold
    temp_var_name = _detect_temperature_variable(ds)
    
    # Temperature data is in Kelvin, convert to Celsius
    temp_daily_c = ds[temp_var_name] - 273.15
    
    # Calculate 1st percentile threshold for the entire dataset
    percentile_1 = float(temp_daily_c.quantile(0.01))
    
    # Calculate temperature days using the 1st percentile threshold
    temp_days_1st = (temp_daily_c < percentile_1).sum(dim='dayofyear')
    temp_days_1st.attrs['units'] = 'days/year'
    temp_days_1st.attrs['long_name'] = f'Temperature Days (<{percentile_1:.1f}°C, 1st percentile)'
    
    print(f"1st percentile threshold: {percentile_1:.1f}°C")
    print(f"Temperature days range: {int(temp_days_1st.min())} - {int(temp_days_1st.max())} days/year")
    print("Note: Using percentile-based threshold appropriate for climate normals data.")
    
    # Step 2: County-level extraction
    print("Extracting temperature days by county...")
    
    # Get raster properties
    height, width = temp_days_1st.shape
    lat_min, lat_max = float(temp_days_1st.lat.min()), float(temp_days_1st.lat.max())
    lon_min, lon_max = float(temp_days_1st.lon.min()), float(temp_days_1st.lon.max())
    
    print(f"  Data grid bounds: {lat_min:.3f}°N to {lat_max:.3f}°N, {lon_min:.3f}° to {lon_max:.3f}°")
    
    # Create transform for rasterization
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    
    # Get coordinate arrays for nearest neighbor lookup
    lats = temp_days_1st.lat.values
    lons = temp_days_1st.lon.values
    
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
            
            # Apply mask to temperature days data
            masked_temp_days = np.where(mask == 1, temp_days_1st.values, np.nan)
            
            # Calculate statistics
            valid_pixels = ~np.isnan(masked_temp_days)
            if valid_pixels.sum() > 0:
                # Standard case: county intersects with grid cells
                mean_temp_days = np.nanmean(masked_temp_days)
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
                        
                        # Get temperature days value from nearest grid cell
                        nearest_temp_days = temp_days_1st.isel(
                            lon=nearest_lon_idx,
                            lat=nearest_lat_idx
                        ).values
                        
                        if not np.isnan(nearest_temp_days):
                            mean_temp_days = float(nearest_temp_days)
                            pixel_count = 1
                            counties_fixed += 1
                        else:
                            mean_temp_days = np.nan
                            pixel_count = 0
                    else:
                        mean_temp_days = np.nan
                        pixel_count = 0
                else:
                    mean_temp_days = np.nan
                    pixel_count = 0
            
            results.append({
                'GEOID': geoid,
                'temp_days_1st': mean_temp_days,
                'percentile_threshold_c': percentile_1,
                'pixel_count': pixel_count
            })
            
        except Exception as e:
            print(f"Warning: Error processing county {geoid}: {e}")
            results.append({
                'GEOID': geoid,
                'temp_days_1st': np.nan,
                'percentile_threshold_c': percentile_1,
                'pixel_count': 0
            })
    
    print(f"  Processed {counties_processed} counties, fixed {counties_fixed} small counties")
    
    if counties_fixed > 0:
        print(f"✅ Fixed {counties_fixed} small counties using nearest neighbor assignment")
    
    return pd.DataFrame(results)


# Legacy process functions for backwards compatibility (will be refactored later)
def process_tas(scenario: str = "historical", 
                region: str = "CONUS",
                output_dir: str = "output",
                include_extremes: bool = True,
                counties_shapefile: Optional[str] = None) -> str:
    """
    Process mean temperature (tas) data for a specific scenario and region.
    
    Args:
        scenario (str): Climate scenario ("historical", "hybrid", "ssp245")
        region (str): Geographic region ("CONUS", "AK", "HI", "PRVI", "GU")
        output_dir (str): Directory for output files
        include_extremes (bool): Whether to include extreme temperature metrics
        counties_shapefile (str, optional): Path to custom county shapefile
        
    Returns:
        str: Path to the output CSV file
    """
    print(f"=== PROCESSING MEAN TEMPERATURE (tas) ===")
    print(f"Scenario: {scenario}")
    print(f"Region: {region}")
    print(f"Include extremes: {include_extremes}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    if include_extremes:
        output_file = os.path.join(
            output_dir, 
            f"county_tas_extremes_timeseries_{region}_{scenario}_1980_2100.csv"
        )
    else:
        output_file = os.path.join(
            output_dir,
            f"county_tas_basic_timeseries_{region}_{scenario}_1980_2100.csv"
        )
    
    # TODO: Connect to batch processing from your existing modules
    print(f"TODO: Implement full time series processing for tas")
    print(f"Would save to: {output_file}")
    
    return output_file


def process_tasmin(scenario: str = "historical", 
                   region: str = "CONUS",
                   output_dir: str = "output",
                   counties_shapefile: Optional[str] = None) -> str:
    """
    Process minimum temperature (tasmin) data with cold extremes.
    
    Args:
        scenario (str): Climate scenario ("historical", "hybrid", "ssp245")
        region (str): Geographic region ("CONUS", "AK", "HI", "PRVI", "GU")
        output_dir (str): Directory for output files
        counties_shapefile (str, optional): Path to custom county shapefile
        
    Returns:
        str: Path to the output CSV file
    """
    print(f"=== PROCESSING MINIMUM TEMPERATURE (tasmin) ===")
    print(f"Scenario: {scenario}")
    print(f"Region: {region}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    output_file = os.path.join(
        output_dir, 
        f"county_tasmin_cold_extremes_timeseries_{region}_{scenario}_1980_2100.csv"
    )
    
    # TODO: Connect to your county_temperature_cold_days module
    print(f"TODO: Implement full time series processing for tasmin")
    print(f"Would save to: {output_file}")
    
    return output_file


def process_tasmax(scenario: str = "historical", 
                   region: str = "CONUS",
                   output_dir: str = "output",
                   counties_shapefile: Optional[str] = None) -> str:
    """
    Process maximum temperature (tasmax) data with hot extremes.
    
    Args:
        scenario (str): Climate scenario ("historical", "hybrid", "ssp245")
        region (str): Geographic region ("CONUS", "AK", "HI", "PRVI", "GU")
        output_dir (str): Directory for output files
        counties_shapefile (str, optional): Path to custom county shapefile
        
    Returns:
        str: Path to the output CSV file
    """
    print(f"=== PROCESSING MAXIMUM TEMPERATURE (tasmax) ===")
    print(f"Scenario: {scenario}")
    print(f"Region: {region}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    output_file = os.path.join(
        output_dir, 
        f"county_tasmax_hot_extremes_timeseries_{region}_{scenario}_1980_2100.csv"
    )
    
    # TODO: Connect to your county_temperature_hot_days module
    print(f"TODO: Implement full time series processing for tasmax")
    print(f"Would save to: {output_file}")
    
    return output_file 