#!/usr/bin/env python3
"""
Test script to debug coordinate issue in CONUS processing pipeline
"""

import xarray as xr
from means.core.regions import REGION_BOUNDS, extract_region
from means.utils.time_util import handle_time_coordinates

def test_coordinate_preservation():
    """Test coordinate preservation through the processing pipeline."""
    
    # Load a file and process it like the pipeline does
    file_path = '/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM/pr/historical/pr_day_NorESM2-LM_historical_r1i1p1f1_gn_1950_v1.1.nc'
    ds = xr.open_dataset(file_path, decode_times=False, cache=False)

    print('Step 1 - Original dataset coordinates:')
    print(f'Lon: {ds.lon.values[:3]}')
    print(f'Lat: {ds.lat.values[:3]}')

    # Handle time coordinates
    ds, time_method = handle_time_coordinates(ds, file_path)

    print('Step 2 - After time handling:')
    print(f'Lon: {ds.lon.values[:3]}')
    print(f'Lat: {ds.lat.values[:3]}')

    # Extract region
    region_ds = extract_region(ds, REGION_BOUNDS['CONUS'])
    var = region_ds['pr']

    print('Step 3 - After region extraction:')
    print(f'Lon: {var.lon.values[:3]}')
    print(f'Lat: {var.lat.values[:3]}')

    # Calculate daily climatology
    if 'dayofyear' in var.coords:
        daily_clim = var.groupby(var.dayofyear).mean(dim='time')
        result = daily_clim.compute()
        
        print('Step 4 - After daily climatology:')
        print(f'Lon: {result.lon.values[:3]}')
        print(f'Lat: {result.lat.values[:3]}')
        print(f'Coordinates: {list(result.coords)}')
        print(f'Lon dtype: {result.lon.dtype}')
        print(f'Lat dtype: {result.lat.dtype}')
        
        # Test what happens when we concat multiple of these
        print('\nStep 5 - Testing concat operation:')
        stacked_data = xr.concat([result, result], dim='year')
        mean_data = stacked_data.mean(dim='year')
        
        print(f'After concat and mean:')
        print(f'Lon: {mean_data.lon.values[:3]}')
        print(f'Lat: {mean_data.lat.values[:3]}')
        print(f'Lon dtype: {mean_data.lon.dtype}')
        print(f'Lat dtype: {mean_data.lat.dtype}')
        
    else:
        print('No dayofyear coordinate found')

    ds.close()

if __name__ == "__main__":
    test_coordinate_preservation() 