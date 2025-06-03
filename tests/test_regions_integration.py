#!/usr/bin/env python3
"""
Integration tests for climate_means.py and regions.py modules.

Tests the interaction between the main climate processing module and the regional
operations module, ensuring that regional functionality works correctly with
climate data processing workflows.
"""

import pytest
import tempfile
import numpy as np
import xarray as xr
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import modules to test
import climate_means
import regions
from regions import REGION_BOUNDS, extract_region, validate_region_bounds


class TestRegionsModuleIntegration:
    """Test integration between regions and climate_means modules."""
    
    def test_regions_imports_successful(self):
        """Test that all required regional imports work correctly."""
        # Test that climate_means can import from regions
        assert hasattr(climate_means, 'REGION_BOUNDS')
        assert hasattr(climate_means, 'extract_region')
        assert hasattr(climate_means, 'validate_region_bounds')
        
        # Test that the imported functions are the same objects
        assert climate_means.REGION_BOUNDS is regions.REGION_BOUNDS
        assert climate_means.extract_region is regions.extract_region
        assert climate_means.validate_region_bounds is regions.validate_region_bounds
    
    def test_region_bounds_structure(self):
        """Test that REGION_BOUNDS has expected structure and regions."""
        # Test that all expected regions exist
        expected_regions = {'CONUS', 'AK', 'HI', 'PRVI', 'GU'}
        assert set(climate_means.REGION_BOUNDS.keys()) == expected_regions
        
        # Test structure of each region
        for region_key, region_info in climate_means.REGION_BOUNDS.items():
            assert 'name' in region_info
            assert 'lon_min' in region_info
            assert 'lon_max' in region_info
            assert 'lat_min' in region_info
            assert 'lat_max' in region_info
            assert isinstance(region_info['name'], str)
            assert isinstance(region_info['lon_min'], (int, float))
            assert isinstance(region_info['lon_max'], (int, float))
            assert isinstance(region_info['lat_min'], (int, float))
            assert isinstance(region_info['lat_max'], (int, float))


class TestRegionalDataOperations:
    """Test regional operations with climate datasets."""
    
    @pytest.fixture
    def sample_global_dataset_0_360(self):
        """Create a sample global dataset with 0-360 longitude system."""
        # Create global grid
        lon = np.linspace(0, 359, 360)  # 0-360 longitude
        lat = np.linspace(-90, 90, 181)  # -90 to 90 latitude
        time = np.arange(0, 10)  # 10 time steps
        
        # Create temperature data
        temp_data = 273.15 + 20 * np.random.random((10, 181, 360))
        
        ds = xr.Dataset({
            'tas': (['time', 'lat', 'lon'], temp_data)
        }, coords={
            'time': time,
            'lat': lat,
            'lon': lon
        })
        
        ds.tas.attrs['units'] = 'K'
        ds.tas.attrs['long_name'] = 'Near-Surface Air Temperature'
        
        return ds
    
    @pytest.fixture
    def sample_global_dataset_minus180_180(self):
        """Create a sample global dataset with -180 to 180 longitude system."""
        # Create global grid
        lon = np.linspace(-180, 179, 360)  # -180 to 180 longitude
        lat = np.linspace(-90, 90, 181)  # -90 to 90 latitude
        time = np.arange(0, 10)  # 10 time steps
        
        # Create temperature data
        temp_data = 273.15 + 20 * np.random.random((10, 181, 360))
        
        ds = xr.Dataset({
            'tas': (['time', 'lat', 'lon'], temp_data)
        }, coords={
            'time': time,
            'lat': lat,
            'lon': lon
        })
        
        ds.tas.attrs['units'] = 'K'
        ds.tas.attrs['long_name'] = 'Near-Surface Air Temperature'
        
        return ds
    
    def test_extract_region_conus_0_360(self, sample_global_dataset_0_360):
        """Test CONUS region extraction from 0-360 longitude dataset."""
        # Extract CONUS region using climate_means module
        conus_bounds = climate_means.REGION_BOUNDS['CONUS']
        conus_ds = climate_means.extract_region(sample_global_dataset_0_360, conus_bounds)
        
        # Verify extraction worked
        assert conus_ds is not None
        assert 'tas' in conus_ds.data_vars
        assert conus_ds.lon.size > 0
        assert conus_ds.lat.size > 0
        
        # Check that coordinates are within expected bounds
        # Note: extract_region should handle coordinate conversion automatically
        lon_min = float(conus_ds.lon.min())
        lon_max = float(conus_ds.lon.max())
        lat_min = float(conus_ds.lat.min())
        lat_max = float(conus_ds.lat.max())
        
        # Should be within reasonable CONUS bounds
        assert lat_min >= 20  # Roughly southern US
        assert lat_max <= 55  # Roughly northern US
    
    def test_extract_region_conus_minus180_180(self, sample_global_dataset_minus180_180):
        """Test CONUS region extraction from -180/180 longitude dataset."""
        # Extract CONUS region using climate_means module
        conus_bounds = climate_means.REGION_BOUNDS['CONUS']
        conus_ds = climate_means.extract_region(sample_global_dataset_minus180_180, conus_bounds)
        
        # Verify extraction worked
        assert conus_ds is not None
        assert 'tas' in conus_ds.data_vars
        assert conus_ds.lon.size > 0
        assert conus_ds.lat.size > 0
        
        # Check longitude bounds for -180/180 system
        lon_min = float(conus_ds.lon.min())
        lon_max = float(conus_ds.lon.max())
        
        # Should be negative values for CONUS in -180/180 system
        assert lon_min < 0
        assert lon_max < 0
        assert lon_min >= -130  # Roughly western US
        assert lon_max <= -65   # Roughly eastern US
    
    def test_extract_all_regions(self, sample_global_dataset_0_360):
        """Test extraction of all defined regions."""
        for region_key in climate_means.REGION_BOUNDS.keys():
            region_bounds = climate_means.REGION_BOUNDS[region_key]
            
            # Skip if region validation fails
            if not climate_means.validate_region_bounds(region_key):
                continue
            
            # Extract region
            region_ds = climate_means.extract_region(sample_global_dataset_0_360, region_bounds)
            
            # Basic checks
            assert region_ds is not None, f"Failed to extract region {region_key}"
            assert 'tas' in region_ds.data_vars, f"Missing 'tas' variable in {region_key}"
            
            # Check that we have some data (not all NaN)
            if region_ds.lon.size > 0 and region_ds.lat.size > 0:
                assert not region_ds.tas.isnull().all().compute(), f"All data is NaN for {region_key}"


class TestRegionalValidation:
    """Test regional validation functions."""
    
    def test_validate_region_bounds_all_regions(self):
        """Test that all predefined regions pass validation."""
        for region_key in climate_means.REGION_BOUNDS.keys():
            assert climate_means.validate_region_bounds(region_key), f"Region {region_key} failed validation"
    
    def test_validate_region_bounds_invalid_region(self):
        """Test validation with invalid region key."""
        assert not climate_means.validate_region_bounds('INVALID_REGION')
        assert not climate_means.validate_region_bounds('NONEXISTENT')
    
