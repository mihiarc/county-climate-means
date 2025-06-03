#!/usr/bin/env python3
"""
Integration tests for time_util.py module.

Tests the time handling utilities and their integration with the main climate
processing modules (climate_means.py and io_util.py), ensuring that time-related
functionality works correctly across the system.
"""

import pytest
import tempfile
import numpy as np
import xarray as xr
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import modules to test
import time_util
import climate_means
import io_util


class TestTimeUtilModuleIntegration:
    """Test integration between time_util and other modules."""
    
    def test_time_util_imports_successful(self):
        """Test that all time utility imports work correctly."""
        # Test that climate_means can import from time_util
        assert hasattr(climate_means, 'generate_climate_periods')
        assert hasattr(climate_means, 'handle_time_coordinates')
        assert hasattr(climate_means, 'reconstruct_time_dataarray')
        assert hasattr(climate_means, 'determine_climatology_type')
        assert hasattr(climate_means, 'add_time_metadata')
        
        # Test that io_util can import from time_util
        assert hasattr(io_util, 'extract_year_from_filename')
        assert hasattr(io_util, 'get_time_decoding_params')
        assert hasattr(io_util, 'try_time_engines')
        
        # Test that the imported functions are the same objects
        assert climate_means.generate_climate_periods is time_util.generate_climate_periods
        assert climate_means.handle_time_coordinates is time_util.handle_time_coordinates
        assert io_util.extract_year_from_filename is time_util.extract_year_from_filename
    
    def test_time_util_functions_accessible(self):
        """Test that all time_util functions are accessible."""
        # Period generation
        assert callable(time_util.generate_climate_periods)
        
        # Time coordinate handling
        assert callable(time_util.handle_time_coordinates)
        
        # Year extraction and filename handling
        assert callable(time_util.extract_year_from_filename)
        assert callable(time_util.sort_files_by_year)
        assert callable(time_util.get_available_years_from_files)
        assert callable(time_util.filter_files_by_year_range)
        
        # Climate normal computations
        assert callable(time_util.reconstruct_time_dataarray)
        assert callable(time_util.determine_climatology_type)
        assert callable(time_util.add_time_metadata)
        
        # Time decoding utilities
        assert callable(time_util.get_time_decoding_params)
        assert callable(time_util.try_time_engines)
        
        # Utility functions
        assert callable(time_util.validate_year_range)
        assert callable(time_util.get_years_in_range)
        assert callable(time_util.is_leap_year)
        assert callable(time_util.get_days_in_year)
        assert callable(time_util.standardize_to_365_days)


class TestPeriodGeneration:
    """Test climate period generation functionality."""
    
    def test_generate_climate_periods_historical(self):
        """Test generation of historical climate periods."""
        data_availability = {
            'historical': {'start': 1950, 'end': 2014}
        }
        
        periods = time_util.generate_climate_periods('historical', data_availability)
        
        # Should generate periods for 1980-2014 (35 periods)
        assert len(periods) == 35
        
        # Check first period (ending in 1980)
        first_period = periods[0]
        assert first_period[0] == 1951  # start_year (1980 - 29)
        assert first_period[1] == 1980  # end_year
        assert first_period[2] == 1980  # target_year
        assert first_period[3] == "historical_1980"  # period_name
        
        # Check last period (ending in 2014)
        last_period = periods[-1]
        assert last_period[0] == 1985  # start_year (2014 - 29)
        assert last_period[1] == 2014  # end_year
        assert last_period[2] == 2014  # target_year
        assert last_period[3] == "historical_2014"  # period_name
    
    def test_generate_climate_periods_future_scenario(self):
        """Test generation of future scenario climate periods."""
        data_availability = {
            'ssp245': {'start': 2015, 'end': 2100}
        }
        
        periods = time_util.generate_climate_periods('ssp245', data_availability)
        
        # Should generate periods for 2015-2100 (86 periods)
        assert len(periods) == 86
        
        # Check first period (ending in 2015)
        first_period = periods[0]
        assert first_period[0] == 1986  # start_year (2015 - 29)
        assert first_period[1] == 2015  # end_year
        assert first_period[2] == 2015  # target_year
        assert first_period[3] == "ssp245_2015"  # period_name
    
    def test_generate_climate_periods_insufficient_data(self):
        """Test period generation with insufficient data."""
        data_availability = {
            'historical': {'start': 1990, 'end': 2014}  # Not enough for 30-year periods starting from 1980
        }
        
        periods = time_util.generate_climate_periods('historical', data_availability)
        
        # Should only generate periods where we have 30 years of data
        # First valid period would end in 2019 (1990 + 29), but data ends in 2014
        # So we get periods ending from 1990+29=2019 down to 2014, but limited by data_end
        # Actually, it should generate periods where start_year >= data_start
        # For target_year 2014: start_year = 2014 - 29 = 1985 < 1990, so no periods
        # For target_year 2013: start_year = 2013 - 29 = 1984 < 1990, so no periods
        # Continue until we find start_year >= 1990
        # target_year where 1990 <= target_year - 29 means target_year >= 2019
        # But data_end is 2014, so no valid periods
        assert len(periods) == 0
    
    def test_generate_climate_periods_integration_with_climate_means(self):
        """Test that climate_means can use generate_climate_periods."""
        data_availability = {
            'historical': {'start': 1950, 'end': 2014}
        }
        
        # Call through climate_means module
        periods = climate_means.generate_climate_periods('historical', data_availability)
        
        assert len(periods) > 0
        assert all(len(period) == 4 for period in periods)  # Each period should have 4 elements


class TestTimeCoordinateHandling:
    """Test time coordinate handling functionality."""
    
    @pytest.fixture
    def sample_daily_dataset(self):
        """Create a sample daily dataset for testing."""
        time = np.arange(0, 365)  # One year of daily data
        lat = np.linspace(25, 50, 10)
        lon = np.linspace(-125, -65, 15)
        
        temp_data = 273.15 + 15 + 10 * np.sin(2 * np.pi * time[:, None, None] / 365) + \
                   np.random.random((365, 10, 15)) * 2
        
        ds = xr.Dataset({
            'tas': (['time', 'lat', 'lon'], temp_data)
        }, coords={
            'time': time,
            'lat': lat,
            'lon': lon
        })
        
        return ds
    
    @pytest.fixture
    def sample_leap_year_dataset(self):
        """Create a sample leap year dataset (366 days)."""
        time = np.arange(0, 366)  # Leap year
        lat = np.linspace(25, 50, 10)
        lon = np.linspace(-125, -65, 15)
        
        temp_data = 273.15 + 15 + 10 * np.sin(2 * np.pi * time[:, None, None] / 366) + \
                   np.random.random((366, 10, 15)) * 2
        
        ds = xr.Dataset({
            'tas': (['time', 'lat', 'lon'], temp_data)
        }, coords={
            'time': time,
            'lat': lat,
            'lon': lon
        })
        
        return ds
    
    def test_handle_time_coordinates_daily_data(self, sample_daily_dataset):
        """Test handling time coordinates for daily data."""
        ds_with_dayofyear, time_method = time_util.handle_time_coordinates(
            sample_daily_dataset, "test_file.nc"
        )
        
        assert time_method == 'daily'
        assert 'dayofyear' in ds_with_dayofyear.coords
        assert len(ds_with_dayofyear.dayofyear) == 365
        assert ds_with_dayofyear.dayofyear.min() == 1
        assert ds_with_dayofyear.dayofyear.max() == 365
    
    def test_handle_time_coordinates_leap_year(self, sample_leap_year_dataset):
        """Test handling time coordinates for leap year data."""
        ds_with_dayofyear, time_method = time_util.handle_time_coordinates(
            sample_leap_year_dataset, "test_leap_year.nc"
        )
        
        assert time_method == 'daily'
        assert 'dayofyear' in ds_with_dayofyear.coords
        # Should be clipped to 365 days
        assert len(ds_with_dayofyear.dayofyear) == 366  # Original length preserved
        assert ds_with_dayofyear.dayofyear.min() == 1
        assert ds_with_dayofyear.dayofyear.max() == 365  # But values clipped to 365
    
    def test_handle_time_coordinates_no_time_dimension(self):
        """Test handling dataset without time coordinate."""
        ds_no_time = xr.Dataset({
            'tas': (['lat', 'lon'], np.random.random((10, 15)))
        }, coords={
            'lat': np.linspace(25, 50, 10),
            'lon': np.linspace(-125, -65, 15)
        })
        
        ds_result, time_method = time_util.handle_time_coordinates(ds_no_time, "no_time.nc")
        
        assert time_method == 'none'
        assert 'dayofyear' not in ds_result.coords
        assert ds_result is ds_no_time  # Should return original dataset unchanged


class TestFilenameHandling:
    """Test filename and year extraction functionality."""
    
    def test_extract_year_from_filename_standard_format(self):
        """Test year extraction from standard NorESM2-LM filename."""
        filename = "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2010.nc"
        year = time_util.extract_year_from_filename(filename)
        assert year == 2010
    
    def test_extract_year_from_filename_with_path(self):
        """Test year extraction from full file path."""
        file_path = "/data/NorESM2-LM/tas/historical/tas_day_NorESM2-LM_historical_r1i1p1f1_gn_1995.nc"
        year = time_util.extract_year_from_filename(file_path)
        assert year == 1995
    
    def test_extract_year_from_filename_invalid_format(self):
        """Test year extraction from invalid filename."""
        invalid_filename = "invalid_file_name.nc"
        year = time_util.extract_year_from_filename(invalid_filename)
        assert year is None
    
    def test_sort_files_by_year(self):
        """Test sorting files by extracted year."""
        files = [
            "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2012.nc",
            "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2010.nc",
            "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2011.nc"
        ]
        
        sorted_files = time_util.sort_files_by_year(files)
        
        expected_years = [2010, 2011, 2012]
        actual_years = [time_util.extract_year_from_filename(f) for f in sorted_files]
        assert actual_years == expected_years
    
    def test_get_available_years_from_files(self):
        """Test getting year range from file list."""
        files = [
            "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_1990.nc",
            "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2000.nc",
            "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2010.nc"
        ]
        
        start_year, end_year = time_util.get_available_years_from_files(files)
        
        assert start_year == 1990
        assert end_year == 2010
    
    def test_get_available_years_from_empty_files(self):
        """Test getting year range from empty file list."""
        start_year, end_year = time_util.get_available_years_from_files([])
        assert start_year == 0
        assert end_year == 0
    
    def test_filter_files_by_year_range(self):
        """Test filtering files by year range."""
        files = [
            "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_1990.nc",
            "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2000.nc",
            "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2010.nc",
            "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2020.nc"
        ]
        
        filtered_files = time_util.filter_files_by_year_range(files, 1995, 2015)
        
        # Should include 2000 and 2010, exclude 1990 and 2020
        assert len(filtered_files) == 2
        years = [time_util.extract_year_from_filename(f) for f in filtered_files]
        assert 2000 in years
        assert 2010 in years
    
    def test_filename_handling_integration_with_io_util(self):
        """Test that io_util can use filename handling functions."""
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "NorESM2-LM"
            tas_hist_dir = data_dir / "tas" / "historical"
            tas_hist_dir.mkdir(parents=True)
            
            # Create dummy files
            for year in [2010, 2011, 2012]:
                filename = f"tas_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}.nc"
                (tas_hist_dir / filename).touch()
            
            # Test through NorESM2FileHandler
            handler = io_util.NorESM2FileHandler(str(data_dir))
            files = handler.get_files_for_period('tas', 'historical', 2010, 2012)
            
            assert len(files) == 3
            # Files should be sorted by year
            years = [handler.extract_year_from_filename(f) for f in files]
            assert years == [2010, 2011, 2012]


class TestClimateNormalComputations:
    """Test climate normal computation and time handling."""
    
    def test_reconstruct_time_dataarray_daily_1d(self):
        """Test reconstructing 1D daily climatology DataArray."""
        # Create 365-day daily climatology data
        daily_data = np.random.random(365)
        batch_year = 2010
        
        da = time_util.reconstruct_time_dataarray(daily_data, batch_year)
        
        assert isinstance(da, xr.DataArray)
        assert 'dayofyear' in da.coords
        assert 'year' in da.coords
        assert len(da.dayofyear) == 365
        assert da.coords['year'] == batch_year
        assert da.dims == ('dayofyear',)
    
    def test_reconstruct_time_dataarray_seasonal_1d(self):
        """Test reconstructing 1D seasonal climatology DataArray."""
        # Create 4-season seasonal data
        seasonal_data = np.random.random(4)
        batch_year = 2010
        
        da = time_util.reconstruct_time_dataarray(seasonal_data, batch_year)
        
        assert isinstance(da, xr.DataArray)
        assert 'season' in da.coords
        assert 'year' in da.coords
        assert len(da.season) == 4
        assert da.coords['year'] == batch_year
        assert da.dims == ('season',)
    
    def test_reconstruct_time_dataarray_daily_3d(self):
        """Test reconstructing 3D daily climatology DataArray."""
        # Create 3D daily climatology data (365, lat, lon)
        daily_3d_data = np.random.random((365, 20, 30))
        batch_year = 2010
        
        da = time_util.reconstruct_time_dataarray(daily_3d_data, batch_year)
        
        assert isinstance(da, xr.DataArray)
        assert 'dayofyear' in da.coords
        assert 'lat' in da.coords
        assert 'lon' in da.coords
        assert 'year' in da.coords
        assert da.shape == (365, 20, 30)
        assert da.coords['year'] == batch_year
        assert da.dims == ('dayofyear', 'lat', 'lon')
    
    def test_reconstruct_time_dataarray_overall_mean_2d(self):
        """Test reconstructing 2D overall mean DataArray."""
        # Create 2D overall mean data (lat, lon)
        mean_2d_data = np.random.random((20, 30))
        batch_year = 2010
        
        da = time_util.reconstruct_time_dataarray(mean_2d_data, batch_year)
        
        assert isinstance(da, xr.DataArray)
        assert 'lat' in da.coords
        assert 'lon' in da.coords
        assert 'year' in da.coords
        assert da.shape == (20, 30)
        assert da.coords['year'] == batch_year
        assert da.dims == ('lat', 'lon')
    
    def test_determine_climatology_type_daily(self):
        """Test determining climatology type for daily data."""
        daily_da = xr.DataArray(
            np.random.random(365),
            coords={'dayofyear': np.arange(1, 366)},
            dims=['dayofyear']
        )
        
        climatology_type = time_util.determine_climatology_type(daily_da)
        assert climatology_type == "daily (365 days)"
    
    def test_determine_climatology_type_seasonal(self):
        """Test determining climatology type for seasonal data."""
        seasonal_da = xr.DataArray(
            np.random.random(4),
            coords={'season': np.arange(4)},
            dims=['season']
        )
        
        climatology_type = time_util.determine_climatology_type(seasonal_da)
        assert climatology_type == "seasonal (4 seasons)"
    
    def test_determine_climatology_type_overall_mean(self):
        """Test determining climatology type for overall mean."""
        mean_da = xr.DataArray(
            np.random.random((20, 30)),
            coords={'lat': np.arange(20), 'lon': np.arange(30)},
            dims=['lat', 'lon']
        )
        
        climatology_type = time_util.determine_climatology_type(mean_da)
        assert climatology_type == "overall mean"
    
    def test_add_time_metadata(self):
        """Test adding time-related metadata to climate normal."""
        # Create a daily climatology DataArray
        result = xr.DataArray(
            np.random.random(365),
            coords={'dayofyear': np.arange(1, 366)},
            dims=['dayofyear']
        )
        
        years = [2000, 2001, 2002, 2003, 2004]
        target_year = 2002
        
        result_with_metadata = time_util.add_time_metadata(result, years, target_year)
        
        # Check that metadata was added
        assert 'long_name' in result_with_metadata.attrs
        assert 'description' in result_with_metadata.attrs
        assert 'target_year' in result_with_metadata.attrs
        assert 'source_years' in result_with_metadata.attrs
        assert 'number_of_years' in result_with_metadata.attrs
        assert 'climatology_type' in result_with_metadata.attrs
        
        assert result_with_metadata.attrs['target_year'] == target_year
        assert result_with_metadata.attrs['number_of_years'] == len(years)
        assert "daily" in result_with_metadata.attrs['climatology_type']


class TestTimeDecodingUtilities:
    """Test time decoding and dataset opening utilities."""
    
    def test_get_time_decoding_params_safe_mode(self):
        """Test getting time decoding parameters in safe mode."""
        params = time_util.get_time_decoding_params(safe_mode=True)
        
        assert isinstance(params, dict)
        assert 'decode_times' in params
        assert 'use_cftime' in params
        assert params['decode_times'] is False
        assert params['use_cftime'] is False
    
    def test_get_time_decoding_params_normal_mode(self):
        """Test getting time decoding parameters in normal mode."""
        params = time_util.get_time_decoding_params(safe_mode=False)
        
        assert isinstance(params, dict)
        assert 'decode_times' in params
        assert 'use_cftime' in params
        assert params['decode_times'] is True
        assert params['use_cftime'] is True
    
    def test_try_time_engines(self):
        """Test getting engine priority list."""
        engines = time_util.try_time_engines()
        
        assert isinstance(engines, list)
        assert len(engines) > 0
        assert 'netcdf4' in engines
        assert 'h5netcdf' in engines
        assert 'scipy' in engines
    
    def test_time_decoding_integration_with_io_util(self):
        """Test integration of time decoding utilities with io_util."""
        # Test that io_util functions use time decoding parameters
        engines = io_util.try_time_engines()
        safe_params = io_util.get_time_decoding_params(safe_mode=True)
        normal_params = io_util.get_time_decoding_params(safe_mode=False)
        
        # Should be the same as calling through time_util
        assert engines == time_util.try_time_engines()
        assert safe_params == time_util.get_time_decoding_params(safe_mode=True)
        assert normal_params == time_util.get_time_decoding_params(safe_mode=False)


class TestUtilityFunctions:
    """Test time utility helper functions."""
    
    def test_validate_year_range_valid(self):
        """Test year range validation with valid range."""
        assert time_util.validate_year_range(1990, 2020) is True
        assert time_util.validate_year_range(2000, 2000) is True  # Single year
    
    def test_validate_year_range_invalid(self):
        """Test year range validation with invalid range."""
        assert time_util.validate_year_range(2020, 1990) is False  # Start > end
    
    def test_validate_year_range_extreme_values(self):
        """Test year range validation with extreme values."""
        # Should pass but generate warnings
        assert time_util.validate_year_range(1800, 1850) is True  # Before typical range
        assert time_util.validate_year_range(2100, 2200) is True  # After typical range
    
    def test_get_years_in_range(self):
        """Test getting list of years in range."""
        years = time_util.get_years_in_range(2010, 2013)
        assert years == [2010, 2011, 2012, 2013]
        
        # Single year
        years = time_util.get_years_in_range(2020, 2020)
        assert years == [2020]
        
        # Invalid range
        years = time_util.get_years_in_range(2020, 2010)
        assert years == []
    
    def test_is_leap_year(self):
        """Test leap year detection."""
        # Standard leap years
        assert time_util.is_leap_year(2000) is True
        assert time_util.is_leap_year(2004) is True
        assert time_util.is_leap_year(2020) is True
        
        # Non-leap years
        assert time_util.is_leap_year(2001) is False
        assert time_util.is_leap_year(2100) is False  # Century year not divisible by 400
        
        # Special cases
        assert time_util.is_leap_year(1900) is False  # Century year not divisible by 400
        assert time_util.is_leap_year(2000) is True   # Century year divisible by 400
    
    def test_get_days_in_year(self):
        """Test getting number of days in year."""
        assert time_util.get_days_in_year(2020) == 366  # Leap year
        assert time_util.get_days_in_year(2021) == 365  # Non-leap year
        assert time_util.get_days_in_year(2000) == 366  # Leap year
        assert time_util.get_days_in_year(1900) == 365  # Non-leap year
    
    def test_standardize_to_365_days_regular_year(self):
        """Test standardizing day-of-year for regular year."""
        day_of_year = np.arange(1, 366)  # 365 days
        standardized = time_util.standardize_to_365_days(day_of_year, 2021)  # Non-leap year
        
        # Should remain unchanged
        np.testing.assert_array_equal(standardized, day_of_year)
    
    def test_standardize_to_365_days_leap_year(self):
        """Test standardizing day-of-year for leap year."""
        day_of_year = np.arange(1, 367)  # 366 days (leap year)
        standardized = time_util.standardize_to_365_days(day_of_year, 2020)  # Leap year
        
        # Should remove day 60 (Feb 29) and adjust subsequent days
        assert len(standardized) == 365
        
        # The original day 60 (Feb 29) should be removed from the sequence
        # But after adjustment, what was day 61 becomes the new day 60
        # So we need to check that the sequence is correct
        
        # Days before Feb 29 should remain unchanged (1-59)
        np.testing.assert_array_equal(standardized[:59], np.arange(1, 60))
        
        # What was originally day 61 should now be day 60
        # What was originally day 62 should now be day 61, etc.
        # So the rest should be 60, 61, 62, ..., 365
        np.testing.assert_array_equal(standardized[59:], np.arange(60, 366))
        
        # Verify no gaps in the sequence
        assert np.array_equal(standardized, np.arange(1, 366))
        
        # The final day should be 365
        assert standardized[-1] == 365
    
    def test_standardize_to_365_days_non_leap_year_data(self):
        """Test standardizing with non-leap year data."""
        day_of_year = np.arange(1, 366)  # 365 days
        standardized = time_util.standardize_to_365_days(day_of_year, 2019)  # Non-leap year
        
        # Should remain unchanged and clipped to 365
        np.testing.assert_array_equal(standardized, np.clip(day_of_year, 1, 365))


class TestWorkflowIntegration:
    """Test integration with main climate processing workflows."""
    
    @pytest.fixture
    def sample_climate_dataset(self):
        """Create a sample climate dataset for workflow testing."""
        time = np.arange(0, 365)  # One year of daily data
        lat = np.linspace(25, 50, 20)
        lon = np.linspace(-125, -65, 30)
        
        temp_data = 273.15 + 15 + 10 * np.sin(2 * np.pi * time[:, None, None] / 365) + \
                   np.random.random((365, 20, 30)) * 2
        
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
    
    def test_time_coordinate_handling_in_workflow(self, sample_climate_dataset):
        """Test time coordinate handling in processing workflow."""
        # Save dataset to temporary file
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as temp_file:
            sample_climate_dataset.to_netcdf(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Mock the workflow components
            mock_handler = MagicMock()
            mock_handler.extract_year_from_filename.return_value = 2010
            
            # Test time coordinate handling through climate_means
            with patch('climate_means.open_dataset_safely') as mock_open:
                mock_open.return_value = sample_climate_dataset
                
                # This should use handle_time_coordinates from time_util
                ds, time_method = climate_means.handle_time_coordinates(sample_climate_dataset, temp_file_path)
                
                assert time_method == 'daily'
                assert 'dayofyear' in ds.coords
                assert len(ds.dayofyear) == 365
        
        finally:
            Path(temp_file_path).unlink(missing_ok=True)
    
    def test_climate_normal_computation_integration(self):
        """Test climate normal computation with time handling."""
        # Create multiple years of daily climatologies
        years = [2010, 2011, 2012, 2013, 2014]
        data_arrays = []
        
        for year in years:
            # Create daily climatology for each year
            daily_values = 273.15 + 15 + 10 * np.sin(2 * np.pi * np.arange(365) / 365) + \
                          np.random.random(365) * 2
            
            # Convert to numpy array as the function expects
            data_arrays.append(daily_values)
        
        target_year = 2012
        
        # Test the climate normal computation
        climate_normal = climate_means.compute_climate_normal(
            data_arrays, years, target_year
        )
        
        # Should have called the time utility function
        assert 'long_name' in climate_normal.attrs
        assert 'description' in climate_normal.attrs
        assert 'target_year' in climate_normal.attrs
        assert 'source_years' in climate_normal.attrs
        assert 'number_of_years' in climate_normal.attrs
        assert 'climatology_type' in climate_normal.attrs
        
        assert climate_normal.attrs['target_year'] == target_year
        assert climate_normal.attrs['number_of_years'] == len(years)
        assert "daily" in climate_normal.attrs['climatology_type']
    
    def test_period_generation_in_workflow(self):
        """Test period generation in processing workflow."""
        # Test through process_climate_data_workflow indirectly
        data_availability = {
            'historical': {'start': 1950, 'end': 2014},
            'ssp245': {'start': 2015, 'end': 2100}
        }
        
        # This would be called in the workflow
        historical_periods = climate_means.generate_climate_periods('historical', data_availability)
        ssp245_periods = climate_means.generate_climate_periods('ssp245', data_availability)
        
        assert len(historical_periods) > 0
        assert len(ssp245_periods) > 0
        
        # Check that periods have proper structure for workflow
        for period in historical_periods[:5]:  # Check first 5 periods
            assert len(period) == 4  # start_year, end_year, target_year, period_name
            assert isinstance(period[3], str)  # period_name should be string
            assert 'historical' in period[3]  # Should contain scenario name


class TestErrorHandling:
    """Test error handling in time utilities."""
    
    def test_extract_year_from_filename_with_none_input(self):
        """Test year extraction with None input."""
        year = time_util.extract_year_from_filename(None)
        assert year is None
    
    def test_extract_year_from_filename_with_empty_string(self):
        """Test year extraction with empty string."""
        year = time_util.extract_year_from_filename("")
        assert year is None
    
    def test_handle_time_coordinates_with_invalid_dataset(self):
        """Test time coordinate handling with invalid dataset."""
        # Create dataset without proper structure
        invalid_ds = xr.Dataset({})
        
        ds_result, time_method = time_util.handle_time_coordinates(invalid_ds, "invalid.nc")
        
        assert time_method == 'none'
        assert ds_result is invalid_ds
    
    def test_reconstruct_time_dataarray_with_unexpected_dimensions(self):
        """Test DataArray reconstruction with unexpected dimensions."""
        # Create 4D data (unexpected)
        data_4d = np.random.random((10, 20, 30, 40))
        
        da = time_util.reconstruct_time_dataarray(data_4d, 2010)
        
        # Should still create a DataArray with generic coordinates
        assert isinstance(da, xr.DataArray)
        assert 'year' in da.coords
        assert da.coords['year'] == 2010
        # Should have generic dimension names
        assert all(dim.startswith('dim_') for dim in da.dims if dim != 'year')
    
    def test_filter_files_by_year_range_with_invalid_files(self):
        """Test file filtering with files that don't have extractable years."""
        files = [
            "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2010.nc",  # Valid
            "invalid_filename.nc",  # Invalid
            "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2020.nc"   # Valid
        ]
        
        filtered = time_util.filter_files_by_year_range(files, 2010, 2020)
        
        # Should only include files with extractable years in range
        assert len(filtered) == 2
        valid_files = [f for f in filtered if "2010" in f or "2020" in f]
        assert len(valid_files) == 2
    
    def test_validate_year_range_edge_cases(self):
        """Test year range validation with edge cases."""
        # Very large range
        assert time_util.validate_year_range(1, 3000) is True
        
        # Negative years (shouldn't happen but should handle gracefully)
        assert time_util.validate_year_range(-100, 100) is True
    
    def test_standardize_to_365_days_with_wrong_length(self):
        """Test day standardization with unexpected array length."""
        # Array that's not 366 days for leap year
        day_of_year = np.arange(1, 301)  # 300 days
        
        standardized = time_util.standardize_to_365_days(day_of_year, 2020)  # Leap year
        
        # Should just clip to 365 without other processing
        expected = np.clip(day_of_year, 1, 365)
        np.testing.assert_array_equal(standardized, expected)


def test_module_level_time_integration():
    """Test module-level integration aspects."""
    # Test that all modules can be imported together
    import time_util
    import climate_means
    import io_util
    
    # Test that functions are properly shared
    assert climate_means.generate_climate_periods is time_util.generate_climate_periods
    assert io_util.extract_year_from_filename is time_util.extract_year_from_filename
    
    # Test that no circular imports occur
    assert hasattr(time_util, 'generate_climate_periods')
    assert hasattr(time_util, 'extract_year_from_filename')
    assert hasattr(time_util, 'handle_time_coordinates')


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 