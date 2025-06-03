#!/usr/bin/env python3
"""
End-to-End Integration Tests for Climate Means Program

This module provides comprehensive end-to-end testing of the complete climate means
processing workflow, from input data ingestion through final output generation.

Tests cover:
- Complete workflow execution with mock data
- Data flow from input to output
- Integration between all major modules
- Error handling and recovery
- Performance characteristics
- Real data processing scenarios (when available)
"""

import pytest
import tempfile
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timedelta
import json
import logging

# Import modules to test
import run_climate_means
import climate_means
from climate_means import process_climate_data_workflow
from io_util import NorESM2FileHandler, save_climate_result
from regions import REGION_BOUNDS
from time_util import generate_climate_periods


class TestClimateWorkflowEndToEnd:
    """End-to-end tests for the complete climate processing workflow."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a complete temporary workspace with mock climate data."""
        workspace = tempfile.mkdtemp()
        workspace_path = Path(workspace)
        
        # Create input data directory structure
        input_dir = workspace_path / "input_data" / "NorESM2-LM"
        output_dir = workspace_path / "output"
        
        # Create directories for variables and scenarios
        variables = ['tas', 'tasmax', 'tasmin', 'pr']
        scenarios = ['historical', 'ssp245', 'ssp585']
        
        for var in variables:
            for scenario in scenarios:
                scenario_dir = input_dir / var / scenario
                scenario_dir.mkdir(parents=True, exist_ok=True)
                
                # Create mock data files for multiple years
                # Historical: 1980-2014 (35 years for testing)
                # Future scenarios: 2015-2050 (sufficient for 30-year normals)
                if scenario == 'historical':
                    years = range(1980, 2015)
                else:
                    years = range(2015, 2051)
                
                for year in years:
                    filename = f"{var}_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc"
                    filepath = scenario_dir / filename
                    
                    # Create realistic mock NetCDF file
                    self._create_mock_netcdf_file(filepath, var, year)
        
        yield {
            'workspace': workspace_path,
            'input_dir': str(input_dir),
            'output_dir': str(output_dir)
        }
        
        # Cleanup
        shutil.rmtree(workspace)
    
    def _create_mock_netcdf_file(self, filepath: Path, variable: str, year: int):
        """Create a realistic mock NetCDF file with proper structure."""
        # Create time coordinate for the entire year
        start_date = pd.Timestamp(f'{year}-01-01')
        if year % 4 == 0:  # Leap year
            days = 366
        else:
            days = 365
        
        time = pd.date_range(start_date, periods=days, freq='D')
        
        # Create spatial coordinates covering different US regions
        # CONUS: roughly 25-50°N, 126-66°W (234-294°E in 0-360)
        # AK: roughly 50-72°N, 170-235°E
        # HI: roughly 18-29°N, 182-205°E
        lat = np.linspace(15, 75, 60)  # Cover all regions
        lon = np.linspace(170, 300, 130)  # Cover all regions in 0-360 system
        
        # Create realistic data based on variable type
        if variable == 'tas':  # Surface air temperature
            # Base temperature with seasonal cycle
            base_temp = 273.15 + 10  # 10°C base
            seasonal_cycle = 15 * np.sin(2 * np.pi * np.arange(days) / 365.25)
            daily_temps = base_temp + seasonal_cycle
            
            # Add spatial gradients (latitude and longitude effects)
            lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
            spatial_temp = 30 - 0.5 * (lat_grid - 40)  # Temperature decreases with latitude
            
            # Create 3D temperature data
            temp_data = np.zeros((days, len(lat), len(lon)))
            for i, daily_temp in enumerate(daily_temps):
                temp_data[i] = daily_temp + spatial_temp + np.random.normal(0, 2, (len(lat), len(lon)))
                
            data_array = temp_data
            
        elif variable == 'tasmax':  # Maximum temperature
            # Similar to tas but higher
            base_temp = 273.15 + 15
            seasonal_cycle = 15 * np.sin(2 * np.pi * np.arange(days) / 365.25)
            daily_temps = base_temp + seasonal_cycle
            
            lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
            spatial_temp = 35 - 0.5 * (lat_grid - 40)
            
            temp_data = np.zeros((days, len(lat), len(lon)))
            for i, daily_temp in enumerate(daily_temps):
                temp_data[i] = daily_temp + spatial_temp + np.random.normal(0, 3, (len(lat), len(lon)))
                
            data_array = temp_data
            
        elif variable == 'tasmin':  # Minimum temperature
            # Similar to tas but lower
            base_temp = 273.15 + 5
            seasonal_cycle = 15 * np.sin(2 * np.pi * np.arange(days) / 365.25)
            daily_temps = base_temp + seasonal_cycle
            
            lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
            spatial_temp = 25 - 0.5 * (lat_grid - 40)
            
            temp_data = np.zeros((days, len(lat), len(lon)))
            for i, daily_temp in enumerate(daily_temps):
                temp_data[i] = daily_temp + spatial_temp + np.random.normal(0, 2, (len(lat), len(lon)))
                
            data_array = temp_data
            
        elif variable == 'pr':  # Precipitation
            # Precipitation with realistic patterns
            base_precip = 0.001  # 1mm/day base
            seasonal_cycle = 0.002 * np.sin(2 * np.pi * np.arange(days) / 365.25 + np.pi/2)  # Summer peak
            daily_precip = base_precip + seasonal_cycle
            
            # Add spatial patterns and randomness
            precip_data = np.zeros((days, len(lat), len(lon)))
            for i, daily_pr in enumerate(daily_precip):
                # Add random precipitation events
                random_events = np.random.exponential(0.005, (len(lat), len(lon)))
                precip_data[i] = np.maximum(0, daily_pr + random_events)
                
            data_array = precip_data
        
        # Create dataset
        ds = xr.Dataset(
            {
                variable: (['time', 'lat', 'lon'], data_array),
            },
            coords={
                'time': time,
                'lat': lat,
                'lon': lon,
            },
            attrs={
                'title': f'Mock {variable} data for testing',
                'institution': 'Test Suite',
                'source': 'NorESM2-LM (mock)',
                'experiment_id': 'historical' if year < 2015 else 'ssp245',
                'variant_label': 'r1i1p1f1',
                'grid_label': 'gn',
                'creation_date': datetime.now().isoformat(),
            }
        )
        
        # Add variable attributes
        if variable in ['tas', 'tasmax', 'tasmin']:
            ds[variable].attrs = {
                'standard_name': 'air_temperature',
                'long_name': f'{variable} - air temperature',
                'units': 'K',
                'cell_methods': 'time: mean'
            }
        elif variable == 'pr':
            ds[variable].attrs = {
                'standard_name': 'precipitation_flux',
                'long_name': 'Precipitation',
                'units': 'kg m-2 s-1',
                'cell_methods': 'time: mean'
            }
        
        # Add time attributes
        ds.time.attrs = {
            'standard_name': 'time',
            'long_name': 'time',
            'bounds': 'time_bnds'
        }
        
        # Add coordinates attributes
        ds.lat.attrs = {
            'standard_name': 'latitude',
            'long_name': 'latitude',
            'units': 'degrees_north'
        }
        
        ds.lon.attrs = {
            'standard_name': 'longitude', 
            'long_name': 'longitude',
            'units': 'degrees_east'
        }
        
        # Save to file
        ds.to_netcdf(filepath)
        ds.close()
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_workflow_execution(self, temp_workspace):
        """Test the complete climate means workflow from start to finish."""
        input_dir = temp_workspace['input_dir']
        output_dir = temp_workspace['output_dir']
        
        # Configuration for testing
        config = {
            'max_workers': 2,           # Minimal workers for testing
            'target_chunk_size': 64,    # Small chunks for testing
            'batch_size': 5,            # Small batches
            'computation_type': 'mixed',
            'memory_safety_margin': 0.8,
            'max_retries': 2
        }
        
        # Test parameters - subset for faster testing
        variables = ['tas', 'pr']  # Test temperature and precipitation
        regions = ['CONUS']        # Test main CONUS region
        scenarios = ['historical'] # Test historical scenario
        
        # Execute the complete workflow
        try:
            process_climate_data_workflow(
                data_directory=input_dir,
                output_directory=output_dir,
                variables=variables,
                regions=regions,
                scenarios=scenarios,
                config=config
            )
            
            # Verify output files were created
            output_path = Path(output_dir)
            assert output_path.exists(), "Output directory should exist"
            
            # Check for expected output files
            output_files = list(output_path.glob("*.nc"))
            assert len(output_files) > 0, "Should have created output files"
            
            # Verify file naming patterns and content
            for output_file in output_files:
                # Check file can be opened
                ds = xr.open_dataset(output_file)
                assert ds is not None, f"Should be able to open output file: {output_file}"
                
                # Verify expected data variables exist
                data_vars = list(ds.data_vars.keys())
                assert len(data_vars) > 0, "Output should contain data variables"
                
                # Verify metadata
                assert 'variable' in ds.attrs or any('variable' in ds[var].attrs for var in data_vars)
                assert 'region' in ds.attrs or any('region' in ds[var].attrs for var in data_vars)
                
                ds.close()
                
        except Exception as e:
            pytest.fail(f"Complete workflow execution failed: {e}")
    
    @pytest.mark.integration
    def test_workflow_with_multiple_variables_and_regions(self, temp_workspace):
        """Test workflow with multiple variables and regions."""
        input_dir = temp_workspace['input_dir']
        output_dir = temp_workspace['output_dir']
        
        config = {
            'max_workers': 2,
            'target_chunk_size': 32,
            'batch_size': 3,
            'computation_type': 'mixed'
        }
        
        # Test with multiple variables and regions
        variables = ['tas', 'tasmax', 'pr']
        regions = ['CONUS', 'AK']  # Test multiple regions
        scenarios = ['historical']
        
        # Execute workflow
        process_climate_data_workflow(
            data_directory=input_dir,
            output_directory=output_dir,
            variables=variables,
            regions=regions,
            scenarios=scenarios,
            config=config
        )
        
        # Verify outputs for each variable-region combination
        output_path = Path(output_dir)
        expected_combinations = len(variables) * len(regions)
        
        output_files = list(output_path.glob("*.nc"))
        # Should have files for each variable-region-period combination
        assert len(output_files) >= expected_combinations, \
            f"Expected at least {expected_combinations} output files, got {len(output_files)}"
    
    @pytest.mark.integration
    def test_workflow_error_handling(self, temp_workspace):
        """Test workflow error handling with invalid inputs."""
        input_dir = temp_workspace['input_dir']
        output_dir = temp_workspace['output_dir']
        
        config = {'max_workers': 1, 'target_chunk_size': 32}
        
        # Test with invalid variable
        with pytest.raises((Exception, SystemExit)):
            process_climate_data_workflow(
                data_directory=input_dir,
                output_directory=output_dir,
                variables=['invalid_variable'],
                regions=['CONUS'],
                scenarios=['historical'],
                config=config
            )
        
        # Test with invalid region
        with pytest.raises((Exception, SystemExit, ValueError)):
            process_climate_data_workflow(
                data_directory=input_dir,
                output_directory=output_dir,
                variables=['tas'],
                regions=['INVALID_REGION'],
                scenarios=['historical'],
                config=config
            )
    
    @pytest.mark.integration
    def test_climate_normal_calculation_accuracy(self, temp_workspace):
        """Test that climate normal calculations produce reasonable results."""
        input_dir = temp_workspace['input_dir']
        output_dir = temp_workspace['output_dir']
        
        config = {
            'max_workers': 1,
            'target_chunk_size': 32,
            'batch_size': 35  # Process all historical years at once
        }
        
        # Process historical temperature data
        process_climate_data_workflow(
            data_directory=input_dir,
            output_directory=output_dir,
            variables=['tas'],
            regions=['CONUS'],
            scenarios=['historical'],
            config=config
        )
        
        # Analyze the output for reasonableness
        output_files = list(Path(output_dir).glob("*tas*CONUS*historical*.nc"))
        assert len(output_files) > 0, "Should have temperature output files"
        
        for output_file in output_files:
            ds = xr.open_dataset(output_file)
            
            # Check that temperature values are reasonable (in Kelvin)
            temp_data = ds['tas'] if 'tas' in ds else ds[list(ds.data_vars)[0]]
            temp_mean = float(temp_data.mean())
            
            # Should be reasonable temperature range (200-350K for surface temps)
            assert 200 < temp_mean < 350, f"Temperature mean {temp_mean}K seems unrealistic"
            
            # Check spatial patterns exist (should vary across CONUS)
            temp_std = float(temp_data.std())
            assert temp_std > 1, "Should have spatial temperature variation"
            
            ds.close()


class TestRunnerEndToEnd:
    """End-to-end tests for the runner script execution."""
    
    @pytest.fixture
    def temp_noresm2_structure(self):
        """Create minimal NorESM2 data structure for runner testing."""
        temp_dir = tempfile.mkdtemp()
        noresm2_path = Path(temp_dir) / "NorESM2-LM"
        
        # Create minimal structure
        for var in ['tas', 'pr']:
            for scenario in ['historical']:
                scenario_dir = noresm2_path / var / scenario
                scenario_dir.mkdir(parents=True, exist_ok=True)
                
                # Create just a few files for quick testing
                for year in [2010, 2011]:
                    filename = f"{var}_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc"
                    filepath = scenario_dir / filename
                    
                    # Create minimal NetCDF file
                    time = pd.date_range(f'{year}-01-01', periods=10, freq='D')
                    lat = np.linspace(30, 40, 5)
                    lon = np.linspace(240, 250, 5)
                    
                    data = np.random.random((10, 5, 5))
                    if var == 'tas':
                        data = 273.15 + 20 * data  # Temperature in K
                    else:
                        data = 0.01 * data  # Precipitation
                    
                    ds = xr.Dataset(
                        {var: (['time', 'lat', 'lon'], data)},
                        coords={'time': time, 'lat': lat, 'lon': lon}
                    )
                    ds.to_netcdf(filepath)
                    ds.close()
        
        yield str(noresm2_path)
        shutil.rmtree(temp_dir)
    
    @patch('run_climate_means.Path.exists')
    @patch('run_climate_means.process_climate_data_workflow')
    @patch('run_climate_means.NorESM2FileHandler')
    def test_runner_noresm2_mode_execution(self, mock_handler_class, mock_workflow, 
                                         mock_path_exists, temp_noresm2_structure):
        """Test runner execution in NorESM2 mode."""
        # Setup mocks
        mock_path_exists.return_value = True
        mock_handler = MagicMock()
        mock_handler.validate_data_availability.return_value = {
            'tas': {'historical': (1980, 2014)},
            'pr': {'historical': (1980, 2014)}
        }
        mock_handler_class.return_value = mock_handler
        
        # Test execution
        try:
            run_climate_means.process_noresm2_data()
            
            # Verify workflow was called
            mock_workflow.assert_called_once()
            
            # Verify handler was created and used
            mock_handler_class.assert_called_once()
            mock_handler.validate_data_availability.assert_called_once()
            
        except Exception as e:
            # Should not fail with proper mocking
            pytest.fail(f"Runner execution failed: {e}")
    
    @patch('run_climate_means.setup_dask_client')
    @patch('run_climate_means.cleanup_dask_resources')
    def test_runner_main_function_execution(self, mock_cleanup, mock_setup):
        """Test runner main function execution."""
        # Setup mocks
        mock_client = MagicMock()
        mock_setup.return_value = mock_client
        
        # Execute main function
        try:
            run_climate_means.main()
            
            # Verify Dask setup and cleanup
            mock_setup.assert_called_once()
            mock_cleanup.assert_called_once_with(mock_client)
            
        except Exception as e:
            pytest.fail(f"Runner main execution failed: {e}")


class TestDataFlowIntegration:
    """Test data flow through the entire pipeline."""
    
    def test_file_handler_to_workflow_integration(self, tmp_path):
        """Test data flow from file handler through workflow components."""
        # Create test data structure
        data_dir = tmp_path / "test_data"
        
        # Create minimal file structure
        var_dir = data_dir / "tas" / "historical"
        var_dir.mkdir(parents=True)
        
        # Create test file
        test_file = var_dir / "tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2010.nc"
        
        # Create minimal but valid NetCDF
        time = pd.date_range('2010-01-01', periods=5, freq='D')
        lat = np.array([30.0, 35.0, 40.0])
        lon = np.array([240.0, 245.0, 250.0])
        
        temp_data = np.random.random((5, 3, 3)) * 20 + 273.15
        
        ds = xr.Dataset(
            {'tas': (['time', 'lat', 'lon'], temp_data)},
            coords={'time': time, 'lat': lat, 'lon': lon}
        )
        ds.to_netcdf(test_file, engine='netcdf4')
        ds.close()
        
        # Test file handler
        handler = NorESM2FileHandler(str(data_dir))
        
        # Test file discovery
        files = handler.get_files_for_period('tas', 'historical', 2010, 2010)
        assert len(files) == 1
        assert str(test_file) in files
        
        # Test year extraction
        year = handler.extract_year_from_filename(str(test_file))
        assert year == 2010
        
        # Test data availability
        availability = handler.validate_data_availability()
        assert 'tas' in availability
        assert 'historical' in availability['tas']
        assert availability['tas']['historical'] == (2010, 2010)
    
    def test_region_extraction_workflow(self):
        """Test region extraction through the workflow."""
        # Create test dataset covering multiple regions
        lat = np.linspace(20, 70, 50)  # Cover CONUS and Alaska
        lon = np.linspace(170, 300, 100)  # Cover all US regions
        time = pd.date_range('2010-01-01', periods=10, freq='D')
        
        # Create temperature data
        temp_data = np.random.random((10, 50, 100)) * 30 + 273.15
        
        ds = xr.Dataset(
            {'tas': (['time', 'lat', 'lon'], temp_data)},
            coords={'time': time, 'lat': lat, 'lon': lon}
        )
        
        # Test region extraction for CONUS
        from regions import extract_region
        
        conus_ds = extract_region(ds, REGION_BOUNDS['CONUS'])
        
        # Verify CONUS region was properly extracted
        assert conus_ds.lat.size > 0, "Should have latitude points in CONUS"
        assert conus_ds.lon.size > 0, "Should have longitude points in CONUS"
        
        # Verify bounds are reasonable for CONUS
        assert conus_ds.lat.min() >= REGION_BOUNDS['CONUS']['lat_min'] - 1
        assert conus_ds.lat.max() <= REGION_BOUNDS['CONUS']['lat_max'] + 1
        
        ds.close()
        conus_ds.close()


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""
    
    @pytest.mark.slow
    def test_workflow_memory_efficiency(self, tmp_path):
        """Test that workflow handles memory efficiently."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create larger test dataset
        data_dir = tmp_path / "large_test"
        var_dir = data_dir / "tas" / "historical"
        var_dir.mkdir(parents=True)
        
        # Create multiple years of data
        for year in range(2010, 2015):
            test_file = var_dir / f"tas_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}.nc"
            
            # Larger spatial grid
            time = pd.date_range(f'{year}-01-01', periods=365, freq='D')
            lat = np.linspace(25, 50, 25)
            lon = np.linspace(235, 295, 60)
            
            temp_data = np.random.random((365, 25, 60)) * 30 + 273.15
            
            ds = xr.Dataset(
                {'tas': (['time', 'lat', 'lon'], temp_data)},
                coords={'time': time, 'lat': lat, 'lon': lon}
            )
            ds.to_netcdf(test_file, engine='netcdf4')
            ds.close()
        
        # Force garbage collection
        gc.collect()
        
        # Test file handler memory usage
        handler = NorESM2FileHandler(str(data_dir))
        availability = handler.validate_data_availability()
        
        # Check memory didn't explode
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Should not increase by more than 200MB for this test
        assert memory_increase < 200, f"Memory usage increased by {memory_increase}MB"
    
    def test_dask_integration_scaling(self):
        """Test Dask integration and scaling behavior."""
        from dask_util import setup_dask_client, cleanup_dask_resources
        
        # Test different worker configurations
        configs = [
            {'max_workers': 1, 'target_chunk_size': 32},
            {'max_workers': 2, 'target_chunk_size': 64},
        ]
        
        for config in configs:
            try:
                client = setup_dask_client(config)
                
                # Verify client is working
                assert client is not None
                assert len(client.scheduler_info()['workers']) <= config['max_workers']
                
                # Cleanup
                cleanup_dask_resources(client)
                
            except Exception as e:
                pytest.fail(f"Dask scaling test failed with config {config}: {e}")


@pytest.mark.integration
def test_full_pipeline_integration():
    """Test the complete pipeline integration across all modules."""
    # Test that all modules can be imported and work together
    assert callable(climate_means.process_climate_data_workflow)
    assert callable(run_climate_means.main)
    assert callable(run_climate_means.process_noresm2_data)
    
    # Test that constants and configurations are consistent
    assert hasattr(climate_means, 'MIN_YEARS_FOR_CLIMATE_NORMAL')
    assert climate_means.MIN_YEARS_FOR_CLIMATE_NORMAL == 30
    
    # Test region definitions
    assert 'CONUS' in REGION_BOUNDS
    assert 'AK' in REGION_BOUNDS
    assert 'HI' in REGION_BOUNDS
    
    # Verify each region has required bounds
    for region_key, region_info in REGION_BOUNDS.items():
        assert 'lat_min' in region_info
        assert 'lat_max' in region_info
        assert 'lon_min' in region_info
        assert 'lon_max' in region_info
        assert 'name' in region_info


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "--tb=short"]) 