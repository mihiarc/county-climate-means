#!/usr/bin/env python3
"""
Integration tests for climate_means.py and io_util.py modules.

Tests the interaction between the main climate processing module and the I/O utilities,
ensuring that imports work correctly and that the modules can work together.
"""

import pytest
import tempfile
import shutil
import numpy as np
import xarray as xr
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Import modules to test
import climate_means
import io_util
from io_util import NorESM2FileHandler, open_dataset_safely, save_climate_result, SAFE_CHUNKS


class TestModuleIntegration:
    """Test integration between different modules."""
    
    def test_imports_successful(self):
        """Test that all required imports work correctly."""
        # Test that climate_means can import from io_util
        assert hasattr(climate_means, 'open_dataset_safely')
        assert hasattr(climate_means, 'NorESM2FileHandler')
        assert hasattr(climate_means, 'save_climate_result')
        assert hasattr(climate_means, 'SAFE_CHUNKS')
        
        # Test that climate_means can import from regions
        assert hasattr(climate_means, 'REGION_BOUNDS')
        assert hasattr(climate_means, 'extract_region')
        assert hasattr(climate_means, 'validate_region_bounds')
        
        # Test that climate_means can import from time_util
        assert hasattr(climate_means, 'generate_climate_periods')
        assert hasattr(climate_means, 'handle_time_coordinates')
        
        # Test that climate_means can import from dask_util
        assert hasattr(climate_means, 'setup_dask_client')
        assert hasattr(climate_means, 'enhanced_performance_monitor')
    
    def test_safe_chunks_consistency(self):
        """Test that SAFE_CHUNKS is consistent between modules."""
        # Check that SAFE_CHUNKS exists in both modules
        assert hasattr(climate_means, 'SAFE_CHUNKS')
        assert hasattr(io_util, 'SAFE_CHUNKS')
        
        # Check that they're the same object (not just equal values)
        assert climate_means.SAFE_CHUNKS is io_util.SAFE_CHUNKS
        
        # Check the structure
        expected_keys = {'time', 'lat', 'lon'}
        assert set(io_util.SAFE_CHUNKS.keys()) == expected_keys
        assert all(isinstance(v, int) for v in io_util.SAFE_CHUNKS.values())


class TestNorESM2FileHandlerIntegration:
    """Test NorESM2FileHandler integration with climate processing functions."""
    
    @pytest.fixture
    def temp_data_structure(self):
        """Create a temporary directory structure mimicking NorESM2-LM data."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "NorESM2-LM"
        
        # Create directory structure
        variables = ['tas', 'pr']
        scenarios = ['historical', 'ssp245']
        
        for var in variables:
            for scenario in scenarios:
                scenario_dir = data_dir / var / scenario
                scenario_dir.mkdir(parents=True, exist_ok=True)
                
                # Create dummy files for a few years
                for year in [2010, 2011, 2012]:
                    filename = f"{var}_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc"
                    filepath = scenario_dir / filename
                    filepath.touch()  # Create empty file
        
        yield str(data_dir)
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_file_handler_creation_from_climate_means(self, temp_data_structure):
        """Test that NorESM2FileHandler can be created from climate_means module."""
        # This tests that the import works and the class is usable
        handler = climate_means.NorESM2FileHandler(temp_data_structure)
        
        assert handler.data_directory == Path(temp_data_structure)
        assert handler.model_name == "NorESM2-LM"
        assert handler.variant_label == "r1i1p1f1"
        assert handler.grid_label == "gn"
    
    def test_file_handler_methods_accessible(self, temp_data_structure):
        """Test that all NorESM2FileHandler methods are accessible from climate_means."""
        handler = climate_means.NorESM2FileHandler(temp_data_structure)
        
        # Test that methods exist and are callable
        assert callable(handler.extract_year_from_filename)
        assert callable(handler.get_files_for_period)
        assert callable(handler.get_available_years)
        assert callable(handler.validate_data_availability)
        
        # Test a simple method call
        year = handler.extract_year_from_filename("tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2010.nc")
        assert year == 2010
    
    def test_get_files_for_period_integration(self, temp_data_structure):
        """Test get_files_for_period method integration."""
        handler = climate_means.NorESM2FileHandler(temp_data_structure)
        
        files = handler.get_files_for_period('tas', 'historical', 2010, 2012)
        assert len(files) == 3  # Should find 2010, 2011, 2012
        
        # Check that files are properly sorted
        years = [handler.extract_year_from_filename(f) for f in files]
        assert years == [2010, 2011, 2012]


class TestDatasetOperations:
    """Test dataset operations across modules."""
    
    @pytest.fixture
    def temp_netcdf_file(self):
        """Create a temporary NetCDF file for testing."""
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        temp_file.close()
        
        # Create sample dataset
        time = pd.date_range('2020-01-01', periods=10, freq='D')
        lat = np.linspace(25, 50, 20)
        lon = np.linspace(-125, -65, 30)
        
        temp_data = 273.15 + 15 + np.random.random((10, 20, 30)) * 10
        
        ds = xr.Dataset({
            'tas': (['time', 'lat', 'lon'], temp_data)
        }, coords={
            'time': time,
            'lat': lat,
            'lon': lon
        })
        
        ds.to_netcdf(temp_file.name)
        
        yield temp_file.name
        
        # Cleanup
        Path(temp_file.name).unlink(missing_ok=True)
    
    def test_open_dataset_safely_integration(self, temp_netcdf_file):
        """Test open_dataset_safely function integration."""
        ds = climate_means.open_dataset_safely(temp_netcdf_file)
        
        assert ds is not None
        assert 'tas' in ds.data_vars
        assert len(ds.time) == 10


class TestWorkflowIntegration:
    """Test integration across the entire workflow."""
    
    def test_climate_means_can_use_io_functions(self):
        """Test that climate_means can successfully use all imported I/O functions."""
        # Test that functions are callable from climate_means module
        assert callable(climate_means.open_dataset_safely)
        assert callable(climate_means.NorESM2FileHandler)
        assert callable(climate_means.save_climate_result)
        
        # Test that constants are accessible
        assert hasattr(climate_means, 'SAFE_CHUNKS')
        assert isinstance(climate_means.SAFE_CHUNKS, dict)
    
    @patch('climate_means.NorESM2FileHandler')
    def test_workflow_components_accessible_through_climate_means(self, mock_handler_class):
        """Test that workflow components are accessible through climate_means module."""
        # Setup mock
        mock_handler = MagicMock()
        mock_handler.validate_data_availability.return_value = {
            'tas': {'historical': (1950, 2014)}
        }
        mock_handler_class.return_value = mock_handler
        
        # Test that the workflow components can be accessed
        # These are used by the runner script
        assert hasattr(climate_means, 'setup_dask_client')
        assert hasattr(climate_means, 'cleanup_dask_resources') 
        assert hasattr(climate_means, 'process_climate_data_workflow')
        assert hasattr(climate_means, 'NorESM2FileHandler')
        
        # Test that NorESM2FileHandler can be instantiated
        handler = climate_means.NorESM2FileHandler("/tmp")
        assert handler is not None
        
        # Note: process_noresm2_data has been moved to run_climate_means.py
        # Integration with that function is tested in test_runner_integration.py
    


class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    def test_file_handler_with_nonexistent_directory(self):
        """Test NorESM2FileHandler error handling with nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            climate_means.NorESM2FileHandler("/nonexistent/path")
    
    def test_open_dataset_safely_with_invalid_file(self):
        """Test open_dataset_safely with invalid file."""
        result = climate_means.open_dataset_safely("/nonexistent/file.nc")
        assert result is None
    
    def test_save_climate_result_with_invalid_path(self):
        """Test save_climate_result error handling."""
        # Create a sample DataArray
        data = xr.DataArray([1, 2, 3], dims=['x'])
        
        # Try to save to an invalid path
        invalid_path = Path("/nonexistent/directory")
        success = climate_means.save_climate_result(
            data, invalid_path, 'test_var', 'test_region', 'test_period'
        )
        
        assert success is False


def test_module_level_integration():
    """Test module-level integration aspects."""
    # Test that both modules can be imported together
    import climate_means
    import io_util
    
    # Test that they don't conflict
    assert climate_means.NorESM2FileHandler is io_util.NorESM2FileHandler
    
    # Test that constants are properly shared
    assert climate_means.SAFE_CHUNKS is io_util.SAFE_CHUNKS


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 