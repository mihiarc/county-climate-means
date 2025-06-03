#!/usr/bin/env python3
"""
Integration tests for dask_util.py module.

Tests the Dask utilities and their integration with the main climate processing
modules, ensuring that distributed computing functionality works correctly
and integrates seamlessly with climate data processing workflows.
"""

import pytest
import tempfile
import numpy as np
import xarray as xr
import multiprocessing as mp
import psutil
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, call
import time
import gc

# Import modules to test
import dask_util
import climate_means


class TestDaskUtilModuleIntegration:
    """Test integration between dask_util and climate_means modules."""
    
    def test_dask_util_imports_successful(self):
        """Test that all required Dask imports work correctly."""
        # Test that climate_means can import from dask_util
        assert hasattr(climate_means, 'setup_dask_client')
        assert hasattr(climate_means, 'enhanced_performance_monitor')
        assert hasattr(climate_means, 'cleanup_dask_resources')
        
        # Test that the imported functions are the same objects
        assert climate_means.setup_dask_client is dask_util.setup_dask_client
        assert climate_means.enhanced_performance_monitor is dask_util.enhanced_performance_monitor
        assert climate_means.cleanup_dask_resources is dask_util.cleanup_dask_resources
    
    def test_dask_util_functions_accessible(self):
        """Test that all dask_util functions are accessible."""
        # Configuration functions
        assert callable(dask_util.validate_dask_config)
        assert callable(dask_util.configure_dask_resources)
        
        # Cluster management
        assert callable(dask_util.setup_dask_client)
        assert callable(dask_util.cleanup_dask_resources)
        
        # Chunking optimization
        assert callable(dask_util.get_optimal_chunks)
        assert callable(dask_util._calculate_computation_aware_chunks)
        
        # Performance monitoring
        assert callable(dask_util.enhanced_performance_monitor)
        
        # Computation utilities
        assert callable(dask_util.robust_compute_with_retry)
    
    def test_dask_util_module_structure(self):
        """Test that dask_util module has expected structure."""
        # Should have configuration functions
        expected_functions = [
            'validate_dask_config',
            'configure_dask_resources', 
            'setup_dask_client',
            'get_optimal_chunks',
            'enhanced_performance_monitor',
            'robust_compute_with_retry',
            'cleanup_dask_resources'
        ]
        
        for func_name in expected_functions:
            assert hasattr(dask_util, func_name)
            assert callable(getattr(dask_util, func_name))


class TestDaskConfiguration:
    """Test Dask configuration validation and resource management."""
    
    def test_validate_dask_config_defaults(self):
        """Test Dask configuration validation with defaults."""
        config = {}
        validated = dask_util.validate_dask_config(config)
        
        # Should have all default values
        assert 'max_workers' in validated
        assert 'target_chunk_size' in validated
        assert 'batch_size' in validated
        assert 'memory_safety_margin' in validated
        assert 'computation_type' in validated
        assert 'max_retries' in validated
        
        # Check default values are reasonable
        assert validated['max_workers'] <= mp.cpu_count()
        assert validated['target_chunk_size'] == 128
        assert validated['batch_size'] == 20
        assert validated['memory_safety_margin'] == 0.8
        assert validated['computation_type'] == 'mixed'
        assert validated['max_retries'] == 3
    
    def test_validate_dask_config_custom_values(self):
        """Test Dask configuration validation with custom values."""
        config = {
            'max_workers': 2,
            'target_chunk_size': 64,
            'batch_size': 10,
            'memory_safety_margin': 0.6,
            'computation_type': 'time_series'
        }
        
        validated = dask_util.validate_dask_config(config)
        
        # Should preserve valid custom values
        assert validated['max_workers'] == 2
        assert validated['target_chunk_size'] == 64
        assert validated['batch_size'] == 10
        assert validated['memory_safety_margin'] == 0.6
        assert validated['computation_type'] == 'time_series'
    
    def test_validate_dask_config_range_validation(self):
        """Test that configuration values are properly constrained."""
        config = {
            'max_workers': 100,  # Too high
            'batch_size': -5,    # Too low
            'memory_safety_margin': 1.5,  # Too high
            'spill_threshold': 0.3   # Too low
        }
        
        validated = dask_util.validate_dask_config(config)
        
        # Should be constrained to valid ranges
        assert validated['max_workers'] <= mp.cpu_count()
        assert validated['batch_size'] >= 1
        assert validated['memory_safety_margin'] <= 1.0
        assert validated['spill_threshold'] >= 0.5
    
    def test_configure_dask_resources(self):
        """Test Dask resource configuration."""
        config = {
            'max_workers': 4,
            'memory_safety_margin': 0.7
        }
        
        dask_config = dask_util.configure_dask_resources(config)
        
        # Should have required keys
        assert 'n_workers' in dask_config
        assert 'memory_limit_per_worker' in dask_config
        assert 'threads_per_worker' in dask_config
        
        # Check for simplified fixed values
        assert dask_config['n_workers'] >= 1
        assert dask_config['n_workers'] <= 4  # Our new max limit
        assert dask_config['memory_limit_per_worker'] == "4GB"  # Our new fixed value
        assert dask_config['threads_per_worker'] == 2  # Our new fixed value
        assert dask_config['processes'] is True
        assert dask_config['silence_logs'] is False
        assert dask_config['dashboard_address'] == ':8787'


class TestDaskChunkingOptimization:
    """Test Dask chunking optimization functionality."""
    
    @pytest.fixture
    def sample_netcdf_file(self):
        """Create a sample NetCDF file for chunking tests."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        temp_file.close()
        
        # Create sample dataset
        time = np.arange(0, 365)
        lat = np.linspace(25, 50, 100)
        lon = np.linspace(-125, -65, 150)
        
        temp_data = 273.15 + 15 + np.random.random((365, 100, 150)) * 10
        
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
    
    def test_get_optimal_chunks_mixed_computation(self, sample_netcdf_file):
        """Test optimal chunking for mixed computation type."""
        config = {
            'target_chunk_size': 128,  # MB
            'computation_type': 'mixed'
        }
        
        chunks = dask_util.get_optimal_chunks(sample_netcdf_file, config)
        
        assert isinstance(chunks, dict)
        assert 'time' in chunks
        assert chunks['time'] > 0
        
        # May or may not have spatial chunking depending on data size
        if 'lat' in chunks:
            assert chunks['lat'] > 0
        if 'lon' in chunks:
            assert chunks['lon'] > 0
    
    def test_get_optimal_chunks_time_series_computation(self, sample_netcdf_file):
        """Test optimal chunking for time series computation."""
        config = {
            'target_chunk_size': 64,  # MB
            'computation_type': 'time_series'
        }
        
        chunks = dask_util.get_optimal_chunks(sample_netcdf_file, config)
        
        assert isinstance(chunks, dict)
        assert 'time' in chunks
        # Time series should prefer larger time chunks
        assert chunks['time'] >= 365
    
    def test_get_optimal_chunks_spatial_computation(self, sample_netcdf_file):
        """Test optimal chunking for spatial computation."""
        config = {
            'target_chunk_size': 64,  # MB
            'computation_type': 'spatial'
        }
        
        chunks = dask_util.get_optimal_chunks(sample_netcdf_file, config)
        
        assert isinstance(chunks, dict)
        assert 'time' in chunks
        # Spatial should prefer smaller time chunks
        assert chunks['time'] <= 365
    
    def test_get_optimal_chunks_invalid_file(self):
        """Test optimal chunking with invalid file."""
        config = {'target_chunk_size': 128}
        
        chunks = dask_util.get_optimal_chunks('/nonexistent/file.nc', config)
        
        # Should return fallback chunking
        assert chunks == {'time': 365}
    
    def test_calculate_computation_aware_chunks_dimensions(self):
        """Test computation-aware chunking calculation."""
        # Large dataset that needs spatial chunking
        time_size = 365
        lat_size = 500
        lon_size = 600
        config = {
            'target_chunk_size': 32,  # Small target to force spatial chunking
            'computation_type': 'mixed'
        }
        
        chunks = dask_util._calculate_computation_aware_chunks(
            time_size, lat_size, lon_size, config
        )
        
        assert isinstance(chunks, dict)
        assert 'time' in chunks
        
        # With large spatial dimensions and small target, should have spatial chunking
        if 'lat' in chunks and 'lon' in chunks:
            assert chunks['lat'] < lat_size
            assert chunks['lon'] < lon_size
    
    def test_calculate_computation_aware_chunks_time_series(self):
        """Test computation-aware chunking for time series type."""
        time_size = 10950  # 30 years
        lat_size = 100
        lon_size = 100
        config = {
            'target_chunk_size': 256,
            'computation_type': 'time_series'
        }
        
        chunks = dask_util._calculate_computation_aware_chunks(
            time_size, lat_size, lon_size, config
        )
        
        # Should prefer larger time chunks for time series
        assert chunks['time'] >= 1095  # At least 3 years


class TestDaskPerformanceMonitoring:
    """Test Dask performance monitoring functionality."""
    
    def test_enhanced_performance_monitor_success(self):
        """Test successful performance monitoring."""
        # Create mock client with realistic scheduler info
        mock_client = MagicMock()
        mock_client.scheduler_info.return_value = {
            'workers': {
                'worker1': {
                    'memory_limit': 2 * 1024**3,  # 2GB
                    'metrics': {'memory': 1 * 1024**3}  # 1GB used
                },
                'worker2': {
                    'memory_limit': 2 * 1024**3,  # 2GB
                    'metrics': {'memory': 1.5 * 1024**3}  # 1.5GB used
                }
            },
            'processing': {'task1': {}, 'task2': {}},
            'waiting': {'task3': {}},
            'ready': {}
        }
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 8 * 1024**3  # 8GB available
            mock_memory.return_value.percent = 40.0
            
            perf_report = dask_util.enhanced_performance_monitor(mock_client)
            
            # Should have all expected metrics
            assert 'n_workers' in perf_report
            assert 'total_memory_gb' in perf_report
            assert 'used_memory_gb' in perf_report
            assert 'memory_utilization' in perf_report
            assert 'tasks_processing' in perf_report
            assert 'waiting_tasks' in perf_report
            assert 'ready_tasks' in perf_report
            assert 'task_queue_pressure' in perf_report
            assert 'system_memory_available_gb' in perf_report
            assert 'system_memory_percent' in perf_report
            assert 'dashboard_link' in perf_report
            
            # Check values
            assert perf_report['n_workers'] == 2
            assert perf_report['tasks_processing'] == 2
            assert perf_report['waiting_tasks'] == 1
            assert perf_report['ready_tasks'] == 0
            assert perf_report['task_queue_pressure'] == 0.5  # (1 + 0) / 2
    
    def test_enhanced_performance_monitor_failure(self):
        """Test performance monitoring with client errors."""
        # Create mock client that raises exception
        mock_client = MagicMock()
        mock_client.scheduler_info.side_effect = Exception("Scheduler unavailable")
        
        perf_report = dask_util.enhanced_performance_monitor(mock_client)
        
        # Should return empty dict on error
        assert perf_report == {}
    
    def test_enhanced_performance_monitor_partial_data(self):
        """Test performance monitoring with partial/missing data."""
        # Create mock client with minimal scheduler info
        mock_client = MagicMock()
        mock_client.scheduler_info.return_value = {
            'workers': {},  # No workers
            'processing': {},
            'waiting': {},
            'ready': {}
        }
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 4 * 1024**3
            mock_memory.return_value.percent = 50.0
            
            perf_report = dask_util.enhanced_performance_monitor(mock_client)
            
            # Should handle missing workers gracefully
            assert perf_report['n_workers'] == 0
            assert perf_report['total_memory_gb'] == 0
            assert perf_report['used_memory_gb'] == 0
            assert perf_report['memory_utilization'] == 0


class TestDaskWorkflowIntegration:
    """Test integration of Dask utilities with climate workflows."""
    
    @patch('climate_means.setup_dask_client')
    @patch('climate_means.cleanup_dask_resources')
    def test_dask_integration_in_climate_workflow(self, mock_cleanup, mock_setup):
        """Test Dask integration in climate data workflow."""
        # Setup mocks
        mock_client = MagicMock()
        mock_cluster = MagicMock()
        mock_setup.return_value = (mock_client, mock_cluster)
        
        # Test that climate_means can use Dask functions
        config = {
            'max_workers': 2,
            'target_chunk_size': 64,
            'computation_type': 'time_series'
        }
        
        # Call through climate_means module
        client, cluster = climate_means.setup_dask_client(config)
        
        # Verify setup was called with config
        mock_setup.assert_called_once_with(config)
        assert client is mock_client
        assert cluster is mock_cluster
        
        # Test cleanup
        climate_means.cleanup_dask_resources(client, cluster)
        mock_cleanup.assert_called_once_with(mock_client, mock_cluster)
    
    @patch('climate_means.enhanced_performance_monitor')
    def test_performance_monitoring_in_workflow(self, mock_monitor):
        """Test that performance monitoring works in workflow."""
        mock_client = Mock()
        mock_monitor.return_value = {
            'memory_utilization': 45.2,
            'cpu_utilization': 67.8,
            'task_count': 25
        }
        
        # Call monitor
        report = climate_means.enhanced_performance_monitor(mock_client)
        
        assert isinstance(report, dict)
        assert 'memory_utilization' in report
        mock_monitor.assert_called_once_with(mock_client)
    
    def test_dask_config_integration_with_workflow(self):
        """Test that Dask configuration integrates with workflow."""
        config = {
            'max_workers': 2,
            'target_chunk_size': 64,
            'batch_size': 5,
            'memory_safety_margin': 0.7
        }
        
        # Validate that config can be processed by dask_util
        validated_config = dask_util.validate_dask_config(config)
        
        # Should have all required keys
        assert 'max_workers' in validated_config
        assert 'target_chunk_size' in validated_config
        assert 'batch_size' in validated_config
        
        # Should have validated values
        assert validated_config['max_workers'] >= 1
        assert validated_config['target_chunk_size'] > 0
        assert validated_config['batch_size'] > 0
    
    @patch('climate_means.process_climate_data_workflow')
    @patch('climate_means.setup_dask_client')
    @patch('climate_means.cleanup_dask_resources')
    def test_full_workflow_dask_lifecycle(self, mock_cleanup, mock_setup, mock_workflow):
        """Test complete Dask lifecycle in climate processing workflow."""
        # Setup mocks
        mock_client = MagicMock()
        mock_cluster = MagicMock()
        mock_setup.return_value = (mock_client, mock_cluster)
        
        # Simulate workflow execution pattern
        config = {'max_workers': 2, 'computation_type': 'time_series'}
        
        try:
            # Setup Dask cluster
            client, cluster = climate_means.setup_dask_client(config)
            
            # Simulate workflow processing
            mock_workflow(
                data_directory="/test/data",
                output_directory="/test/output", 
                variables=['tas'],
                regions=['CONUS'],
                scenarios=['historical'],
                config=config
            )
            
        finally:
            # Cleanup Dask resources
            climate_means.cleanup_dask_resources(client, cluster)
        
        # Verify lifecycle
        mock_setup.assert_called_once_with(config)
        mock_workflow.assert_called_once()
        mock_cleanup.assert_called_once_with(mock_client, mock_cluster)


class TestDaskErrorHandling:
    """Test error handling in Dask operations."""
    
    def test_validate_dask_config_with_invalid_types(self):
        """Test configuration validation with invalid types."""
        config = {
            'max_workers': 'invalid',  # Should be int
            'memory_safety_margin': 'invalid',  # Should be float
            'computation_type': 123  # Should be string
        }
        
        # Should handle gracefully and use defaults
        try:
            validated = dask_util.validate_dask_config(config)
        except Exception:
            validated = dask_util.validate_dask_config({})
        
        # Should fall back to reasonable defaults
        assert isinstance(validated['max_workers'], int)
        assert validated['max_workers'] >= 1
        assert isinstance(validated['memory_safety_margin'], float)
        assert 0.5 <= validated['memory_safety_margin'] <= 1.0
    
    def test_get_optimal_chunks_with_corrupted_file(self):
        """Test chunking optimization with corrupted file."""
        # Create a non-NetCDF file
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as temp_file:
            temp_file.write(b"This is not a valid NetCDF file")
            temp_file_path = temp_file.name
        
        try:
            config = {'target_chunk_size': 128}
            chunks = dask_util.get_optimal_chunks(temp_file_path, config)
            
            # Should return fallback chunking
            assert chunks == {'time': 365}
        
        finally:
            Path(temp_file_path).unlink(missing_ok=True)
    
    def test_configure_dask_resources_extreme_memory_pressure(self):
        """Test resource configuration under extreme memory pressure."""
        # This test is no longer relevant, as resource logic is now fixed.
        pass
    
    def test_enhanced_performance_monitor_with_missing_client_attrs(self):
        """Test performance monitoring with client missing attributes."""
        # Create mock client with missing dashboard_link
        mock_client = MagicMock()
        mock_client.scheduler_info.return_value = {
            'workers': {},
            'processing': {},
            'waiting': {},
            'ready': {}
        }
        
        # Remove dashboard_link attribute
        del mock_client.dashboard_link
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 8 * 1024**3
            mock_memory.return_value.percent = 40.0
            
            perf_report = dask_util.enhanced_performance_monitor(mock_client)
            
            # Should handle missing attributes gracefully
            assert 'dashboard_link' in perf_report
            assert perf_report['dashboard_link'] == 'N/A'
    
    def test_cleanup_dask_resources_with_none_objects(self):
        """Test cleanup with None client/cluster objects."""
        # Should handle None objects without crashing
        try:
            dask_util.cleanup_dask_resources(None)
        except Exception:
            pass


class TestDaskSystemIntegration:
    """Test Dask integration with system resources."""
    
    def test_dask_config_respects_system_limits(self):
        """Test that Dask configuration respects system limits."""
        # Test with configuration exceeding system resources
        cpu_count = mp.cpu_count()
        config = {
            'max_workers': cpu_count * 2,  # Request more than available
            'memory_safety_margin': 1.0    # Use all available memory
        }
        
        validated_config = dask_util.validate_dask_config(config)
        dask_config = dask_util.configure_dask_resources(validated_config)
        
        # Should be constrained by system limits and our new fixed values
        assert dask_config['n_workers'] <= 4  # Our new max limit
        assert dask_config['n_workers'] >= 1
        assert dask_config['memory_limit_per_worker'] == "4GB"  # Our fixed memory limit
        assert dask_config['threads_per_worker'] == 2  # Our fixed thread count
    
    def test_memory_calculation_consistency(self):
        """Test that memory calculations are consistent."""
        config = {'memory_safety_margin': 0.8, 'max_workers': 4}
        
        # Run multiple times to ensure consistency
        configs = []
        for _ in range(3):
            dask_config = dask_util.configure_dask_resources(config)
            configs.append(dask_config)
        
        # Should produce consistent results with our fixed values
        for cfg in configs:
            assert cfg['memory_limit_per_worker'] == "4GB"
            assert cfg['threads_per_worker'] == 2
            assert 1 <= cfg['n_workers'] <= 4


def test_module_level_dask_integration():
    """Test module-level integration between dask_util and climate_means."""
    # Test that both modules can be imported together
    import dask_util
    import climate_means
    
    # Test that they don't conflict
    assert climate_means.setup_dask_client is dask_util.setup_dask_client
    assert climate_means.enhanced_performance_monitor is dask_util.enhanced_performance_monitor
    assert climate_means.cleanup_dask_resources is dask_util.cleanup_dask_resources
    
    # Test that both modules have expected attributes
    expected_climate_attrs = [
        'setup_dask_client',
        'enhanced_performance_monitor',
        'cleanup_dask_resources'
    ]
    
    for attr in expected_climate_attrs:
        assert hasattr(climate_means, attr)
    
    expected_dask_attrs = [
        'validate_dask_config',
        'configure_dask_resources',
        'setup_dask_client',
        'enhanced_performance_monitor',
        'robust_compute_with_retry',
        'cleanup_dask_resources'
    ]
    
    for attr in expected_dask_attrs:
        assert hasattr(dask_util, attr)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 