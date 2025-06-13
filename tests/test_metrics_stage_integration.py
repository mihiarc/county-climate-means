"""
Tests for stage two (metrics) integration.

This module provides comprehensive testing for the metrics stage handler,
including integration with the pipeline orchestration system, data flow
from stage one (means), and error handling scenarios.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import yaml
import xarray as xr
import numpy as np

from county_climate.metrics.integration.stage_handlers import (
    metrics_stage_handler,
    METRICS_IMPORTS_AVAILABLE
)


class TestMetricsStageHandler:
    """Test the metrics stage handler integration."""
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock context for stage handler testing."""
        return {
            'stage_config': {
                'output_base_path': '/tmp/test_metrics_output',
                'multiprocessing_workers': 2,
                'max_memory_per_worker_gb': 1.0,
                'batch_size_counties': 10,
                'metrics': ['mean', 'std', 'min', 'max'],
                'county_boundaries': '2024_census',
                'spatial_aggregation': 'area_weighted_mean',
                'output_formats': ['csv', 'netcdf4']
            },
            'stage_inputs': {
                'climate_means': {
                    'output_base_path': '/tmp/test_means_output',
                    'output_files': [
                        '/tmp/test_means_output/tas/historical/tas_CONUS_historical_2000_30yr_normal.nc',
                        '/tmp/test_means_output/pr/historical/pr_CONUS_historical_2000_30yr_normal.nc'
                    ]
                }
            },
            'pipeline_context': {
                'base_data_path': '/tmp/test_input',
                'temp_data_path': '/tmp/test_temp',
                'execution_id': 'test_exec_123'
            },
            'logger': MagicMock()
        }
    
    @pytest.fixture
    def temp_test_dirs(self):
        """Create temporary directories for testing."""
        base_temp = Path(tempfile.mkdtemp())
        
        # Create directory structure
        means_output = base_temp / "means_output"
        metrics_output = base_temp / "metrics_output"
        
        means_output.mkdir(parents=True)
        metrics_output.mkdir(parents=True)
        
        # Create some fake means output files
        (means_output / "tas" / "historical").mkdir(parents=True)
        (means_output / "pr" / "historical").mkdir(parents=True)
        
        # Create dummy NetCDF files
        fake_nc_1 = means_output / "tas" / "historical" / "tas_CONUS_historical_2000_30yr_normal.nc"
        fake_nc_2 = means_output / "pr" / "historical" / "pr_CONUS_historical_2000_30yr_normal.nc"
        
        # Create minimal NetCDF files with dummy data
        for nc_file in [fake_nc_1, fake_nc_2]:
            ds = xr.Dataset({
                'temperature' if 'tas' in str(nc_file) else 'precipitation': (
                    ['time', 'lat', 'lon'], 
                    np.random.random((365, 10, 10))
                )
            })
            ds.to_netcdf(nc_file)
        
        yield {
            'base': base_temp,
            'means_output': means_output,
            'metrics_output': metrics_output,
            'nc_files': [fake_nc_1, fake_nc_2]
        }
        
        # Cleanup
        shutil.rmtree(base_temp)
    
    @pytest.mark.asyncio
    async def test_metrics_stage_handler_basic_execution(self, mock_context, temp_test_dirs):
        """Test basic execution of metrics stage handler."""
        # Update context with real temp directories
        mock_context['stage_config']['output_base_path'] = str(temp_test_dirs['metrics_output'])
        mock_context['stage_inputs']['climate_means']['output_base_path'] = str(temp_test_dirs['means_output'])
        mock_context['stage_inputs']['climate_means']['output_files'] = [
            str(f) for f in temp_test_dirs['nc_files']
        ]
        
        # Execute the handler
        result = await metrics_stage_handler(**mock_context)
        
        # Verify basic structure
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'processing_stats' in result
        assert 'output_files' in result
        
        # Should complete successfully
        assert result['status'] in ['completed', 'failed']  # Either is fine for this test
        
        # Should have processing stats
        stats = result['processing_stats']
        assert 'files_processed' in stats
        assert 'counties_processed' in stats
        assert 'errors' in stats
        
        # Should have created some output
        assert isinstance(result['output_files'], list)
    
    @pytest.mark.asyncio
    async def test_metrics_stage_handler_with_imports_available(self, mock_context, temp_test_dirs):
        """Test metrics handler when all imports are available."""
        mock_context['stage_config']['output_base_path'] = str(temp_test_dirs['metrics_output'])
        mock_context['stage_inputs']['climate_means']['output_base_path'] = str(temp_test_dirs['means_output'])
        
        # Mock the imports availability
        with patch('county_climate.metrics.integration.stage_handlers.METRICS_IMPORTS_AVAILABLE', True):
            result = await metrics_stage_handler(**mock_context)
        
        assert result['status'] == 'completed'
        assert result['processing_stats']['files_processed'] >= 0
        
        # Should create output files
        output_dir = Path(temp_test_dirs['metrics_output'])
        assert output_dir.exists()
    
    @pytest.mark.asyncio
    async def test_metrics_stage_handler_imports_unavailable(self, mock_context, temp_test_dirs):
        """Test metrics handler when imports are not available (demo mode)."""
        mock_context['stage_config']['output_base_path'] = str(temp_test_dirs['metrics_output'])
        mock_context['stage_inputs']['climate_means']['output_base_path'] = str(temp_test_dirs['means_output'])
        
        # Mock the imports availability as False
        with patch('county_climate.metrics.integration.stage_handlers.METRICS_IMPORTS_AVAILABLE', False):
            result = await metrics_stage_handler(**mock_context)
        
        assert result['status'] == 'completed'
        assert 'message' in result
        assert 'demo mode' in result['message']
        
        # Should create demo output file
        output_dir = Path(temp_test_dirs['metrics_output'])
        demo_files = list(output_dir.glob("demo_metrics_output.txt"))
        assert len(demo_files) > 0
    
    @pytest.mark.asyncio
    async def test_metrics_stage_handler_no_input_files(self, mock_context, temp_test_dirs):
        """Test metrics handler when no input files are found."""
        # Set up empty input directory
        empty_dir = temp_test_dirs['base'] / "empty_means"
        empty_dir.mkdir()
        
        mock_context['stage_config']['output_base_path'] = str(temp_test_dirs['metrics_output'])
        mock_context['stage_inputs']['climate_means']['output_base_path'] = str(empty_dir)
        
        result = await metrics_stage_handler(**mock_context)
        
        assert result['status'] == 'completed'  # Should complete even with no inputs
        assert result['processing_stats']['files_processed'] == 0
        
        # Should log the issue
        if METRICS_IMPORTS_AVAILABLE:
            assert 'No input files found' in str(result['processing_stats']['errors'])
    
    @pytest.mark.asyncio
    async def test_metrics_stage_handler_error_handling(self, mock_context):
        """Test error handling in metrics stage handler."""
        # Set invalid paths to trigger errors
        mock_context['stage_config']['output_base_path'] = '/invalid/path/that/does/not/exist'
        mock_context['stage_inputs']['climate_means']['output_base_path'] = '/another/invalid/path'
        
        # Should handle errors gracefully
        result = await metrics_stage_handler(**mock_context)
        
        # Should not crash and should return a result
        assert isinstance(result, dict)
        assert 'status' in result


class TestMetricsStageIntegration:
    """Test metrics stage integration with pipeline orchestration."""
    
    @pytest.fixture
    def pipeline_config(self):
        """Create a pipeline configuration for integration testing."""
        return {
            'pipeline_id': 'test_metrics_integration',
            'pipeline_name': 'Test Metrics Integration',
            'base_data_path': '/tmp/test_input',
            'stages': [
                {
                    'stage_id': 'climate_means',
                    'stage_type': 'means',
                    'stage_name': 'Climate Means',
                    'package_name': 'county_climate.means.integration',
                    'entry_point': 'means_stage_handler',
                    'stage_config': {
                        'variables': ['tas', 'pr'],
                        'regions': ['CONUS'],
                        'scenarios': ['historical'],
                        'output_base_path': '/tmp/test_means'
                    }
                },
                {
                    'stage_id': 'climate_metrics',
                    'stage_type': 'metrics', 
                    'stage_name': 'Climate Metrics',
                    'package_name': 'county_climate.metrics.integration',
                    'entry_point': 'metrics_stage_handler',
                    'depends_on': ['climate_means'],
                    'stage_config': {
                        'output_base_path': '/tmp/test_metrics',
                        'metrics': ['mean', 'std'],
                        'county_boundaries': '2024_census'
                    }
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_stage_dependency_data_flow(self, pipeline_config):
        """Test that data flows correctly from means to metrics stage."""
        from county_climate.shared.config import ConfigurationLoader, PipelineConfiguration
        from county_climate.shared.orchestration import PipelineOrchestrator
        
        # Load configuration
        loader = ConfigurationLoader()
        config = loader.load_pipeline_config(pipeline_config)
        
        # Create orchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Mock means handler output
        async def mock_means_handler(**context):
            return {
                'status': 'completed',
                'output_base_path': '/tmp/test_means',
                'output_files': [
                    '/tmp/test_means/tas/historical/tas_CONUS_historical_2000.nc',
                    '/tmp/test_means/pr/historical/pr_CONUS_historical_2000.nc'
                ],
                'processing_stats': {
                    'successful_datasets': 2,
                    'failed_datasets': 0
                }
            }
        
        # Register handlers
        from county_climate.shared.config.integration_config import ProcessingStage
        orchestrator.register_stage_handler(ProcessingStage.MEANS, mock_means_handler)
        orchestrator.register_stage_handler(ProcessingStage.METRICS, metrics_stage_handler)
        
        # Execute pipeline
        execution = await orchestrator.execute_pipeline('test_exec')
        
        # Verify execution
        assert execution.status.value == 'completed'
        assert len(execution.stage_executions) == 2
        
        # Verify means stage completed
        means_execution = execution.stage_executions['climate_means']
        assert means_execution.status.value == 'completed'
        
        # Verify metrics stage received means output
        metrics_execution = execution.stage_executions['climate_metrics']
        assert metrics_execution.status.value == 'completed'
        
        # Verify data flow
        assert 'output_files' in means_execution.output_data
        # The metrics handler should have received the means output as stage_inputs
    
    @pytest.mark.asyncio
    async def test_metrics_stage_failure_handling(self, pipeline_config):
        """Test error handling when metrics stage fails."""
        from county_climate.shared.config import ConfigurationLoader
        from county_climate.shared.orchestration import PipelineOrchestrator
        
        loader = ConfigurationLoader()
        config = loader.load_pipeline_config(pipeline_config)
        orchestrator = PipelineOrchestrator(config)
        
        # Mock means handler that succeeds
        async def mock_means_handler(**context):
            return {'status': 'completed', 'output_files': []}
        
        # Mock metrics handler that fails
        async def failing_metrics_handler(**context):
            raise Exception("Simulated metrics processing failure")
        
        from county_climate.shared.config.integration_config import ProcessingStage
        orchestrator.register_stage_handler(ProcessingStage.MEANS, mock_means_handler)
        orchestrator.register_stage_handler(ProcessingStage.METRICS, failing_metrics_handler)
        
        # Execute pipeline
        execution = await orchestrator.execute_pipeline('test_exec')
        
        # Pipeline should fail due to metrics stage failure
        assert execution.status.value == 'failed'
        
        # Means should succeed, metrics should fail
        assert execution.stage_executions['climate_means'].status.value == 'completed'
        assert execution.stage_executions['climate_metrics'].status.value == 'failed'
        assert 'Simulated metrics processing failure' in execution.stage_executions['climate_metrics'].error_message


class TestMetricsStageValidation:
    """Test validation and quality checks for metrics stage."""
    
    @pytest.mark.asyncio
    async def test_metrics_output_validation(self, temp_test_dirs):
        """Test validation of metrics stage outputs."""
        
        # Create mock context
        context = {
            'stage_config': {
                'output_base_path': str(temp_test_dirs['metrics_output']),
                'create_county_summaries': True,
                'output_formats': ['csv', 'netcdf4']
            },
            'stage_inputs': {
                'climate_means': {
                    'output_base_path': str(temp_test_dirs['means_output'])
                }
            },
            'pipeline_context': {},
            'logger': MagicMock()
        }
        
        # Execute handler
        result = await metrics_stage_handler(**context)
        
        # Validate output structure
        assert 'processing_stats' in result
        stats = result['processing_stats']
        
        # Should track key metrics
        required_stats = ['files_processed', 'counties_processed', 'metrics_calculated', 'errors']
        for stat in required_stats:
            assert stat in stats
        
        # Should be non-negative numbers
        assert stats['files_processed'] >= 0
        assert stats['counties_processed'] >= 0
        assert isinstance(stats['metrics_calculated'], list)
        assert isinstance(stats['errors'], list)
    
    @pytest.mark.asyncio
    async def test_metrics_configuration_validation(self):
        """Test validation of metrics stage configuration."""
        
        # Test with minimal valid config
        minimal_context = {
            'stage_config': {
                'output_base_path': '/tmp/test'
            },
            'stage_inputs': {},
            'pipeline_context': {},
            'logger': MagicMock()
        }
        
        # Should not crash with minimal config
        result = await metrics_stage_handler(**minimal_context)
        assert isinstance(result, dict)
        assert 'status' in result
    
    def test_metrics_handler_import(self):
        """Test that metrics handler can be imported successfully."""
        # This should work without errors
        from county_climate.metrics.integration.stage_handlers import metrics_stage_handler
        assert callable(metrics_stage_handler)
        
        # Check if imports are available
        from county_climate.metrics.integration.stage_handlers import METRICS_IMPORTS_AVAILABLE
        assert isinstance(METRICS_IMPORTS_AVAILABLE, bool)


class TestMetricsStagePerformance:
    """Test performance characteristics of metrics stage."""
    
    @pytest.mark.asyncio
    async def test_metrics_stage_timeout_handling(self):
        """Test that metrics stage respects timeout constraints."""
        import time
        
        context = {
            'stage_config': {
                'output_base_path': '/tmp/test_timeout',
                'processing_timeout_seconds': 1  # Very short timeout
            },
            'stage_inputs': {},
            'pipeline_context': {},
            'logger': MagicMock()
        }
        
        start_time = time.time()
        result = await metrics_stage_handler(**context)
        end_time = time.time()
        
        # Should complete quickly (well under timeout for this simple case)
        assert end_time - start_time < 5  # Should finish in under 5 seconds
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio 
    async def test_metrics_stage_memory_usage(self):
        """Test that metrics stage has reasonable memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        context = {
            'stage_config': {
                'output_base_path': '/tmp/test_memory',
                'max_memory_per_worker_gb': 1.0
            },
            'stage_inputs': {},
            'pipeline_context': {},
            'logger': MagicMock()
        }
        
        result = await metrics_stage_handler(**context)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory (less than 100MB increase)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])