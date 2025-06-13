"""
End-to-end pipeline integration tests.

This module tests the complete pipeline from configuration loading through
all three stages (means, metrics, validation) to ensure proper integration
and data flow between stages.
"""

import asyncio
import pytest
import tempfile
import shutil
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
import xarray as xr
import numpy as np

from county_climate.shared.config import ConfigurationLoader, PipelineConfiguration
from county_climate.shared.orchestration import PipelineOrchestrator
from county_climate.shared.config.integration_config import ProcessingStage


class TestEndToEndPipeline:
    """Test complete pipeline execution with all three stages."""
    
    @pytest.fixture
    def temp_pipeline_dirs(self):
        """Create temporary directory structure for pipeline testing."""
        base_temp = Path(tempfile.mkdtemp())
        
        # Create directory structure
        input_dir = base_temp / "input"
        means_output = base_temp / "means_output" 
        metrics_output = base_temp / "metrics_output"
        validation_output = base_temp / "validation_output"
        
        for d in [input_dir, means_output, metrics_output, validation_output]:
            d.mkdir(parents=True)
        
        yield {
            'base': base_temp,
            'input': input_dir,
            'means_output': means_output,
            'metrics_output': metrics_output,
            'validation_output': validation_output
        }
        
        # Cleanup
        shutil.rmtree(base_temp)
    
    @pytest.fixture
    def full_pipeline_config(self, temp_pipeline_dirs):
        """Create a complete pipeline configuration for testing."""
        return {
            'pipeline_id': 'test_full_pipeline',
            'pipeline_name': 'Test Full Pipeline',
            'pipeline_version': '1.0.0',
            'environment': 'development',
            'base_data_path': str(temp_pipeline_dirs['input']),
            'global_resource_limits': {
                'max_memory_gb': 4.0,
                'max_cpu_cores': 2,
                'max_processing_time_hours': 1.0
            },
            'stages': [
                {
                    'stage_id': 'climate_means',
                    'stage_type': 'means',
                    'stage_name': 'Climate Means Processing',
                    'package_name': 'county_climate.means.integration',
                    'entry_point': 'means_stage_handler',
                    'stage_config': {
                        'variables': ['tas'],
                        'regions': ['CONUS'],
                        'scenarios': ['historical'],
                        'multiprocessing_workers': 2,
                        'output_base_path': str(temp_pipeline_dirs['means_output']),
                        'enable_rich_progress': False
                    }
                },
                {
                    'stage_id': 'climate_metrics',
                    'stage_type': 'metrics',
                    'stage_name': 'Climate Metrics Processing',
                    'package_name': 'county_climate.metrics.integration',
                    'entry_point': 'metrics_stage_handler',
                    'depends_on': ['climate_means'],
                    'stage_config': {
                        'output_base_path': str(temp_pipeline_dirs['metrics_output']),
                        'metrics': ['mean', 'std', 'min', 'max'],
                        'county_boundaries': '2024_census'
                    }
                },
                {
                    'stage_id': 'validation',
                    'stage_type': 'validation',
                    'stage_name': 'Pipeline Validation',
                    'package_name': 'county_climate.shared.validation',
                    'entry_point': 'validate_complete_pipeline',
                    'depends_on': ['climate_metrics'],
                    'stage_config': {
                        'means_output_path': str(temp_pipeline_dirs['means_output']),
                        'metrics_output_path': str(temp_pipeline_dirs['metrics_output']),
                        'validation_report_path': str(temp_pipeline_dirs['validation_output']),
                        'create_validation_report': True,
                        'validation_checks': ['data_completeness', 'file_integrity']
                    }
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_full_pipeline_execution_with_mocks(self, full_pipeline_config, temp_pipeline_dirs):
        """Test complete pipeline execution with mocked stage handlers."""
        
        # Load configuration
        loader = ConfigurationLoader()
        config = loader.load_pipeline_config(full_pipeline_config)
        
        # Create orchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Mock stage handlers
        async def mock_means_handler(**context):
            # Create fake output files
            output_dir = Path(context['stage_config']['output_base_path'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create fake NetCDF files
            fake_files = []
            for var in ['tas']:
                var_dir = output_dir / var / 'historical'
                var_dir.mkdir(parents=True, exist_ok=True)
                fake_file = var_dir / f'{var}_CONUS_historical_2000_30yr_normal.nc'
                
                # Create dummy NetCDF
                ds = xr.Dataset({
                    var: (['time', 'lat', 'lon'], np.random.random((365, 50, 50)))
                })
                ds.to_netcdf(fake_file)
                fake_files.append(str(fake_file))
            
            return {
                'status': 'completed',
                'output_base_path': str(output_dir),
                'output_files': fake_files,
                'processing_stats': {
                    'successful_datasets': 1,
                    'failed_datasets': 0,
                    'total_processing_time': 120.5
                }
            }
        
        # Import real handlers
        from county_climate.metrics.integration.stage_handlers import metrics_stage_handler
        from county_climate.shared.validation.pipeline_validator import validate_complete_pipeline
        
        # Register handlers
        orchestrator.register_stage_handler(ProcessingStage.MEANS, mock_means_handler)
        orchestrator.register_stage_handler(ProcessingStage.METRICS, metrics_stage_handler)
        orchestrator.register_stage_handler(ProcessingStage.VALIDATION, validate_complete_pipeline)
        
        # Execute pipeline
        execution = await orchestrator.execute_pipeline('test_full_exec')
        
        # Verify execution completed
        assert execution.status.value == 'completed'
        assert len(execution.stage_executions) == 3
        
        # Verify each stage completed
        means_exec = execution.stage_executions['climate_means']
        metrics_exec = execution.stage_executions['climate_metrics'] 
        validation_exec = execution.stage_executions['validation']
        
        assert means_exec.status.value == 'completed'
        assert metrics_exec.status.value == 'completed'
        assert validation_exec.status.value == 'completed'
        
        # Verify data flow
        assert 'output_files' in means_exec.output_data
        assert 'processing_stats' in metrics_exec.output_data
        assert 'validation_passed' in validation_exec.output_data
        
        # Verify files were created
        means_files = list(temp_pipeline_dirs['means_output'].rglob('*.nc'))
        assert len(means_files) > 0
        
        # Metrics should create some output
        metrics_files = list(temp_pipeline_dirs['metrics_output'].glob('*'))
        assert len(metrics_files) > 0
        
        # Validation should create a report
        validation_files = list(temp_pipeline_dirs['validation_output'].glob('*.txt'))
        assert len(validation_files) > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_failure_propagation(self, full_pipeline_config):
        """Test that failures in one stage properly stop the pipeline."""
        
        loader = ConfigurationLoader()
        config = loader.load_pipeline_config(full_pipeline_config)
        orchestrator = PipelineOrchestrator(config)
        
        # Mock failing means handler
        async def failing_means_handler(**context):
            raise Exception("Simulated means processing failure")
        
        # Import real handlers for other stages
        from county_climate.metrics.integration.stage_handlers import metrics_stage_handler
        from county_climate.shared.validation.pipeline_validator import validate_complete_pipeline
        
        # Register handlers
        orchestrator.register_stage_handler(ProcessingStage.MEANS, failing_means_handler)
        orchestrator.register_stage_handler(ProcessingStage.METRICS, metrics_stage_handler) 
        orchestrator.register_stage_handler(ProcessingStage.VALIDATION, validate_complete_pipeline)
        
        # Execute pipeline
        execution = await orchestrator.execute_pipeline('test_failure_exec')
        
        # Pipeline should fail
        assert execution.status.value == 'failed'
        
        # First stage should fail
        assert execution.stage_executions['climate_means'].status.value == 'failed'
        assert 'Simulated means processing failure' in execution.stage_executions['climate_means'].error_message
        
        # Dependent stages should be skipped
        assert execution.stage_executions['climate_metrics'].status.value == 'skipped'
        assert execution.stage_executions['validation'].status.value == 'skipped'
    
    @pytest.mark.asyncio
    async def test_pipeline_partial_failure_handling(self, full_pipeline_config, temp_pipeline_dirs):
        """Test pipeline behavior when middle stage fails."""
        
        loader = ConfigurationLoader()
        config = loader.load_pipeline_config(full_pipeline_config)
        orchestrator = PipelineOrchestrator(config)
        
        # Mock successful means handler
        async def successful_means_handler(**context):
            return {
                'status': 'completed',
                'output_files': ['fake_file.nc'],
                'processing_stats': {'successful_datasets': 1}
            }
        
        # Mock failing metrics handler
        async def failing_metrics_handler(**context):
            raise Exception("Simulated metrics processing failure")
        
        # Import real validation handler
        from county_climate.shared.validation.pipeline_validator import validate_complete_pipeline
        
        # Register handlers
        orchestrator.register_stage_handler(ProcessingStage.MEANS, successful_means_handler)
        orchestrator.register_stage_handler(ProcessingStage.METRICS, failing_metrics_handler)
        orchestrator.register_stage_handler(ProcessingStage.VALIDATION, validate_complete_pipeline)
        
        # Execute pipeline
        execution = await orchestrator.execute_pipeline('test_partial_failure_exec')
        
        # Pipeline should fail overall
        assert execution.status.value == 'failed'
        
        # First stage should succeed
        assert execution.stage_executions['climate_means'].status.value == 'completed'
        
        # Second stage should fail
        assert execution.stage_executions['climate_metrics'].status.value == 'failed'
        assert 'Simulated metrics processing failure' in execution.stage_executions['climate_metrics'].error_message
        
        # Third stage should be skipped
        assert execution.stage_executions['validation'].status.value == 'skipped'
    
    @pytest.mark.asyncio
    async def test_pipeline_configuration_validation(self, temp_pipeline_dirs):
        """Test pipeline configuration validation catches issues."""
        
        # Create invalid configuration (missing dependency)
        invalid_config = {
            'pipeline_id': 'invalid_pipeline',
            'pipeline_name': 'Invalid Pipeline',
            'base_data_path': str(temp_pipeline_dirs['input']),
            'stages': [
                {
                    'stage_id': 'climate_metrics',
                    'stage_type': 'metrics',
                    'stage_name': 'Orphaned Metrics',
                    'package_name': 'county_climate.metrics.integration',
                    'entry_point': 'metrics_stage_handler',
                    'depends_on': ['nonexistent_stage']  # Invalid dependency
                }
            ]
        }
        
        # Should raise validation error
        loader = ConfigurationLoader()
        with pytest.raises(ValueError, match="depends on nonexistent_stage"):
            loader.load_pipeline_config(invalid_config)
    
    @pytest.mark.asyncio
    async def test_pipeline_execution_timing(self, full_pipeline_config):
        """Test that pipeline execution timing is properly tracked."""
        
        loader = ConfigurationLoader()
        config = loader.load_pipeline_config(full_pipeline_config)
        orchestrator = PipelineOrchestrator(config)
        
        # Mock handlers with artificial delays
        async def slow_means_handler(**context):
            await asyncio.sleep(0.1)  # 100ms delay
            return {'status': 'completed', 'output_files': []}
        
        async def slow_metrics_handler(**context):
            await asyncio.sleep(0.05)  # 50ms delay
            return {'status': 'completed', 'processing_stats': {'files_processed': 0}}
        
        async def slow_validation_handler(**context):
            await asyncio.sleep(0.02)  # 20ms delay
            return {'status': 'completed', 'validation_passed': True}
        
        # Register handlers
        orchestrator.register_stage_handler(ProcessingStage.MEANS, slow_means_handler)
        orchestrator.register_stage_handler(ProcessingStage.METRICS, slow_metrics_handler)
        orchestrator.register_stage_handler(ProcessingStage.VALIDATION, slow_validation_handler)
        
        # Execute pipeline
        execution = await orchestrator.execute_pipeline('test_timing_exec')
        
        # Verify timing information
        assert execution.status.value == 'completed'
        assert execution.start_time is not None
        assert execution.end_time is not None
        assert execution.end_time > execution.start_time
        
        # Each stage should have timing info
        for stage_id in ['climate_means', 'climate_metrics', 'validation']:
            stage_exec = execution.stage_executions[stage_id]
            assert stage_exec.start_time is not None
            assert stage_exec.end_time is not None
            assert stage_exec.duration_seconds is not None
            assert stage_exec.duration_seconds > 0
    
    def test_pipeline_config_file_loading(self, full_pipeline_config, temp_pipeline_dirs):
        """Test loading pipeline configuration from YAML file."""
        
        # Write config to file
        config_file = temp_pipeline_dirs['base'] / 'test_pipeline.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(full_pipeline_config, f)
        
        # Load configuration
        loader = ConfigurationLoader()
        config = loader.load_pipeline_config(config_file)
        
        # Verify configuration
        assert config.pipeline_id == 'test_full_pipeline'
        assert len(config.stages) == 3
        
        # Verify execution order
        execution_order = config.get_execution_order()
        assert len(execution_order) == 3
        assert 'climate_means' in execution_order[0]
        assert 'climate_metrics' in execution_order[1]
        assert 'validation' in execution_order[2]
    
    @pytest.mark.asyncio
    async def test_pipeline_resource_limits(self, full_pipeline_config):
        """Test that pipeline respects resource limits."""
        
        # Modify config with strict resource limits
        full_pipeline_config['global_resource_limits']['max_processing_time_hours'] = 0.001  # Very short
        full_pipeline_config['stages'][0]['resource_limits'] = {
            'max_memory_gb': 0.1,  # Very low
            'max_cpu_cores': 1,
            'max_processing_time_hours': 0.001
        }
        
        loader = ConfigurationLoader()
        config = loader.load_pipeline_config(full_pipeline_config)
        orchestrator = PipelineOrchestrator(config)
        
        # Mock fast handlers
        async def fast_handler(**context):
            return {'status': 'completed'}
        
        # Register handlers
        orchestrator.register_stage_handler(ProcessingStage.MEANS, fast_handler)
        orchestrator.register_stage_handler(ProcessingStage.METRICS, fast_handler)
        orchestrator.register_stage_handler(ProcessingStage.VALIDATION, fast_handler)
        
        # Execute pipeline - should complete quickly
        execution = await orchestrator.execute_pipeline('test_resource_exec')
        
        # Should complete (handlers are fast enough)
        assert execution.status.value == 'completed'
        
        # Verify resource limits are accessible in configuration
        stage_config = config.get_stage_by_id('climate_means')
        assert stage_config.resource_limits.max_memory_gb == 0.1
        assert stage_config.resource_limits.max_cpu_cores == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])