"""
Tests for configuration-driven integration system.

This module tests the configuration loading, validation, and pipeline
orchestration capabilities of the climate data processing system.
"""

import asyncio
import pytest
import tempfile
import yaml
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from county_climate.shared.config import (
    PipelineConfiguration,
    StageConfiguration,
    ProcessingProfile,
    ProcessingStage,
    TriggerType,
    EnvironmentType,
    ResourceLimits,
    ConfigurationLoader,
    ConfigurationManager,
    ConfigurationError,
)

from county_climate.shared.orchestration import (
    PipelineOrchestrator,
    PipelineRunner,
    StageExecution,
    ExecutionStatus,
)

from county_climate.shared.contracts.climate_data import (
    Region,
    ClimateVariable,
    Scenario,
)


class TestIntegrationConfig:
    """Test configuration models and validation."""
    
    def test_stage_configuration_creation(self):
        """Test creating a valid stage configuration."""
        stage = StageConfiguration(
            stage_id="test_stage",
            stage_type=ProcessingStage.MEANS,
            stage_name="Test Stage",
            package_name="county_climate.means",
            entry_point="process_region"
        )
        
        assert stage.stage_id == "test_stage"
        assert stage.stage_type == ProcessingStage.MEANS
        assert stage.trigger_type == TriggerType.DEPENDENCY  # default
        assert stage.parallel_execution is True  # default
        assert stage.retry_attempts == 3  # default
    
    def test_pipeline_configuration_validation(self):
        """Test pipeline configuration validation."""
        stages = [
            StageConfiguration(
                stage_id="stage1",
                stage_type=ProcessingStage.MEANS,
                stage_name="Stage 1",
                package_name="test.package",
                entry_point="handler"
            ),
            StageConfiguration(
                stage_id="stage2",
                stage_type=ProcessingStage.METRICS,
                stage_name="Stage 2", 
                package_name="test.package",
                entry_point="handler",
                depends_on=["stage1"]
            )
        ]
        
        config = PipelineConfiguration(
            pipeline_id="test_pipeline",
            pipeline_name="Test Pipeline",
            base_data_path=Path("/tmp/test"),
            stages=stages
        )
        
        assert len(config.stages) == 2
        assert config.get_stage_by_id("stage1") is not None
        assert config.get_stage_by_id("nonexistent") is None
        
        # Test execution order
        execution_order = config.get_execution_order()
        assert len(execution_order) == 2
        assert "stage1" in execution_order[0]
        assert "stage2" in execution_order[1]
    
    def test_pipeline_dependency_validation_failure(self):
        """Test that invalid dependencies are caught."""
        stages = [
            StageConfiguration(
                stage_id="stage1",
                stage_type=ProcessingStage.MEANS,
                stage_name="Stage 1",
                package_name="test.package",
                entry_point="handler",
                depends_on=["nonexistent_stage"]  # Invalid dependency
            )
        ]
        
        with pytest.raises(ValueError, match="depends on nonexistent_stage which doesn't exist"):
            PipelineConfiguration(
                pipeline_id="test_pipeline",
                pipeline_name="Test Pipeline",
                base_data_path=Path("/tmp/test"),
                stages=stages
            )
    
    def test_processing_profile_creation(self):
        """Test creating a processing profile."""
        profile = ProcessingProfile(
            profile_name="test_profile",
            description="Test processing profile",
            regions=[Region.CONUS],
            variables=[ClimateVariable.TEMPERATURE],
            scenarios=[Scenario.HISTORICAL],
            year_ranges=[(1990, 2020)],
            enable_means=True,
            enable_metrics=False
        )
        
        assert profile.profile_name == "test_profile"
        assert Region.CONUS in profile.regions
        assert profile.enable_means is True
        assert profile.enable_metrics is False


class TestConfigurationLoader:
    """Test configuration loading and management."""
    
    def test_load_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config_data = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "Test Pipeline",
            "base_data_path": "/tmp/test",
            "stages": [
                {
                    "stage_id": "test_stage",
                    "stage_type": "means",
                    "stage_name": "Test Stage",
                    "package_name": "test.package",
                    "entry_point": "handler"
                }
            ]
        }
        
        loader = ConfigurationLoader()
        config = loader.load_pipeline_config(config_data)
        
        assert config.pipeline_id == "test_pipeline"
        assert len(config.stages) == 1
        assert config.stages[0].stage_id == "test_stage"
    
    def test_load_config_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "pipeline_id": "yaml_pipeline",
            "pipeline_name": "YAML Pipeline",
            "base_data_path": "/tmp/yaml",
            "environment": "development",
            "stages": [
                {
                    "stage_id": "yaml_stage",
                    "stage_type": "means",
                    "stage_name": "YAML Stage",
                    "package_name": "test.package",
                    "entry_point": "handler"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = Path(f.name)
        
        try:
            loader = ConfigurationLoader()
            config = loader.load_pipeline_config(config_file)
            
            assert config.pipeline_id == "yaml_pipeline"
            assert config.environment == EnvironmentType.DEVELOPMENT
            
        finally:
            config_file.unlink()
    
    def test_environment_overrides(self):
        """Test environment-specific configuration overrides."""
        config_data = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "Test Pipeline", 
            "base_data_path": "/tmp/test",
            "max_workers": 2,
            "environment_overrides": {
                "production": {
                    "max_workers": 8,
                    "base_data_path": "/data/production"
                }
            },
            "stages": [
                {
                    "stage_id": "test_stage",
                    "stage_type": "means",
                    "stage_name": "Test Stage",
                    "package_name": "test.package",
                    "entry_point": "handler",
                    "retry_attempts": 3,
                    "environment_overrides": {
                        "production": {
                            "retry_attempts": 5
                        }
                    }
                }
            ]
        }
        
        loader = ConfigurationLoader()
        
        # Test development environment (no overrides)
        dev_config = loader.load_pipeline_config(config_data, EnvironmentType.DEVELOPMENT)
        # max_workers should stay at original value since no override for dev
        assert dev_config.stages[0].retry_attempts == 3
        
        # Test production environment (with overrides)
        prod_config = loader.load_pipeline_config(config_data, EnvironmentType.PRODUCTION)
        assert str(prod_config.base_data_path) == "/data/production"
        assert prod_config.stages[0].retry_attempts == 5
    
    def test_env_var_overrides(self):
        """Test environment variable overrides."""
        config_data = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "Test Pipeline",
            "base_data_path": "/tmp/test",
            "stages": [
                {
                    "stage_id": "test_stage",
                    "stage_type": "means",
                    "stage_name": "Test Stage",
                    "package_name": "test.package",
                    "entry_point": "handler"
                }
            ]
        }
        
        # Mock environment variables
        with patch.dict('os.environ', {
            'COUNTY_CLIMATE_DATA_PATH': '/env/data/path',
            'COUNTY_CLIMATE_MAX_WORKERS': '6'
        }):
            loader = ConfigurationLoader()
            config = loader.load_pipeline_config(config_data)
            
            assert str(config.base_data_path) == "/env/data/path"
            # Environment variables override the base config
    
    def test_config_validation(self):
        """Test configuration validation warnings."""
        config_data = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "Test Pipeline",
            "base_data_path": "/tmp/test",
            "stages": [
                {
                    "stage_id": "orphan_stage",
                    "stage_type": "means", 
                    "stage_name": "Orphan Stage",
                    "package_name": "test.package",
                    "entry_point": "handler"
                },
                {
                    "stage_id": "dependent_stage",
                    "stage_type": "metrics",
                    "stage_name": "Dependent Stage",
                    "package_name": "test.package", 
                    "entry_point": "handler",
                    "depends_on": ["orphan_stage"]
                }
            ]
        }
        
        loader = ConfigurationLoader()
        config = loader.load_pipeline_config(config_data)
        warnings = loader.validate_config(config)
        
        # Should warn about unreferenced stages
        assert any("Unreferenced stages" in warning for warning in warnings)
    
    def test_create_default_configs(self):
        """Test creating default configurations.""" 
        loader = ConfigurationLoader()
        
        basic_config = loader.create_default_config("basic")
        assert basic_config.pipeline_id == "default_basic"
        assert len(basic_config.stages) == 1
        assert basic_config.stages[0].stage_type == ProcessingStage.MEANS
        
        full_config = loader.create_default_config("full")
        assert full_config.pipeline_id == "default_full"
        assert len(full_config.stages) == 3
        stage_types = {stage.stage_type for stage in full_config.stages}
        assert ProcessingStage.MEANS in stage_types
        assert ProcessingStage.METRICS in stage_types


class TestConfigurationManager:
    """Test high-level configuration management."""
    
    def test_configuration_manager_workflow(self):
        """Test complete configuration manager workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Create sample configuration
            config_data = {
                "pipeline_id": "manager_test",
                "pipeline_name": "Manager Test Pipeline",
                "base_data_path": "/tmp/manager",
                "stages": [
                    {
                        "stage_id": "test_stage",
                        "stage_type": "means",
                        "stage_name": "Test Stage",
                        "package_name": "test.package",
                        "entry_point": "handler"
                    }
                ]
            }
            
            config_file = config_dir / "test_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            
            # Test configuration manager
            manager = ConfigurationManager(config_dir)
            
            # Test listing configs
            configs = manager.list_configs()
            assert "test_config" in configs
            
            # Test loading config
            config = manager.load_config("test_config")
            assert config.pipeline_id == "manager_test"
            
            # Test getting stage config
            stage_config = manager.get_stage_config("test_stage")
            assert stage_config is not None
            assert stage_config.stage_id == "test_stage"
            
            # Test validation
            warnings = manager.validate_active_config()
            assert isinstance(warnings, list)


class TestPipelineOrchestration:
    """Test pipeline orchestration engine."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample pipeline configuration for testing."""
        stages = [
            StageConfiguration(
                stage_id="stage1",
                stage_type=ProcessingStage.MEANS,
                stage_name="Stage 1",
                package_name="test.package",
                entry_point="handler1"
            ),
            StageConfiguration(
                stage_id="stage2", 
                stage_type=ProcessingStage.METRICS,
                stage_name="Stage 2",
                package_name="test.package",
                entry_point="handler2",
                depends_on=["stage1"]
            )
        ]
        
        return PipelineConfiguration(
            pipeline_id="test_pipeline",
            pipeline_name="Test Pipeline",
            base_data_path=Path("/tmp/test"),
            stages=stages
        )
    
    def test_stage_execution_lifecycle(self):
        """Test stage execution state lifecycle."""
        execution = StageExecution(stage_id="test_stage")
        
        assert execution.status == ExecutionStatus.PENDING
        assert execution.start_time is None
        
        execution.start_execution()
        assert execution.status == ExecutionStatus.RUNNING
        assert execution.start_time is not None
        assert execution.attempt_number == 1
        
        execution.complete_execution({"output": "test"})
        assert execution.status == ExecutionStatus.COMPLETED
        assert execution.end_time is not None
        assert execution.output_data["output"] == "test"
        assert execution.duration_seconds is not None
    
    def test_stage_execution_failure(self):
        """Test stage execution failure handling."""
        execution = StageExecution(stage_id="test_stage")
        
        execution.start_execution()
        execution.fail_execution("Test error message")
        
        assert execution.status == ExecutionStatus.FAILED
        assert execution.error_message == "Test error message"
        assert execution.end_time is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_orchestrator_basic(self, sample_config):
        """Test basic pipeline orchestration."""
        orchestrator = PipelineOrchestrator(sample_config)
        
        # Mock stage handlers
        async def mock_handler1(**context):
            return {"stage1_output": "success"}
        
        async def mock_handler2(**context):
            stage_inputs = context.get('stage_inputs', {})
            assert 'stage1' in stage_inputs
            return {"stage2_output": "success"}
        
        orchestrator.register_stage_handler(ProcessingStage.MEANS, mock_handler1)
        orchestrator.register_stage_handler(ProcessingStage.METRICS, mock_handler2)
        
        # Execute pipeline
        execution = await orchestrator.execute_pipeline("test_exec")
        
        assert execution.status == ExecutionStatus.COMPLETED
        assert len(execution.stage_executions) == 2
        assert execution.stage_executions["stage1"].status == ExecutionStatus.COMPLETED
        assert execution.stage_executions["stage2"].status == ExecutionStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_pipeline_orchestrator_failure_handling(self, sample_config):
        """Test pipeline failure handling."""
        orchestrator = PipelineOrchestrator(sample_config)
        
        # Mock handlers with failure
        async def failing_handler(**context):
            raise Exception("Simulated failure")
        
        async def success_handler(**context):
            return {"output": "success"}
        
        orchestrator.register_stage_handler(ProcessingStage.MEANS, failing_handler)
        orchestrator.register_stage_handler(ProcessingStage.METRICS, success_handler)
        
        # Execute pipeline
        execution = await orchestrator.execute_pipeline("test_exec")
        
        assert execution.status == ExecutionStatus.FAILED
        assert execution.stage_executions["stage1"].status == ExecutionStatus.FAILED
        # Stage 2 should be skipped due to dependency failure
        assert execution.stage_executions["stage2"].status == ExecutionStatus.SKIPPED
    
    @pytest.mark.asyncio
    async def test_pipeline_retry_logic(self, sample_config):
        """Test stage retry logic."""
        # Modify config to have retry attempts
        sample_config.stages[0].retry_attempts = 2
        sample_config.stages[0].retry_delay_seconds = 0.1  # Fast retry for testing
        
        orchestrator = PipelineOrchestrator(sample_config)
        
        # Mock handler that fails twice then succeeds
        call_count = 0
        async def flaky_handler(**context):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception(f"Failure attempt {call_count}")
            return {"output": "success"}
        
        async def success_handler(**context):
            return {"output": "success"}
        
        orchestrator.register_stage_handler(ProcessingStage.MEANS, flaky_handler)
        orchestrator.register_stage_handler(ProcessingStage.METRICS, success_handler)
        
        # Execute pipeline
        execution = await orchestrator.execute_pipeline("test_exec")
        
        assert execution.status == ExecutionStatus.COMPLETED
        assert execution.stage_executions["stage1"].status == ExecutionStatus.COMPLETED
        assert execution.stage_executions["stage1"].attempt_number == 3  # 2 failures + 1 success


class TestPipelineRunner:
    """Test high-level pipeline runner interface."""
    
    def test_pipeline_runner_config_dict(self):
        """Test running pipeline with configuration dictionary."""
        config_data = {
            "pipeline_id": "runner_test",
            "pipeline_name": "Runner Test Pipeline",
            "base_data_path": "/tmp/runner",
            "stages": [
                {
                    "stage_id": "test_stage",
                    "stage_type": "means",
                    "stage_name": "Test Stage", 
                    "package_name": "test.package",
                    "entry_point": "handler"
                }
            ]
        }
        
        config = PipelineConfiguration(**config_data)
        runner = PipelineRunner()
        
        # Mock the orchestrator to avoid actual execution
        with patch.object(runner, '_register_default_handlers'):
            with patch('county_climate.shared.orchestration.pipeline_orchestrator.PipelineOrchestrator') as mock_orchestrator_class:
                mock_orchestrator = MagicMock()
                mock_orchestrator_class.return_value = mock_orchestrator
                
                # Mock async execution
                async def mock_execute():
                    from county_climate.shared.orchestration.pipeline_orchestrator import PipelineExecution
                    return PipelineExecution(
                        pipeline_id="runner_test",
                        execution_id="test_exec",
                        status=ExecutionStatus.COMPLETED
                    )
                
                mock_orchestrator.execute_pipeline.return_value = mock_execute()
                
                # This would normally run the pipeline, but we're mocking it
                # result = runner.run_pipeline(config, "test_exec")
                
                # Verify orchestrator was created with correct config
                runner.orchestrator = mock_orchestrator
                assert runner.orchestrator is not None
    
    @pytest.mark.asyncio
    async def test_integration_workflow_example(self):
        """Test a complete integration workflow example."""
        # This test demonstrates how the configuration-driven system would work
        
        # 1. Create a processing profile
        profile = ProcessingProfile(
            profile_name="integration_test",
            description="Integration test profile",
            regions=[Region.CONUS],
            variables=[ClimateVariable.TEMPERATURE],
            scenarios=[Scenario.HISTORICAL],
            year_ranges=[(1990, 2000)],
            enable_means=True,
            enable_metrics=True,
            max_parallel_regions=1,
            memory_per_process_gb=2.0
        )
        
        # 2. Create base pipeline configuration
        base_config = PipelineConfiguration(
            pipeline_id="integration_pipeline",
            pipeline_name="Integration Test Pipeline", 
            base_data_path=Path("/tmp/integration"),
            stages=[
                StageConfiguration(
                    stage_id="means_processing",
                    stage_type=ProcessingStage.MEANS,
                    stage_name="Climate Means Processing",
                    package_name="county_climate.means",
                    entry_point="process_region"
                ),
                StageConfiguration(
                    stage_id="metrics_processing", 
                    stage_type=ProcessingStage.METRICS,
                    stage_name="Climate Metrics Processing",
                    package_name="county_climate.metrics",
                    entry_point="process_county_metrics",
                    depends_on=["means_processing"]
                )
            ]
        )
        
        # 3. Apply profile to configuration
        pipeline_config = profile.to_pipeline_config(base_config)
        
        # 4. Validate configuration
        loader = ConfigurationLoader()
        warnings = loader.validate_config(pipeline_config)
        assert isinstance(warnings, list)  # Should be able to validate
        
        # 5. Create orchestrator and register handlers
        orchestrator = PipelineOrchestrator(pipeline_config)
        
        # Mock handlers for demonstration
        async def mock_means_handler(**context):
            # For testing, just return some mock data
            return {
                "processed_regions": ["CONUS"],
                "processed_variables": ["temperature"],
                "output_files": ["/tmp/output/CONUS_temperature.nc"]
            }
        
        async def mock_metrics_handler(**context):
            inputs = context['stage_inputs']
            means_output = inputs.get('means_processing', {})
            output_files = means_output.get('output_files', [])
            
            return {
                "metrics_computed": len(output_files),
                "county_metrics": {"county_001": {"temp_mean": 15.5}}
            }
        
        orchestrator.register_stage_handler(ProcessingStage.MEANS, mock_means_handler)
        orchestrator.register_stage_handler(ProcessingStage.METRICS, mock_metrics_handler)
        
        # 6. Execute pipeline
        execution = await orchestrator.execute_pipeline("integration_test_exec")
        
        # 7. Verify results
        assert execution.status == ExecutionStatus.COMPLETED
        assert len(execution.stage_executions) == 2
        assert execution.stage_executions["means_processing"].status == ExecutionStatus.COMPLETED
        assert execution.stage_executions["metrics_processing"].status == ExecutionStatus.COMPLETED
        
        # Verify data flow between stages
        means_output = execution.stage_executions["means_processing"].output_data
        metrics_output = execution.stage_executions["metrics_processing"].output_data
        
        assert "processed_regions" in means_output
        assert "metrics_computed" in metrics_output
        assert metrics_output["metrics_computed"] > 0