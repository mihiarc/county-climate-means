"""
Tests for Pydantic data contracts.

This module tests the data contracts to ensure they provide proper
validation, serialization, and type safety for pipeline integration.
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from pydantic import ValidationError

from county_climate.shared.contracts.climate_data import (
    ClimateDatasetContract,
    ClimateVariable,
    Region,
    Scenario,
    ProcessingType,
    ProcessingStatus,
    ValidationStatus,
    SpatialBoundsContract,
    QualityMetricsContract,
    ProcessingMetadataContract,
    DataAccessContract,
    ValidationResultContract,
    CatalogQueryContract,
    CatalogResponseContract
)

from county_climate.shared.contracts.pipeline_interface import (
    PipelineConfigContract,
    ProcessingMode,
    ProcessingStatusContract,
    ErrorContract,
    DataflowContract,
    HealthCheckContract,
    HealthStatus,
    ErrorSeverity
)

from county_climate.shared.validation.validators import (
    ClimateDataValidator,
    FilePathValidator,
    CoordinateValidator,
    TemporalValidator
)


class TestClimateDataContracts:
    """Test cases for climate data contracts."""
    
    def test_spatial_bounds_contract(self):
        """Test spatial bounds contract validation."""
        # Valid bounds
        bounds = SpatialBoundsContract(
            min_longitude=-125.0,
            max_longitude=-66.5,
            min_latitude=25.0,
            max_latitude=49.0
        )
        assert bounds.min_longitude == -125.0
        assert bounds.to_list() == [-125.0, -66.5, 25.0, 49.0]
        
        # Invalid longitude range
        with pytest.raises(ValidationError):
            SpatialBoundsContract(
                min_longitude=50.0,
                max_longitude=-50.0,  # max < min
                min_latitude=25.0,
                max_latitude=49.0
            )
        
        # Invalid latitude values
        with pytest.raises(ValidationError):
            SpatialBoundsContract(
                min_longitude=-125.0,
                max_longitude=-66.5,
                min_latitude=-100.0,  # Invalid latitude
                max_latitude=49.0
            )
    
    def test_quality_metrics_contract(self):
        """Test quality metrics contract validation."""
        # Valid quality metrics
        quality = QualityMetricsContract(
            quality_score=0.95,
            completeness_ratio=0.99,
            coordinate_validation=True,
            temporal_validation=True,
            data_range_validation=True,
            quality_flags=["minor_gaps"],
            validation_errors=[]
        )
        assert quality.quality_score == 0.95
        assert "minor_gaps" in quality.quality_flags
        
        # Invalid quality score
        with pytest.raises(ValidationError):
            QualityMetricsContract(
                quality_score=1.5,  # > 1.0
                completeness_ratio=0.99,
                coordinate_validation=True,
                temporal_validation=True,
                data_range_validation=True
            )
        
        # Invalid quality flag
        with pytest.raises(ValidationError):
            QualityMetricsContract(
                quality_score=0.95,
                completeness_ratio=0.99,
                coordinate_validation=True,
                temporal_validation=True,
                data_range_validation=True,
                quality_flags=["invalid_flag"]
            )
    
    def test_data_access_contract(self):
        """Test data access contract validation."""
        # Valid data access
        access = DataAccessContract(
            file_path="output/data/CONUS/tas/file.nc",
            file_size_bytes=1024,
            checksum="a" * 64,  # Valid SHA256
            download_url="/api/v1/datasets/test/download"
        )
        assert access.file_path == "output/data/CONUS/tas/file.nc"
        assert len(access.checksum) == 64
        
        # Invalid file path (absolute)
        with pytest.raises(ValidationError):
            DataAccessContract(
                file_path="/absolute/path/file.nc",
                file_size_bytes=1024,
                checksum="a" * 64
            )
        
        # Invalid checksum
        with pytest.raises(ValidationError):
            DataAccessContract(
                file_path="output/file.nc",
                file_size_bytes=1024,
                checksum="invalid_checksum"
            )
    
    def test_climate_dataset_contract(self):
        """Test main climate dataset contract."""
        # Create valid dataset
        dataset = self._create_valid_dataset()
        
        # Test properties
        assert dataset.id == "tas_CONUS_historical_1995"
        assert dataset.variable == ClimateVariable.TEMPERATURE
        assert dataset.region == Region.CONUS
        assert dataset.scenario == Scenario.HISTORICAL
        assert dataset.target_year == 1995
        
        # Test computed properties
        assert dataset.is_ready_for_metrics  # Should be True for valid dataset
        
        # Test download info
        download_info = dataset.get_download_info()
        assert "file_path" in download_info
        assert "checksum" in download_info
        assert "file_size_mb" in download_info
    
    def test_dataset_id_validation(self):
        """Test dataset ID format validation."""
        dataset = self._create_valid_dataset()
        
        # Valid ID should work
        assert dataset.id == "tas_CONUS_historical_1995"
        
        # Invalid ID format should fail
        with pytest.raises(ValidationError):
            dataset.id = "invalid_id_format"
    
    def test_dataset_consistency_validation(self):
        """Test dataset ID consistency with components."""
        # This should fail because ID doesn't match components
        with pytest.raises(ValidationError):
            ClimateDatasetContract(
                id="wrong_id_format",
                variable=ClimateVariable.TEMPERATURE,
                region=Region.CONUS,
                scenario=Scenario.HISTORICAL,
                target_year=1995,
                internal_processing_type=ProcessingType.HISTORICAL,
                status=ProcessingStatus.COMPLETED,
                spatial_bounds=self._create_valid_spatial_bounds(),
                quality_metrics=self._create_valid_quality_metrics(),
                processing_metadata=self._create_valid_processing_metadata(),
                data_access=self._create_valid_data_access(),
                validation_result=self._create_valid_validation_result()
            )
    
    def test_catalog_query_contract(self):
        """Test catalog query contract."""
        # Valid query
        query = CatalogQueryContract(
            variables=[ClimateVariable.TEMPERATURE],
            regions=[Region.CONUS],
            scenarios=[Scenario.HISTORICAL],
            year_start=1990,
            year_end=2000,
            min_quality_score=0.95,
            limit=100,
            sort_by="target_year"
        )
        assert len(query.variables) == 1
        assert query.year_start == 1990
        
        # Invalid year range
        with pytest.raises(ValidationError):
            CatalogQueryContract(
                year_start=2000,
                year_end=1990  # end < start
            )
    
    def test_catalog_response_contract(self):
        """Test catalog response contract."""
        datasets = [self._create_valid_dataset()]
        
        response = CatalogResponseContract(
            query_timestamp=datetime.now(timezone.utc),
            total_matches=1,
            returned_count=1,
            offset=0,
            has_more=False,
            datasets=datasets,
            summary={"test": "data"}
        )
        
        assert response.total_matches == 1
        assert len(response.datasets) == 1
        
        # Invalid returned_count
        with pytest.raises(ValidationError):
            CatalogResponseContract(
                query_timestamp=datetime.now(timezone.utc),
                total_matches=1,
                returned_count=2,  # Doesn't match datasets length
                offset=0,
                has_more=False,
                datasets=datasets
            )
    
    def _create_valid_spatial_bounds(self) -> SpatialBoundsContract:
        """Create valid spatial bounds for testing."""
        return SpatialBoundsContract(
            min_longitude=-125.0,
            max_longitude=-66.5,
            min_latitude=25.0,
            max_latitude=49.0
        )
    
    def _create_valid_quality_metrics(self) -> QualityMetricsContract:
        """Create valid quality metrics for testing."""
        return QualityMetricsContract(
            quality_score=0.98,
            completeness_ratio=0.99,
            coordinate_validation=True,
            temporal_validation=True,
            data_range_validation=True
        )
    
    def _create_valid_processing_metadata(self) -> ProcessingMetadataContract:
        """Create valid processing metadata for testing."""
        return ProcessingMetadataContract(
            processing_timestamp=datetime.now(timezone.utc),
            processing_duration_seconds=120.0,
            processing_version="0.1.0",
            source_files=["test_file.nc"],
            temporal_window_start=1966,
            temporal_window_end=1995
        )
    
    def _create_valid_data_access(self) -> DataAccessContract:
        """Create valid data access for testing."""
        # Create a temporary test file for validation
        test_file = Path("output/data/test_file.nc")
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.touch()  # Create empty file for testing
        
        return DataAccessContract(
            file_path="output/data/test_file.nc",
            file_size_bytes=1024,
            checksum="a" * 64
        )
    
    def _create_valid_validation_result(self) -> ValidationResultContract:
        """Create valid validation result for testing."""
        return ValidationResultContract(
            validation_status=ValidationStatus.PASSED,
            validation_timestamp=datetime.now(timezone.utc),
            validator_version="1.0.0",
            coordinate_validation=True,
            temporal_validation=True,
            data_range_validation=True,
            metadata_validation=True,
            file_integrity_validation=True
        )
    
    def _create_valid_dataset(self) -> ClimateDatasetContract:
        """Create valid dataset for testing."""
        return ClimateDatasetContract(
            id="tas_CONUS_historical_1995",
            variable=ClimateVariable.TEMPERATURE,
            region=Region.CONUS,
            scenario=Scenario.HISTORICAL,
            target_year=1995,
            internal_processing_type=ProcessingType.HISTORICAL,
            status=ProcessingStatus.COMPLETED,
            spatial_bounds=self._create_valid_spatial_bounds(),
            quality_metrics=self._create_valid_quality_metrics(),
            processing_metadata=self._create_valid_processing_metadata(),
            data_access=self._create_valid_data_access(),
            validation_result=self._create_valid_validation_result()
        )


class TestPipelineInterfaceContracts:
    """Test cases for pipeline interface contracts."""
    
    def test_pipeline_config_contract(self):
        """Test pipeline configuration contract."""
        config = PipelineConfigContract(
            pipeline_id="test_pipeline",
            pipeline_name="Test Pipeline",
            pipeline_version="1.0.0",
            processing_mode=ProcessingMode.MULTIPROCESSING,
            max_workers=6,
            input_data_path="/data/input",
            output_data_path="/data/output",
            catalog_path="/data/catalog.yaml"
        )
        
        assert config.pipeline_id == "test_pipeline"
        assert config.processing_mode == ProcessingMode.MULTIPROCESSING
        assert config.max_workers == 6
        
        # Invalid worker count
        with pytest.raises(ValidationError):
            PipelineConfigContract(
                pipeline_id="test",
                pipeline_name="Test",
                pipeline_version="1.0.0",
                processing_mode=ProcessingMode.MULTIPROCESSING,
                max_workers=0,  # Invalid
                input_data_path="/data/input",
                output_data_path="/data/output",
                catalog_path="/data/catalog.yaml"
            )
    
    def test_processing_status_contract(self):
        """Test processing status contract."""
        status = ProcessingStatusContract(
            pipeline_id="test_pipeline",
            stage="means_processing",
            status="running",
            total_tasks=100,
            completed_tasks=50,
            failed_tasks=2,
            active_workers=6,
            start_time=datetime.now(timezone.utc),
            last_update=datetime.now(timezone.utc),
            throughput_tasks_per_hour=25.0,
            avg_task_duration_seconds=144.0,
            memory_usage_gb=8.5,
            cpu_usage_percent=75.0,
            current_quality_score=0.97,
            error_rate=0.02
        )
        
        assert status.progress_percentage == 50.0
        assert not status.is_complete
        
        # Invalid completed tasks
        with pytest.raises(ValidationError):
            ProcessingStatusContract(
                pipeline_id="test_pipeline",
                stage="means_processing",
                status="running",
                total_tasks=100,
                completed_tasks=150,  # > total_tasks
                failed_tasks=0,
                active_workers=6,
                start_time=datetime.now(timezone.utc),
                last_update=datetime.now(timezone.utc),
                throughput_tasks_per_hour=25.0,
                avg_task_duration_seconds=144.0,
                memory_usage_gb=8.5,
                cpu_usage_percent=75.0,
                current_quality_score=0.97,
                error_rate=0.02
            )
    
    def test_error_contract(self):
        """Test error contract."""
        error = ErrorContract(
            error_id="test_error_001",
            pipeline_id="test_pipeline",
            stage="means_processing",
            error_type="FileNotFoundError",
            error_message="Input file not found",
            severity=ErrorSeverity.HIGH,
            timestamp=datetime.now(timezone.utc),
            dataset_id="tas_CONUS_historical_1995"
        )
        
        assert error.error_id == "test_error_001"
        assert error.severity == ErrorSeverity.HIGH
        assert not error.is_retryable  # Default
    
    def test_health_check_contract(self):
        """Test health check contract."""
        health = HealthCheckContract(
            pipeline_id="test_pipeline",
            overall_status=HealthStatus.HEALTHY,
            check_timestamp=datetime.now(timezone.utc),
            database_status=HealthStatus.HEALTHY,
            file_system_status=HealthStatus.HEALTHY,
            memory_status=HealthStatus.HEALTHY,
            cpu_status=HealthStatus.HEALTHY,
            memory_usage_percent=65.0,
            cpu_usage_percent=45.0,
            disk_usage_percent=78.0,
            active_connections=12,
            recent_error_count=0,
            avg_response_time_ms=125.0,
            throughput_last_hour=145.0
        )
        
        assert health.is_healthy
        assert not health.needs_attention


class TestValidators:
    """Test cases for validation utilities."""
    
    def test_file_path_validator(self):
        """Test file path validation."""
        # Valid relative paths
        assert FilePathValidator.validate_relative_path("output/data/file.nc")
        assert FilePathValidator.validate_netcdf_extension("file.nc")
        
        # Invalid paths
        assert not FilePathValidator.validate_relative_path("/absolute/path")
        assert not FilePathValidator.validate_relative_path("path/../with/dotdot")
        assert not FilePathValidator.validate_netcdf_extension("file.txt")
    
    def test_coordinate_validator(self):
        """Test coordinate validation."""
        # Valid coordinates
        assert CoordinateValidator.validate_longitude(-125.0)
        assert CoordinateValidator.validate_longitude(180.0)
        assert CoordinateValidator.validate_latitude(45.0)
        assert CoordinateValidator.validate_bounds_consistency(-125.0, -66.5)
        
        # Invalid coordinates
        assert not CoordinateValidator.validate_longitude(-200.0)
        assert not CoordinateValidator.validate_latitude(100.0)
        assert not CoordinateValidator.validate_bounds_consistency(50.0, -50.0)
    
    def test_temporal_validator(self):
        """Test temporal validation."""
        # Valid years and ranges
        assert TemporalValidator.validate_year(1995)
        assert TemporalValidator.validate_year_range(1980, 2020)
        assert TemporalValidator.validate_climatology_window(1966, 1995, 1980)
        
        # Invalid years and ranges
        assert not TemporalValidator.validate_year(1800)
        assert not TemporalValidator.validate_year(2200)
        assert not TemporalValidator.validate_year_range(2000, 1980)
        assert not TemporalValidator.validate_climatology_window(1980, 2009, 1970)
    
    def test_climate_data_validator(self):
        """Test climate data validator."""
        validator = ClimateDataValidator()
        
        # Create a valid dataset for testing
        dataset = self._create_test_dataset()
        
        # Validate the dataset
        validation_result = validator.validate_contract(dataset)
        
        # Check validation result structure
        assert hasattr(validation_result, 'validation_status')
        assert hasattr(validation_result, 'is_valid')
        assert hasattr(validation_result, 'coordinate_validation')
        assert hasattr(validation_result, 'temporal_validation')
        assert hasattr(validation_result, 'data_range_validation')
        assert hasattr(validation_result, 'metadata_validation')
        assert hasattr(validation_result, 'file_integrity_validation')
    
    def _create_test_dataset(self) -> ClimateDatasetContract:
        """Create a test dataset for validation."""
        return ClimateDatasetContract(
            id="tas_CONUS_historical_1995",
            variable=ClimateVariable.TEMPERATURE,
            region=Region.CONUS,
            scenario=Scenario.HISTORICAL,
            target_year=1995,
            internal_processing_type=ProcessingType.HISTORICAL,
            status=ProcessingStatus.COMPLETED,
            spatial_bounds=SpatialBoundsContract(
                min_longitude=-125.0,
                max_longitude=-66.5,
                min_latitude=25.0,
                max_latitude=49.0
            ),
            quality_metrics=QualityMetricsContract(
                quality_score=0.98,
                completeness_ratio=0.99,
                coordinate_validation=True,
                temporal_validation=True,
                data_range_validation=True
            ),
            processing_metadata=ProcessingMetadataContract(
                processing_timestamp=datetime.now(timezone.utc),
                processing_duration_seconds=120.0,
                processing_version="0.1.0",
                source_files=["test_file.nc"],
                temporal_window_start=1966,
                temporal_window_end=1995
            ),
            data_access=DataAccessContract(
                file_path="output/data/test_file.nc",
                file_size_bytes=1024,
                checksum="a" * 64
            ),
            validation_result=ValidationResultContract(
                validation_status=ValidationStatus.PASSED,
                validation_timestamp=datetime.now(timezone.utc),
                validator_version="1.0.0",
                coordinate_validation=True,
                temporal_validation=True,
                data_range_validation=True,
                metadata_validation=True,
                file_integrity_validation=True
            )
        )


class TestSerialization:
    """Test contract serialization and deserialization."""
    
    def test_dataset_serialization(self):
        """Test dataset contract serialization."""
        dataset = self._create_test_dataset()
        
        # Test dict serialization
        dataset_dict = dataset.model_dump()
        assert isinstance(dataset_dict, dict)
        assert dataset_dict['id'] == "tas_CONUS_historical_1995"
        
        # Test JSON serialization
        dataset_json = dataset.model_dump_json()
        assert isinstance(dataset_json, str)
        
        # Test round-trip
        dataset_restored = ClimateDatasetContract.model_validate_json(dataset_json)
        assert dataset.id == dataset_restored.id
        assert dataset.variable == dataset_restored.variable
        assert dataset.quality_metrics.quality_score == dataset_restored.quality_metrics.quality_score
    
    def test_schema_generation(self):
        """Test JSON schema generation."""
        schema = ClimateDatasetContract.model_json_schema()
        
        assert 'properties' in schema
        assert 'id' in schema['properties']
        assert 'variable' in schema['properties']
        assert 'quality_metrics' in schema['properties']
        
        # Check that enums are properly defined
        if '$defs' in schema and 'ClimateVariable' in schema['$defs']:
            variable_enum = schema['$defs']['ClimateVariable']['enum']
            assert 'pr' in variable_enum
            assert 'tas' in variable_enum
    
    def _create_test_dataset(self) -> ClimateDatasetContract:
        """Create a test dataset for serialization testing."""
        return ClimateDatasetContract(
            id="tas_CONUS_historical_1995",
            variable=ClimateVariable.TEMPERATURE,
            region=Region.CONUS,
            scenario=Scenario.HISTORICAL,
            target_year=1995,
            internal_processing_type=ProcessingType.HISTORICAL,
            status=ProcessingStatus.COMPLETED,
            spatial_bounds=SpatialBoundsContract(
                min_longitude=-125.0,
                max_longitude=-66.5,
                min_latitude=25.0,
                max_latitude=49.0
            ),
            quality_metrics=QualityMetricsContract(
                quality_score=0.98,
                completeness_ratio=0.99,
                coordinate_validation=True,
                temporal_validation=True,
                data_range_validation=True
            ),
            processing_metadata=ProcessingMetadataContract(
                processing_timestamp=datetime.now(timezone.utc),
                processing_duration_seconds=120.0,
                processing_version="0.1.0",
                source_files=["test_file.nc"],
                temporal_window_start=1966,
                temporal_window_end=1995
            ),
            data_access=DataAccessContract(
                file_path="output/data/test_file.nc",
                file_size_bytes=1024,
                checksum="a" * 64
            ),
            validation_result=ValidationResultContract(
                validation_status=ValidationStatus.PASSED,
                validation_timestamp=datetime.now(timezone.utc),
                validator_version="1.0.0",
                coordinate_validation=True,
                temporal_validation=True,
                data_range_validation=True,
                metadata_validation=True,
                file_integrity_validation=True
            )
        )