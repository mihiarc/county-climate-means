"""
Example usage of Pydantic data contracts for climate data processing.

This module demonstrates how to use the standardized contracts for
type-safe integration between climate processing packages.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Import contracts
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
    CatalogQueryContract
)

from county_climate.shared.contracts.pipeline_interface import (
    PipelineConfigContract,
    ProcessingMode,
    LogLevel
)

from county_climate.shared.validation.validators import ClimateDataValidator


def create_example_dataset() -> ClimateDatasetContract:
    """Create an example climate dataset contract."""
    
    # Create spatial bounds
    spatial_bounds = SpatialBoundsContract(
        min_longitude=-125.0,
        max_longitude=-66.5,
        min_latitude=25.0,
        max_latitude=49.0
    )
    
    # Create quality metrics
    quality_metrics = QualityMetricsContract(
        quality_score=0.98,
        completeness_ratio=0.99,
        coordinate_validation=True,
        temporal_validation=True,
        data_range_validation=True,
        quality_flags=["minor_gaps"],
        validation_errors=[]
    )
    
    # Create processing metadata
    processing_metadata = ProcessingMetadataContract(
        processing_timestamp=datetime.now(timezone.utc),
        processing_duration_seconds=125.5,
        processing_version="0.1.0",
        processing_software="climate-means",
        multiprocessing_workers=6,
        batch_size=15,
        memory_usage_gb=12.5,
        source_files=["tas_day_NorESM2-LM_historical_r1i1p1f1_gn_1995.nc"],
        source_checksums={"input_file": "abc123def456"},
        temporal_window_start=1966,
        temporal_window_end=1995,
        temporal_window_length=30
    )
    
    # Create data access info
    data_access = DataAccessContract(
        file_path="output/data/CONUS/tas/tas_CONUS_historical_1995_climatology.nc",
        file_size_bytes=50331648,
        file_format="netcdf4",
        checksum="a1b2c3d4e5f67890123456789012345678901234567890123456789012345678",
        compression_level=4,
        download_url="/api/v1/datasets/tas_CONUS_historical_1995/download",
        api_endpoint="/api/v1/datasets/tas_CONUS_historical_1995"
    )
    
    # Create validation result
    validation_result = ValidationResultContract(
        validation_status=ValidationStatus.PASSED,
        validation_timestamp=datetime.now(timezone.utc),
        validator_version="1.0.0",
        coordinate_validation=True,
        temporal_validation=True,
        data_range_validation=True,
        metadata_validation=True,
        file_integrity_validation=True,
        validation_messages=["All validations passed"],
        validation_warnings=["Minor data gaps in source"],
        validation_errors=[]
    )
    
    # Create the main dataset contract
    dataset = ClimateDatasetContract(
        id="tas_CONUS_historical_1995",
        variable=ClimateVariable.TEMPERATURE,
        region=Region.CONUS,
        scenario=Scenario.HISTORICAL,
        target_year=1995,
        internal_processing_type=ProcessingType.HISTORICAL,
        status=ProcessingStatus.COMPLETED,
        spatial_bounds=spatial_bounds,
        quality_metrics=quality_metrics,
        processing_metadata=processing_metadata,
        data_access=data_access,
        validation_result=validation_result,
        description="30-year climate normal for temperature in CONUS for year 1995",
        units="K",
        temporal_resolution="daily_climatology",
        spatial_resolution="0.25_degree",
        metrics_compatible=True
    )
    
    return dataset


def create_example_pipeline_config() -> PipelineConfigContract:
    """Create an example pipeline configuration contract."""
    
    config = PipelineConfigContract(
        pipeline_id="climate_means_production",
        pipeline_name="Climate Means Production Pipeline",
        pipeline_version="0.1.0",
        processing_mode=ProcessingMode.MULTIPROCESSING,
        max_workers=6,
        batch_size=15,
        max_memory_per_worker_gb=4.0,
        max_processing_time_hours=12,
        temp_storage_gb=100.0,
        min_quality_score=0.95,
        max_error_rate=0.05,
        upstream_pipeline=None,
        downstream_pipelines=["climate_metrics", "climate_extremes"],
        input_data_path="/data/climate/norESM2",
        output_data_path="/data/climate/normals",
        catalog_path="/data/climate/catalog.yaml",
        health_check_interval_seconds=60,
        log_level=LogLevel.INFO,
        enable_metrics=True
    )
    
    return config


def create_example_catalog_query() -> CatalogQueryContract:
    """Create an example catalog query contract."""
    
    query = CatalogQueryContract(
        variables=[ClimateVariable.TEMPERATURE, ClimateVariable.PRECIPITATION],
        regions=[Region.CONUS, Region.ALASKA],
        scenarios=[Scenario.HISTORICAL, Scenario.SSP245],
        year_start=1990,
        year_end=2020,
        min_quality_score=0.95,
        validation_status=ValidationStatus.PASSED,
        status=ProcessingStatus.COMPLETED,
        limit=100,
        offset=0,
        sort_by="target_year",
        sort_order="asc",
        include_metadata=True,
        metrics_compatible_only=True
    )
    
    return query


def demonstrate_validation():
    """Demonstrate contract validation capabilities."""
    
    print("=== Contract Validation Demonstration ===\n")
    
    # Create example dataset
    dataset = create_example_dataset()
    print(f"Created dataset: {dataset.id}")
    print(f"Variable: {dataset.variable}, Region: {dataset.region}")
    print(f"Quality Score: {dataset.quality_metrics.quality_score}")
    print(f"Is ready for metrics: {dataset.is_ready_for_metrics}")
    print()
    
    # Validate using the validator
    validator = ClimateDataValidator()
    validation_result = validator.validate_contract(dataset)
    
    print("Validation Results:")
    print(f"  Status: {validation_result.validation_status}")
    print(f"  Is Valid: {validation_result.is_valid}")
    print(f"  Coordinate Validation: {validation_result.coordinate_validation}")
    print(f"  Temporal Validation: {validation_result.temporal_validation}")
    print(f"  Data Range Validation: {validation_result.data_range_validation}")
    print(f"  Metadata Validation: {validation_result.metadata_validation}")
    print(f"  File Integrity Validation: {validation_result.file_integrity_validation}")
    
    if validation_result.validation_messages:
        print("  Messages:", validation_result.validation_messages)
    
    if validation_result.validation_warnings:
        print("  Warnings:", validation_result.validation_warnings)
    
    if validation_result.validation_errors:
        print("  Errors:", validation_result.validation_errors)
    
    print()


def demonstrate_serialization():
    """Demonstrate contract serialization capabilities."""
    
    print("=== Contract Serialization Demonstration ===\n")
    
    # Create dataset and config
    dataset = create_example_dataset()
    config = create_example_pipeline_config()
    query = create_example_catalog_query()
    
    # Serialize to dictionaries
    dataset_dict = dataset.dict()
    config_dict = config.dict()
    query_dict = query.dict()
    
    print(f"Dataset serialized to {len(dataset_dict)} fields")
    print(f"Config serialized to {len(config_dict)} fields")
    print(f"Query serialized to {len(query_dict)} fields")
    print()
    
    # Demonstrate JSON schema generation
    dataset_schema = dataset.schema()
    print(f"Generated JSON schema with {len(dataset_schema['properties'])} properties")
    print("Schema example properties:")
    for prop in list(dataset_schema['properties'].keys())[:5]:
        print(f"  - {prop}")
    print()
    
    # Demonstrate round-trip serialization
    dataset_json = dataset.json()
    dataset_restored = ClimateDatasetContract.parse_raw(dataset_json)
    
    print(f"Round-trip serialization successful: {dataset.id == dataset_restored.id}")
    print()


def demonstrate_type_safety():
    """Demonstrate type safety features of contracts."""
    
    print("=== Type Safety Demonstration ===\n")
    
    # Valid creation
    try:
        valid_dataset = create_example_dataset()
        print(f"✓ Valid dataset created: {valid_dataset.id}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Invalid enum value
    try:
        ClimateDatasetContract(
            id="test_invalid_enum",
            variable="invalid_variable",  # This should fail
            region=Region.CONUS,
            scenario=Scenario.HISTORICAL,
            target_year=1995,
            # ... other required fields would be needed
        )
        print("✗ Should have failed with invalid variable")
    except Exception as e:
        print(f"✓ Correctly caught invalid variable: {type(e).__name__}")
    
    # Invalid year range
    try:
        dataset = create_example_dataset()
        dataset.target_year = 1800  # Invalid year
        print("✗ Should have failed with invalid year")
    except Exception as e:
        print(f"✓ Correctly caught invalid year: {type(e).__name__}")
    
    # Invalid quality score
    try:
        QualityMetricsContract(
            quality_score=1.5,  # Invalid (> 1.0)
            completeness_ratio=0.95,
            coordinate_validation=True,
            temporal_validation=True,
            data_range_validation=True
        )
        print("✗ Should have failed with invalid quality score")
    except Exception as e:
        print(f"✓ Correctly caught invalid quality score: {type(e).__name__}")
    
    print()


def demonstrate_query_filtering():
    """Demonstrate query contract filtering capabilities."""
    
    print("=== Query Filtering Demonstration ===\n")
    
    # Create various queries
    queries = [
        CatalogQueryContract(
            variables=[ClimateVariable.TEMPERATURE],
            regions=[Region.CONUS],
            year_start=1990,
            year_end=2000
        ),
        CatalogQueryContract(
            scenarios=[Scenario.SSP245],
            min_quality_score=0.98,
            metrics_compatible_only=True
        ),
        CatalogQueryContract(
            limit=50,
            sort_by="target_year",
            sort_order="desc"
        )
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}:")
        if query.variables:
            print(f"  Variables: {[v.value for v in query.variables]}")
        if query.regions:
            print(f"  Regions: {[r.value for r in query.regions]}")
        if query.scenarios:
            print(f"  Scenarios: {[s.value for s in query.scenarios]}")
        if query.year_start:
            print(f"  Year range: {query.year_start}-{query.year_end}")
        if query.min_quality_score:
            print(f"  Min quality: {query.min_quality_score}")
        print(f"  Metrics compatible only: {query.metrics_compatible_only}")
        print(f"  Sort: {query.sort_by} ({query.sort_order})")
        print()


if __name__ == "__main__":
    """Run all demonstrations."""
    
    print("Climate Data Contracts - Usage Examples")
    print("=" * 50)
    print()
    
    demonstrate_validation()
    demonstrate_serialization()
    demonstrate_type_safety()
    demonstrate_query_filtering()
    
    print("All demonstrations completed successfully!")