"""
Example configurations for climate data processing pipelines.

This module provides sample configurations that demonstrate how to set up
configuration-driven climate data processing workflows.
"""

from pathlib import Path
from typing import Dict, Any

from ..config.integration_config import (
    PipelineConfiguration,
    StageConfiguration,
    ProcessingProfile,
    ProcessingStage,
    TriggerType,
    EnvironmentType,
    ResourceLimits,
    DataFlowConfiguration,
    DataFlowType,
)
from ..contracts.climate_data import Region, ClimateVariable, Scenario


def create_basic_means_pipeline() -> PipelineConfiguration:
    """Create a basic pipeline for climate means processing only."""
    
    means_stage = StageConfiguration(
        stage_id="climate_means",
        stage_type=ProcessingStage.MEANS,
        stage_name="Climate Means Processing",
        package_name="county_climate.means",
        entry_point="process_region",
        trigger_type=TriggerType.MANUAL,
        resource_limits=ResourceLimits(
            max_memory_gb=8.0,
            max_cpu_cores=6,
            max_processing_time_hours=4.0,
            priority=8
        ),
        stage_config={
            "variables": ["temperature", "precipitation"],
            "regions": ["CONUS"],
            "scenarios": ["historical"],
            "year_range": [1990, 2020],
            "batch_size_years": 15,
            "multiprocessing_workers": 6
        }
    )
    
    validation_stage = StageConfiguration(
        stage_id="output_validation",
        stage_type=ProcessingStage.VALIDATION,
        stage_name="Output Data Validation",
        package_name="county_climate.shared.validation",
        entry_point="validate_means_output",
        depends_on=["climate_means"],
        trigger_type=TriggerType.DEPENDENCY,
        resource_limits=ResourceLimits(
            max_memory_gb=2.0,
            max_cpu_cores=2,
            max_processing_time_hours=0.5,
            priority=5
        )
    )
    
    return PipelineConfiguration(
        pipeline_id="basic_means_pipeline",
        pipeline_name="Basic Climate Means Processing Pipeline",
        pipeline_version="1.0.0",
        environment=EnvironmentType.PRODUCTION,
        base_data_path=Path("/data/climate/normals"),
        temp_data_path=Path("/tmp/climate_processing"),
        log_path=Path("/var/log/climate/means.log"),
        stages=[means_stage, validation_stage],
        global_resource_limits=ResourceLimits(
            max_memory_gb=16.0,
            max_cpu_cores=8,
            max_processing_time_hours=6.0
        ),
        global_config={
            "data_source": "NEX-GDDP-CMIP6",
            "model": "NorESM2-LM",
            "climate_normals_period": 30,
            "output_format": "netcdf4",
            "compression_level": 4
        },
        enable_monitoring=True,
        monitoring_interval_seconds=30,
        description="Processes climate means for basic temperature and precipitation variables",
        tags=["production", "means", "basic"]
    )


def create_full_pipeline() -> PipelineConfiguration:
    """Create a complete pipeline with means, metrics, and visualization."""
    
    means_stage = StageConfiguration(
        stage_id="climate_means",
        stage_type=ProcessingStage.MEANS,
        stage_name="Climate Means Processing",
        package_name="county_climate.means",
        entry_point="process_region",
        trigger_type=TriggerType.MANUAL,
        resource_limits=ResourceLimits(
            max_memory_gb=12.0,
            max_cpu_cores=6,
            max_processing_time_hours=8.0,
            priority=10
        ),
        stage_config={
            "variables": ["temperature", "precipitation", "temperature_max", "temperature_min"],
            "regions": ["CONUS", "AK", "HI"],
            "scenarios": ["historical", "ssp245"],
            "year_range": [1980, 2020],
            "enable_quality_checks": True
        }
    )
    
    metrics_stage = StageConfiguration(
        stage_id="county_metrics",
        stage_type=ProcessingStage.METRICS,
        stage_name="County-Level Climate Metrics",
        package_name="county_climate.metrics",
        entry_point="process_county_metrics",
        depends_on=["climate_means"],
        trigger_type=TriggerType.DEPENDENCY,
        resource_limits=ResourceLimits(
            max_memory_gb=8.0,
            max_cpu_cores=4,
            max_processing_time_hours=4.0,
            priority=8
        ),
        stage_config={
            "metrics": ["mean", "percentiles", "trends"],
            "county_boundaries": "2024_census",
            "spatial_aggregation": "area_weighted_mean"
        }
    )
    
    extremes_stage = StageConfiguration(
        stage_id="climate_extremes",
        stage_type=ProcessingStage.EXTREMES,
        stage_name="Climate Extremes Analysis",
        package_name="county_climate.metrics.extremes",
        entry_point="process_extremes",
        depends_on=["county_metrics"],
        trigger_type=TriggerType.DEPENDENCY,
        resource_limits=ResourceLimits(
            max_memory_gb=6.0,
            max_cpu_cores=4,
            max_processing_time_hours=2.0,
            priority=6
        ),
        stage_config={
            "extreme_indices": ["heat_waves", "cold_spells", "heavy_precipitation"],
            "threshold_percentiles": [90, 95, 99]
        }
    )
    
    visualization_stage = StageConfiguration(
        stage_id="visualization",
        stage_type=ProcessingStage.VISUALIZATION,
        stage_name="Data Visualization",
        package_name="county_climate.visualization",
        entry_point="create_visualizations",
        depends_on=["county_metrics", "climate_extremes"],
        optional_depends_on=["climate_extremes"],
        trigger_type=TriggerType.DEPENDENCY,
        resource_limits=ResourceLimits(
            max_memory_gb=4.0,
            max_cpu_cores=2,
            max_processing_time_hours=1.0,
            priority=3
        ),
        stage_config={
            "output_formats": ["png", "html", "pdf"],
            "create_interactive_maps": True,
            "create_summary_reports": True
        }
    )
    
    validation_stage = StageConfiguration(
        stage_id="final_validation",
        stage_type=ProcessingStage.VALIDATION,
        stage_name="Complete Pipeline Validation",
        package_name="county_climate.shared.validation",
        entry_point="validate_complete_pipeline",
        depends_on=["visualization"],
        trigger_type=TriggerType.DEPENDENCY,
        resource_limits=ResourceLimits(
            max_memory_gb=2.0,
            max_cpu_cores=1,
            max_processing_time_hours=0.5,
            priority=5
        )
    )
    
    # Define data flows between stages
    data_flows = [
        DataFlowConfiguration(
            flow_id="means_to_metrics",
            source_stage="climate_means",
            target_stage="county_metrics",
            flow_type=DataFlowType.FILE_BASED,
            data_contracts=["ClimateDatasetContract"],
            quality_requirements={"min_quality_score": 0.95},
            enable_monitoring=True
        ),
        DataFlowConfiguration(
            flow_id="metrics_to_extremes",
            source_stage="county_metrics", 
            target_stage="climate_extremes",
            flow_type=DataFlowType.FILE_BASED,
            data_contracts=["CountyMetricsContract"],
            quality_requirements={"min_completeness": 0.90}
        )
    ]
    
    return PipelineConfiguration(
        pipeline_id="full_climate_pipeline",
        pipeline_name="Complete Climate Data Processing Pipeline",
        pipeline_version="2.0.0",
        environment=EnvironmentType.PRODUCTION,
        base_data_path=Path("/data/climate"),
        temp_data_path=Path("/tmp/climate_full"),
        log_path=Path("/var/log/climate/full_pipeline.log"),
        stages=[means_stage, metrics_stage, extremes_stage, visualization_stage, validation_stage],
        data_flows=data_flows,
        global_resource_limits=ResourceLimits(
            max_memory_gb=32.0,
            max_cpu_cores=16,
            max_processing_time_hours=24.0
        ),
        global_config={
            "data_source": "NEX-GDDP-CMIP6",
            "model": "NorESM2-LM",
            "output_crs": "EPSG:4326",
            "enable_caching": True,
            "cache_directory": "/data/cache",
            "notification_email": "admin@climate-processing.org"
        },
        enable_monitoring=True,
        monitoring_interval_seconds=60,
        continue_on_stage_failure=False,
        description="Complete climate data processing pipeline from means through visualization",
        tags=["production", "complete", "means", "metrics", "extremes", "visualization"]
    )


def create_development_pipeline() -> PipelineConfiguration:
    """Create a pipeline optimized for development and testing."""
    
    quick_means_stage = StageConfiguration(
        stage_id="dev_means",
        stage_type=ProcessingStage.MEANS,
        stage_name="Development Means Processing",
        package_name="county_climate.means",
        entry_point="process_region",
        trigger_type=TriggerType.MANUAL,
        resource_limits=ResourceLimits(
            max_memory_gb=4.0,
            max_cpu_cores=2,
            max_processing_time_hours=1.0,
            priority=5
        ),
        stage_config={
            "variables": ["temperature"],  # Just one variable for speed
            "regions": ["CONUS"],  # Just one region
            "scenarios": ["historical"],
            "year_range": [2015, 2016],  # Just 2 years for quick testing
            "batch_size_years": 2,
            "multiprocessing_workers": 2,
            "enable_debug_output": True
        },
        environment_overrides={
            EnvironmentType.DEVELOPMENT: {
                "stage_config": {
                    "enable_debug_output": True,
                    "save_intermediate_files": True
                }
            }
        }
    )
    
    quick_metrics_stage = StageConfiguration(
        stage_id="dev_metrics",
        stage_type=ProcessingStage.METRICS,
        stage_name="Development Metrics Processing",
        package_name="county_climate.metrics",
        entry_point="process_county_metrics",
        depends_on=["dev_means"],
        trigger_type=TriggerType.DEPENDENCY,
        resource_limits=ResourceLimits(
            max_memory_gb=2.0,
            max_cpu_cores=2,
            max_processing_time_hours=0.5,
            priority=5
        ),
        stage_config={
            "metrics": ["mean"],  # Just basic metrics
            "sample_counties": 10,  # Process only 10 counties for speed
            "enable_debug_output": True
        }
    )
    
    return PipelineConfiguration(
        pipeline_id="development_pipeline",
        pipeline_name="Development and Testing Pipeline",
        pipeline_version="1.0.0-dev",
        environment=EnvironmentType.DEVELOPMENT,
        base_data_path=Path("/tmp/dev_climate"),
        temp_data_path=Path("/tmp/dev_climate/temp"),
        log_path=Path("/tmp/dev_climate/logs/dev.log"),
        stages=[quick_means_stage, quick_metrics_stage],
        global_resource_limits=ResourceLimits(
            max_memory_gb=8.0,
            max_cpu_cores=4,
            max_processing_time_hours=2.0
        ),
        global_config={
            "debug_mode": True,
            "enable_profiling": True,
            "save_intermediate_results": True,
            "fast_mode": True
        },
        enable_monitoring=True,
        monitoring_interval_seconds=10,  # More frequent monitoring for dev
        continue_on_stage_failure=True,  # Continue for debugging
        description="Fast pipeline for development and testing",
        tags=["development", "testing", "fast", "debug"]
    )


def create_conus_temperature_profile() -> ProcessingProfile:
    """Create a processing profile for CONUS temperature analysis."""
    
    return ProcessingProfile(
        profile_name="conus_temperature_comprehensive",
        description="Comprehensive temperature analysis for Continental United States",
        regions=[Region.CONUS],
        variables=[
            ClimateVariable.TEMPERATURE,
            ClimateVariable.MAX_TEMPERATURE,
            ClimateVariable.MIN_TEMPERATURE
        ],
        scenarios=[Scenario.HISTORICAL, Scenario.SSP245, Scenario.SSP585],
        year_ranges=[(1980, 2014), (2015, 2044), (2045, 2074), (2075, 2100)],
        enable_means=True,
        enable_metrics=True,
        enable_extremes=True,
        enable_visualization=True,
        max_parallel_regions=1,  # Only one region
        max_parallel_variables=3,  # All temperature variables
        memory_per_process_gb=6.0
    )


def create_multi_region_precipitation_profile() -> ProcessingProfile:
    """Create a processing profile for precipitation analysis across multiple regions."""
    
    return ProcessingProfile(
        profile_name="multi_region_precipitation",
        description="Precipitation analysis for all US regions",
        regions=[Region.CONUS, Region.ALASKA, Region.HAWAII, Region.PUERTO_RICO],
        variables=[ClimateVariable.PRECIPITATION],
        scenarios=[Scenario.HISTORICAL, Scenario.SSP245],
        year_ranges=[(1990, 2020), (2021, 2050)],
        enable_means=True,
        enable_metrics=True,
        enable_extremes=False,  # Skip extremes for faster processing
        enable_visualization=False,  # Skip visualization for faster processing
        max_parallel_regions=2,  # Process 2 regions at once
        max_parallel_variables=1,  # Only one variable
        memory_per_process_gb=4.0
    )


def create_sample_config_files() -> Dict[str, Any]:
    """Create sample configuration files as dictionaries for export."""
    
    configs = {}
    
    # Basic means pipeline
    basic_config = create_basic_means_pipeline()
    configs["pipeline_basic_means.yaml"] = basic_config.dict()
    
    # Full pipeline
    full_config = create_full_pipeline()
    configs["pipeline_full_processing.yaml"] = full_config.dict()
    
    # Development pipeline
    dev_config = create_development_pipeline()
    configs["pipeline_development.yaml"] = dev_config.dict()
    
    # Processing profiles
    conus_profile = create_conus_temperature_profile()
    configs["profile_conus_temperature.yaml"] = conus_profile.dict()
    
    precip_profile = create_multi_region_precipitation_profile()
    configs["profile_multi_region_precipitation.yaml"] = precip_profile.dict()
    
    # Environment-specific overrides example
    configs["environment_overrides_example.yaml"] = {
        "pipeline_id": "example_with_environments",
        "pipeline_name": "Pipeline with Environment Overrides",
        "base_data_path": "/data/default",
        "max_workers": 4,
        "environment_overrides": {
            "development": {
                "base_data_path": "/tmp/dev_data",
                "max_workers": 2,
                "enable_debug": True
            },
            "staging": {
                "base_data_path": "/staging/data",
                "max_workers": 6,
                "enable_monitoring": True
            },
            "production": {
                "base_data_path": "/prod/data",
                "max_workers": 16,
                "enable_monitoring": True,
                "enable_alerting": True,
                "log_level": "INFO"
            }
        },
        "stages": [
            {
                "stage_id": "example_stage",
                "stage_type": "means",
                "stage_name": "Example Stage",
                "package_name": "county_climate.means",
                "entry_point": "process_region",
                "retry_attempts": 3,
                "environment_overrides": {
                    "development": {
                        "retry_attempts": 1,  # Fail fast in dev
                        "enable_debug_output": True
                    },
                    "production": {
                        "retry_attempts": 5,  # More resilient in prod
                        "enable_performance_monitoring": True
                    }
                }
            }
        ]
    }
    
    return configs


if __name__ == "__main__":
    """Export sample configurations to files."""
    import yaml
    from pathlib import Path
    
    output_dir = Path("sample_configs")
    output_dir.mkdir(exist_ok=True)
    
    configs = create_sample_config_files()
    
    for filename, config_data in configs.items():
        output_file = output_dir / filename
        with open(output_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        print(f"Created: {output_file}")
    
    print(f"\nSample configurations created in {output_dir}/")
    print("Files created:")
    for filename in configs.keys():
        print(f"  - {filename}")