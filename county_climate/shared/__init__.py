"""
Shared data contracts and utilities for climate data processing pipeline.

This package provides standardized interfaces between the climate means
and climate metrics processing components.
"""

__version__ = "1.0.0"

from .contracts.climate_data import (
    ClimateDatasetContract,
    ClimateVariable,
    Region,
    Scenario,
    QualityMetricsContract,
    SpatialBoundsContract,
    ProcessingMetadataContract,
    DataAccessContract,
    ValidationResultContract,
    CatalogQueryContract,
    CatalogResponseContract
)

from .contracts.pipeline_interface import (
    PipelineConfigContract,
    DataflowContract,
    ProcessingStatusContract,
    ErrorContract,
    HealthCheckContract
)

from .validation.validators import (
    ClimateDataValidator,
    FilePathValidator,
    CoordinateValidator,
    TemporalValidator
)

__all__ = [
    # Climate data contracts
    "ClimateDatasetContract",
    "ClimateVariable", 
    "Region",
    "Scenario",
    "QualityMetricsContract",
    "SpatialBoundsContract",
    "ProcessingMetadataContract",
    "DataAccessContract",
    "ValidationResultContract",
    "CatalogQueryContract",
    "CatalogResponseContract",
    
    # Pipeline interface contracts
    "PipelineConfigContract",
    "DataflowContract", 
    "ProcessingStatusContract",
    "ErrorContract",
    "HealthCheckContract",
    
    # Validators
    "ClimateDataValidator",
    "FilePathValidator",
    "CoordinateValidator", 
    "TemporalValidator",
]