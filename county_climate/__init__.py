"""
County Climate Data Processing Package

A comprehensive package for processing county-level climate data including:
- Climate means (30-year normals) processing
- Climate metrics and extremes calculation
- Shared data contracts and validation
"""

__version__ = "0.1.0"
__author__ = "Climate Data Processing Team"

# Import main submodules
from . import means
from . import metrics
from . import shared

# Import key functionality
from .shared.contracts.climate_data import ClimateDatasetContract
from .shared.validation.validators import ClimateDataValidator

__all__ = [
    "means",
    "metrics", 
    "shared",
    "ClimateDatasetContract",
    "ClimateDataValidator"
]