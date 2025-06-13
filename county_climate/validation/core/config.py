"""Validation configuration and constants."""

from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field


class ValidationConfig(BaseModel):
    """Configuration for validation phase."""
    
    # Data expectations
    expected_counties: int = 3109  # CONUS counties
    expected_years: Set[int] = Field(default_factory=lambda: set(range(1980, 2101)))
    expected_scenarios: List[str] = ["historical", "ssp245", "ssp585"]
    
    # Climate variable ranges
    temperature_ranges: Dict[str, tuple] = {
        "mean_temp_c": (-40, 50),
        "min_temp_c": (-50, 40),
        "max_temp_c": (-30, 60),
    }
    
    precipitation_ranges: Dict[str, tuple] = {
        "annual_precip_mm": (0, 5000),
        "precip_gt_50mm": (0, 365),
        "precip_gt_100mm": (0, 365),
    }
    
    # Outlier detection thresholds
    iqr_multiplier: float = 1.5
    z_score_threshold: float = 3.0
    modified_z_threshold: float = 3.5
    
    # Temporal consistency thresholds
    max_year_to_year_temp_change: float = 5.0  # degrees C
    max_year_to_year_precip_change: float = 500.0  # mm
    
    # Logical relationship tolerances
    temp_logic_tolerance: float = 0.1  # degrees C
    
    # Output settings
    output_dir: str = "validation_outputs"
    save_plots: bool = True
    plot_dpi: int = 300
    
    # Validation modules to run
    run_completeness_check: bool = True
    run_spatial_check: bool = True
    run_temporal_check: bool = True
    run_logical_check: bool = True
    run_plausibility_check: bool = True
    run_outlier_detection: bool = True
    run_visualization: bool = True
    
    # Climate metrics to validate
    climate_metrics: List[str] = [
        "mean_temp_c", "min_temp_c", "max_temp_c",
        "hot_days_gt_35c", "hot_days_gt_40c",
        "cold_days_lt_0c", "cold_days_lt_neg10c",
        "frost_free_days", "growing_degree_days",
        "cooling_degree_days", "heating_degree_days",
        "annual_precip_mm", "precip_gt_50mm", "precip_gt_100mm",
        "dry_days", "longest_dry_spell",
        "temp_range_c", "extreme_heat_index",
        "extreme_cold_index", "temp_variability",
        "precip_variability", "drought_index",
        "extreme_precip_index", "climate_stress_index",
        "heat_wave_events", "cold_snap_events"
    ]


class DataQualityThresholds(BaseModel):
    """Thresholds for data quality assessment."""
    
    # Missing data thresholds (percentage)
    max_missing_overall: float = 5.0
    max_missing_per_metric: float = 10.0
    max_missing_per_county: float = 15.0
    
    # Completeness thresholds
    min_counties_per_scenario: int = 3000
    min_years_per_series: int = 100
    
    # Quality score thresholds
    excellent_threshold: float = 0.01  # < 1% issues
    good_threshold: float = 0.05       # < 5% issues
    fair_threshold: float = 0.10       # < 10% issues
    # Anything above fair_threshold is considered POOR