"""Data contracts for validation phase (Phase 3)."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from county_climate.validation.core.validator import ValidationSeverity


class ValidationIssueContract(BaseModel):
    """Contract for individual validation issues."""
    severity: ValidationSeverity
    category: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ValidationMetricsContract(BaseModel):
    """Contract for validation metrics."""
    completeness: Optional[Dict[str, Any]] = None
    spatial: Optional[Dict[str, Any]] = None
    temporal: Optional[Dict[str, Any]] = None
    logical: Optional[Dict[str, Any]] = None
    plausibility: Optional[Dict[str, Any]] = None
    outlier_analysis: Optional[Dict[str, Any]] = None
    precipitation_analysis: Optional[Dict[str, Any]] = None


class ValidationResultContract(BaseModel):
    """Contract for complete validation results."""
    validator_name: str
    dataset_path: str
    start_time: datetime
    end_time: Optional[datetime] = None
    issues: List[ValidationIssueContract] = Field(default_factory=list)
    metrics: ValidationMetricsContract = Field(default_factory=ValidationMetricsContract)
    quality_score: Optional[str] = None
    passed: bool = True
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate validation duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def issue_summary(self) -> Dict[str, int]:
        """Get issue counts by severity."""
        summary = {"critical": 0, "warning": 0, "info": 0}
        for issue in self.issues:
            summary[issue.severity.value] += 1
        return summary


class ValidationStageInputContract(BaseModel):
    """Input contract for validation stage."""
    metrics_output_path: str = Field(description="Path to metrics output from Phase 2")
    county_shapefile_path: Optional[str] = Field(
        default=None,
        description="Path to county shapefile for spatial analysis"
    )
    output_dir: str = Field(
        default="validation_outputs",
        description="Directory for validation outputs"
    )
    run_visualizations: bool = Field(
        default=True,
        description="Whether to generate visualization plots"
    )
    validators_to_run: List[str] = Field(
        default=["qaqc", "spatial", "precipitation"],
        description="List of validators to execute"
    )


class ValidationStageOutputContract(BaseModel):
    """Output contract for validation stage."""
    validation_results: Dict[str, ValidationResultContract] = Field(
        description="Results from each validator"
    )
    overall_quality_score: str = Field(
        description="Overall quality assessment"
    )
    passed: bool = Field(
        description="Whether validation passed overall"
    )
    report_paths: Dict[str, str] = Field(
        description="Paths to generated reports"
    )
    visualization_paths: Optional[Dict[str, str]] = Field(
        default=None,
        description="Paths to generated visualizations"
    )
    summary: Dict[str, Any] = Field(
        description="Summary statistics and findings"
    )


class ValidationConfigContract(BaseModel):
    """Configuration contract for validation stage."""
    expected_counties: int = 3109
    expected_scenarios: List[str] = ["historical", "ssp245", "ssp585"]
    iqr_multiplier: float = 1.5
    z_score_threshold: float = 3.0
    max_missing_overall: float = 5.0
    max_missing_per_metric: float = 10.0
    enable_rich_progress: bool = True
    save_intermediate_results: bool = True