"""Base validator class and validation result structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import pandas as pd

from pydantic import BaseModel, Field


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(BaseModel):
    """Individual validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ValidationResult(BaseModel):
    """Complete validation result for a dataset."""
    validator_name: str
    dataset_path: str
    start_time: datetime
    end_time: Optional[datetime] = None
    issues: List[ValidationIssue] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    quality_score: Optional[str] = None
    passed: bool = True
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate validation duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def issue_count(self) -> Dict[str, int]:
        """Count issues by severity."""
        counts = {severity.value: 0 for severity in ValidationSeverity}
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return counts
    
    def add_issue(self, severity: ValidationSeverity, category: str, 
                  message: str, details: Optional[Dict[str, Any]] = None):
        """Add a validation issue."""
        issue = ValidationIssue(
            severity=severity,
            category=category,
            message=message,
            details=details
        )
        self.issues.append(issue)
        
        # Update passed status based on critical issues
        if severity == ValidationSeverity.CRITICAL:
            self.passed = False


class BaseValidator(ABC):
    """Abstract base class for all validators."""
    
    def __init__(self, name: str, output_dir: Optional[Path] = None):
        """Initialize validator."""
        self.name = name
        self.output_dir = output_dir or Path("validation_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Initialize result
        self.result: Optional[ValidationResult] = None
        
    @abstractmethod
    def validate(self, data: pd.DataFrame, **kwargs) -> ValidationResult:
        """
        Perform validation on the dataset.
        
        Args:
            data: DataFrame to validate
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationResult object with all findings
        """
        pass
    
    def _initialize_result(self, dataset_path: str) -> ValidationResult:
        """Initialize a new validation result."""
        self.result = ValidationResult(
            validator_name=self.name,
            dataset_path=dataset_path,
            start_time=datetime.now()
        )
        return self.result
    
    def _finalize_result(self) -> ValidationResult:
        """Finalize the validation result."""
        if self.result:
            self.result.end_time = datetime.now()
            self._calculate_quality_score()
        return self.result
    
    def _calculate_quality_score(self):
        """Calculate overall quality score based on issues."""
        if not self.result:
            return
            
        issue_counts = self.result.issue_count
        
        if issue_counts["critical"] > 0:
            self.result.quality_score = "POOR"
        elif issue_counts["warning"] > 10:
            self.result.quality_score = "FAIR"
        elif issue_counts["warning"] > 0:
            self.result.quality_score = "GOOD"
        else:
            self.result.quality_score = "EXCELLENT"
    
    def save_report(self, format: str = "json") -> Path:
        """Save validation report to file."""
        if not self.result:
            raise ValueError("No validation result to save")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_report_{timestamp}.{format}"
        filepath = self.output_dir / filename
        
        if format == "json":
            with open(filepath, "w") as f:
                f.write(self.result.model_dump_json(indent=2))
        elif format == "csv":
            # Save issues as CSV
            issues_df = pd.DataFrame([
                {
                    "severity": issue.severity.value,
                    "category": issue.category,
                    "message": issue.message,
                    "timestamp": issue.timestamp
                }
                for issue in self.result.issues
            ])
            issues_df.to_csv(filepath, index=False)
        
        self.logger.info(f"Saved validation report to {filepath}")
        return filepath