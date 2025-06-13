"""Precipitation-specific validation and investigation."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from ..core.validator import BaseValidator, ValidationResult, ValidationSeverity
from ..core.config import ValidationConfig


class PrecipitationValidator(BaseValidator):
    """
    Specialized validator for precipitation data quality and relationships.
    Investigates specific precipitation data inconsistencies and patterns.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None,
                 output_dir: Optional[Path] = None):
        """Initialize precipitation validator."""
        super().__init__(name="precipitation_validator", output_dir=output_dir)
        self.config = config or ValidationConfig()
        
    def validate(self, data: pd.DataFrame, dataset_path: str = "climate_data.csv") -> ValidationResult:
        """
        Validate precipitation-specific patterns and relationships.
        
        Args:
            data: Climate data DataFrame
            dataset_path: Path to dataset for reporting
            
        Returns:
            ValidationResult with precipitation findings
        """
        self.logger.info(f"Starting precipitation validation for {dataset_path}")
        self._initialize_result(dataset_path)
        
        self.df = data
        
        # Run precipitation-specific validations
        validation_results = {}
        
        # Check precipitation vs high precipitation days relationship
        precip_relationship = self._validate_precipitation_relationships()
        validation_results['precipitation_relationships'] = precip_relationship
        
        # Investigate problem patterns
        problem_patterns = self._investigate_problem_patterns()
        validation_results['problem_patterns'] = problem_patterns
        
        # Geographic distribution of issues
        geographic_issues = self._analyze_geographic_distribution()
        validation_results['geographic_distribution'] = geographic_issues
        
        # Temporal investigation
        temporal_patterns = self._investigate_temporal_patterns()
        validation_results['temporal_patterns'] = temporal_patterns
        
        # Store results
        self.result.metrics['precipitation_analysis'] = validation_results
        
        return self._finalize_result()
    
    def _validate_precipitation_relationships(self) -> Dict:
        """Validate relationships between precipitation metrics."""
        self.logger.info("Validating precipitation relationships...")
        
        results = {}
        
        # Check if required columns exist
        precip_cols = ['annual_precip_mm', 'precip_gt_50mm', 'precip_gt_100mm']
        missing_cols = [col for col in precip_cols if col not in self.df.columns]
        
        if missing_cols:
            self.result.add_issue(
                ValidationSeverity.WARNING,
                "precipitation",
                f"Missing precipitation columns: {missing_cols}",
                {"missing_columns": missing_cols}
            )
            return results
        
        # Analyze relationship between annual precipitation and high precipitation days
        self.df['avg_precip_per_high_day'] = np.where(
            self.df['precip_gt_50mm'] > 0,
            self.df['annual_precip_mm'] / self.df['precip_gt_50mm'],
            0
        )
        
        # Find problematic cases
        # Case 1: High annual precipitation but few high precipitation days
        high_annual_low_days = self.df[
            (self.df['annual_precip_mm'] > 2000) & 
            (self.df['precip_gt_50mm'] < 10)
        ]
        
        if len(high_annual_low_days) > 0:
            self.result.add_issue(
                ValidationSeverity.WARNING,
                "precipitation",
                f"Found {len(high_annual_low_days)} records with high annual precipitation but few high precipitation days",
                {
                    "count": len(high_annual_low_days),
                    "example_geoids": high_annual_low_days['GEOID'].unique()[:5].tolist()
                }
            )
        
        results['high_annual_low_days'] = len(high_annual_low_days)
        
        # Case 2: Low annual precipitation but many high precipitation days
        low_annual_high_days = self.df[
            (self.df['annual_precip_mm'] < 500) & 
            (self.df['precip_gt_50mm'] > 20)
        ]
        
        if len(low_annual_high_days) > 0:
            self.result.add_issue(
                ValidationSeverity.WARNING,
                "precipitation",
                f"Found {len(low_annual_high_days)} records with low annual precipitation but many high precipitation days",
                {
                    "count": len(low_annual_high_days),
                    "example_geoids": low_annual_high_days['GEOID'].unique()[:5].tolist()
                }
            )
        
        results['low_annual_high_days'] = len(low_annual_high_days)
        
        # Case 3: Implausible average precipitation per high day
        implausible_avg = self.df[
            (self.df['avg_precip_per_high_day'] > 500) |  # >500mm per high precip day
            ((self.df['avg_precip_per_high_day'] < 50) & (self.df['precip_gt_50mm'] > 0))
        ]
        
        if len(implausible_avg) > 0:
            self.result.add_issue(
                ValidationSeverity.CRITICAL,
                "precipitation",
                f"Found {len(implausible_avg)} records with implausible average precipitation per high day",
                {"count": len(implausible_avg)}
            )
        
        results['implausible_average'] = len(implausible_avg)
        
        # Check consistency between 50mm and 100mm days
        inconsistent_days = self.df[self.df['precip_gt_100mm'] > self.df['precip_gt_50mm']]
        
        if len(inconsistent_days) > 0:
            self.result.add_issue(
                ValidationSeverity.CRITICAL,
                "precipitation",
                f"Found {len(inconsistent_days)} records where days >100mm exceed days >50mm",
                {"count": len(inconsistent_days)}
            )
        
        results['inconsistent_threshold_days'] = len(inconsistent_days)
        
        return results
    
    def _investigate_problem_patterns(self) -> Dict:
        """Investigate patterns in problematic precipitation data."""
        self.logger.info("Investigating problem patterns...")
        
        patterns = {}
        
        # Identify all problematic records
        problem_mask = (
            # High annual, low days
            ((self.df['annual_precip_mm'] > 2000) & (self.df['precip_gt_50mm'] < 10)) |
            # Low annual, high days
            ((self.df['annual_precip_mm'] < 500) & (self.df['precip_gt_50mm'] > 20)) |
            # Inconsistent thresholds
            (self.df['precip_gt_100mm'] > self.df['precip_gt_50mm'])
        )
        
        problem_records = self.df[problem_mask]
        
        if len(problem_records) == 0:
            return patterns
        
        # Pattern by scenario
        scenario_problems = problem_records.groupby('scenario').size().to_dict()
        patterns['by_scenario'] = scenario_problems
        
        # Pattern by decade
        problem_records['decade'] = (problem_records['year'] // 10) * 10
        decade_problems = problem_records.groupby('decade').size().to_dict()
        patterns['by_decade'] = decade_problems
        
        # Counties with persistent problems
        county_problem_counts = problem_records.groupby('GEOID').size()
        persistent_problem_counties = county_problem_counts[county_problem_counts > 10]
        patterns['persistent_problem_counties'] = {
            'count': len(persistent_problem_counties),
            'top_counties': persistent_problem_counties.head(10).to_dict()
        }
        
        # Correlation with other metrics
        if 'mean_temp_c' in self.df.columns:
            # Check if problems correlate with extreme temperatures
            problem_temps = problem_records['mean_temp_c'].describe().to_dict()
            normal_temps = self.df[~problem_mask]['mean_temp_c'].describe().to_dict()
            
            patterns['temperature_correlation'] = {
                'problem_records_temp_stats': problem_temps,
                'normal_records_temp_stats': normal_temps
            }
        
        return patterns
    
    def _analyze_geographic_distribution(self) -> Dict:
        """Analyze geographic distribution of precipitation issues."""
        self.logger.info("Analyzing geographic distribution of issues...")
        
        distribution = {}
        
        # Identify problem records
        problem_mask = (
            ((self.df['annual_precip_mm'] > 2000) & (self.df['precip_gt_50mm'] < 10)) |
            ((self.df['annual_precip_mm'] < 500) & (self.df['precip_gt_50mm'] > 20)) |
            (self.df['precip_gt_100mm'] > self.df['precip_gt_50mm'])
        )
        
        problem_records = self.df[problem_mask]
        
        if len(problem_records) == 0:
            return distribution
        
        # Group by state (first 2 digits of GEOID)
        problem_records['state_fips'] = problem_records['GEOID'].str[:2]
        state_issues = problem_records.groupby('state_fips').size().to_dict()
        
        distribution['issues_by_state'] = state_issues
        distribution['top_problem_states'] = dict(
            sorted(state_issues.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        # Identify spatial clusters
        problem_counties = problem_records['GEOID'].unique()
        distribution['total_problem_counties'] = len(problem_counties)
        
        # Check if problems are concentrated in specific regions
        county_issue_density = problem_records.groupby('GEOID').size()
        high_density_counties = county_issue_density[county_issue_density > 50]
        
        if len(high_density_counties) > 0:
            distribution['high_issue_density_counties'] = {
                'count': len(high_density_counties),
                'counties': high_density_counties.head(10).to_dict()
            }
            
            self.result.add_issue(
                ValidationSeverity.WARNING,
                "precipitation_geographic",
                f"Found {len(high_density_counties)} counties with >50 precipitation data issues",
                {"count": len(high_density_counties)}
            )
        
        return distribution
    
    def _investigate_temporal_patterns(self) -> Dict:
        """Investigate temporal patterns in precipitation issues."""
        self.logger.info("Investigating temporal patterns...")
        
        temporal = {}
        
        # Identify problem records
        problem_mask = (
            ((self.df['annual_precip_mm'] > 2000) & (self.df['precip_gt_50mm'] < 10)) |
            ((self.df['annual_precip_mm'] < 500) & (self.df['precip_gt_50mm'] > 20)) |
            (self.df['precip_gt_100mm'] > self.df['precip_gt_50mm'])
        )
        
        problem_records = self.df[problem_mask]
        
        if len(problem_records) == 0:
            return temporal
        
        # Issues by year
        yearly_issues = problem_records.groupby('year').size()
        temporal['issues_by_year'] = {
            'min_year': int(yearly_issues.index.min()),
            'max_year': int(yearly_issues.index.max()),
            'peak_year': int(yearly_issues.idxmax()),
            'peak_count': int(yearly_issues.max())
        }
        
        # Check for trends
        years = yearly_issues.index.values
        counts = yearly_issues.values
        
        if len(years) > 10:
            # Simple linear trend
            z = np.polyfit(years, counts, 1)
            temporal['trend'] = {
                'slope': float(z[0]),
                'increasing': z[0] > 0,
                'significant': abs(z[0]) > 1.0
            }
            
            if abs(z[0]) > 1.0:
                trend_dir = "increasing" if z[0] > 0 else "decreasing"
                self.result.add_issue(
                    ValidationSeverity.INFO,
                    "precipitation_temporal",
                    f"Precipitation issues show {trend_dir} trend over time",
                    {"slope": float(z[0]), "direction": trend_dir}
                )
        
        # Check for sudden changes
        if len(yearly_issues) > 1:
            year_to_year_changes = yearly_issues.diff().abs()
            max_change = year_to_year_changes.max()
            max_change_year = year_to_year_changes.idxmax()
            
            if max_change > 100:
                temporal['sudden_changes'] = {
                    'max_change': int(max_change),
                    'year': int(max_change_year)
                }
                
                self.result.add_issue(
                    ValidationSeverity.WARNING,
                    "precipitation_temporal",
                    f"Sudden change in precipitation issues: {int(max_change)} increase in year {int(max_change_year)}",
                    {"change": int(max_change), "year": int(max_change_year)}
                )
        
        return temporal