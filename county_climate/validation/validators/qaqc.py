"""Comprehensive QA/QC validator for climate data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging

from ..core.validator import BaseValidator, ValidationResult, ValidationSeverity
from ..core.config import ValidationConfig


class QAQCValidator(BaseValidator):
    """
    Comprehensive QA/QC validator for climate datasets organized by county and year.
    Validates data completeness, spatial consistency, temporal consistency, 
    logical relationships, and physical plausibility.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None, 
                 output_dir: Optional[Path] = None):
        """Initialize QA/QC validator with configuration."""
        super().__init__(name="climate_qaqc", output_dir=output_dir)
        self.config = config or ValidationConfig()
        
    def validate(self, data: pd.DataFrame, dataset_path: str = "climate_data.csv") -> ValidationResult:
        """
        Perform comprehensive validation on climate dataset.
        
        Args:
            data: Climate data DataFrame
            dataset_path: Path to the dataset for reporting
            
        Returns:
            ValidationResult with all findings
        """
        self.logger.info(f"Starting QA/QC validation for {dataset_path}")
        self._initialize_result(dataset_path)
        
        # Store data for validation
        self.df = data
        
        # Log basic data info
        self._log_data_info()
        
        # Run validation checks
        if self.config.run_completeness_check:
            completeness_results = self._validate_data_completeness()
            self.result.metrics["completeness"] = completeness_results
            
        if self.config.run_spatial_check:
            spatial_results = self._validate_spatial_consistency()
            self.result.metrics["spatial"] = spatial_results
            
        if self.config.run_temporal_check:
            temporal_results = self._validate_temporal_consistency()
            self.result.metrics["temporal"] = temporal_results
            
        if self.config.run_logical_check:
            logical_results = self._validate_logical_relationships()
            self.result.metrics["logical"] = logical_results
            
        if self.config.run_plausibility_check:
            plausibility_results = self._validate_physical_plausibility()
            self.result.metrics["plausibility"] = plausibility_results
        
        return self._finalize_result()
    
    def _log_data_info(self):
        """Log basic information about the dataset."""
        self.logger.info(f"Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
        self.logger.info(f"Date range: {self.df['year'].min()} - {self.df['year'].max()}")
        self.logger.info(f"Unique counties: {self.df['GEOID'].nunique():,}")
        self.logger.info(f"Scenarios: {', '.join(self.df['scenario'].unique())}")
    
    def _validate_data_completeness(self) -> Dict:
        """Check for missing data patterns."""
        self.logger.info("Validating data completeness...")
        results = {}
        
        # Check for missing values
        missing_summary = self.df.isnull().sum()
        missing_pct = (missing_summary / len(self.df)) * 100
        
        results['missing_values'] = missing_summary[missing_summary > 0].to_dict()
        results['missing_percentages'] = missing_pct[missing_pct > 0].to_dict()
        
        # Flag metrics with high missing data
        for metric, pct in missing_pct.items():
            if pct > self.config.max_missing_per_metric:
                self.result.add_issue(
                    ValidationSeverity.WARNING,
                    "completeness",
                    f"Metric '{metric}' has {pct:.1f}% missing values",
                    {"metric": metric, "missing_pct": pct}
                )
        
        # Check for duplicate records
        duplicates = self.df.duplicated(subset=['GEOID', 'year', 'scenario']).sum()
        results['duplicate_records'] = duplicates
        
        if duplicates > 0:
            self.result.add_issue(
                ValidationSeverity.CRITICAL,
                "completeness",
                f"Found {duplicates} duplicate records (same GEOID, year, scenario)",
                {"count": duplicates}
            )
        
        # Check temporal completeness
        expected_years = self.config.expected_years
        incomplete_series = []
        
        for (geoid, scenario), group in self.df.groupby(['GEOID', 'scenario']):
            years_present = set(group['year'])
            missing_years = expected_years - years_present
            
            if missing_years:
                incomplete_series.append({
                    'GEOID': geoid,
                    'scenario': scenario,
                    'missing_years': len(missing_years),
                    'total_expected': len(expected_years)
                })
        
        results['incomplete_time_series'] = len(incomplete_series)
        results['incomplete_series_sample'] = incomplete_series[:10]
        
        if incomplete_series:
            self.result.add_issue(
                ValidationSeverity.WARNING,
                "completeness",
                f"Found {len(incomplete_series)} county-scenario combinations with incomplete time series",
                {"count": len(incomplete_series)}
            )
        
        return results
    
    def _validate_spatial_consistency(self) -> Dict:
        """Validate spatial patterns and county coverage."""
        self.logger.info("Validating spatial consistency...")
        results = {}
        
        # Check county coverage across scenarios
        county_scenario_coverage = self.df.groupby('GEOID')['scenario'].apply(
            lambda x: set(x.unique())
        ).to_dict()
        
        all_scenarios = set(self.config.expected_scenarios)
        counties_missing_scenarios = {}
        
        for geoid, scenarios in county_scenario_coverage.items():
            missing = all_scenarios - scenarios
            if missing:
                counties_missing_scenarios[geoid] = list(missing)
        
        results['counties_missing_scenarios'] = len(counties_missing_scenarios)
        
        if counties_missing_scenarios:
            self.result.add_issue(
                ValidationSeverity.WARNING,
                "spatial",
                f"Found {len(counties_missing_scenarios)} counties missing one or more scenarios",
                {"count": len(counties_missing_scenarios)}
            )
        
        # Check actual vs expected county count
        actual_counties = self.df['GEOID'].nunique()
        expected_counties = self.config.expected_counties
        results['actual_counties'] = actual_counties
        results['expected_counties'] = expected_counties
        
        if actual_counties < expected_counties * 0.95:  # Allow 5% tolerance
            self.result.add_issue(
                ValidationSeverity.WARNING,
                "spatial",
                f"Dataset has {actual_counties} counties, expected ~{expected_counties}",
                {"actual": actual_counties, "expected": expected_counties}
            )
        
        # Check for spatial outliers
        results['spatial_outliers'] = self._detect_spatial_outliers()
        
        return results
    
    def _detect_spatial_outliers(self) -> Dict:
        """Detect counties with extreme climate values."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        climate_metrics = [col for col in numeric_cols 
                          if col not in ['GEOID', 'year', 'pixel_count']]
        
        spatial_outliers = {}
        
        for metric in climate_metrics[:10]:  # Limit to first 10 metrics for efficiency
            outlier_counties = []
            
            for (year, scenario), group in self.df.groupby(['year', 'scenario']):
                Q1 = group[metric].quantile(0.25)
                Q3 = group[metric].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.config.iqr_multiplier * IQR
                upper_bound = Q3 + self.config.iqr_multiplier * IQR
                
                outliers = group[
                    (group[metric] < lower_bound) | (group[metric] > upper_bound)
                ]['GEOID'].tolist()
                
                outlier_counties.extend(outliers)
            
            if outlier_counties:
                outlier_freq = pd.Series(outlier_counties).value_counts()
                spatial_outliers[metric] = outlier_freq.head(5).to_dict()
        
        return spatial_outliers
    
    def _validate_temporal_consistency(self) -> Dict:
        """Validate temporal trends and consistency."""
        self.logger.info("Validating temporal consistency...")
        results = {}
        
        # Check for unrealistic year-to-year changes
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        climate_metrics = [col for col in numeric_cols 
                          if col not in ['GEOID', 'year', 'pixel_count']]
        
        temporal_issues = {}
        
        for metric in climate_metrics[:5]:  # Sample metrics for efficiency
            extreme_changes = []
            
            for (geoid, scenario), group in self.df.groupby(['GEOID', 'scenario']):
                if len(group) < 2:
                    continue
                
                group_sorted = group.sort_values('year')
                year_diff = group_sorted[metric].diff().abs()
                
                # Define thresholds based on metric type
                if 'temp' in metric.lower():
                    threshold = self.config.max_year_to_year_temp_change
                elif 'precip' in metric.lower():
                    threshold = self.config.max_year_to_year_precip_change
                else:
                    threshold = group_sorted[metric].std() * 3
                
                extreme_idx = year_diff > threshold
                if extreme_idx.any():
                    extreme_changes.append({
                        'GEOID': geoid,
                        'scenario': scenario,
                        'years': group_sorted.loc[extreme_idx, 'year'].tolist()
                    })
            
            if extreme_changes:
                temporal_issues[metric] = len(extreme_changes)
                self.result.add_issue(
                    ValidationSeverity.WARNING,
                    "temporal",
                    f"Metric '{metric}' has {len(extreme_changes)} extreme year-to-year changes",
                    {"metric": metric, "count": len(extreme_changes)}
                )
        
        results['temporal_issues'] = temporal_issues
        
        # Check for monotonic trends that might indicate errors
        results['monotonic_trends'] = self._check_monotonic_trends()
        
        return results
    
    def _check_monotonic_trends(self) -> Dict:
        """Check for unrealistic monotonic trends."""
        monotonic_counties = {}
        
        # Check temperature trends
        for temp_metric in ['mean_temp_c', 'min_temp_c', 'max_temp_c']:
            if temp_metric not in self.df.columns:
                continue
                
            for (geoid, scenario), group in self.df.groupby(['GEOID', 'scenario']):
                if len(group) < 20:  # Need sufficient data
                    continue
                
                sorted_group = group.sort_values('year')
                values = sorted_group[temp_metric].values
                
                # Check if strictly increasing/decreasing
                if np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0):
                    if geoid not in monotonic_counties:
                        monotonic_counties[geoid] = []
                    monotonic_counties[geoid].append(f"{temp_metric}_{scenario}")
        
        if monotonic_counties:
            self.result.add_issue(
                ValidationSeverity.INFO,
                "temporal",
                f"Found {len(monotonic_counties)} counties with monotonic temperature trends",
                {"count": len(monotonic_counties)}
            )
        
        return {"counties_with_monotonic_trends": len(monotonic_counties)}
    
    def _validate_logical_relationships(self) -> Dict:
        """Validate logical relationships between variables."""
        self.logger.info("Validating logical relationships...")
        results = {}
        
        # Temperature relationships: min <= mean <= max
        temp_cols = ['min_temp_c', 'mean_temp_c', 'max_temp_c']
        if all(col in self.df.columns for col in temp_cols):
            tolerance = self.config.temp_logic_tolerance
            
            # Check min <= mean
            min_mean_violations = self.df[
                self.df['min_temp_c'] > self.df['mean_temp_c'] + tolerance
            ]
            
            # Check mean <= max
            mean_max_violations = self.df[
                self.df['mean_temp_c'] > self.df['max_temp_c'] + tolerance
            ]
            
            # Check min <= max
            min_max_violations = self.df[
                self.df['min_temp_c'] > self.df['max_temp_c'] + tolerance
            ]
            
            results['temperature_logic_violations'] = {
                'min_gt_mean': len(min_mean_violations),
                'mean_gt_max': len(mean_max_violations),
                'min_gt_max': len(min_max_violations)
            }
            
            total_violations = len(min_mean_violations) + len(mean_max_violations) + len(min_max_violations)
            if total_violations > 0:
                self.result.add_issue(
                    ValidationSeverity.CRITICAL,
                    "logical",
                    f"Found {total_violations} temperature relationship violations",
                    results['temperature_logic_violations']
                )
        
        # Precipitation relationships
        if 'annual_precip_mm' in self.df.columns and 'precip_gt_50mm' in self.df.columns:
            # Days with >50mm precip shouldn't exceed 365
            excess_precip_days = self.df[self.df['precip_gt_50mm'] > 365]
            
            if len(excess_precip_days) > 0:
                self.result.add_issue(
                    ValidationSeverity.CRITICAL,
                    "logical",
                    f"Found {len(excess_precip_days)} records with >365 high precipitation days",
                    {"count": len(excess_precip_days)}
                )
            
            results['excess_precipitation_days'] = len(excess_precip_days)
        
        return results
    
    def _validate_physical_plausibility(self) -> Dict:
        """Check if values are within physically plausible ranges."""
        self.logger.info("Validating physical plausibility...")
        results = {}
        
        plausibility_issues = {}
        
        # Check temperature ranges
        for metric, (min_val, max_val) in self.config.temperature_ranges.items():
            if metric not in self.df.columns:
                continue
                
            out_of_range = self.df[
                (self.df[metric] < min_val) | (self.df[metric] > max_val)
            ]
            
            if len(out_of_range) > 0:
                plausibility_issues[metric] = len(out_of_range)
                self.result.add_issue(
                    ValidationSeverity.CRITICAL,
                    "plausibility",
                    f"Metric '{metric}' has {len(out_of_range)} values outside plausible range [{min_val}, {max_val}]",
                    {"metric": metric, "count": len(out_of_range), "range": [min_val, max_val]}
                )
        
        # Check precipitation ranges
        for metric, (min_val, max_val) in self.config.precipitation_ranges.items():
            if metric not in self.df.columns:
                continue
                
            out_of_range = self.df[
                (self.df[metric] < min_val) | (self.df[metric] > max_val)
            ]
            
            if len(out_of_range) > 0:
                plausibility_issues[metric] = len(out_of_range)
                self.result.add_issue(
                    ValidationSeverity.CRITICAL,
                    "plausibility",
                    f"Metric '{metric}' has {len(out_of_range)} values outside plausible range [{min_val}, {max_val}]",
                    {"metric": metric, "count": len(out_of_range), "range": [min_val, max_val]}
                )
        
        results['plausibility_issues'] = plausibility_issues
        
        return results