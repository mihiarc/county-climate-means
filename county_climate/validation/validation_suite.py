"""
Comprehensive validation suite integrating all enhanced validators.

This module provides a unified interface for running all validation
checks with enhanced features including geographic visualization,
advanced pattern detection, and comprehensive reporting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal
import logging
import json
from datetime import datetime

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from .core.config import ValidationConfig
from .core.validator import ValidationResult
from .validators.qaqc_fixed import QAQCValidator
from .validators.spatial_fixed import SpatialOutliersValidator
from .validators.enhanced_spatial import EnhancedSpatialOutliersValidator
from .validators.enhanced_precipitation import EnhancedPrecipitationValidator
from .visualization.climate_visualizer import ClimateVisualizer
from .visualization.geographic_visualizer import GeographicVisualizer


logger = logging.getLogger(__name__)


class ValidationSuite:
    """
    Comprehensive validation suite with all enhanced features.
    
    Provides:
    - Unified interface for all validators
    - Geographic visualization integration
    - Advanced reporting capabilities
    - Executive summaries
    - Export to multiple formats
    """
    
    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize validation suite.
        
        Args:
            config: Validation configuration
            output_dir: Directory for outputs
        """
        self.config = config or ValidationConfig()
        self.output_dir = Path(output_dir) if output_dir else Path("validation_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize validators
        self.validators = {
            'qaqc': QAQCValidator(config, self.output_dir),
            'spatial': EnhancedSpatialOutliersValidator(config, output_dir=self.output_dir),
            'precipitation': EnhancedPrecipitationValidator(config, self.output_dir)
        }
        
        # Results storage
        self.results = {}
        self.df = None
        
    def run_all_validations(
        self,
        data: pd.DataFrame,
        validators_to_run: Optional[List[str]] = None,
        include_geographic: bool = True,
        create_dashboards: bool = True,
        export_format: Literal['json', 'html', 'pdf', 'all'] = 'json'
    ) -> Dict[str, ValidationResult]:
        """
        Run all validation checks with enhanced features.
        
        Args:
            data: Climate data DataFrame
            validators_to_run: Specific validators to run (default: all)
            include_geographic: Include geographic visualizations
            create_dashboards: Create visualization dashboards
            export_format: Export format for reports
            
        Returns:
            Dictionary of validation results
        """
        logger.info("Starting comprehensive validation suite")
        self.df = data
        
        # Determine which validators to run
        if validators_to_run is None:
            validators_to_run = list(self.validators.keys())
        
        # Run validators
        for name in validators_to_run:
            if name in self.validators:
                logger.info(f"Running {name} validator...")
                self.results[name] = self.validators[name].validate(data)
            else:
                logger.warning(f"Unknown validator: {name}")
        
        # Create visualizations if requested
        if create_dashboards:
            self._create_visualization_dashboards()
        
        # Create geographic visualizations if requested
        if include_geographic and self._check_geographic_available():
            self._create_geographic_visualizations()
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        # Export in requested formats
        self._export_report(report, export_format)
        
        # Generate executive summary
        self.generate_executive_report()
        
        return self.results
    
    def create_outlier_county_profiles(self, top_n: int = 20) -> Dict:
        """
        Create detailed profiles for top outlier counties.
        
        Args:
            top_n: Number of counties to profile
            
        Returns:
            Dictionary of county profiles
        """
        logger.info(f"Creating profiles for top {top_n} outlier counties")
        
        if 'spatial' in self.validators and self.df is not None:
            spatial_validator = self.validators['spatial']
            profiles = spatial_validator.create_county_profiles(top_n=top_n)
            return profiles
        else:
            logger.warning("Spatial validator not available or no data loaded")
            return {}
    
    def create_climate_change_maps(
        self,
        scenarios: Optional[List[str]] = None
    ) -> None:
        """
        Create climate change projection maps.
        
        Args:
            scenarios: Climate scenarios to map
        """
        if not self._check_geographic_available():
            logger.warning("Geographic visualization not available")
            return
            
        logger.info("Creating climate change projection maps")
        
        geo_viz = GeographicVisualizer(self.df, self.output_dir / 'geographic')
        
        if scenarios is None:
            scenarios = ['ssp245', 'ssp585']
        
        for scenario in scenarios:
            if scenario in self.df['scenario'].unique():
                geo_viz.create_change_maps(scenario=scenario)
    
    def generate_executive_report(self) -> None:
        """Generate executive summary report."""
        logger.info("Generating executive summary report")
        
        summary = {
            'report_date': datetime.now().isoformat(),
            'data_summary': {
                'total_records': len(self.df) if self.df is not None else 0,
                'counties': self.df['fips'].nunique() if self.df is not None else 0,
                'year_range': f"{self.df['year'].min()}-{self.df['year'].max()}" if self.df is not None else "N/A",
                'scenarios': self.df['scenario'].unique().tolist() if self.df is not None else []
            },
            'validation_summary': {},
            'key_findings': [],
            'recommendations': []
        }
        
        # Summarize each validator's results
        for name, result in self.results.items():
            summary['validation_summary'][name] = {
                'passed': result.passed,
                'quality_score': result.quality_score,
                'issues': {
                    'critical': len([i for i in result.issues if i.severity.value == 'critical']),
                    'warning': len([i for i in result.issues if i.severity.value == 'warning']),
                    'info': len([i for i in result.issues if i.severity.value == 'info'])
                }
            }
        
        # Extract key findings
        if 'qaqc' in self.results:
            qaqc_issues = self.results['qaqc'].issues
            if any(i.severity.value == 'critical' for i in qaqc_issues):
                summary['key_findings'].append(
                    "Critical data quality issues found requiring immediate attention"
                )
        
        if 'spatial' in self.results:
            spatial_metrics = self.results['spatial'].metrics
            if 'summary' in spatial_metrics and spatial_metrics['summary'].get('persistent_outlier_counties', 0) > 10:
                summary['key_findings'].append(
                    f"Found {spatial_metrics['summary']['persistent_outlier_counties']} persistent spatial outliers"
                )
        
        if 'precipitation' in self.results:
            precip_issues = [i for i in self.results['precipitation'].issues 
                           if 'impossible' in i.category]
            if precip_issues:
                summary['key_findings'].append(
                    "Detected impossible precipitation patterns in some counties"
                )
        
        # Generate recommendations
        # Convert quality scores to numeric
        quality_scores = []
        for r in self.results.values():
            if r.quality_score:
                score_map = {'EXCELLENT': 0.95, 'GOOD': 0.85, 'FAIR': 0.75, 'POOR': 0.5}
                quality_scores.append(score_map.get(r.quality_score, 0.7))
        
        overall_quality = np.mean(quality_scores) if quality_scores else 0.7
        
        if overall_quality < 0.7:
            summary['recommendations'].append(
                "Consider comprehensive data review before using for analysis"
            )
        
        if any(not r.passed for r in self.results.values()):
            summary['recommendations'].append(
                "Address critical validation failures before proceeding"
            )
        
        # Save executive summary
        exec_path = self.output_dir / 'executive_summary.json'
        with open(exec_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also create a markdown version
        self._create_executive_summary_markdown(summary)
    
    def _create_visualization_dashboards(self) -> None:
        """Create all visualization dashboards."""
        logger.info("Creating visualization dashboards")
        
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Standard climate visualizer
        # Temporarily rename fips to GEOID for compatibility with visualizer
        df_viz = self.df.copy()
        if 'fips' in df_viz.columns and 'GEOID' not in df_viz.columns:
            df_viz['GEOID'] = df_viz['fips']
        
        climate_viz = ClimateVisualizer(viz_dir)
        climate_viz.create_overview_dashboard(df_viz)
        
        # Create temperature analysis if temperature data is present
        if any(col in df_viz.columns for col in ['mean_temp_c', 'min_temp_c', 'max_temp_c']):
            climate_viz.create_temperature_analysis(df_viz)
        
        # Create precipitation analysis if precipitation data is present  
        required_precip_cols = ['annual_precip_mm', 'precip_gt_50mm']
        has_all_precip = all(col in df_viz.columns for col in required_precip_cols)
        
        # Also check if the data is meaningful (not all zeros from placeholders)
        if has_all_precip:
            precip_data_meaningful = (
                df_viz['annual_precip_mm'].sum() > 0 and 
                df_viz['precip_gt_50mm'].sum() > 0
            )
            if precip_data_meaningful:
                climate_viz.create_precipitation_analysis(df_viz)
            else:
                logger.info("Skipping precipitation analysis - data appears to be placeholder values")
        elif 'annual_precip_mm' in df_viz.columns:
            logger.info("Skipping precipitation analysis - missing required columns for full analysis")
            
        # Create validation summary
        climate_viz.create_validation_summary_plots(self.results)
        
        # Enhanced precipitation diagnostics
        if 'precipitation' in self.validators and 'precipitation' in self.results:
            # Skip if validator wasn't run or doesn't have data
            pass
    
    def _create_geographic_visualizations(self) -> None:
        """Create geographic visualizations."""
        logger.info("Creating geographic visualizations")
        
        geo_dir = self.output_dir / 'geographic'
        geo_dir.mkdir(exist_ok=True)
        
        try:
            geo_viz = GeographicVisualizer(self.df, geo_dir)
            
            # Current and historical climate maps
            geo_viz.create_climate_maps(time_period="current")
            geo_viz.create_climate_maps(time_period="historical")
            
            # Data coverage maps
            geo_viz.create_coverage_map(metric="completeness")
            
            # Outlier maps if available
            if 'spatial' in self.results:
                # Get outlier severity data
                spatial_validator = self.validators['spatial']
                outlier_results = spatial_validator._detect_all_outliers()
                severity_df = spatial_validator.classify_outlier_severity(outlier_results)
                
                if not severity_df.empty:
                    outlier_df = severity_df[['fips', 'severity']].copy()
                    outlier_df['is_outlier'] = True
                    geo_viz.create_outlier_maps(outlier_df, "Spatial Outliers by Severity")
                    
        except Exception as e:
            logger.error(f"Error creating geographic visualizations: {e}")
    
    def _check_geographic_available(self) -> bool:
        """Check if geographic visualization is available."""
        try:
            import geopandas
            import contextily
            return True
        except ImportError:
            return False
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive validation report."""
        report = {
            'metadata': {
                'report_date': datetime.now().isoformat(),
                'validators_run': list(self.results.keys()),
                'data_records': len(self.df) if self.df is not None else 0,
                'counties': self.df['fips'].nunique() if self.df is not None else 0
            },
            'validation_results': {},
            'summary_statistics': {},
            'quality_scores': {}
        }
        
        # Add results from each validator
        for name, result in self.results.items():
            report['validation_results'][name] = {
                'passed': result.passed,
                'quality_score': result.quality_score,
                'duration': result.duration,
                'issues_summary': {
                    'total': len(result.issues),
                    'by_severity': {
                        'critical': len([i for i in result.issues if i.severity.value == 'critical']),
                        'warning': len([i for i in result.issues if i.severity.value == 'warning']),
                        'info': len([i for i in result.issues if i.severity.value == 'info'])
                    }
                },
                'key_findings': result.metrics
            }
            
            # Convert quality score to numeric if possible
            if result.quality_score:
                try:
                    score = float(result.quality_score)
                except (ValueError, TypeError):
                    # If it's a string like "EXCELLENT", map to numeric
                    score_map = {'EXCELLENT': 0.95, 'GOOD': 0.85, 'FAIR': 0.75, 'POOR': 0.5}
                    score = score_map.get(result.quality_score, 0.7)
                report['quality_scores'][name] = score
            else:
                # Calculate based on issues
                total_issues = len(result.issues)
                critical_issues = len([i for i in result.issues if i.severity.value == 'critical'])
                if critical_issues > 0:
                    score = 0.5  # Poor
                elif total_issues > 10:
                    score = 0.75  # Fair
                elif total_issues > 5:
                    score = 0.85  # Good
                else:
                    score = 0.95  # Excellent
                report['quality_scores'][name] = score
        
        # Overall quality score
        if report['quality_scores']:
            report['overall_quality_score'] = np.mean(list(report['quality_scores'].values()))
        else:
            report['overall_quality_score'] = 0.0
        
        # Add data summary statistics
        if self.df is not None:
            report['summary_statistics'] = {
                'temporal_coverage': {
                    'start_year': int(self.df['year'].min()),
                    'end_year': int(self.df['year'].max()),
                    'total_years': int(self.df['year'].nunique())
                },
                'scenarios': self.df['scenario'].unique().tolist(),
                'variables_present': [col for col in ['mean_temp_c', 'annual_precip_cm', 
                                                     'min_temp_c', 'max_temp_c'] 
                                    if col in self.df.columns]
            }
        
        return report
    
    def _export_report(self, report: Dict, format: str) -> None:
        """Export report in requested format."""
        logger.info(f"Exporting report in {format} format")
        
        if format in ['json', 'all']:
            json_path = self.output_dir / 'validation_report.json'
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyJSONEncoder)
        
        if format in ['html', 'all']:
            self._create_html_report(report)
        
        if format in ['pdf', 'all']:
            logger.info("PDF export requires additional dependencies")
            # PDF generation would require additional libraries
    
    def _create_html_report(self, report: Dict) -> None:
        """Create HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Climate Data Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .quality-score {{ font-size: 24px; font-weight: bold; }}
                .passed {{ color: #27ae60; }}
                .failed {{ color: #e74c3c; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
            </style>
        </head>
        <body>
            <h1>Climate Data Validation Report</h1>
            <p>Generated: {report['metadata']['report_date']}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Overall Quality Score: <span class="quality-score">{report['overall_quality_score']:.2f}</span></p>
                <p>Records Analyzed: {report['metadata']['data_records']:,}</p>
                <p>Counties: {report['metadata']['counties']:,}</p>
            </div>
            
            <h2>Validation Results</h2>
            <table>
                <tr>
                    <th>Validator</th>
                    <th>Status</th>
                    <th>Quality Score</th>
                    <th>Critical Issues</th>
                    <th>Warnings</th>
                </tr>
        """
        
        for name, result in report['validation_results'].items():
            status_class = 'passed' if result['passed'] else 'failed'
            status_text = 'PASSED' if result['passed'] else 'FAILED'
            
            html_content += f"""
                <tr>
                    <td>{name.upper()}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result['quality_score'] if isinstance(result['quality_score'], str) else f"{result['quality_score']:.2f}"}</td>
                    <td>{result['issues_summary']['by_severity']['critical']}</td>
                    <td>{result['issues_summary']['by_severity']['warning']}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        html_path = self.output_dir / 'validation_report.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
    
    def _create_executive_summary_markdown(self, summary: Dict) -> None:
        """Create markdown version of executive summary."""
        md_content = f"""# Climate Data Validation Executive Summary

**Report Date:** {summary['report_date']}

## Data Overview
- **Total Records:** {summary['data_summary']['total_records']:,}
- **Counties:** {summary['data_summary']['counties']:,}
- **Time Period:** {summary['data_summary']['year_range']}
- **Scenarios:** {', '.join(summary['data_summary']['scenarios'])}

## Validation Summary

| Validator | Status | Quality Score | Critical Issues | Warnings |
|-----------|--------|--------------|-----------------|----------|
"""
        
        for name, result in summary['validation_summary'].items():
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            # Format quality score appropriately
            if isinstance(result['quality_score'], str):
                score_display = result['quality_score']
            else:
                score_display = f"{result['quality_score']:.2f}"
            md_content += f"| {name.upper()} | {status} | {score_display} | "
            md_content += f"{result['issues']['critical']} | {result['issues']['warning']} |\n"
        
        md_content += f"""
## Key Findings

"""
        for finding in summary['key_findings']:
            md_content += f"- {finding}\n"
        
        md_content += f"""
## Recommendations

"""
        for rec in summary['recommendations']:
            md_content += f"1. {rec}\n"
        
        md_path = self.output_dir / 'EXECUTIVE_SUMMARY.md'
        with open(md_path, 'w') as f:
            f.write(md_content)