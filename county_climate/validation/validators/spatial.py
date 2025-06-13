"""Spatial outlier detection and analysis for climate data."""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats

from ..core.validator import BaseValidator, ValidationResult, ValidationSeverity
from ..core.config import ValidationConfig
from county_climate.shared.data import CountyBoundariesManager


class SpatialOutliersValidator(BaseValidator):
    """
    Identifies counties with extreme climate patterns using multiple statistical methods.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None,
                 shapefile_path: Optional[str] = None,
                 output_dir: Optional[Path] = None):
        """Initialize spatial outliers validator."""
        super().__init__(name="spatial_outliers", output_dir=output_dir)
        self.config = config or ValidationConfig()
        self.shapefile_path = shapefile_path
        self.gdf = None
        self._county_manager = None
        
    def validate(self, data: pd.DataFrame, dataset_path: str = "climate_data.csv") -> ValidationResult:
        """
        Identify spatial outliers in climate data.
        
        Args:
            data: Climate data DataFrame
            dataset_path: Path to dataset for reporting
            
        Returns:
            ValidationResult with outlier findings
        """
        self.logger.info(f"Starting spatial outlier detection for {dataset_path}")
        self._initialize_result(dataset_path)
        
        self.df = data
        
        # Load shapefile if provided
        if self.shapefile_path:
            self._load_shapefile()
        
        # Run outlier detection methods
        outlier_results = {}
        
        # IQR method
        iqr_outliers = self._detect_outliers_iqr()
        outlier_results['iqr_method'] = iqr_outliers
        
        # Z-score method
        zscore_outliers = self._detect_outliers_zscore()
        outlier_results['zscore_method'] = zscore_outliers
        
        # Modified Z-score method
        modified_zscore_outliers = self._detect_outliers_modified_zscore()
        outlier_results['modified_zscore_method'] = modified_zscore_outliers
        
        # Persistent outliers across methods
        persistent_outliers = self._identify_persistent_outliers(outlier_results)
        outlier_results['persistent_outliers'] = persistent_outliers
        
        # Geographic clustering of outliers
        if self.gdf is not None:
            cluster_results = self._analyze_geographic_clusters(persistent_outliers)
            outlier_results['geographic_clusters'] = cluster_results
        
        # Store results
        self.result.metrics['outlier_analysis'] = outlier_results
        
        # Report findings
        self._report_outlier_findings(outlier_results)
        
        return self._finalize_result()
    
    def _load_shapefile(self):
        """Load county shapefile for geographic analysis."""
        try:
            if self.shapefile_path:
                # Legacy: load from shapefile path
                self.gdf = gpd.read_file(self.shapefile_path)
                self.logger.info(f"Loaded shapefile with {len(self.gdf)} counties")
            else:
                # Modern: use CountyBoundariesManager
                if not self._county_manager:
                    self._county_manager = CountyBoundariesManager()
                self.gdf = self._county_manager.load_counties(simplified=False)
                self.logger.info(f"Loaded {len(self.gdf)} counties from modern format")
        except Exception as e:
            self.logger.warning(f"Could not load county boundaries: {e}")
            self.result.add_issue(
                ValidationSeverity.WARNING,
                "spatial",
                f"Failed to load county boundaries for geographic analysis: {str(e)}"
            )
    
    def _detect_outliers_iqr(self) -> Dict[str, Dict]:
        """Detect outliers using Interquartile Range method."""
        self.logger.info("Detecting outliers using IQR method...")
        
        outliers = {}
        climate_metrics = [col for col in self.config.climate_metrics 
                          if col in self.df.columns]
        
        for metric in climate_metrics:
            metric_outliers = {}
            
            for scenario in self.df['scenario'].unique():
                scenario_data = self.df[self.df['scenario'] == scenario]
                
                # Calculate IQR for each year
                yearly_outliers = []
                
                for year in scenario_data['year'].unique():
                    year_data = scenario_data[scenario_data['year'] == year]
                    
                    Q1 = year_data[metric].quantile(0.25)
                    Q3 = year_data[metric].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - self.config.iqr_multiplier * IQR
                    upper_bound = Q3 + self.config.iqr_multiplier * IQR
                    
                    outlier_mask = (year_data[metric] < lower_bound) | (year_data[metric] > upper_bound)
                    outlier_counties = year_data.loc[outlier_mask, 'GEOID'].tolist()
                    
                    if outlier_counties:
                        yearly_outliers.extend(outlier_counties)
                
                # Count frequency of outliers
                if yearly_outliers:
                    outlier_freq = pd.Series(yearly_outliers).value_counts()
                    # Keep counties that are outliers in >10% of years
                    persistent = outlier_freq[outlier_freq > len(scenario_data['year'].unique()) * 0.1]
                    metric_outliers[scenario] = persistent.to_dict()
            
            if metric_outliers:
                outliers[metric] = metric_outliers
        
        return outliers
    
    def _detect_outliers_zscore(self) -> Dict[str, Dict]:
        """Detect outliers using Z-score method."""
        self.logger.info("Detecting outliers using Z-score method...")
        
        outliers = {}
        climate_metrics = [col for col in self.config.climate_metrics 
                          if col in self.df.columns]
        
        for metric in climate_metrics:
            metric_outliers = {}
            
            for scenario in self.df['scenario'].unique():
                scenario_data = self.df[self.df['scenario'] == scenario]
                
                # Calculate mean and std for the metric across all counties and years
                mean_val = scenario_data[metric].mean()
                std_val = scenario_data[metric].std()
                
                # Calculate z-scores
                scenario_data['z_score'] = np.abs((scenario_data[metric] - mean_val) / std_val)
                
                # Find outliers
                outlier_data = scenario_data[scenario_data['z_score'] > self.config.z_score_threshold]
                
                if len(outlier_data) > 0:
                    # Count by county
                    outlier_counts = outlier_data.groupby('GEOID').size()
                    # Keep counties with multiple outlier instances
                    persistent = outlier_counts[outlier_counts > 5]
                    metric_outliers[scenario] = persistent.to_dict()
            
            if metric_outliers:
                outliers[metric] = metric_outliers
        
        return outliers
    
    def _detect_outliers_modified_zscore(self) -> Dict[str, Dict]:
        """Detect outliers using Modified Z-score (MAD) method."""
        self.logger.info("Detecting outliers using Modified Z-score method...")
        
        outliers = {}
        climate_metrics = [col for col in self.config.climate_metrics 
                          if col in self.df.columns]
        
        for metric in climate_metrics:
            metric_outliers = {}
            
            for scenario in self.df['scenario'].unique():
                scenario_data = self.df[self.df['scenario'] == scenario]
                
                # Calculate median and MAD
                median_val = scenario_data[metric].median()
                mad = np.median(np.abs(scenario_data[metric] - median_val))
                
                # Calculate modified z-scores
                if mad != 0:
                    modified_z = 0.6745 * (scenario_data[metric] - median_val) / mad
                    scenario_data['modified_z_score'] = np.abs(modified_z)
                    
                    # Find outliers
                    outlier_data = scenario_data[
                        scenario_data['modified_z_score'] > self.config.modified_z_threshold
                    ]
                    
                    if len(outlier_data) > 0:
                        # Count by county
                        outlier_counts = outlier_data.groupby('GEOID').size()
                        # Keep counties with multiple outlier instances
                        persistent = outlier_counts[outlier_counts > 5]
                        metric_outliers[scenario] = persistent.to_dict()
            
            if metric_outliers:
                outliers[metric] = metric_outliers
        
        return outliers
    
    def _identify_persistent_outliers(self, outlier_results: Dict) -> Dict[str, List[str]]:
        """Identify counties that are outliers across multiple methods and metrics."""
        self.logger.info("Identifying persistent outliers across methods...")
        
        # Collect all outlier counties
        all_outliers = []
        
        for method, method_results in outlier_results.items():
            if method == 'persistent_outliers':
                continue
                
            for metric, metric_results in method_results.items():
                for scenario, counties in metric_results.items():
                    all_outliers.extend(list(counties.keys()))
        
        # Count frequency
        outlier_freq = pd.Series(all_outliers).value_counts()
        
        # Identify persistent outliers (appear in multiple analyses)
        persistent_threshold = 3  # Must appear as outlier in at least 3 analyses
        persistent_outliers = outlier_freq[outlier_freq >= persistent_threshold]
        
        return {
            'counties': persistent_outliers.index.tolist(),
            'frequency': persistent_outliers.to_dict()
        }
    
    def _analyze_geographic_clusters(self, persistent_outliers: Dict) -> Dict:
        """Analyze geographic clustering of outliers."""
        if not persistent_outliers['counties']:
            return {}
        
        self.logger.info("Analyzing geographic clusters of outliers...")
        
        # Filter shapefile to outlier counties
        outlier_gdf = self.gdf[self.gdf['GEOID'].isin(persistent_outliers['counties'])]
        
        if len(outlier_gdf) == 0:
            return {}
        
        # Calculate centroids
        outlier_gdf['centroid'] = outlier_gdf.geometry.centroid
        
        # Simple clustering based on state
        state_clusters = outlier_gdf.groupby('STATEFP')['GEOID'].count().to_dict()
        
        return {
            'outlier_counties_by_state': state_clusters,
            'total_outlier_counties': len(outlier_gdf)
        }
    
    def _report_outlier_findings(self, outlier_results: Dict):
        """Report key findings from outlier analysis."""
        # Report persistent outliers
        if 'persistent_outliers' in outlier_results:
            persistent = outlier_results['persistent_outliers']
            if persistent['counties']:
                self.result.add_issue(
                    ValidationSeverity.INFO,
                    "spatial_outliers",
                    f"Found {len(persistent['counties'])} counties as persistent outliers across methods",
                    {
                        "count": len(persistent['counties']),
                        "top_outliers": dict(list(persistent['frequency'].items())[:5])
                    }
                )
        
        # Report method-specific findings
        for method in ['iqr_method', 'zscore_method', 'modified_zscore_method']:
            if method in outlier_results:
                method_data = outlier_results[method]
                metric_count = len(method_data)
                total_outliers = sum(
                    len(counties) 
                    for metric_data in method_data.values() 
                    for counties in metric_data.values()
                )
                
                if total_outliers > 0:
                    self.result.add_issue(
                        ValidationSeverity.INFO,
                        "spatial_outliers",
                        f"{method}: Found outliers in {metric_count} metrics with {total_outliers} total detections",
                        {"metrics": metric_count, "detections": total_outliers}
                    )