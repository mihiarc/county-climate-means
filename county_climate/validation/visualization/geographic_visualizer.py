"""
Geographic visualization for climate data validation.

This module provides advanced mapping and spatial visualization capabilities
for climate data validation, including choropleth maps, outlier visualization,
and climate change projections.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    import geopandas as gpd

try:
    import geopandas as gpd
    import contextily as ctx
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    gpd = None
    ctx = None

from county_climate.shared.data.county_boundaries import CountyBoundariesManager

logger = logging.getLogger(__name__)


class GeographicVisualizer:
    """
    Advanced geographic visualization for climate data validation.
    
    Provides choropleth maps, spatial pattern analysis, and climate
    change projection visualizations.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: Union[str, Path],
        county_boundaries_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the geographic visualizer.
        
        Args:
            df: Climate data DataFrame
            output_dir: Directory for saving visualizations
            county_boundaries_path: Path to county boundaries file
        """
        if not GEOSPATIAL_AVAILABLE:
            raise ImportError(
                "Geospatial libraries not available. "
                "Install with: pip install geopandas contextily"
            )
            
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize county boundaries
        self.boundaries_manager = CountyBoundariesManager()
        self.gdf = self._load_county_boundaries(county_boundaries_path)
        
        # Color schemes for different visualizations
        self.temp_colors = LinearSegmentedColormap.from_list(
            'temp', ['#2166ac', '#f7f7f7', '#b2182b']
        )
        self.precip_colors = LinearSegmentedColormap.from_list(
            'precip', ['#f7fbff', '#08519c']
        )
        self.change_colors = LinearSegmentedColormap.from_list(
            'change', ['#b2182b', '#f7f7f7', '#2166ac']
        )
        
    def _load_county_boundaries(
        self, 
        boundaries_path: Optional[Union[str, Path]] = None
    ) -> "gpd.GeoDataFrame":
        """Load county boundaries for mapping."""
        if boundaries_path:
            return gpd.read_file(boundaries_path)
        else:
            # Use the centralized boundaries manager
            return self.boundaries_manager.get_boundaries(format='geodataframe')
    
    def create_climate_maps(
        self,
        variables: Optional[List[str]] = None,
        time_period: str = "current"
    ) -> None:
        """
        Generate choropleth maps for climate variables.
        
        Args:
            variables: List of variables to map (default: all)
            time_period: "current" (2015-2020) or "historical" (1980-2014)
        """
        if variables is None:
            variables = ['mean_temp_c', 'annual_precip_cm']
            
        logger.info(f"Creating climate maps for {time_period} period")
        
        # Filter data for time period
        if time_period == "current":
            year_range = (2015, 2020)
        else:
            year_range = (1980, 2014)
            
        period_data = self.df[
            (self.df['year'] >= year_range[0]) & 
            (self.df['year'] <= year_range[1])
        ]
        
        # Calculate averages by county
        county_avgs = period_data.groupby('fips')[variables].mean()
        
        # Create maps
        fig, axes = plt.subplots(1, len(variables), figsize=(16, 8))
        if len(variables) == 1:
            axes = [axes]
            
        for idx, var in enumerate(variables):
            ax = axes[idx]
            
            # Merge data with geometry
            plot_gdf = self.gdf.merge(
                county_avgs[[var]], 
                left_on='GEOID', 
                right_index=True,
                how='left'
            )
            
            # Determine color map
            cmap = self.temp_colors if 'temp' in var else self.precip_colors
            
            # Create choropleth
            plot_gdf.plot(
                column=var,
                ax=ax,
                cmap=cmap,
                legend=True,
                legend_kwds={
                    'label': self._get_variable_label(var),
                    'orientation': 'horizontal',
                    'pad': 0.05
                },
                missing_kwds={'color': 'lightgray'}
            )
            
            # Add title
            ax.set_title(
                f"{self._get_variable_label(var)} ({year_range[0]}-{year_range[1]})",
                fontsize=14,
                fontweight='bold'
            )
            ax.axis('off')
            
        plt.tight_layout()
        output_path = self.output_dir / f"climate_maps_{time_period}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved climate maps to {output_path}")
        
    def create_outlier_maps(
        self,
        outlier_data: pd.DataFrame,
        title: str = "Spatial Outliers"
    ) -> None:
        """
        Create maps showing spatial distribution of outliers.
        
        Args:
            outlier_data: DataFrame with 'fips' and outlier indicators
            title: Map title
        """
        logger.info("Creating spatial outlier maps")
        
        # Merge outlier data with geometry
        plot_gdf = self.gdf.merge(
            outlier_data,
            left_on='GEOID',
            right_on='fips',
            how='left'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot base map (non-outliers)
        plot_gdf[plot_gdf['is_outlier'] != True].plot(
            ax=ax,
            color='lightgray',
            edgecolor='white',
            linewidth=0.5
        )
        
        # Plot outliers with different colors by severity
        if 'severity' in plot_gdf.columns:
            colors = {'extreme': '#d73027', 'moderate': '#fee08b', 'mild': '#1a9850'}
            for severity, color in colors.items():
                subset = plot_gdf[plot_gdf['severity'] == severity]
                if not subset.empty:
                    subset.plot(
                        ax=ax,
                        color=color,
                        edgecolor='black',
                        linewidth=1,
                        label=severity.capitalize()
                    )
        else:
            # Simple outlier plotting
            outliers = plot_gdf[plot_gdf['is_outlier'] == True]
            if not outliers.empty:
                outliers.plot(
                    ax=ax,
                    color='#d73027',
                    edgecolor='black',
                    linewidth=1,
                    label='Outlier'
                )
        
        # Add legend and title
        ax.legend(loc='lower left', frameon=True, fancybox=True)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add state boundaries for context
        try:
            states = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            states[states['name'] == 'United States of America'].boundary.plot(
                ax=ax, color='black', linewidth=2, alpha=0.5
            )
        except:
            pass
            
        plt.tight_layout()
        output_path = self.output_dir / "spatial_outliers_map.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved outlier map to {output_path}")
        
    def create_change_maps(
        self,
        scenario: str = "ssp245",
        variables: Optional[List[str]] = None
    ) -> None:
        """
        Show climate change projections (future vs historical).
        
        Args:
            scenario: Climate scenario to use
            variables: Variables to show (default: temperature and precipitation)
        """
        if variables is None:
            variables = ['mean_temp_c', 'annual_precip_cm']
            
        logger.info(f"Creating climate change maps for {scenario}")
        
        # Calculate historical baseline (1980-2014)
        historical = self.df[
            (self.df['scenario'] == 'historical') &
            (self.df['year'] >= 1980) & 
            (self.df['year'] <= 2014)
        ].groupby('fips')[variables].mean()
        
        # Calculate future projection (2050-2080)
        future = self.df[
            (self.df['scenario'] == scenario) &
            (self.df['year'] >= 2050) & 
            (self.df['year'] <= 2080)
        ].groupby('fips')[variables].mean()
        
        # Calculate changes
        changes = future - historical
        
        # Create maps
        fig, axes = plt.subplots(1, len(variables), figsize=(16, 8))
        if len(variables) == 1:
            axes = [axes]
            
        for idx, var in enumerate(variables):
            ax = axes[idx]
            
            # Merge with geometry
            plot_gdf = self.gdf.merge(
                changes[[var]], 
                left_on='GEOID', 
                right_index=True,
                how='left'
            )
            
            # Create choropleth
            plot_gdf.plot(
                column=var,
                ax=ax,
                cmap=self.change_colors,
                legend=True,
                legend_kwds={
                    'label': f"Change in {self._get_variable_label(var)}",
                    'orientation': 'horizontal',
                    'pad': 0.05
                },
                missing_kwds={'color': 'lightgray'}
            )
            
            # Add title
            ax.set_title(
                f"Projected Change in {self._get_variable_label(var)}\n"
                f"({scenario.upper()}: 2050-2080 vs 1980-2014)",
                fontsize=14,
                fontweight='bold'
            )
            ax.axis('off')
            
        plt.tight_layout()
        output_path = self.output_dir / f"climate_change_maps_{scenario}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved climate change maps to {output_path}")
        
    def create_coverage_map(self, metric: str = "completeness") -> None:
        """
        Display data quality and coverage by county.
        
        Args:
            metric: Quality metric to display
        """
        logger.info(f"Creating data coverage map for {metric}")
        
        # Calculate coverage metrics by county
        if metric == "completeness":
            # Calculate percentage of non-null values
            coverage = self.df.groupby('fips').apply(
                lambda x: x[['mean_temp_c', 'annual_precip_cm']].notna().mean().mean()
            ) * 100
        elif metric == "years":
            # Count unique years per county
            coverage = self.df.groupby('fips')['year'].nunique()
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        # Convert to DataFrame
        coverage_df = pd.DataFrame({'fips': coverage.index, metric: coverage.values})
        
        # Merge with geometry
        plot_gdf = self.gdf.merge(
            coverage_df,
            left_on='GEOID',
            right_on='fips',
            how='left'
        )
        
        # Create map
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color map based on metric
        if metric == "completeness":
            cmap = 'RdYlGn'
            label = "Data Completeness (%)"
        else:
            cmap = 'viridis'
            label = "Years of Data"
            
        plot_gdf.plot(
            column=metric,
            ax=ax,
            cmap=cmap,
            legend=True,
            legend_kwds={
                'label': label,
                'orientation': 'vertical'
            },
            missing_kwds={'color': 'lightgray', 'label': 'No Data'}
        )
        
        ax.set_title(
            f"Data Coverage by County: {label}",
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        ax.axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / f"data_coverage_map_{metric}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved coverage map to {output_path}")
        
    def _get_variable_label(self, var: str) -> str:
        """Get human-readable label for variable."""
        labels = {
            'mean_temp_c': 'Mean Temperature (°C)',
            'min_temp_c': 'Minimum Temperature (°C)',
            'max_temp_c': 'Maximum Temperature (°C)',
            'annual_precip_cm': 'Annual Precipitation (cm)',
            'high_precip_days': 'High Precipitation Days',
            'hot_days': 'Hot Days (>32°C)',
            'cold_days': 'Cold Days (<0°C)'
        }
        return labels.get(var, var)
        
    def create_all_maps(self) -> None:
        """Generate all available map types."""
        logger.info("Creating all geographic visualizations")
        
        # Current climate maps
        self.create_climate_maps(time_period="current")
        
        # Historical climate maps
        self.create_climate_maps(time_period="historical")
        
        # Climate change projections
        if 'ssp245' in self.df['scenario'].unique():
            self.create_change_maps(scenario="ssp245")
        if 'ssp585' in self.df['scenario'].unique():
            self.create_change_maps(scenario="ssp585")
            
        # Data coverage
        self.create_coverage_map(metric="completeness")
        self.create_coverage_map(metric="years")
        
        logger.info("Completed all geographic visualizations")