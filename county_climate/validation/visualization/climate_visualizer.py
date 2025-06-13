"""Climate data visualization for validation outputs."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import warnings

warnings.filterwarnings('ignore')


class ClimateVisualizer:
    """
    Creates comprehensive visualizations of climate data for validation.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize visualizer."""
        self.output_dir = output_dir or Path("validation_outputs/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_overview_dashboard(self, df: pd.DataFrame, title: str = "Climate Data Overview"):
        """Create overview dashboard with key statistics."""
        self.logger.info("Creating overview dashboard...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Data completeness heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        if len(missing_pct) > 0:
            sns.barplot(x=missing_pct.values, y=missing_pct.index, ax=ax1)
            ax1.set_xlabel('Missing Data (%)')
            ax1.set_title('Data Completeness by Variable')
        else:
            ax1.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=16)
            ax1.set_title('Data Completeness')
        
        # 2. Record count by scenario
        ax2 = fig.add_subplot(gs[0, 2])
        scenario_counts = df['scenario'].value_counts()
        scenario_counts.plot(kind='bar', ax=ax2)
        ax2.set_title('Records by Scenario')
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('Record Count')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Temperature distribution
        ax3 = fig.add_subplot(gs[1, 0])
        if 'mean_temp_c' in df.columns:
            df['mean_temp_c'].hist(bins=50, ax=ax3, alpha=0.7)
            ax3.set_xlabel('Mean Temperature (°C)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Temperature Distribution')
            ax3.axvline(df['mean_temp_c'].mean(), color='red', linestyle='--', 
                       label=f"Mean: {df['mean_temp_c'].mean():.1f}°C")
            ax3.legend()
        
        # 4. Precipitation distribution
        ax4 = fig.add_subplot(gs[1, 1])
        if 'annual_precip_mm' in df.columns:
            df['annual_precip_mm'].hist(bins=50, ax=ax4, alpha=0.7)
            ax4.set_xlabel('Annual Precipitation (mm)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Precipitation Distribution')
            ax4.axvline(df['annual_precip_mm'].mean(), color='blue', linestyle='--',
                       label=f"Mean: {df['annual_precip_mm'].mean():.0f}mm")
            ax4.legend()
        
        # 5. Temporal coverage
        ax5 = fig.add_subplot(gs[1, 2])
        yearly_counts = df.groupby('year').size()
        yearly_counts.plot(ax=ax5)
        ax5.set_xlabel('Year')
        ax5.set_ylabel('Record Count')
        ax5.set_title('Temporal Coverage')
        ax5.grid(True, alpha=0.3)
        
        # 6. County coverage
        ax6 = fig.add_subplot(gs[2, :])
        county_counts = df.groupby('GEOID')['year'].count()
        ax6.hist(county_counts.values, bins=30, alpha=0.7)
        ax6.set_xlabel('Number of Year-Scenario Records per County')
        ax6.set_ylabel('Number of Counties')
        ax6.set_title('County Data Completeness Distribution')
        ax6.axvline(county_counts.mean(), color='green', linestyle='--',
                   label=f"Mean: {county_counts.mean():.0f} records/county")
        ax6.legend()
        
        # Add main title
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # Save
        output_path = self.output_dir / "overview_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved overview dashboard to {output_path}")
        return output_path
    
    def create_temperature_analysis(self, df: pd.DataFrame):
        """Create temperature-specific analysis plots."""
        self.logger.info("Creating temperature analysis plots...")
        
        temp_cols = ['mean_temp_c', 'min_temp_c', 'max_temp_c']
        available_cols = [col for col in temp_cols if col in df.columns]
        
        if not available_cols:
            self.logger.warning("No temperature columns found")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 1. Temperature correlations
        if len(available_cols) > 1:
            corr = df[available_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[0], square=True)
            axes[0].set_title('Temperature Metric Correlations')
        
        # 2. Temperature ranges by scenario
        if 'scenario' in df.columns and 'mean_temp_c' in df.columns:
            df.boxplot(column='mean_temp_c', by='scenario', ax=axes[1])
            axes[1].set_title('Mean Temperature by Scenario')
            axes[1].set_xlabel('Scenario')
            axes[1].set_ylabel('Temperature (°C)')
        
        # 3. Temperature extremes
        if 'hot_days_gt_35c' in df.columns and 'cold_days_lt_0c' in df.columns:
            axes[2].scatter(df['cold_days_lt_0c'], df['hot_days_gt_35c'], 
                          alpha=0.5, s=1)
            axes[2].set_xlabel('Cold Days (<0°C)')
            axes[2].set_ylabel('Hot Days (>35°C)')
            axes[2].set_title('Temperature Extremes Relationship')
        
        # 4. Temperature trend over time
        if 'mean_temp_c' in df.columns:
            yearly_temp = df.groupby(['year', 'scenario'])['mean_temp_c'].mean().unstack()
            yearly_temp.plot(ax=axes[3], marker='o', markersize=2)
            axes[3].set_xlabel('Year')
            axes[3].set_ylabel('Mean Temperature (°C)')
            axes[3].set_title('Temperature Trends by Scenario')
            axes[3].legend(title='Scenario')
            axes[3].grid(True, alpha=0.3)
        
        plt.suptitle('Temperature Analysis', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / "temperature_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved temperature analysis to {output_path}")
        return output_path
    
    def create_precipitation_analysis(self, df: pd.DataFrame):
        """Create precipitation-specific analysis plots."""
        self.logger.info("Creating precipitation analysis plots...")
        
        precip_cols = ['annual_precip_mm', 'precip_gt_50mm', 'precip_gt_100mm']
        available_cols = [col for col in precip_cols if col in df.columns]
        
        if not available_cols:
            self.logger.warning("No precipitation columns found")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 1. Precipitation distribution by scenario
        if 'annual_precip_mm' in df.columns and 'scenario' in df.columns:
            for scenario in df['scenario'].unique():
                scenario_data = df[df['scenario'] == scenario]
                axes[0].hist(scenario_data['annual_precip_mm'], bins=30, 
                           alpha=0.5, label=scenario)
            axes[0].set_xlabel('Annual Precipitation (mm)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Precipitation Distribution by Scenario')
            axes[0].legend()
        
        # 2. Precipitation vs high precipitation days
        if 'annual_precip_mm' in df.columns and 'precip_gt_50mm' in df.columns:
            axes[1].scatter(df['precip_gt_50mm'], df['annual_precip_mm'], 
                          alpha=0.3, s=1)
            axes[1].set_xlabel('Days with >50mm Precipitation')
            axes[1].set_ylabel('Annual Precipitation (mm)')
            axes[1].set_title('Annual Precipitation vs High Precipitation Days')
            
            # Add trend line
            z = np.polyfit(df['precip_gt_50mm'].dropna(), 
                          df['annual_precip_mm'].dropna(), 1)
            p = np.poly1d(z)
            axes[1].plot(df['precip_gt_50mm'].sort_values(), 
                        p(df['precip_gt_50mm'].sort_values()),
                        "r--", alpha=0.8, label=f'Trend: y={z[0]:.1f}x+{z[1]:.0f}')
            axes[1].legend()
        
        # 3. Extreme precipitation patterns
        if 'precip_gt_100mm' in df.columns:
            yearly_extremes = df.groupby('year')['precip_gt_100mm'].mean()
            yearly_extremes.plot(ax=axes[2], marker='o', markersize=3)
            axes[2].set_xlabel('Year')
            axes[2].set_ylabel('Average Days >100mm')
            axes[2].set_title('Extreme Precipitation Trend')
            axes[2].grid(True, alpha=0.3)
        
        # 4. Dry days analysis
        if 'dry_days' in df.columns:
            axes[3].hist(df['dry_days'], bins=50, alpha=0.7)
            axes[3].set_xlabel('Dry Days per Year')
            axes[3].set_ylabel('Frequency')
            axes[3].set_title('Distribution of Dry Days')
            axes[3].axvline(df['dry_days'].mean(), color='red', linestyle='--',
                          label=f"Mean: {df['dry_days'].mean():.0f} days")
            axes[3].legend()
        
        plt.suptitle('Precipitation Analysis', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / "precipitation_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved precipitation analysis to {output_path}")
        return output_path
    
    def create_validation_summary_plots(self, validation_results: Dict):
        """Create summary plots from validation results."""
        self.logger.info("Creating validation summary plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. Issues by severity
        if 'issue_counts' in validation_results:
            issue_counts = validation_results['issue_counts']
            colors = {'critical': 'red', 'warning': 'orange', 'info': 'blue'}
            
            severities = list(issue_counts.keys())
            counts = list(issue_counts.values())
            colors_list = [colors.get(s, 'gray') for s in severities]
            
            axes[0].bar(severities, counts, color=colors_list)
            axes[0].set_title('Validation Issues by Severity')
            axes[0].set_ylabel('Count')
            
            # Add value labels on bars
            for i, (severity, count) in enumerate(zip(severities, counts)):
                axes[0].text(i, count + 0.5, str(count), ha='center')
        
        # 2. Issues by category
        if 'issues_by_category' in validation_results:
            category_counts = validation_results['issues_by_category']
            categories = list(category_counts.keys())
            counts = list(category_counts.values())
            
            axes[1].barh(categories, counts)
            axes[1].set_title('Issues by Category')
            axes[1].set_xlabel('Count')
        
        # 3. Data quality metrics
        if 'quality_metrics' in validation_results:
            metrics = validation_results['quality_metrics']
            
            # Create a simple quality score visualization
            quality_score = metrics.get('overall_score', 0)
            axes[2].pie([quality_score, 100-quality_score], 
                       labels=['Pass', 'Fail'],
                       colors=['green', 'lightgray'],
                       startangle=90,
                       counterclock=False)
            axes[2].set_title(f'Overall Quality Score: {quality_score:.1f}%')
        
        # 4. Completeness summary
        if 'completeness_summary' in validation_results:
            completeness = validation_results['completeness_summary']
            
            labels = list(completeness.keys())
            values = list(completeness.values())
            
            axes[3].bar(labels, values)
            axes[3].set_title('Data Completeness Summary')
            axes[3].set_ylabel('Percentage Complete')
            axes[3].set_ylim(0, 105)
            
            # Add value labels
            for i, (label, value) in enumerate(zip(labels, values)):
                axes[3].text(i, value + 1, f'{value:.1f}%', ha='center')
        
        plt.suptitle('Validation Summary', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / "validation_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved validation summary to {output_path}")
        return output_path