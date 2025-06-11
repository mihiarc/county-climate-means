#!/usr/bin/env python3
"""
Script to visualize minimum temperature data from NetCDF and county-level extreme cold results.

Creates multiple maps:
1. Input NetCDF minimum temperature data as a raster
2. County-level annual mean minimum temperature as a choropleth map
3. County-level very cold days (10th percentile) map
4. County-level extremely cold days (5th percentile) map
5. County-level ultra-extreme cold days (1st percentile) map
"""

import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
from metrics.utils.climate_crs_handler import load_and_prepare_netcdf
import warnings
warnings.filterwarnings('ignore')

def load_netcdf_data():
    """Load and process NetCDF minimum temperature data using the standardized CRS handler."""
    print("Loading NetCDF minimum temperature data...")
    
    # Use the standardized CRS handler
    ds = load_and_prepare_netcdf('tasmin_CONUS_historical_1980_30yr_normal.nc')
    
    # Convert from Kelvin to Celsius and calculate annual mean
    temp_min_daily_c = ds['__xarray_dataarray_variable__'] - 273.15
    annual_mean_min_temp = temp_min_daily_c.mean(dim='dayofyear')
    
    ds.close()
    return annual_mean_min_temp

def load_county_results():
    """Load county-level extreme cold temperature results."""
    print("Loading county extreme cold temperature results...")
    
    # Load shapefile and results
    counties = gpd.read_file('tl_2024_us_county/tl_2024_us_county.shp')
    results = pd.read_csv('output/test/county_extreme_cold_metrics.csv')
    
    # Ensure GEOID columns are properly formatted as 5-character strings with leading zeros
    counties['GEOID'] = counties['GEOID'].astype(str).str.zfill(5)
    results['GEOID'] = results['GEOID'].astype(str).str.zfill(5)
    
    print(f"County GEOID example: {counties['GEOID'].iloc[0]}")
    print(f"Results GEOID example: {results['GEOID'].iloc[0]}")
    
    # Merge county boundaries with temperature data
    counties_with_data = counties.merge(results, on='GEOID', how='left')
    
    # Convert to appropriate CRS for visualization
    counties_with_data = counties_with_data.to_crs('EPSG:4326')
    
    return counties_with_data

def create_cold_temperature_colormap():
    """Create a custom colormap for cold temperature data."""
    # Colors from warm (light) to very cold (dark blue/purple)
    colors = ['#FFEEEE', '#E6F3FF', '#CCE7FF', '#99D6FF', '#66C2FF', '#3399FF', '#0066CC', '#003D99', '#002966']
    return LinearSegmentedColormap.from_list('cold_temperature', colors, N=256)

def create_cold_days_colormap():
    """Create a custom colormap for cold days data."""
    # Colors from few days (light) to many days (intense cold colors)
    colors = ['#F0F8FF', '#E6F3FF', '#B3D9FF', '#80BFFF', '#4D9FFF', '#1A7FFF', '#0066CC', '#004499', '#002266']
    return LinearSegmentedColormap.from_list('cold_days', colors, N=256)

def plot_netcdf_min_temperature(annual_mean_min_temp):
    """Create map of input NetCDF minimum temperature data."""
    print("Creating NetCDF minimum temperature map...")
    
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map extent for CONUS
    ax.set_extent([-130, -60, 15, 55], ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.3)
    
    # Create custom colormap
    cmap = create_cold_temperature_colormap()
    
    # Plot temperature data
    im = ax.pcolormesh(
        annual_mean_min_temp.lon, 
        annual_mean_min_temp.lat, 
        annual_mean_min_temp.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=-15,
        vmax=25,
        shading='auto'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Annual Mean Minimum Temperature (°C)', fontsize=12)
    
    # Add title and labels
    ax.set_title('Annual Mean Minimum Temperature from NetCDF Climate Data\n(CONUS 1980 30-Year Normal)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    plt.tight_layout()
    plt.savefig('min_temperature_netcdf_input.png', dpi=300, bbox_inches='tight')
    print("NetCDF minimum temperature map saved as 'min_temperature_netcdf_input.png'")
    return fig

def plot_county_mean_min_temperature(counties_with_data):
    """Create choropleth map of county-level mean minimum temperature results."""
    print("Creating county-level mean minimum temperature map...")
    
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map extent for CONUS
    ax.set_extent([-130, -60, 15, 55], ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.8, alpha=0.7, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.3)
    
    # Filter to only CONUS (exclude Alaska, Hawaii, territories)
    conus_counties = counties_with_data[
        (counties_with_data.geometry.centroid.x > -130) & 
        (counties_with_data.geometry.centroid.x < -60) &
        (counties_with_data.geometry.centroid.y > 15) & 
        (counties_with_data.geometry.centroid.y < 55)
    ].copy()
    
    # Create custom colormap
    cmap = create_cold_temperature_colormap()
    
    # Plot counties with temperature data
    counties_with_temp = conus_counties.dropna(subset=['annual_mean_min_temp_c'])
    
    im = counties_with_temp.plot(
        column='annual_mean_min_temp_c',
        ax=ax,
        cmap=cmap,
        vmin=-15,
        vmax=25,
        edgecolor='white',
        linewidth=0.1,
        transform=ccrs.PlateCarree(),
        legend=False
    )
    
    # Plot counties without data in gray
    counties_no_data = conus_counties[conus_counties['annual_mean_min_temp_c'].isna()]
    if not counties_no_data.empty:
        counties_no_data.plot(
            ax=ax,
            color='lightgray',
            edgecolor='white',
            linewidth=0.1,
            transform=ccrs.PlateCarree(),
            alpha=0.7
        )
    
    # Add colorbar manually
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-15, vmax=25))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Annual Mean Minimum Temperature (°C)', fontsize=12)
    
    # Add title
    ax.set_title('County-Level Annual Mean Minimum Temperature\n(Calculated from NetCDF Climate Data)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add statistics text
    valid_counties = counties_with_temp['annual_mean_min_temp_c'].count()
    total_counties = len(conus_counties)
    mean_temp = counties_with_temp['annual_mean_min_temp_c'].mean()
    
    stats_text = f"Coverage: {valid_counties}/{total_counties} counties ({valid_counties/total_counties*100:.1f}%)\n"
    stats_text += f"Mean: {mean_temp:.1f} °C"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('min_temperature_county_results.png', dpi=300, bbox_inches='tight')
    print("County minimum temperature map saved as 'min_temperature_county_results.png'")
    return fig

def plot_very_cold_days_10th(counties_with_data):
    """Create choropleth map of very cold days (10th percentile)."""
    print("Creating very cold days (10th percentile) map...")
    
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map extent for CONUS
    ax.set_extent([-130, -60, 15, 55], ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.8, alpha=0.7, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.3)
    
    # Filter to only CONUS
    conus_counties = counties_with_data[
        (counties_with_data.geometry.centroid.x > -130) & 
        (counties_with_data.geometry.centroid.x < -60) &
        (counties_with_data.geometry.centroid.y > 15) & 
        (counties_with_data.geometry.centroid.y < 55)
    ].copy()
    
    # Create custom colormap for cold days
    cmap_days = create_cold_days_colormap()
    
    # Plot counties with very cold days data
    counties_with_cold_days = conus_counties.dropna(subset=['very_cold_days_10th'])
    
    if not counties_with_cold_days.empty:
        # Determine appropriate value range
        max_days = counties_with_cold_days['very_cold_days_10th'].max()
        vmax = min(max_days, 200)  # Cap at 200 days for better color resolution
        
        im = counties_with_cold_days.plot(
            column='very_cold_days_10th',
            ax=ax,
            cmap=cmap_days,
            vmin=0,
            vmax=vmax,
            edgecolor='white',
            linewidth=0.1,
            transform=ccrs.PlateCarree(),
            legend=False
        )
    
    # Plot counties without data in gray
    counties_no_data = conus_counties[conus_counties['very_cold_days_10th'].isna()]
    if not counties_no_data.empty:
        counties_no_data.plot(
            ax=ax,
            color='lightgray',
            edgecolor='white',
            linewidth=0.1,
            transform=ccrs.PlateCarree(),
            alpha=0.7
        )
    
    # Add colorbar manually
    if not counties_with_cold_days.empty:
        sm = plt.cm.ScalarMappable(cmap=cmap_days, norm=plt.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Very Cold Days (<-9.9°C, 10th percentile)', fontsize=12)
    
    # Add title
    ax.set_title('County-Level Very Cold Days (10th Percentile)\n(Days per year below 10th percentile threshold)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add statistics text
    if not counties_with_cold_days.empty:
        valid_counties = counties_with_cold_days['very_cold_days_10th'].count()
        total_counties = len(conus_counties)
        mean_days = counties_with_cold_days['very_cold_days_10th'].mean()
        
        stats_text = f"Coverage: {valid_counties}/{total_counties} counties ({valid_counties/total_counties*100:.1f}%)\n"
        stats_text += f"Mean: {mean_days:.1f} days/year\n"
        stats_text += f"Note: Climate normals (30-year averages)"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('min_temperature_very_cold_days_10th.png', dpi=300, bbox_inches='tight')
    print("Very cold days (10th percentile) map saved as 'min_temperature_very_cold_days_10th.png'")
    return fig

def plot_extremely_cold_days_5th(counties_with_data):
    """Create choropleth map of extremely cold days (5th percentile)."""
    print("Creating extremely cold days (5th percentile) map...")
    
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map extent for CONUS
    ax.set_extent([-130, -60, 15, 55], ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.8, alpha=0.7, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.3)
    
    # Filter to only CONUS
    conus_counties = counties_with_data[
        (counties_with_data.geometry.centroid.x > -130) & 
        (counties_with_data.geometry.centroid.x < -60) &
        (counties_with_data.geometry.centroid.y > 15) & 
        (counties_with_data.geometry.centroid.y < 55)
    ].copy()
    
    # Create custom colormap for extremely cold days (more intense colors)
    colors = ['#F0F8FF', '#D6EFFF', '#ADC7FF', '#85A0FF', '#5C7AFF', '#3355FF', '#0A2FFF', '#0020CC', '#001299']
    cmap_days_5 = LinearSegmentedColormap.from_list('extremely_cold_days', colors, N=256)
    
    # Plot counties with extremely cold days data
    counties_with_cold_days = conus_counties.dropna(subset=['extremely_cold_days_5th'])
    
    if not counties_with_cold_days.empty:
        # Determine appropriate value range
        max_days = counties_with_cold_days['extremely_cold_days_5th'].max()
        vmax = min(max_days, 150)  # Cap at 150 days for better color resolution
        
        im = counties_with_cold_days.plot(
            column='extremely_cold_days_5th',
            ax=ax,
            cmap=cmap_days_5,
            vmin=0,
            vmax=vmax,
            edgecolor='white',
            linewidth=0.1,
            transform=ccrs.PlateCarree(),
            legend=False
        )
    
    # Plot counties without data in gray
    counties_no_data = conus_counties[conus_counties['extremely_cold_days_5th'].isna()]
    if not counties_no_data.empty:
        counties_no_data.plot(
            ax=ax,
            color='lightgray',
            edgecolor='white',
            linewidth=0.1,
            transform=ccrs.PlateCarree(),
            alpha=0.7
        )
    
    # Add colorbar manually
    if not counties_with_cold_days.empty:
        sm = plt.cm.ScalarMappable(cmap=cmap_days_5, norm=plt.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Extremely Cold Days (<-14.2°C, 5th percentile)', fontsize=12)
    
    # Add title
    ax.set_title('County-Level Extremely Cold Days (5th Percentile)\n(Days per year below 5th percentile threshold)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add statistics text
    if not counties_with_cold_days.empty:
        valid_counties = counties_with_cold_days['extremely_cold_days_5th'].count()
        total_counties = len(conus_counties)
        mean_days = counties_with_cold_days['extremely_cold_days_5th'].mean()
        
        stats_text = f"Coverage: {valid_counties}/{total_counties} counties ({valid_counties/total_counties*100:.1f}%)\n"
        stats_text += f"Mean: {mean_days:.1f} days/year\n"
        stats_text += f"Note: Climate normals (30-year averages)"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('min_temperature_extremely_cold_days_5th.png', dpi=300, bbox_inches='tight')
    print("Extremely cold days (5th percentile) map saved as 'min_temperature_extremely_cold_days_5th.png'")
    return fig

def plot_ultra_extreme_cold_days_1st(counties_with_data):
    """Create choropleth map of ultra-extreme cold days (1st percentile)."""
    print("Creating ultra-extreme cold days (1st percentile) map...")
    
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map extent for CONUS
    ax.set_extent([-130, -60, 15, 55], ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.8, alpha=0.7, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.3)
    
    # Filter to only CONUS
    conus_counties = counties_with_data[
        (counties_with_data.geometry.centroid.x > -130) & 
        (counties_with_data.geometry.centroid.x < -60) &
        (counties_with_data.geometry.centroid.y > 15) & 
        (counties_with_data.geometry.centroid.y < 55)
    ].copy()
    
    # Create custom colormap for ultra-extreme cold days (very intense colors)
    colors = ['#F5F8FF', '#E6F0FF', '#C7D9FF', '#A8C2FF', '#89ABFF', '#6A94FF', '#4B7DFF', '#2C66FF', '#0D4FFF', '#0033CC']
    cmap_days_1 = LinearSegmentedColormap.from_list('ultra_extreme_cold_days', colors, N=256)
    
    # Plot counties with ultra-extreme cold days data
    counties_with_cold_days = conus_counties.dropna(subset=['ultra_cold_days_1st'])
    
    if not counties_with_cold_days.empty:
        # Determine appropriate value range
        max_days = counties_with_cold_days['ultra_cold_days_1st'].max()
        vmax = max(max_days, 10)  # Ensure minimum range of 10 days for visibility
        
        im = counties_with_cold_days.plot(
            column='ultra_cold_days_1st',
            ax=ax,
            cmap=cmap_days_1,
            vmin=0,
            vmax=vmax,
            edgecolor='white',
            linewidth=0.1,
            transform=ccrs.PlateCarree(),
            legend=False
        )
    
    # Plot counties without data in gray
    counties_no_data = conus_counties[conus_counties['ultra_cold_days_1st'].isna()]
    if not counties_no_data.empty:
        counties_no_data.plot(
            ax=ax,
            color='lightgray',
            edgecolor='white',
            linewidth=0.1,
            transform=ccrs.PlateCarree(),
            alpha=0.7
        )
    
    # Add colorbar manually
    if not counties_with_cold_days.empty:
        sm = plt.cm.ScalarMappable(cmap=cmap_days_1, norm=plt.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Ultra-Extreme Cold Days (<-21.1°C, 1st percentile)', fontsize=12)
    
    # Add title
    ax.set_title('County-Level Ultra-Extreme Cold Days (1st Percentile)\n(Days per year below 1st percentile threshold)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add statistics text
    if not counties_with_cold_days.empty:
        valid_counties = counties_with_cold_days['ultra_cold_days_1st'].count()
        total_counties = len(conus_counties)
        mean_days = counties_with_cold_days['ultra_cold_days_1st'].mean()
        
        stats_text = f"Coverage: {valid_counties}/{total_counties} counties ({valid_counties/total_counties*100:.1f}%)\n"
        stats_text += f"Mean: {mean_days:.1f} days/year\n"
        stats_text += f"Note: Climate normals (30-year averages)"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('min_temperature_ultra_extreme_cold_days_1st.png', dpi=300, bbox_inches='tight')
    print("Ultra-extreme cold days (1st percentile) map saved as 'min_temperature_ultra_extreme_cold_days_1st.png'")
    return fig

def create_comparison_stats(annual_mean_min_temp, counties_with_data):
    """Create a comparison plot showing statistics."""
    print("Creating comparison statistics...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # NetCDF data histogram
    netcdf_data = annual_mean_min_temp.values.flatten()
    netcdf_data = netcdf_data[~np.isnan(netcdf_data)]
    
    axes[0,0].hist(netcdf_data, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0,0].set_xlabel('Annual Mean Minimum Temperature (°C)')
    axes[0,0].set_ylabel('Frequency (Grid Cells)')
    axes[0,0].set_title('Distribution of NetCDF Grid Cell Values')
    axes[0,0].axvline(np.mean(netcdf_data), color='blue', linestyle='--', 
                label=f'Mean: {np.mean(netcdf_data):.1f} °C')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # County mean minimum temperature histogram
    county_data = counties_with_data['annual_mean_min_temp_c'].dropna()
    
    axes[0,1].hist(county_data, bins=50, alpha=0.7, color='lightcyan', edgecolor='black')
    axes[0,1].set_xlabel('Annual Mean Minimum Temperature (°C)')
    axes[0,1].set_ylabel('Frequency (Counties)')
    axes[0,1].set_title('Distribution of County-Level Mean Minimum Temperature')
    axes[0,1].axvline(county_data.mean(), color='blue', linestyle='--', 
                label=f'Mean: {county_data.mean():.1f} °C')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Very cold days (10th percentile) histogram
    very_cold_data = counties_with_data['very_cold_days_10th'].dropna()
    
    axes[1,0].hist(very_cold_data, bins=30, alpha=0.7, color='lightsteelblue', edgecolor='black')
    axes[1,0].set_xlabel('Very Cold Days (days/year)')
    axes[1,0].set_ylabel('Frequency (Counties)')
    axes[1,0].set_title('Distribution of Very Cold Days (10th Percentile)')
    axes[1,0].axvline(very_cold_data.mean(), color='blue', linestyle='--', 
                label=f'Mean: {very_cold_data.mean():.1f} days')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Extremely cold days (5th percentile) histogram
    extremely_cold_data = counties_with_data['extremely_cold_days_5th'].dropna()
    
    axes[1,1].hist(extremely_cold_data, bins=30, alpha=0.7, color='royalblue', edgecolor='black')
    axes[1,1].set_xlabel('Extremely Cold Days (days/year)')
    axes[1,1].set_ylabel('Frequency (Counties)')
    axes[1,1].set_title('Distribution of Extremely Cold Days (5th Percentile)')
    axes[1,1].axvline(extremely_cold_data.mean(), color='darkblue', linestyle='--', 
                label=f'Mean: {extremely_cold_data.mean():.1f} days')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('min_temperature_comparison_stats.png', dpi=300, bbox_inches='tight')
    print("Comparison stats saved as 'min_temperature_comparison_stats.png'")
    return fig

def main():
    """Main function to create all visualizations."""
    print("=== MINIMUM TEMPERATURE DATA VISUALIZATION ===")
    print()
    
    # Load data
    annual_mean_min_temp = load_netcdf_data()
    counties_with_data = load_county_results()
    
    # Create maps
    print("\nCreating visualizations...")
    fig1 = plot_netcdf_min_temperature(annual_mean_min_temp)
    fig2 = plot_county_mean_min_temperature(counties_with_data)
    fig3 = plot_very_cold_days_10th(counties_with_data)
    fig4 = plot_extremely_cold_days_5th(counties_with_data)
    fig5 = plot_ultra_extreme_cold_days_1st(counties_with_data)
    fig6 = create_comparison_stats(annual_mean_min_temp, counties_with_data)
    
    print("\n=== VISUALIZATION COMPLETE ===")
    print("Created files:")
    print("  1. min_temperature_netcdf_input.png - Input NetCDF minimum temperature raster data")
    print("  2. min_temperature_county_results.png - County-level mean minimum temperature choropleth")
    print("  3. min_temperature_very_cold_days_10th.png - Very cold days (10th percentile)")
    print("  4. min_temperature_extremely_cold_days_5th.png - Extremely cold days (5th percentile)")
    print("  5. min_temperature_ultra_extreme_cold_days_1st.png - Ultra-extreme cold days (1st percentile)")
    print("  6. min_temperature_comparison_stats.png - Statistical comparison")
    
    # Show basic statistics
    valid_temp_data = counties_with_data['annual_mean_min_temp_c'].dropna()
    valid_10th_data = counties_with_data['very_cold_days_10th'].dropna()
    valid_5th_data = counties_with_data['extremely_cold_days_5th'].dropna()
    valid_1st_data = counties_with_data['ultra_cold_days_1st'].dropna()
    print(f"\nData Summary:")
    print(f"  NetCDF grid cells: {annual_mean_min_temp.size:,}")
    print(f"  Total counties: {len(counties_with_data):,}")
    print(f"  Counties with temperature data: {len(valid_temp_data):,}")
    print(f"  Counties with 10th percentile data: {len(valid_10th_data):,}")
    print(f"  Counties with 5th percentile data: {len(valid_5th_data):,}")
    print(f"  Counties with 1st percentile data: {len(valid_1st_data):,}")
    print(f"  Coverage: {len(valid_temp_data)/len(counties_with_data)*100:.1f}%")
    print(f"  Mean annual minimum temperature: {valid_temp_data.mean():.1f} °C")
    print(f"  Mean very cold days (10th): {valid_10th_data.mean():.1f} days/year")
    print(f"  Mean extremely cold days (5th): {valid_5th_data.mean():.1f} days/year")
    print(f"  Mean ultra-extreme cold days (1st): {valid_1st_data.mean():.1f} days/year")

if __name__ == "__main__":
    main() 