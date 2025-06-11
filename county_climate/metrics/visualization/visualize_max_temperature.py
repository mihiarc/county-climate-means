#!/usr/bin/env python3
"""
Script to visualize maximum temperature data from NetCDF and county-level results.

Creates multiple maps:
1. Input NetCDF maximum temperature data as a raster
2. County-level annual mean temperature as a choropleth map
3. County-level high temperature days (90th percentile) map
4. County-level very high temperature days (95th percentile) map
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
    """Load and process NetCDF maximum temperature data using the standardized CRS handler."""
    print("Loading NetCDF maximum temperature data...")
    
    # Use the standardized CRS handler
    ds = load_and_prepare_netcdf('tas_CONUS_historical_1980_30yr_normal.nc')
    
    # Convert from Kelvin to Celsius and calculate annual mean
    temp_daily_c = ds['__xarray_dataarray_variable__'] - 273.15
    annual_mean_temp = temp_daily_c.mean(dim='dayofyear')
    
    ds.close()
    return annual_mean_temp

def load_county_results():
    """Load county-level temperature results."""
    print("Loading county temperature results...")
    
    # Load shapefile and results
    counties = gpd.read_file('tl_2024_us_county/tl_2024_us_county.shp')
    results = pd.read_csv('output/test/county_temperature_percentile_metrics.csv')
    
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

def create_temperature_colormap():
    """Create a custom colormap for temperature data."""
    # Colors from cold (blue/purple) to hot (red/orange)
    colors = ['#000080', '#0000FF', '#4169E1', '#87CEEB', '#90EE90', '#FFFF00', '#FFA500', '#FF4500', '#DC143C']
    return LinearSegmentedColormap.from_list('temperature', colors, N=256)

def create_temp_days_colormap():
    """Create a custom colormap for temperature days data."""
    # Colors from few days (light) to many days (intense)
    colors = ['#FFFACD', '#FFE4B5', '#FFA500', '#FF6347', '#DC143C', '#B22222', '#8B0000']
    return LinearSegmentedColormap.from_list('temp_days', colors, N=256)

def plot_netcdf_temperature(annual_mean_temp):
    """Create map of input NetCDF temperature data."""
    print("Creating NetCDF temperature map...")
    
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
    cmap = create_temperature_colormap()
    
    # Plot temperature data
    im = ax.pcolormesh(
        annual_mean_temp.lon, 
        annual_mean_temp.lat, 
        annual_mean_temp.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=-5,
        vmax=30,
        shading='auto'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Annual Mean Temperature (°C)', fontsize=12)
    
    # Add title and labels
    ax.set_title('Annual Mean Temperature from NetCDF Climate Data\n(CONUS 1980 30-Year Normal)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    plt.tight_layout()
    plt.savefig('max_temperature_netcdf_input.png', dpi=300, bbox_inches='tight')
    print("NetCDF map saved as 'max_temperature_netcdf_input.png'")
    return fig

def plot_county_mean_temperature(counties_with_data):
    """Create choropleth map of county-level mean temperature results."""
    print("Creating county-level mean temperature map...")
    
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
    cmap = create_temperature_colormap()
    
    # Plot counties with temperature data
    counties_with_temp = conus_counties.dropna(subset=['annual_mean_temp_c'])
    
    im = counties_with_temp.plot(
        column='annual_mean_temp_c',
        ax=ax,
        cmap=cmap,
        vmin=-5,
        vmax=30,
        edgecolor='white',
        linewidth=0.1,
        transform=ccrs.PlateCarree(),
        legend=False
    )
    
    # Plot counties without data in gray
    counties_no_data = conus_counties[conus_counties['annual_mean_temp_c'].isna()]
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
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-5, vmax=30))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Annual Mean Temperature (°C)', fontsize=12)
    
    # Add title
    ax.set_title('County-Level Annual Mean Temperature\n(Calculated from NetCDF Climate Data)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add statistics text
    valid_counties = counties_with_temp['annual_mean_temp_c'].count()
    total_counties = len(conus_counties)
    mean_temp = counties_with_temp['annual_mean_temp_c'].mean()
    
    stats_text = f"Coverage: {valid_counties}/{total_counties} counties ({valid_counties/total_counties*100:.1f}%)\n"
    stats_text += f"Mean: {mean_temp:.1f} °C"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('max_temperature_county_results.png', dpi=300, bbox_inches='tight')
    print("County map saved as 'max_temperature_county_results.png'")
    return fig

def plot_high_temperature_days_90th(counties_with_data):
    """Create choropleth map of high temperature days (90th percentile)."""
    print("Creating high temperature days (90th percentile) map...")
    
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
    
    # Create custom colormap for temperature days
    cmap_days = create_temp_days_colormap()
    
    # Plot counties with high temperature days data
    counties_with_temp_days = conus_counties.dropna(subset=['high_temp_days_90th'])
    
    if not counties_with_temp_days.empty:
        # Determine appropriate value range
        max_days = counties_with_temp_days['high_temp_days_90th'].max()
        vmax = min(max_days, 200)  # Cap at 200 days for better color resolution
        
        im = counties_with_temp_days.plot(
            column='high_temp_days_90th',
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
    counties_no_data = conus_counties[conus_counties['high_temp_days_90th'].isna()]
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
    if not counties_with_temp_days.empty:
        sm = plt.cm.ScalarMappable(cmap=cmap_days, norm=plt.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('High Temperature Days (>24.4°C, 90th percentile)', fontsize=12)
    
    # Add title
    ax.set_title('County-Level High Temperature Days (90th Percentile)\n(Days per year above 90th percentile threshold)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add statistics text
    if not counties_with_temp_days.empty:
        valid_counties = counties_with_temp_days['high_temp_days_90th'].count()
        total_counties = len(conus_counties)
        mean_days = counties_with_temp_days['high_temp_days_90th'].mean()
        
        stats_text = f"Coverage: {valid_counties}/{total_counties} counties ({valid_counties/total_counties*100:.1f}%)\n"
        stats_text += f"Mean: {mean_days:.1f} days/year\n"
        stats_text += f"Note: Climate normals (30-year averages)"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('max_temperature_high_days_90th.png', dpi=300, bbox_inches='tight')
    print("High temperature days (90th percentile) map saved as 'max_temperature_high_days_90th.png'")
    return fig

def plot_very_high_temperature_days_95th(counties_with_data):
    """Create choropleth map of very high temperature days (95th percentile)."""
    print("Creating very high temperature days (95th percentile) map...")
    
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
    
    # Create custom colormap for temperature days (more intense colors for 95th percentile)
    colors = ['#FFF8DC', '#FFE4B5', '#FFA500', '#FF6347', '#DC143C', '#B22222', '#8B0000', '#800000']
    cmap_days_95 = LinearSegmentedColormap.from_list('temp_days_95', colors, N=256)
    
    # Plot counties with very high temperature days data
    counties_with_temp_days = conus_counties.dropna(subset=['high_temp_days_95th'])
    
    if not counties_with_temp_days.empty:
        # Determine appropriate value range
        max_days = counties_with_temp_days['high_temp_days_95th'].max()
        vmax = min(max_days, 150)  # Cap at 150 days for better color resolution
        
        im = counties_with_temp_days.plot(
            column='high_temp_days_95th',
            ax=ax,
            cmap=cmap_days_95,
            vmin=0,
            vmax=vmax,
            edgecolor='white',
            linewidth=0.1,
            transform=ccrs.PlateCarree(),
            legend=False
        )
    
    # Plot counties without data in gray
    counties_no_data = conus_counties[conus_counties['high_temp_days_95th'].isna()]
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
    if not counties_with_temp_days.empty:
        sm = plt.cm.ScalarMappable(cmap=cmap_days_95, norm=plt.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Very High Temperature Days (>26.7°C, 95th percentile)', fontsize=12)
    
    # Add title
    ax.set_title('County-Level Very High Temperature Days (95th Percentile)\n(Days per year above 95th percentile threshold)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add statistics text
    if not counties_with_temp_days.empty:
        valid_counties = counties_with_temp_days['high_temp_days_95th'].count()
        total_counties = len(conus_counties)
        mean_days = counties_with_temp_days['high_temp_days_95th'].mean()
        
        stats_text = f"Coverage: {valid_counties}/{total_counties} counties ({valid_counties/total_counties*100:.1f}%)\n"
        stats_text += f"Mean: {mean_days:.1f} days/year\n"
        stats_text += f"Note: Climate normals (30-year averages)"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('max_temperature_very_high_days_95th.png', dpi=300, bbox_inches='tight')
    print("Very high temperature days (95th percentile) map saved as 'max_temperature_very_high_days_95th.png'")
    return fig

def create_comparison_stats(annual_mean_temp, counties_with_data):
    """Create a comparison plot showing statistics."""
    print("Creating comparison statistics...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # NetCDF data histogram
    netcdf_data = annual_mean_temp.values.flatten()
    netcdf_data = netcdf_data[~np.isnan(netcdf_data)]
    
    axes[0,0].hist(netcdf_data, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,0].set_xlabel('Annual Mean Temperature (°C)')
    axes[0,0].set_ylabel('Frequency (Grid Cells)')
    axes[0,0].set_title('Distribution of NetCDF Grid Cell Values')
    axes[0,0].axvline(np.mean(netcdf_data), color='red', linestyle='--', 
                label=f'Mean: {np.mean(netcdf_data):.1f} °C')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # County mean temperature histogram
    county_data = counties_with_data['annual_mean_temp_c'].dropna()
    
    axes[0,1].hist(county_data, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,1].set_xlabel('Annual Mean Temperature (°C)')
    axes[0,1].set_ylabel('Frequency (Counties)')
    axes[0,1].set_title('Distribution of County-Level Mean Temperature')
    axes[0,1].axvline(county_data.mean(), color='red', linestyle='--', 
                label=f'Mean: {county_data.mean():.1f} °C')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # High temperature days (90th percentile) histogram
    high_days_data = counties_with_data['high_temp_days_90th'].dropna()
    
    axes[1,0].hist(high_days_data, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].set_xlabel('High Temperature Days (days/year)')
    axes[1,0].set_ylabel('Frequency (Counties)')
    axes[1,0].set_title('Distribution of High Temperature Days (90th Percentile)')
    axes[1,0].axvline(high_days_data.mean(), color='red', linestyle='--', 
                label=f'Mean: {high_days_data.mean():.1f} days')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Very high temperature days (95th percentile) histogram
    very_high_days_data = counties_with_data['high_temp_days_95th'].dropna()
    
    axes[1,1].hist(very_high_days_data, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[1,1].set_xlabel('Very High Temperature Days (days/year)')
    axes[1,1].set_ylabel('Frequency (Counties)')
    axes[1,1].set_title('Distribution of Very High Temperature Days (95th Percentile)')
    axes[1,1].axvline(very_high_days_data.mean(), color='darkred', linestyle='--', 
                label=f'Mean: {very_high_days_data.mean():.1f} days')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('max_temperature_comparison_stats.png', dpi=300, bbox_inches='tight')
    print("Comparison stats saved as 'max_temperature_comparison_stats.png'")
    return fig

def main():
    """Main function to create all visualizations."""
    print("=== MAXIMUM TEMPERATURE DATA VISUALIZATION ===")
    print()
    
    # Load data
    annual_mean_temp = load_netcdf_data()
    counties_with_data = load_county_results()
    
    # Create maps
    print("\nCreating visualizations...")
    fig1 = plot_netcdf_temperature(annual_mean_temp)
    fig2 = plot_county_mean_temperature(counties_with_data)
    fig3 = plot_high_temperature_days_90th(counties_with_data)
    fig4 = plot_very_high_temperature_days_95th(counties_with_data)
    fig5 = create_comparison_stats(annual_mean_temp, counties_with_data)
    
    print("\n=== VISUALIZATION COMPLETE ===")
    print("Created files:")
    print("  1. max_temperature_netcdf_input.png - Input NetCDF raster data")
    print("  2. max_temperature_county_results.png - County-level mean temperature choropleth")
    print("  3. max_temperature_high_days_90th.png - High temperature days (90th percentile)")
    print("  4. max_temperature_very_high_days_95th.png - Very high temperature days (95th percentile)")
    print("  5. max_temperature_comparison_stats.png - Statistical comparison")
    
    # Show basic statistics
    valid_temp_data = counties_with_data['annual_mean_temp_c'].dropna()
    valid_90th_data = counties_with_data['high_temp_days_90th'].dropna()
    valid_95th_data = counties_with_data['high_temp_days_95th'].dropna()
    print(f"\nData Summary:")
    print(f"  NetCDF grid cells: {annual_mean_temp.size:,}")
    print(f"  Total counties: {len(counties_with_data):,}")
    print(f"  Counties with temperature data: {len(valid_temp_data):,}")
    print(f"  Counties with 90th percentile data: {len(valid_90th_data):,}")
    print(f"  Counties with 95th percentile data: {len(valid_95th_data):,}")
    print(f"  Coverage: {len(valid_temp_data)/len(counties_with_data)*100:.1f}%")
    print(f"  Mean annual temperature: {valid_temp_data.mean():.1f} °C")
    print(f"  Mean high temperature days (90th): {valid_90th_data.mean():.1f} days/year")
    print(f"  Mean very high temperature days (95th): {valid_95th_data.mean():.1f} days/year")

if __name__ == "__main__":
    main() 