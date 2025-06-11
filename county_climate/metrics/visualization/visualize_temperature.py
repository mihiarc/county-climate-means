#!/usr/bin/env python3
"""
Script to visualize temperature data from NetCDF and county-level results.

Creates multiple maps:
1. Input NetCDF temperature data as a raster
2. County-level temperature results as choropleth maps
3. Statistical comparison plots
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
    """Load and process NetCDF temperature data using the standardized CRS handler."""
    print("Loading NetCDF temperature data...")
    
    # Use the standardized CRS handler
    ds = load_and_prepare_netcdf('tas_CONUS_historical_1980_30yr_normal.nc')
    
    # Get temperature data and convert from Kelvin to Celsius
    temp_data = ds['__xarray_dataarray_variable__'] - 273.15
    
    # Calculate annual mean temperature
    temp_annual = temp_data.mean(dim='dayofyear')
    
    print(f"NetCDF data summary:")
    print(f"  Geographic extent: {ds.lat.min().values:.1f}°N to {ds.lat.max().values:.1f}°N, {ds.lon.min().values:.1f}°W to {ds.lon.max().values:.1f}°W")
    print(f"  Grid size: {len(ds.lat)} x {len(ds.lon)} pixels")
    print(f"  Temperature range: {temp_annual.values.min():.1f}°C to {temp_annual.values.max():.1f}°C")
    print(f"  Mean temperature: {np.nanmean(temp_annual.values):.1f}°C")
    
    ds.close()
    return temp_annual

def load_county_data():
    """Load county-level temperature results."""
    print("Loading county temperature results...")
    
    try:
        # Try to load the county results
        county_df = pd.read_csv('county_temperature_sample.csv')
        print(f"Loaded {len(county_df)} county records")
        
        # Load county shapefile
        counties_gdf = gpd.read_file('tl_2024_us_county/tl_2024_us_county.shp')
        
        # Ensure GEOID columns are both strings with proper formatting
        counties_gdf['GEOID'] = counties_gdf['GEOID'].astype(str).str.zfill(5)
        county_df['GEOID'] = county_df['GEOID'].astype(str).str.zfill(5)
        
        print(f"Sample GEOIDs - Shapefile: {counties_gdf['GEOID'].iloc[0]}, CSV: {county_df['GEOID'].iloc[0]}")
        
        # Merge with temperature data
        counties_merged = counties_gdf.merge(county_df, on='GEOID', how='left')
        
        print(f"County data summary:")
        print(f"  Counties with data: {counties_merged['annual_mean_temp_c'].notna().sum()}")
        print(f"  Temperature range: {counties_merged['annual_mean_temp_c'].min():.1f}°C to {counties_merged['annual_mean_temp_c'].max():.1f}°C")
        print(f"  Mean temperature: {counties_merged['annual_mean_temp_c'].mean():.1f}°C")
        
        return counties_merged
        
    except FileNotFoundError:
        print("County temperature results not found. Will only create NetCDF visualization.")
        return None

def create_temperature_colormap():
    """Create a temperature-appropriate colormap."""
    # Create a blue-to-red colormap suitable for temperature
    colors = ['#08519c', '#3182bd', '#6baed6', '#bdd7e7', '#eff3ff', 
              '#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
    return LinearSegmentedColormap.from_list('temperature', colors, N=256)

def plot_netcdf_temperature(temp_data):
    """Create raster map of NetCDF temperature data."""
    print("Creating NetCDF temperature map...")
    
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent to CONUS
    ax.set_extent([-125, -65, 20, 46], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    
    # Create temperature colormap
    temp_cmap = create_temperature_colormap()
    
    # Plot temperature data
    temp_plot = ax.pcolormesh(
        temp_data.lon, temp_data.lat, temp_data.values,
        transform=ccrs.PlateCarree(),
        cmap=temp_cmap,
        shading='auto'
    )
    
    # Add colorbar
    cbar = plt.colorbar(temp_plot, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8, aspect=40)
    cbar.set_label('Annual Mean Temperature (°C)', fontsize=12)
    
    # Add title and labels
    plt.title('NetCDF Input: Annual Mean Temperature (1980 30-Year Normal)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    plt.tight_layout()
    plt.savefig('temperature_netcdf_input.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: temperature_netcdf_input.png")

def plot_county_temperature(counties_gdf):
    """Create choropleth map of county-level temperature results."""
    if counties_gdf is None:
        print("No county data available for mapping.")
        return
        
    print("Creating county temperature map...")
    
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent to CONUS
    ax.set_extent([-125, -65, 20, 50], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.7)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    
    # Create temperature colormap
    temp_cmap = create_temperature_colormap()
    
    # Plot counties with temperature data
    counties_with_data = counties_gdf[counties_gdf['annual_mean_temp_c'].notna()]
    
    if len(counties_with_data) > 0:
        counties_with_data.plot(
            column='annual_mean_temp_c',
            ax=ax,
            cmap=temp_cmap,
            linewidth=0.1,
            edgecolor='white',
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            legend=True,
            legend_kwds={
                'label': 'Annual Mean Temperature (°C)',
                'orientation': 'horizontal',
                'pad': 0.05,
                'shrink': 0.8,
                'aspect': 40
            }
        )
    
    # Plot counties without data in gray
    counties_no_data = counties_gdf[counties_gdf['annual_mean_temp_c'].isna()]
    if len(counties_no_data) > 0:
        counties_no_data.plot(
            ax=ax,
            color='lightgray',
            linewidth=0.1,
            edgecolor='white',
            alpha=0.5,
            transform=ccrs.PlateCarree()
        )
    
    plt.title('County-Level Annual Mean Temperature Results', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    plt.tight_layout()
    plt.savefig('temperature_county_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: temperature_county_results.png")

def plot_hot_cold_days(counties_gdf):
    """Create maps showing hot days (>30°C) and cold days (<0°C)."""
    if counties_gdf is None or 'hot_days_30c' not in counties_gdf.columns:
        print("No hot/cold days data available for mapping.")
        return
        
    print("Creating hot and cold days maps...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                   subplot_kw={'projection': ccrs.PlateCarree()})
    
    for ax in [ax1, ax2]:
        ax.set_extent([-125, -65, 20, 50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.7)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    
    # Hot days map
    counties_with_hot = counties_gdf[counties_gdf['hot_days_30c'].notna()]
    if len(counties_with_hot) > 0:
        counties_with_hot.plot(
            column='hot_days_30c',
            ax=ax1,
            cmap='Reds',
            linewidth=0.1,
            edgecolor='white',
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            legend=True,
            legend_kwds={
                'label': 'Hot Days > 30°C',
                'orientation': 'horizontal',
                'pad': 0.05,
                'shrink': 0.8
            }
        )
    
    ax1.set_title('Annual Hot Days (> 30°C)', fontsize=14, fontweight='bold')
    
    # Cold days map
    counties_with_cold = counties_gdf[counties_gdf['cold_days_0c'].notna()]
    if len(counties_with_cold) > 0:
        counties_with_cold.plot(
            column='cold_days_0c',
            ax=ax2,
            cmap='Blues',
            linewidth=0.1,
            edgecolor='white',
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            legend=True,
            legend_kwds={
                'label': 'Cold Days < 0°C',
                'orientation': 'horizontal',
                'pad': 0.05,
                'shrink': 0.8
            }
        )
    
    ax2.set_title('Annual Cold Days (< 0°C)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('temperature_hot_days.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: temperature_hot_days.png")

def create_comparison_stats(temp_netcdf, counties_gdf):
    """Create statistical comparison plots."""
    if counties_gdf is None:
        print("No county data available for comparison.")
        return
        
    print("Creating statistical comparison plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Temperature distribution histogram
    county_temps = counties_gdf['annual_mean_temp_c'].dropna()
    netcdf_temps = temp_netcdf.values.flatten()
    netcdf_temps = netcdf_temps[~np.isnan(netcdf_temps)]
    
    ax1.hist(county_temps, bins=30, alpha=0.7, label=f'County-level (n={len(county_temps)})', color='blue')
    ax1.hist(netcdf_temps, bins=30, alpha=0.7, label=f'NetCDF grid (n={len(netcdf_temps)})', color='red')
    ax1.set_xlabel('Annual Mean Temperature (°C)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Temperature Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Coverage statistics
    total_counties = len(counties_gdf)
    counties_with_data = counties_gdf['annual_mean_temp_c'].notna().sum()
    coverage_pct = (counties_with_data / total_counties) * 100
    
    ax2.bar(['Total Counties', 'With Data', 'Missing Data'], 
            [total_counties, counties_with_data, total_counties - counties_with_data],
            color=['gray', 'green', 'red'], alpha=0.7)
    ax2.set_ylabel('Number of Counties')
    ax2.set_title(f'Data Coverage: {coverage_pct:.1f}%')
    ax2.grid(True, alpha=0.3)
    
    # Add text annotation
    ax2.text(0.5, 0.95, f'{counties_with_data}/{total_counties} counties\n({coverage_pct:.1f}%)', 
             transform=ax2.transAxes, ha='center', va='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Temperature statistics by state (if state info available)
    if 'STATEFP' in counties_gdf.columns:
        state_stats = counties_gdf.groupby('STATEFP')['annual_mean_temp_c'].agg(['mean', 'count']).reset_index()
        state_stats = state_stats[state_stats['count'] >= 5]  # Only states with 5+ counties
        
        # Sample some states for display
        sample_states = state_stats.head(10)
        
        ax3.bar(range(len(sample_states)), sample_states['mean'], 
                color='orange', alpha=0.7)
        ax3.set_xlabel('State (FIPS Code)')
        ax3.set_ylabel('Mean Temperature (°C)')
        ax3.set_title('Average Temperature by State (Sample)')
        ax3.set_xticks(range(len(sample_states)))
        ax3.set_xticklabels(sample_states['STATEFP'], rotation=45)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'State information\nnot available', 
                transform=ax3.transAxes, ha='center', va='center')
        ax3.set_title('State Analysis')
    
    # 4. Summary statistics table
    ax4.axis('off')
    
    if len(county_temps) > 0:
        stats_data = [
            ['Metric', 'County-Level', 'NetCDF Grid'],
            ['Count', f'{len(county_temps):,}', f'{len(netcdf_temps):,}'],
            ['Mean (°C)', f'{county_temps.mean():.1f}', f'{netcdf_temps.mean():.1f}'],
            ['Std Dev (°C)', f'{county_temps.std():.1f}', f'{netcdf_temps.std():.1f}'],
            ['Min (°C)', f'{county_temps.min():.1f}', f'{netcdf_temps.min():.1f}'],
            ['Max (°C)', f'{county_temps.max():.1f}', f'{netcdf_temps.max():.1f}'],
            ['Median (°C)', f'{county_temps.median():.1f}', f'{np.median(netcdf_temps):.1f}']
        ]
        
        table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the header
        for i in range(3):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('temperature_comparison_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: temperature_comparison_stats.png")

def main():
    """Main function to generate all temperature visualizations."""
    print("=== TEMPERATURE VISUALIZATION SUITE ===")
    
    # Load data
    temp_netcdf = load_netcdf_data()
    counties_gdf = load_county_data()
    
    # Create visualizations
    plot_netcdf_temperature(temp_netcdf)
    plot_county_temperature(counties_gdf)
    plot_hot_cold_days(counties_gdf)
    create_comparison_stats(temp_netcdf, counties_gdf)
    
    print("\n=== VISUALIZATION COMPLETE ===")
    print("Generated files:")
    print("  - temperature_netcdf_input.png")
    print("  - temperature_county_results.png")
    print("  - temperature_hot_days.png")
    print("  - temperature_comparison_stats.png")

if __name__ == "__main__":
    main() 