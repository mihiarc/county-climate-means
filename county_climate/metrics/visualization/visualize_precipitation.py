#!/usr/bin/env python3
"""
Script to visualize precipitation data from NetCDF and county-level results.

Creates two maps:
1. Input NetCDF precipitation data as a raster
2. County-level precipitation results as a choropleth map
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
    """Load and process NetCDF precipitation data using the standardized CRS handler."""
    print("Loading NetCDF precipitation data...")
    
    # Use the standardized CRS handler
    ds = load_and_prepare_netcdf('pr_CONUS_historical_1980_30yr_normal.nc')
    
    # Calculate annual precipitation
    pr_daily_mm = ds['__xarray_dataarray_variable__'] * 86400  # Convert to mm/day
    annual_precip = pr_daily_mm.sum(dim='dayofyear')  # Sum to get annual total
    
    ds.close()
    return annual_precip

def load_county_results():
    """Load county-level precipitation results."""
    print("Loading county precipitation results...")
    
    # Load shapefile and results
    counties = gpd.read_file('tl_2024_us_county/tl_2024_us_county.shp')
    results = pd.read_csv('county_precipitation_with_heavy_rain.csv')  # Updated to use enhanced file
    
    # Ensure GEOID columns are properly formatted as 5-character strings with leading zeros
    counties['GEOID'] = counties['GEOID'].astype(str).str.zfill(5)
    results['GEOID'] = results['GEOID'].astype(str).str.zfill(5)
    
    print(f"County GEOID example: {counties['GEOID'].iloc[0]}")
    print(f"Results GEOID example: {results['GEOID'].iloc[0]}")
    
    # Merge county boundaries with precipitation data
    counties_with_data = counties.merge(results, on='GEOID', how='left')
    
    # Convert to appropriate CRS for visualization
    counties_with_data = counties_with_data.to_crs('EPSG:4326')
    
    return counties_with_data

def create_precipitation_colormap():
    """Create a custom colormap for precipitation data."""
    # Colors from dry (brown/tan) to wet (blue/green)
    colors = ['#8B4513', '#DEB887', '#F0E68C', '#90EE90', '#20B2AA', '#4169E1', '#0000CD']
    return LinearSegmentedColormap.from_list('precipitation', colors, N=256)

def plot_netcdf_precipitation(annual_precip):
    """Create map of input NetCDF precipitation data."""
    print("Creating NetCDF precipitation map...")
    
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
    cmap = create_precipitation_colormap()
    
    # Plot precipitation data
    im = ax.pcolormesh(
        annual_precip.lon, 
        annual_precip.lat, 
        annual_precip.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=0,
        vmax=2000,
        shading='auto'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Annual Precipitation (mm/year)', fontsize=12)
    
    # Add title and labels
    ax.set_title('Annual Precipitation from NetCDF Climate Data\n(CONUS 1980 30-Year Normal)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    plt.tight_layout()
    plt.savefig('precipitation_netcdf_input.png', dpi=300, bbox_inches='tight')
    print("NetCDF map saved as 'precipitation_netcdf_input.png'")
    return fig

def plot_county_precipitation(counties_with_data):
    """Create choropleth map of county-level precipitation results."""
    print("Creating county-level precipitation map...")
    
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
    cmap = create_precipitation_colormap()
    
    # Plot counties with precipitation data
    counties_with_precip = conus_counties.dropna(subset=['annual_precipitation_mm'])
    
    im = counties_with_precip.plot(
        column='annual_precipitation_mm',
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=2000,
        edgecolor='white',
        linewidth=0.1,
        transform=ccrs.PlateCarree(),
        legend=False
    )
    
    # Plot counties without data in gray
    counties_no_data = conus_counties[conus_counties['annual_precipitation_mm'].isna()]
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
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=2000))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Annual Precipitation (mm/year)', fontsize=12)
    
    # Add title
    ax.set_title('County-Level Annual Precipitation\n(Calculated from NetCDF Climate Data)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add statistics text
    valid_counties = counties_with_precip['annual_precipitation_mm'].count()
    total_counties = len(conus_counties)
    mean_precip = counties_with_precip['annual_precipitation_mm'].mean()
    
    stats_text = f"Coverage: {valid_counties}/{total_counties} counties ({valid_counties/total_counties*100:.1f}%)\n"
    stats_text += f"Mean: {mean_precip:.0f} mm/year"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('precipitation_county_results.png', dpi=300, bbox_inches='tight')
    print("County map saved as 'precipitation_county_results.png'")
    return fig

def create_comparison_stats(annual_precip, counties_with_data):
    """Create a comparison plot showing statistics."""
    print("Creating comparison statistics...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # NetCDF data histogram
    netcdf_data = annual_precip.values.flatten()
    netcdf_data = netcdf_data[~np.isnan(netcdf_data)]
    
    axes[0,0].hist(netcdf_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_xlabel('Annual Precipitation (mm/year)')
    axes[0,0].set_ylabel('Frequency (Grid Cells)')
    axes[0,0].set_title('Distribution of NetCDF Grid Cell Values')
    axes[0,0].axvline(np.mean(netcdf_data), color='red', linestyle='--', 
                label=f'Mean: {np.mean(netcdf_data):.0f} mm')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # County annual precipitation histogram
    county_data = counties_with_data['annual_precipitation_mm'].dropna()
    
    axes[0,1].hist(county_data, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,1].set_xlabel('Annual Precipitation (mm/year)')
    axes[0,1].set_ylabel('Frequency (Counties)')
    axes[0,1].set_title('Distribution of County-Level Annual Precipitation')
    axes[0,1].axvline(county_data.mean(), color='red', linestyle='--', 
                label=f'Mean: {county_data.mean():.0f} mm')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # High precipitation days histogram
    high_days_data = counties_with_data['high_precip_days_95th'].dropna()
    
    axes[1,0].hist(high_days_data, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].set_xlabel('High Precipitation Days (days/year)')
    axes[1,0].set_ylabel('Frequency (Counties)')
    axes[1,0].set_title('Distribution of High Precipitation Days (95th Percentile)')
    axes[1,0].axvline(high_days_data.mean(), color='red', linestyle='--', 
                label=f'Mean: {high_days_data.mean():.1f} days')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Scatter plot: Annual precipitation vs High precipitation days
    valid_data = counties_with_data.dropna(subset=['annual_precipitation_mm', 'high_precip_days_95th'])
    
    axes[1,1].scatter(valid_data['annual_precipitation_mm'], valid_data['high_precip_days_95th'], 
                     alpha=0.6, s=10, color='purple')
    axes[1,1].set_xlabel('Annual Precipitation (mm/year)')
    axes[1,1].set_ylabel('High Precipitation Days (days/year)')
    axes[1,1].set_title('Annual Precipitation vs High Precipitation Days')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = valid_data['annual_precipitation_mm'].corr(valid_data['high_precip_days_95th'])
    axes[1,1].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[1,1].transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('precipitation_comparison_stats.png', dpi=300, bbox_inches='tight')
    print("Comparison stats saved as 'precipitation_comparison_stats.png'")
    return fig

def plot_high_precipitation_days(counties_with_data):
    """Create choropleth map of high precipitation days (95th percentile)."""
    print("Creating high precipitation days map...")
    
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
    
    # Create custom colormap for precipitation days (different from precipitation amount)
    # Colors from dry (light yellow) to wet (deep blue/purple)
    colors = ['#FFFFA0', '#FFE066', '#FFA500', '#FF6B35', '#D2691E', '#8B4513', '#4169E1', '#191970']
    cmap_days = LinearSegmentedColormap.from_list('precip_days', colors, N=256)
    
    # Plot counties with high precipitation days data
    counties_with_precip_days = conus_counties.dropna(subset=['high_precip_days_95th'])
    
    if not counties_with_precip_days.empty:
        # Determine appropriate value range
        max_days = counties_with_precip_days['high_precip_days_95th'].max()
        vmax = min(max_days, 100)  # Cap at 100 days for better color resolution
        
        im = counties_with_precip_days.plot(
            column='high_precip_days_95th',
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
    counties_no_data = conus_counties[conus_counties['high_precip_days_95th'].isna()]
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
    if not counties_with_precip_days.empty:
        sm = plt.cm.ScalarMappable(cmap=cmap_days, norm=plt.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('High Precipitation Days (>5.0 mm/day, 95th percentile)', fontsize=12)
    
    # Add title
    ax.set_title('County-Level High Precipitation Days (95th Percentile)\n(Days per year with >5.0 mm precipitation)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add statistics text
    if not counties_with_precip_days.empty:
        valid_counties = counties_with_precip_days['high_precip_days_95th'].count()
        total_counties = len(conus_counties)
        mean_days = counties_with_precip_days['high_precip_days_95th'].mean()
        
        stats_text = f"Coverage: {valid_counties}/{total_counties} counties ({valid_counties/total_counties*100:.1f}%)\n"
        stats_text += f"Mean: {mean_days:.1f} days/year\n"
        stats_text += f"Note: Climate normals (30-year averages)"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('precipitation_high_days_95th.png', dpi=300, bbox_inches='tight')
    print("High precipitation days map saved as 'precipitation_high_days_95th.png'")
    return fig

def main():
    """Main function to create all visualizations."""
    print("=== PRECIPITATION DATA VISUALIZATION ===")
    print()
    
    # Load data
    annual_precip = load_netcdf_data()
    counties_with_data = load_county_results()
    
    # Create maps
    print("\nCreating visualizations...")
    fig1 = plot_netcdf_precipitation(annual_precip)
    fig2 = plot_county_precipitation(counties_with_data)
    fig3 = create_comparison_stats(annual_precip, counties_with_data)
    fig4 = plot_high_precipitation_days(counties_with_data)
    
    print("\n=== VISUALIZATION COMPLETE ===")
    print("Created files:")
    print("  1. precipitation_netcdf_input.png - Input NetCDF raster data")
    print("  2. precipitation_county_results.png - County-level choropleth map")
    print("  3. precipitation_comparison_stats.png - Enhanced statistical comparison")
    print("  4. precipitation_high_days_95th.png - High precipitation days map")
    
    # Show basic statistics
    valid_data = counties_with_data['annual_precipitation_mm'].dropna()
    valid_days_data = counties_with_data['high_precip_days_95th'].dropna()
    print(f"\nData Summary:")
    print(f"  NetCDF grid cells: {annual_precip.size:,}")
    print(f"  Total counties: {len(counties_with_data):,}")
    print(f"  Counties with precipitation data: {len(valid_data):,}")
    print(f"  Counties with high precipitation days data: {len(valid_days_data):,}")
    print(f"  Coverage: {len(valid_data)/len(counties_with_data)*100:.1f}%")
    print(f"  Mean annual precipitation: {valid_data.mean():.1f} mm/year")
    print(f"  Mean high precipitation days: {valid_days_data.mean():.1f} days/year")

if __name__ == "__main__":
    main() 