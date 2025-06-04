#!/usr/bin/env python3
"""
Alaska Climate Normals Visualization Script

Visualizes climate normal data from NetCDF files with proper geographic context.
Supports temperature and precipitation variables.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import argparse
from datetime import datetime

def setup_alaska_map(ax, extent=None):
    """Setup a map projection focused on Alaska."""
    if extent is None:
        # Default Alaska bounds (roughly)
        extent = [-180, -120, 50, 75]  # [west, east, south, north]
    
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, color='black')
    ax.add_feature(cfeature.STATES, linewidth=0.5, color='gray', alpha=0.7)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.6)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    return ax

def get_variable_config(variable_name, data_array):
    """Get visualization configuration for different climate variables."""
    
    # Convert temperature from Kelvin to Celsius if needed
    if variable_name in ['tas', 'tasmax', 'tasmin']:
        if data_array.max() > 100:  # Likely in Kelvin
            data_array = data_array - 273.15
            units = '°C'
        else:
            units = '°C'
    elif variable_name == 'pr':
        # Precipitation is usually in kg/m2/s, convert to mm/day
        data_array = data_array * 86400  # seconds per day
        units = 'mm/day'
    else:
        units = data_array.attrs.get('units', '')
    
    # Define colormaps and ranges for different variables
    configs = {
        'tas': {
            'cmap': 'RdYlBu_r',
            'vmin': -40,
            'vmax': 20,
            'title': 'Mean Temperature',
            'units': units,
            'extend': 'both'
        },
        'tasmax': {
            'cmap': 'Reds',
            'vmin': -30,
            'vmax': 30,
            'title': 'Maximum Temperature',
            'units': units,
            'extend': 'both'
        },
        'tasmin': {
            'cmap': 'Blues_r',
            'vmin': -50,
            'vmax': 10,
            'title': 'Minimum Temperature',
            'units': units,
            'extend': 'both'
        },
        'pr': {
            'cmap': 'YlGnBu',
            'vmin': 0,
            'vmax': 10,
            'title': 'Precipitation',
            'units': units,
            'extend': 'max'
        }
    }
    
    return configs.get(variable_name, {
        'cmap': 'viridis',
        'vmin': None,
        'vmax': None,
        'title': variable_name.upper(),
        'units': units,
        'extend': 'neither'
    }), data_array

def create_detailed_visualization(file_path, output_dir=None, save_plot=True):
    """Create a detailed visualization of the climate data."""
    
    # Load the data
    print(f"Loading data from: {file_path}")
    ds = xr.open_dataset(file_path)
    
    # Get the main variable (assuming it's the first data variable)
    var_names = list(ds.data_vars)
    if not var_names:
        print("No data variables found in the file!")
        return
    
    var_name = var_names[0]
    data = ds[var_name]
    
    print(f"Variable: {var_name}")
    print(f"Dimensions: {data.dims}")
    print(f"Shape: {data.shape}")
    
    # Get variable configuration
    config, data = get_variable_config(var_name, data)
    
    # Extract metadata
    target_year = ds.attrs.get('target_year', 'Unknown')
    period_type = ds.attrs.get('period_type', 'Unknown')
    region = ds.attrs.get('region', 'AK')
    
    # Create the figure
    fig = plt.figure(figsize=(15, 10))
    
    # Main map
    ax1 = plt.subplot(2, 2, (1, 3), projection=ccrs.PlateCarree())
    ax1 = setup_alaska_map(ax1)
    
    # Plot the data
    if 'dayofyear' in data.dims:
        # For climatology data, show annual mean
        annual_mean = data.mean(dim='dayofyear')
        im = annual_mean.plot(
            ax=ax1,
            transform=ccrs.PlateCarree(),
            cmap=config['cmap'],
            vmin=config['vmin'],
            vmax=config['vmax'],
            extend=config['extend'],
            add_colorbar=False
        )
        plot_title = f"Annual Mean {config['title']}"
    else:
        # For other data, plot as is
        im = data.plot(
            ax=ax1,
            transform=ccrs.PlateCarree(),
            cmap=config['cmap'],
            vmin=config['vmin'],
            vmax=config['vmax'],
            extend=config['extend'],
            add_colorbar=False
        )
        plot_title = config['title']
    
    ax1.set_title(f"{plot_title}\n{region} - {period_type.title()} Period - Target Year {target_year}", 
                  fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label(f"{config['title']} ({config['units']})", fontsize=12)
    
    # Seasonal analysis if dayofyear dimension exists
    if 'dayofyear' in data.dims:
        seasons = {
            'Winter (DJF)': [335, 366, 1, 59],  # Dec, Jan, Feb
            'Spring (MAM)': [60, 151],           # Mar, Apr, May
            'Summer (JJA)': [152, 243],          # Jun, Jul, Aug
            'Fall (SON)': [244, 334]             # Sep, Oct, Nov
        }
        
        for i, (season_name, days) in enumerate(seasons.items()):
            ax = plt.subplot(2, 4, 5 + i, projection=ccrs.PlateCarree())
            ax = setup_alaska_map(ax, extent=[-180, -120, 55, 72])
            
            # Select seasonal data
            if len(days) == 2:
                seasonal_data = data.sel(dayofyear=slice(days[0], days[1])).mean(dim='dayofyear')
            else:  # Winter spans year boundary
                winter_data = xr.concat([
                    data.sel(dayofyear=slice(days[0], days[1])),
                    data.sel(dayofyear=slice(days[2], days[3]))
                ], dim='dayofyear')
                seasonal_data = winter_data.mean(dim='dayofyear')
            
            im_season = seasonal_data.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=config['cmap'],
                vmin=config['vmin'],
                vmax=config['vmax'],
                extend=config['extend'],
                add_colorbar=False
            )
            
            ax.set_title(season_name, fontsize=10)
    
    # Add statistics text
    stats_text = f"""
Data Statistics:
• Min: {float(data.min()):.2f} {config['units']}
• Max: {float(data.max()):.2f} {config['units']}
• Mean: {float(data.mean()):.2f} {config['units']}
• Std: {float(data.std()):.2f} {config['units']}

File Info:
• Variable: {var_name}
• Target Year: {target_year}
• Period: {period_type}
• Region: {region}
• Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plot:
        if output_dir is None:
            output_dir = Path(file_path).parent / 'visualizations'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Create output filename
        file_stem = Path(file_path).stem
        output_file = output_dir / f"{file_stem}_visualization.png"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    
    plt.show()
    
    # Close dataset
    ds.close()

def create_simple_visualization(file_path, output_dir=None):
    """Create a simple, quick visualization."""
    
    print(f"Loading data from: {file_path}")
    ds = xr.open_dataset(file_path)
    
    var_names = list(ds.data_vars)
    var_name = var_names[0]
    data = ds[var_name]
    
    config, data = get_variable_config(var_name, data)
    
    # Create simple plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax = setup_alaska_map(ax)
    
    # Plot data (use annual mean if dayofyear exists)
    if 'dayofyear' in data.dims:
        plot_data = data.mean(dim='dayofyear')
    else:
        plot_data = data
    
    im = plot_data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=config['cmap'],
        vmin=config['vmin'],
        vmax=config['vmax'],
        extend=config['extend'],
        cbar_kwargs={'label': f"{config['title']} ({config['units']})"}
    )
    
    target_year = ds.attrs.get('target_year', 'Unknown')
    period_type = ds.attrs.get('period_type', 'Unknown')
    region = ds.attrs.get('region', 'AK')
    
    ax.set_title(f"{config['title']} - {region} - {period_type.title()} - Target Year {target_year}", 
                 fontsize=14, fontweight='bold')
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        file_stem = Path(file_path).stem
        output_file = output_dir / f"{file_stem}_simple.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Simple visualization saved to: {output_file}")
    
    plt.show()
    ds.close()

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Visualize Alaska climate normals data')
    parser.add_argument('file_path', help='Path to NetCDF file to visualize')
    parser.add_argument('--output-dir', '-o', help='Output directory for saved plots')
    parser.add_argument('--simple', '-s', action='store_true', 
                       help='Create simple visualization instead of detailed')
    parser.add_argument('--no-save', action='store_true', 
                       help='Do not save plots to file')
    
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    print(f"🗺️  Creating visualization for: {file_path.name}")
    
    try:
        if args.simple:
            create_simple_visualization(file_path, args.output_dir)
        else:
            create_detailed_visualization(file_path, args.output_dir, not args.no_save)
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # If no command line args, use the example file
    import sys
    if len(sys.argv) == 1:
        # Default to the file mentioned by the user
        example_file = "output/alaska_normals/tas/historical/tas_AK_historical_1980_30yr_normal.nc"
        if Path(example_file).exists():
            print(f"🗺️  Creating visualization for: {example_file}")
            create_detailed_visualization(example_file)
        else:
            print(f"Example file not found: {example_file}")
            print("Usage: python visualize_alaska_climate.py <path_to_netcdf_file>")
    else:
        main() 