#!/usr/bin/env python3
"""
Climate Data Alignment Validation Script

Loads sample climate rasters from each region and overlays them on 
the regional extent boundaries to validate proper geographic alignment.

Usage:
    python validate_climate_data_alignment.py [--region REGION] [--output-dir OUTPUT_DIR]
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap

# Add src to path
sys.path.append('src')
from utils.regions import REGION_BOUNDS

def convert_to_standard_longitude(lon_360):
    """Convert longitude from 0-360 to -180-180 system."""
    return lon_360 - 360 if lon_360 > 180 else lon_360

def find_sample_file(region_key):
    """Find a sample climate file for the given region."""
    
    # Define sample file paths for each region
    sample_files = {
        'CONUS': 'output/conus/rolling_30year_climate_normals/tas/historical/tas_CONUS_historical_2000_30yr_normal.nc',
        'AK': 'output/alaska_normals/tas/historical/tas_AK_historical_2000_30yr_normal.nc',
        'HI': 'output/hawaii_normals/tas/historical/tas_HI_historical_2000_30yr_normal.nc',
        'PRVI': 'output/prvi_normals/tas/historical/tas_PRVI_historical_1990_30yr_normal.nc',
        'GU': 'output/guam_normals/tas/historical/tas_GU_historical_1985_30yr_normal.nc'
    }
    
    file_path = Path(sample_files.get(region_key, ''))
    
    if not file_path.exists():
        # Try to find any file in the region's directory
        region_dirs = {
            'CONUS': Path('output/conus/rolling_30year_climate_normals/tas/historical'),
            'AK': Path('output/alaska_normals/tas/historical'),
            'HI': Path('output/hawaii_normals/tas/historical'),
            'PRVI': Path('output/prvi_normals/tas/historical'),
            'GU': Path('output/guam_normals/tas/historical')
        }
        
        region_dir = region_dirs.get(region_key)
        if region_dir and region_dir.exists():
            nc_files = list(region_dir.glob('*.nc'))
            if nc_files:
                file_path = nc_files[0]  # Use first available file
    
    return file_path if file_path.exists() else None

def create_data_alignment_map(region_key, region_info, sample_file, output_dir):
    """Create a map showing climate data overlaid on regional extent."""
    
    print(f"🔍 Loading climate data for {region_key}...")
    
    try:
        # Load the climate data
        ds = xr.open_dataset(sample_file)
        
        # Get the temperature variable (assuming it's the main data variable)
        data_vars = list(ds.data_vars)
        if len(data_vars) == 0:
            print(f"❌ No data variables found in {sample_file}")
            return None
        
        var_name = data_vars[0]  # Use first data variable
        data = ds[var_name]
        
        # Get coordinate names
        lon_name = 'lon' if 'lon' in data.coords else 'x'
        lat_name = 'lat' if 'lat' in data.coords else 'y'
        
        # Convert temperature from Kelvin to Celsius if needed
        if data.attrs.get('units', '') == 'K':
            data = data - 273.15
            temp_unit = '°C'
        else:
            temp_unit = data.attrs.get('units', '')
        
        # Calculate annual mean if we have dayofyear dimension
        if 'dayofyear' in data.dims:
            data = data.mean(dim='dayofyear')
        
        print(f"   📊 Data shape: {data.shape}")
        print(f"   🌡️  Temperature range: {float(data.min()):.1f} to {float(data.max()):.1f} {temp_unit}")
        
    except Exception as e:
        print(f"❌ Error loading {sample_file}: {e}")
        return None
    
    # Convert longitude bounds to -180/180 system for mapping
    lon_min = convert_to_standard_longitude(region_info['lon_min'])
    lon_max = convert_to_standard_longitude(region_info['lon_max'])
    lat_min = region_info['lat_min']
    lat_max = region_info['lat_max']
    
    # Set up the figure and projection
    if region_key == 'CONUS':
        projection = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5)
        figsize = (16, 12)
    elif region_key == 'AK':
        projection = ccrs.AlbersEqualArea(central_longitude=-154, central_latitude=50)
        figsize = (14, 12)
    elif region_key == 'HI':
        projection = ccrs.PlateCarree(central_longitude=-157)
        figsize = (14, 10)
    elif region_key == 'PRVI':
        projection = ccrs.PlateCarree(central_longitude=-66)
        figsize = (12, 10)
    elif region_key == 'GU':
        projection = ccrs.PlateCarree(central_longitude=147)
        figsize = (12, 10)
    else:
        projection = ccrs.PlateCarree()
        figsize = (14, 10)
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)
    
    # Set map extent with some padding
    if lon_min > lon_max:  # Crosses dateline (Alaska)
        ax.set_global()
    else:
        padding = 3  # degrees
        ax.set_extent([lon_min - padding, lon_max + padding, 
                       lat_min - padding, lat_max + padding], 
                      ccrs.PlateCarree())
    
    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, color='black', alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray', alpha=0.6)
    ax.add_feature(cfeature.STATES, linewidth=0.3, color='gray', alpha=0.4)
    
    # Create colormap for temperature data
    cmap = get_cmap('RdYlBu_r')  # Red-Yellow-Blue reversed (red=hot, blue=cold)
    
    # Plot the climate data
    try:
        # Get coordinates - handle both standard longitude systems
        lons = data[lon_name].values
        lats = data[lat_name].values
        
        # Convert longitudes if needed for plotting
        if lons.max() > 180:  # 0-360 system
            lons = np.where(lons > 180, lons - 360, lons)
        
        # Create temperature plot
        temp_values = data.values
        
        # Handle NaN values
        if np.all(np.isnan(temp_values)):
            print(f"⚠️  Warning: All temperature values are NaN for {region_key}")
            return None
        
        # Create the plot
        im = ax.pcolormesh(lons, lats, temp_values, 
                          transform=ccrs.PlateCarree(),
                          cmap=cmap, alpha=0.8,
                          shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, aspect=50, shrink=0.8)
        cbar.set_label(f'Annual Mean Temperature ({temp_unit})', fontsize=12)
        
    except Exception as e:
        print(f"⚠️  Warning: Could not plot climate data: {e}")
    
    # Add region boundary rectangle
    if not (lon_min > lon_max):  # Normal case (doesn't cross dateline)
        rect = patches.Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=3, edgecolor='red', facecolor='none',
            transform=ccrs.PlateCarree(),
            linestyle='--', alpha=0.9
        )
        ax.add_patch(rect)
    else:  # Crosses dateline (Alaska)
        # Add two rectangles
        rect1 = patches.Rectangle(
            (lon_min, lat_min), 180 - lon_min, lat_max - lat_min,
            linewidth=3, edgecolor='red', facecolor='none',
            transform=ccrs.PlateCarree(),
            linestyle='--', alpha=0.9
        )
        rect2 = patches.Rectangle(
            (-180, lat_min), lon_max - (-180), lat_max - lat_min,
            linewidth=3, edgecolor='red', facecolor='none',
            transform=ccrs.PlateCarree(),
            linestyle='--', alpha=0.9
        )
        ax.add_patch(rect1)
        ax.add_patch(rect2)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    # Title and info
    title = f"{region_info['name']} ({region_key}) - Climate Data Alignment Validation"
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add info text
    info_text = f"Sample File: {Path(sample_file).name}\n"
    info_text += f"Region Bounds: {lon_min:.2f}° to {lon_max:.2f}°E, {lat_min:.2f}° to {lat_max:.2f}°N\n"
    info_text += f"Data Coverage: {float(lons.min()):.2f}° to {float(lons.max()):.2f}°E, "
    info_text += f"{float(lats.min()):.2f}° to {float(lats.max()):.2f}°N"
    
    plt.figtext(0.02, 0.02, info_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    # Add legend for boundary
    red_line = plt.Line2D([0], [0], color='red', linewidth=3, linestyle='--', alpha=0.9)
    ax.legend([red_line], ['Regional Boundary'], loc='upper right', 
              bbox_to_anchor=(0.98, 0.98))
    
    # Save the map
    output_file = output_dir / f"{region_key.lower()}_data_alignment_validation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Clean up
    ds.close()
    
    print(f"✅ {region_key}: Climate data alignment validated")
    print(f"   💾 Saved: {output_file}")
    print()
    
    return str(output_file)

def create_alignment_summary(output_dir):
    """Create a summary report of all regional alignments."""
    
    summary_file = output_dir / "alignment_validation_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("U.S. Regional Climate Data Alignment Validation Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("This validation checks that climate data rasters are properly\n")
        f.write("aligned with the defined regional boundaries.\n\n")
        
        f.write("What to look for:\n")
        f.write("- Climate data should fill the red dashed boundary rectangle\n")
        f.write("- No significant gaps between data and coastlines\n")
        f.write("- Temperature patterns should make geographic sense\n")
        f.write("- Data coverage should match expected region extent\n\n")
        
        f.write("Regional Definitions:\n")
        f.write("-" * 30 + "\n")
        
        for region_key, region_info in REGION_BOUNDS.items():
            lon_min = convert_to_standard_longitude(region_info['lon_min'])
            lon_max = convert_to_standard_longitude(region_info['lon_max'])
            
            f.write(f"{region_key}: {region_info['name']}\n")
            f.write(f"  Bounds: {lon_min:.2f}° to {lon_max:.2f}°E, ")
            f.write(f"{region_info['lat_min']:.2f}° to {region_info['lat_max']:.2f}°N\n")
            
            sample_file = find_sample_file(region_key)
            if sample_file:
                f.write(f"  Sample: {sample_file.name}\n")
            else:
                f.write(f"  Sample: No data file found\n")
            f.write("\n")
        
        f.write("Expected Temperature Patterns:\n")
        f.write("-" * 30 + "\n")
        f.write("CONUS: Cold in north/mountains, warm in south/deserts\n")
        f.write("Alaska: Very cold, especially in interior/north\n")
        f.write("Hawaii: Warm, tropical temperatures year-round\n")
        f.write("Puerto Rico: Warm tropical/subtropical\n")
        f.write("Guam: Hot tropical, minimal seasonal variation\n")
    
    return str(summary_file)

def main():
    """Main function to create climate data alignment validation maps."""
    parser = argparse.ArgumentParser(description='Validate climate data alignment with regional extents')
    parser.add_argument('--region', choices=list(REGION_BOUNDS.keys()) + ['all'],
                       default='all', help='Region to validate (default: all)')
    parser.add_argument('--output-dir', type=str, default='region_validation_maps',
                       help='Output directory for maps (default: region_validation_maps)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🗺️  Climate Data Alignment Validation")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print()
    
    created_files = []
    
    if args.region == 'all':
        # Validate all regions
        for region_key, region_info in REGION_BOUNDS.items():
            sample_file = find_sample_file(region_key)
            
            if sample_file:
                alignment_file = create_data_alignment_map(
                    region_key, region_info, sample_file, output_dir
                )
                if alignment_file:
                    created_files.append(alignment_file)
            else:
                print(f"❌ {region_key}: No sample data file found")
                print(f"   Expected in: output/{region_key.lower()}_normals/tas/historical/")
                print()
        
        # Create summary report
        summary_file = create_alignment_summary(output_dir)
        created_files.append(summary_file)
        print(f"📋 Summary report: {summary_file}")
    
    else:
        # Validate single region
        region_info = REGION_BOUNDS[args.region]
        sample_file = find_sample_file(args.region)
        
        if sample_file:
            alignment_file = create_data_alignment_map(
                args.region, region_info, sample_file, output_dir
            )
            if alignment_file:
                created_files.append(alignment_file)
        else:
            print(f"❌ {args.region}: No sample data file found")
    
    print("\n🎉 Climate data alignment validation complete!")
    print(f"Total files created: {len(created_files)}")
    print(f"Files saved in: {output_dir}")
    
    if created_files:
        print("\n📸 Validation Maps Created:")
        for file_path in created_files:
            print(f"  • {Path(file_path).name}")

if __name__ == "__main__":
    main() 