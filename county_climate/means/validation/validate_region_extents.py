#!/usr/bin/env python3
"""
U.S. Regional Extents Validation Script

Creates maps to visually validate the geographic bounds for all U.S. regions:
- CONUS (Continental United States)
- Alaska (AK)
- Hawaii (HI)
- Puerto Rico & Virgin Islands (PRVI)
- Guam & Northern Mariana Islands (GU)

Usage:
    python validate_region_extents.py [--region REGION] [--output-dir OUTPUT_DIR]
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np

from county_climate.means.core.regions import REGION_BOUNDS

def convert_to_standard_longitude(lon_360):
    """Convert longitude from 0-360 to -180-180 system."""
    return lon_360 - 360 if lon_360 > 180 else lon_360

def create_region_map(region_key, region_info, output_dir):
    """Create a detailed map for a specific region."""
    
    # Convert longitude bounds to -180/180 system for mapping
    lon_min = convert_to_standard_longitude(region_info['lon_min'])
    lon_max = convert_to_standard_longitude(region_info['lon_max'])
    lat_min = region_info['lat_min']
    lat_max = region_info['lat_max']
    
    # Handle longitude crossing dateline
    if lon_min > lon_max:  # Crosses dateline (like Alaska)
        lon_extent = [lon_min - 5, 180, -180, lon_max + 5]
        central_lon = (lon_min + lon_max + 360) / 2 if lon_min > 0 else (lon_min + lon_max) / 2
    else:
        lon_extent = [lon_min - 5, lon_max + 5]
        central_lon = (lon_min + lon_max) / 2
    
    lat_extent = [lat_min - 2, lat_max + 2]
    central_lat = (lat_min + lat_max) / 2
    
    # Create figure with appropriate projection
    if region_key == 'CONUS':
        projection = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5)
        figsize = (15, 10)
    elif region_key == 'AK':
        projection = ccrs.AlbersEqualArea(central_longitude=-154, central_latitude=50)
        figsize = (12, 10)
    elif region_key == 'HI':
        projection = ccrs.PlateCarree(central_longitude=-157)
        figsize = (12, 8)
    elif region_key == 'PRVI':
        projection = ccrs.PlateCarree(central_longitude=-66)
        figsize = (10, 8)
    elif region_key == 'GU':
        projection = ccrs.PlateCarree(central_longitude=147)
        figsize = (10, 8)
    else:
        projection = ccrs.PlateCarree()
        figsize = (12, 8)
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)
    
    # Set extent
    if len(lon_extent) == 4:  # Crosses dateline
        # For regions crossing dateline, use global extent and zoom manually
        ax.set_global()
    else:
        ax.set_extent([lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]], ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray')
    ax.add_feature(cfeature.STATES, linewidth=0.3, color='gray', alpha=0.7)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    
    # Add region boundary rectangle
    if not (lon_min > lon_max):  # Normal case (doesn't cross dateline)
        rect = patches.Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=3, edgecolor='red', facecolor='red', alpha=0.2,
            transform=ccrs.PlateCarree()
        )
        ax.add_patch(rect)
    else:  # Crosses dateline (like Alaska)
        # Add two rectangles
        rect1 = patches.Rectangle(
            (lon_min, lat_min), 180 - lon_min, lat_max - lat_min,
            linewidth=3, edgecolor='red', facecolor='red', alpha=0.2,
            transform=ccrs.PlateCarree()
        )
        rect2 = patches.Rectangle(
            (-180, lat_min), lon_max - (-180), lat_max - lat_min,
            linewidth=3, edgecolor='red', facecolor='red', alpha=0.2,
            transform=ccrs.PlateCarree()
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
    title = f"{region_info['name']} ({region_key}) - Region Extent Validation"
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add coordinate info as text
    coord_text = f"Longitude: {region_info['lon_min']:.2f}° to {region_info['lon_max']:.2f}°E (0-360 system)\n"
    coord_text += f"Standard: {lon_min:.2f}° to {lon_max:.2f}° (-180/180 system)\n"
    coord_text += f"Latitude: {lat_min:.2f}° to {lat_max:.2f}°N"
    
    plt.figtext(0.02, 0.02, coord_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Save the map
    output_file = output_dir / f"{region_key.lower()}_extent_validation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ {region_key}: {region_info['name']}")
    print(f"   📍 Bounds: {lon_min:.2f}° to {lon_max:.2f}°E, {lat_min:.2f}° to {lat_max:.2f}°N")
    print(f"   💾 Saved: {output_file}")
    print()
    
    return str(output_file)

def create_overview_map(output_dir):
    """Create an overview map showing all U.S. regions."""
    
    # Create figure with global projection
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplots for different views
    # Main overview (top)
    ax1 = plt.subplot(2, 3, (1, 3), projection=ccrs.PlateCarree())
    ax1.set_global()
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray')
    ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
    ax1.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    
    # Color scheme for regions
    colors = {
        'CONUS': ('blue', 'Continental United States'),
        'AK': ('green', 'Alaska'),
        'HI': ('orange', 'Hawaii'),
        'PRVI': ('purple', 'Puerto Rico & Virgin Islands'),
        'GU': ('red', 'Guam & Northern Mariana Islands')
    }
    
    # Add all region rectangles to overview
    for region_key, region_info in REGION_BOUNDS.items():
        color, label = colors[region_key]
        
        # Convert to standard longitude
        lon_min = convert_to_standard_longitude(region_info['lon_min'])
        lon_max = convert_to_standard_longitude(region_info['lon_max'])
        lat_min = region_info['lat_min']
        lat_max = region_info['lat_max']
        
        if not (lon_min > lon_max):  # Normal case
            rect = patches.Rectangle(
                (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
                transform=ccrs.PlateCarree(), label=label
            )
            ax1.add_patch(rect)
        else:  # Crosses dateline
            # Two rectangles for Alaska
            rect1 = patches.Rectangle(
                (lon_min, lat_min), 180 - lon_min, lat_max - lat_min,
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
                transform=ccrs.PlateCarree()
            )
            rect2 = patches.Rectangle(
                (-180, lat_min), lon_max - (-180), lat_max - lat_min,
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
                transform=ccrs.PlateCarree(), label=label
            )
            ax1.add_patch(rect1)
            ax1.add_patch(rect2)
    
    ax1.set_title('U.S. Climate Processing Regions Overview', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='lower left', bbox_to_anchor=(0, 0))
    
    # Detail views for smaller regions
    detail_regions = [
        ('HI', 4, [-180, -150, 15, 30]),
        ('PRVI', 5, [-70, -62, 16, 20]),
        ('GU', 6, [140, 150, 10, 25])
    ]
    
    for region_key, subplot_pos, extent in detail_regions:
        ax = plt.subplot(2, 3, subplot_pos, projection=ccrs.PlateCarree())
        ax.set_extent(extent, ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        
        # Add region rectangle
        region_info = REGION_BOUNDS[region_key]
        color, label = colors[region_key]
        
        lon_min = convert_to_standard_longitude(region_info['lon_min'])
        lon_max = convert_to_standard_longitude(region_info['lon_max'])
        lat_min = region_info['lat_min']
        lat_max = region_info['lat_max']
        
        rect = patches.Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
            transform=ccrs.PlateCarree()
        )
        ax.add_patch(rect)
        
        ax.set_title(f'{region_key}: {label}', fontsize=12, fontweight='bold')
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
    
    plt.tight_layout()
    
    # Save overview map
    overview_file = output_dir / "us_regions_overview.png"
    plt.savefig(overview_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    return str(overview_file)

def main():
    """Main function to create region validation maps."""
    parser = argparse.ArgumentParser(description='Validate U.S. regional extents for climate processing')
    parser.add_argument('--region', choices=list(REGION_BOUNDS.keys()) + ['all'], 
                       default='all', help='Region to map (default: all)')
    parser.add_argument('--output-dir', type=str, default='region_validation_maps',
                       help='Output directory for maps (default: region_validation_maps)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🗺️  U.S. Regional Extents Validation")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print()
    
    created_files = []
    
    if args.region == 'all':
        # Create overview map first
        print("Creating overview map...")
        overview_file = create_overview_map(output_dir)
        created_files.append(overview_file)
        print(f"📊 Overview map saved: {overview_file}")
        print()
        
        # Create individual region maps
        print("Creating individual region maps...")
        for region_key, region_info in REGION_BOUNDS.items():
            region_file = create_region_map(region_key, region_info, output_dir)
            created_files.append(region_file)
    else:
        # Create single region map
        region_info = REGION_BOUNDS[args.region]
        region_file = create_region_map(args.region, region_info, output_dir)
        created_files.append(region_file)
    
    print("🎉 Region validation maps created successfully!")
    print(f"Total files created: {len(created_files)}")
    print(f"Files saved in: {output_dir}")
    
    # Print summary table
    print("\n📋 Regional Summary:")
    print("-" * 80)
    print(f"{'Region':<8} {'Name':<35} {'Longitude Range':<20} {'Latitude Range':<15}")
    print("-" * 80)
    
    for region_key, region_info in REGION_BOUNDS.items():
        lon_min = convert_to_standard_longitude(region_info['lon_min'])
        lon_max = convert_to_standard_longitude(region_info['lon_max'])
        
        lon_range = f"{lon_min:.1f}° to {lon_max:.1f}°"
        lat_range = f"{region_info['lat_min']:.1f}° to {region_info['lat_max']:.1f}°"
        
        print(f"{region_key:<8} {region_info['name']:<35} {lon_range:<20} {lat_range:<15}")

if __name__ == "__main__":
    main() 