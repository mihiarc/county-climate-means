#!/usr/bin/env python3
"""
Unified Regional Climate Data Visualizer

A comprehensive visualization system for all climate regions with proper
coordinate reference systems and geographic projections for each region.
Supports data validation and visual inspection across all regions.
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
from typing import Dict, Any, Optional, Tuple, List
import logging

# Import our region definitions
from means.core.regions import REGION_BOUNDS, get_region_crs_info

logger = logging.getLogger(__name__)


class RegionalVisualizer:
    """
    Unified visualizer for all climate regions with region-specific projections.
    """
    
    def __init__(self):
        self.region_projections = self._setup_region_projections()
        self.variable_configs = self._setup_variable_configs()
    
    def _setup_region_projections(self) -> Dict[str, Dict[str, Any]]:
        """Setup appropriate map projections for each region."""
        projections = {}
        
        for region_key in REGION_BOUNDS.keys():
            crs_info = get_region_crs_info(region_key)
            
            if region_key == 'CONUS':
                # Albers Equal Area for CONUS
                projections[region_key] = {
                    'projection': ccrs.AlbersEqualArea(
                        central_longitude=-96,
                        central_latitude=37.5,
                        standard_parallels=(29.5, 45.5)
                    ),
                    'extent': [-125, -65, 25, 50],
                    'features': ['coastline', 'borders', 'states', 'lakes']
                }
            
            elif region_key == 'AK':
                # Alaska Albers
                projections[region_key] = {
                    'projection': ccrs.AlbersEqualArea(
                        central_longitude=-154,
                        central_latitude=50,
                        standard_parallels=(55, 65)
                    ),
                    'extent': [-180, -120, 50, 75],
                    'features': ['coastline', 'borders', 'states', 'ocean', 'land']
                }
            
            elif region_key == 'HI':
                # Hawaii - use UTM Zone 4N
                projections[region_key] = {
                    'projection': ccrs.UTM(zone=4),
                    'extent': [-178, -154, 18, 29],
                    'features': ['coastline', 'ocean', 'land']
                }
            
            elif region_key == 'PRVI':
                # Puerto Rico - use UTM Zone 19N
                projections[region_key] = {
                    'projection': ccrs.UTM(zone=19),
                    'extent': [-68, -64, 17, 19],
                    'features': ['coastline', 'borders', 'ocean', 'land']
                }
            
            elif region_key == 'GU':
                # Guam - use UTM Zone 55N
                projections[region_key] = {
                    'projection': ccrs.UTM(zone=55),
                    'extent': [144, 147, 13, 21],
                    'features': ['coastline', 'ocean', 'land']
                }
            
            else:
                # Default to PlateCarree
                projections[region_key] = {
                    'projection': ccrs.PlateCarree(),
                    'extent': crs_info['extent'],
                    'features': ['coastline', 'borders', 'ocean', 'land']
                }
        
        return projections
    
    def _setup_variable_configs(self) -> Dict[str, Dict[str, Any]]:
        """Setup visualization configurations for different climate variables."""
        return {
            'tas': {
                'cmap': 'RdYlBu_r',
                'vmin': -40,
                'vmax': 30,
                'title': 'Mean Temperature',
                'units': '°C',
                'extend': 'both',
                'conversion': lambda x: x - 273.15 if x.max() > 100 else x
            },
            'tasmax': {
                'cmap': 'Reds',
                'vmin': -30,
                'vmax': 40,
                'title': 'Maximum Temperature',
                'units': '°C',
                'extend': 'both',
                'conversion': lambda x: x - 273.15 if x.max() > 100 else x
            },
            'tasmin': {
                'cmap': 'Blues_r',
                'vmin': -50,
                'vmax': 20,
                'title': 'Minimum Temperature',
                'units': '°C',
                'extend': 'both',
                'conversion': lambda x: x - 273.15 if x.max() > 100 else x
            },
            'pr': {
                'cmap': 'YlGnBu',
                'vmin': 0,
                'vmax': 15,
                'title': 'Precipitation',
                'units': 'mm/day',
                'extend': 'max',
                'conversion': lambda x: x * 86400  # kg/m2/s to mm/day
            }
        }
    
    def setup_regional_map(self, ax, region_key: str) -> None:
        """Setup a map with region-specific projection and features."""
        if region_key not in self.region_projections:
            logger.warning(f"Unknown region {region_key}, using default projection")
            region_key = 'CONUS'
        
        proj_info = self.region_projections[region_key]
        
        # Set extent
        ax.set_extent(proj_info['extent'], crs=ccrs.PlateCarree())
        
        # Add features based on region
        feature_configs = {
            'coastline': {'linewidth': 0.8, 'color': 'black'},
            'borders': {'linewidth': 0.8, 'color': 'black'},
            'states': {'linewidth': 0.5, 'color': 'gray', 'alpha': 0.7},
            'ocean': {'facecolor': 'lightblue', 'alpha': 0.2},
            'lakes': {'facecolor': 'lightblue', 'alpha': 0.3}
        }
        
        for feature_name in proj_info['features']:
            if feature_name in feature_configs:
                feature = getattr(cfeature, feature_name.upper())
                ax.add_feature(feature, **feature_configs[feature_name])
        
        # Add gridlines (adjust based on region size)
        if region_key in ['PRVI', 'GU']:
            # Smaller regions need finer gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7,
                            x_inline=False, y_inline=False)
        else:
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7)
        
        gl.top_labels = False
        gl.right_labels = False
    
    def get_variable_config(self, variable_name: str, data_array: xr.DataArray) -> Tuple[Dict[str, Any], xr.DataArray]:
        """Get visualization configuration and convert data for a variable."""
        config = self.variable_configs.get(variable_name, {
            'cmap': 'viridis',
            'vmin': None,
            'vmax': None,
            'title': variable_name.upper(),
            'units': data_array.attrs.get('units', ''),
            'extend': 'neither',
            'conversion': lambda x: x
        })
        
        # Apply conversion if specified
        if 'conversion' in config:
            try:
                converted_data = config['conversion'](data_array)
                return config, converted_data
            except Exception as e:
                logger.warning(f"Failed to convert {variable_name}: {e}")
                return config, data_array
        
        return config, data_array
    
    def detect_region_from_data(self, ds: xr.Dataset) -> str:
        """Detect the region from dataset attributes or coordinate bounds."""
        # First check attributes
        if 'region' in ds.attrs:
            region = ds.attrs['region']
            if region in REGION_BOUNDS:
                return region
        
        # Try to detect from coordinate bounds
        lon_name = 'lon' if 'lon' in ds.coords else 'x'
        lat_name = 'lat' if 'lat' in ds.coords else 'y'
        
        lon_min = float(ds[lon_name].min())
        lon_max = float(ds[lon_name].max())
        lat_min = float(ds[lat_name].min())
        lat_max = float(ds[lat_name].max())
        
        # Convert to 0-360 if needed for comparison
        if lon_min < 0:
            lon_min = lon_min + 360 if lon_min < 0 else lon_min
            lon_max = lon_max + 360 if lon_max < 0 else lon_max
        
        # Check which region bounds best match
        for region_key, bounds in REGION_BOUNDS.items():
            if (abs(lon_min - bounds['lon_min']) < 5 and
                abs(lon_max - bounds['lon_max']) < 5 and
                abs(lat_min - bounds['lat_min']) < 2 and
                abs(lat_max - bounds['lat_max']) < 2):
                return region_key
        
        # Default to CONUS if can't detect
        logger.warning("Could not detect region from data, defaulting to CONUS")
        return 'CONUS'
    
    def create_comprehensive_visualization(self, file_path: str, output_dir: Optional[str] = None, 
                                         region_key: Optional[str] = None, save_plot: bool = True) -> None:
        """Create a comprehensive visualization with multiple views."""
        
        print(f"🗺️  Loading data from: {file_path}")
        ds = xr.open_dataset(file_path)
        
        # Get the main variable
        var_names = list(ds.data_vars)
        if not var_names:
            print("❌ No data variables found in the file!")
            return
        
        var_name = var_names[0]
        data = ds[var_name]
        
        # Handle cases where variable name is generic
        if var_name == '__xarray_dataarray_variable__':
            # Try to infer variable name from filename
            file_stem = Path(file_path).stem
            if 'tas_' in file_stem:
                var_name = 'tas'
            elif 'pr_' in file_stem:
                var_name = 'pr'
            elif 'tasmax_' in file_stem:
                var_name = 'tasmax'
            elif 'tasmin_' in file_stem:
                var_name = 'tasmin'
        
        # Check if coordinates are geographic or grid indices
        lon_name = 'lon' if 'lon' in ds.coords else 'x'
        lat_name = 'lat' if 'lat' in ds.coords else 'y'
        
        lon_min = float(ds[lon_name].min())
        lon_max = float(ds[lon_name].max())
        lat_min = float(ds[lat_name].min())
        lat_max = float(ds[lat_name].max())
        
        print(f"🔍 Debug: Checking coordinates - Lon: {lon_min}-{lon_max}, Lat: {lat_min}-{lat_max}")
        
        # Check if coordinates look like grid indices (0-based, small integers)
        coords_are_indices = (
            (lon_min == 0 and lon_max < 1000 and lat_min == 0 and lat_max < 1000) or
            (str(ds[lon_name].dtype) in ['int32', 'int64'] and str(ds[lat_name].dtype) in ['int32', 'int64'])
        )
        
        print(f"🔍 Debug: coords_are_indices = {coords_are_indices}")
        
        if coords_are_indices:
            print(f"⚠️  Warning: Coordinates appear to be grid indices, not geographic coordinates")
            print(f"   Lon range: {lon_min:.0f} to {lon_max:.0f} (dtype: {ds[lon_name].dtype})")
            print(f"   Lat range: {lat_min:.0f} to {lat_max:.0f} (dtype: {ds[lat_name].dtype})")
            print(f"   This file needs coordinate reconstruction for proper visualization")
            print(f"   The data processing may have lost the geographic coordinate information")
            return
        
        # Detect region if not provided
        if region_key is None:
            region_key = self.detect_region_from_data(ds)
        
        print(f"📊 Variable: {var_name}")
        print(f"🌍 Region: {region_key} ({REGION_BOUNDS[region_key]['name']})")
        print(f"📐 Dimensions: {data.dims}")
        print(f"📏 Shape: {data.shape}")
        
        # Get variable configuration
        config, data = self.get_variable_config(var_name, data)
        
        # Extract metadata from filename and attributes
        file_stem = Path(file_path).stem
        
        # Try to extract target year from filename (e.g., "tas_AK_ssp245_2049_30yr_normal")
        target_year = ds.attrs.get('target_year', 'Unknown')
        if target_year == 'Unknown':
            # Try to extract from filename
            import re
            year_match = re.search(r'_(\d{4})_', file_stem)
            if year_match:
                target_year = year_match.group(1)
        
        # Try to extract period type from filename
        period_type = ds.attrs.get('period_type', 'Unknown')
        if period_type == 'Unknown':
            if 'historical' in file_stem.lower():
                period_type = 'Historical'
            elif 'ssp245' in file_stem.lower():
                period_type = 'SSP2-4.5 Projection'
            elif 'ssp585' in file_stem.lower():
                period_type = 'SSP5-8.5 Projection'
            elif '30yr' in file_stem.lower():
                period_type = '30-Year Climate Normal'
        
        region_name = REGION_BOUNDS[region_key]['name']
        
        # Create the figure with region-specific projection
        proj_info = self.region_projections[region_key]
        fig = plt.figure(figsize=(20, 14))
        
        # Main map (takes up left 2/3 of top row)
        ax_main = plt.subplot(2, 3, (1, 2), projection=proj_info['projection'])
        self.setup_regional_map(ax_main, region_key)
        
        # Plot the main data
        if 'dayofyear' in data.dims:
            # For climatology data, show annual mean
            annual_mean = data.mean(dim='dayofyear')
            plot_data = annual_mean
            plot_title = f"Annual Mean {config['title']}"
        else:
            plot_data = data
            plot_title = config['title']
        
        # Simple pcolormesh plot (fixes gray NaN overlay)
        im = ax_main.pcolormesh(
            plot_data.lon, plot_data.lat, plot_data.values,
            transform=ccrs.PlateCarree(),
            cmap=config['cmap'],
            vmin=config['vmin'],
            vmax=config['vmax'],
            shading='auto'
        )
        
        ax_main.set_title(f"{plot_title}\n{region_name} - {period_type} - Target Year {target_year}", 
                         fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar with better positioning
        cbar = plt.colorbar(im, ax=ax_main, orientation='horizontal', 
                           pad=0.08, aspect=40, shrink=0.8)
        cbar.set_label(f"{config['title']} ({config['units']})", fontsize=12, fontweight='bold')
        
        # Seasonal analysis if dayofyear dimension exists
        if 'dayofyear' in data.dims:
            seasons = {
                'Winter (DJF)': [335, 366, 1, 59],
                'Spring (MAM)': [60, 151],
                'Summer (JJA)': [152, 243],
                'Fall (SON)': [244, 334]
            }
            
            # Create seasonal plots in a 2x2 grid on the right side and bottom
            seasonal_positions = [
                (1, 3),  # Top right - Winter
                (2, 1),  # Bottom left - Spring  
                (2, 2),  # Bottom center - Summer
                (2, 3)   # Bottom right - Fall
            ]
            
            for i, (season_name, days) in enumerate(seasons.items()):
                row, col = seasonal_positions[i]
                ax = plt.subplot(2, 3, (row-1)*3 + col, projection=proj_info['projection'])
                self.setup_regional_map(ax, region_key)
                
                # Select seasonal data
                if len(days) == 2:
                    seasonal_data = data.sel(dayofyear=slice(days[0], days[1])).mean(dim='dayofyear')
                else:  # Winter spans year boundary
                    winter_data = xr.concat([
                        data.sel(dayofyear=slice(days[0], days[1])),
                        data.sel(dayofyear=slice(days[2], days[3]))
                    ], dim='dayofyear')
                    seasonal_data = winter_data.mean(dim='dayofyear')
                
                ax.pcolormesh(
                    seasonal_data.lon, seasonal_data.lat, seasonal_data.values,
                    transform=ccrs.PlateCarree(),
                    cmap=config['cmap'],
                    vmin=config['vmin'],
                    vmax=config['vmax'],
                    shading='auto'
                )
                
                ax.set_title(season_name, fontsize=10)
        
        # Add comprehensive statistics in a better position
        stats_text = f"""Data Statistics:
Min: {float(plot_data.min()):.2f} {config['units']}
Max: {float(plot_data.max()):.2f} {config['units']}
Mean: {float(plot_data.mean()):.2f} {config['units']}
Std: {float(plot_data.std()):.2f} {config['units']}

File Information:
Variable: {var_name}
Region: {region_name} ({region_key})
Target Year: {target_year}
Period: {period_type}
Projection: {type(proj_info['projection']).__name__}
Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Coordinate Info:
Lon range: {float(ds.lon.min()):.2f} to {float(ds.lon.max()):.2f}
Lat range: {float(ds.lat.min()):.2f} to {float(ds.lat.max()):.2f}
Grid points: {ds.lon.size} × {ds.lat.size}"""
        
        # Position stats box in upper right, outside the plot area
        fig.text(0.98, 0.98, stats_text, fontsize=8, verticalalignment='top', 
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Adjust layout with more space
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        
        if save_plot:
            if output_dir is None:
                output_dir = Path(file_path).parent / 'visualizations'
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output filename
            file_stem = Path(file_path).stem
            output_file = output_dir / f"{file_stem}_{region_key}_comprehensive.png"
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved to: {output_file}")
        
        plt.show()
        ds.close()
    
    def create_simple_visualization(self, file_path: str, output_dir: Optional[str] = None,
                                  region_key: Optional[str] = None) -> None:
        """Create a simple, quick visualization for data validation."""
        
        print(f"🗺️  Loading data from: {file_path}")
        ds = xr.open_dataset(file_path)
        
        var_names = list(ds.data_vars)
        if not var_names:
            print("❌ No data variables found in the file!")
            return
        
        var_name = var_names[0]
        data = ds[var_name]
        
        # Handle cases where variable name is generic
        if var_name == '__xarray_dataarray_variable__':
            # Try to infer variable name from filename
            file_stem = Path(file_path).stem
            if 'tas_' in file_stem:
                var_name = 'tas'
            elif 'pr_' in file_stem:
                var_name = 'pr'
            elif 'tasmax_' in file_stem:
                var_name = 'tasmax'
            elif 'tasmin_' in file_stem:
                var_name = 'tasmin'
        
        # Check if coordinates are geographic or grid indices
        lon_name = 'lon' if 'lon' in ds.coords else 'x'
        lat_name = 'lat' if 'lat' in ds.coords else 'y'
        
        lon_min = float(ds[lon_name].min())
        lon_max = float(ds[lon_name].max())
        lat_min = float(ds[lat_name].min())
        lat_max = float(ds[lat_name].max())
        
        print(f"🔍 Debug: Checking coordinates - Lon: {lon_min}-{lon_max}, Lat: {lat_min}-{lat_max}")
        
        # Check if coordinates look like grid indices (0-based, small integers)
        coords_are_indices = (
            (lon_min == 0 and lon_max < 1000 and lat_min == 0 and lat_max < 1000) or
            (str(ds[lon_name].dtype) in ['int32', 'int64'] and str(ds[lat_name].dtype) in ['int32', 'int64'])
        )
        
        print(f"🔍 Debug: coords_are_indices = {coords_are_indices}")
        
        if coords_are_indices:
            print(f"⚠️  Warning: Coordinates appear to be grid indices, not geographic coordinates")
            print(f"   Lon range: {lon_min:.0f} to {lon_max:.0f} (dtype: {ds[lon_name].dtype})")
            print(f"   Lat range: {lat_min:.0f} to {lat_max:.0f} (dtype: {ds[lat_name].dtype})")
            print(f"   This file needs coordinate reconstruction for proper visualization")
            print(f"   The data processing may have lost the geographic coordinate information")
            ds.close()
            return
        
        # Detect region if not provided
        if region_key is None:
            region_key = self.detect_region_from_data(ds)
        
        config, data = self.get_variable_config(var_name, data)
        
        # Create simple plot with region-specific projection
        proj_info = self.region_projections[region_key]
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': proj_info['projection']})
        self.setup_regional_map(ax, region_key)
        
        # Plot data (use annual mean if dayofyear exists)
        if 'dayofyear' in data.dims:
            plot_data = data.mean(dim='dayofyear')
        else:
            plot_data = data
        
        # Simple pcolormesh plot (fixes gray NaN overlay)
        im = ax.pcolormesh(
            plot_data.lon, plot_data.lat, plot_data.values,
            transform=ccrs.PlateCarree(),
            cmap=config['cmap'],
            vmin=config['vmin'],
            vmax=config['vmax'],
            shading='auto'
        )
        
        # Add colorbar manually
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, aspect=40, shrink=0.8)
        cbar.set_label(f"{config['title']} ({config['units']})", fontsize=12)
        
        target_year = ds.attrs.get('target_year', 'Unknown')
        period_type = ds.attrs.get('period_type', 'Unknown')
        region_name = REGION_BOUNDS[region_key]['name']
        
        ax.set_title(f"{config['title']} - {region_name} - {period_type.title()} - Target Year {target_year}", 
                    fontsize=14, fontweight='bold')
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            file_stem = Path(file_path).stem
            output_file = output_dir / f"{file_stem}_{region_key}_simple.png"
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            print(f"💾 Simple visualization saved to: {output_file}")
        
        plt.show()
        ds.close()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Visualize climate data for any region')
    parser.add_argument('file_path', help='Path to NetCDF file or glob pattern')
    parser.add_argument('--output-dir', '-o', help='Output directory for saved plots')
    parser.add_argument('--region', '-r', choices=list(REGION_BOUNDS.keys()),
                       help='Force specific region (auto-detect if not provided)')
    parser.add_argument('--simple', '-s', action='store_true',
                       help='Create simple visualization instead of comprehensive')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plots to file')
    
    args = parser.parse_args()
    
    visualizer = RegionalVisualizer()
    
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    try:
        if args.simple:
            visualizer.create_simple_visualization(
                str(file_path), args.output_dir, args.region
            )
        else:
            visualizer.create_comprehensive_visualization(
                str(file_path), args.output_dir, args.region, not args.no_save
            )
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 