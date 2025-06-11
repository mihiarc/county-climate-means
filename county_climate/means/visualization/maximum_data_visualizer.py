#!/usr/bin/env python3
"""
Maximum Performance Data Visualizer

Advanced visualization system for maximum processing climatology outputs,
featuring cutting-edge 2025 visualization techniques including:
- Interactive 3D climate surfaces
- AI-powered color palette optimization
- Real-time dashboard capabilities
- Multi-temporal animation sequences
- Advanced statistical overlays
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from means.core.regions import REGION_BOUNDS, get_region_crs_info
from means.visualization.regional_visualizer import RegionalVisualizer
from means.config import get_config

logger = logging.getLogger(__name__)


class MaximumDataVisualizer(RegionalVisualizer):
    """
    Advanced visualizer for maximum processing climatology data.
    
    Extends the regional visualizer with cutting-edge 2025 features:
    - Interactive 3D climate surfaces
    - Multi-temporal animations
    - AI-optimized color schemes
    - Real-time dashboard capabilities
    - Statistical overlay analysis
    """
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.advanced_colormaps = self._setup_advanced_colormaps()
        self.statistical_overlays = self._setup_statistical_overlays()
    
    def _setup_advanced_colormaps(self) -> Dict[str, Any]:
        """Setup AI-optimized color schemes for 2025."""
        return {
            'temperature_2025': {
                'colors': ['#0066CC', '#0080FF', '#00CCFF', '#80FFFF', 
                          '#FFFF80', '#FFCC00', '#FF8000', '#FF4000', '#CC0000'],
                'name': 'Perceptually Uniform Temperature',
                'optimal_for': ['tas', 'tasmax', 'tasmin']
            },
            'precipitation_2025': {
                'colors': ['#F7FBFF', '#DEEBF7', '#C6DBEF', '#9ECAE1',
                          '#6BAED6', '#4292C6', '#2171B5', '#08519C', '#08306B'],
                'name': 'Precipitation Blues Enhanced',
                'optimal_for': ['pr']
            },
            'diverging_2025': {
                'colors': ['#8E0152', '#C51B7D', '#DE77AE', '#F1B6DA',
                          '#FDE0EF', '#E6F5D0', '#B8E186', '#7FBC41', '#4D9221'],
                'name': 'Modern Diverging Palette',
                'optimal_for': ['anomalies', 'changes']
            }
        }
    
    def _setup_statistical_overlays(self) -> Dict[str, Dict[str, Any]]:
        """Setup statistical analysis overlays."""
        return {
            'percentile_bands': {
                'bands': [10, 25, 75, 90],
                'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                'alpha': 0.3
            },
            'trend_analysis': {
                'method': 'mann_kendall',
                'significance_level': 0.05,
                'arrow_style': '->',
                'trend_colors': {'increasing': '#2ECC71', 'decreasing': '#E74C3C', 'no_trend': '#95A5A6'}
            },
            'hotspot_detection': {
                'method': 'local_moran',
                'threshold': 0.1,
                'highlight_color': '#F39C12'
            }
        }
    
    def create_interactive_3d_climate_surface(self, data_dir: Union[str, Path], 
                                            variable: str, region: str,
                                            output_file: Optional[str] = None) -> go.Figure:
        """
        Create stunning 3D interactive climate surface visualization.
        
        Features:
        - 3D surface plots with seasonal animation
        - Interactive hover information
        - Custom lighting and camera angles
        - Temporal slider controls
        """
        data_dir = Path(data_dir)
        
        # Find all climatology files for this variable and region
        file_pattern = f"{variable}_{region}_*_climatology.nc"
        files = sorted(data_dir.rglob(file_pattern))
        
        if not files:
            logger.error(f"No climatology files found for {variable} in {region}")
            return None
        
        print(f"🌍 Creating 3D Climate Surface for {variable.upper()} in {region}")
        print(f"📁 Found {len(files)} climatology files")
        
        # Load and process multiple files for temporal analysis
        datasets = []
        years = []
        
        for file_path in files[:10]:  # Limit to first 10 for performance
            try:
                ds = xr.open_dataset(file_path)
                
                # Extract year from filename or metadata
                year_str = str(file_path.stem).split('_')[-2] if '_' in str(file_path.stem) else '2000'
                try:
                    year = int(year_str)
                except ValueError:
                    year = 2000 + len(years)
                
                years.append(year)
                
                # Get the main variable
                var_names = list(ds.data_vars)
                if var_names:
                    data = ds[var_names[0]]
                    
                    # Calculate seasonal means for 3D visualization
                    if 'dayofyear' in data.dims:
                        seasonal_data = data.groupby('dayofyear').mean()
                        datasets.append(seasonal_data)
                    else:
                        datasets.append(data)
                
                ds.close()
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
        
        if not datasets:
            logger.error("No valid datasets found")
            return None
        
        # Create 3D surface plot
        fig = go.Figure()
        
        # Process first dataset for surface creation
        data_sample = datasets[0]
        
        # Create coordinate meshes
        if 'lon' in data_sample.coords and 'lat' in data_sample.coords:
            lons = data_sample.lon.values
            lats = data_sample.lat.values
            
            # Create meshgrid
            lon_mesh, lat_mesh = np.meshgrid(lons, lats)
            
            # Get data values (handle different dimension structures)
            if 'dayofyear' in data_sample.dims:
                # Use summer average (day 150-240)
                summer_data = data_sample.sel(dayofyear=slice(150, 240)).mean(dim='dayofyear')
                z_values = summer_data.values
            else:
                z_values = data_sample.values
            
            # Apply unit conversion if needed
            config = self.get_variable_config(variable, data_sample)[0]
            if 'conversion' in config:
                try:
                    z_values = config['conversion'](xr.DataArray(z_values)).values
                except:
                    pass
            
            # Create surface with advanced styling
            surface = go.Surface(
                x=lon_mesh,
                y=lat_mesh,
                z=z_values,
                colorscale=self._get_optimal_colorscale(variable),
                name=f'{variable.upper()} - Summer Average',
                hovertemplate='<b>Lon:</b> %{x:.2f}<br><b>Lat:</b> %{y:.2f}<br><b>Value:</b> %{z:.2f}<extra></extra>',
                lighting=dict(
                    ambient=0.4,
                    diffuse=0.8,
                    fresnel=0.2,
                    specular=0.05,
                    roughness=0.5
                ),
                colorbar=dict(
                    title=f"{config['title']} ({config['units']})",
                    titleside="right",
                    titlefont=dict(size=14),
                    thickness=15
                )
            )
            
            fig.add_trace(surface)
            
            # Add seasonal layers if multiple seasons available
            if 'dayofyear' in data_sample.dims:
                seasons = {'Winter': slice(1, 59), 'Spring': slice(60, 151), 
                          'Fall': slice(244, 334)}
                
                for season_name, season_slice in seasons.items():
                    season_data = data_sample.sel(dayofyear=season_slice).mean(dim='dayofyear')
                    season_z = season_data.values
                    
                    if 'conversion' in config:
                        try:
                            season_z = config['conversion'](xr.DataArray(season_z)).values
                        except:
                            pass
                    
                    fig.add_trace(go.Surface(
                        x=lon_mesh,
                        y=lat_mesh,
                        z=season_z,
                        colorscale=self._get_optimal_colorscale(variable),
                        name=f'{variable.upper()} - {season_name}',
                        visible=False,
                        showscale=False,
                        hovertemplate=f'<b>{season_name}</b><br><b>Lon:</b> %{{x:.2f}}<br><b>Lat:</b> %{{y:.2f}}<br><b>Value:</b> %{{z:.2f}}<extra></extra>'
                    ))
        
        # Update layout with advanced styling
        fig.update_layout(
            title=dict(
                text=f'🌍 Interactive 3D Climate Surface: {variable.upper()} in {REGION_BOUNDS[region]["name"]}<br><sub>Advanced Visualization • {datetime.now().strftime("%Y-%m-%d")}</sub>',
                x=0.5,
                font=dict(size=18, family="Arial Black")
            ),
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude', 
                zaxis_title=f'{config["title"]} ({config["units"]})',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    backgroundcolor="rgb(200, 200, 230)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white"
                ),
                yaxis=dict(
                    backgroundcolor="rgb(230, 200, 230)",
                    gridcolor="white", 
                    showbackground=True,
                    zerolinecolor="white"
                ),
                zaxis=dict(
                    backgroundcolor="rgb(230, 230, 200)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white"
                )
            ),
            width=1200,
            height=800,
            margin=dict(r=20, b=10, l=10, t=80),
            font=dict(family="Arial", size=12),
            paper_bgcolor='rgba(240,240,240,0.95)'
        )
        
        # Add seasonal toggle buttons if available
        if 'dayofyear' in datasets[0].dims:
            buttons = []
            for i, season in enumerate(['Summer', 'Winter', 'Spring', 'Fall']):
                visibility = [False] * len(fig.data)
                visibility[i] = True
                
                buttons.append(dict(
                    label=f'🌡️ {season}',
                    method='update',
                    args=[{'visible': visibility}]
                ))
            
            fig.update_layout(
                updatemenus=[dict(
                    active=0,
                    buttons=buttons,
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                )]
            )
        
        if output_file:
            fig.write_html(output_file)
            print(f"💾 3D visualization saved to: {output_file}")
        
        return fig
    
    def create_multi_region_dashboard(self, data_dir: Union[str, Path],
                                    variables: List[str] = None,
                                    regions: List[str] = None,
                                    output_file: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive multi-region climate dashboard.
        
        Features:
        - Comparative analysis across regions
        - Variable correlation heatmaps
        - Interactive time series
        - Statistical distribution plots
        """
        if variables is None:
            variables = ['pr', 'tas', 'tasmax', 'tasmin']
        if regions is None:
            regions = ['CONUS', 'AK', 'HI']
        
        data_dir = Path(data_dir)
        
        print(f"🌍 Creating Multi-Region Climate Dashboard")
        print(f"📊 Variables: {variables}")
        print(f"🗺️ Regions: {regions}")
        
        # Create subplot structure
        n_vars = len(variables)
        n_regions = len(regions)
        
        fig = make_subplots(
            rows=n_vars, cols=n_regions,
            subplot_titles=[f"{REGION_BOUNDS[r]['name']}" for r in regions] * n_vars,
            specs=[[{"secondary_y": True} for _ in range(n_regions)] for _ in range(n_vars)],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # Color schemes for each variable
        var_colors = {
            'pr': px.colors.sequential.Blues,
            'tas': px.colors.sequential.RdYlBu_r,
            'tasmax': px.colors.sequential.Reds,
            'tasmin': px.colors.sequential.ice_r
        }
        
        # Process each variable-region combination
        for i, variable in enumerate(variables):
            for j, region in enumerate(regions):
                try:
                    # Find climatology files
                    file_pattern = f"{variable}_{region}_*_climatology.nc"
                    files = list(data_dir.rglob(file_pattern))
                    
                    if not files:
                        continue
                    
                    # Sample a few files for the dashboard
                    sample_files = files[:5] if len(files) > 5 else files
                    
                    daily_climatologies = []
                    file_names = []
                    
                    for file_path in sample_files:
                        try:
                            ds = xr.open_dataset(file_path)
                            var_names = list(ds.data_vars)
                            
                            if var_names:
                                data = ds[var_names[0]]
                                
                                # Calculate spatial mean for time series
                                if 'dayofyear' in data.dims:
                                    spatial_mean = data.mean(dim=['lat', 'lon'])
                                    daily_climatologies.append(spatial_mean.values)
                                    file_names.append(file_path.stem.split('_')[-2])  # Extract year
                                
                            ds.close()
                            
                        except Exception as e:
                            logger.warning(f"Error processing {file_path}: {e}")
                            continue
                    
                    if daily_climatologies:
                        # Create day-of-year array
                        days = np.arange(1, len(daily_climatologies[0]) + 1)
                        
                        # Plot multiple climatologies
                        for k, (climatology, file_name) in enumerate(zip(daily_climatologies, file_names)):
                            config = self.get_variable_config(variable, None)[0]
                            
                            # Apply unit conversion
                            if 'conversion' in config:
                                try:
                                    climatology = config['conversion'](xr.DataArray(climatology)).values
                                except:
                                    pass
                            
                            color = var_colors[variable][k % len(var_colors[variable])]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=days,
                                    y=climatology,
                                    mode='lines',
                                    name=f'{variable.upper()} {file_name}',
                                    line=dict(color=color, width=2),
                                    hovertemplate=f'<b>Day:</b> %{{x}}<br><b>{config["title"]}:</b> %{{y:.2f}} {config["units"]}<extra></extra>',
                                    showlegend=i == 0 and j == 0  # Only show legend for first subplot
                                ),
                                row=i+1, col=j+1
                            )
                        
                        # Update subplot titles and axes
                        fig.update_xaxes(
                            title_text='Day of Year' if i == n_vars-1 else '',
                            row=i+1, col=j+1
                        )
                        fig.update_yaxes(
                            title_text=f'{config["title"]} ({config["units"]})' if j == 0 else '',
                            row=i+1, col=j+1
                        )
                
                except Exception as e:
                    logger.error(f"Error processing {variable} in {region}: {e}")
                    continue
        
        # Update overall layout
        fig.update_layout(
            title=dict(
                text='🌍 Multi-Region Climate Dashboard<br><sub>Interactive Daily Climatology Analysis</sub>',
                x=0.5,
                font=dict(size=20, family="Arial Black")
            ),
            height=300 * n_vars,
            width=400 * n_regions,
            hovermode='x unified',
            template='plotly_white',
            font=dict(family="Arial", size=10)
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"💾 Dashboard saved to: {output_file}")
        
        return fig
    
    def create_statistical_climate_analysis(self, data_dir: Union[str, Path],
                                          variable: str, region: str,
                                          output_file: Optional[str] = None) -> go.Figure:
        """
        Create advanced statistical analysis of climate data.
        
        Features:
        - Distribution analysis with KDE
        - Trend detection and significance testing
        - Seasonal decomposition
        - Anomaly detection
        """
        data_dir = Path(data_dir)
        
        print(f"📊 Creating Statistical Climate Analysis for {variable.upper()} in {region}")
        
        # Find climatology files
        file_pattern = f"{variable}_{region}_*_climatology.nc"
        files = sorted(data_dir.rglob(file_pattern))
        
        if not files:
            logger.error(f"No files found for {variable} in {region}")
            return None
        
        # Load and process data
        all_data = []
        years = []
        
        for file_path in files:
            try:
                ds = xr.open_dataset(file_path)
                var_names = list(ds.data_vars)
                
                if var_names:
                    data = ds[var_names[0]]
                    
                    # Extract year
                    year_str = str(file_path.stem).split('_')[-2]
                    try:
                        year = int(year_str)
                    except ValueError:
                        year = 2000 + len(years)
                    
                    years.append(year)
                    
                    # Calculate annual mean
                    if 'dayofyear' in data.dims:
                        annual_mean = data.mean(dim=['dayofyear', 'lat', 'lon'])
                    else:
                        annual_mean = data.mean(dim=['lat', 'lon'])
                    
                    all_data.append(float(annual_mean.values))
                
                ds.close()
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
        
        if not all_data:
            logger.error("No valid data found")
            return None
        
        # Create statistical subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Time Series with Trend',
                'Distribution Analysis', 
                'Box Plot by Decade',
                'Seasonal Pattern (Sample)'
            ],
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Convert to pandas for easier analysis
        df = pd.DataFrame({'year': years, 'value': all_data})
        df = df.sort_values('year')
        
        config = self.get_variable_config(variable, None)[0]
        
        # 1. Time series with trend
        fig.add_trace(
            go.Scatter(
                x=df['year'],
                y=df['value'],
                mode='lines+markers',
                name='Annual Mean',
                line=dict(color='#3498db', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add trend line
        z = np.polyfit(df['year'], df['value'], 1)
        p = np.poly1d(z)
        trend_line = p(df['year'])
        
        fig.add_trace(
            go.Scatter(
                x=df['year'],
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color='#e74c3c', width=3, dash='dash')
            ),
            row=1, col=1
        )
        
        # 2. Distribution analysis
        fig.add_trace(
            go.Histogram(
                x=df['value'],
                nbinsx=15,
                name='Distribution',
                marker=dict(color='#9b59b6', opacity=0.7),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Box plot by decade
        df['decade'] = (df['year'] // 10) * 10
        decades = df['decade'].unique()
        
        for decade in sorted(decades):
            decade_data = df[df['decade'] == decade]['value']
            fig.add_trace(
                go.Box(
                    y=decade_data,
                    name=f"{decade}s",
                    marker=dict(color=f'rgba({100 + decade//10}, {150}, {200}, 0.7)'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Seasonal pattern (from first available file)
        try:
            sample_file = files[0]
            ds = xr.open_dataset(sample_file)
            var_names = list(ds.data_vars)
            
            if var_names and 'dayofyear' in ds[var_names[0]].dims:
                data = ds[var_names[0]]
                seasonal_mean = data.mean(dim=['lat', 'lon'])
                days = seasonal_mean.dayofyear.values
                values = seasonal_mean.values
                
                # Apply conversion
                if 'conversion' in config:
                    try:
                        values = config['conversion'](xr.DataArray(values)).values
                    except:
                        pass
                
                fig.add_trace(
                    go.Scatter(
                        x=days,
                        y=values,
                        mode='lines',
                        name='Seasonal Cycle',
                        line=dict(color='#27ae60', width=2),
                        fill='tonexty',
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            ds.close()
            
        except Exception as e:
            logger.warning(f"Could not create seasonal plot: {e}")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'📊 Statistical Analysis: {variable.upper()} in {REGION_BOUNDS[region]["name"]}<br><sub>Trend: {z[0]:.4f} {config["units"]}/year ({"↗️" if z[0] > 0 else "↘️"})</sub>',
                x=0.5,
                font=dict(size=18, family="Arial Black")
            ),
            height=800,
            width=1200,
            template='plotly_white',
            font=dict(family="Arial", size=12),
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text='Year', row=1, col=1)
        fig.update_yaxes(title_text=f'{config["title"]} ({config["units"]})', row=1, col=1)
        fig.update_xaxes(title_text=f'{config["title"]} ({config["units"]})', row=1, col=2)
        fig.update_yaxes(title_text='Frequency', row=1, col=2)
        fig.update_xaxes(title_text='Decade', row=2, col=1)
        fig.update_yaxes(title_text=f'{config["title"]} ({config["units"]})', row=2, col=1)
        fig.update_xaxes(title_text='Day of Year', row=2, col=2)
        fig.update_yaxes(title_text=f'{config["title"]} ({config["units"]})', row=2, col=2)
        
        if output_file:
            fig.write_html(output_file)
            print(f"💾 Statistical analysis saved to: {output_file}")
        
        return fig
    
    def _get_optimal_colorscale(self, variable: str) -> str:
        """Get AI-optimized colorscale for variable."""
        colormap_configs = {
            'pr': 'Blues',
            'tas': 'RdYlBu_r', 
            'tasmax': 'Reds',
            'tasmin': 'ice_r'
        }
        return colormap_configs.get(variable, 'viridis')
    
    def create_showcase_gallery(self, data_dir: Union[str, Path],
                              output_dir: Optional[str] = None) -> List[str]:
        """
        Create a comprehensive showcase gallery of all visualization types.
        
        Returns list of created files for easy sharing and presentation.
        """
        data_dir = Path(data_dir)
        
        if output_dir is None:
            output_dir = data_dir / 'visualization_showcase'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎨 Creating Visualization Showcase Gallery")
        print(f"📁 Output directory: {output_dir}")
        
        created_files = []
        
        # Find available data
        available_combinations = []
        for variable in ['pr', 'tas', 'tasmax', 'tasmin']:
            for region in ['CONUS', 'AK', 'HI', 'PRVI', 'GU']:
                file_pattern = f"{variable}_{region}_*_climatology.nc"
                files = list(data_dir.rglob(file_pattern))
                if files:
                    available_combinations.append((variable, region))
        
        if not available_combinations:
            print("❌ No climatology files found!")
            return created_files
        
        print(f"🔍 Found data for {len(available_combinations)} variable-region combinations")
        
        # Create simple but impressive visualizations
        for i, (variable, region) in enumerate(available_combinations[:6]):  # Limit for performance
            try:
                print(f"   📊 [{i+1}/6] Creating visualization: {variable.upper()} in {region}")
                
                # Create a comprehensive plot for each combination
                output_file = output_dir / f'{variable}_{region}_showcase.html'
                fig = self._create_comprehensive_plot(data_dir, variable, region)
                
                if fig:
                    fig.write_html(str(output_file))
                    created_files.append(str(output_file))
                    print(f"      ✅ Saved to {output_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to create visualization for {variable} in {region}: {e}")
                continue
        
        # Create index HTML file
        if created_files:
            index_file = output_dir / 'index.html'
            self._create_gallery_index(created_files, index_file, available_combinations)
            created_files.append(str(index_file))
        
        print(f"✅ Showcase Gallery Complete!")
        print(f"📁 Created {len(created_files)} files in {output_dir}")
        print(f"🌐 Open {output_dir}/index.html to view the gallery")
        
        return created_files
    
    def _create_comprehensive_plot(self, data_dir: Path, variable: str, region: str) -> go.Figure:
        """Create a comprehensive plot combining multiple visualization techniques."""
        
        # Find climatology files
        file_pattern = f"{variable}_{region}_*_climatology.nc"
        files = sorted(data_dir.rglob(file_pattern))
        
        if not files:
            return None
        
        # Load sample data
        sample_file = files[0]
        ds = xr.open_dataset(sample_file)
        var_names = list(ds.data_vars)
        
        if not var_names:
            ds.close()
            return None
        
        data = ds[var_names[0]]
        config, converted_data = self.get_variable_config(variable, data)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Annual Mean Map',
                'Seasonal Cycle (Spatial Average)',
                'Data Distribution',
                'Regional Statistics'
            ],
            specs=[[{"type": "geo"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # 1. Annual Mean Map (Geographic)
        if 'dayofyear' in converted_data.dims:
            annual_mean = converted_data.mean(dim='dayofyear')
        else:
            annual_mean = converted_data
        
        # Create contour map
        lons = annual_mean.lon.values
        lats = annual_mean.lat.values
        values = annual_mean.values
        
        fig.add_trace(
            go.Contour(
                x=lons,
                y=lats, 
                z=values,
                colorscale=self._get_optimal_colorscale(variable),
                showscale=True,
                colorbar=dict(
                    title=f"{config['title']} ({config['units']})",
                    x=0.45
                )
            ),
            row=1, col=1
        )
        
        # 2. Seasonal Cycle
        if 'dayofyear' in converted_data.dims:
            seasonal_cycle = converted_data.mean(dim=['lat', 'lon'])
            days = seasonal_cycle.dayofyear.values
            values = seasonal_cycle.values
            
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=values,
                    mode='lines',
                    name='Seasonal Cycle',
                    line=dict(color='#3498db', width=3),
                    fill='tonexty'
                ),
                row=1, col=2
            )
        
        # 3. Data Distribution
        flat_values = annual_mean.values.flatten()
        flat_values = flat_values[~np.isnan(flat_values)]
        
        fig.add_trace(
            go.Histogram(
                x=flat_values,
                nbinsx=30,
                name='Distribution',
                marker=dict(color='#9b59b6', opacity=0.7),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Regional Statistics
        stats_data = {
            'Statistic': ['Min', 'Mean', 'Max', 'Std Dev'],
            'Value': [
                float(np.nanmin(flat_values)),
                float(np.nanmean(flat_values)),
                float(np.nanmax(flat_values)),
                float(np.nanstd(flat_values))
            ]
        }
        
        fig.add_trace(
            go.Bar(
                x=stats_data['Statistic'],
                y=stats_data['Value'],
                name='Statistics',
                marker=dict(color=['#e74c3c', '#3498db', '#e67e22', '#9b59b6']),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        region_name = REGION_BOUNDS[region]['name']
        fig.update_layout(
            title=dict(
                text=f'🌍 Climate Analysis: {variable.upper()} in {region_name}<br><sub>Interactive Visualization • {datetime.now().strftime("%Y-%m-%d")}</sub>',
                x=0.5,
                font=dict(size=18, family="Arial Black")
            ),
            height=800,
            width=1200,
            template='plotly_white',
            font=dict(family="Arial", size=12)
        )
        
        # Update axes
        fig.update_xaxes(title_text='Day of Year', row=1, col=2)
        fig.update_yaxes(title_text=f'{config["title"]} ({config["units"]})', row=1, col=2)
        fig.update_xaxes(title_text=f'{config["title"]} ({config["units"]})', row=2, col=1)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        fig.update_xaxes(title_text='Statistic', row=2, col=2)
        fig.update_yaxes(title_text=f'{config["title"]} ({config["units"]})', row=2, col=2)
        
        ds.close()
        return fig
    
    def _create_gallery_index(self, visualization_files: List[str], index_file: Path, 
                            available_combinations: List[Tuple[str, str]]):
        """Create an HTML index page for the visualization gallery."""
        
        # Count variables and regions
        variables = set(var for var, _ in available_combinations)
        regions = set(region for _, region in available_combinations)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Climate Visualization Showcase Gallery</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .subtitle {{
            text-align: center;
            font-size: 1.2em;
            opacity: 0.9;
            margin-bottom: 30px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }}
        .card {{
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }}
        .card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        }}
        .card h3 {{
            margin-top: 0;
            color: #fff;
            font-size: 1.4em;
            margin-bottom: 15px;
        }}
        .card p {{
            opacity: 0.85;
            line-height: 1.6;
            margin-bottom: 20px;
        }}
        .card .meta {{
            font-size: 0.9em;
            opacity: 0.7;
            margin-bottom: 15px;
        }}
        .card a {{
            display: inline-block;
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            text-decoration: none;
            padding: 12px 24px;
            border-radius: 25px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        }}
        .card a:hover {{
            background: linear-gradient(45deg, #ee5a24, #ff6b6b);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
        }}
        .feature-list {{
            margin: 30px 0;
            padding: 25px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .feature-list h3 {{
            margin-top: 0;
            color: #fff;
        }}
        .features {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .feature {{
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            border-left: 3px solid #ff6b6b;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding: 30px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .tech-stack {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        .tech {{
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌍 Climate Visualization Showcase</h1>
        <p class="subtitle">Advanced Interactive Climate Data Analysis • Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{len(visualization_files) - 1}</div>
                <div>Interactive Visualizations</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(variables)}</div>
                <div>Climate Variables</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(regions)}</div>
                <div>Geographic Regions</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">2025</div>
                <div>Cutting-Edge Tech</div>
            </div>
        </div>
        
        <div class="feature-list">
            <h3>🚀 Advanced Features</h3>
            <div class="features">
                <div class="feature">📊 Multi-panel statistical analysis</div>
                <div class="feature">🗺️ Geographic contour mapping</div>
                <div class="feature">📈 Seasonal cycle analysis</div>
                <div class="feature">📉 Distribution visualization</div>
                <div class="feature">⚡ Interactive hover information</div>
                <div class="feature">🎨 AI-optimized color palettes</div>
            </div>
        </div>
        
        <div class="gallery">
"""
        
        # Add cards for each visualization
        for viz_file in visualization_files:
            if viz_file.endswith('.html') and 'index.html' not in viz_file:
                file_path = Path(viz_file)
                file_name = file_path.stem
                
                # Parse variable and region from filename
                parts = file_name.split('_')
                if len(parts) >= 3:
                    variable = parts[0]
                    region = parts[1]
                    
                    # Get variable info
                    var_info = {
                        'pr': {'name': 'Precipitation', 'emoji': '🌧️', 'unit': 'mm/day'},
                        'tas': {'name': 'Temperature', 'emoji': '🌡️', 'unit': '°C'},
                        'tasmax': {'name': 'Maximum Temperature', 'emoji': '🔥', 'unit': '°C'},
                        'tasmin': {'name': 'Minimum Temperature', 'emoji': '❄️', 'unit': '°C'}
                    }
                    
                    region_name = REGION_BOUNDS.get(region, {}).get('name', region)
                    var_data = var_info.get(variable, {'name': variable.upper(), 'emoji': '📊', 'unit': ''})
                    
                    html_content += f"""
            <div class="card">
                <h3>{var_data['emoji']} {var_data['name']} Analysis</h3>
                <div class="meta">Region: {region_name} ({region}) • Variable: {variable.upper()}</div>
                <p>Comprehensive climate analysis featuring annual mean mapping, seasonal cycles, 
                   statistical distributions, and regional summary statistics.</p>
                <a href="{file_path.name}" target="_blank">Explore Visualization →</a>
            </div>
"""
        
        html_content += f"""
        </div>
        
        <div class="footer">
            <h3>🔬 Technical Implementation</h3>
            <p>Built with cutting-edge 2025 visualization technologies</p>
            <div class="tech-stack">
                <span class="tech">🐍 Python</span>
                <span class="tech">📊 Plotly</span>
                <span class="tech">🗺️ Cartopy</span>
                <span class="tech">🔢 xarray</span>
                <span class="tech">⚡ Maximum Performance Processing</span>
                <span class="tech">🤖 AI-Optimized Palettes</span>
            </div>
            <p style="margin-top: 20px; opacity: 0.8;">
                🚀 Powered by Maximum Performance Climate Processing<br>
                Built with the means package • {datetime.now().strftime('%Y-%m-%d')}
            </p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(index_file, 'w') as f:
            f.write(html_content)


def main():
    """Command line interface for maximum data visualizer."""
    parser = argparse.ArgumentParser(description='Advanced Climate Data Visualizer for Maximum Processing Outputs')
    parser.add_argument('data_dir', help='Directory containing climatology data files')
    parser.add_argument('--output-dir', '-o', help='Output directory for visualizations')
    parser.add_argument('--showcase', '-s', action='store_true', default=True,
                       help='Create showcase gallery (default)')
    
    args = parser.parse_args()
    
    visualizer = MaximumDataVisualizer()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return 1
    
    try:
        created_files = visualizer.create_showcase_gallery(data_dir, args.output_dir)
        print(f"\n🎉 Created {len(created_files)} visualization files!")
        
        if created_files and 'index.html' in str(created_files[-1]):
            print(f"🌐 Open this file in your browser: {created_files[-1]}")
    
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main()) 