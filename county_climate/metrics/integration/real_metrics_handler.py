"""
Real implementation of metrics stage handler for county-level climate metrics.

This module provides the actual processing logic for Phase 2, calculating
county-level statistics from Phase 1 climate means outputs.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from ..utils.county_handler import load_us_counties


async def real_metrics_stage_handler(**context) -> Dict[str, Any]:
    """
    Real implementation of climate metrics processing.
    
    This handler:
    1. Loads county boundaries
    2. Processes climate means files from Phase 1
    3. Calculates county-level statistics
    4. Outputs results in multiple formats
    """
    stage_config = context['stage_config']
    stage_inputs = context.get('stage_inputs', {})
    pipeline_context = context['pipeline_context']
    logger = context['logger']
    
    logger.info("Starting real climate metrics processing")
    start_time = time.time()
    
    # Extract configuration
    if 'input_means_path' in stage_config:
        means_output_path = Path(stage_config['input_means_path'])
        logger.info(f"Using direct input path: {means_output_path}")
    else:
        means_output_path = Path(stage_inputs.get('climate_means', {}).get(
            'output_base_path',
            '/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/means'
        ))
    
    output_base_path = Path(stage_config.get(
        'output_base_path',
        '/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/metrics'
    ))
    
    # Create output directory
    try:
        output_base_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        return {
            'status': 'failed',
            'error_message': str(e),
            'processing_stats': {'files_processed': 0}
        }
    
    # Get processing parameters
    variables = stage_config.get('variables', ['tas'])
    regions = stage_config.get('regions', ['CONUS'])
    scenarios = stage_config.get('scenarios', ['historical', 'ssp245'])
    metrics = stage_config.get('metrics', ['mean', 'std', 'min', 'max'])
    percentiles = stage_config.get('percentiles', [])
    
    # Load county boundaries
    logger.info("Loading county boundaries...")
    try:
        counties = load_us_counties()
        logger.info(f"Loaded {len(counties)} counties")
        
        # Filter counties by region if needed
        if 'CONUS' in regions:
            counties = counties[
                (counties.geometry.bounds.minx >= -130) & 
                (counties.geometry.bounds.maxx <= -60) &
                (counties.geometry.bounds.miny >= 20) &
                (counties.geometry.bounds.maxy <= 50)
            ]
            logger.info(f"Filtered to {len(counties)} CONUS counties")
    except Exception as e:
        logger.error(f"Failed to load county boundaries: {e}")
        return {
            'status': 'failed',
            'error_message': f"County loading failed: {e}",
            'processing_stats': {'files_processed': 0}
        }
    
    # Initialize results tracking
    results = {
        'status': 'completed',
        'output_files': [],
        'processing_stats': {
            'files_processed': 0,
            'counties_processed': 0,
            'metrics_calculated': metrics + [f'p{p}' for p in percentiles],
            'errors': [],
            'processing_details': []
        }
    }
    
    # Find and process climate files
    for variable in variables:
        for scenario in scenarios:
            # Find files for this variable/scenario combination
            pattern = f"{variable}/*{scenario}*/*.nc"
            nc_files = list(means_output_path.glob(pattern))
            
            if not nc_files:
                logger.warning(f"No files found for {variable}/{scenario}")
                continue
            
            logger.info(f"Processing {len(nc_files)} files for {variable}/{scenario}")
            
            # Process each file
            for nc_file in nc_files:
                try:
                    sample_counties = stage_config.get('sample_counties', None)
                    result = await process_single_climate_file(
                        nc_file, counties, output_base_path, 
                        variable, metrics, percentiles, logger,
                        sample_counties=sample_counties
                    )
                    
                    if result['status'] == 'success':
                        results['output_files'].append(result['output_file'])
                        results['processing_stats']['files_processed'] += 1
                        results['processing_stats']['counties_processed'] += result['counties_processed']
                        results['processing_stats']['processing_details'].append({
                            'file': nc_file.name,
                            'counties': result['counties_processed'],
                            'processing_time': result['processing_time']
                        })
                    else:
                        results['processing_stats']['errors'].append(result['error'])
                        
                except Exception as e:
                    logger.error(f"Error processing {nc_file}: {e}")
                    results['processing_stats']['errors'].append(str(e))
    
    # Calculate total processing time
    processing_time = time.time() - start_time
    results['processing_time_seconds'] = processing_time
    
    # Log summary
    logger.info(f"Metrics processing completed in {processing_time:.1f} seconds")
    logger.info(f"Processed {results['processing_stats']['files_processed']} files")
    logger.info(f"Total counties processed: {results['processing_stats']['counties_processed']}")
    logger.info(f"Created {len(results['output_files'])} output files")
    
    if results['processing_stats']['errors']:
        logger.warning(f"Encountered {len(results['processing_stats']['errors'])} errors")
    
    return results


async def process_single_climate_file(
    nc_file: Path, 
    counties_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    variable_name: str,
    metrics: List[str],
    percentiles: List[float],
    logger: logging.Logger,
    sample_counties: Optional[int] = None
) -> Dict[str, Any]:
    """Process a single climate file and calculate county statistics."""
    
    start_time = time.time()
    logger.info(f"Processing {nc_file.name}")
    
    try:
        # Open the climate dataset
        ds = xr.open_dataset(nc_file)
        
        # Get the climate variable
        if variable_name not in ds:
            return {
                'status': 'error',
                'error': f"Variable {variable_name} not found in {nc_file.name}"
            }
        
        climate_data = ds[variable_name]
        
        # Convert longitude from 0-360 to -180-180 if needed
        lons = ds.lon.values
        if lons.max() > 180:
            lons = np.where(lons > 180, lons - 360, lons)
            ds = ds.assign_coords(lon=lons)
            climate_data = ds[variable_name]
        
        # Calculate annual mean (average across all days)
        annual_data = climate_data.mean(dim='dayofyear')
        
        # Process counties
        county_results = []
        
        # Process all counties or sample if specified
        if sample_counties is not None and sample_counties < len(counties_gdf):
            counties_to_process = counties_gdf.sample(sample_counties, random_state=42)
            logger.info(f"Processing sample of {sample_counties} counties")
        else:
            counties_to_process = counties_gdf
        
        for idx, county in counties_to_process.iterrows():
            # Get county bounds
            bounds = county.geometry.bounds
            
            # Select data within county bounds (with small buffer)
            buffer = 0.1  # degrees
            county_data = annual_data.sel(
                lon=slice(bounds[0] - buffer, bounds[2] + buffer),
                lat=slice(bounds[1] - buffer, bounds[3] + buffer)
            )
            
            if county_data.size > 0:
                # Convert to numpy array for calculations
                data_values = county_data.values.flatten()
                data_values = data_values[~np.isnan(data_values)]
                
                if len(data_values) > 0:
                    # Calculate statistics
                    county_stats = {
                        'GEOID': county['GEOID'],
                        'NAME': county.get('NAME', 'Unknown'),
                        'STATE': county.get('STATEFP', 'Unknown'),
                        'num_grid_points': len(data_values)
                    }
                    
                    # Basic metrics
                    if 'mean' in metrics:
                        county_stats['mean'] = float(np.mean(data_values))
                    if 'std' in metrics:
                        # Only calculate std if we have multiple data points
                        if len(data_values) > 1:
                            county_stats['std'] = float(np.std(data_values))
                        else:
                            county_stats['std'] = 0.0  # No variation with single point
                    if 'min' in metrics:
                        county_stats['min'] = float(np.min(data_values))
                    if 'max' in metrics:
                        county_stats['max'] = float(np.max(data_values))
                    
                    # Percentiles
                    for p in percentiles:
                        county_stats[f'p{p}'] = float(np.percentile(data_values, p))
                    
                    # Convert temperature from Kelvin to Celsius if needed
                    if variable_name in ['tas', 'tasmax', 'tasmin']:
                        for key in county_stats:
                            # Only convert actual temperature values, not std or num_grid_points
                            if key in ['mean', 'min', 'max'] or key.startswith('p'):
                                county_stats[key] -= 273.15
                    
                    county_results.append(county_stats)
        
        # Create DataFrame and save
        if county_results:
            df = pd.DataFrame(county_results)
            
            # Create output filename
            output_name = f"{nc_file.stem}_county_metrics.csv"
            output_file = output_dir / output_name
            
            # Save as CSV
            df.to_csv(output_file, index=False)
            
            processing_time = time.time() - start_time
            logger.info(f"Saved metrics for {len(df)} counties in {processing_time:.1f}s")
            
            return {
                'status': 'success',
                'output_file': str(output_file),
                'counties_processed': len(df),
                'processing_time': processing_time
            }
        else:
            return {
                'status': 'error',
                'error': 'No county data extracted'
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'error': f"Processing error: {str(e)}"
        }
    finally:
        if 'ds' in locals():
            ds.close()