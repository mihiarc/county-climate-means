"""
Stage handlers that bridge existing metrics processing to the new orchestration system.

These handlers wrap the existing metrics processing components to work with
the configuration-driven pipeline orchestrator.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import glob
import os

try:
    from ..utils.county_handler import load_us_counties
    import geopandas as gpd
    import xarray as xr
    import numpy as np
    METRICS_IMPORTS_AVAILABLE = True
except ImportError as e:
    METRICS_IMPORTS_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Metrics imports not available: {e}")


async def metrics_stage_handler(**context) -> Dict[str, Any]:
    """
    Stage handler for climate metrics processing.
    
    This handler:
    1. Loads county boundaries from census shapefiles
    2. Reads climate means outputs from Phase 1
    3. Calculates county-level statistics (mean, std, min, max, percentiles)
    4. Exports results in CSV format (extensible to other formats)
    """
    # Extract the nested stage configuration
    stage_config_dict = context['stage_config']
    stage_config = stage_config_dict.get('stage_config', {})  # Get the nested stage_config
    stage_inputs = context.get('stage_inputs', {})
    pipeline_context = context['pipeline_context']
    logger = context['logger']
    
    logger.info("Starting climate metrics processing stage")
    start_time = time.time()
    
    # Extract configuration
    # Check if input path is specified directly in stage config (for Phase 2 only runs)
    if 'input_means_path' in stage_config:
        means_output_path = Path(stage_config['input_means_path'])
        logger.info(f"Using direct input path from config: {means_output_path}")
    else:
        # Otherwise get from stage inputs (normal pipeline flow)
        means_output_path = Path(stage_inputs.get('climate_means', {}).get(
            'output_base_path',
            '/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/means'
        ))
    
    # Use organized output structure
    from county_climate.shared.config.output_paths import OrganizedOutputPaths
    organized_paths = OrganizedOutputPaths()
    output_base_path = Path(stage_config.get(
        'output_base_path',
        str(organized_paths.county_metrics_base)
    ))
    
    # Create output directory
    try:
        output_base_path.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        logger.error(f"Failed to create output directory {output_base_path}: {e}")
        return {
            'status': 'failed',
            'processing_stats': {
                'files_processed': 0,
                'counties_processed': 0,
                'metrics_calculated': [],
                'errors': [f"Failed to create output directory: {e}"]
            },
            'output_files': [],
            'error_message': f"Failed to create output directory: {e}"
        }
    
    # Initialize results
    results = {
        'status': 'completed',
        'output_files': [],
        'processing_stats': {
            'files_processed': 0,
            'counties_processed': 0,
            'metrics_calculated': [],
            'errors': []
        }
    }
    
    if not METRICS_IMPORTS_AVAILABLE:
        logger.error("Required metrics processing libraries not available")
        logger.error("Please install: geopandas, pyproj")
        return {
            'status': 'failed',
            'error_message': 'Required libraries not available for metrics processing',
            'processing_stats': {
                'files_processed': 0,
                'counties_processed': 0,
                'metrics_calculated': [],
                'errors': ['Missing required libraries: geopandas, pyproj']
            },
            'output_files': []
        }
        
    else:
        logger.info("Processing climate metrics with full libraries")
        
        try:
            # Get list of climate means files
            nc_files = list(means_output_path.rglob("*.nc")) if means_output_path.exists() else []
            logger.info(f"Found {len(nc_files)} climate means files to process")
            
            if not nc_files:
                logger.warning("No climate means files found to process")
                results['processing_stats']['errors'].append("No input files found")
            else:
                # Load county boundaries (simplified)
                logger.info("Loading county boundaries...")
                
                # Import the real processing function
                from .real_metrics_handler import process_single_climate_file
                
                # Get processing parameters from config
                variables = stage_config.get('variables', ['tas'])
                metrics = stage_config.get('metrics', ['mean', 'std', 'min', 'max'])
                percentiles = stage_config.get('percentiles', [])
                
                # Load county boundaries
                counties = load_us_counties()
                logger.info(f"Loaded {len(counties)} counties")
                
                # Filter to CONUS if specified
                if 'CONUS' in stage_config.get('regions', ['CONUS']):
                    counties = counties[
                        (counties.geometry.bounds.minx >= -130) & 
                        (counties.geometry.bounds.maxx <= -60) &
                        (counties.geometry.bounds.miny >= 20) &
                        (counties.geometry.bounds.maxy <= 50)
                    ]
                    logger.info(f"Filtered to {len(counties)} CONUS counties")
                
                # Process climate files
                total_counties_processed = 0
                for nc_file in nc_files:
                    try:
                        logger.info(f"Processing {nc_file.name}")
                        
                        # Determine variable from filename
                        variable = None
                        for var in variables:
                            if var in str(nc_file):
                                variable = var
                                break
                        
                        if not variable:
                            logger.warning(f"Could not determine variable for {nc_file.name}")
                            continue
                        
                        # Process the file
                        sample_counties = stage_config.get('sample_counties', None)
                        result = await process_single_climate_file(
                            nc_file, counties, output_base_path,
                            variable, metrics, percentiles, logger,
                            sample_counties=sample_counties
                        )
                        
                        if result['status'] == 'success':
                            results['output_files'].append(result['output_file'])
                            results['processing_stats']['files_processed'] += 1
                            total_counties_processed += result['counties_processed']
                        else:
                            results['processing_stats']['errors'].append(result['error'])
                        
                    except Exception as e:
                        logger.error(f"Error processing {nc_file}: {e}")
                        results['processing_stats']['errors'].append(str(e))
                
                results['processing_stats']['counties_processed'] = total_counties_processed
                results['processing_stats']['metrics_calculated'] = metrics + [f'p{p}' for p in percentiles]
                
        except Exception as e:
            logger.error(f"Error in metrics processing: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    results['processing_time_seconds'] = processing_time
    
    # Log summary
    logger.info(f"Metrics processing completed in {processing_time:.1f} seconds")
    logger.info(f"Processed {results['processing_stats']['files_processed']} files")
    logger.info(f"Created {len(results['output_files'])} output files")
    
    return results