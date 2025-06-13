"""
Stage handler for climate means processing using the refactored parallel variables architecture.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Check if means processing components are available
MEANS_IMPORTS_AVAILABLE = True
try:
    from county_climate.means.core.regional_climate_processor_v2 import (
        RegionalClimateProcessorV2, RegionalProcessingConfigV2
    )
except ImportError:
    MEANS_IMPORTS_AVAILABLE = False


def means_stage_handler_v2(**context) -> Dict[str, Any]:
    """
    Refactored handler for the climate means processing stage.
    
    This version properly parallelizes variables while keeping years sequential
    to avoid NetCDF file conflicts.
    
    Expected configuration parameters:
        variables (list): Climate variables to process
        regions (list): Geographic regions to process
        scenarios (list): Climate scenarios to process
        max_workers (int): Maximum workers (default: number of variables)
        enable_progress_tracking (bool): Enable Rich progress display
    """
    # Extract the nested stage configuration
    stage_config_dict = context['stage_config']
    stage_config = stage_config_dict.get('stage_config', {})
    pipeline_context = context['pipeline_context']
    logger = context['logger']
    
    logger.info("Starting climate means processing stage (V2 - Parallel Variables)")
    start_time = time.time()
    
    # Check if means imports are available
    if not MEANS_IMPORTS_AVAILABLE:
        logger.error("Means processing components not available")
        return {
            'status': 'failed',
            'error': 'Means processing components not available',
            'output_files': [],
            'processing_stats': {'total_datasets': 0, 'successful_datasets': 0},
        }
    
    try:
        # Extract configuration parameters
        variables = stage_config.get('variables', ['tas'])
        regions = stage_config.get('regions', ['CONUS'])
        scenarios = stage_config.get('scenarios', ['historical'])
        
        # Processing parameters
        max_workers = stage_config.get('max_workers', None)  # None = one per variable
        
        # Paths
        input_base_path = Path(pipeline_context.get('base_data_path', '/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM'))
        output_base_path = Path(stage_config.get('output_base_path', '/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/means'))
        
        # Create output directory
        output_base_path.mkdir(parents=True, exist_ok=True)
        
        # Results tracking
        all_results = {}
        output_files = []
        processing_stats = {
            'total_tasks': len(regions) * len(scenarios),
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_processing_time': 0,
        }
        
        # Progress tracking
        use_progress_tracking = stage_config.get('enable_progress_tracking', True)
        
        logger.info(f"Processing {len(variables)} variables for {len(regions)} regions across {len(scenarios)} scenarios")
        logger.info(f"Total processing tasks: {processing_stats['total_tasks']}")
        logger.info(f"Parallel architecture: {len(variables)} workers (one per variable)")
        
        # Process each region/scenario combination
        for region in regions:
            for scenario in scenarios:
                task_name = f"{region}/{scenario}"
                
                try:
                    logger.info(f"Processing {region}/{scenario} with parallel variables: {', '.join(variables)}")
                    
                    # Create configuration for this task
                    config = RegionalProcessingConfigV2(
                        region_key=region,
                        variables=variables,
                        input_data_dir=input_base_path,
                        output_base_dir=output_base_path,
                        max_workers=max_workers,
                        use_progress_tracking=use_progress_tracking
                    )
                    
                    # Create processor
                    processor = RegionalClimateProcessorV2(config)
                    
                    # Process this region/scenario
                    task_start_time = time.time()
                    result = processor.process_single_scenario(scenario)
                    task_time = time.time() - task_start_time
                    
                    # Store results
                    all_results[task_name] = result
                    
                    # Extract successful variables
                    if 'variables' in result:
                        successful_vars = [
                            v for v, r in result['variables'].items() 
                            if r.get('status') == 'success'
                        ]
                        failed_vars = [
                            v for v, r in result['variables'].items() 
                            if r.get('status') != 'success'
                        ]
                        
                        if successful_vars:
                            processing_stats['successful_tasks'] += 1
                            logger.info(f"✅ {task_name}: {len(successful_vars)}/{len(variables)} variables succeeded in {task_time:.1f}s")
                        else:
                            processing_stats['failed_tasks'] += 1
                            logger.error(f"❌ {task_name}: All variables failed")
                            
                        # Collect output files
                        for var, var_result in result['variables'].items():
                            if var_result.get('status') == 'success':
                                for year in var_result.get('target_years_processed', []):
                                    # Construct output file path
                                    period = 'historical' if year < 2015 else scenario
                                    output_file = output_base_path / var / period / f"{var}_{region}_{period}_{year}_30yr_normal.nc"
                                    if output_file.exists():
                                        output_files.append(str(output_file))
                    else:
                        processing_stats['failed_tasks'] += 1
                        logger.error(f"❌ {task_name}: Processing failed - {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    processing_stats['failed_tasks'] += 1
                    logger.error(f"Error processing {task_name}: {e}")
                    all_results[task_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        # Calculate summary statistics
        total_time = time.time() - start_time
        processing_stats['total_processing_time'] = total_time
        
        success_rate = (processing_stats['successful_tasks'] / 
                       max(processing_stats['total_tasks'], 1)) * 100
        
        logger.info(f"🎉 Means processing completed in {total_time/60:.1f} minutes")
        logger.info(f"✅ Success rate: {success_rate:.1f}% ({processing_stats['successful_tasks']}/{processing_stats['total_tasks']} tasks)")
        
        return {
            'status': 'completed' if success_rate > 0 else 'failed',
            'output_files': output_files,
            'processing_stats': processing_stats,
            'results_by_task': all_results,
            'output_base_path': str(output_base_path),
        }
        
    except Exception as e:
        logger.error(f"Fatal error in means processing: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'output_files': [],
            'processing_stats': processing_stats,
        }