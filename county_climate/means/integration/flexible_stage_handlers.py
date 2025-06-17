"""
Flexible stage handlers that support multiple scenarios and climate models.

These handlers use the refactored flexible processing system.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio

try:
    # Try flexible imports first
    from county_climate.means.core.regional_climate_processor_v3 import (
        FlexibleRegionalClimateProcessor,
        create_flexible_regional_processor
    )
    from county_climate.means.core.flexible_config import FlexibleRegionalProcessingConfig
    FLEXIBLE_IMPORTS_AVAILABLE = True
except ImportError:
    FLEXIBLE_IMPORTS_AVAILABLE = False
    # Fall back to original imports
    try:
        from county_climate.means.core.regional_climate_processor import (
            RegionalClimateProcessor,
            RegionalProcessingConfig,
        )
        from county_climate.means.core.regions import REGION_BOUNDS
        from county_climate.means.utils.io_util import NorESM2FileHandler
        MEANS_IMPORTS_AVAILABLE = True
    except ImportError as e:
        MEANS_IMPORTS_AVAILABLE = False
        import logging
        logging.getLogger(__name__).warning(f"Means imports not available: {e}")


async def flexible_means_stage_handler(**context) -> Dict[str, Any]:
    """
    Flexible stage handler for climate means processing.
    
    Supports multiple climate models and configurable scenarios.
    
    Configuration options:
        enable_rich_progress (bool): Enable Rich progress tracking display. Default: True
        variables (list): Climate variables to process
        regions (list): Geographic regions to process
        scenarios (list): Climate scenarios to process (e.g., ['ssp585', 'ssp245'])
        model_id (str): Climate model ID (default: "NorESM2-LM")
        multiprocessing_workers (int): Number of parallel workers
        batch_size_years (int): Years to process in each batch
        max_memory_per_worker_gb (float): Memory limit per worker
    """
    # Extract configuration
    stage_config_dict = context['stage_config']
    stage_config = stage_config_dict.get('stage_config', {})
    pipeline_context = context['pipeline_context']
    logger = context['logger']
    
    logger.info("Starting flexible climate means processing stage")
    start_time = time.time()
    
    # Check if flexible imports are available
    if not FLEXIBLE_IMPORTS_AVAILABLE:
        logger.warning("Flexible processing not available, falling back to original handler")
        # Import and use original handler
        from county_climate.means.integration.stage_handlers import means_stage_handler
        return await means_stage_handler(**context)
    
    try:
        # Extract configuration parameters
        variables = stage_config.get('variables', ['tas'])
        regions = stage_config.get('regions', ['CONUS'])
        scenarios = stage_config.get('scenarios', ['historical'])
        model_id = stage_config.get('model_id', 'NorESM2-LM')
        
        # Processing parameters
        multiprocessing_workers = stage_config.get('multiprocessing_workers', 16)
        batch_size_years = stage_config.get('batch_size_years', 10)
        max_memory_per_worker_gb = stage_config.get('max_memory_per_worker_gb', 3.5)
        enable_rich_progress = stage_config.get('enable_rich_progress', True)
        
        # Paths
        input_base_path = Path(pipeline_context.get('base_data_path', '/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM'))
        output_base_path = Path(stage_config.get('output_base_path', '/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/means'))
        
        # Create output directory
        output_base_path.mkdir(parents=True, exist_ok=True)
        
        # Results tracking
        processed_datasets = []
        output_files = []
        processing_stats = {
            'total_datasets': 0,
            'successful_datasets': 0,
            'failed_datasets': 0,
            'total_processing_time': 0,
        }
        
        # Calculate total processing tasks
        total_tasks = len(variables) * len(regions) * len(scenarios)
        logger.info(f"Processing {len(variables)} variables for {len(regions)} regions across {len(scenarios)} scenarios")
        logger.info(f"Using climate model: {model_id}")
        logger.info(f"Total processing tasks: {total_tasks}")
        
        # Process each combination of region and scenario
        for region in regions:
            for scenario in scenarios:
                task_name = f"{region}/{scenario}"
                
                try:
                    logger.info(f"Processing all variables ({', '.join(variables)}) for {region} in {scenario} scenario")
                    
                    # Create flexible processor for this specific scenario
                    processor = create_flexible_regional_processor(
                        region_key=region,
                        variables=variables,
                        scenario=scenario,
                        model_id=model_id,
                        max_cores=multiprocessing_workers,
                        cores_per_variable=multiprocessing_workers // len(variables),
                        batch_size_years=batch_size_years,
                        max_memory_per_process_gb=max_memory_per_worker_gb,
                        enable_rich_progress=enable_rich_progress
                    )
                    
                    # Process this region/scenario combination
                    dataset_start_time = time.time()
                    
                    logger.info(f"Starting flexible processing for {region}/{scenario}")
                    result = processor.process_all_variables()
                    
                    dataset_time = time.time() - dataset_start_time
                    
                    # Track results for each variable
                    successful_variables = []
                    failed_variables = []
                    
                    for variable in variables:
                        processing_stats['total_datasets'] += 1
                        processing_stats['total_processing_time'] += dataset_time / len(variables)
                        
                        if result and variable in result and result[variable].get('status') != 'error':
                            processing_stats['successful_datasets'] += 1
                            successful_variables.append(variable)
                            
                            # Count output files created
                            # Check all possible output directories based on scenario processing
                            output_patterns = [
                                output_base_path / variable / "historical" / f"*{region}*.nc",
                                output_base_path / variable / "hybrid" / f"*{region}*.nc",
                                output_base_path / variable / scenario / f"*{region}*.nc",
                            ]
                            
                            created_files = []
                            for pattern in output_patterns:
                                if pattern.parent.exists():
                                    created_files.extend(list(pattern.parent.glob(pattern.name)))
                            
                            output_files.extend([str(f) for f in created_files])
                            
                            processed_datasets.append({
                                'variable': variable,
                                'region': region,
                                'scenario': scenario,
                                'model': model_id,
                                'processing_time_seconds': dataset_time,
                                'output_files': [str(f) for f in created_files],
                            })
                        else:
                            processing_stats['failed_datasets'] += 1
                            failed_variables.append(variable)
                    
                    # Log results
                    if successful_variables:
                        logger.info(f"Successfully processed {len(successful_variables)}/{len(variables)} variables for {region}/{scenario} in {dataset_time:.1f}s")
                        logger.info(f"Successful: {', '.join(successful_variables)}")
                    if failed_variables:
                        logger.error(f"Failed variables for {region}/{scenario}: {', '.join(failed_variables)}")
                
                except Exception as e:
                    # Mark all variables for this region/scenario as failed
                    for variable in variables:
                        processing_stats['failed_datasets'] += 1
                        processing_stats['total_datasets'] += 1
                    
                    logger.error(f"Error processing {region}/{scenario}: {e}")
        
        # Calculate summary statistics
        total_time = time.time() - start_time
        processing_stats['total_processing_time'] = total_time
        
        success_rate = (processing_stats['successful_datasets'] / 
                       max(processing_stats['total_datasets'], 1)) * 100
        
        logger.info(f"Flexible means processing completed in {total_time:.1f}s")
        logger.info(f"Success rate: {success_rate:.1f}% ({processing_stats['successful_datasets']}/{processing_stats['total_datasets']})")
        
        # Return results for downstream stages
        return {
            'status': 'completed',
            'output_files': output_files,
            'processed_datasets': processed_datasets,
            'processing_stats': processing_stats,
            'output_base_path': str(output_base_path),
            'total_processing_time_seconds': total_time,
            'success_rate': success_rate,
            'model_id': model_id,
        }
        
    except Exception as e:
        logger.error(f"Flexible climate means processing failed: {e}")
        raise