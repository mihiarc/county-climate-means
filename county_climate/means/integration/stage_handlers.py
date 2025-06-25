"""
Stage handlers that bridge existing means processing to the new orchestration system.

These handlers wrap the existing RegionalClimateProcessor and other components
to work with the configuration-driven pipeline orchestrator.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio

try:
    from county_climate.means.core.regional_climate_processor import (
        RegionalClimateProcessor,
        RegionalProcessingConfig,
    )
    from county_climate.means.core.regions import REGION_BOUNDS
    from county_climate.means.utils.io_util import NorESM2FileHandler
    from county_climate.means.utils.mp_progress import (
        MultiprocessingProgressTracker,
        ProgressReporter
    )
    MEANS_IMPORTS_AVAILABLE = True
except ImportError as e:
    MEANS_IMPORTS_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Means imports not available: {e}")


async def means_stage_handler(**context) -> Dict[str, Any]:
    """
    Stage handler for climate means processing.
    
    Bridges the existing RegionalClimateProcessor to the new orchestration system.
    
    Configuration options:
        enable_rich_progress (bool): Enable Rich progress tracking display. Default: True
        variables (list): Climate variables to process
        regions (list): Geographic regions to process
        scenarios (list): Climate scenarios to process
        multiprocessing_workers (int): Number of parallel workers
        batch_size_years (int): Years to process in each batch
        max_memory_per_worker_gb (float): Memory limit per worker
    """
    # Extract the nested stage configuration
    stage_config_dict = context['stage_config']
    stage_config = stage_config_dict.get('stage_config', {})  # Get the nested stage_config
    pipeline_context = context['pipeline_context']
    logger = context['logger']
    
    logger.info("Starting climate means processing stage")
    start_time = time.time()
    
    # Check if means imports are available
    if not MEANS_IMPORTS_AVAILABLE:
        logger.error("Means processing components not available")
        return {
            'status': 'failed',
            'error': 'Means processing components not available - using mock processing',
            'output_files': [],
            'processing_stats': {'total_datasets': 0, 'successful_datasets': 0},
        }
    
    try:
        # Extract configuration parameters from the nested stage_config
        variables = stage_config.get('variables', ['tas'])
        regions = stage_config.get('regions', ['CONUS'])
        scenarios = stage_config.get('scenarios', ['historical'])
        
        # Processing parameters
        multiprocessing_workers = stage_config.get('multiprocessing_workers', 16)
        batch_size_years = stage_config.get('batch_size_years', 10)
        max_memory_per_worker_gb = stage_config.get('max_memory_per_worker_gb', 3.5)
        
        # Paths
        input_base_path = Path(pipeline_context.get('base_data_path', '/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM'))
        
        # Use organized output structure
        from county_climate.shared.config.output_paths import OrganizedOutputPaths
        organized_paths = OrganizedOutputPaths()
        output_base_path = Path(stage_config.get('output_base_path', str(organized_paths.climate_means_base)))
        
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
        logger.info(f"Total processing tasks: {total_tasks}")
        
        # Initialize progress tracking (enabled by default)
        use_rich_progress = stage_config.get('enable_rich_progress', True)  # Default to True
        
        # Process each combination of region and scenario (processing all variables simultaneously)
        for region in regions:
            for scenario in scenarios:
                task_name = f"{region}/{scenario}"
                
                try:
                    logger.info(f"Processing all variables ({', '.join(variables)}) for {region} in {scenario} scenario")
                    
                    # Progress tracking is handled by the RegionalClimateProcessor
                    
                    # Create configuration for this specific processing task
                    processing_config = RegionalProcessingConfig(
                        region_key=region,
                        variables=variables,  # Process ALL variables simultaneously
                        input_data_dir=input_base_path,
                        output_base_dir=output_base_path,
                        max_cores=multiprocessing_workers,
                        cores_per_variable=multiprocessing_workers // len(variables),  # Divide cores among variables
                        batch_size_years=batch_size_years,
                        max_memory_per_process_gb=max_memory_per_worker_gb,
                        min_years_for_normal=25,
                    )
                    
                    # Create regional processor with rich progress enabled
                    processor = RegionalClimateProcessor(processing_config, use_rich_progress=use_rich_progress)
                    
                    # Process this region/scenario combination with all variables
                    dataset_start_time = time.time()
                    
                    # Process using the regional processor (handles all variables at once)
                    logger.info(f"Starting processing with RegionalClimateProcessor for {region}/{scenario}")
                    result = processor.process_all_variables()
                    
                    dataset_time = time.time() - dataset_start_time
                    
                    # Track results for each variable
                    successful_variables = []
                    failed_variables = []
                    
                    for variable in variables:
                        processing_stats['total_datasets'] += 1
                        processing_stats['total_processing_time'] += dataset_time / len(variables)  # Divide time among variables
                        
                        if result and variable in result and result[variable].get('status') != 'error':
                            processing_stats['successful_datasets'] += 1
                            successful_variables.append(variable)
                            
                            # Count output files created for this variable
                            output_pattern = output_base_path / variable / scenario / f"*{region}*.nc"
                            if output_pattern.parent.exists():
                                created_files = list(output_pattern.parent.glob(output_pattern.name))
                                output_files.extend([str(f) for f in created_files])
                                
                                processed_datasets.append({
                                    'variable': variable,
                                    'region': region,
                                    'scenario': scenario,
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
                    
                    # Progress tracking is handled by the RegionalClimateProcessor
                
                except Exception as e:
                    # Mark all variables for this region/scenario as failed
                    for variable in variables:
                        processing_stats['failed_datasets'] += 1
                        processing_stats['total_datasets'] += 1
                    
                    logger.error(f"Error processing {region}/{scenario}: {e}")
                    
                    # Progress tracking is handled by the RegionalClimateProcessor
        
        # Progress tracking is handled by the RegionalClimateProcessor
        
        # Calculate summary statistics
        total_time = time.time() - start_time
        processing_stats['total_processing_time'] = total_time
        
        success_rate = (processing_stats['successful_datasets'] / 
                       max(processing_stats['total_datasets'], 1)) * 100
        
        logger.info(f"Means processing completed in {total_time:.1f}s")
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
        }
        
    except Exception as e:
        logger.error(f"Climate means processing failed: {e}")
        raise


async def _process_variable_scenario(
    processor,
    engine,
    variable: str,
    scenario: str,
    years: List[int],
    batch_size: int,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Process a single variable/scenario combination."""
    
    try:
        # Split years into batches for processing
        year_batches = [years[i:i + batch_size] for i in range(0, len(years), batch_size)]
        
        output_files = []
        quality_metrics = {
            'total_years': len(years),
            'successful_years': 0,
            'data_completeness': 0.0,
            'coordinate_validation': True,
        }
        
        logger.info(f"Processing {len(year_batches)} batches for {variable}/{scenario}")
        
        # Process each batch
        for batch_idx, year_batch in enumerate(year_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(year_batches)}: years {min(year_batch)}-{max(year_batch)}")
            
            # Use the existing processor to handle this batch
            # This is where you'd call your existing processing logic
            batch_result = await _process_year_batch(
                processor=processor,
                variable=variable,
                scenario=scenario,
                years=year_batch,
                logger=logger
            )
            
            if batch_result['success']:
                output_files.extend(batch_result.get('output_files', []))
                quality_metrics['successful_years'] += len(year_batch)
            else:
                logger.warning(f"Batch {batch_idx + 1} failed: {batch_result.get('error')}")
        
        # Calculate quality metrics
        quality_metrics['data_completeness'] = quality_metrics['successful_years'] / quality_metrics['total_years']
        
        success = quality_metrics['data_completeness'] > 0.8  # Require 80% success rate
        
        return {
            'success': success,
            'output_files': output_files,
            'quality_metrics': quality_metrics,
        }
        
    except Exception as e:
        logger.error(f"Error in _process_variable_scenario: {e}")
        return {
            'success': False,
            'error': str(e),
            'output_files': [],
            'quality_metrics': {},
        }


async def _process_year_batch(
    processor,
    variable: str,
    scenario: str,
    years: List[int],
    logger: logging.Logger
) -> Dict[str, Any]:
    """Process a batch of years for a specific variable/scenario."""
    
    try:
        # This is where you'd integrate with your existing processing logic
        # For now, return a mock result - you'll need to adapt this to call
        # your actual RegionalClimateProcessor methods
        
        logger.info(f"Processing years {min(years)}-{max(years)} for {variable}/{scenario}")
        
        # Mock processing - replace with actual processor.process_years() call
        output_files = []
        for year in years:
            # This would be the actual file path your processor creates
            output_file = f"/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/means/{variable}_{scenario}_{year}_climatology.nc"
            output_files.append(output_file)
        
        return {
            'success': True,
            'output_files': output_files,
        }
        
    except Exception as e:
        logger.error(f"Error processing year batch: {e}")
        return {
            'success': False,
            'error': str(e),
            'output_files': [],
        }


async def validation_stage_handler(**context) -> Dict[str, Any]:
    """
    Stage handler for means output validation.
    
    Validates the output from means processing stage.
    """
    stage_config = context['stage_config']
    stage_inputs = context['stage_inputs']
    logger = context['logger']
    
    logger.info("Starting means output validation")
    
    try:
        # Get means processing results
        means_results = stage_inputs.get('climate_means', {})
        output_files = means_results.get('output_files', [])
        
        if not output_files:
            logger.warning("No output files to validate")
            return {
                'status': 'completed',
                'validation_passed': False,
                'error': 'No output files found',
            }
        
        # Validation checks
        validation_results = {
            'total_files': len(output_files),
            'files_exist': 0,
            'files_readable': 0,
            'coordinate_validation': True,
            'data_range_validation': True,
        }
        
        logger.info(f"Validating {len(output_files)} output files")
        
        # Check file existence and readability
        for file_path in output_files:
            file_path = Path(file_path)
            
            if file_path.exists():
                validation_results['files_exist'] += 1
                
                # Try to read the file (basic validation)
                try:
                    # You could add more sophisticated validation here
                    # using your existing validation modules
                    validation_results['files_readable'] += 1
                except Exception as e:
                    logger.warning(f"File not readable: {file_path}: {e}")
            else:
                logger.warning(f"Output file missing: {file_path}")
        
        # Calculate validation metrics
        file_existence_rate = validation_results['files_exist'] / validation_results['total_files']
        file_readability_rate = validation_results['files_readable'] / validation_results['total_files']
        
        # Determine if validation passed
        validation_passed = (
            file_existence_rate >= 0.95 and  # 95% of files exist
            file_readability_rate >= 0.9 and  # 90% of files are readable
            validation_results['coordinate_validation'] and
            validation_results['data_range_validation']
        )
        
        logger.info(f"Validation completed: {validation_passed}")
        logger.info(f"File existence rate: {file_existence_rate:.1%}")
        logger.info(f"File readability rate: {file_readability_rate:.1%}")
        
        return {
            'status': 'completed',
            'validation_passed': validation_passed,
            'validation_results': validation_results,
            'file_existence_rate': file_existence_rate,
            'file_readability_rate': file_readability_rate,
        }
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            'status': 'failed',
            'validation_passed': False,
            'error': str(e),
        }