"""
Stage handler for Zarr-based climate processing.

This handler integrates the Zarr processing capabilities into the 
orchestrated pipeline framework.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from county_climate.shared.contracts import (
    PipelineStageConfig,
    ProcessingState,
    StageResult,
    StageStatus
)
from county_climate.shared.integration import BaseStageHandler
from county_climate.means.core.zarr_climate_processor import ZarrClimateProcessor
from county_climate.shared.utils.zarr_utils import (
    netcdf_to_zarr, create_multiscale_zarr, validate_zarr_store
)
from county_climate.shared.utils.kerchunk_utils import (
    create_single_file_reference, create_multi_file_reference,
    create_climate_data_catalog
)
from county_climate.shared.utils.dask_utils import create_distributed_client

logger = logging.getLogger(__name__)


class ZarrStageHandler(BaseStageHandler):
    """Handler for Zarr-based climate data processing stages."""
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.processor = None
    
    def validate_config(self, config: PipelineStageConfig) -> bool:
        """Validate Zarr stage configuration."""
        required_fields = {
            'data_preparation': ['input', 'output', 'options'],
            'climate_means_zarr': ['input', 'output', 'processing'],
            'export_validation': ['export', 'validation']
        }
        
        stage_name = config.name
        if stage_name in required_fields:
            for field in required_fields[stage_name]:
                if field not in config.config:
                    logger.error(f"Missing required field '{field}' in {stage_name} config")
                    return False
        
        return True
    
    def setup(self, config: PipelineStageConfig, state: ProcessingState) -> None:
        """Set up Zarr processing environment."""
        # Extract Dask configuration
        dask_config = state.metadata.get('dask', {})
        
        # Create Dask client if needed
        if config.name in ['climate_means_zarr', 'data_preparation']:
            self.client = create_distributed_client(
                n_workers=dask_config.get('n_workers', 4),
                threads_per_worker=dask_config.get('threads_per_worker', 2),
                memory_limit=dask_config.get('memory_limit', '8GB')
            )
            logger.info(f"Dask dashboard available at: {self.client.dashboard_link}")
    
    def execute(self, config: PipelineStageConfig, state: ProcessingState) -> StageResult:
        """Execute Zarr processing stage."""
        try:
            if config.name == 'data_preparation':
                return self._execute_data_preparation(config, state)
            elif config.name == 'climate_means_zarr':
                return self._execute_climate_means(config, state)
            elif config.name == 'export_validation':
                return self._execute_export_validation(config, state)
            else:
                raise ValueError(f"Unknown stage: {config.name}")
                
        except Exception as e:
            logger.error(f"Stage {config.name} failed: {str(e)}")
            return StageResult(
                stage_name=config.name,
                status=StageStatus.FAILED,
                metrics={'error': str(e)}
            )
    
    def cleanup(self, config: PipelineStageConfig, state: ProcessingState) -> None:
        """Clean up resources."""
        if self.client:
            self.client.close()
            self.client = None
    
    def _execute_data_preparation(
        self, config: PipelineStageConfig, state: ProcessingState
    ) -> StageResult:
        """Execute data preparation stage."""
        logger.info("Starting data preparation stage")
        
        cfg = config.config
        input_cfg = cfg['input']
        output_cfg = cfg['output']
        options = cfg['options']
        
        # Ensure output directories exist
        Path(output_cfg['base_path']).mkdir(parents=True, exist_ok=True)
        Path(output_cfg['kerchunk_path']).mkdir(parents=True, exist_ok=True)
        
        catalog = {}
        total_files = 0
        
        for variable in input_cfg['variables']:
            for scenario in input_cfg['scenarios']:
                # Find input files
                pattern = input_cfg['file_pattern'].format(
                    variable=variable,
                    scenario=scenario
                )
                base_path = Path(input_cfg['base_path'])
                files = sorted(base_path.glob(pattern))
                
                if not files:
                    logger.warning(f"No files found for {variable}/{scenario}")
                    continue
                
                logger.info(f"Processing {len(files)} files for {variable}/{scenario}")
                total_files += len(files)
                
                if options['conversion_method'] == 'kerchunk':
                    # Create Kerchunk references
                    ref_path = Path(output_cfg['kerchunk_path']) / f"{variable}_{scenario}.json"
                    
                    if len(files) == 1:
                        create_single_file_reference(
                            files[0],
                            ref_path,
                            inline_threshold=options['kerchunk']['inline_threshold']
                        )
                    else:
                        create_multi_file_reference(
                            files,
                            ref_path,
                            parallel=True,
                            max_workers=options['kerchunk']['parallel_jobs']
                        )
                    
                    catalog[f"{variable}/{scenario}"] = str(ref_path)
                    
                else:  # Direct Zarr conversion
                    zarr_path = Path(output_cfg['base_path']) / variable / f"{scenario}.zarr"
                    
                    # Convert first file or create from multiple
                    if len(files) == 1:
                        netcdf_to_zarr(
                            files[0],
                            zarr_path,
                            chunks=options['zarr']['chunks'],
                            variable=variable,
                            consolidated=options['consolidate_metadata']
                        )
                    else:
                        # TODO: Implement multi-file Zarr creation
                        logger.warning("Multi-file Zarr creation not yet implemented")
                    
                    catalog[f"{variable}/{scenario}"] = str(zarr_path)
        
        # Save catalog
        catalog_path = Path(output_cfg['catalog_path'])
        with open(catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)
        
        # Update state
        state.metadata['zarr_catalog'] = str(catalog_path)
        state.metadata['total_files_processed'] = total_files
        
        return StageResult(
            stage_name=config.name,
            status=StageStatus.COMPLETED,
            metrics={
                'total_files': total_files,
                'catalog_entries': len(catalog),
                'method': options['conversion_method']
            },
            artifacts={'catalog_path': str(catalog_path)}
        )
    
    def _execute_climate_means(
        self, config: PipelineStageConfig, state: ProcessingState
    ) -> StageResult:
        """Execute climate means calculation using Zarr."""
        logger.info("Starting Zarr-based climate means calculation")
        
        cfg = config.config
        processing_cfg = cfg['processing']
        
        # Load catalog
        catalog_path = state.metadata.get('zarr_catalog')
        if not catalog_path:
            raise ValueError("No Zarr catalog found in state")
        
        with open(catalog_path, 'r') as f:
            catalog = json.load(f)
        
        output_base = Path(cfg['output']['base_path'])
        output_base.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Process each variable/region/period combination
        for variable in ['pr', 'tas', 'tasmax', 'tasmin']:
            for region in processing_cfg['regions']:
                for period in processing_cfg['periods']:
                    # Skip if no data available
                    scenario_key = f"{variable}/{period['scenarios'][0]}"
                    if scenario_key not in catalog:
                        logger.warning(f"No data for {scenario_key}")
                        continue
                    
                    # Create processor
                    processor = ZarrClimateProcessor(
                        variable=variable,
                        scenario=period['scenarios'][0],
                        region=region,
                        use_kerchunk=cfg['input'].get('use_kerchunk', True),
                        dask_client=self.client,
                        chunk_strategy=processing_cfg.get('chunk_strategy', 'auto')
                    )
                    
                    # Process period
                    input_path = catalog[scenario_key]
                    output_path = output_base / variable / region / f"{period['name']}.zarr"
                    
                    try:
                        result_ds = processor.calculate_rolling_normals_zarr(
                            input_path,
                            output_path,
                            period['start_year'],
                            period['end_year'],
                            window_size=processing_cfg['window_size'],
                            min_years=processing_cfg['min_years']
                        )
                        
                        # Create multiscale if requested
                        if cfg['output'].get('multiscale', False):
                            processor.create_multiscale_normals(
                                output_path,
                                output_path.parent / 'multiscale',
                                scales=cfg['output'].get('scales', [1, 2, 4, 8])
                            )
                        
                        results.append({
                            'variable': variable,
                            'region': region,
                            'period': period['name'],
                            'output': str(output_path)
                        })
                        
                    except Exception as e:
                        logger.error(f"Failed to process {variable}/{region}/{period['name']}: {e}")
        
        return StageResult(
            stage_name=config.name,
            status=StageStatus.COMPLETED,
            metrics={
                'processed_combinations': len(results),
                'output_format': cfg['output']['format']
            },
            artifacts={'results': results}
        )
    
    def _execute_export_validation(
        self, config: PipelineStageConfig, state: ProcessingState
    ) -> StageResult:
        """Execute export and validation stage."""
        logger.info("Starting export and validation")
        
        cfg = config.config
        
        # Validate Zarr stores
        validation_results = []
        
        if 'results' in state.artifacts:
            for result in state.artifacts['results']:
                output_path = result['output']
                if Path(output_path).exists():
                    try:
                        info = validate_zarr_store(output_path)
                        validation_results.append({
                            'path': output_path,
                            'valid': True,
                            'info': info
                        })
                    except Exception as e:
                        validation_results.append({
                            'path': output_path,
                            'valid': False,
                            'error': str(e)
                        })
        
        # Generate validation report if requested
        if cfg['validation'].get('generate_report', False):
            report_path = Path(cfg['validation']['report_path'])
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Simple JSON report for now
            with open(report_path, 'w') as f:
                json.dump({
                    'validation_results': validation_results,
                    'summary': {
                        'total_stores': len(validation_results),
                        'valid_stores': sum(1 for r in validation_results if r['valid'])
                    }
                }, f, indent=2)
        
        return StageResult(
            stage_name=config.name,
            status=StageStatus.COMPLETED,
            metrics={
                'stores_validated': len(validation_results),
                'validation_passed': sum(1 for r in validation_results if r['valid'])
            },
            artifacts={'validation_results': validation_results}
        )