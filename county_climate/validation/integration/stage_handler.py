"""Stage handler for Phase 3 validation integration."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import json
from datetime import datetime, timezone

from county_climate.validation.validators import (
    QAQCValidator,
    SpatialOutliersValidator,
    PrecipitationValidator
)
from county_climate.validation.visualization import ClimateVisualizer
from county_climate.validation.core import ValidationConfig


async def validation_stage_handler(**context) -> Dict[str, Any]:
    """
    Stage handler for Phase 3 - Validation and QA/QC of climate data.
    
    This stage validates outputs from Phase 1 (means) and Phase 2 (metrics),
    performing comprehensive quality checks and generating reports.
    
    Configuration options:
        metrics_output_path (str): Path to metrics outputs from Phase 2
        output_dir (str): Directory for validation outputs
        validators_to_run (list): List of validators to execute
        validation_config (dict): Validation configuration parameters
        generate_maps (bool): Generate map visualizations
        generate_timeseries (bool): Generate timeseries plots
        generate_distributions (bool): Generate distribution plots
    """
    logger = logging.getLogger(__name__)
    
    # Extract configuration
    stage_config_dict = context['stage_config']
    stage_config = stage_config_dict.get('stage_config', {})
    
    # Initialize validation config
    validation_config = ValidationConfig(**stage_config.get('validation_config', {}))
    
    # Output directory - use organized structure
    from county_climate.shared.config.output_paths import OrganizedOutputPaths
    organized_paths = OrganizedOutputPaths()
    output_dir = Path(stage_config.get('output_dir', str(organized_paths.validation_base)))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("Starting validation stage")
        
        # Get input paths
        metrics_output_path = Path(stage_config.get('metrics_output_path'))
        if not metrics_output_path.exists():
            raise FileNotFoundError(f"Metrics output not found: {metrics_output_path}")
        
        # Find CSV files in metrics output directory
        csv_files = list(metrics_output_path.glob("**/*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {metrics_output_path}")
        
        logger.info(f"Found {len(csv_files)} CSV files to validate")
        
        # Initialize validators
        validators_to_run = stage_config.get('validators_to_run', ['qaqc', 'spatial_outliers', 'precipitation'])
        validators = {}
        
        if 'qaqc' in validators_to_run:
            validators['qaqc'] = QAQCValidator(
                config=validation_config,
                output_dir=output_dir / 'reports' / 'by_validator' / 'qaqc'
            )
        
        if 'spatial_outliers' in validators_to_run:
            validators['spatial_outliers'] = SpatialOutliersValidator(
                config=validation_config,
                output_dir=output_dir / 'reports' / 'by_validator' / 'spatial_outliers'
            )
        
        if 'precipitation' in validators_to_run:
            validators['precipitation'] = PrecipitationValidator(
                config=validation_config,
                output_dir=output_dir / 'reports' / 'by_validator' / 'precipitation'
            )
        
        # Run validation on each file
        all_results = []
        validation_reports = {}
        
        for csv_file in csv_files:
            logger.info(f"Validating {csv_file.name}")
            
            # Load data
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"Loaded {len(df)} records from {csv_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {csv_file}: {e}")
                continue
            
            # Run each validator
            file_results = {}
            for validator_name, validator in validators.items():
                logger.info(f"Running {validator_name} on {csv_file.name}")
                
                try:
                    result = validator.validate(df, dataset_path=str(csv_file))
                    file_results[validator_name] = {
                        'passed': result.passed,
                        'quality_score': result.quality_score,
                        'issue_count': len(result.issues),
                        'metrics': result.metrics
                    }
                    
                    # Save report
                    report_path = validator.save_report(format='json')
                    if validator_name not in validation_reports:
                        validation_reports[validator_name] = []
                    validation_reports[validator_name].append(str(report_path))
                    
                except Exception as e:
                    logger.error(f"Validator {validator_name} failed on {csv_file.name}: {e}")
                    file_results[validator_name] = {
                        'passed': False,
                        'error': str(e)
                    }
            
            all_results.append({
                'file': str(csv_file),
                'results': file_results
            })
        
        # Generate visualizations if requested
        visualization_paths = []
        if stage_config.get('generate_maps', False) or stage_config.get('generate_timeseries', False):
            logger.info("Generating visualizations")
            
            visualizer = ClimateVisualizer(output_dir=output_dir / 'visualizations')
            
            # Try to create visualizations for each data file
            for csv_file in csv_files[:5]:  # Limit to first 5 files
                try:
                    df = pd.read_csv(csv_file)
                    
                    if stage_config.get('generate_maps', False):
                        map_path = visualizer.create_overview_dashboard(
                            df, 
                            title=f"Climate Data Overview - {csv_file.stem}"
                        )
                        if map_path:
                            visualization_paths.append(str(map_path))
                    
                    if stage_config.get('generate_timeseries', False):
                        # Create temperature analysis if available
                        if any(col.startswith('tas') for col in df.columns):
                            temp_path = visualizer.create_temperature_analysis(df)
                            if temp_path:
                                visualization_paths.append(str(temp_path))
                        
                        # Create precipitation analysis if available
                        if 'pr' in df.columns or any(col.startswith('pr') for col in df.columns):
                            precip_path = visualizer.create_precipitation_analysis(df)
                            if precip_path:
                                visualization_paths.append(str(precip_path))
                                
                except Exception as e:
                    logger.error(f"Failed to create visualizations for {csv_file}: {e}")
        
        # Create summary report
        summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'files_validated': len(all_results),
            'validators_run': list(validators.keys()),
            'validation_results': all_results,
            'report_paths': validation_reports,
            'visualization_paths': visualization_paths,
            'overall_passed': all(
                all(r.get('passed', False) for r in result['results'].values())
                for result in all_results
            )
        }
        
        # Save summary report
        summary_path = output_dir / 'reports' / 'summary' / 'validation_summary.json'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Validation complete. Summary saved to {summary_path}")
        
        return {
            'status': 'completed',
            'summary_path': str(summary_path),
            'files_validated': len(all_results),
            'overall_passed': summary['overall_passed'],
            'report_paths': validation_reports,
            'visualization_paths': visualization_paths
        }
        
    except Exception as e:
        logger.error(f"Validation stage failed: {e}")
        raise


# Create instance for registration
validation_stage_handler_instance = validation_stage_handler