"""
Pipeline validation handler for the orchestration system.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


async def validate_complete_pipeline(**context) -> Dict[str, Any]:
    """
    Validate the complete pipeline outputs.
    
    This handler performs validation checks on the outputs from
    previous pipeline stages.
    """
    stage_config = context['stage_config']
    stage_inputs = context.get('stage_inputs', {})
    logger = context['logger']
    
    logger.info("Starting pipeline validation")
    
    validation_checks = stage_config.get('validation_checks', [
        'data_completeness',
        'file_integrity'
    ])
    
    validation_results = {
        'status': 'completed',
        'validation_passed': True,
        'checks_performed': [],
        'issues_found': [],
        'validated_files': 0,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Get output paths from previous stages
    means_output = Path(stage_config.get('means_output_path', '/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/means'))
    metrics_output = Path(stage_config.get('metrics_output_path', '/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/metrics'))
    
    # Check if output directories exist
    if not means_output.exists():
        validation_results['validation_passed'] = False
        validation_results['issues_found'].append(f"Means output directory not found: {means_output}")
        logger.error(f"Means output directory not found: {means_output}")
    
    if not metrics_output.exists():
        validation_results['validation_passed'] = False
        validation_results['issues_found'].append(f"Metrics output directory not found: {metrics_output}")
        logger.error(f"Metrics output directory not found: {metrics_output}")
    
    # Validate means output files
    if 'data_completeness' in validation_checks and means_output.exists():
        logger.info("Checking data completeness for means outputs")
        
        # Count NetCDF files
        nc_files = list(means_output.rglob('*.nc'))
        validation_results['validated_files'] = len(nc_files)
        
        if len(nc_files) == 0:
            validation_results['validation_passed'] = False
            validation_results['issues_found'].append("No NetCDF files found in means output")
        else:
            logger.info(f"Found {len(nc_files)} NetCDF files in means output")
            validation_results['checks_performed'].append('data_completeness')
            
            # Check file sizes
            small_files = [f for f in nc_files if f.stat().st_size < 1024 * 1024]  # Files < 1MB
            if small_files:
                validation_results['issues_found'].append(
                    f"Found {len(small_files)} files smaller than 1MB"
                )
    
    # Validate metrics output files
    if 'file_integrity' in validation_checks and metrics_output.exists():
        logger.info("Checking file integrity for metrics outputs")
        
        # Check for expected output formats
        expected_formats = stage_config.get('expected_formats', ['netcdf4', 'csv', 'parquet'])
        for fmt in expected_formats:
            if fmt == 'netcdf4':
                files = list(metrics_output.rglob('*.nc'))
            elif fmt == 'csv':
                files = list(metrics_output.rglob('*.csv'))
            elif fmt == 'parquet':
                files = list(metrics_output.rglob('*.parquet'))
            else:
                continue
                
            if not files:
                validation_results['issues_found'].append(
                    f"No {fmt} files found in metrics output"
                )
            else:
                logger.info(f"Found {len(files)} {fmt} files in metrics output")
        
        validation_results['checks_performed'].append('file_integrity')
    
    # Create validation report if requested
    if stage_config.get('create_validation_report', True):
        report_path = Path(stage_config.get(
            'validation_report_path',
            '/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/validation'
        ))
        report_path.mkdir(parents=True, exist_ok=True)
        
        report_file = report_path / f"validation_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"Pipeline Validation Report\n")
            f.write(f"========================\n\n")
            f.write(f"Timestamp: {validation_results['timestamp']}\n")
            f.write(f"Validation Passed: {validation_results['validation_passed']}\n")
            f.write(f"Files Validated: {validation_results['validated_files']}\n\n")
            
            f.write(f"Checks Performed:\n")
            for check in validation_results['checks_performed']:
                f.write(f"  - {check}\n")
            
            if validation_results['issues_found']:
                f.write(f"\nIssues Found:\n")
                for issue in validation_results['issues_found']:
                    f.write(f"  - {issue}\n")
            else:
                f.write(f"\nNo issues found.\n")
        
        logger.info(f"Validation report saved to: {report_file}")
        validation_results['report_file'] = str(report_file)
    
    # Log summary
    if validation_results['validation_passed']:
        logger.info(f"✅ Pipeline validation passed. Validated {validation_results['validated_files']} files.")
    else:
        logger.warning(f"⚠️ Pipeline validation found issues: {len(validation_results['issues_found'])} issues")
    
    return validation_results