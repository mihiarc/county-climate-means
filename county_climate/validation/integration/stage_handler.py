"""Stage handler for Phase 3 validation integration."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from county_climate.shared.contracts.pipeline_interface import (
    PipelineStageInterface,
    StageStatus,
    StageResult
)
from county_climate.shared.contracts.validation_contracts import (
    ValidationStageInputContract,
    ValidationStageOutputContract,
    ValidationResultContract
)
from county_climate.validation.validators import (
    QAQCValidator,
    SpatialOutliersValidator,
    PrecipitationValidator
)
from county_climate.validation.visualization import ClimateVisualizer
from county_climate.validation.core import ValidationConfig


class ValidationStageHandler(PipelineStageInterface):
    """
    Handler for Phase 3 - Validation and QA/QC of climate data.
    
    This stage validates outputs from Phase 1 (means) and Phase 2 (metrics),
    performing comprehensive quality checks and generating reports.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize validation stage handler."""
        super().__init__(stage_name="validation", config=config or {})
        self.logger = logging.getLogger(__name__)
        
        # Initialize validation config
        self.validation_config = ValidationConfig(**self.config.get('validation_config', {}))
        
        # Output directory
        self.output_dir = Path(self.config.get('output_dir', 'validation_outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate stage inputs."""
        try:
            # Parse input contract
            input_contract = ValidationStageInputContract(**inputs)
            
            # Check if metrics file exists
            metrics_path = Path(input_contract.metrics_output_path)
            if not metrics_path.exists():
                self.logger.error(f"Metrics file not found: {metrics_path}")
                return False
                
            # Check shapefile if provided
            if input_contract.county_shapefile_path:
                shapefile_path = Path(input_contract.county_shapefile_path)
                if not shapefile_path.exists():
                    self.logger.warning(f"Shapefile not found: {shapefile_path}")
                    # Not a fatal error - can use modern format
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
            
    async def execute(self, inputs: Dict[str, Any]) -> StageResult:
        """Execute validation stage."""
        try:
            self.update_status(StageStatus.RUNNING)
            
            # Parse inputs
            input_contract = ValidationStageInputContract(**inputs)
            
            # Load climate data
            self.logger.info(f"Loading climate data from {input_contract.metrics_output_path}")
            climate_data = pd.read_csv(input_contract.metrics_output_path)
            self.logger.info(f"Loaded {len(climate_data)} records")
            
            # Initialize validators
            validators = self._initialize_validators(input_contract)
            
            # Run validations
            validation_results = {}
            
            for validator_name, validator in validators.items():
                if validator_name in input_contract.validators_to_run:
                    self.logger.info(f"Running {validator_name} validator...")
                    
                    result = validator.validate(
                        climate_data,
                        dataset_path=input_contract.metrics_output_path
                    )
                    
                    # Convert to contract
                    result_contract = ValidationResultContract(
                        validator_name=result.validator_name,
                        dataset_path=result.dataset_path,
                        start_time=result.start_time,
                        end_time=result.end_time,
                        issues=[
                            {
                                "severity": issue.severity,
                                "category": issue.category,
                                "message": issue.message,
                                "details": issue.details,
                                "timestamp": issue.timestamp
                            }
                            for issue in result.issues
                        ],
                        metrics=result.metrics,
                        quality_score=result.quality_score,
                        passed=result.passed
                    )
                    
                    validation_results[validator_name] = result_contract
                    
                    # Save individual report
                    report_path = validator.save_report(format="json")
                    self.logger.info(f"Saved {validator_name} report to {report_path}")
            
            # Generate visualizations if requested
            visualization_paths = {}
            if input_contract.run_visualizations:
                self.logger.info("Generating visualizations...")
                visualizer = ClimateVisualizer(output_dir=self.output_dir / "visualizations")
                
                # Create overview dashboard
                overview_path = visualizer.create_overview_dashboard(
                    climate_data,
                    title="Climate Data Validation Overview"
                )
                visualization_paths['overview'] = str(overview_path)
                
                # Create analysis plots
                temp_path = visualizer.create_temperature_analysis(climate_data)
                if temp_path:
                    visualization_paths['temperature'] = str(temp_path)
                    
                precip_path = visualizer.create_precipitation_analysis(climate_data)
                if precip_path:
                    visualization_paths['precipitation'] = str(precip_path)
                    
                # Create validation summary
                validation_summary = self._create_validation_summary(validation_results)
                summary_path = visualizer.create_validation_summary_plots(validation_summary)
                visualization_paths['validation_summary'] = str(summary_path)
            
            # Calculate overall quality
            overall_quality = self._calculate_overall_quality(validation_results)
            overall_passed = all(r.passed for r in validation_results.values())
            
            # Create output contract
            output = ValidationStageOutputContract(
                validation_results={
                    name: result.model_dump()
                    for name, result in validation_results.items()
                },
                overall_quality_score=overall_quality,
                passed=overall_passed,
                report_paths={
                    name: str(self.output_dir / f"{name}_report.json")
                    for name in validation_results.keys()
                },
                visualization_paths=visualization_paths if visualization_paths else None,
                summary=self._create_summary(validation_results, climate_data)
            )
            
            # Save final report
            self._save_final_report(output)
            
            self.update_status(StageStatus.COMPLETED)
            
            return StageResult(
                status=StageStatus.COMPLETED,
                outputs=output.model_dump(),
                metrics={
                    "records_validated": len(climate_data),
                    "validators_run": len(validation_results),
                    "overall_quality": overall_quality,
                    "passed": overall_passed
                }
            )
            
        except Exception as e:
            self.logger.error(f"Validation stage failed: {e}")
            self.update_status(StageStatus.FAILED, str(e))
            
            return StageResult(
                status=StageStatus.FAILED,
                outputs={},
                error=str(e)
            )
            
    def get_resource_requirements(self) -> Dict[str, Any]:
        """Get resource requirements for validation stage."""
        return {
            "memory_gb": 8,
            "cpu_cores": 2,
            "estimated_runtime_minutes": 15,
            "disk_space_gb": 2
        }
        
    def _initialize_validators(self, input_contract: ValidationStageInputContract) -> Dict[str, Any]:
        """Initialize validation modules."""
        validators = {}
        
        # QA/QC Validator
        if "qaqc" in input_contract.validators_to_run:
            validators["qaqc"] = QAQCValidator(
                config=self.validation_config,
                output_dir=self.output_dir / "qaqc"
            )
            
        # Spatial Outliers Validator
        if "spatial" in input_contract.validators_to_run:
            validators["spatial"] = SpatialOutliersValidator(
                config=self.validation_config,
                shapefile_path=input_contract.county_shapefile_path,
                output_dir=self.output_dir / "spatial"
            )
            
        # Precipitation Validator
        if "precipitation" in input_contract.validators_to_run:
            validators["precipitation"] = PrecipitationValidator(
                config=self.validation_config,
                output_dir=self.output_dir / "precipitation"
            )
            
        return validators
        
    def _calculate_overall_quality(self, validation_results: Dict[str, ValidationResultContract]) -> str:
        """Calculate overall quality score based on all validations."""
        if not validation_results:
            return "UNKNOWN"
            
        # Collect all quality scores
        scores = []
        score_map = {"EXCELLENT": 4, "GOOD": 3, "FAIR": 2, "POOR": 1}
        
        for result in validation_results.values():
            if result.quality_score and result.quality_score in score_map:
                scores.append(score_map[result.quality_score])
                
        if not scores:
            return "UNKNOWN"
            
        # Calculate average
        avg_score = sum(scores) / len(scores)
        
        # Map back to quality level
        if avg_score >= 3.5:
            return "EXCELLENT"
        elif avg_score >= 2.5:
            return "GOOD"
        elif avg_score >= 1.5:
            return "FAIR"
        else:
            return "POOR"
            
    def _create_validation_summary(self, validation_results: Dict[str, ValidationResultContract]) -> Dict[str, Any]:
        """Create summary data for visualization."""
        summary = {
            "issue_counts": {},
            "issues_by_category": {},
            "quality_metrics": {},
            "completeness_summary": {}
        }
        
        # Aggregate issue counts
        total_issues = {"critical": 0, "warning": 0, "info": 0}
        category_counts = {}
        
        for name, result in validation_results.items():
            issue_summary = result.issue_summary
            
            for severity, count in issue_summary.items():
                total_issues[severity] += count
                
            for issue in result.issues:
                category = issue["category"]
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1
                
        summary["issue_counts"] = total_issues
        summary["issues_by_category"] = category_counts
        
        # Quality metrics
        passed_count = sum(1 for r in validation_results.values() if r.passed)
        total_count = len(validation_results)
        
        summary["quality_metrics"] = {
            "overall_score": (passed_count / total_count) * 100 if total_count > 0 else 0
        }
        
        # Completeness from QA/QC if available
        if "qaqc" in validation_results:
            qaqc_metrics = validation_results["qaqc"].metrics
            if "completeness" in qaqc_metrics:
                completeness = qaqc_metrics["completeness"]
                
                # Calculate completeness percentages
                if "missing_percentages" in completeness:
                    missing_pcts = completeness["missing_percentages"]
                    avg_completeness = 100 - (sum(missing_pcts.values()) / len(missing_pcts) if missing_pcts else 0)
                else:
                    avg_completeness = 100
                    
                summary["completeness_summary"] = {
                    "data": avg_completeness,
                    "temporal": 95.0,  # Example
                    "spatial": 98.0    # Example
                }
                
        return summary
        
    def _create_summary(self, validation_results: Dict[str, ValidationResultContract], 
                       climate_data: pd.DataFrame) -> Dict[str, Any]:
        """Create overall summary of validation results."""
        summary = {
            "total_records": len(climate_data),
            "date_range": f"{climate_data['year'].min()}-{climate_data['year'].max()}",
            "scenarios": list(climate_data['scenario'].unique()),
            "counties": climate_data['GEOID'].nunique(),
            "validators_run": list(validation_results.keys()),
            "issues_found": {}
        }
        
        # Aggregate issues
        for name, result in validation_results.items():
            issue_counts = result.issue_summary
            summary["issues_found"][name] = {
                "total": sum(issue_counts.values()),
                "by_severity": issue_counts,
                "passed": result.passed
            }
            
        return summary
        
    def _save_final_report(self, output: ValidationStageOutputContract):
        """Save final validation report."""
        report_path = self.output_dir / "validation_report.json"
        
        with open(report_path, 'w') as f:
            f.write(output.model_dump_json(indent=2))
            
        self.logger.info(f"Saved final validation report to {report_path}")


# Create a function-based wrapper for pipeline integration
def validation_stage_handler(**context):
    """Function-based wrapper for ValidationStageHandler class.
    
    This wrapper allows the class-based handler to work with the
    pipeline orchestration system.
    """
    # Extract the nested stage configuration
    stage_config_dict = context['stage_config']
    stage_config = stage_config_dict.get('stage_config', {})  # Get the nested stage_config
    stage_inputs = context.get('stage_inputs', {})
    pipeline_context = context['pipeline_context']
    logger = context['logger']
    
    # Create handler instance with config
    handler = ValidationStageHandler(config=stage_config)
    
    # Run validation synchronously since it's already async internally
    import asyncio
    
    async def run_validation():
        # Prepare inputs - need to get metrics path from previous stage
        validation_inputs = {
            'metrics_output_path': stage_inputs.get('county_metrics_all', {}).get(
                'output_files', 
                [stage_config.get('metrics_output_path', '/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/metrics/county_climate_metrics_complete.csv')]
            )[0] if stage_inputs.get('county_metrics_all', {}).get('output_files') else stage_config.get('metrics_output_path', '/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/metrics/county_climate_metrics_complete.csv'),
            'shapefile_path': stage_config.get('shapefile_path'),
            'validators_to_run': stage_config.get('validators_to_run', ['qaqc', 'spatial', 'precipitation']),
            'run_visualizations': stage_config.get('run_visualizations', True)
        }
        
        # Validate inputs
        inputs_valid = await handler.validate_inputs(validation_inputs)
        if not inputs_valid:
            return {
                'status': 'failed',
                'error': 'Invalid inputs for validation stage'
            }
        
        # Execute validation
        result = await handler.execute(validation_inputs)
        return result
    
    # Run the async function
    return asyncio.run(run_validation())