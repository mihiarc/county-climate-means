"""
Pipeline-Aware Workflow for Climate Means Processing

Provides workflow functions that are aware of downstream pipeline requirements
and automatically prepare outputs for seamless integration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict

from means.workflow.integrated_processor import IntegratedClimateProcessor, ProcessingWorkflow
from means.pipeline.interface import PipelineInterface
from means.config import get_config
from means.core.regions import REGION_BOUNDS

logger = logging.getLogger(__name__)


class PipelineAwareWorkflow:
    """
    Workflow manager that automatically configures processing based on
    downstream pipeline requirements.
    
    This class provides high-level functions that make it easy to process
    climate means data with optimal settings for specific downstream pipelines.
    """
    
    def __init__(self, output_base_dir: Optional[Path] = None):
        if output_base_dir is None:
            config = get_config()
            output_base_dir = config.paths.output_base_dir
        
        self.output_base_dir = Path(output_base_dir)
        self.pipeline_interface = PipelineInterface(output_base_dir)
        
        # Default workflow configurations for different downstream pipelines
        self.pipeline_workflows = {
            "climate_extremes": ProcessingWorkflow(
                regions=["CONUS", "AK", "HI", "PRVI", "GU"],
                variables=["pr", "tas", "tasmax", "tasmin"],
                enable_catalog=True,
                enable_organization=True,
                enable_metadata_enhancement=True,
                enable_pipeline_bridge=True,
                use_scenario_year_structure=True,
                create_bridge_files=True,
                validate_for_downstream=True,
                max_workers=6,
                cores_per_variable=2,
                batch_size_years=2
            ),
            "climate_metrics": ProcessingWorkflow(
                regions=["CONUS", "AK", "HI", "PRVI", "GU"],
                variables=["pr", "tas", "tasmax", "tasmin"],
                enable_catalog=True,
                enable_organization=True,
                enable_metadata_enhancement=True,
                enable_pipeline_bridge=True,
                use_scenario_year_structure=True,
                create_bridge_files=True,
                validate_for_downstream=True,
                max_workers=8,  # Higher for metrics pipeline
                cores_per_variable=3,
                batch_size_years=1  # Smaller batches for memory efficiency
            ),
            "default": ProcessingWorkflow(
                regions=["CONUS"],
                variables=["pr", "tas"],
                enable_catalog=True,
                enable_organization=True,
                enable_metadata_enhancement=True,
                enable_pipeline_bridge=True,
                use_scenario_year_structure=True,
                create_bridge_files=True,
                validate_for_downstream=True,
                max_workers=6,
                cores_per_variable=2,
                batch_size_years=2
            )
        }
    
    def process_for_pipeline(self, 
                           downstream_pipeline: str,
                           regions: Optional[List[str]] = None,
                           variables: Optional[List[str]] = None,
                           custom_workflow: Optional[ProcessingWorkflow] = None) -> Dict[str, Any]:
        """
        Process climate means data optimized for a specific downstream pipeline.
        
        Args:
            downstream_pipeline: Name of downstream pipeline ("climate_extremes", "climate_metrics", etc.)
            regions: Override default regions
            variables: Override default variables
            custom_workflow: Custom workflow configuration
            
        Returns:
            Dictionary with processing results and pipeline preparation status
        """
        logger.info(f"Processing climate means for {downstream_pipeline} pipeline")
        
        # Get workflow configuration
        if custom_workflow:
            workflow = custom_workflow
        else:
            workflow = self.pipeline_workflows.get(
                downstream_pipeline, 
                self.pipeline_workflows["default"]
            )
        
        # Apply overrides
        if regions:
            workflow.regions = regions
        if variables:
            workflow.variables = variables
        
        # Create integrated processor
        processor = IntegratedClimateProcessor(workflow)
        
        # Process all regions
        results = processor.process_all_regions()
        
        # Add pipeline-specific export
        try:
            export_results = processor.export_for_downstream_pipeline(downstream_pipeline)
            results["pipeline_export"] = export_results
        except Exception as e:
            logger.warning(f"Failed to create pipeline export for {downstream_pipeline}: {e}")
            results["pipeline_export"] = {"status": "failed", "error": str(e)}
        
        # Add validation results
        try:
            validation_results = processor.validate_downstream_readiness()
            results["downstream_validation"] = validation_results
        except Exception as e:
            logger.warning(f"Failed to validate downstream readiness: {e}")
            results["downstream_validation"] = {"ready_for_downstream": False, "error": str(e)}
        
        logger.info(f"Completed processing for {downstream_pipeline} pipeline")
        return results
    
    def process_for_extremes_pipeline(self, 
                                    regions: Optional[List[str]] = None,
                                    variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process climate means data optimized for climate extremes pipeline.
        
        Args:
            regions: Regions to process (default: all regions)
            variables: Variables to process (default: all variables)
            
        Returns:
            Processing results with extremes pipeline configuration
        """
        return self.process_for_pipeline(
            "climate_extremes", 
            regions=regions, 
            variables=variables
        )
    
    def process_for_metrics_pipeline(self,
                                   regions: Optional[List[str]] = None,
                                   variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process climate means data optimized for climate metrics pipeline.
        
        Args:
            regions: Regions to process (default: all regions)
            variables: Variables to process (default: all variables)
            
        Returns:
            Processing results with metrics pipeline configuration
        """
        return self.process_for_pipeline(
            "climate_metrics",
            regions=regions,
            variables=variables
        )
    
    def create_custom_workflow(self,
                             downstream_pipeline: str,
                             regions: List[str],
                             variables: List[str],
                             processing_options: Optional[Dict[str, Any]] = None) -> ProcessingWorkflow:
        """
        Create a custom workflow configuration.
        
        Args:
            downstream_pipeline: Target downstream pipeline
            regions: Regions to process
            variables: Variables to process
            processing_options: Custom processing options
            
        Returns:
            Custom ProcessingWorkflow configuration
        """
        # Start with default workflow for the pipeline
        base_workflow = self.pipeline_workflows.get(
            downstream_pipeline,
            self.pipeline_workflows["default"]
        )
        
        # Create custom workflow
        custom_workflow = ProcessingWorkflow(
            regions=regions,
            variables=variables,
            enable_catalog=True,
            enable_organization=True,
            enable_metadata_enhancement=True,
            enable_pipeline_bridge=True,
            use_scenario_year_structure=True,
            create_bridge_files=True,
            validate_for_downstream=True,
            max_workers=base_workflow.max_workers,
            cores_per_variable=base_workflow.cores_per_variable,
            batch_size_years=base_workflow.batch_size_years
        )
        
        # Apply custom processing options
        if processing_options:
            for key, value in processing_options.items():
                if hasattr(custom_workflow, key):
                    setattr(custom_workflow, key, value)
        
        return custom_workflow
    
    def process_single_region(self,
                            region: str,
                            downstream_pipeline: str = "climate_extremes",
                            variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a single region for downstream pipeline.
        
        Args:
            region: Region code to process
            downstream_pipeline: Target downstream pipeline
            variables: Variables to process (default: all)
            
        Returns:
            Processing results for the single region
        """
        if region not in REGION_BOUNDS:
            raise ValueError(f"Invalid region: {region}")
        
        if variables is None:
            variables = ["pr", "tas", "tasmax", "tasmin"]
        
        return self.process_for_pipeline(
            downstream_pipeline,
            regions=[region],
            variables=variables
        )
    
    def process_subset(self,
                      regions: List[str],
                      variables: List[str],
                      downstream_pipeline: str = "climate_extremes",
                      processing_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a subset of regions and variables.
        
        Args:
            regions: List of region codes
            variables: List of variable names
            downstream_pipeline: Target downstream pipeline
            processing_options: Custom processing options
            
        Returns:
            Processing results for the subset
        """
        # Validate inputs
        invalid_regions = [r for r in regions if r not in REGION_BOUNDS]
        if invalid_regions:
            raise ValueError(f"Invalid regions: {invalid_regions}")
        
        valid_variables = ["pr", "tas", "tasmax", "tasmin"]
        invalid_variables = [v for v in variables if v not in valid_variables]
        if invalid_variables:
            raise ValueError(f"Invalid variables: {invalid_variables}")
        
        # Create custom workflow
        custom_workflow = self.create_custom_workflow(
            downstream_pipeline,
            regions,
            variables,
            processing_options
        )
        
        return self.process_for_pipeline(
            downstream_pipeline,
            custom_workflow=custom_workflow
        )
    
    def get_pipeline_requirements(self, downstream_pipeline: str) -> Dict[str, Any]:
        """
        Get recommended configuration for a downstream pipeline.
        
        Args:
            downstream_pipeline: Name of downstream pipeline
            
        Returns:
            Dictionary with recommended configuration
        """
        workflow = self.pipeline_workflows.get(
            downstream_pipeline,
            self.pipeline_workflows["default"]
        )
        
        return {
            "recommended_regions": workflow.regions,
            "recommended_variables": workflow.variables,
            "processing_settings": {
                "max_workers": workflow.max_workers,
                "cores_per_variable": workflow.cores_per_variable,
                "batch_size_years": workflow.batch_size_years
            },
            "output_features": {
                "scenario_year_structure": workflow.use_scenario_year_structure,
                "catalog_enabled": workflow.enable_catalog,
                "metadata_enhancement": workflow.enable_metadata_enhancement,
                "pipeline_bridge": workflow.enable_pipeline_bridge
            }
        }
    
    def validate_pipeline_readiness(self, downstream_pipeline: str) -> Dict[str, Any]:
        """
        Validate that current outputs are ready for a specific downstream pipeline.
        
        Args:
            downstream_pipeline: Name of downstream pipeline
            
        Returns:
            Validation results
        """
        return self.pipeline_interface.validate_for_downstream()
    
    def create_pipeline_export(self, 
                             downstream_pipeline: str,
                             export_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Create export files for a specific downstream pipeline.
        
        Args:
            downstream_pipeline: Name of downstream pipeline
            export_dir: Directory for export files
            
        Returns:
            Export results
        """
        if downstream_pipeline == "climate_extremes":
            return self.pipeline_interface.export_for_extremes_pipeline(export_dir)
        else:
            processor = IntegratedClimateProcessor(self.pipeline_workflows["default"])
            return processor.export_for_downstream_pipeline(downstream_pipeline, export_dir)


# Convenience functions for common workflows
def process_all_for_extremes(**kwargs) -> Dict[str, Any]:
    """Process all regions and variables for climate extremes pipeline."""
    workflow = PipelineAwareWorkflow()
    return workflow.process_for_extremes_pipeline(**kwargs)


def process_all_for_metrics(**kwargs) -> Dict[str, Any]:
    """Process all regions and variables for climate metrics pipeline."""
    workflow = PipelineAwareWorkflow()
    return workflow.process_for_metrics_pipeline(**kwargs)


def process_region_for_pipeline(region: str, 
                              pipeline: str = "climate_extremes",
                              **kwargs) -> Dict[str, Any]:
    """Process a single region for specified pipeline."""
    workflow = PipelineAwareWorkflow()
    return workflow.process_single_region(region, pipeline, **kwargs)