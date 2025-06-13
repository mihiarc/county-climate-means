"""
Integrated Climate Processing Workflow

Provides a high-level workflow that integrates climate means processing
with the new catalog system, output organization, and pipeline bridge features.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

from county_climate.means.core.regional_climate_processor import RegionalClimateProcessor, RegionalProcessingConfig
from county_climate.means.output.catalog import CatalogManager
from county_climate.means.output.organization import OutputOrganizer
from county_climate.means.output.metadata import DownstreamMetadataEnhancer
from county_climate.means.pipeline.interface import PipelineInterface
from county_climate.means.pipeline.bridge import PipelineBridge, BridgeConfiguration
from county_climate.means.config import get_config
from county_climate.means.core.regions import REGION_BOUNDS

logger = logging.getLogger(__name__)


@dataclass
class ProcessingWorkflow:
    """Configuration for integrated processing workflow."""
    
    # Processing configuration
    regions: List[str]
    variables: List[str]
    enable_catalog: bool = True
    enable_organization: bool = True
    enable_metadata_enhancement: bool = True
    enable_pipeline_bridge: bool = True
    
    # Output configuration
    use_scenario_year_structure: bool = True
    create_bridge_files: bool = True
    validate_for_downstream: bool = True
    
    # Processing options
    max_workers: int = 6
    cores_per_variable: int = 2
    batch_size_years: int = 2


class IntegratedClimateProcessor:
    """
    Integrated processor that combines climate means calculation with
    downstream pipeline preparation.
    
    Provides a unified workflow that:
    1. Processes climate means using regional processor
    2. Organizes outputs in scenario/year structure
    3. Enhances metadata for downstream consumption
    4. Creates catalog and pipeline bridge files
    5. Validates outputs for downstream pipelines
    """
    
    def __init__(self, workflow_config: ProcessingWorkflow):
        self.workflow = workflow_config
        self.config = get_config()
        
        # Initialize components
        self.catalog_manager = CatalogManager(self.config.paths.output_base_dir)
        self.output_organizer = OutputOrganizer(self.config.paths.output_base_dir)
        self.metadata_enhancer = DownstreamMetadataEnhancer()
        self.pipeline_interface = PipelineInterface(self.config.paths.output_base_dir)
        
        # Initialize output structure if enabled
        if self.workflow.use_scenario_year_structure:
            self._initialize_output_structure()
    
    def _initialize_output_structure(self):
        """Initialize the downstream-oriented output structure."""
        scenarios = ["historical", "ssp245"]  # Default scenarios
        if "ssp585" in self.config.processing.scenario_year_ranges:
            scenarios.append("ssp585")
        
        self.output_organizer.initialize_structure(scenarios)
        logger.info("Initialized scenario/year output structure")
    
    def process_all_regions(self) -> Dict[str, Any]:
        """
        Process all regions with integrated workflow.
        
        Returns:
            Dictionary with processing results and pipeline preparation status
        """
        logger.info(f"Starting integrated processing for {len(self.workflow.regions)} regions")
        
        overall_start = time.time()
        results = {
            "processing_results": {},
            "organization_results": {},
            "catalog_results": {},
            "pipeline_bridge_results": {},
            "workflow_summary": {},
            "start_time": datetime.now().isoformat()
        }
        
        # Step 1: Process all regions
        logger.info("Step 1: Processing climate means for all regions")
        for region in self.workflow.regions:
            region_results = self._process_region(region)
            results["processing_results"][region] = region_results
        
        # Step 2: Organize outputs if enabled
        if self.workflow.enable_organization:
            logger.info("Step 2: Organizing outputs for downstream consumption")
            org_results = self._organize_outputs()
            results["organization_results"] = org_results
        
        # Step 3: Create/update catalog if enabled
        if self.workflow.enable_catalog:
            logger.info("Step 3: Creating/updating data catalog")
            catalog_results = self._create_catalog()
            results["catalog_results"] = catalog_results
        
        # Step 4: Create pipeline bridge if enabled
        if self.workflow.enable_pipeline_bridge:
            logger.info("Step 4: Creating pipeline bridge for downstream integration")
            bridge_results = self._create_pipeline_bridge()
            results["pipeline_bridge_results"] = bridge_results
        
        # Step 5: Generate workflow summary
        overall_duration = time.time() - overall_start
        results["workflow_summary"] = self._generate_workflow_summary(
            results, overall_duration
        )
        results["end_time"] = datetime.now().isoformat()
        
        logger.info(f"Integrated workflow completed in {overall_duration:.1f} seconds")
        return results
    
    def _process_region(self, region: str) -> Dict[str, Any]:
        """Process a single region using the regional processor."""
        logger.info(f"Processing region: {region}")
        
        try:
            # Create regional processor configuration
            region_config = RegionalProcessingConfig(
                region_key=region,
                variables=self.workflow.variables,
                input_data_dir=self.config.paths.input_data_dir,
                output_base_dir=self.config.paths.output_base_dir,
                max_cores=self.workflow.max_workers,
                cores_per_variable=self.workflow.cores_per_variable,
                batch_size_years=self.workflow.batch_size_years
            )
            
            # Create and run processor
            processor = RegionalClimateProcessor(region_config, use_rich_progress=True)
            region_results = processor.process_all_variables()
            
            logger.info(f"Completed processing for region: {region}")
            return {
                "status": "completed",
                "variables_processed": list(region_results.keys()),
                "results": region_results
            }
            
        except Exception as e:
            logger.error(f"Failed to process region {region}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "variables_processed": []
            }
    
    def _organize_outputs(self) -> Dict[str, Any]:
        """Organize outputs into downstream-friendly structure."""
        try:
            # Find output files from regional processing
            output_base = self.config.paths.output_base_dir
            
            # Look for files in regional output directories
            organization_stats = {"total_processed": 0, "total_failed": 0}
            
            for region in self.workflow.regions:
                regional_dir = output_base / "data" / region
                if regional_dir.exists():
                    stats = self.output_organizer.organize_directory(
                        regional_dir, copy_files=True
                    )
                    organization_stats["total_processed"] += stats["processed"]
                    organization_stats["total_failed"] += stats["failed"]
            
            # Validate organization
            validation_results = self.output_organizer.validate_organization()
            
            return {
                "status": "completed",
                "organization_stats": organization_stats,
                "validation_results": validation_results
            }
            
        except Exception as e:
            logger.error(f"Failed to organize outputs: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _create_catalog(self) -> Dict[str, Any]:
        """Create or update the data catalog."""
        try:
            # Update catalog from organized output directory
            catalog = self.catalog_manager.update_catalog()
            
            # Get catalog statistics
            stats = catalog.get_summary_stats()
            
            return {
                "status": "completed",
                "catalog_file": str(self.catalog_manager.catalog_file),
                "total_datasets": stats["total_datasets"],
                "valid_datasets": stats["valid_datasets"],
                "scenarios": stats["scenarios"],
                "variables": stats["variables"],
                "regions": stats["regions"]
            }
            
        except Exception as e:
            logger.error(f"Failed to create catalog: {e}")
            return {
                "status": "failed", 
                "error": str(e)
            }
    
    def _create_pipeline_bridge(self) -> Dict[str, Any]:
        """Create pipeline bridge for downstream integration."""
        try:
            # Configure bridge
            bridge_config = BridgeConfiguration(
                target_pipelines=self.config.pipeline_integration.supported_pipelines,
                create_ready_signals=self.workflow.create_bridge_files,
                validate_outputs=self.workflow.validate_for_downstream
            )
            
            # Create bridge
            bridge = PipelineBridge(
                self.config.paths.output_base_dir,
                bridge_config
            )
            
            # Create full bridge setup
            bridge_results = bridge.create_full_bridge()
            
            return bridge_results
            
        except Exception as e:
            logger.error(f"Failed to create pipeline bridge: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _generate_workflow_summary(self, 
                                 results: Dict[str, Any], 
                                 duration: float) -> Dict[str, Any]:
        """Generate comprehensive workflow summary."""
        # Count successful/failed regions
        successful_regions = []
        failed_regions = []
        
        for region, region_result in results["processing_results"].items():
            if region_result.get("status") == "completed":
                successful_regions.append(region)
            else:
                failed_regions.append(region)
        
        # Catalog statistics
        catalog_stats = results.get("catalog_results", {})
        
        # Organization statistics
        org_stats = results.get("organization_results", {}).get("organization_stats", {})
        
        # Pipeline bridge status
        bridge_status = results.get("pipeline_bridge_results", {}).get("status", "unknown")
        
        summary = {
            "workflow_duration_seconds": duration,
            "regions_requested": len(self.workflow.regions),
            "regions_successful": len(successful_regions),
            "regions_failed": len(failed_regions),
            "successful_regions": successful_regions,
            "failed_regions": failed_regions,
            "variables_processed": self.workflow.variables,
            "total_datasets_cataloged": catalog_stats.get("total_datasets", 0),
            "valid_datasets": catalog_stats.get("valid_datasets", 0),
            "files_organized": org_stats.get("total_processed", 0),
            "organization_failures": org_stats.get("total_failed", 0),
            "pipeline_bridge_status": bridge_status,
            "downstream_ready": bridge_status == "success",
            "workflow_features": {
                "catalog_enabled": self.workflow.enable_catalog,
                "organization_enabled": self.workflow.enable_organization,
                "metadata_enhancement_enabled": self.workflow.enable_metadata_enhancement,
                "pipeline_bridge_enabled": self.workflow.enable_pipeline_bridge,
                "scenario_year_structure": self.workflow.use_scenario_year_structure
            }
        }
        
        return summary
    
    def validate_downstream_readiness(self) -> Dict[str, Any]:
        """Validate that outputs are ready for downstream processing."""
        return self.pipeline_interface.validate_for_downstream()
    
    def export_for_downstream_pipeline(self, 
                                     pipeline_name: str,
                                     export_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Export configuration and manifest for specific downstream pipeline."""
        if pipeline_name == "climate_extremes":
            return self.pipeline_interface.export_for_extremes_pipeline(export_dir)
        else:
            # Generic export
            if export_dir is None:
                export_dir = self.config.paths.output_base_dir / "exports" / pipeline_name
            
            export_dir = Path(export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Create downstream configuration
            downstream_config = self.pipeline_interface.create_downstream_config(pipeline_name)
            config_file = export_dir / f"{pipeline_name}_config.yaml"
            downstream_config.save(config_file)
            
            return {
                "export_directory": str(export_dir),
                "config_file": str(config_file),
                "pipeline_name": pipeline_name
            }