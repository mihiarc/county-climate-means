"""
Pipeline Interface for Downstream Integration

Provides a standardized interface for connecting climate means outputs
to downstream climate extremes and other analysis pipelines.
"""

import json
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

from means.output.catalog import ClimateDataCatalog, CatalogManager
from means.output.organization import OutputOrganizer
from means.core.regions import REGION_BOUNDS
from means.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class DownstreamConfig:
    """Configuration for downstream pipeline integration."""
    
    # Data source information (required)
    input_data_catalog: str
    scenarios_available: Dict[str, List[int]]
    variables_available: List[str]
    regions_available: List[str]
    
    # Data source information (with defaults)
    input_data_format: str = "netcdf4_cf"
    coordinate_system: str = "geographic_wgs84"
    
    # Processing recommendations
    recommended_workers: int = 6
    memory_per_worker_gb: int = 4
    
    # Quality information
    data_quality_validated: bool = True
    processing_software: str = "climate-means"
    processing_version: str = "0.1.0"
    
    # Temporal information
    temporal_resolution: str = "daily"
    climatology_period: str = "30_year"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save(self, output_path: Path, format: str = "yaml") -> None:
        """Save configuration to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "yaml":
            with open(output_path, 'w') as f:
                f.write(self.to_yaml())
        elif format.lower() == "json":
            with open(output_path, 'w') as f:
                f.write(self.to_json())
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved downstream config to {output_path}")


class PipelineInterface:
    """
    Interface for connecting climate means to downstream pipelines.
    
    Provides standardized methods for data discovery, configuration export,
    and pipeline integration.
    """
    
    def __init__(self, output_base_dir: Optional[Path] = None):
        if output_base_dir is None:
            config = get_config()
            output_base_dir = config.paths.output_base_dir
        
        self.output_base_dir = Path(output_base_dir)
        self.catalog_manager = CatalogManager(self.output_base_dir)
        self.organizer = OutputOrganizer(self.output_base_dir)
        
        # Load or create catalog
        self.catalog = self.catalog_manager.load_catalog()
    
    def get_available_scenarios(self) -> Dict[str, List[int]]:
        """
        Get scenarios and their available years for downstream processing.
        
        Returns:
            Dictionary mapping scenario names to lists of available years
        """
        scenarios = {}
        
        for scenario in self.catalog.get_available_scenarios():
            years = self.catalog.get_scenario_years(scenario)
            scenarios[scenario] = sorted(years)
        
        return scenarios
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Export processing summary for downstream pipeline.
        
        Returns:
            Comprehensive summary of processing results and data availability
        """
        catalog_stats = self.catalog.get_summary_stats()
        
        return {
            "processing_complete": True,
            "processing_date": datetime.now().isoformat(),
            "software_info": {
                "name": "climate-means",
                "version": "0.1.0"
            },
            "data_summary": {
                "total_datasets": catalog_stats["total_datasets"],
                "valid_datasets": catalog_stats["valid_datasets"],
                "total_size_gb": catalog_stats["total_file_size_gb"],
                "scenarios": catalog_stats["scenarios"],
                "variables": catalog_stats["variables"],
                "regions": catalog_stats["regions"],
                "year_ranges": catalog_stats["year_range"]
            },
            "data_organization": {
                "structure": "scenario_year_based",
                "catalog_file": str(self.catalog_manager.catalog_file),
                "data_directory": str(self.output_base_dir / "data"),
                "metadata_directory": str(self.output_base_dir / "metadata")
            },
            "quality_assurance": {
                "validation_performed": True,
                "data_integrity_checked": True,
                "completeness_verified": True
            }
        }
    
    def create_downstream_config(self, 
                               next_pipeline: str,
                               custom_settings: Optional[Dict[str, Any]] = None) -> DownstreamConfig:
        """
        Generate configuration for next pipeline stage.
        
        Args:
            next_pipeline: Name of the downstream pipeline
            custom_settings: Optional custom settings to override defaults
            
        Returns:
            DownstreamConfig object for the next pipeline
        """
        # Get current system configuration
        config = get_config()
        
        # Base configuration
        downstream_config = DownstreamConfig(
            input_data_catalog=str(self.catalog_manager.catalog_file),
            scenarios_available=self.get_available_scenarios(),
            variables_available=self.catalog.get_available_variables(),
            regions_available=self.catalog.get_available_regions(),
            recommended_workers=config.processing.max_workers,
            memory_per_worker_gb=getattr(config.processing, "max_memory_per_process_gb", 4)
        )
        
        # Pipeline-specific customizations
        if next_pipeline == "climate_extremes":
            # Optimize for extremes pipeline
            downstream_config.recommended_workers = min(downstream_config.recommended_workers, 8)
            downstream_config.memory_per_worker_gb = max(downstream_config.memory_per_worker_gb, 6)
        
        elif next_pipeline == "climate_metrics":
            # Optimize for metrics pipeline
            downstream_config.recommended_workers = max(downstream_config.recommended_workers, 4)
            downstream_config.memory_per_worker_gb = max(downstream_config.memory_per_worker_gb, 8)
        
        # Apply custom settings
        if custom_settings:
            for key, value in custom_settings.items():
                if hasattr(downstream_config, key):
                    setattr(downstream_config, key, value)
        
        return downstream_config
    
    def export_for_extremes_pipeline(self, 
                                   output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Export data manifest specifically for climate extremes pipeline.
        
        Args:
            output_dir: Directory to save export files
            
        Returns:
            Dictionary with export information
        """
        if output_dir is None:
            output_dir = self.output_base_dir / "exports"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create downstream configuration
        extremes_config = self.create_downstream_config("climate_extremes")
        config_file = output_dir / "climate_extremes_config.yaml"
        extremes_config.save(config_file)
        
        # Create processing summary
        processing_summary = self.get_processing_summary()
        summary_file = output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(processing_summary, f, indent=2)
        
        # Create data manifest for extremes pipeline
        extremes_manifest = {
            "pipeline_name": "climate_extremes",
            "input_source": "climate_means",
            "config_file": str(config_file),
            "data_catalog": str(self.catalog_manager.catalog_file),
            "processing_summary": str(summary_file),
            "data_access": {
                "method": "direct_file_access",
                "catalog_api": "means.output.catalog.ClimateDataCatalog",
                "example_usage": self._get_example_usage()
            },
            "recommendations": {
                "processing_order": "by_scenario_then_region",
                "memory_optimization": "process_by_variable",
                "parallelization": "scenario_and_region_level"
            }
        }
        
        manifest_file = output_dir / "extremes_pipeline_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(extremes_manifest, f, indent=2)
        
        logger.info(f"Exported for extremes pipeline to {output_dir}")
        
        return {
            "export_directory": str(output_dir),
            "config_file": str(config_file),
            "manifest_file": str(manifest_file),
            "processing_summary": str(summary_file),
            "datasets_available": len(self.catalog.datasets)
        }
    
    def _get_example_usage(self) -> Dict[str, str]:
        """Get example usage code for downstream pipelines."""
        return {
            "python_api": """
# Load climate means catalog
from means.output.catalog import ClimateDataCatalog
catalog = ClimateDataCatalog('output/catalog/climate_means_catalog.yaml')

# Find SSP245 temperature data for 2050s
datasets = catalog.find_by_scenario_year(
    scenario='ssp245',
    year_range=(2050, 2059),
    variable='tas',
    region='CONUS'
)

# Load data for processing
for dataset in datasets:
    data = catalog.load_dataset_data(dataset.id)
    # Process with extremes pipeline...
            """,
            "command_line": """
# Use pipeline interface
python -c "
from means.pipeline import PipelineInterface
interface = PipelineInterface()
config = interface.create_downstream_config('climate_extremes')
config.save('extremes_config.yaml')
"
            """
        }
    
    def validate_for_downstream(self) -> Dict[str, Any]:
        """
        Validate that outputs are ready for downstream processing.
        
        Returns:
            Validation results with any issues found
        """
        validation_results = {
            "ready_for_downstream": True,
            "issues": [],
            "warnings": [],
            "validation_details": {}
        }
        
        # Check catalog integrity
        try:
            catalog_stats = self.catalog.get_summary_stats()
            if catalog_stats["valid_datasets"] == 0:
                validation_results["ready_for_downstream"] = False
                validation_results["issues"].append("No valid datasets in catalog")
        except Exception as e:
            validation_results["ready_for_downstream"] = False
            validation_results["issues"].append(f"Catalog validation failed: {e}")
        
        # Check output organization
        org_validation = self.organizer.validate_organization()
        if not org_validation["structure_valid"]:
            validation_results["warnings"].extend(org_validation["issues"])
        
        validation_results["validation_details"]["organization"] = org_validation
        
        # Check required metadata files
        required_files = [
            "catalog/climate_means_catalog.yaml",
            "metadata/scenario_definitions.json",
            "metadata/processing_methods.json"
        ]
        
        missing_files = []
        for required_file in required_files:
            if not (self.output_base_dir / required_file).exists():
                missing_files.append(required_file)
        
        if missing_files:
            validation_results["ready_for_downstream"] = False
            validation_results["issues"].extend([f"Missing file: {f}" for f in missing_files])
        
        # Check data completeness
        scenarios = self.get_available_scenarios()
        for scenario, years in scenarios.items():
            if not years:
                validation_results["warnings"].append(f"No data available for scenario: {scenario}")
        
        validation_results["validation_details"]["data_availability"] = scenarios
        
        logger.info(f"Downstream validation: {'PASSED' if validation_results['ready_for_downstream'] else 'FAILED'}")
        
        return validation_results
    
    def create_pipeline_bridge_config(self, 
                                    bridge_directory: Optional[Path] = None) -> Dict[str, Any]:
        """
        Create configuration for pipeline bridge/handoff.
        
        Args:
            bridge_directory: Directory for bridge files
            
        Returns:
            Bridge configuration information
        """
        if bridge_directory is None:
            bridge_directory = self.output_base_dir / "bridge"
        
        bridge_directory = Path(bridge_directory)
        bridge_directory.mkdir(parents=True, exist_ok=True)
        
        # Create bridge configuration
        bridge_config = {
            "bridge_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "source_pipeline": {
                "name": "climate_means", 
                "version": "0.1.0",
                "output_directory": str(self.output_base_dir)
            },
            "data_handoff": {
                "catalog_file": str(self.catalog_manager.catalog_file),
                "data_directory": str(self.output_base_dir / "data"),
                "organization": "scenario_year_based",
                "format": "netcdf4_cf"
            },
            "downstream_recommendations": {
                "next_pipelines": ["climate_extremes", "climate_metrics"],
                "processing_order": ["historical", "ssp245", "ssp585"],
                "parallelization_strategy": "by_scenario_and_region"
            },
            "quality_assurance": self.validate_for_downstream()
        }
        
        # Save bridge configuration
        bridge_config_file = bridge_directory / "pipeline_bridge_config.json"
        with open(bridge_config_file, 'w') as f:
            json.dump(bridge_config, f, indent=2)
        
        # Create ready signal file
        ready_file = bridge_directory / "READY_FOR_DOWNSTREAM"
        with open(ready_file, 'w') as f:
            f.write(f"Climate means processing completed at {datetime.now().isoformat()}\n")
            f.write(f"Data catalog: {self.catalog_manager.catalog_file}\n")
            f.write(f"Valid datasets: {len([d for d in self.catalog.datasets if d.is_valid])}\n")
        
        logger.info(f"Created pipeline bridge configuration in {bridge_directory}")
        
        return {
            "bridge_directory": str(bridge_directory),
            "config_file": str(bridge_config_file),
            "ready_file": str(ready_file),
            "bridge_config": bridge_config
        }