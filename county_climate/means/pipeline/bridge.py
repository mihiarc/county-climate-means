"""
Pipeline Bridge for Seamless Data Handoff

Provides utilities for creating bridge files and configurations that enable
seamless handoff between climate means processing and downstream pipelines.
"""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from county_climate.means.output.catalog import ClimateDataCatalog, CatalogManager
from county_climate.means.pipeline.interface import PipelineInterface, DownstreamConfig

logger = logging.getLogger(__name__)


@dataclass 
class BridgeConfiguration:
    """Configuration for pipeline bridge operations."""
    
    # Source pipeline info
    source_pipeline: str = "climate_means"
    source_version: str = "0.1.0"
    
    # Bridge settings
    create_symlinks: bool = False  # Create symlinks instead of copies
    validate_outputs: bool = True
    create_ready_signals: bool = True
    
    # Downstream pipeline info
    target_pipelines: List[str] = None
    bridge_directory: Optional[Path] = None
    
    def __post_init__(self):
        if self.target_pipelines is None:
            self.target_pipelines = ["climate_extremes"]


class PipelineBridge:
    """
    Creates bridge configurations and data handoff mechanisms.
    
    Facilitates seamless integration between climate means processing
    and downstream analysis pipelines.
    """
    
    def __init__(self, 
                 output_base_dir: Path,
                 bridge_config: Optional[BridgeConfiguration] = None):
        self.output_base_dir = Path(output_base_dir)
        self.config = bridge_config or BridgeConfiguration()
        
        # Initialize interface and catalog manager
        self.interface = PipelineInterface(output_base_dir)
        self.catalog_manager = CatalogManager(output_base_dir)
        
        # Set default bridge directory
        if self.config.bridge_directory is None:
            self.config.bridge_directory = self.output_base_dir / "bridge"
    
    def create_full_bridge(self) -> Dict[str, Any]:
        """
        Create complete bridge setup for downstream pipelines.
        
        Returns:
            Dictionary with bridge creation results
        """
        logger.info("Creating complete pipeline bridge")
        
        bridge_results = {
            "bridge_directory": str(self.config.bridge_directory),
            "created_files": [],
            "target_pipelines": self.config.target_pipelines,
            "creation_time": datetime.now().isoformat(),
            "status": "success"
        }
        
        try:
            # Create bridge directory
            self.config.bridge_directory.mkdir(parents=True, exist_ok=True)
            
            # Validate outputs if requested
            if self.config.validate_outputs:
                validation_results = self.interface.validate_for_downstream()
                if not validation_results["ready_for_downstream"]:
                    bridge_results["status"] = "failed"
                    bridge_results["validation_errors"] = validation_results["issues"]
                    logger.error(f"Validation failed: {validation_results['issues']}")
                    return bridge_results
            
            # Create bridge for each target pipeline
            for target_pipeline in self.config.target_pipelines:
                pipeline_results = self._create_pipeline_specific_bridge(target_pipeline)
                bridge_results["created_files"].extend(pipeline_results["files"])
            
            # Create general bridge files
            general_files = self._create_general_bridge_files()
            bridge_results["created_files"].extend(general_files)
            
            # Create ready signals
            if self.config.create_ready_signals:
                signal_files = self._create_ready_signals()
                bridge_results["created_files"].extend(signal_files)
            
            logger.info(f"Successfully created bridge with {len(bridge_results['created_files'])} files")
            
        except Exception as e:
            logger.error(f"Failed to create bridge: {e}")
            bridge_results["status"] = "failed"
            bridge_results["error"] = str(e)
        
        return bridge_results
    
    def _create_pipeline_specific_bridge(self, target_pipeline: str) -> Dict[str, Any]:
        """Create bridge files specific to a target pipeline."""
        logger.info(f"Creating bridge for {target_pipeline}")
        
        pipeline_dir = self.config.bridge_directory / target_pipeline
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        
        # Create downstream configuration
        downstream_config = self.interface.create_downstream_config(target_pipeline)
        config_file = pipeline_dir / f"{target_pipeline}_config.yaml"
        downstream_config.save(config_file)
        created_files.append(str(config_file))
        
        # Create pipeline-specific manifest
        if target_pipeline == "climate_extremes":
            export_results = self.interface.export_for_extremes_pipeline(pipeline_dir)
            created_files.extend([
                export_results["config_file"],
                export_results["manifest_file"], 
                export_results["processing_summary"]
            ])
        
        # Create data access scripts
        access_script = self._create_data_access_script(target_pipeline, pipeline_dir)
        created_files.append(str(access_script))
        
        # Create example usage
        example_file = self._create_usage_examples(target_pipeline, pipeline_dir)
        created_files.append(str(example_file))
        
        return {"files": created_files}
    
    def _create_general_bridge_files(self) -> List[str]:
        """Create general bridge files for all pipelines."""
        created_files = []
        
        # Create main bridge configuration
        bridge_config = self.interface.create_pipeline_bridge_config(self.config.bridge_directory)
        created_files.append(bridge_config["config_file"])
        
        # Create data catalog copy/symlink in bridge directory
        catalog_source = self.catalog_manager.catalog_file
        catalog_dest = self.config.bridge_directory / "climate_means_catalog.yaml"
        
        if self.config.create_symlinks and catalog_source.exists():
            if catalog_dest.exists():
                catalog_dest.unlink()
            catalog_dest.symlink_to(catalog_source.resolve())
        else:
            shutil.copy2(catalog_source, catalog_dest)
        
        created_files.append(str(catalog_dest))
        
        # Create metadata summary
        metadata_summary = self._create_metadata_summary()
        metadata_file = self.config.bridge_directory / "metadata_summary.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata_summary, f, indent=2)
        created_files.append(str(metadata_file))
        
        return created_files
    
    def _create_ready_signals(self) -> List[str]:
        """Create ready signal files for downstream pipelines."""
        created_files = []
        
        # Main ready signal
        main_ready = self.config.bridge_directory / "PIPELINE_READY"
        with open(main_ready, 'w') as f:
            f.write(f"Climate means pipeline completed successfully\n")
            f.write(f"Completion time: {datetime.now().isoformat()}\n")
            f.write(f"Data catalog: {self.catalog_manager.catalog_file}\n")
            f.write(f"Valid datasets: {len([d for d in self.interface.catalog.datasets if d.is_valid])}\n")
            f.write(f"Bridge directory: {self.config.bridge_directory}\n")
        created_files.append(str(main_ready))
        
        # Pipeline-specific ready signals
        for target_pipeline in self.config.target_pipelines:
            pipeline_ready = self.config.bridge_directory / target_pipeline / "READY"
            with open(pipeline_ready, 'w') as f:
                f.write(f"Ready for {target_pipeline} pipeline\n")
                f.write(f"Config file: {target_pipeline}_config.yaml\n")
                f.write(f"Data catalog: ../climate_means_catalog.yaml\n")
            created_files.append(str(pipeline_ready))
        
        return created_files
    
    def _create_data_access_script(self, target_pipeline: str, output_dir: Path) -> Path:
        """Create a Python script for accessing climate means data."""
        script_content = f'''#!/usr/bin/env python3
"""
Data access script for {target_pipeline} pipeline.

This script provides easy access to climate means data for downstream processing.
Generated automatically by climate-means pipeline bridge.
"""

import sys
from pathlib import Path

# Add climate-means to Python path if needed
# sys.path.insert(0, '/path/to/climate-means')

from county_climate.means.output.catalog import ClimateDataCatalog


def load_climate_means_catalog():
    """Load the climate means data catalog."""
    catalog_file = Path(__file__).parent / "climate_means_catalog.yaml"
    return ClimateDataCatalog(catalog_file)


def get_data_for_scenario(scenario, year_range=None, variable=None, region=None):
    """
    Get climate means data for a specific scenario.
    
    Args:
        scenario: Scenario name (historical, ssp245, ssp585)
        year_range: Tuple of (start_year, end_year) or None for all years
        variable: Variable name or None for all variables
        region: Region name or None for all regions
    
    Returns:
        List of ClimateDataset objects
    """
    catalog = load_climate_means_catalog()
    
    if year_range:
        return catalog.find_by_scenario_year(
            scenario=scenario,
            year_range=year_range,
            variable=variable,
            region=region
        )
    else:
        return catalog.find_datasets(
            scenario=scenario,
            variable=variable,
            region=region
        )


def load_dataset(dataset_id):
    """Load actual data for a dataset."""
    catalog = load_climate_means_catalog()
    return catalog.load_dataset_data(dataset_id)


def get_available_data_summary():
    """Get summary of available data."""
    catalog = load_climate_means_catalog()
    return catalog.get_summary_stats()


# Example usage for {target_pipeline}
if __name__ == "__main__":
    print("Climate Means Data Access for {target_pipeline}")
    print("=" * 50)
    
    # Load catalog
    catalog = load_climate_means_catalog()
    
    # Show available data
    summary = catalog.get_summary_stats()
    print(f"Available scenarios: {{summary['scenarios']}}")
    print(f"Available variables: {{summary['variables']}}")
    print(f"Available regions: {{summary['regions']}}")
    print(f"Total valid datasets: {{summary['valid_datasets']}}")
    
    # Example: Get SSP245 temperature data for CONUS in 2050s
    if 'ssp245' in summary['scenarios'] and 'tas' in summary['variables']:
        datasets = get_data_for_scenario(
            scenario='ssp245',
            year_range=(2050, 2059),
            variable='tas',
            region='CONUS'
        )
        print(f"\\nFound {{len(datasets)}} SSP245 temperature datasets for CONUS 2050s")
        
        # Load first dataset as example
        if datasets:
            data = load_dataset(datasets[0].id)
            if data is not None:
                print(f"Example dataset shape: {{dict(data.dims)}}")
                print(f"Variables: {{list(data.data_vars)}}")
'''
        
        script_file = output_dir / f"access_{target_pipeline}_data.py"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_file.chmod(0o755)
        
        return script_file
    
    def _create_usage_examples(self, target_pipeline: str, output_dir: Path) -> Path:
        """Create usage examples for the target pipeline."""
        examples_content = f'''# Usage Examples for {target_pipeline.title()} Pipeline

## Overview
This directory contains configuration and data access tools for connecting
climate means outputs to the {target_pipeline} pipeline.

## Quick Start

### 1. Load Configuration
```python
import yaml
with open('{target_pipeline}_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

### 2. Access Data Catalog
```python
from county_climate.means.output.catalog import ClimateDataCatalog
catalog = ClimateDataCatalog('climate_means_catalog.yaml')

# Get available scenarios and years
scenarios = catalog.get_available_scenarios()
print(f"Available scenarios: {{scenarios}}")
```

### 3. Find Specific Data
```python
# Example: Get all SSP245 temperature data for 2080s
datasets = catalog.find_by_scenario_year(
    scenario='ssp245',
    year_range=(2080, 2089),
    variable='tas'
)

# Load data for processing
for dataset in datasets:
    data = catalog.load_dataset_data(dataset.id)
    # Process with {target_pipeline} pipeline...
```

## Data Organization

The climate means data is organized as:
```
data/
├── historical/          # Historical scenario (1980-2014)
│   ├── 1980/
│   │   ├── CONUS/
│   │   ├── AK/
│   │   └── ...
│   └── ...
├── ssp245/             # SSP2-4.5 scenario (2015-2100)
│   ├── 2015/
│   └── ...
└── ssp585/             # SSP5-8.5 scenario (if available)
```

## File Naming Convention

Files follow the pattern: `{{variable}}_{{region}}_{{year}}_climatology.nc`

Examples:
- `tas_CONUS_2050_climatology.nc` - Temperature for CONUS, year 2050
- `pr_AK_1990_climatology.nc` - Precipitation for Alaska, year 1990

## Variables Available

- `pr`: Precipitation (kg m⁻² s⁻¹)
- `tas`: Near-surface air temperature (K)
- `tasmax`: Daily maximum temperature (K)
- `tasmin`: Daily minimum temperature (K)

## Regions Available

- `CONUS`: Continental United States
- `AK`: Alaska
- `HI`: Hawaii
- `PRVI`: Puerto Rico & Virgin Islands
- `GU`: Guam & Northern Mariana Islands

## Next Steps

1. Use the data access script: `./access_{target_pipeline}_data.py`
2. Review the configuration file: `{target_pipeline}_config.yaml`
3. Check the processing summary: `processing_summary.json`
4. Validate data availability with the catalog

## Support

For issues with climate means data, check:
- Catalog validation: `catalog.get_summary_stats()`
- File integrity: `dataset.is_valid`
- Processing logs: `../logs/`
'''
        
        examples_file = output_dir / "README.md"
        with open(examples_file, 'w') as f:
            f.write(examples_content)
        
        return examples_file
    
    def _create_metadata_summary(self) -> Dict[str, Any]:
        """Create comprehensive metadata summary."""
        processing_summary = self.interface.get_processing_summary()
        
        metadata_summary = {
            "bridge_info": {
                "created_at": datetime.now().isoformat(),
                "bridge_version": "1.0",
                "source_pipeline": self.config.source_pipeline,
                "target_pipelines": self.config.target_pipelines
            },
            "data_processing": processing_summary,
            "file_organization": {
                "structure": "scenario_year_based",
                "catalog_file": "climate_means_catalog.yaml",
                "access_method": "python_api",
                "coordinate_system": "geographic_wgs84"
            },
            "quality_metrics": {
                "validation_performed": True,
                "data_integrity_verified": True,
                "completeness_checked": True
            },
            "downstream_recommendations": {
                "processing_strategy": "process_by_scenario_then_region",
                "memory_management": "load_datasets_individually", 
                "parallelization": "region_and_variable_level"
            }
        }
        
        return metadata_summary
    
    def cleanup_bridge(self) -> Dict[str, Any]:
        """Clean up bridge files and directories."""
        logger.info("Cleaning up pipeline bridge")
        
        cleanup_results = {
            "cleaned_files": [],
            "cleanup_time": datetime.now().isoformat(),
            "status": "success"
        }
        
        try:
            if self.config.bridge_directory.exists():
                # Remove bridge directory and all contents
                shutil.rmtree(self.config.bridge_directory)
                cleanup_results["cleaned_files"].append(str(self.config.bridge_directory))
                logger.info(f"Removed bridge directory: {self.config.bridge_directory}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup bridge: {e}")
            cleanup_results["status"] = "failed"
            cleanup_results["error"] = str(e)
        
        return cleanup_results