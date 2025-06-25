#!/usr/bin/env python3
"""
Configuration Management for Climate Means Package

Centralized configuration for data paths, output directories, and processing settings.
Supports environment variables, config files, and programmatic overrides.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass, field

# Import the new organized output paths
try:
    from county_climate.shared.config.output_paths import OrganizedOutputPaths
except ImportError:
    # Fallback if import fails
    OrganizedOutputPaths = None

logger = logging.getLogger(__name__)

@dataclass
class DataPaths:
    """Data path configuration."""
    input_data_dir: str = "/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    output_base_dir: str = "/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/organized"
    
    # Use organized output structure
    use_organized_structure: bool = True
    output_version: str = "v1.0"
    
    # Legacy regional output directories (for backward compatibility)
    conus_output_dir: str = "output/conus_normals"
    alaska_output_dir: str = "output/alaska_normals"
    hawaii_output_dir: str = "output/hawaii_normals"
    prvi_output_dir: str = "output/prvi_normals"
    guam_output_dir: str = "output/guam_normals"
    
    # Validation and visualization outputs
    validation_output_dir: str = "output/validation"
    visualization_output_dir: str = "output/visualizations"
    
    def __post_init__(self):
        """Convert string paths to Path objects and resolve environment variables."""
        # Expand environment variables
        self.input_data_dir = os.path.expandvars(self.input_data_dir)
        self.output_base_dir = os.path.expandvars(self.output_base_dir)
        
        # Convert to Path objects
        self.input_data_dir = Path(self.input_data_dir)
        self.output_base_dir = Path(self.output_base_dir)
        
        # Initialize organized paths if enabled
        if self.use_organized_structure and OrganizedOutputPaths:
            self._organized_paths = OrganizedOutputPaths(
                base_path=str(self.output_base_dir),
                version=self.output_version
            )
        else:
            self._organized_paths = None
            # Use legacy paths
            self.conus_output_dir = Path(self.conus_output_dir)
            self.alaska_output_dir = Path(self.alaska_output_dir)
            self.hawaii_output_dir = Path(self.hawaii_output_dir)
            self.prvi_output_dir = Path(self.prvi_output_dir)
            self.guam_output_dir = Path(self.guam_output_dir)
            self.validation_output_dir = Path(self.validation_output_dir)
            self.visualization_output_dir = Path(self.visualization_output_dir)
    
    def get_means_output_path(self, scenario: str, region: str) -> Path:
        """Get output path for climate means using organized structure."""
        if self._organized_paths:
            return self._organized_paths.get_means_output_path(scenario, region)
        else:
            # Fallback to legacy structure
            return self.get_regional_output_dir_legacy(region) / scenario
    
    def get_means_filename(self, variable: str, region: str, scenario: str, 
                          start_year: int, end_year: Optional[int] = None) -> str:
        """Get standardized filename for climate means."""
        if self._organized_paths:
            return self._organized_paths.get_means_filename(
                variable, region, scenario, start_year, end_year
            )
        else:
            # Legacy naming convention
            return f"{variable}_{region}_{scenario}_{start_year}_30yr_normal.nc"
    
    def get_regional_output_dir_legacy(self, region_key: str) -> Path:
        """Get legacy regional output directory."""
        region_dirs = {
            'CONUS': self.conus_output_dir,
            'AK': self.alaska_output_dir,
            'HI': self.hawaii_output_dir,
            'PRVI': self.prvi_output_dir,
            'GU': self.guam_output_dir,
        }
        return region_dirs.get(region_key, self.output_base_dir)

@dataclass
class OutputConfig:
    """Output organization and catalog configuration."""
    # Output organization strategy
    organization_strategy: str = "scenario_year"  # "scenario_year" or "legacy_regional"
    
    # Catalog settings
    create_catalog: bool = True
    catalog_format: str = "yaml"  # "yaml" or "json"
    include_checksums: bool = True
    auto_update_catalog: bool = True
    
    # Metadata settings
    export_spatial_metadata: bool = True
    include_processing_metadata: bool = True
    
    # File organization
    scenario_year_structure: bool = True
    flatten_internal_types: bool = True  # Hide internal processing distinctions

@dataclass
class PipelineIntegrationConfig:
    """Configuration for pipeline integration features."""
    # Bridge settings
    create_bridge_files: bool = True
    downstream_pipeline_hint: str = "climate_extremes"
    export_processing_config: bool = True
    
    # Validation settings
    validate_outputs_for_downstream: bool = True
    create_ready_signals: bool = True
    
    # Supported downstream pipelines
    supported_pipelines: list = field(default_factory=lambda: ["climate_extremes", "climate_metrics"])

@dataclass
class ProcessingConfig:
    """Processing configuration settings."""
    # Multiprocessing settings
    max_workers: int = 4
    batch_size: int = 15
    max_retries: int = 3
    
    # Memory management
    memory_conservative: bool = True
    chunk_size_mb: int = 100
    max_memory_per_process_gb: int = 4
    
    # Processing options
    processing_type: str = "sequential"  # or "multiprocessing"
    safe_mode: bool = True
    
    # Climate calculation settings
    climate_period_length: int = 30  # years
    
    # File handling
    compression_level: int = 4
    file_format: str = "netcdf4"
    
    # Scenario/year mapping settings
    scenario_year_ranges: dict = field(default_factory=lambda: {
        "historical": (1980, 2014),
        "ssp245": (2015, 2100),
        "ssp585": (2015, 2100)
    })

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    console_output: bool = True

@dataclass
class ClimateConfig:
    """Main configuration class combining all settings."""
    paths: DataPaths = field(default_factory=DataPaths)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    pipeline_integration: PipelineIntegrationConfig = field(default_factory=PipelineIntegrationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Initialize configuration from environment variables and config files."""
        self._load_from_environment()
        self._load_from_config_file()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Data paths
        if "CLIMATE_INPUT_DIR" in os.environ:
            self.paths.input_data_dir = os.environ["CLIMATE_INPUT_DIR"]
        
        if "CLIMATE_OUTPUT_DIR" in os.environ:
            self.paths.output_base_dir = os.environ["CLIMATE_OUTPUT_DIR"]
        
        # Processing settings
        if "CLIMATE_MAX_WORKERS" in os.environ:
            try:
                self.processing.max_workers = int(os.environ["CLIMATE_MAX_WORKERS"])
            except ValueError:
                logger.warning("Invalid CLIMATE_MAX_WORKERS value, using default")
        
        if "CLIMATE_BATCH_SIZE" in os.environ:
            try:
                self.processing.batch_size = int(os.environ["CLIMATE_BATCH_SIZE"])
            except ValueError:
                logger.warning("Invalid CLIMATE_BATCH_SIZE value, using default")
        
        # Logging
        if "CLIMATE_LOG_LEVEL" in os.environ:
            self.logging.level = os.environ["CLIMATE_LOG_LEVEL"].upper()
        
        if "CLIMATE_LOG_FILE" in os.environ:
            self.logging.log_file = os.environ["CLIMATE_LOG_FILE"]
    
    def _load_from_config_file(self):
        """Load configuration from YAML config file."""
        config_paths = [
            Path.cwd() / "climate_config.yaml",
            Path.home() / ".climate_means" / "config.yaml",
            Path("/etc/climate_means/config.yaml")
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                    
                    self._update_from_dict(config_data)
                    logger.info(f"Loaded configuration from {config_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        if "paths" in config_dict:
            for key, value in config_dict["paths"].items():
                if hasattr(self.paths, key):
                    setattr(self.paths, key, value)
        
        if "processing" in config_dict:
            for key, value in config_dict["processing"].items():
                if hasattr(self.processing, key):
                    setattr(self.processing, key, value)
        
        if "output" in config_dict:
            for key, value in config_dict["output"].items():
                if hasattr(self.output, key):
                    setattr(self.output, key, value)
        
        if "pipeline_integration" in config_dict:
            for key, value in config_dict["pipeline_integration"].items():
                if hasattr(self.pipeline_integration, key):
                    setattr(self.pipeline_integration, key, value)
        
        if "logging" in config_dict:
            for key, value in config_dict["logging"].items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)
    
    def save_config(self, config_path: Path):
        """Save current configuration to YAML file."""
        config_dict = {
            "paths": {
                "input_data_dir": str(self.paths.input_data_dir),
                "output_base_dir": str(self.paths.output_base_dir),
                "conus_output_dir": str(self.paths.conus_output_dir),
                "alaska_output_dir": str(self.paths.alaska_output_dir),
                "hawaii_output_dir": str(self.paths.hawaii_output_dir),
                "prvi_output_dir": str(self.paths.prvi_output_dir),
                "guam_output_dir": str(self.paths.guam_output_dir),
                "validation_output_dir": str(self.paths.validation_output_dir),
                "visualization_output_dir": str(self.paths.visualization_output_dir),
            },
            "processing": {
                "max_workers": self.processing.max_workers,
                "batch_size": self.processing.batch_size,
                "max_retries": self.processing.max_retries,
                "memory_conservative": self.processing.memory_conservative,
                "chunk_size_mb": self.processing.chunk_size_mb,
                "max_memory_per_process_gb": self.processing.max_memory_per_process_gb,
                "processing_type": self.processing.processing_type,
                "safe_mode": self.processing.safe_mode,
                "climate_period_length": self.processing.climate_period_length,
                "compression_level": self.processing.compression_level,
                "file_format": self.processing.file_format,
                "scenario_year_ranges": self.processing.scenario_year_ranges,
            },
            "output": {
                "organization_strategy": self.output.organization_strategy,
                "create_catalog": self.output.create_catalog,
                "catalog_format": self.output.catalog_format,
                "include_checksums": self.output.include_checksums,
                "auto_update_catalog": self.output.auto_update_catalog,
                "export_spatial_metadata": self.output.export_spatial_metadata,
                "include_processing_metadata": self.output.include_processing_metadata,
                "scenario_year_structure": self.output.scenario_year_structure,
                "flatten_internal_types": self.output.flatten_internal_types,
            },
            "pipeline_integration": {
                "create_bridge_files": self.pipeline_integration.create_bridge_files,
                "downstream_pipeline_hint": self.pipeline_integration.downstream_pipeline_hint,
                "export_processing_config": self.pipeline_integration.export_processing_config,
                "validate_outputs_for_downstream": self.pipeline_integration.validate_outputs_for_downstream,
                "create_ready_signals": self.pipeline_integration.create_ready_signals,
                "supported_pipelines": self.pipeline_integration.supported_pipelines,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "log_file": self.logging.log_file,
                "console_output": self.logging.console_output,
            }
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
    
    def create_output_directories(self):
        """Create all output directories if they don't exist."""
        directories = [
            self.paths.output_base_dir,
            self.paths.conus_output_dir,
            self.paths.alaska_output_dir,
            self.paths.hawaii_output_dir,
            self.paths.prvi_output_dir,
            self.paths.guam_output_dir,
            self.paths.validation_output_dir,
            self.paths.visualization_output_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def validate_paths(self) -> bool:
        """Validate that required paths exist and are accessible."""
        # Check input data directory
        if not self.paths.input_data_dir.exists():
            logger.error(f"Input data directory does not exist: {self.paths.input_data_dir}")
            return False
        
        if not self.paths.input_data_dir.is_dir():
            logger.error(f"Input data path is not a directory: {self.paths.input_data_dir}")
            return False
        
        # Check if we can create output directories
        try:
            self.create_output_directories()
        except PermissionError:
            logger.error(f"Cannot create output directories in: {self.paths.output_base_dir}")
            return False
        
        return True
    
    def get_regional_output_dir(self, region_key: str) -> Path:
        """Get the output directory for a specific region."""
        if self.paths.use_organized_structure and hasattr(self.paths, '_organized_paths') and self.paths._organized_paths:
            # Use organized structure - return base means directory
            # The actual scenario/region subdirectory will be created by get_means_output_path
            return self.paths._organized_paths.climate_means_base
        else:
            # Legacy structure
            region_dirs = {
                'CONUS': self.paths.conus_output_dir,
                'AK': self.paths.alaska_output_dir,
                'HI': self.paths.hawaii_output_dir,
                'PRVI': self.paths.prvi_output_dir,
                'GU': self.paths.guam_output_dir,
            }
            
            return region_dirs.get(region_key, self.paths.output_base_dir)


# Global configuration instance
_config = None

def get_config() -> ClimateConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = ClimateConfig()
    return _config

def set_config(config: ClimateConfig):
    """Set the global configuration instance."""
    global _config
    _config = config

def reset_config():
    """Reset configuration to defaults."""
    global _config
    _config = ClimateConfig()

# Convenience functions for common operations
def get_input_data_dir() -> Path:
    """Get the input data directory."""
    return get_config().paths.input_data_dir

def get_output_base_dir() -> Path:
    """Get the base output directory."""
    return get_config().paths.output_base_dir

def get_regional_output_dir(region_key: str) -> Path:
    """Get the output directory for a specific region."""
    return get_config().get_regional_output_dir(region_key)

def get_processing_config() -> ProcessingConfig:
    """Get the processing configuration."""
    return get_config().processing

def create_sample_config(config_path: Path = None):
    """Create a sample configuration file."""
    if config_path is None:
        config_path = Path.cwd() / "climate_config.yaml"
    
    config = ClimateConfig()
    config.save_config(config_path)
    
    print(f"Sample configuration created at: {config_path}")
    print("Edit this file to customize your settings.")
    
    return config_path 