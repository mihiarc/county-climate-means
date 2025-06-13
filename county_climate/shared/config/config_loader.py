"""
Configuration loader and validator for climate data processing pipelines.

This module provides functionality to load, validate, and manage pipeline
configurations from various sources (YAML, JSON, environment variables).
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import yaml
from pydantic import ValidationError

from .integration_config import (
    PipelineConfiguration,
    ProcessingProfile,
    EnvironmentType,
    StageConfiguration,
    ProcessingStage,
    TriggerType,
    ResourceLimits,
)


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


class ConfigurationLoader:
    """Loads and validates pipeline configurations from various sources."""
    
    def __init__(self, config_search_paths: Optional[List[Path]] = None):
        """Initialize configuration loader.
        
        Args:
            config_search_paths: Directories to search for configuration files.
                                Defaults to [current_dir, ~/.county-climate, /etc/county-climate]
        """
        if config_search_paths is None:
            config_search_paths = [
                Path.cwd(),
                Path.home() / ".county-climate",
                Path("/etc/county-climate"),
            ]
        
        self.search_paths = [Path(p) for p in config_search_paths]
        self._config_cache: Dict[str, PipelineConfiguration] = {}
        self._profile_cache: Dict[str, ProcessingProfile] = {}
    
    def load_pipeline_config(
        self, 
        config_source: Union[str, Path, Dict[str, Any]],
        environment: Optional[EnvironmentType] = None,
        validate: bool = True
    ) -> PipelineConfiguration:
        """Load pipeline configuration from various sources.
        
        Args:
            config_source: Configuration file path, dict, or config name to search for
            environment: Environment type for environment-specific overrides
            validate: Whether to validate the configuration
            
        Returns:
            Validated pipeline configuration
            
        Raises:
            ConfigurationError: If configuration cannot be loaded or validated
        """
        try:
            # Load raw configuration data
            if isinstance(config_source, dict):
                config_data = config_source
            elif isinstance(config_source, (str, Path)):
                config_data = self._load_config_file(config_source)
            else:
                raise ConfigurationError(f"Unsupported config source type: {type(config_source)}")
            
            # Apply environment-specific overrides
            if environment:
                config_data = self._apply_environment_overrides(config_data, environment)
            
            # Apply environment variable overrides
            config_data = self._apply_env_var_overrides(config_data)
            
            # Create and validate configuration
            if validate:
                config = PipelineConfiguration(**config_data)
            else:
                config = PipelineConfiguration.construct(**config_data)
            
            return config
            
        except (ValidationError, ValueError, yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Failed to load pipeline configuration: {e}")
    
    def load_processing_profile(
        self,
        profile_source: Union[str, Path, Dict[str, Any]]
    ) -> ProcessingProfile:
        """Load processing profile configuration.
        
        Args:
            profile_source: Profile file path, dict, or profile name
            
        Returns:
            Processing profile configuration
        """
        try:
            if isinstance(profile_source, dict):
                profile_data = profile_source
            elif isinstance(profile_source, (str, Path)):
                profile_data = self._load_config_file(profile_source)
            else:
                raise ConfigurationError(f"Unsupported profile source type: {type(profile_source)}")
            
            return ProcessingProfile(**profile_data)
            
        except (ValidationError, ValueError, yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Failed to load processing profile: {e}")
    
    def find_config_file(self, config_name: str) -> Optional[Path]:
        """Find configuration file in search paths.
        
        Args:
            config_name: Name of configuration file (with or without extension)
            
        Returns:
            Path to configuration file if found, None otherwise
        """
        # Try different extensions
        extensions = ['.yaml', '.yml', '.json']
        
        for search_path in self.search_paths:
            if not search_path.exists():
                continue
                
            # Try exact name first
            config_path = search_path / config_name
            if config_path.exists():
                return config_path
            
            # Try with extensions
            for ext in extensions:
                config_path = search_path / f"{config_name}{ext}"
                if config_path.exists():
                    return config_path
        
        return None
    
    def list_available_configs(self) -> List[str]:
        """List all available configuration files in search paths."""
        configs = set()
        extensions = {'.yaml', '.yml', '.json'}
        
        for search_path in self.search_paths:
            if not search_path.exists():
                continue
                
            for file_path in search_path.glob("*"):
                if file_path.suffix in extensions:
                    configs.add(file_path.stem)
        
        return sorted(list(configs))
    
    def validate_config(self, config: PipelineConfiguration) -> List[str]:
        """Validate configuration and return list of warnings/issues.
        
        Args:
            config: Pipeline configuration to validate
            
        Returns:
            List of validation warnings (empty if no issues)
        """
        warnings = []
        
        # Check for unreferenced stages
        referenced_stages = set()
        for stage in config.stages:
            referenced_stages.update(stage.depends_on)
            referenced_stages.update(stage.optional_depends_on)
        
        stage_ids = {stage.stage_id for stage in config.stages}
        unreferenced = stage_ids - referenced_stages
        if unreferenced:
            warnings.append(f"Unreferenced stages (no dependencies): {unreferenced}")
        
        # Check for resource limit consistency
        for stage in config.stages:
            if (config.global_resource_limits.max_memory_gb and 
                stage.resource_limits.max_memory_gb and
                stage.resource_limits.max_memory_gb > config.global_resource_limits.max_memory_gb):
                warnings.append(
                    f"Stage {stage.stage_id} memory limit exceeds global limit"
                )
        
        # Check for data flow consistency
        flow_sources = {flow.source_stage for flow in config.data_flows}
        flow_targets = {flow.target_stage for flow in config.data_flows}
        
        stages_without_inputs = stage_ids - flow_targets
        stages_without_outputs = stage_ids - flow_sources
        
        if len(stages_without_inputs) > 1:  # Allow one root stage
            warnings.append(f"Multiple stages without data inputs: {stages_without_inputs}")
        
        if len(stages_without_outputs) > 1:  # Allow one leaf stage
            warnings.append(f"Multiple stages without data outputs: {stages_without_outputs}")
        
        return warnings
    
    def create_default_config(self, config_type: str = "basic") -> PipelineConfiguration:
        """Create a default pipeline configuration.
        
        Args:
            config_type: Type of default config ("basic", "full", "development")
            
        Returns:
            Default pipeline configuration
        """
        base_config = {
            "pipeline_id": f"default_{config_type}",
            "pipeline_name": f"Default {config_type.title()} Pipeline",
            "environment": "development",
            "base_data_path": "/tmp/climate_data",
            "stages": [],
        }
        
        if config_type == "basic":
            base_config["stages"] = [
                {
                    "stage_id": "means_processor",
                    "stage_type": "means",
                    "stage_name": "Climate Means Processing",
                    "package_name": "county_climate.means",
                    "entry_point": "process_region",
                    "trigger_type": "manual",
                }
            ]
        elif config_type == "full":
            base_config["stages"] = [
                {
                    "stage_id": "means_processor",
                    "stage_type": "means",
                    "stage_name": "Climate Means Processing",
                    "package_name": "county_climate.means",
                    "entry_point": "process_region",
                    "trigger_type": "manual",
                },
                {
                    "stage_id": "metrics_processor",
                    "stage_type": "metrics", 
                    "stage_name": "Climate Metrics Processing",
                    "package_name": "county_climate.metrics",
                    "entry_point": "process_county_metrics",
                    "depends_on": ["means_processor"],
                    "trigger_type": "dependency",
                },
                {
                    "stage_id": "validation",
                    "stage_type": "validation",
                    "stage_name": "Data Validation",
                    "package_name": "county_climate.shared.validation",
                    "entry_point": "validate_pipeline_output",
                    "depends_on": ["metrics_processor"],
                    "trigger_type": "dependency",
                }
            ]
        
        return PipelineConfiguration(**base_config)
    
    def _load_config_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        file_path = Path(file_path)
        
        # If it's just a name, search for it
        if not file_path.exists():
            found_path = self.find_config_file(str(file_path))
            if found_path:
                file_path = found_path
            else:
                raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if file_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif file_path.suffix == '.json':
                return json.load(f)
            else:
                raise ConfigurationError(f"Unsupported file format: {file_path.suffix}")
    
    def _apply_environment_overrides(
        self, 
        config_data: Dict[str, Any], 
        environment: EnvironmentType
    ) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides."""
        config_copy = config_data.copy()
        
        # Apply global environment overrides
        if "environment_overrides" in config_copy:
            env_overrides = config_copy["environment_overrides"].get(environment.value, {})
            config_copy.update(env_overrides)
        
        # Apply stage-specific environment overrides
        if "stages" in config_copy:
            for stage in config_copy["stages"]:
                if "environment_overrides" in stage:
                    env_overrides = stage["environment_overrides"].get(environment.value, {})
                    stage.update(env_overrides)
        
        return config_copy
    
    def _apply_env_var_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        config_copy = config_data.copy()
        
        # Common environment variable mappings
        env_mappings = {
            "COUNTY_CLIMATE_DATA_PATH": "base_data_path",
            "COUNTY_CLIMATE_TEMP_PATH": "temp_data_path", 
            "COUNTY_CLIMATE_LOG_PATH": "log_path",
            "COUNTY_CLIMATE_MAX_WORKERS": "global_resource_limits.max_cpu_cores",
            "COUNTY_CLIMATE_MAX_MEMORY": "global_resource_limits.max_memory_gb",
            "COUNTY_CLIMATE_ENVIRONMENT": "environment",
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert value to appropriate type
                if config_path.endswith(("_cores", "max_workers")):
                    value = int(value)
                elif config_path.endswith(("_gb", "_memory")):
                    value = float(value)
                elif config_path.endswith("_path"):
                    value = Path(value)
                
                # Set nested configuration value
                self._set_nested_config(config_copy, config_path, value)
        
        return config_copy
    
    def _set_nested_config(self, config: Dict[str, Any], path: str, value: Any):
        """Set nested configuration value using dot notation."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value


class ConfigurationManager:
    """High-level configuration management interface."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Primary configuration directory
        """
        search_paths = [config_dir] if config_dir else None
        self.loader = ConfigurationLoader(search_paths)
        self.active_config: Optional[PipelineConfiguration] = None
    
    def load_config(
        self, 
        config_name: str = "pipeline",
        environment: Optional[str] = None
    ) -> PipelineConfiguration:
        """Load and activate pipeline configuration.
        
        Args:
            config_name: Name of configuration to load
            environment: Environment type (development, production, etc.)
            
        Returns:
            Loaded pipeline configuration
        """
        env_type = EnvironmentType(environment) if environment else None
        self.active_config = self.loader.load_pipeline_config(config_name, env_type)
        return self.active_config
    
    def get_stage_config(self, stage_id: str) -> Optional[StageConfiguration]:
        """Get configuration for specific stage."""
        if not self.active_config:
            raise ConfigurationError("No active configuration loaded")
        
        return self.active_config.get_stage_by_id(stage_id)
    
    def list_configs(self) -> List[str]:
        """List available configurations."""
        return self.loader.list_available_configs()
    
    def validate_active_config(self) -> List[str]:
        """Validate currently active configuration."""
        if not self.active_config:
            raise ConfigurationError("No active configuration loaded")
        
        return self.loader.validate_config(self.active_config)
    
    def create_sample_configs(self, output_dir: Path):
        """Create sample configuration files.
        
        Args:
            output_dir: Directory to write sample configurations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create basic pipeline config
        basic_config = self.loader.create_default_config("basic")
        with open(output_dir / "pipeline_basic.yaml", 'w') as f:
            yaml.dump(basic_config.dict(), f, default_flow_style=False, indent=2)
        
        # Create full pipeline config
        full_config = self.loader.create_default_config("full")
        with open(output_dir / "pipeline_full.yaml", 'w') as f:
            yaml.dump(full_config.dict(), f, default_flow_style=False, indent=2)
        
        # Create sample processing profile
        sample_profile = {
            "profile_name": "conus_temperature_historical",
            "description": "Process CONUS temperature data for historical period",
            "regions": ["CONUS"],
            "variables": ["temperature"],
            "scenarios": ["historical"],
            "year_ranges": [[1990, 2020]],
            "enable_means": True,
            "enable_metrics": True,
            "max_parallel_regions": 1,
            "max_parallel_variables": 2,
            "memory_per_process_gb": 4.0,
        }
        
        with open(output_dir / "profile_sample.yaml", 'w') as f:
            yaml.dump(sample_profile, f, default_flow_style=False, indent=2)