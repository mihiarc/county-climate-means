#!/usr/bin/env python3
"""
Configuration Management for Climate Data Processing

This module provides centralized configuration management using dataclasses
and environment variable support.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from .constants import *


@dataclass
class ClimateProcessingConfig:
    """
    Main configuration class for climate data processing.
    
    This class centralizes all configuration settings and provides
    reasonable defaults with environment variable overrides.
    """
    
    # Data directories
    input_data_dir: str = DEFAULT_INPUT_DATA_DIR
    output_base_dir: str = DEFAULT_OUTPUT_BASE_DIR
    
    # Processing scope
    variables: List[str] = field(default_factory=lambda: CLIMATE_VARIABLES.copy())
    regions: List[str] = field(default_factory=lambda: REGIONS.copy())
    scenarios: List[str] = field(default_factory=lambda: CLIMATE_SCENARIOS.copy())
    
    # Time periods
    historical_start_year: int = HISTORICAL_START_YEAR
    historical_end_year: int = HISTORICAL_END_YEAR
    hybrid_start_year: int = HYBRID_START_YEAR
    hybrid_end_year: int = HYBRID_END_YEAR
    future_start_year: int = FUTURE_START_YEAR
    max_future_year: int = MAX_FUTURE_YEAR
    
    # Climate normal parameters
    min_years_for_normal: int = MIN_YEARS_FOR_CLIMATE_NORMAL
    standard_normal_years: int = STANDARD_CLIMATE_NORMAL_YEARS
    
    # Performance settings
    max_workers: int = OPTIMAL_WORKERS
    max_cores: int = MAX_CORES
    cores_per_variable: int = CORES_PER_VARIABLE
    batch_size_years: int = BATCH_SIZE_YEARS
    max_memory_per_process_gb: float = MAX_MEMORY_PER_PROCESS_GB
    
    # Processing control
    timeout_per_file: int = TIMEOUT_PER_FILE
    max_retries: int = MAX_RETRIES
    progress_interval: int = PROGRESS_INTERVAL
    memory_check_interval: int = MEMORY_CHECK_INTERVAL
    
    # I/O settings
    safe_chunks: Dict[str, int] = field(default_factory=lambda: SAFE_CHUNKS.copy())
    netcdf_engines: List[str] = field(default_factory=lambda: NETCDF_ENGINES.copy())
    
    # Monitoring settings
    progress_status_file: str = PROGRESS_STATUS_FILE
    progress_log_file: str = PROGRESS_LOG_FILE
    status_update_interval: int = STATUS_UPDATE_INTERVAL
    
    # Logging settings
    log_level: str = LOG_LEVEL
    log_format: str = LOG_FORMAT
    
    # Memory management
    garbage_collection_interval: int = GARBAGE_COLLECTION_INTERVAL
    memory_warning_threshold_gb: float = MEMORY_WARNING_THRESHOLD_GB
    memory_critical_threshold_gb: float = MEMORY_CRITICAL_THRESHOLD_GB
    
    # Skip problematic files
    corrupted_files: set = field(default_factory=lambda: CORRUPTED_FILES.copy())
    
    def __post_init__(self):
        """Post-initialization validation and environment variable overrides."""
        self._apply_environment_overrides()
        self._validate_configuration()
        self._setup_logging()
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides to configuration."""
        
        # Data directories
        if 'CLIMATE_INPUT_DIR' in os.environ:
            self.input_data_dir = os.environ['CLIMATE_INPUT_DIR']
        
        if 'CLIMATE_OUTPUT_DIR' in os.environ:
            self.output_base_dir = os.environ['CLIMATE_OUTPUT_DIR']
        
        # Performance settings
        if 'CLIMATE_MAX_WORKERS' in os.environ:
            try:
                self.max_workers = int(os.environ['CLIMATE_MAX_WORKERS'])
            except ValueError:
                logging.warning(f"Invalid CLIMATE_MAX_WORKERS value: {os.environ['CLIMATE_MAX_WORKERS']}")
        
        if 'CLIMATE_MAX_MEMORY' in os.environ:
            try:
                self.max_memory_per_process_gb = float(os.environ['CLIMATE_MAX_MEMORY'])
            except ValueError:
                logging.warning(f"Invalid CLIMATE_MAX_MEMORY value: {os.environ['CLIMATE_MAX_MEMORY']}")
        
        # Logging level
        if 'CLIMATE_LOG_LEVEL' in os.environ:
            self.log_level = os.environ['CLIMATE_LOG_LEVEL'].upper()
        
        # Variables to process
        if 'CLIMATE_VARIABLES' in os.environ:
            variables = os.environ['CLIMATE_VARIABLES'].split(',')
            self.variables = [v.strip() for v in variables if v.strip()]
        
        # Regions to process
        if 'CLIMATE_REGIONS' in os.environ:
            regions = os.environ['CLIMATE_REGIONS'].split(',')
            self.regions = [r.strip() for r in regions if r.strip()]
    
    def _validate_configuration(self):
        """Validate configuration settings."""
        
        # Skip directory validation during import to avoid circular dependencies
        # Directory validation will happen when actually using the configuration
        
        # Validate variables
        valid_variables = set(CLIMATE_VARIABLES)
        invalid_vars = set(self.variables) - valid_variables
        if invalid_vars:
            raise ValueError(f"Invalid variables specified: {invalid_vars}")
        
        # Validate regions
        valid_regions = {'CONUS', 'AK', 'HI', 'PRVI', 'GU'}
        invalid_regions = set(self.regions) - valid_regions
        if invalid_regions:
            raise ValueError(f"Invalid regions specified: {invalid_regions}")
        
        # Validate performance settings
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be positive: {self.max_workers}")
        
        if self.max_memory_per_process_gb <= 0:
            raise ValueError(f"max_memory_per_process_gb must be positive: {self.max_memory_per_process_gb}")
        
        # Validate year ranges
        if self.historical_start_year >= self.historical_end_year:
            raise ValueError("historical_start_year must be less than historical_end_year")
        
        if self.hybrid_start_year >= self.hybrid_end_year:
            raise ValueError("hybrid_start_year must be less than hybrid_end_year")
    
    def validate_directories(self):
        """Validate directories - separate method to avoid circular imports."""
        input_path = Path(self.input_data_dir)
        if not input_path.exists():
            raise ValueError(f"Input data directory does not exist: {self.input_data_dir}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        numeric_level = getattr(logging, self.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {self.log_level}')
        
        logging.basicConfig(
            level=numeric_level,
            format=self.log_format,
            force=True  # Override any existing configuration
        )
    
    def get_output_dir(self, variable: str, period_type: str) -> Path:
        """Get output directory for a specific variable and period type."""
        return Path(self.output_base_dir) / variable / period_type
    
    def get_processing_targets(self) -> Dict[str, Dict[str, int]]:
        """Get processing targets for progress tracking."""
        targets = {}
        for variable in self.variables:
            if variable in PROCESSING_TARGETS:
                targets[variable] = PROCESSING_TARGETS[variable].copy()
        return targets
    
    def is_file_corrupted(self, file_path: str) -> bool:
        """Check if a file is in the known corrupted files list."""
        return Path(file_path).name in self.corrupted_files
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'input_data_dir': self.input_data_dir,
            'output_base_dir': self.output_base_dir,
            'variables': self.variables,
            'regions': self.regions,
            'scenarios': self.scenarios,
            'max_workers': self.max_workers,
            'max_memory_per_process_gb': self.max_memory_per_process_gb,
            'log_level': self.log_level
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClimateProcessingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


def get_default_config() -> ClimateProcessingConfig:
    """Get default configuration with environment variable overrides."""
    return ClimateProcessingConfig()


def get_production_config() -> ClimateProcessingConfig:
    """Get production configuration with conservative settings."""
    config = ClimateProcessingConfig()
    
    # More conservative settings for production
    config.max_workers = min(config.max_workers, 4)
    config.max_memory_per_process_gb = min(config.max_memory_per_process_gb, 3.0)
    config.timeout_per_file = 600  # 10 minutes for production
    config.max_retries = 3
    config.log_level = 'INFO'
    
    return config


def get_development_config() -> ClimateProcessingConfig:
    """Get development configuration with debug settings."""
    config = ClimateProcessingConfig()
    
    # Development settings
    config.max_workers = 2  # Fewer workers for development
    config.variables = ['pr']  # Just one variable for testing
    config.log_level = 'DEBUG'
    config.progress_interval = 1  # More frequent progress updates
    
    return config


def get_testing_config() -> ClimateProcessingConfig:
    """Get configuration for testing with minimal processing."""
    config = ClimateProcessingConfig()
    
    # Testing settings
    config.max_workers = 1
    config.variables = ['pr']
    config.historical_start_year = 2010
    config.historical_end_year = 2012  # Only 3 years for testing
    config.hybrid_start_year = 2015
    config.hybrid_end_year = 2016  # Only 2 years for testing
    config.min_years_for_normal = 2  # Reduced for testing
    config.log_level = 'DEBUG'
    
    return config 