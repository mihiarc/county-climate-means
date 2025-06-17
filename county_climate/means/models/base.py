#!/usr/bin/env python3
"""
Abstract base class for climate model handlers.

Provides a flexible interface for supporting different Global Climate Models (GCMs)
with their specific file structures, naming conventions, and scenarios.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Pattern
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Configuration for a climate scenario."""
    name: str
    start_year: int
    end_year: int
    description: str = ""
    parent_scenario: Optional[str] = None  # For scenarios that branch from others


@dataclass
class ModelConfig:
    """Configuration for a climate model."""
    model_name: str
    model_id: str  # e.g., "NorESM2-LM", "GFDL-ESM4"
    institution: str
    ensemble_member: str = "r1i1p1f1"  # Default ensemble member
    grid_label: str = "gn"  # Default grid label
    
    # File structure configuration
    variable_dir_pattern: str = "{variable}/{scenario}"  # How variables/scenarios are organized
    filename_pattern: str = "{variable}_day_{model_id}_{scenario}_{ensemble}_{grid}_{year}.nc"
    
    # Supported scenarios with their configurations
    scenarios: Dict[str, ScenarioConfig] = field(default_factory=dict)
    
    # Additional metadata
    resolution: Optional[str] = None
    notes: Optional[str] = None


class ClimateModelHandler(ABC):
    """
    Abstract base class for handling different climate models.
    
    Each climate model may have different:
    - Directory structures
    - File naming conventions
    - Available scenarios
    - Time period coverage
    - Variable naming
    """
    
    def __init__(self, base_path: Path, config: ModelConfig):
        """
        Initialize the climate model handler.
        
        Args:
            base_path: Base directory containing model data
            config: Model-specific configuration
        """
        self.base_path = Path(base_path)
        self.config = config
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate the handler setup."""
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {self.base_path}")
        
        if not self.config.scenarios:
            raise ValueError(f"No scenarios defined for model {self.config.model_id}")
    
    @abstractmethod
    def get_filename_pattern(self) -> Pattern:
        """
        Get the compiled regex pattern for matching this model's files.
        
        Returns:
            Compiled regex pattern for filename matching
        """
        pass
    
    @abstractmethod
    def extract_year_from_filename(self, filename: str) -> Optional[int]:
        """
        Extract the year from a filename.
        
        Args:
            filename: Name of the file (not full path)
            
        Returns:
            Year as integer, or None if not found
        """
        pass
    
    @abstractmethod
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """
        Extract all metadata from a filename.
        
        Args:
            filename: Name of the file (not full path)
            
        Returns:
            Dictionary with keys like 'variable', 'scenario', 'year', 'ensemble', etc.
        """
        pass
    
    def get_files_for_period(self, variable: str, scenario: str, 
                           start_year: int, end_year: int) -> List[Path]:
        """
        Get list of files for a variable, scenario, and year range.
        
        Args:
            variable: Climate variable (pr, tas, tasmax, tasmin)
            scenario: Climate scenario (historical, ssp245, ssp585, etc.)
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            
        Returns:
            List of file paths sorted by year
        """
        # Validate scenario
        if scenario not in self.config.scenarios:
            logger.warning(f"Scenario {scenario} not configured for {self.config.model_id}")
            return []
        
        # Build directory path using pattern
        rel_path = self.config.variable_dir_pattern.format(
            variable=variable,
            scenario=scenario,
            model_id=self.config.model_id
        )
        scenario_dir = self.base_path / rel_path
        
        if not scenario_dir.exists():
            logger.warning(f"Directory not found: {scenario_dir}")
            return []
        
        # Find files matching the year range
        files = []
        pattern = self.get_filename_pattern()
        
        for file_path in scenario_dir.glob("*.nc"):
            if pattern.match(file_path.name):
                year = self.extract_year_from_filename(file_path.name)
                if year and start_year <= year <= end_year:
                    files.append(file_path)
        
        # Sort by year
        files.sort(key=lambda x: self.extract_year_from_filename(x.name) or 0)
        
        logger.debug(f"Found {len(files)} files for {variable} {scenario} {start_year}-{end_year}")
        return files
    
    def get_available_years(self, variable: str, scenario: str) -> Tuple[int, int]:
        """
        Get the available year range for a variable and scenario.
        
        Args:
            variable: Climate variable
            scenario: Climate scenario
            
        Returns:
            Tuple of (start_year, end_year) or (0, 0) if no data found
        """
        scenario_dir = self.base_path / self.config.variable_dir_pattern.format(
            variable=variable,
            scenario=scenario,
            model_id=self.config.model_id
        )
        
        if not scenario_dir.exists():
            return 0, 0
        
        years = []
        pattern = self.get_filename_pattern()
        
        for file_path in scenario_dir.glob("*.nc"):
            if pattern.match(file_path.name):
                year = self.extract_year_from_filename(file_path.name)
                if year:
                    years.append(year)
        
        if years:
            return min(years), max(years)
        return 0, 0
    
    def validate_data_availability(self) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """
        Check data availability for all variables and scenarios.
        
        Returns:
            Dictionary with structure: {variable: {scenario: (start_year, end_year)}}
        """
        availability = {}
        variables = ['pr', 'tas', 'tasmax', 'tasmin']
        
        for variable in variables:
            availability[variable] = {}
            for scenario_name in self.config.scenarios:
                start_year, end_year = self.get_available_years(variable, scenario_name)
                if start_year > 0:
                    availability[variable][scenario_name] = (start_year, end_year)
                    logger.info(f"{self.config.model_id} {variable} {scenario_name}: {start_year}-{end_year}")
        
        return availability
    
    def get_hybrid_files_for_period(self, variable: str, target_year: int,
                                   projection_scenario: str = None,
                                   window_years: int = 30) -> Tuple[List[Path], Dict[str, int]]:
        """
        Get files for a period that may span historical and future scenarios.
        
        Args:
            variable: Climate variable
            target_year: End year of the period
            projection_scenario: Which projection to use (ssp245, ssp585, etc.)
            window_years: Length of the period (default 30)
            
        Returns:
            Tuple of (file_paths, scenario_counts)
        """
        start_year = target_year - window_years + 1
        all_files = []
        scenario_counts = {}
        
        # Determine transition year between historical and projections
        hist_config = self.config.scenarios.get('historical')
        if not hist_config:
            raise ValueError(f"No historical scenario configured for {self.config.model_id}")
        
        hist_end_year = hist_config.end_year
        
        # If no projection scenario specified, use the first available
        if not projection_scenario:
            # Find first projection scenario
            for scenario_name, scenario_config in self.config.scenarios.items():
                if scenario_name != 'historical' and scenario_config.start_year > hist_end_year:
                    projection_scenario = scenario_name
                    break
        
        if not projection_scenario:
            raise ValueError(f"No projection scenario found for {self.config.model_id}")
        
        # Get files year by year
        for year in range(start_year, target_year + 1):
            if year <= hist_end_year:
                # Use historical data
                hist_files = self.get_files_for_period(variable, 'historical', year, year)
                if hist_files:
                    all_files.extend(hist_files)
                    scenario_counts['historical'] = scenario_counts.get('historical', 0) + 1
            else:
                # Use projection data
                proj_files = self.get_files_for_period(variable, projection_scenario, year, year)
                if proj_files:
                    all_files.extend(proj_files)
                    scenario_counts[projection_scenario] = scenario_counts.get(projection_scenario, 0) + 1
        
        logger.info(f"Hybrid files for {variable} target {target_year}: " +
                   " + ".join([f"{count} {scenario}" for scenario, count in scenario_counts.items()]))
        
        return all_files, scenario_counts
    
    def get_scenario_config(self, scenario: str) -> Optional[ScenarioConfig]:
        """Get configuration for a specific scenario."""
        return self.config.scenarios.get(scenario)
    
    def get_supported_scenarios(self) -> List[str]:
        """Get list of supported scenario names."""
        return list(self.config.scenarios.keys())
    
    def __repr__(self) -> str:
        """String representation of the handler."""
        return f"{self.__class__.__name__}(model={self.config.model_id}, path={self.base_path})"