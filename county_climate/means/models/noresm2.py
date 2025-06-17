#!/usr/bin/env python3
"""
NorESM2-LM climate model handler implementation.

Handles the specific file structure and naming conventions for NorESM2-LM data.
"""

import re
from pathlib import Path
from typing import Dict, Optional, Pattern

from .base import ClimateModelHandler, ModelConfig, ScenarioConfig


class NorESM2Handler(ClimateModelHandler):
    """Handler for NorESM2-LM climate model data."""
    
    def __init__(self, base_path: Path):
        """
        Initialize NorESM2-LM handler with default configuration.
        
        Args:
            base_path: Base directory containing NorESM2-LM data
        """
        # Define NorESM2-LM specific configuration
        config = ModelConfig(
            model_name="Norwegian Earth System Model version 2 - Low resolution",
            model_id="NorESM2-LM",
            institution="NCC",
            ensemble_member="r1i1p1f1",
            grid_label="gn",
            variable_dir_pattern="{variable}/{scenario}",
            filename_pattern="{variable}_day_{model_id}_{scenario}_{ensemble}_{grid}_{year}.nc",
            scenarios={
                "historical": ScenarioConfig(
                    name="historical",
                    start_year=1850,
                    end_year=2014,
                    description="Historical climate simulation"
                ),
                "ssp126": ScenarioConfig(
                    name="ssp126",
                    start_year=2015,
                    end_year=2100,
                    description="SSP1-2.6 - Sustainability scenario",
                    parent_scenario="historical"
                ),
                "ssp245": ScenarioConfig(
                    name="ssp245",
                    start_year=2015,
                    end_year=2100,
                    description="SSP2-4.5 - Middle of the road scenario",
                    parent_scenario="historical"
                ),
                "ssp370": ScenarioConfig(
                    name="ssp370",
                    start_year=2015,
                    end_year=2100,
                    description="SSP3-7.0 - Regional rivalry scenario",
                    parent_scenario="historical"
                ),
                "ssp585": ScenarioConfig(
                    name="ssp585",
                    start_year=2015,
                    end_year=2100,
                    description="SSP5-8.5 - Fossil-fueled development scenario",
                    parent_scenario="historical"
                )
            },
            resolution="250km",
            notes="Low resolution version of NorESM2"
        )
        
        super().__init__(base_path, config)
        
        # Compile regex patterns for NorESM2-LM
        self._filename_pattern = None
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        # Pattern: variable_day_NorESM2-LM_scenario_r1i1p1f1_gn_YYYY.nc
        # Also handle version suffixes: _YYYY_v1.1.nc
        pattern = (
            r'^(?P<variable>\w+)_'
            r'(?P<frequency>day)_'
            r'(?P<model_id>NorESM2-LM)_'
            r'(?P<scenario>\w+)_'
            r'(?P<ensemble>r\d+i\d+p\d+f\d+)_'
            r'(?P<grid>\w+)_'
            r'(?P<year>\d{4})'
            r'(?:_v(?P<version>[\d.]+))?'
            r'\.nc$'
        )
        self._filename_pattern = re.compile(pattern)
    
    def get_filename_pattern(self) -> Pattern:
        """Get the compiled regex pattern for NorESM2-LM files."""
        return self._filename_pattern
    
    def extract_year_from_filename(self, filename: str) -> Optional[int]:
        """
        Extract year from NorESM2-LM filename.
        
        Expected formats:
        - pr_day_NorESM2-LM_historical_r1i1p1f1_gn_1980.nc
        - tas_day_NorESM2-LM_ssp245_r1i1p1f1_gn_2050_v1.1.nc
        """
        match = self._filename_pattern.match(filename)
        if match:
            return int(match.group('year'))
        
        # Fallback to simple pattern
        match = re.search(r'_(\d{4})(?:_v[\d.]+)?\.nc$', filename)
        if match:
            return int(match.group(1))
        
        return None
    
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """Extract all metadata from NorESM2-LM filename."""
        metadata = {}
        
        match = self._filename_pattern.match(filename)
        if match:
            metadata = match.groupdict()
            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}
            # Convert year to int for consistency
            if 'year' in metadata:
                metadata['year'] = int(metadata['year'])
        
        return metadata
    
    def validate_filename(self, filename: str) -> bool:
        """Check if a filename matches NorESM2-LM conventions."""
        return bool(self._filename_pattern.match(filename))
    
    def construct_filename(self, variable: str, scenario: str, year: int,
                         ensemble: str = None, version: str = None) -> str:
        """
        Construct a NorESM2-LM filename from components.
        
        Args:
            variable: Climate variable
            scenario: Climate scenario
            year: Year
            ensemble: Ensemble member (defaults to r1i1p1f1)
            version: Optional version string
            
        Returns:
            Constructed filename
        """
        if ensemble is None:
            ensemble = self.config.ensemble_member
        
        filename = (
            f"{variable}_day_{self.config.model_id}_{scenario}_"
            f"{ensemble}_{self.config.grid_label}_{year}"
        )
        
        if version:
            filename += f"_v{version}"
        
        filename += ".nc"
        return filename