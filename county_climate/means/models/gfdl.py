#!/usr/bin/env python3
"""
GFDL-ESM4 climate model handler implementation.

Example implementation showing how to add support for another GCM.
"""

import re
from pathlib import Path
from typing import Dict, Optional, Pattern

from .base import ClimateModelHandler, ModelConfig, ScenarioConfig


class GFDLESM4Handler(ClimateModelHandler):
    """Handler for GFDL-ESM4 climate model data."""
    
    def __init__(self, base_path: Path):
        """
        Initialize GFDL-ESM4 handler with configuration.
        
        Args:
            base_path: Base directory containing GFDL-ESM4 data
        """
        # Define GFDL-ESM4 specific configuration
        config = ModelConfig(
            model_name="Geophysical Fluid Dynamics Laboratory Earth System Model v4",
            model_id="GFDL-ESM4",
            institution="NOAA-GFDL",
            ensemble_member="r1i1p1f1",
            grid_label="gr1",
            # GFDL uses a different directory structure
            variable_dir_pattern="{scenario}/{variable}",
            filename_pattern="{variable}_day_{model_id}_{scenario}_{ensemble}_{grid}_{year}.nc",
            scenarios={
                "historical": ScenarioConfig(
                    name="historical",
                    start_year=1850,
                    end_year=2014,
                    description="Historical climate simulation"
                ),
                "ssp119": ScenarioConfig(
                    name="ssp119",
                    start_year=2015,
                    end_year=2100,
                    description="SSP1-1.9 - Very low emissions scenario",
                    parent_scenario="historical"
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
                "ssp460": ScenarioConfig(
                    name="ssp460",
                    start_year=2015,
                    end_year=2100,
                    description="SSP4-6.0 - Inequality scenario",
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
            resolution="100km",
            notes="GFDL Earth System Model version 4"
        )
        
        super().__init__(base_path, config)
        
        # Compile regex patterns for GFDL-ESM4
        self._filename_pattern = None
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        # GFDL might use slightly different conventions
        # Pattern: variable_day_GFDL-ESM4_scenario_r1i1p1f1_gr1_YYYY.nc
        pattern = (
            r'^(?P<variable>\w+)_'
            r'(?P<frequency>day)_'
            r'(?P<model_id>GFDL-ESM4)_'
            r'(?P<scenario>\w+)_'
            r'(?P<ensemble>r\d+i\d+p\d+f\d+)_'
            r'(?P<grid>\w+)_'
            r'(?P<year>\d{4})'
            r'\.nc$'
        )
        self._filename_pattern = re.compile(pattern)
    
    def get_filename_pattern(self) -> Pattern:
        """Get the compiled regex pattern for GFDL-ESM4 files."""
        return self._filename_pattern
    
    def extract_year_from_filename(self, filename: str) -> Optional[int]:
        """Extract year from GFDL-ESM4 filename."""
        match = self._filename_pattern.match(filename)
        if match:
            return int(match.group('year'))
        
        # Fallback pattern
        match = re.search(r'_(\d{4})\.nc$', filename)
        if match:
            return int(match.group(1))
        
        return None
    
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """Extract all metadata from GFDL-ESM4 filename."""
        metadata = {}
        
        match = self._filename_pattern.match(filename)
        if match:
            metadata = match.groupdict()
            # Convert year to int
            if 'year' in metadata:
                metadata['year'] = int(metadata['year'])
        
        return metadata