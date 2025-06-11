"""
Climate data configuration - Integrated with Climate Means Catalog

This configuration was automatically generated to integrate the county_climate_metrics
package with the climate means processing pipeline catalog system.

Generated from catalog: output/catalog/climate_means_catalog.yaml
"""

import os
import glob
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import the catalog bridge adapter
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from means.pipeline.catalog_bridge import CatalogToBridgeAdapter

# Initialize catalog adapter
CATALOG_PATH = str(Path(__file__).parent.parent.parent.parent / "output/catalog/climate_means_catalog.yaml")
_catalog_adapter = CatalogToBridgeAdapter(CATALOG_PATH)

# Data configuration - derived from climate means catalog
DATA_CONFIG = {
    'base_path': 'output',
    'regions': {'CONUS': {'name': 'Continental United States', 'counties': 3114, 'path': 'CONUS'}, 'AK': {'name': 'Alaska', 'counties': 30, 'path': 'AK'}},
    'variables': ['pr', 'tas', 'tasmax', 'tasmin'],
    'scenarios': {'historical': {'name': 'HISTORICAL', 'pattern': '*_historical_*', 'years': range(1980, 2015)}, 'ssp245': {'name': 'SSP245', 'pattern': '*_ssp245_*', 'years': range(2015, 2101)}}
}


def get_data_files(region: str, variable: str, base_path: Optional[str] = None) -> List[Tuple[int, str, str]]:
    """
    Get list of data files for a region/variable combination from climate means catalog.
    
    Args:
        region: Region code (CONUS, AK, HI, PRVI, GU)
        variable: Variable name (pr, tas, tasmin, tasmax)  
        base_path: Ignored - uses catalog directly
        
    Returns:
        List of (year, scenario, filepath) tuples
    """
    return _catalog_adapter.get_data_files(region, variable, base_path)


def filter_files_by_scenario(files: List[Tuple[int, str, str]], scenario: str) -> List[Tuple[int, str, str]]:
    """Filter file list to specific scenario."""
    return _catalog_adapter.filter_files_by_scenario(files, scenario)


def filter_files_by_year_range(files: List[Tuple[int, str, str]], start_year: int, end_year: int) -> List[Tuple[int, str, str]]:
    """Filter file list to specific year range."""
    return _catalog_adapter.filter_files_by_year_range(files, start_year, end_year)


def get_available_years(region: str, variable: str, scenario: Optional[str] = None) -> List[int]:
    """Get list of available years for region/variable/scenario."""
    return _catalog_adapter.get_available_years(region, variable, scenario)


def validate_data_availability(region: str, variable: str, scenario: str, year: int) -> bool:
    """Check if specific data file exists."""
    return _catalog_adapter.validate_data_availability(region, variable, scenario, year)


def get_data_summary() -> Dict:
    """Get summary of all available data."""
    return _catalog_adapter.get_data_summary()


def print_data_summary():
    """Print comprehensive data availability summary."""
    _catalog_adapter.print_data_summary()
