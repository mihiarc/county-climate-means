"""
Regional configuration for multi-region climate data processing.

Defines configuration for processing climate data across all U.S. regions:
- CONUS: Continental United States
- AK: Alaska  
- HI: Hawaii
- PRVI: Puerto Rico and Virgin Islands
- GU: Guam and Northern Mariana Islands

Each region has different coordinate reference systems, file naming patterns,
and data availability.
"""

import geopandas as gpd
import re
from typing import Dict, List, Optional


# Regional configuration mapping
REGION_CONFIG = {
    'CONUS': {
        'name': 'Continental United States',
        'file_pattern': '{variable}_CONUS_*_30yr_normal.nc',
        'data_path_template': '/home/mihiarc/repos/county_climate_means/output/data/CONUS',
        'has_precipitation': True,
        'has_temperature': True,
        'state_filter': None,  # All continental states
        'fips_pattern': None,   # No specific FIPS filtering
        'excluded_states': ['02', '15', '72', '78', '66', '69']  # AK, HI, PR, VI, GU, MP
    },
    'AK': {
        'name': 'Alaska',
        'file_pattern': '{variable}_AK_*_30yr_normal.nc',
        'data_path_template': '/home/mihiarc/repos/county_climate_means/output/data/AK',
        'has_precipitation': True,
        'has_temperature': True,
        'state_filter': ['02'],  # Alaska state FIPS code
        'fips_pattern': r'^02\d{3}$',  # Alaska counties start with 02
        'excluded_states': None
    },
    'HI': {
        'name': 'Hawaii', 
        'file_pattern': '{variable}_HI_*_30yr_normal.nc',
        'data_path_template': '/home/mihiarc/repos/county_climate_means/output/data/HI',
        'has_precipitation': True,  # Complete precipitation data available
        'has_temperature': True,
        'state_filter': ['15'],  # Hawaii state FIPS code
        'fips_pattern': r'^15\d{3}$',  # Hawaii counties start with 15
        'excluded_states': None
    },
    'PRVI': {
        'name': 'Puerto Rico and Virgin Islands',
        'file_pattern': '{variable}_PRVI_*_30yr_normal.nc', 
        'data_path_template': '/home/mihiarc/repos/county_climate_means/output/data/PRVI',
        'has_precipitation': True,  # Complete precipitation data available
        'has_temperature': True,
        'state_filter': ['72', '78'],  # PR=72, VI=78
        'fips_pattern': r'^(72|78)\d{3}$',  # PR/VI counties
        'excluded_states': None
    },
    'GU': {
        'name': 'Guam and Northern Mariana Islands',
        'file_pattern': '{variable}_GU_*_30yr_normal.nc',
        'data_path_template': '/home/mihiarc/repos/county_climate_means/output/data/GU',
        'has_precipitation': True,  # Complete precipitation data available
        'has_temperature': True,
        'state_filter': ['66', '69'],  # GU=66, MP=69
        'fips_pattern': r'^(66|69)\d{3}$',  # GU/MP counties
        'excluded_states': None
    }
}

# Climate variable configuration
VARIABLE_CONFIG = {
    'pr': {
        'name': 'Precipitation',
        'long_name': 'Daily Precipitation',
        'units_input': 'kg m^(-2) s^(-1)',
        'units_output': 'mm/day',
        'conversion_factor': 86400,  # Convert from kg/m²/s to mm/day
        'available_regions': ['CONUS', 'AK', 'HI', 'PRVI', 'GU']  # All regions have complete precipitation data
    },
    'tas': {
        'name': 'Mean Temperature',
        'long_name': 'Near-Surface Air Temperature',
        'units_input': 'K',
        'units_output': '°C',
        'conversion_factor': -273.15,  # Convert from Kelvin to Celsius
        'available_regions': ['CONUS', 'AK', 'HI', 'PRVI', 'GU']
    },
    'tasmin': {
        'name': 'Minimum Temperature',
        'long_name': 'Minimum Near-Surface Air Temperature',
        'units_input': 'K',
        'units_output': '°C',
        'conversion_factor': -273.15,
        'available_regions': ['CONUS', 'AK', 'HI', 'PRVI', 'GU']
    },
    'tasmax': {
        'name': 'Maximum Temperature',
        'long_name': 'Maximum Near-Surface Air Temperature',
        'units_input': 'K',
        'units_output': '°C',
        'conversion_factor': -273.15,
        'available_regions': ['CONUS', 'AK', 'HI', 'PRVI', 'GU']
    }
}

# Climate scenarios configuration
SCENARIO_CONFIG = {
    'historical': {
        'name': 'Historical',
        'description': 'Historical observational data',
        'year_range': (1950, 2014),
        'file_pattern': '*_historical_*_climatology.nc'
    },
    'ssp245': {
        'name': 'SSP245',
        'description': 'Climate projection scenario 2.45 W/m²',
        'year_range': (2015, 2100),
        'file_pattern': '*_ssp245_*_climatology.nc'
    },
    'ssp585': {
        'name': 'SSP585', 
        'description': 'Climate projection scenario 5.85 W/m²',
        'year_range': (2015, 2100),
        'file_pattern': '*_ssp585_*_climatology.nc'
    }
}


def get_available_regions(variable: str = None) -> List[str]:
    """
    Get list of available regions, optionally filtered by variable availability.
    
    Args:
        variable (str, optional): Climate variable to filter by ('pr', 'tas', 'tasmin', 'tasmax')
        
    Returns:
        List[str]: List of available region keys
    """
    if variable is None:
        return list(REGION_CONFIG.keys())
    
    if variable not in VARIABLE_CONFIG:
        raise ValueError(f"Unknown variable '{variable}'. Available: {list(VARIABLE_CONFIG.keys())}")
    
    return VARIABLE_CONFIG[variable]['available_regions']


def get_region_info(region_key: str) -> Dict:
    """
    Get configuration information for a specific region.
    
    Args:
        region_key (str): Region identifier ('CONUS', 'AK', 'HI', 'PRVI', 'GU')
        
    Returns:
        Dict: Region configuration
        
    Raises:
        ValueError: If region_key is not valid
    """
    if region_key not in REGION_CONFIG:
        raise ValueError(f"Unknown region '{region_key}'. Available: {list(REGION_CONFIG.keys())}")
    
    return REGION_CONFIG[region_key].copy()


def get_variable_info(variable: str) -> Dict:
    """
    Get configuration information for a specific climate variable.
    
    Args:
        variable (str): Variable identifier ('pr', 'tas', 'tasmin', 'tasmax')
        
    Returns:
        Dict: Variable configuration
        
    Raises:
        ValueError: If variable is not valid
    """
    if variable not in VARIABLE_CONFIG:
        raise ValueError(f"Unknown variable '{variable}'. Available: {list(VARIABLE_CONFIG.keys())}")
    
    return VARIABLE_CONFIG[variable].copy()


def get_scenario_info(scenario: str) -> Dict:
    """
    Get configuration information for a specific climate scenario.
    
    Args:
        scenario (str): Scenario identifier ('historical', 'hybrid', 'ssp245')
        
    Returns:
        Dict: Scenario configuration
        
    Raises:
        ValueError: If scenario is not valid
    """
    if scenario not in SCENARIO_CONFIG:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(SCENARIO_CONFIG.keys())}")
    
    return SCENARIO_CONFIG[scenario].copy()


def filter_counties_by_region(counties_gdf: gpd.GeoDataFrame, region_key: str) -> gpd.GeoDataFrame:
    """
    Filter county boundaries to include only counties from the specified region.
    
    Args:
        counties_gdf (geopandas.GeoDataFrame): All county boundaries
        region_key (str): Region identifier ('CONUS', 'AK', 'HI', 'PRVI', 'GU')
        
    Returns:
        geopandas.GeoDataFrame: Filtered counties for the region
        
    Raises:
        ValueError: If region_key is not valid
    """
    if region_key not in REGION_CONFIG:
        raise ValueError(f"Unknown region '{region_key}'. Available: {list(REGION_CONFIG.keys())}")
    
    config = REGION_CONFIG[region_key]
    
    if region_key == 'CONUS':
        # CONUS includes all states except territories
        excluded_states = config['excluded_states']
        region_counties = counties_gdf[~counties_gdf['GEOID'].str[:2].isin(excluded_states)]
    else:
        # Filter by FIPS pattern for territories and states
        fips_pattern = config['fips_pattern'] 
        region_counties = counties_gdf[counties_gdf['GEOID'].str.match(fips_pattern)]
    
    print(f"  Region {region_key} ({config['name']}): {len(region_counties)} counties")
    return region_counties


def validate_region_variable_combination(region_key: str, variable: str) -> bool:
    """
    Check if a region has data available for a specific variable.
    
    Args:
        region_key (str): Region identifier
        variable (str): Variable identifier
        
    Returns:
        bool: True if combination is valid, False otherwise
        
    Raises:
        ValueError: If region_key or variable is not valid
    """
    region_info = get_region_info(region_key)
    variable_info = get_variable_info(variable)
    
    # Check if region is in the variable's available regions
    if region_key not in variable_info['available_regions']:
        return False
    
    # Check region-specific availability flags
    if variable == 'pr' and not region_info.get('has_precipitation', False):
        return False
    
    if variable in ['tas', 'tasmin', 'tasmax'] and not region_info.get('has_temperature', False):
        return False
    
    return True


def get_valid_combinations() -> Dict[str, List[str]]:
    """
    Get all valid region-variable combinations.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping variables to their available regions
    """
    combinations = {}
    
    for variable in VARIABLE_CONFIG.keys():
        valid_regions = []
        for region in REGION_CONFIG.keys():
            if validate_region_variable_combination(region, variable):
                valid_regions.append(region)
        combinations[variable] = valid_regions
    
    return combinations


def print_region_summary():
    """Print a summary of all regions and their data availability."""
    print("=== REGIONAL DATA AVAILABILITY SUMMARY ===")
    print()
    
    for region_key, config in REGION_CONFIG.items():
        print(f"{region_key}: {config['name']}")
        print(f"  Precipitation: {'✓' if config.get('has_precipitation', False) else '✗'}")
        print(f"  Temperature: {'✓' if config.get('has_temperature', False) else '✗'}")
        
        if config.get('state_filter'):
            print(f"  State FIPS: {', '.join(config['state_filter'])}")
        if config.get('fips_pattern'):
            print(f"  FIPS Pattern: {config['fips_pattern']}")
        print()


def print_variable_summary():
    """Print a summary of all variables and their available regions."""
    print("=== VARIABLE AVAILABILITY SUMMARY ===")
    print()
    
    for variable, config in VARIABLE_CONFIG.items():
        print(f"{variable}: {config['name']}")
        print(f"  Available in: {', '.join(config['available_regions'])}")
        print(f"  Units: {config['units_input']} → {config['units_output']}")
        print() 