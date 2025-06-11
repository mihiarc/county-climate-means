"""
File discovery utilities for climate data processing.

Handles finding and organizing climate data files across different regions,
scenarios, and time periods. Supports the complex file naming patterns
used in climate data archives.
"""

import os
import glob
import re
from typing import List, Tuple, Dict, Optional
from .region_config import REGION_CONFIG, SCENARIO_CONFIG, get_region_info, get_scenario_info


def get_file_list(region_key: str, variable: str, base_path: Optional[str] = None) -> List[Tuple[int, str, str]]:
    """
    Get list of all climate files organized by year and scenario for a specific region and variable.
    
    Args:
        region_key (str): Region identifier ('CONUS', 'AK', 'HI', 'PRVI', 'GU')
        variable (str): Climate variable ('pr', 'tas', 'tasmin', 'tasmax')
        base_path (str, optional): Override base path for data files
        
    Returns:
        List[Tuple[int, str, str]]: List of tuples (year, scenario, filepath)
        
    Raises:
        ValueError: If region_key is not valid
    """
    region_config = get_region_info(region_key)
    
    # Use provided base_path or default from region config
    if base_path is None:
        data_base_path = region_config['data_path_template']
    else:
        # If base_path is provided, construct the region-specific path
        # Use the actual region directory names that exist
        data_base_path = os.path.join(base_path, 'data', region_key)
    
    files = []
    
    # The structure is: {data_base_path}/{variable}/*.nc (files are all in the variable directory)
    variable_path = os.path.join(data_base_path, variable)
    
    if not os.path.exists(variable_path):
        print(f"  Warning: Variable path not found: {variable_path}")
        return files
    
    # Find all NetCDF files in the variable directory
    file_pattern = "*.nc"
    full_pattern = os.path.join(variable_path, file_pattern)
    all_files = glob.glob(full_pattern)
    
    # Process each file and determine its scenario based on filename or year
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # Extract year from filename
        year = extract_year_from_filename(file_path, "")  # Pass empty scenario for now
        if year is None:
            continue
            
        # Determine scenario - prioritize filename-based detection over year-based
        scenario_key = None
        
        # First, try to extract scenario from filename (more reliable)
        for scen_key in SCENARIO_CONFIG.keys():
            if scen_key in filename:
                scenario_key = scen_key
                break
                
        # If no scenario found in filename, fall back to year-based detection
        if scenario_key is None:
            for scen_key, scen_config in SCENARIO_CONFIG.items():
                year_range = scen_config['year_range']
                if year_range[0] <= year <= year_range[1]:
                    scenario_key = scen_key
                    break
        
        if scenario_key is None:
            print(f"  Warning: Could not determine scenario for {filename}")
            continue
        
        files.append((year, scenario_key, file_path))
    
    # Sort by year for consistent processing order
    files.sort(key=lambda x: x[0])
    
    print(f"  Found {len(files)} files for {region_key} {variable}")
    return files


def extract_year_from_filename(filepath: str, scenario: str) -> Optional[int]:
    """
    Extract year from climate data filename.
    
    Handles various filename patterns used across different scenarios and regions.
    
    Args:
        filepath (str): Full path to the climate data file
        scenario (str): Climate scenario ('historical', 'hybrid', 'ssp245') - can be empty
        
    Returns:
        Optional[int]: Extracted year, or None if not found
    """
    filename = os.path.basename(filepath)
    
    # Patterns for actual file formats we see:
    # pr_CONUS_pr_day_NorESM2-LM_historical_r1i1p1f1_gn_1950_v1.1_climatology.nc
    # pr_CONUS_pr_day_NorESM2-LM_ssp245_r1i1p1f1_gn_2020_v1.1_climatology.nc
    
    patterns = [
        # Primary pattern: extract year from the standard climatology filename
        r'_(\d{4})_v[\d\.]+_climatology\.nc$',
        
        # Alternative pattern if scenario is provided
        rf'{scenario}_.*?_(\d{{4}})_.*?\.nc' if scenario else None,
        
        # Fallback pattern: just look for 4-digit year followed by version
        r'_(\d{4})_v',
        
        # Final fallback: any 4-digit year
        r'(\d{4})',
    ]
    
    # Filter out None patterns
    patterns = [p for p in patterns if p is not None]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                year = int(match.group(1))
                # Basic validation - climate data should be reasonable years
                if 1900 <= year <= 2200:
                    return year
            except (ValueError, IndexError):
                continue
    
    return None


def discover_available_data(base_path: str, region_key: Optional[str] = None, 
                          variable: Optional[str] = None) -> Dict[str, Dict[str, List[int]]]:
    """
    Discover all available climate data files and organize by region and variable.
    
    Args:
        base_path (str): Base directory containing climate data
        region_key (str, optional): Specific region to check (default: all regions)
        variable (str, optional): Specific variable to check (default: all variables)
        
    Returns:
        Dict[str, Dict[str, List[int]]]: Nested dictionary structure:
            {region: {variable: [list_of_years]}}
    """
    available_data = {}
    
    # Determine which regions to check
    regions_to_check = [region_key] if region_key else list(REGION_CONFIG.keys())
    
    # Determine which variables to check
    variables_to_check = [variable] if variable else ['pr', 'tas', 'tasmin', 'tasmax']
    
    for region in regions_to_check:
        available_data[region] = {}
        
        for var in variables_to_check:
            try:
                # Get file list for this region/variable combination
                file_list = get_file_list(region, var, base_path)
                
                # Extract just the years
                years = [year for year, scenario, filepath in file_list]
                available_data[region][var] = sorted(list(set(years)))  # Remove duplicates and sort
                
            except Exception as e:
                print(f"  Warning: Error checking {region} {var}: {e}")
                available_data[region][var] = []
    
    return available_data


def validate_file_availability(region_key: str, variable: str, year: int, 
                             base_path: Optional[str] = None) -> Optional[str]:
    """
    Check if a specific climate data file exists and return its path.
    
    Args:
        region_key (str): Region identifier
        variable (str): Climate variable
        year (int): Year to check
        base_path (str, optional): Override base path
        
    Returns:
        Optional[str]: File path if found, None otherwise
    """
    file_list = get_file_list(region_key, variable, base_path)
    
    for file_year, scenario, filepath in file_list:
        if file_year == year:
            if os.path.exists(filepath):
                return filepath
            else:
                print(f"  Warning: File listed but not found: {filepath}")
                return None
    
    return None


def get_scenario_for_year(year: int) -> Optional[str]:
    """
    Determine which climate scenario a given year belongs to.
    
    Args:
        year (int): Year to check
        
    Returns:
        Optional[str]: Scenario name ('historical', 'hybrid', 'ssp245') or None
    """
    for scenario_key, scenario_config in SCENARIO_CONFIG.items():
        year_range = scenario_config['year_range']
        if year_range[0] <= year <= year_range[1]:
            return scenario_key
    
    return None


def get_years_for_scenario(scenario: str) -> Tuple[int, int]:
    """
    Get the year range for a specific scenario.
    
    Args:
        scenario (str): Scenario identifier
        
    Returns:
        Tuple[int, int]: (start_year, end_year)
        
    Raises:
        ValueError: If scenario is not valid
    """
    scenario_config = get_scenario_info(scenario)
    return scenario_config['year_range']


def build_file_path(region_key: str, variable: str, year: int, scenario: str, 
                   base_path: Optional[str] = None) -> str:
    """
    Build expected file path for a specific region, variable, year, and scenario.
    
    Args:
        region_key (str): Region identifier
        variable (str): Climate variable
        year (int): Year
        scenario (str): Climate scenario
        base_path (str, optional): Override base path
        
    Returns:
        str: Expected file path
    """
    region_config = get_region_info(region_key)
    
    # Use provided base_path or default from region config
    if base_path is None:
        base_path = region_config['data_path_template']
    
    # Build filename using standard pattern
    filename = f"{variable}_{region_key}_{scenario}_{year}_30yr_normal.nc"
    
    # Combine with scenario subdirectory
    file_path = os.path.join(base_path, scenario, filename)
    
    return file_path


def print_data_availability_summary(base_path: str):
    """
    Print a comprehensive summary of available climate data.
    
    Args:
        base_path (str): Base directory containing climate data
    """
    print("=== CLIMATE DATA AVAILABILITY SUMMARY ===")
    print(f"Base path: {base_path}")
    print()
    
    available_data = discover_available_data(base_path)
    
    for region_key, region_data in available_data.items():
        region_config = get_region_info(region_key)
        print(f"{region_key}: {region_config['name']}")
        
        for variable, years in region_data.items():
            if years:
                year_range = f"{min(years)}-{max(years)}"
                print(f"  {variable}: {len(years)} files ({year_range})")
            else:
                print(f"  {variable}: No files found")
        print()


def get_missing_files(region_key: str, variable: str, base_path: Optional[str] = None) -> List[Tuple[int, str]]:
    """
    Identify missing files for a region/variable combination across all expected years.
    
    Args:
        region_key (str): Region identifier
        variable (str): Climate variable
        base_path (str, optional): Override base path
        
    Returns:
        List[Tuple[int, str]]: List of (year, scenario) tuples for missing files
    """
    # Get all files that should exist
    expected_files = []
    for scenario_key, scenario_config in SCENARIO_CONFIG.items():
        start_year, end_year = scenario_config['year_range']
        for year in range(start_year, end_year + 1):
            expected_files.append((year, scenario_key))
    
    # Get files that actually exist
    existing_files = get_file_list(region_key, variable, base_path)
    existing_years_scenarios = {(year, scenario) for year, scenario, filepath in existing_files}
    
    # Find missing files
    missing_files = []
    for year, scenario in expected_files:
        if (year, scenario) not in existing_years_scenarios:
            missing_files.append((year, scenario))
    
    return missing_files 