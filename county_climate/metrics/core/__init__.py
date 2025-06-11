"""
Core climate metrics processing functions.

This module provides the main API for processing climate data by variable type.
Each function processes a specific climate variable across the full time series
(1980-2100) and calculates county-level metrics.

Available climate variables:
- pr: Precipitation 
- tas: Near-surface air temperature
- tasmin: Minimum near-surface air temperature
- tasmax: Maximum near-surface air temperature

Example usage:
    >>> from metrics.core import process_pr, process_tas, process_all_variables
    >>> pr_results = process_pr(scenario="historical", region="CONUS")
    >>> temp_results = process_tas(scenario="ssp245", region="AK")
    >>> all_results = process_all_variables(scenario="ssp245", region="CONUS")
"""

import os
from typing import Dict, List, Optional
from .precipitation import process_pr
from .temperature import process_tas, process_tasmin, process_tasmax


def process_all_variables(scenario: str = "historical",
                         region: str = "CONUS", 
                         output_dir: str = "output",
                         variables: Optional[List[str]] = None,
                         include_extremes: bool = True,
                         counties_shapefile: Optional[str] = None) -> Dict[str, str]:
    """
    Process all climate variables for a specific scenario and region.
    
    This is a convenience function that runs all individual processing functions
    to generate a complete set of county-level climate metrics.
    
    Args:
        scenario (str): Climate scenario ("historical", "hybrid", "ssp245")
        region (str): Geographic region ("CONUS", "AK", "HI", "PRVI", "GU")
        output_dir (str): Directory for output files
        variables (List[str], optional): Specific variables to process. 
                                       If None, processes all: ["pr", "tas", "tasmin", "tasmax"]
        include_extremes (bool): Whether to include extreme metrics (applies to pr and tas)
        counties_shapefile (str, optional): Path to custom county shapefile
        
    Returns:
        Dict[str, str]: Dictionary mapping variable names to output file paths
        
    Example:
        >>> # Process all variables
        >>> results = process_all_variables(scenario="ssp245", region="AK")
        >>> print("Files created:")
        >>> for var, file_path in results.items():
        ...     print(f"  {var}: {file_path}")
        
        >>> # Process only temperature variables
        >>> temp_results = process_all_variables(
        ...     scenario="historical", 
        ...     region="CONUS",
        ...     variables=["tas", "tasmin", "tasmax"]
        ... )
    """
    print("=" * 60)
    print("PROCESSING ALL CLIMATE VARIABLES")
    print("=" * 60)
    print(f"Scenario: {scenario}")
    print(f"Region: {region}")
    print(f"Output directory: {output_dir}")
    print(f"Include extremes: {include_extremes}")
    
    # Set default variables if not specified
    if variables is None:
        variables = ["pr", "tas", "tasmin", "tasmax"]
    
    print(f"Variables to process: {', '.join(variables)}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results
    results = {}
    
    # Process each variable
    for variable in variables:
        try:
            if variable == "pr":
                print(f"\n--- Processing variable {variable} ({variables.index(variable)+1}/{len(variables)}) ---")
                results[variable] = process_pr(
                    scenario=scenario,
                    region=region,
                    output_dir=output_dir,
                    include_extremes=include_extremes,
                    counties_shapefile=counties_shapefile
                )
                
            elif variable == "tas":
                print(f"\n--- Processing variable {variable} ({variables.index(variable)+1}/{len(variables)}) ---")
                results[variable] = process_tas(
                    scenario=scenario,
                    region=region,
                    output_dir=output_dir,
                    include_extremes=include_extremes,
                    counties_shapefile=counties_shapefile
                )
                
            elif variable == "tasmin":
                print(f"\n--- Processing variable {variable} ({variables.index(variable)+1}/{len(variables)}) ---")
                results[variable] = process_tasmin(
                    scenario=scenario,
                    region=region,
                    output_dir=output_dir,
                    counties_shapefile=counties_shapefile
                )
                
            elif variable == "tasmax":
                print(f"\n--- Processing variable {variable} ({variables.index(variable)+1}/{len(variables)}) ---")
                results[variable] = process_tasmax(
                    scenario=scenario,
                    region=region,
                    output_dir=output_dir,
                    counties_shapefile=counties_shapefile
                )
                
            else:
                print(f"WARNING: Unknown variable '{variable}' - skipping")
                continue
                
            print(f"✓ Completed {variable}")
            
        except Exception as e:
            print(f"ERROR processing {variable}: {e}")
            # Continue with other variables
            continue
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print("Output files:")
    for var, file_path in results.items():
        print(f"  {var}: {file_path}")
    
    return results


__all__ = [
    'process_pr',
    'process_tas', 
    'process_tasmin',
    'process_tasmax',
    'process_all_variables'
]
