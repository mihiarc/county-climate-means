"""
Climate Means Catalog Bridge for County Climate Metrics Integration

This module creates a seamless bridge between the climate means catalog and
the county_climate_metrics package, allowing the metrics pipeline to consume
data directly from our scenario/year organized catalog without requiring
file structure changes.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from ..output.catalog import ClimateDataCatalog


@dataclass
class MetricsDataConfig:
    """Configuration for county metrics pipeline integration."""
    catalog_path: str
    base_path: str
    regions: Dict[str, Dict[str, Any]]
    variables: List[str]
    scenarios: Dict[str, Dict[str, Any]]


class CatalogToBridgeAdapter:
    """
    Adapter that makes climate means catalog data accessible to county metrics pipeline.
    
    This class translates between our scenario/year organized catalog and the
    region/variable file structure expected by the county_climate_metrics package.
    """
    
    def __init__(self, catalog_path: str):
        """
        Initialize adapter with catalog path.
        
        Args:
            catalog_path: Path to climate means catalog YAML file
        """
        from pathlib import Path
        self.catalog = ClimateDataCatalog(Path(catalog_path))
        self.catalog_path = catalog_path
        
        # Load catalog metadata
        with open(catalog_path, 'r') as f:
            self.catalog_metadata = yaml.safe_load(f)
    
    def get_data_files(self, region: str, variable: str, base_path: Optional[str] = None) -> List[Tuple[int, str, str]]:
        """
        Get list of data files for a region/variable combination, compatible with
        county_climate_metrics data_config.get_data_files() format.
        
        Args:
            region: Region code (CONUS, AK, HI, PRVI, GU)
            variable: Variable name (pr, tas, tasmin, tasmax)
            base_path: Ignored - we use catalog directly
            
        Returns:
            List of (year, scenario, filepath) tuples
        """
        # Find all datasets for this region and variable
        datasets = []
        for dataset in self.catalog_metadata['datasets']:
            if (dataset['region'] == region and 
                dataset['variable'] == variable):
                datasets.append(dataset)
        
        # Convert to the expected format
        results = []
        for dataset in datasets:
            year = dataset['target_year']
            scenario = dataset['scenario']
            catalog_filepath = dataset['file_path']
            
            # Map catalog paths to actual file structure
            # Catalog has: output/data/{scenario}/{year}/{region}/filename.nc
            # Actual has: output/data/{region}/{variable}/filename.nc
            
            # Get the project root
            catalog_dir = os.path.dirname(self.catalog_path)
            project_root = os.path.dirname(catalog_dir)
            
            # Build the actual file path based on the real structure
            # Look for files matching the pattern in the actual location
            import glob
            actual_data_dir = os.path.join(project_root, 'output', 'data', region, variable)
            
            # Build a pattern to match files for this year and scenario
            # The actual files have complex names like:
            # pr_CONUS_pr_day_NorESM2-LM_historical_r1i1p1f1_gn_1950_v1.1_climatology.nc
            pattern = f"{variable}_{region}_{variable}_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}_*.nc"
            full_pattern = os.path.join(actual_data_dir, pattern)
            
            matching_files = glob.glob(full_pattern)
            
            if matching_files:
                # Use the first matching file
                filepath = matching_files[0]
                results.append((year, scenario, filepath))
            else:
                print(f"Warning: No files found matching pattern: {full_pattern}")
        
        # Sort by year for consistent processing
        results.sort(key=lambda x: x[0])
        return results
    
    def filter_files_by_scenario(self, files: List[Tuple[int, str, str]], scenario: str) -> List[Tuple[int, str, str]]:
        """Filter file list to specific scenario."""
        return [(year, scen, path) for year, scen, path in files if scen == scenario]
    
    def filter_files_by_year_range(self, files: List[Tuple[int, str, str]], 
                                  start_year: int, end_year: int) -> List[Tuple[int, str, str]]:
        """Filter file list to specific year range."""
        return [(year, scen, path) for year, scen, path in files if start_year <= year <= end_year]
    
    def get_available_years(self, region: str, variable: str, scenario: Optional[str] = None) -> List[int]:
        """Get list of available years for region/variable/scenario."""
        files = self.get_data_files(region, variable)
        
        if scenario:
            files = self.filter_files_by_scenario(files, scenario)
        
        return sorted(list(set(year for year, _, _ in files)))
    
    def validate_data_availability(self, region: str, variable: str, scenario: str, year: int) -> bool:
        """Check if specific data file exists."""
        files = self.get_data_files(region, variable)
        
        for file_year, file_scenario, filepath in files:
            if file_year == year and file_scenario == scenario:
                return os.path.exists(filepath)
        
        return False
    
    def get_data_summary(self) -> Dict:
        """Get summary of all available data."""
        # Extract summary from catalog metadata
        summary = {
            'regions': len(set(ds['region'] for ds in self.catalog_metadata['datasets'])),
            'variables': len(set(ds['variable'] for ds in self.catalog_metadata['datasets'])),
            'scenarios': len(set(ds['scenario'] for ds in self.catalog_metadata['datasets'])),
            'details': {}
        }
        
        # Group by region and variable
        for dataset in self.catalog_metadata['datasets']:
            region = dataset['region']
            variable = dataset['variable']
            scenario = dataset['scenario']
            year = dataset['target_year']
            
            if region not in summary['details']:
                summary['details'][region] = {}
            if variable not in summary['details'][region]:
                summary['details'][region][variable] = {
                    'total_files': 0,
                    'scenarios': set(),
                    'years': []
                }
            
            summary['details'][region][variable]['total_files'] += 1
            summary['details'][region][variable]['scenarios'].add(scenario)
            summary['details'][region][variable]['years'].append(year)
        
        # Convert sets to lists and get year ranges
        for region_data in summary['details'].values():
            for var_data in region_data.values():
                var_data['scenarios'] = list(var_data['scenarios'])
                if var_data['years']:
                    var_data['year_range'] = (min(var_data['years']), max(var_data['years']))
                else:
                    var_data['year_range'] = (None, None)
        
        return summary
    
    def create_metrics_data_config(self) -> MetricsDataConfig:
        """
        Create a data configuration compatible with county_climate_metrics.
        
        Returns:
            MetricsDataConfig object that can replace the DATA_CONFIG in county_climate_metrics
        """
        # Extract regions and scenarios from catalog
        regions = {}
        scenarios = {}
        variables = set()
        
        for dataset in self.catalog_metadata['datasets']:
            region = dataset['region']
            scenario = dataset['scenario']
            variable = dataset['variable']
            year = dataset['target_year']
            
            variables.add(variable)
            
            # Build region config
            if region not in regions:
                regions[region] = {
                    'name': self._get_region_full_name(region),
                    'counties': self._get_region_county_count(region),
                    'path': region
                }
            
            # Build scenario config
            if scenario not in scenarios:
                scenarios[scenario] = {
                    'name': scenario.upper(),
                    'pattern': f'*_{scenario}_*',
                    'years': []
                }
            scenarios[scenario]['years'].append(year)
        
        # Convert year lists to ranges
        for scenario_config in scenarios.values():
            years = scenario_config['years']
            if years:
                scenario_config['years'] = range(min(years), max(years) + 1)
            else:
                scenario_config['years'] = range(1950, 1951)  # Default empty range
        
        return MetricsDataConfig(
            catalog_path=self.catalog_path,
            base_path=str(Path(self.catalog_path).parent.parent),  # Go up from catalog to project root
            regions=regions,
            variables=sorted(list(variables)),
            scenarios=scenarios
        )
    
    def _get_region_full_name(self, region: str) -> str:
        """Get full name for region code."""
        region_names = {
            'CONUS': 'Continental United States',
            'AK': 'Alaska',
            'HI': 'Hawaii',
            'PRVI': 'Puerto Rico and Virgin Islands',
            'GU': 'Guam and Northern Mariana Islands'
        }
        return region_names.get(region, region)
    
    def _get_region_county_count(self, region: str) -> int:
        """Get approximate county count for region."""
        county_counts = {
            'CONUS': 3114,
            'AK': 30,
            'HI': 5,
            'PRVI': 81,
            'GU': 5
        }
        return county_counts.get(region, 0)
    
    def print_data_summary(self):
        """Print comprehensive data availability summary."""
        print("=" * 60)
        print("CLIMATE MEANS CATALOG DATA SUMMARY")
        print("=" * 60)
        
        summary = self.get_data_summary()
        print(f"Catalog: {self.catalog_path}")
        print(f"Regions: {summary['regions']}")
        print(f"Variables: {summary['variables']}")
        print(f"Scenarios: {summary['scenarios']}")
        print()
        
        total_files = 0
        for region, region_data in summary['details'].items():
            print(f"{region}: {self._get_region_full_name(region)}")
            
            for variable, var_data in region_data.items():
                total_files += var_data['total_files']
                year_range = var_data['year_range']
                year_str = f"{year_range[0]}-{year_range[1]}" if year_range[0] else "No data"
                scenarios_str = ", ".join(var_data['scenarios'])
                
                print(f"  {variable}: {var_data['total_files']} files ({year_str}) - {scenarios_str}")
            print()
        
        print(f"Total datasets: {total_files:,}")


def create_integrated_data_config(catalog_path: str) -> str:
    """
    Create a new data_config.py file for county_climate_metrics that uses our catalog.
    
    Args:
        catalog_path: Path to climate means catalog
        
    Returns:
        Generated Python code for the new data_config.py
    """
    adapter = CatalogToBridgeAdapter(catalog_path)
    config = adapter.create_metrics_data_config()
    
    # Generate Python code for the new data_config.py
    code = f'''"""
Climate data configuration - Integrated with Climate Means Catalog

This configuration was automatically generated to integrate the county_climate_metrics
package with the climate means processing pipeline catalog system.

Generated from catalog: {catalog_path}
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
CATALOG_PATH = "{catalog_path}"
_catalog_adapter = CatalogToBridgeAdapter(CATALOG_PATH)

# Data configuration - derived from climate means catalog
DATA_CONFIG = {{
    'base_path': '{config.base_path}',
    'regions': {repr(config.regions)},
    'variables': {repr(config.variables)},
    'scenarios': {repr(config.scenarios)}
}}


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
'''
    
    return code


def integrate_county_metrics_package(county_metrics_path: str, catalog_path: str):
    """
    Integrate the county_climate_metrics package with our climate means catalog.
    
    Args:
        county_metrics_path: Path to county_climate_metrics package
        catalog_path: Path to climate means catalog
    """
    print("🔗 Integrating county_climate_metrics with climate means catalog...")
    
    # Create the bridge adapter
    adapter = CatalogToBridgeAdapter(catalog_path)
    
    # Generate the new data_config.py
    new_config_code = create_integrated_data_config(catalog_path)
    
    # Write the new configuration file
    config_path = os.path.join(county_metrics_path, "metrics", "utils", "data_config.py")
    
    # Backup the original
    backup_path = config_path + ".original"
    if os.path.exists(config_path) and not os.path.exists(backup_path):
        os.rename(config_path, backup_path)
        print(f"📦 Backed up original data_config.py to {backup_path}")
    
    # Write the new configuration
    with open(config_path, 'w') as f:
        f.write(new_config_code)
    
    print(f"✅ Created integrated data_config.py at {config_path}")
    
    # Update the main.py to use the correct climate normals base path
    main_py_path = os.path.join(county_metrics_path, "metrics", "main.py")
    
    # Read current main.py content
    with open(main_py_path, 'r') as f:
        main_content = f.read()
    
    # Update the climate normals base path in setup_paths()
    catalog_dir = os.path.dirname(catalog_path)
    climate_base_path = os.path.dirname(catalog_dir)  # Go up from catalog to output
    
    updated_content = main_content.replace(
        "'climate_normals_base': Path(\"/home/mihiarc/repos/county_climate_means/output\")",
        f"'climate_normals_base': Path(\"{climate_base_path}\")"
    )
    
    # Write back the updated main.py
    with open(main_py_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✅ Updated main.py with correct climate data path")
    
    # Print integration summary
    adapter.print_data_summary()
    
    print("🎯 Integration complete! The county_climate_metrics package can now consume")
    print("   data directly from the climate means catalog without any file structure changes.")