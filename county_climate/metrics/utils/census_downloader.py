"""
U.S. Census County Boundaries Downloader

This module provides utilities to download county boundary data from the 
U.S. Census Bureau's TIGERweb API and convert it to shapefiles for use
in climate analysis pipelines.

Features:
- Download 2024 county boundaries from TIGERweb REST API
- Support for all U.S. states, territories, and DC
- Convert GeoJSON to shapefile format
- Caching and resumable downloads
- Regional filtering capabilities
- Integration with climate pipeline requirements

Usage:
    downloader = CensusCountyDownloader()
    shapefile_path = downloader.download_counties()
    
    # Download specific regions only
    shapefile_path = downloader.download_counties(regions=['CONUS', 'AK'])
"""

import os
import json
import requests
import zipfile
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import time

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import shape
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


@dataclass
class DownloadConfig:
    """Configuration for county boundary downloads."""
    cache_dir: str = "cache/census_data"
    output_dir: str = "tl_2024_us_county"
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 300
    chunk_size: int = 1000  # Features per request
    

class CensusCountyDownloader:
    """
    Downloads U.S. county boundaries from Census TIGERweb API.
    
    This class provides methods to download county boundary data from the
    U.S. Census Bureau's TIGERweb REST API, with support for caching,
    regional filtering, and conversion to shapefile format.
    """
    
    # TIGERweb API endpoints
    BASE_URL = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb"
    COUNTY_ENDPOINT = f"{BASE_URL}/State_County/MapServer/11"
    
    # Regional FIPS code mappings for climate pipeline integration
    REGION_FIPS = {
        'CONUS': [str(i).zfill(2) for i in range(1, 57) if i not in [2, 15, 60, 66, 69, 72, 78]],  # Continental US
        'AK': ['02'],      # Alaska
        'HI': ['15'],      # Hawaii  
        'PRVI': ['72'],    # Puerto Rico
        'GU': ['66'],      # Guam
        'VI': ['78'],      # Virgin Islands (part of PRVI region)
        'AS': ['60'],      # American Samoa
        'MP': ['69'],      # Northern Mariana Islands (part of GU region)
    }
    
    def __init__(self, config: Optional[DownloadConfig] = None):
        """
        Initialize the Census County Downloader.
        
        Args:
            config: Download configuration options
        """
        self.config = config or DownloadConfig()
        self.logger = self._setup_logging()
        
        # Create cache and output directories
        os.makedirs(self.config.cache_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if not HAS_GEOPANDAS:
            self.logger.warning(
                "GeoPandas not available. Install with: pip install geopandas"
                " for full shapefile conversion capabilities."
            )
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the downloader."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def download_counties(self, 
                         regions: Optional[List[str]] = None,
                         force_refresh: bool = False,
                         output_format: str = 'shapefile') -> str:
        """
        Download county boundaries for specified regions.
        
        Args:
            regions: List of region codes (CONUS, AK, HI, PRVI, GU).
                    If None, downloads all regions.
            force_refresh: Force re-download even if cached data exists
            output_format: Output format ('shapefile', 'geojson')
            
        Returns:
            Path to the downloaded/created county boundaries file
            
        Raises:
            ValueError: If invalid region specified or dependencies missing
            requests.RequestException: If API request fails
        """
        if regions is None:
            regions = ['CONUS', 'AK', 'HI', 'PRVI', 'GU']
        
        # Validate regions
        invalid_regions = [r for r in regions if r not in self.REGION_FIPS]
        if invalid_regions:
            raise ValueError(f"Invalid regions: {invalid_regions}. "
                           f"Valid regions: {list(self.REGION_FIPS.keys())}")
        
        self.logger.info(f"Downloading county boundaries for regions: {regions}")
        
        # Check cache first
        cache_file = os.path.join(self.config.cache_dir, "all_counties.geojson")
        if not force_refresh and os.path.exists(cache_file):
            self.logger.info(f"Using cached county data: {cache_file}")
            county_data = self._load_cached_data(cache_file)
        else:
            # Download from API
            county_data = self._download_from_api()
            self._save_to_cache(county_data, cache_file)
        
        # Filter by regions if specified
        if regions != ['CONUS', 'AK', 'HI', 'PRVI', 'GU']:  # Not all regions
            county_data = self._filter_by_regions(county_data, regions)
        
        # Convert to requested output format
        if output_format == 'shapefile':
            return self._convert_to_shapefile(county_data, regions)
        elif output_format == 'geojson':
            return self._save_as_geojson(county_data, regions)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _download_from_api(self) -> Dict[str, Any]:
        """
        Download county data from TIGERweb API.
        
        Returns:
            GeoJSON-like dictionary with county features
            
        Raises:
            requests.RequestException: If API request fails
        """
        self.logger.info("Downloading county boundaries from TIGERweb API...")
        
        # Get total feature count first
        count_params = {
            'where': '1=1',
            'returnCountOnly': 'true',
            'f': 'json'
        }
        
        count_response = self._make_request(self.COUNTY_ENDPOINT + "/query", count_params)
        total_count = count_response.get('count', 0)
        self.logger.info(f"Total counties to download: {total_count:,}")
        
        # Download in chunks to handle large datasets
        all_features = []
        offset = 0
        
        while offset < total_count:
            self.logger.info(f"Downloading counties {offset:,} - {min(offset + self.config.chunk_size, total_count):,}")
            
            params = {
                'where': '1=1',
                'outFields': '*',
                'returnGeometry': 'true',
                'f': 'geojson',
                'resultOffset': offset,
                'resultRecordCount': self.config.chunk_size
            }
            
            chunk_data = self._make_request(self.COUNTY_ENDPOINT + "/query", params)
            
            if 'features' in chunk_data:
                all_features.extend(chunk_data['features'])
                self.logger.info(f"Downloaded {len(chunk_data['features'])} features")
            
            offset += self.config.chunk_size
            
            # Rate limiting
            time.sleep(0.1)
        
        # Construct GeoJSON structure
        geojson_data = {
            "type": "FeatureCollection",
            "features": all_features,
            "metadata": {
                "source": "U.S. Census Bureau TIGERweb",
                "vintage": "2024",
                "downloaded": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_features": len(all_features)
            }
        }
        
        self.logger.info(f"Successfully downloaded {len(all_features):,} county features")
        return geojson_data
    
    def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.
        
        Args:
            url: Request URL
            params: Request parameters
            
        Returns:
            JSON response data
            
        Raises:
            requests.RequestException: If all retries fail
        """
        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(
                    url, 
                    params=params, 
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    self.logger.error(f"Failed to download after {self.config.max_retries} attempts: {e}")
                    raise
                
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
    
    def _load_cached_data(self, cache_file: str) -> Dict[str, Any]:
        """Load county data from cache file."""
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_to_cache(self, data: Dict[str, Any], cache_file: str) -> None:
        """Save county data to cache file."""
        self.logger.info(f"Caching county data to {cache_file}")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'))  # Compact JSON
    
    def _filter_by_regions(self, county_data: Dict[str, Any], regions: List[str]) -> Dict[str, Any]:
        """
        Filter county data by specified regions.
        
        Args:
            county_data: GeoJSON county data
            regions: List of region codes to include
            
        Returns:
            Filtered GeoJSON data
        """
        # Get FIPS codes for requested regions
        target_fips = []
        for region in regions:
            target_fips.extend(self.REGION_FIPS[region])
        
        self.logger.info(f"Filtering counties for FIPS codes: {target_fips}")
        
        # Filter features
        filtered_features = []
        for feature in county_data['features']:
            # Get state FIPS from GEOID (first 2 characters)
            geoid = feature['properties'].get('GEOID', '')
            state_fips = geoid[:2] if len(geoid) >= 2 else ''
            
            if state_fips in target_fips:
                filtered_features.append(feature)
        
        # Update data structure
        filtered_data = county_data.copy()
        filtered_data['features'] = filtered_features
        filtered_data['metadata']['filtered_regions'] = regions
        filtered_data['metadata']['filtered_count'] = len(filtered_features)
        
        self.logger.info(f"Filtered to {len(filtered_features)} counties for regions: {regions}")
        return filtered_data
    
    def _convert_to_shapefile(self, county_data: Dict[str, Any], regions: List[str]) -> str:
        """
        Convert GeoJSON county data to shapefile format.
        
        Args:
            county_data: GeoJSON county data
            regions: List of regions (for filename)
            
        Returns:
            Path to created shapefile
            
        Raises:
            ImportError: If geopandas not available
            ValueError: If conversion fails
        """
        if not HAS_GEOPANDAS:
            raise ImportError(
                "GeoPandas required for shapefile conversion. "
                "Install with: pip install geopandas"
            )
        
        self.logger.info("Converting GeoJSON to shapefile format...")
        
        try:
            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(
                county_data['features'],
                crs='EPSG:4326'  # WGS84
            )
            
            # Generate output filename
            if len(regions) == 1:
                region_suffix = f"_{regions[0]}"
            elif set(regions) == {'CONUS', 'AK', 'HI', 'PRVI', 'GU'}:
                region_suffix = "_all"
            else:
                region_suffix = f"_{'_'.join(sorted(regions))}"
            
            output_path = os.path.join(
                self.config.output_dir,
                f"tl_2024_us_county{region_suffix}"
            )
            
            # Ensure output directory exists
            os.makedirs(output_path, exist_ok=True)
            
            # Write shapefile
            shapefile_path = os.path.join(output_path, "tl_2024_us_county.shp")
            gdf.to_file(shapefile_path)
            
            # Create metadata file
            metadata_path = os.path.join(output_path, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(county_data.get('metadata', {}), f, indent=2)
            
            self.logger.info(f"Shapefile created: {shapefile_path}")
            self.logger.info(f"Contains {len(gdf)} county features")
            
            return shapefile_path
            
        except Exception as e:
            self.logger.error(f"Failed to convert to shapefile: {e}")
            raise ValueError(f"Shapefile conversion failed: {e}")
    
    def _save_as_geojson(self, county_data: Dict[str, Any], regions: List[str]) -> str:
        """
        Save county data as GeoJSON file.
        
        Args:
            county_data: GeoJSON county data
            regions: List of regions (for filename)
            
        Returns:
            Path to created GeoJSON file
        """
        # Generate output filename
        if len(regions) == 1:
            region_suffix = f"_{regions[0]}"
        elif set(regions) == {'CONUS', 'AK', 'HI', 'PRVI', 'GU'}:
            region_suffix = "_all"
        else:
            region_suffix = f"_{'_'.join(sorted(regions))}"
        
        output_file = os.path.join(
            self.config.output_dir,
            f"tl_2024_us_county{region_suffix}.geojson"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(county_data, f, indent=2)
        
        self.logger.info(f"GeoJSON saved: {output_file}")
        return output_file
    
    def get_available_regions(self) -> Dict[str, List[str]]:
        """
        Get available region codes and their FIPS mappings.
        
        Returns:
            Dictionary mapping region codes to FIPS codes
        """
        return self.REGION_FIPS.copy()
    
    def validate_dependencies(self) -> Dict[str, bool]:
        """
        Check if required dependencies are available.
        
        Returns:
            Dictionary showing availability of dependencies
        """
        deps = {
            'requests': True,  # Always available (required)
            'geopandas': HAS_GEOPANDAS,
            'pandas': HAS_GEOPANDAS,  # Comes with geopandas
            'shapely': HAS_GEOPANDAS,  # Comes with geopandas
        }
        return deps
    
    def get_county_info(self, geoid: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific county by GEOID.
        
        Args:
            geoid: 5-digit FIPS county code
            
        Returns:
            County information dictionary or None if not found
        """
        params = {
            'where': f"GEOID='{geoid}'",
            'outFields': '*',
            'returnGeometry': 'false',
            'f': 'json'
        }
        
        try:
            response = self._make_request(self.COUNTY_ENDPOINT + "/query", params)
            features = response.get('features', [])
            
            if features:
                return features[0]['attributes']
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get county info for {geoid}: {e}")
            return None


def download_us_counties(regions: Optional[List[str]] = None,
                        output_dir: str = "tl_2024_us_county",
                        force_refresh: bool = False) -> str:
    """
    Convenience function to download U.S. county boundaries.
    
    Args:
        regions: List of region codes (CONUS, AK, HI, PRVI, GU)
        output_dir: Output directory for shapefile
        force_refresh: Force re-download even if cached
        
    Returns:
        Path to created shapefile
        
    Example:
        # Download all U.S. counties
        shapefile_path = download_us_counties()
        
        # Download only CONUS and Alaska
        shapefile_path = download_us_counties(['CONUS', 'AK'])
    """
    config = DownloadConfig(output_dir=output_dir)
    downloader = CensusCountyDownloader(config)
    return downloader.download_counties(
        regions=regions,
        force_refresh=force_refresh,
        output_format='shapefile'
    )


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Check dependencies
    downloader = CensusCountyDownloader()
    deps = downloader.validate_dependencies()
    
    print("Dependency Check:")
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"  {dep}: {status}")
    
    if not deps['geopandas']:
        print("\n⚠️  GeoPandas not available. Install with:")
        print("pip install geopandas")
        sys.exit(1)
    
    # Download counties
    try:
        print("\nDownloading county boundaries...")
        shapefile_path = download_us_counties(['CONUS', 'AK'])
        print(f"✅ County boundaries downloaded: {shapefile_path}")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)