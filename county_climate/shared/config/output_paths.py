"""
Centralized output path configuration for the climate data pipeline.
Supports the new organized directory structure.
"""
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class OrganizedOutputPaths:
    """Manages output paths for the organized climate data structure."""
    
    def __init__(self, base_path: Optional[str] = None, version: Optional[str] = None):
        """
        Initialize output paths.
        
        Args:
            base_path: Base directory for outputs. Defaults to environment variable or standard location.
            version: Version directory name. Defaults to 'v1.0' or from environment.
        """
        # Base path configuration
        self.base_path = Path(
            base_path or 
            os.environ.get('CLIMATE_OUTPUT_BASE') or 
            '/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/organized'
        )
        
        # Version configuration
        self.version = version or os.environ.get('CLIMATE_OUTPUT_VERSION', 'v1.0')
        self.versioned_path = self.base_path / self.version
        
        # Ensure base directories exist
        self.versioned_path.mkdir(parents=True, exist_ok=True)
        
    @property
    def climate_means_base(self) -> Path:
        """L1 Climate means output directory."""
        return self.versioned_path / 'L1_climate_means'
    
    @property
    def county_metrics_base(self) -> Path:
        """L2 County metrics output directory."""
        return self.versioned_path / 'L2_county_metrics'
    
    @property
    def validation_base(self) -> Path:
        """L3 Validation output directory."""
        return self.versioned_path / 'L3_validation'
    
    @property
    def analysis_base(self) -> Path:
        """Analysis outputs directory."""
        return self.versioned_path / 'analysis'
    
    @property
    def catalog_dir(self) -> Path:
        """Catalog directory."""
        return self.versioned_path / 'catalog'
    
    @property
    def metadata_dir(self) -> Path:
        """Metadata directory."""
        return self.versioned_path / 'metadata'
    
    @property
    def logs_dir(self) -> Path:
        """Logs directory (outside version)."""
        return self.base_path / 'logs'
    
    @property
    def temp_dir(self) -> Path:
        """Temporary files directory (outside version)."""
        return self.base_path / 'temp'
    
    def get_means_output_path(self, scenario: str, region: str) -> Path:
        """
        Get output path for climate means files.
        
        Args:
            scenario: Climate scenario (historical, ssp245, ssp585)
            region: Region code (CONUS, AK, HI, PRVI, GU)
            
        Returns:
            Path to the output directory
        """
        path = self.climate_means_base / 'netcdf' / scenario / region
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_means_filename(self, variable: str, region: str, scenario: str, 
                          start_year: int, end_year: Optional[int] = None) -> str:
        """
        Generate standardized filename for climate means.
        
        Args:
            variable: Climate variable (pr, tas, tasmax, tasmin)
            region: Region code
            scenario: Climate scenario
            start_year: Start year of the period
            end_year: End year (defaults to start_year + 29 for 30-year normal)
            
        Returns:
            Standardized filename
        """
        if end_year is None:
            end_year = start_year + 29
        period = f"{start_year}-{end_year}"
        return f"{variable}_{region}_{scenario}_{period}_30yr_mean.nc"
    
    def get_metrics_output_path(self, region: str, format: str = 'csv') -> Path:
        """
        Get output path for county metrics files.
        
        Args:
            region: Region code
            format: Output format (csv, parquet, netcdf)
            
        Returns:
            Path to the output directory
        """
        path = self.county_metrics_base / 'by_region' / region / format
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_metrics_filename(self, region: str, variable: str, scenario: str,
                           start_year: int, format: str = 'csv') -> str:
        """
        Generate standardized filename for county metrics.
        
        Args:
            region: Region code
            variable: Climate variable or 'all_variables'
            scenario: Climate scenario
            start_year: Start year of the period
            format: Output format
            
        Returns:
            Standardized filename
        """
        end_year = start_year + 29
        period = f"{start_year}-{end_year}"
        return f"{region}_{variable}_{scenario}_{period}_metrics.{format}"
    
    def get_validation_output_path(self, validator_type: str) -> Path:
        """
        Get output path for validation results.
        
        Args:
            validator_type: Type of validator (qaqc, spatial_outliers, precipitation)
            
        Returns:
            Path to the output directory
        """
        path = self.validation_base / 'reports' / 'by_validator' / validator_type
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_visualization_path(self, viz_type: str = 'maps') -> Path:
        """
        Get output path for visualizations.
        
        Args:
            viz_type: Type of visualization (maps, timeseries, distributions)
            
        Returns:
            Path to the output directory
        """
        path = self.validation_base / 'visualizations' / viz_type
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def to_dict(self) -> Dict[str, str]:
        """Convert paths to dictionary for configuration."""
        return {
            'base_path': str(self.base_path),
            'version': self.version,
            'climate_means_base': str(self.climate_means_base),
            'county_metrics_base': str(self.county_metrics_base),
            'validation_base': str(self.validation_base),
            'analysis_base': str(self.analysis_base),
            'catalog_dir': str(self.catalog_dir),
            'metadata_dir': str(self.metadata_dir),
            'logs_dir': str(self.logs_dir),
            'temp_dir': str(self.temp_dir)
        }
    
    def __str__(self) -> str:
        """String representation of output paths."""
        return f"OrganizedOutputPaths(base={self.base_path}, version={self.version})"


# Convenience function for backward compatibility
def get_output_paths(base_path: Optional[str] = None, version: Optional[str] = None) -> OrganizedOutputPaths:
    """Get configured output paths instance."""
    return OrganizedOutputPaths(base_path, version)