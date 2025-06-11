"""
Validation utilities for climate data contracts and processing.

These validators provide comprehensive validation for climate data,
ensuring data quality and contract compliance across the pipeline.
"""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import xarray as xr
import numpy as np
from pydantic import ValidationError

from ..contracts.climate_data import (
    ClimateDatasetContract,
    ValidationResultContract,
    ValidationStatus,
    QualityMetricsContract,
    SpatialBoundsContract
)


class ClimateDataValidator:
    """
    Comprehensive validator for climate datasets and their contracts.
    
    Validates both the contract compliance and the actual data quality
    of climate datasets to ensure pipeline integrity.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, applies stricter validation rules
        """
        self.strict_mode = strict_mode
        
        # Physical data ranges for validation
        self.variable_ranges = {
            'pr': (0.0, 0.001),  # Precipitation: 0 to ~100mm/day in kg/m²/s
            'tas': (200.0, 350.0),  # Temperature: -73°C to 77°C in Kelvin
            'tasmax': (200.0, 350.0),  # Max temperature
            'tasmin': (200.0, 350.0),  # Min temperature
        }
        
        # Expected coordinate ranges by region
        self.region_bounds = {
            'CONUS': {'lon': (-125.0, -66.5), 'lat': (25.0, 49.0)},
            'AK': {'lon': (-180.0, -129.0), 'lat': (54.0, 71.5)},
            'HI': {'lon': (-161.0, -154.5), 'lat': (18.5, 22.5)},
            'PRVI': {'lon': (-68.0, -64.5), 'lat': (17.5, 18.5)},
            'GU': {'lon': (144.5, 145.0), 'lat': (13.2, 13.7)}
        }
    
    def validate_contract(self, dataset: ClimateDatasetContract) -> ValidationResultContract:
        """
        Validate a climate dataset contract for compliance and consistency.
        
        Args:
            dataset: Climate dataset contract to validate
            
        Returns:
            ValidationResultContract with detailed validation results
        """
        validation_messages = []
        validation_warnings = []
        validation_errors = []
        
        # Test each validation component
        coordinate_valid = self._validate_coordinates(dataset, validation_messages, validation_warnings, validation_errors)
        temporal_valid = self._validate_temporal_data(dataset, validation_messages, validation_warnings, validation_errors)
        data_range_valid = self._validate_data_ranges(dataset, validation_messages, validation_warnings, validation_errors)
        metadata_valid = self._validate_metadata(dataset, validation_messages, validation_warnings, validation_errors)
        file_integrity_valid = self._validate_file_integrity(dataset, validation_messages, validation_warnings, validation_errors)
        
        # Determine overall status
        if validation_errors:
            status = ValidationStatus.FAILED
        elif validation_warnings and self.strict_mode:
            status = ValidationStatus.FAILED
        else:
            status = ValidationStatus.PASSED
        
        return ValidationResultContract(
            validation_status=status,
            validation_timestamp=datetime.now(timezone.utc),
            validator_version="1.0.0",
            coordinate_validation=coordinate_valid,
            temporal_validation=temporal_valid,
            data_range_validation=data_range_valid,
            metadata_validation=metadata_valid,
            file_integrity_validation=file_integrity_valid,
            validation_messages=validation_messages,
            validation_warnings=validation_warnings,
            validation_errors=validation_errors
        )
    
    def validate_dataset_file(self, file_path: str, expected_contract: ClimateDatasetContract) -> Tuple[bool, List[str]]:
        """
        Validate the actual NetCDF file against its contract.
        
        Args:
            file_path: Path to NetCDF file
            expected_contract: Expected dataset contract
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Check file exists
            if not Path(file_path).exists():
                errors.append(f"File does not exist: {file_path}")
                return False, errors
            
            # Open dataset
            with xr.open_dataset(file_path) as ds:
                # Validate variable presence
                if expected_contract.variable not in ds.data_vars:
                    errors.append(f"Variable {expected_contract.variable} not found in dataset")
                
                # Validate coordinates
                coord_errors = self._validate_netcdf_coordinates(ds, expected_contract)
                errors.extend(coord_errors)
                
                # Validate data values
                data_errors = self._validate_netcdf_data_values(ds, expected_contract)
                errors.extend(data_errors)
                
                # Validate spatial bounds
                spatial_errors = self._validate_netcdf_spatial_bounds(ds, expected_contract)
                errors.extend(spatial_errors)
        
        except Exception as e:
            errors.append(f"Error reading NetCDF file: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _validate_coordinates(self, dataset: ClimateDatasetContract, messages: List[str], 
                            warnings: List[str], errors: List[str]) -> bool:
        """Validate coordinate system and spatial bounds."""
        try:
            # Check spatial bounds are reasonable
            bounds = dataset.spatial_bounds
            region_expected = self.region_bounds.get(dataset.region)
            
            if region_expected:
                # Check longitude bounds
                if (bounds.min_longitude < region_expected['lon'][0] - 5 or 
                    bounds.max_longitude > region_expected['lon'][1] + 5):
                    warnings.append(f"Longitude bounds {bounds.min_longitude}-{bounds.max_longitude} "
                                  f"outside expected range for {dataset.region}")
                
                # Check latitude bounds
                if (bounds.min_latitude < region_expected['lat'][0] - 2 or 
                    bounds.max_latitude > region_expected['lat'][1] + 2):
                    warnings.append(f"Latitude bounds {bounds.min_latitude}-{bounds.max_latitude} "
                                  f"outside expected range for {dataset.region}")
            
            messages.append("Coordinate validation completed")
            return True
            
        except Exception as e:
            errors.append(f"Coordinate validation failed: {str(e)}")
            return False
    
    def _validate_temporal_data(self, dataset: ClimateDatasetContract, messages: List[str],
                               warnings: List[str], errors: List[str]) -> bool:
        """Validate temporal data consistency."""
        try:
            # Check target year is within processing window
            proc_meta = dataset.processing_metadata
            if not (proc_meta.temporal_window_start <= dataset.target_year <= proc_meta.temporal_window_end):
                errors.append(f"Target year {dataset.target_year} not in temporal window "
                             f"{proc_meta.temporal_window_start}-{proc_meta.temporal_window_end}")
                return False
            
            # Check temporal window length
            window_length = proc_meta.temporal_window_end - proc_meta.temporal_window_start + 1
            if window_length != proc_meta.temporal_window_length:
                warnings.append(f"Temporal window length mismatch: calculated {window_length}, "
                               f"reported {proc_meta.temporal_window_length}")
            
            messages.append("Temporal validation completed")
            return True
            
        except Exception as e:
            errors.append(f"Temporal validation failed: {str(e)}")
            return False
    
    def _validate_data_ranges(self, dataset: ClimateDatasetContract, messages: List[str],
                             warnings: List[str], errors: List[str]) -> bool:
        """Validate data values are within physical ranges."""
        try:
            # For contract validation, we check if quality metrics indicate valid ranges
            quality = dataset.quality_metrics
            
            if not quality.data_range_validation:
                errors.append("Data range validation failed according to quality metrics")
                return False
            
            if quality.quality_score < 0.8:
                warnings.append(f"Low quality score: {quality.quality_score}")
            
            # Check for specific quality flags
            concerning_flags = {'outlier_values', 'data_range_errors'}
            if any(flag in quality.quality_flags for flag in concerning_flags):
                warnings.append(f"Concerning quality flags: {quality.quality_flags}")
            
            messages.append("Data range validation completed")
            return True
            
        except Exception as e:
            errors.append(f"Data range validation failed: {str(e)}")
            return False
    
    def _validate_metadata(self, dataset: ClimateDatasetContract, messages: List[str],
                          warnings: List[str], errors: List[str]) -> bool:
        """Validate metadata completeness and consistency."""
        try:
            # Check required metadata fields
            if not dataset.units:
                warnings.append("Units not specified")
            
            if not dataset.description:
                warnings.append("Description not provided")
            
            # Check processing metadata completeness
            proc_meta = dataset.processing_metadata
            if not proc_meta.source_files:
                errors.append("No source files listed in processing metadata")
                return False
            
            if proc_meta.processing_duration_seconds <= 0:
                warnings.append("Invalid processing duration")
            
            # Check data access metadata
            if not Path(dataset.data_access.file_path).suffix == '.nc':
                warnings.append("File does not have .nc extension")
            
            messages.append("Metadata validation completed")
            return True
            
        except Exception as e:
            errors.append(f"Metadata validation failed: {str(e)}")
            return False
    
    def _validate_file_integrity(self, dataset: ClimateDatasetContract, messages: List[str],
                                warnings: List[str], errors: List[str]) -> bool:
        """Validate file integrity and accessibility."""
        try:
            file_path = Path(dataset.data_access.file_path)
            
            # Check file exists
            if not file_path.exists():
                errors.append(f"File does not exist: {file_path}")
                return False
            
            # Check file size matches
            actual_size = file_path.stat().st_size
            expected_size = dataset.data_access.file_size_bytes
            
            if abs(actual_size - expected_size) > 1024:  # Allow 1KB difference
                warnings.append(f"File size mismatch: actual {actual_size}, expected {expected_size}")
            
            # Validate checksum if possible
            if dataset.data_access.checksum:
                if not self._verify_checksum(file_path, dataset.data_access.checksum):
                    errors.append("File checksum verification failed")
                    return False
            
            messages.append("File integrity validation completed")
            return True
            
        except Exception as e:
            errors.append(f"File integrity validation failed: {str(e)}")
            return False
    
    def _validate_netcdf_coordinates(self, ds: xr.Dataset, contract: ClimateDatasetContract) -> List[str]:
        """Validate NetCDF coordinate variables."""
        errors = []
        
        # Check for required coordinate dimensions
        required_dims = {'lat', 'lon', 'time'}
        missing_dims = required_dims - set(ds.dims.keys())
        if missing_dims:
            errors.append(f"Missing coordinate dimensions: {missing_dims}")
        
        # Validate coordinate values
        if 'lat' in ds.coords:
            lat_values = ds.coords['lat'].values
            if np.any(lat_values < -90) or np.any(lat_values > 90):
                errors.append("Latitude values outside valid range [-90, 90]")
        
        if 'lon' in ds.coords:
            lon_values = ds.coords['lon'].values
            # Allow both -180-180 and 0-360 conventions
            if not (np.all((lon_values >= -180) & (lon_values <= 360))):
                errors.append("Longitude values outside valid range [-180, 360]")
        
        return errors
    
    def _validate_netcdf_data_values(self, ds: xr.Dataset, contract: ClimateDatasetContract) -> List[str]:
        """Validate NetCDF data variable values."""
        errors = []
        
        variable = contract.variable
        if variable in ds.data_vars:
            data = ds[variable].values
            
            # Check for expected data range
            if variable in self.variable_ranges:
                min_val, max_val = self.variable_ranges[variable]
                if np.any(data < min_val) or np.any(data > max_val):
                    actual_min, actual_max = np.nanmin(data), np.nanmax(data)
                    errors.append(f"Data values outside expected range [{min_val}, {max_val}]: "
                                f"actual range [{actual_min:.3f}, {actual_max:.3f}]")
            
            # Check for excessive missing values
            if np.isnan(data).sum() / data.size > 0.5:
                errors.append("More than 50% of data values are missing")
        
        return errors
    
    def _validate_netcdf_spatial_bounds(self, ds: xr.Dataset, contract: ClimateDatasetContract) -> List[str]:
        """Validate NetCDF spatial bounds match contract."""
        errors = []
        
        if 'lat' in ds.coords and 'lon' in ds.coords:
            actual_bounds = SpatialBoundsContract(
                min_longitude=float(ds.coords['lon'].min()),
                max_longitude=float(ds.coords['lon'].max()),
                min_latitude=float(ds.coords['lat'].min()),
                max_latitude=float(ds.coords['lat'].max())
            )
            
            expected_bounds = contract.spatial_bounds
            
            # Allow small tolerance for floating point differences
            tolerance = 0.1
            if (abs(actual_bounds.min_longitude - expected_bounds.min_longitude) > tolerance or
                abs(actual_bounds.max_longitude - expected_bounds.max_longitude) > tolerance or
                abs(actual_bounds.min_latitude - expected_bounds.min_latitude) > tolerance or
                abs(actual_bounds.max_latitude - expected_bounds.max_latitude) > tolerance):
                errors.append(f"Spatial bounds mismatch: "
                             f"actual {actual_bounds.to_list()}, "
                             f"expected {expected_bounds.to_list()}")
        
        return errors
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            actual_checksum = sha256_hash.hexdigest()
            return actual_checksum.lower() == expected_checksum.lower()
        
        except Exception:
            return False


class FilePathValidator:
    """Validator for file paths and directory structures."""
    
    @staticmethod
    def validate_relative_path(path: str) -> bool:
        """Validate that path is relative and safe."""
        return not (path.startswith('/') or '..' in path or '\\' in path)
    
    @staticmethod
    def validate_netcdf_extension(path: str) -> bool:
        """Validate NetCDF file extension."""
        return Path(path).suffix.lower() in ['.nc', '.nc4']
    
    @staticmethod
    def validate_path_pattern(path: str, pattern: str) -> bool:
        """Validate path matches expected pattern."""
        return re.match(pattern, path) is not None


class CoordinateValidator:
    """Validator for geographic coordinates."""
    
    @staticmethod
    def validate_longitude(lon: float) -> bool:
        """Validate longitude value."""
        return -180 <= lon <= 360
    
    @staticmethod
    def validate_latitude(lat: float) -> bool:
        """Validate latitude value."""
        return -90 <= lat <= 90
    
    @staticmethod
    def validate_bounds_consistency(min_val: float, max_val: float) -> bool:
        """Validate min/max bounds consistency."""
        return min_val <= max_val


class TemporalValidator:
    """Validator for temporal data."""
    
    @staticmethod
    def validate_year(year: int) -> bool:
        """Validate year is within reasonable range."""
        return 1950 <= year <= 2100
    
    @staticmethod
    def validate_year_range(start_year: int, end_year: int) -> bool:
        """Validate year range consistency."""
        return start_year <= end_year and TemporalValidator.validate_year(start_year) and TemporalValidator.validate_year(end_year)
    
    @staticmethod
    def validate_climatology_window(start_year: int, end_year: int, target_year: int) -> bool:
        """Validate climatology window contains target year."""
        return start_year <= target_year <= end_year