#!/usr/bin/env python3
"""
Single Variable Climate Processor

Processes all years for a single climate variable sequentially to avoid
NetCDF file conflicts. Designed to run in parallel with other variable processors.
"""

import logging
import gc
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xarray as xr
from datetime import datetime

from county_climate.means.utils.io_util import NorESM2FileHandler
from county_climate.means.core.regions import REGION_BOUNDS, extract_region
from county_climate.means.utils.time_util import handle_time_coordinates, extract_year_from_filename
from county_climate.means.utils.mp_progress import ProgressReporter


class SingleVariableProcessor:
    """Processes a single climate variable for a specific region/scenario."""
    
    def __init__(
        self,
        variable: str,
        region_key: str,
        scenario: str,
        input_data_dir: Path,
        output_base_dir: Path,
        min_years_for_normal: int = 25,
        progress_queue: Optional[object] = None
    ):
        """Initialize the single variable processor.
        
        Args:
            variable: Climate variable to process (pr, tas, tasmax, tasmin)
            region_key: Region identifier (CONUS, AK, HI, PRVI, GU)
            scenario: Climate scenario (historical, ssp245, ssp585)
            input_data_dir: Base directory for input climate data
            output_base_dir: Base directory for output files
            min_years_for_normal: Minimum years required for climate normal
            progress_queue: Queue for progress updates
        """
        self.variable = variable
        self.region_key = region_key
        self.scenario = scenario
        self.input_data_dir = input_data_dir
        self.output_base_dir = output_base_dir
        self.min_years_for_normal = min_years_for_normal
        self.progress_queue = progress_queue
        
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.{variable}_{region_key}_{scenario}")
        
        # Initialize file handler
        self.file_handler = NorESM2FileHandler(input_data_dir)
        
        # Progress reporter
        self.progress_reporter = ProgressReporter(progress_queue) if progress_queue else None
        
    def process(self) -> Dict:
        """Process all years for this variable/region/scenario combination.
        
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Starting processing for {self.variable}/{self.region_key}/{self.scenario}")
        start_time = time.time()
        
        results = {
            'variable': self.variable,
            'region': self.region_key,
            'scenario': self.scenario,
            'target_years_processed': [],
            'files_processed': 0,
            'errors': [],
            'status': 'success'
        }
        
        try:
            # Determine target years based on scenario
            target_years = self._get_target_years()
            
            # Process each target year sequentially
            for target_year in target_years:
                try:
                    year_result = self._process_target_year(target_year)
                    if year_result['status'] == 'success':
                        results['target_years_processed'].append(target_year)
                        results['files_processed'] += year_result['files_processed']
                    elif year_result['status'] == 'skipped':
                        # Still count as processed if output already exists
                        results['target_years_processed'].append(target_year)
                        
                except Exception as e:
                    self.logger.error(f"Error processing target year {target_year}: {e}")
                    results['errors'].append({
                        'target_year': target_year,
                        'error': str(e)
                    })
                    
        except Exception as e:
            self.logger.error(f"Fatal error in variable processing: {e}")
            results['status'] = 'error'
            results['errors'].append({
                'error': str(e),
                'type': 'fatal'
            })
            
        # Calculate processing time
        results['processing_time_seconds'] = time.time() - start_time
        
        return results
    
    def _get_target_years(self) -> List[int]:
        """Determine target years based on scenario.
        
        Returns:
            List of target years to process
        """
        if self.scenario == 'historical':
            return list(range(1980, 2015))  # 1980-2014
        elif self.scenario == 'ssp245' or self.scenario == 'ssp585':
            return list(range(2015, 2101))  # 2015-2100
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")
    
    def _process_target_year(self, target_year: int) -> Dict:
        """Process a single target year (30-year climate normal).
        
        Args:
            target_year: Target year for the climate normal
            
        Returns:
            Dictionary with processing results for this year
        """
        result = {
            'target_year': target_year,
            'status': 'pending',
            'files_processed': 0,
            'output_file': None
        }
        
        # Determine output file path
        period_type = self._get_period_type(target_year)
        output_dir = self.output_base_dir / self.variable / period_type
        output_file = output_dir / f"{self.variable}_{self.region_key}_{period_type}_{target_year}_30yr_normal.nc"
        
        # Check if output already exists
        if output_file.exists():
            self.logger.debug(f"Output already exists for {target_year}, skipping")
            result['status'] = 'skipped'
            result['output_file'] = str(output_file)
            
            # Update progress even for skipped files
            if self.progress_reporter:
                self.progress_reporter.update_task(
                    self.variable,
                    advance=30,  # Approximate files that would have been processed
                    current_item=f"Skipped {target_year} (exists)"
                )
            return result
        
        # Get files for 30-year window
        files = self._get_files_for_target_year(target_year)
        
        if len(files) < self.min_years_for_normal:
            self.logger.warning(f"Insufficient files for {target_year}: {len(files)} < {self.min_years_for_normal}")
            result['status'] = 'insufficient_data'
            return result
        
        # Process files to get daily climatologies
        daily_climatologies = []
        years_used = []
        
        for file_path in files:
            try:
                # Ensure file_path is a Path object
                if not isinstance(file_path, Path):
                    file_path = Path(file_path)
                    
                year, daily_clim = self._process_single_file(file_path)
                if year is not None and daily_clim is not None:
                    daily_climatologies.append(daily_clim)
                    years_used.append(year)
                    result['files_processed'] += 1
                    
                    # Update progress for each file
                    if self.progress_reporter:
                        self.progress_reporter.update_task(
                            self.variable,
                            advance=1,
                            current_item=f"Year {target_year}: {file_path.name}"
                        )
                        
            except Exception as e:
                self.logger.warning(f"Error processing file {file_path}: {e}")
                continue
        
        # Check if we have enough valid data
        if len(daily_climatologies) < self.min_years_for_normal:
            self.logger.warning(f"Insufficient valid climatologies for {target_year}")
            result['status'] = 'insufficient_valid_data'
            return result
        
        # Compute climate normal
        try:
            climate_normal = self._compute_climate_normal(daily_climatologies, years_used, target_year)
            
            # Save result
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Add metadata
            # Validate region key
            if self.region_key not in REGION_BOUNDS:
                self.logger.error(f"Invalid region key: {self.region_key}. Valid keys: {list(REGION_BOUNDS.keys())}")
                region_name = self.region_key
            else:
                region_info = REGION_BOUNDS[self.region_key]
                if isinstance(region_info, dict) and 'name' in region_info:
                    region_name = region_info['name']
                else:
                    self.logger.warning(f"Unexpected REGION_BOUNDS structure for {self.region_key}: {type(region_info)}")
                    region_name = self.region_key
            
            climate_normal.attrs.update({
                'title': f'{self.variable.upper()} 30-Year {period_type.title()} Climate Normal ({self.region_key}) - Target Year {target_year}',
                'variable': self.variable,
                'region': self.region_key,
                'region_name': region_name,
                'scenario': self.scenario,
                'target_year': target_year,
                'period_type': period_type,
                'num_years': len(years_used),
                'source_years': f"{min(years_used)}-{max(years_used)}",
                'processing': f'Single variable processor for {self.variable}',
                'source': 'NorESM2-LM climate model',
                'method': '30-year rolling climate normal',
                'created': datetime.now().isoformat()
            })
            
            climate_normal.to_netcdf(output_file)
            
            result['status'] = 'success'
            result['output_file'] = str(output_file)
            
            # Clean up memory
            del daily_climatologies, climate_normal
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error computing climate normal for {target_year}: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
            
        return result
    
    def _get_period_type(self, target_year: int) -> str:
        """Determine period type based on target year.
        
        Args:
            target_year: Target year
            
        Returns:
            Period type (historical, hybrid, or scenario name)
        """
        if target_year < 2015:
            return 'historical'
        elif target_year < 2045 and self.scenario == 'historical':
            return 'hybrid'
        else:
            return self.scenario
    
    def _get_files_for_target_year(self, target_year: int) -> List[Path]:
        """Get list of files needed for 30-year window.
        
        Args:
            target_year: Target year for climate normal
            
        Returns:
            List of file paths for the 30-year window
        """
        start_year = target_year - 29
        end_year = target_year
        
        if self.scenario == 'historical':
            # For historical, we might need to handle the transition period
            if target_year >= 2015:
                # This is a hybrid period
                hist_start = max(1985, start_year)
                hist_end = min(2014, end_year)
                ssp_start = max(2015, start_year)
                ssp_end = end_year
                
                hist_files = self.file_handler.get_files_for_period(
                    self.variable, 'historical', hist_start, hist_end
                )
                ssp_files = self.file_handler.get_files_for_period(
                    self.variable, 'ssp245', ssp_start, ssp_end
                )
                return hist_files + ssp_files
            else:
                # Pure historical
                return self.file_handler.get_files_for_period(
                    self.variable, 'historical', start_year, end_year
                )
        else:
            # Future scenarios
            return self.file_handler.get_files_for_period(
                self.variable, self.scenario, start_year, end_year
            )
    
    def _process_single_file(self, file_path: Path) -> Tuple[Optional[int], Optional[xr.DataArray]]:
        """Process a single NetCDF file to extract regional daily climatology.
        
        Args:
            file_path: Path to NetCDF file
            
        Returns:
            Tuple of (year, daily_climatology) or (None, None) if failed
        """
        try:
            # Extract year from filename
            year = extract_year_from_filename(file_path)
            
            # Open and process file
            with xr.open_dataset(file_path) as ds:
                # Extract regional data
                # Debug: Check region bounds
                if self.region_key not in REGION_BOUNDS:
                    self.logger.error(f"Region key '{self.region_key}' not in REGION_BOUNDS. Available: {list(REGION_BOUNDS.keys())}")
                    return None, None
                    
                region_info = REGION_BOUNDS[self.region_key]
                if not isinstance(region_info, dict):
                    self.logger.error(f"REGION_BOUNDS['{self.region_key}'] is not a dict: {type(region_info)}, value: {region_info}")
                    return None, None
                    
                regional_data = extract_region(ds, region_info)
                
                if regional_data is None:
                    return None, None
                
                # Fix time coordinates
                regional_data, _ = handle_time_coordinates(regional_data, str(file_path))
                
                # Calculate daily climatology
                if self.variable in regional_data:
                    var_data = regional_data[self.variable]
                    
                    # For daily data, group by day of year
                    daily_clim = var_data.groupby('time.dayofyear').mean(dim='time')
                    
                    return year, daily_clim
                    
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            
        return None, None
    
    def _compute_climate_normal(self, daily_climatologies: List[xr.DataArray], 
                               years_used: List[int], target_year: int) -> xr.DataArray:
        """Compute 30-year climate normal from daily climatologies.
        
        Args:
            daily_climatologies: List of daily climatology arrays
            years_used: List of years used
            target_year: Target year for the normal
            
        Returns:
            Climate normal DataArray
        """
        # Stack all daily climatologies along a new 'year' dimension
        stacked_data = xr.concat(daily_climatologies, dim='year')
        stacked_data['year'] = years_used
        
        # Calculate the mean across all years for each day of year
        climate_normal = stacked_data.mean(dim='year')
        
        return climate_normal


def process_single_variable(args: Dict) -> Dict:
    """Function wrapper for multiprocessing.
    
    Args:
        args: Dictionary with processing parameters
        
    Returns:
        Processing results
    """
    processor = SingleVariableProcessor(
        variable=args['variable'],
        region_key=args['region_key'],
        scenario=args['scenario'],
        input_data_dir=Path(args['input_data_dir']),
        output_base_dir=Path(args['output_base_dir']),
        min_years_for_normal=args.get('min_years_for_normal', 25),
        progress_queue=args.get('progress_queue')
    )
    
    return processor.process()