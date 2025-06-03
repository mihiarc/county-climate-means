#!/usr/bin/env python3
"""
Comprehensive Climate Normals Processing Pipeline

This script processes rolling 30-year climate normals for all four climate variables:
- pr (precipitation)
- tas (near-surface air temperature) 
- tasmax (daily maximum near-surface air temperature)
- tasmin (daily minimum near-surface air temperature)

Processing approaches:
1. Historical normals: 1980-2014 (using only historical data)
2. Hybrid normals: 2015-2044 (combining historical + ssp245 data for 30-year windows)
3. Future normals: 2045+ (using only ssp245 data for 30-year windows)

Output structure:
output/rolling_30year_climate_normals/
├── pr/
│   ├── historical/    # 1980-2014 normals
│   ├── hybrid/        # 2015-2044 normals (historical + ssp245)
│   └── ssp245/        # 2045+ normals (ssp245 only)
├── tas/
├── tasmax/
└── tasmin/
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import xarray as xr
import gc
from typing import List, Dict, Tuple

# Import our modules
from io_util import NorESM2FileHandler
from regions import REGION_BOUNDS, extract_region
from time_util import handle_time_coordinates
from climate_means import compute_climate_normal, calculate_daily_climatology

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_DATA_DIR = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
OUTPUT_BASE_DIR = "output/rolling_30year_climate_normals"
VARIABLES = ['pr', 'tas', 'tasmax', 'tasmin']
REGIONS = ['CONUS']  # Focus on CONUS for now
MIN_YEARS_FOR_NORMAL = 25  # Minimum years needed for a climate normal

def process_single_file_for_climatology(file_path: str, variable_name: str, 
                                       region_key: str, file_handler) -> Tuple[int, xr.DataArray]:
    """
    Process a single file to extract daily climatology for a region.
    
    Returns:
        Tuple of (year, daily_climatology) or (None, None) if failed
    """
    try:
        # Extract year from filename
        year = file_handler.extract_year_from_filename(file_path)
        if year is None:
            logger.warning(f"Could not extract year from {file_path}")
            return None, None
        
        # Open dataset
        ds = xr.open_dataset(file_path, decode_times=False, chunks={'time': 365})
        
        # Check if variable exists
        if variable_name not in ds.data_vars:
            logger.warning(f"Variable {variable_name} not found in {file_path}")
            ds.close()
            return None, None
        
        # Handle time coordinates
        ds, time_method = handle_time_coordinates(ds, file_path)
        
        # Extract region
        region_bounds = REGION_BOUNDS[region_key]
        region_ds = extract_region(ds, region_bounds)
        var = region_ds[variable_name]
        
        # Calculate daily climatology
        climatology = calculate_daily_climatology(var, time_method, file_path)
        if climatology is None:
            ds.close()
            return None, None
        
        # Compute result
        result = climatology.compute()
        
        # Close dataset and cleanup
        ds.close()
        del ds, region_ds, var, climatology
        gc.collect()
        
        return year, result
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None, None


def process_historical_normals(variable: str, region_key: str, file_handler) -> None:
    """
    Process historical rolling 30-year climate normals (1980-2014).
    Each target year uses the preceding 30 years of historical data.
    """
    logger.info(f"Processing historical normals for {variable} {region_key}")
    
    # Output directory
    output_dir = Path(OUTPUT_BASE_DIR) / variable / "historical"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target years for historical normals
    target_years = list(range(1980, 2015))  # 1980-2014
    
    all_normals = []
    all_target_years = []
    
    for target_year in target_years:
        # 30-year period ending at target_year
        start_year = target_year - 29
        end_year = target_year
        
        logger.info(f"  Processing target year {target_year} (period: {start_year}-{end_year})")
        
        # Get files for this period (historical only)
        files = file_handler.get_files_for_period(variable, 'historical', start_year, end_year)
        
        if len(files) < MIN_YEARS_FOR_NORMAL:
            logger.warning(f"    Insufficient files ({len(files)}) for {target_year}")
            continue
        
        # Process files to get daily climatologies
        daily_climatologies = []
        years_used = []
        
        for file_path in files:
            year, daily_clim = process_single_file_for_climatology(
                file_path, variable, region_key, file_handler
            )
            if year is not None and daily_clim is not None:
                daily_climatologies.append(daily_clim)
                years_used.append(year)
        
        if len(daily_climatologies) >= MIN_YEARS_FOR_NORMAL:
            logger.info(f"    Computing normal from {len(daily_climatologies)} years")
            
            # Compute 30-year climate normal
            data_arrays = [clim.values for clim in daily_climatologies]
            climate_normal = compute_climate_normal(data_arrays, years_used, target_year)
            
            if climate_normal is not None:
                # Add metadata
                climate_normal.attrs.update({
                    'title': f'{variable.upper()} 30-Year Climate Normal ({region_key}) - Target Year {target_year}',
                    'variable': variable,
                    'region': region_key,
                    'target_year': target_year,
                    'period_type': 'historical',
                    'period_used': f"{min(years_used)}-{max(years_used)}",
                    'num_years': len(years_used),
                    'creation_date': str(pd.Timestamp.now()),
                    'source': 'NorESM2-LM climate model',
                    'scenario': 'historical',
                    'method': '30-year rolling climate normal',
                    'processing': 'Sequential processing'
                })
                
                # Save individual file
                output_file = output_dir / f"{variable}_{region_key}_historical_{target_year}_30yr_normal.nc"
                climate_normal.to_netcdf(output_file)
                
                # Store for combined dataset
                all_normals.append(climate_normal)
                all_target_years.append(target_year)
                
                logger.info(f"    ✓ Saved normal for {target_year}")
                
                # Cleanup
                del climate_normal, daily_climatologies
                gc.collect()
            else:
                logger.error(f"    ✗ Failed to compute normal for {target_year}")
        else:
            logger.warning(f"    Insufficient processed files ({len(daily_climatologies)}) for {target_year}")
    
    # Create combined dataset
    if all_normals:
        logger.info(f"Creating combined historical dataset with {len(all_normals)} normals")
        combined = xr.concat(all_normals, dim='target_year')
        combined = combined.assign_coords(target_year=all_target_years)
        
        # Add global attributes
        combined.attrs.update({
            'title': f'{variable.upper()} Historical 30-Year Climate Normals ({region_key}) 1980-2014',
            'description': f'Historical 30-year rolling climate normals for {variable}',
            'variable': variable,
            'region': region_key,
            'target_years': f"{min(all_target_years)}-{max(all_target_years)}",
            'num_normals': len(all_target_years),
            'creation_date': str(pd.Timestamp.now()),
            'source': 'NorESM2-LM climate model',
            'scenario': 'historical',
            'method': 'Historical rolling 30-year climate normals'
        })
        
        # Save combined file
        combined_file = output_dir / f"{variable}_{region_key}_historical_1980-2014_all_normals.nc"
        combined.to_netcdf(combined_file)
        
        logger.info(f"✓ Historical normals completed for {variable} {region_key}")
        logger.info(f"  Individual files: {output_dir}/{variable}_{region_key}_historical_YYYY_30yr_normal.nc")
        logger.info(f"  Combined file: {combined_file}")


def process_hybrid_normals(variable: str, region_key: str, file_handler) -> None:
    """
    Process hybrid rolling 30-year climate normals (2015-2044).
    Each target year combines historical and ssp245 data for a 30-year window.
    """
    logger.info(f"Processing hybrid normals for {variable} {region_key}")
    
    # Output directory
    output_dir = Path(OUTPUT_BASE_DIR) / variable / "hybrid"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target years for hybrid normals (where we need historical + ssp245)
    target_years = list(range(2015, 2045))  # 2015-2044
    
    all_normals = []
    all_target_years = []
    
    for target_year in target_years:
        logger.info(f"  Processing hybrid target year {target_year}")
        
        # Use hybrid file collection (combines historical + ssp245)
        all_files, scenario_counts = file_handler.get_hybrid_files_for_period(variable, target_year, 30)
        
        if len(all_files) < MIN_YEARS_FOR_NORMAL:
            logger.warning(f"    Insufficient files ({len(all_files)}) for {target_year}")
            continue
        
        hist_count = scenario_counts['historical']
        ssp245_count = scenario_counts['ssp245']
        logger.info(f"    Using {hist_count} historical + {ssp245_count} SSP245 files")
        
        # Process files to get daily climatologies
        daily_climatologies = []
        years_used = []
        scenarios_used = []
        
        for file_path in all_files:
            year, daily_clim = process_single_file_for_climatology(
                file_path, variable, region_key, file_handler
            )
            if year is not None and daily_clim is not None:
                daily_climatologies.append(daily_clim)
                years_used.append(year)
                
                # Determine scenario from file path
                if 'historical' in str(file_path):
                    scenarios_used.append('historical')
                elif 'ssp245' in str(file_path):
                    scenarios_used.append('ssp245')
                else:
                    scenarios_used.append('unknown')
        
        if len(daily_climatologies) >= MIN_YEARS_FOR_NORMAL:
            logger.info(f"    Computing normal from {len(daily_climatologies)} years")
            
            # Compute 30-year climate normal
            data_arrays = [clim.values for clim in daily_climatologies]
            climate_normal = compute_climate_normal(data_arrays, years_used, target_year)
            
            if climate_normal is not None:
                # Count scenarios actually used
                final_hist_count = scenarios_used.count('historical')
                final_ssp245_count = scenarios_used.count('ssp245')
                
                # Add metadata
                climate_normal.attrs.update({
                    'title': f'{variable.upper()} 30-Year Hybrid Climate Normal ({region_key}) - Target Year {target_year}',
                    'variable': variable,
                    'region': region_key,
                    'target_year': target_year,
                    'period_type': 'hybrid',
                    'period_used': f"{min(years_used)}-{max(years_used)}",
                    'num_years': len(years_used),
                    'historical_years': final_hist_count,
                    'ssp245_years': final_ssp245_count,
                    'creation_date': str(pd.Timestamp.now()),
                    'source': 'NorESM2-LM climate model',
                    'scenarios': 'historical + ssp245',
                    'method': '30-year hybrid rolling climate normal',
                    'processing': 'Sequential processing'
                })
                
                # Save individual file
                output_file = output_dir / f"{variable}_{region_key}_hybrid_{target_year}_30yr_normal.nc"
                climate_normal.to_netcdf(output_file)
                
                # Store for combined dataset
                all_normals.append(climate_normal)
                all_target_years.append(target_year)
                
                logger.info(f"    ✓ Saved hybrid normal for {target_year} ({final_hist_count}H+{final_ssp245_count}S)")
                
                # Cleanup
                del climate_normal, daily_climatologies
                gc.collect()
            else:
                logger.error(f"    ✗ Failed to compute normal for {target_year}")
        else:
            logger.warning(f"    Insufficient processed files ({len(daily_climatologies)}) for {target_year}")
    
    # Create combined dataset
    if all_normals:
        logger.info(f"Creating combined hybrid dataset with {len(all_normals)} normals")
        combined = xr.concat(all_normals, dim='target_year')
        combined = combined.assign_coords(target_year=all_target_years)
        
        # Add global attributes
        combined.attrs.update({
            'title': f'{variable.upper()} Hybrid 30-Year Climate Normals ({region_key}) 2015-2044',
            'description': f'Hybrid 30-year rolling climate normals combining historical and SSP245 data',
            'variable': variable,
            'region': region_key,
            'target_years': f"{min(all_target_years)}-{max(all_target_years)}",
            'num_normals': len(all_target_years),
            'creation_date': str(pd.Timestamp.now()),
            'source': 'NorESM2-LM climate model',
            'scenarios': 'historical + ssp245',
            'method': 'Hybrid rolling 30-year climate normals'
        })
        
        # Save combined file
        combined_file = output_dir / f"{variable}_{region_key}_hybrid_2015-2044_all_normals.nc"
        combined.to_netcdf(combined_file)
        
        logger.info(f"✓ Hybrid normals completed for {variable} {region_key}")
        logger.info(f"  Individual files: {output_dir}/{variable}_{region_key}_hybrid_YYYY_30yr_normal.nc")
        logger.info(f"  Combined file: {combined_file}")


def process_future_normals(variable: str, region_key: str, file_handler, max_year: int) -> None:
    """
    Process future rolling 30-year climate normals (2045+).
    Each target year uses only ssp245 data for a 30-year window.
    """
    logger.info(f"Processing future normals for {variable} {region_key}")
    
    # Output directory
    output_dir = Path(OUTPUT_BASE_DIR) / variable / "ssp245"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target years for future normals (only ssp245 data)
    target_years = list(range(2045, min(max_year + 1, 2101)))
    
    if not target_years:
        logger.info(f"  No future years to process for {variable}")
        return
    
    all_normals = []
    all_target_years = []
    
    for target_year in target_years:
        # 30-year period ending at target_year
        start_year = target_year - 29
        end_year = target_year
        
        logger.info(f"  Processing future target year {target_year} (period: {start_year}-{end_year})")
        
        # Get files for this period (ssp245 only)
        files = file_handler.get_files_for_period(variable, 'ssp245', start_year, end_year)
        
        if len(files) < MIN_YEARS_FOR_NORMAL:
            logger.warning(f"    Insufficient files ({len(files)}) for {target_year}")
            continue
        
        # Process files to get daily climatologies
        daily_climatologies = []
        years_used = []
        
        for file_path in files:
            year, daily_clim = process_single_file_for_climatology(
                file_path, variable, region_key, file_handler
            )
            if year is not None and daily_clim is not None:
                daily_climatologies.append(daily_clim)
                years_used.append(year)
        
        if len(daily_climatologies) >= MIN_YEARS_FOR_NORMAL:
            logger.info(f"    Computing normal from {len(daily_climatologies)} years")
            
            # Compute 30-year climate normal
            data_arrays = [clim.values for clim in daily_climatologies]
            climate_normal = compute_climate_normal(data_arrays, years_used, target_year)
            
            if climate_normal is not None:
                # Add metadata
                climate_normal.attrs.update({
                    'title': f'{variable.upper()} 30-Year Future Climate Normal ({region_key}) - Target Year {target_year}',
                    'variable': variable,
                    'region': region_key,
                    'target_year': target_year,
                    'period_type': 'future',
                    'period_used': f"{min(years_used)}-{max(years_used)}",
                    'num_years': len(years_used),
                    'creation_date': str(pd.Timestamp.now()),
                    'source': 'NorESM2-LM climate model',
                    'scenario': 'ssp245',
                    'method': '30-year rolling climate normal',
                    'processing': 'Sequential processing'
                })
                
                # Save individual file
                output_file = output_dir / f"{variable}_{region_key}_ssp245_{target_year}_30yr_normal.nc"
                climate_normal.to_netcdf(output_file)
                
                # Store for combined dataset
                all_normals.append(climate_normal)
                all_target_years.append(target_year)
                
                logger.info(f"    ✓ Saved future normal for {target_year}")
                
                # Cleanup
                del climate_normal, daily_climatologies
                gc.collect()
            else:
                logger.error(f"    ✗ Failed to compute normal for {target_year}")
        else:
            logger.warning(f"    Insufficient processed files ({len(daily_climatologies)}) for {target_year}")
    
    # Create combined dataset
    if all_normals:
        logger.info(f"Creating combined future dataset with {len(all_normals)} normals")
        combined = xr.concat(all_normals, dim='target_year')
        combined = combined.assign_coords(target_year=all_target_years)
        
        # Add global attributes
        combined.attrs.update({
            'title': f'{variable.upper()} Future 30-Year Climate Normals ({region_key}) {min(all_target_years)}-{max(all_target_years)}',
            'description': f'Future 30-year rolling climate normals using SSP245 data',
            'variable': variable,
            'region': region_key,
            'target_years': f"{min(all_target_years)}-{max(all_target_years)}",
            'num_normals': len(all_target_years),
            'creation_date': str(pd.Timestamp.now()),
            'source': 'NorESM2-LM climate model',
            'scenario': 'ssp245',
            'method': 'Future rolling 30-year climate normals'
        })
        
        # Save combined file
        combined_file = output_dir / f"{variable}_{region_key}_ssp245_{min(all_target_years)}-{max(all_target_years)}_all_normals.nc"
        combined.to_netcdf(combined_file)
        
        logger.info(f"✓ Future normals completed for {variable} {region_key}")
        logger.info(f"  Individual files: {output_dir}/{variable}_{region_key}_ssp245_YYYY_30yr_normal.nc")
        logger.info(f"  Combined file: {combined_file}")


def main():
    """Main processing function."""
    logger.info("🌍 Starting Comprehensive Climate Normals Processing Pipeline")
    logger.info(f"Input directory: {INPUT_DATA_DIR}")
    logger.info(f"Output directory: {OUTPUT_BASE_DIR}")
    logger.info(f"Variables: {VARIABLES}")
    logger.info(f"Regions: {REGIONS}")
    
    # Initialize file handler
    try:
        file_handler = NorESM2FileHandler(INPUT_DATA_DIR)
        
        # Get data availability
        availability = file_handler.validate_data_availability()
        logger.info("\nData availability summary:")
        for var, scenarios in availability.items():
            logger.info(f"  {var}:")
            for scenario, (start, end) in scenarios.items():
                logger.info(f"    {scenario}: {start}-{end}")
        
    except Exception as e:
        logger.error(f"Failed to initialize file handler: {e}")
        return
    
    # Process each variable and region combination
    for variable in VARIABLES:
        for region_key in REGIONS:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Processing {variable.upper()} for {region_key}")
            logger.info(f"{'='*60}")
            
            try:
                # 1. Historical normals (1980-2014)
                logger.info(f"\n📊 Step 1: Historical normals (1980-2014)")
                process_historical_normals(variable, region_key, file_handler)
                
                # 2. Hybrid normals (2015-2044) - combines historical + ssp245
                logger.info(f"\n🔀 Step 2: Hybrid normals (2015-2044)")
                process_hybrid_normals(variable, region_key, file_handler)
                
                # 3. Future normals (2045+) - ssp245 only
                if variable in availability and 'ssp245' in availability[variable]:
                    _, max_year = availability[variable]['ssp245']
                    logger.info(f"\n🚀 Step 3: Future normals (2045-{max_year})")
                    process_future_normals(variable, region_key, file_handler, max_year)
                
                logger.info(f"✅ Completed {variable.upper()} for {region_key}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {variable} {region_key}: {e}")
                continue
    
    logger.info(f"\n🎉 Climate normals processing pipeline completed!")
    logger.info(f"📁 Results saved to: {OUTPUT_BASE_DIR}/")


if __name__ == "__main__":
    main() 