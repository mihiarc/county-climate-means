#!/usr/bin/env python3
"""
Climate Data Processing Runner

This script handles the execution of climate data processing workflows using the
core functionality provided by climate_means.py.

Usage:
    python run_climate_means.py                # Run basic demonstration
    python run_climate_means.py noresm2        # Process NorESM2-LM data
    python run_climate_means.py example        # Run example workflow
    python run_climate_means.py help           # Show usage information
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# Import all the necessary functions and classes from climate_means
from climate_means import (
    # Regional utilities
    REGION_BOUNDS,
    validate_region_bounds,
    
    # Time utilities
    generate_climate_periods,
    
    # I/O utilities
    NorESM2FileHandler,
    
    # Core processing workflow
    process_climate_data_workflow,
    
    # Logger
    logger
)

def main():
    """Main function demonstrating the consolidated climate processing functionality."""
    logger.info("Starting Climate Data Processing - Sequential Processing")
    
    # Example configuration (for logging/monitoring purposes)
    config = {
        'processing_type': 'sequential',
        'max_workers': 1,           # Sequential processing
        'batch_size': 10,
        'memory_conservative': True
    }
    
    try:
        # Display available regions
        logger.info("Available regions:")
        for region_key, region_info in REGION_BOUNDS.items():
            logger.info(f"  {region_key}: {region_info['name']}")
        
        # Example: validate a region
        if validate_region_bounds('CONUS'):
            logger.info("CONUS region validation passed")
        
        # Example: generate climate periods
        data_availability = {
            'historical': {'start': 1950, 'end': 2014},
            'ssp245': {'start': 2015, 'end': 2100}
        }
        
        periods = generate_climate_periods('historical', data_availability)
        logger.info(f"Generated {len(periods)} historical periods")
        
        # Test NorESM2 file handler if data is available
        noresm2_path = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
        if Path(noresm2_path).exists():
            logger.info("Testing NorESM2-LM file handler...")
            try:
                file_handler = NorESM2FileHandler(noresm2_path)
                availability = file_handler.validate_data_availability()
                logger.info("NorESM2-LM data availability validated successfully")
            except Exception as e:
                logger.warning(f"Could not validate NorESM2-LM data: {e}")
        else:
            logger.info("NorESM2-LM data path not found - skipping validation")
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise
    
    logger.info("Climate data processing demonstration completed successfully")


def example_usage():
    """
    Example usage of the climate processing functions.
    Modify the paths and parameters according to your data setup.
    """
    
    # Configuration for sequential processing
    config = {
        'processing_type': 'sequential',     # Sequential processing approach
        'memory_conservative': True,         # Conservative memory usage
        'batch_size': 15,                   # Batch size for processing
        'max_retries': 3                    # Maximum retry attempts
    }
    
    # Define what to process
    variables = ['tas', 'tasmax', 'tasmin', 'pr']  # Temperature and precipitation
    regions = ['CONUS', 'AK', 'HI']                # US regions
    scenarios = ['historical', 'ssp245', 'ssp585'] # Climate scenarios
    
    # Data and output directories (modify these paths)
    data_directory = "/path/to/your/climate/netcdf/files"
    output_directory = "/path/to/output/climate/normals"
    
    # Run the complete workflow
    try:
        process_climate_data_workflow(
            data_directory=data_directory,
            output_directory=output_directory,
            variables=variables,
            regions=regions,
            scenarios=scenarios,
            config=config
        )
        print("Climate data processing completed successfully!")
        
    except Exception as e:
        print(f"Error in climate data processing: {e}")
        logger.error(f"Processing failed: {e}")


def process_noresm2_data():
    """
    Process NorESM2-LM climate data specifically.
    This function is configured for the NorESM2-LM dataset structure.
    """
    
    # Configuration optimized for NorESM2-LM data (sequential processing)
    config = {
        'processing_type': 'sequential',     # Sequential processing approach
        'memory_conservative': True,         # Very conservative memory usage
        'batch_size': 5,                    # Smaller batches (5 years instead of 10)
        'max_retries': 3                    # Retry failed computations
    }
    
    # NorESM2-LM data configuration
    data_directory = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
    output_directory = "/home/mihiarc/repos/county_climate/output/noresm2_climate_normals"
    
    # Focus on specific request: precipitation (pr) for historical CONUS
    variables = ['pr']  # Only precipitation
    regions = ['CONUS']  # Only CONUS region
    scenarios = ['historical']  # Only historical scenario
    
    logger.info("Starting NorESM2-LM climate data processing (sequential)")
    logger.info(f"Data directory: {data_directory}")
    logger.info(f"Output directory: {output_directory}")
    logger.info(f"Variables: {variables}")
    logger.info(f"Regions: {regions}")
    logger.info(f"Scenarios: {scenarios}")
    
    try:
        # Validate data availability first
        file_handler = NorESM2FileHandler(data_directory)
        data_availability = file_handler.validate_data_availability()
        
        logger.info("Data availability summary:")
        for variable, scenarios_data in data_availability.items():
            for scenario, (start_year, end_year) in scenarios_data.items():
                logger.info(f"  {variable} {scenario}: {start_year}-{end_year}")
        
        # Run the complete workflow
        process_climate_data_workflow(
            data_directory=data_directory,
            output_directory=output_directory,
            variables=variables,
            regions=regions,
            scenarios=scenarios,
            config=config
        )
        
        logger.info("✓ NorESM2-LM climate data processing completed successfully!")
        print("✓ NorESM2-LM climate data processing completed successfully!")
        
    except Exception as e:
        error_msg = f"Error processing NorESM2-LM data: {e}"
        logger.error(error_msg)
        print(f"✗ {error_msg}")
        raise


def test_noresm2_small():
    """
    Test NorESM2 processing with a small subset of files.
    """
    
    # Configuration for testing (sequential processing)
    config = {
        'processing_type': 'sequential',     # Sequential processing approach  
        'memory_conservative': True,         # Very conservative memory usage
        'batch_size': 3,                    # Small batches for testing
        'max_retries': 2                    # Retry failed computations
    }
    
    data_directory = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
    
    logger.info("Starting NorESM2-LM small test (sequential processing)")
    logger.info(f"Data directory: {data_directory}")
    
    try:
        # Initialize file handler
        file_handler = NorESM2FileHandler(data_directory)
        
        # Get a few files to test with (first 5 years)
        files = file_handler.get_files_for_period('pr', 'historical', 1950, 1954)
        logger.info(f"Testing with {len(files)} files for years 1950-1954")
        
        if len(files) < 3:
            logger.error("Not enough files found for testing")
            return
        
        try:
            # Test processing one file
            first_file = files[0]
            logger.info(f"Testing with single file: {first_file}")
            
            # Simple test - just open and check the file
            from io_util import open_dataset_safely
            ds = open_dataset_safely(first_file)
            if ds:
                logger.info(f"Successfully opened file. Variables: {list(ds.data_vars)}")
                logger.info(f"Dimensions: {dict(ds.sizes)}")
                if 'pr' in ds.data_vars:
                    logger.info(f"Precipitation variable shape: {ds.pr.shape}")
                ds.close()
            else:
                logger.error("Failed to open test file")
            
        except Exception as e:
            logger.error(f"Error in file test: {e}")
        
        logger.info("✓ NorESM2-LM test completed successfully!")
        print("✓ NorESM2-LM test completed successfully!")
        
    except Exception as e:
        error_msg = f"Error in NorESM2-LM test: {e}"
        logger.error(error_msg)
        print(f"✗ {error_msg}")
        raise


def print_help():
    """Print usage information."""
    print("Climate Data Processing Runner")
    print("=" * 50)
    print()
    print("Usage:")
    print("  python run_climate_means.py           # Run basic demonstration")
    print("  python run_climate_means.py noresm2   # Process NorESM2-LM data")
    print("  python run_climate_means.py test      # Test with small NorESM2 subset")
    print("  python run_climate_means.py diagnose  # Diagnose NorESM2 files for problems")
    print("  python run_climate_means.py simple    # Simple processing (sequential)")
    print("  python run_climate_means.py rolling   # Calculate rolling 30-year climate normals (1980-2014)")
    print("  python run_climate_means.py ssp245    # Calculate hybrid 30-year normals (2015-2100) using historical+SSP245")
    print("  python run_climate_means.py example   # Run example workflow")
    print("  python run_climate_means.py help      # Show this help")
    print()
    print("Description:")
    print("  This script provides different modes for processing climate data:")
    print()
    print("  - Basic demo: Tests core functionality and validates setup")
    print("  - NorESM2: Processes NorESM2-LM climate model data")
    print("  - Test: Processes small subset of NorESM2 data for validation")
    print("  - Example: Shows how to customize processing for your data")
    print()
    print("Processing Approach:")
    print("  All processing uses sequential (non-distributed) approach for reliability")
    print("  with NetCDF files. This avoids thread-safety issues and provides")
    print("  better memory management for climate data processing.")
    print()
    print("Configuration:")
    print("  Edit the configuration dictionaries in each function to customize:")
    print("  - Processing type (sequential)")
    print("  - Memory usage (conservative memory management)")
    print("  - Processing parameters (batch_size)")
    print("  - Input/output directories")
    print("  - Variables, regions, and scenarios to process")


def diagnose_noresm2_files():
    """
    Diagnose NorESM2 files to identify problematic ones without using Dask.
    This will help identify corrupted files or other issues.
    """
    
    data_directory = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
    
    logger.info("Starting NorESM2-LM file diagnosis (no Dask)")
    logger.info(f"Data directory: {data_directory}")
    
    try:
        # Initialize file handler
        file_handler = NorESM2FileHandler(data_directory)
        
        # Get first few precipitation files
        files = file_handler.get_files_for_period('pr', 'historical', 1950, 1954)
        logger.info(f"Found {len(files)} files for 1950-1954")
        
        good_files = []
        bad_files = []
        
        for i, file_path in enumerate(files):
            logger.info(f"Testing file {i+1}/{len(files)}: {Path(file_path).name}")
            
            try:
                # Try to open without chunks first
                import xarray as xr
                ds = xr.open_dataset(file_path, decode_times=False)
                
                # Basic checks
                logger.info(f"  Variables: {list(ds.data_vars)}")
                logger.info(f"  Dimensions: {dict(ds.sizes)}")
                
                if 'pr' in ds.data_vars:
                    pr = ds.pr
                    logger.info(f"  PR shape: {pr.shape}")
                    logger.info(f"  PR dtype: {pr.dtype}")
                    
                    # Try to load a small subset
                    sample = pr.isel(time=0, lat=slice(0, 10), lon=slice(0, 10))
                    values = sample.load()
                    logger.info(f"  Sample loaded successfully: {values.shape}")
                    
                    good_files.append(file_path)
                    logger.info(f"  ✓ File OK")
                else:
                    logger.warning(f"  No 'pr' variable found")
                    bad_files.append(file_path)
                
                ds.close()
                
            except Exception as e:
                logger.error(f"  ✗ File failed: {e}")
                bad_files.append(file_path)
        
        logger.info(f"\nDiagnosis complete:")
        logger.info(f"  Good files: {len(good_files)}")
        logger.info(f"  Bad files: {len(bad_files)}")
        
        if bad_files:
            logger.warning("Problematic files:")
            for bad_file in bad_files:
                logger.warning(f"  - {Path(bad_file).name}")
        
        if good_files:
            logger.info("Good files:")
            for good_file in good_files[:3]:  # Show first 3
                logger.info(f"  - {Path(good_file).name}")
        
        return good_files, bad_files
        
    except Exception as e:
        logger.error(f"Error in file diagnosis: {e}")
        raise


def simple_noresm2_processing():
    """
    Simple processing of NorESM2 data without Dask for the full historical period.
    This will process precipitation data for CONUS from 1950-2014.
    """
    
    data_directory = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
    output_directory = "/home/mihiarc/repos/county_climate/output/full_historical_processing"
    
    logger.info("Starting full historical NorESM2-LM processing (no Dask)")
    logger.info(f"Data directory: {data_directory}")
    logger.info(f"Output directory: {output_directory}")
    
    # Create output directory
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize file handler
        file_handler = NorESM2FileHandler(data_directory)
        
        # Get full historical period files
        files = file_handler.get_files_for_period('pr', 'historical', 1950, 2014)
        logger.info(f"Found {len(files)} files for full historical period (1950-2014)")
        
        # Import necessary modules
        import xarray as xr
        from regions import REGION_BOUNDS, extract_region
        from time_util import handle_time_coordinates
        
        daily_climatologies = []
        years = []
        
        # Process in batches to manage memory
        batch_size = 10  # Process 10 years at a time
        total_files = len(files)
        
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = files[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}: files {batch_start+1}-{batch_end}")
            
            for i, file_path in enumerate(batch_files):
                file_idx = batch_start + i + 1
                logger.info(f"  Processing file {file_idx}/{total_files}: {Path(file_path).name}")
                
                try:
                    # Open file
                    ds = xr.open_dataset(file_path, decode_times=False)
                    
                    # Handle time coordinates
                    ds, time_method = handle_time_coordinates(ds, file_path)
                    
                    # Extract CONUS region
                    conus_bounds = REGION_BOUNDS['CONUS']
                    region_ds = extract_region(ds, conus_bounds)
                    
                    # Get precipitation variable
                    pr = region_ds.pr
                    
                    # Calculate daily climatology (mean for each day of year)
                    if 'dayofyear' in pr.coords:
                        daily_clim = pr.groupby(pr.dayofyear).mean(dim='time')
                        
                        # Store for later use
                        daily_climatologies.append(daily_clim)
                        year = file_handler.extract_year_from_filename(file_path)
                        years.append(year)
                        
                        logger.info(f"    ✓ Added climatology for year {year} (shape: {daily_clim.shape})")
                    else:
                        logger.warning(f"    No dayofyear coordinate found")
                    
                    ds.close()
                    
                except Exception as e:
                    logger.error(f"    ✗ Failed to process: {e}")
            
            # Log progress
            logger.info(f"  Batch complete. Total climatologies collected: {len(daily_climatologies)}")
            
            # Garbage collection after each batch
            import gc
            gc.collect()
        
        # Create climate normal from all years
        if len(daily_climatologies) >= 30:  # Need at least 30 years for climate normal
            logger.info(f"Creating climate normal from {len(daily_climatologies)} years of data")
            
            # Combine all daily climatologies in smaller chunks to manage memory
            chunk_size = 20
            combined_chunks = []
            
            for chunk_start in range(0, len(daily_climatologies), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(daily_climatologies))
                chunk_clims = daily_climatologies[chunk_start:chunk_end]
                chunk_years = years[chunk_start:chunk_end]
                
                logger.info(f"  Processing chunk {chunk_start//chunk_size + 1}: years {chunk_years[0]}-{chunk_years[-1]}")
                
                # Combine this chunk
                chunk_combined = xr.concat(chunk_clims, dim='year')
                chunk_combined = chunk_combined.assign_coords(year=chunk_years)
                
                # Calculate mean for this chunk
                chunk_mean = chunk_combined.mean(dim='year')
                combined_chunks.append(chunk_mean)
                
                # Clean up
                del chunk_combined, chunk_clims
                gc.collect()
            
            # Now combine the chunk means
            logger.info(f"Combining {len(combined_chunks)} chunk means into final climate normal")
            final_combined = xr.concat(combined_chunks, dim='chunk')
            climate_normal = final_combined.mean(dim='chunk')
            
            # Add comprehensive metadata
            climate_normal.attrs.update({
                'title': 'Precipitation Climate Normal (CONUS) - Full Historical Period',
                'variable': 'pr',
                'region': 'CONUS',
                'years_used': f"{min(years)}-{max(years)}",
                'num_years': len(years),
                'creation_date': str(pd.Timestamp.now()),
                'units': 'kg m-2 s-1',
                'source': 'NorESM2-LM climate model',
                'scenario': 'historical',
                'method': 'Daily climatology averaged over years',
                'processing': 'Sequential processing without Dask'
            })
            
            # Save result
            output_file = output_path / f"pr_CONUS_historical_{min(years)}-{max(years)}_climate_normal.nc"
            climate_normal.to_netcdf(output_file)
            
            logger.info(f"✓ Climate normal saved to: {output_file}")
            logger.info(f"  Shape: {climate_normal.shape}")
            logger.info(f"  Data range: {float(climate_normal.min()):.2e} to {float(climate_normal.max()):.2e}")
            logger.info(f"  Years processed: {min(years)}-{max(years)} ({len(years)} years)")
            
            print(f"✓ Full historical processing completed successfully!")
            print(f"  Processed {len(years)} years of data ({min(years)}-{max(years)})")
            print(f"  Output: {output_file}")
            print(f"  Climate normal shape: {climate_normal.shape}")
            
        else:
            logger.warning(f"Not enough files processed ({len(daily_climatologies)}) to create climate normal (need ≥30)")
        
    except Exception as e:
        logger.error(f"Error in full historical processing: {e}")
        raise


def rolling_30year_climate_normals():
    """
    Calculate rolling 30-year climate normals for each year from 1980-2014.
    For each target year, calculates a climate normal based on the preceding 30 years.
    
    Example:
    - Year 1980: climate normal from 1951-1980
    - Year 1981: climate normal from 1952-1981
    - ...
    - Year 2014: climate normal from 1985-2014
    """
    
    data_directory = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
    output_directory = "/home/mihiarc/repos/county_climate/output/rolling_30year_normals"
    
    logger.info("Starting rolling 30-year climate normals calculation")
    logger.info(f"Data directory: {data_directory}")
    logger.info(f"Output directory: {output_directory}")
    
    # Create output directory
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize file handler
        file_handler = NorESM2FileHandler(data_directory)
        
        # Import necessary modules
        import xarray as xr
        from regions import REGION_BOUNDS, extract_region
        from time_util import handle_time_coordinates
        import gc
        
        # Target years (1980-2014)
        target_years = list(range(1980, 2015))
        logger.info(f"Calculating rolling normals for {len(target_years)} target years: {target_years[0]}-{target_years[-1]}")
        
        all_rolling_normals = []
        all_target_years = []
        
        for target_year in target_years:
            # Calculate the 30-year period ending at target_year
            start_year = target_year - 29  # 30 years including target_year
            end_year = target_year
            
            logger.info(f"Processing target year {target_year}: using data from {start_year}-{end_year}")
            
            # Get files for this 30-year period
            files = file_handler.get_files_for_period('pr', 'historical', start_year, end_year)
            
            if len(files) < 25:  # Need at least 25 years for a meaningful climate normal
                logger.warning(f"  Only {len(files)} files found for {start_year}-{end_year}, skipping")
                continue
                
            logger.info(f"  Found {len(files)} files for 30-year period")
            
            # Process files for this period
            daily_climatologies = []
            years_used = []
            
            for i, file_path in enumerate(files):
                try:
                    # Open file
                    ds = xr.open_dataset(file_path, decode_times=False)
                    
                    # Handle time coordinates
                    ds, time_method = handle_time_coordinates(ds, file_path)
                    
                    # Extract CONUS region
                    conus_bounds = REGION_BOUNDS['CONUS']
                    region_ds = extract_region(ds, conus_bounds)
                    
                    # Get precipitation variable
                    pr = region_ds.pr
                    
                    # Calculate daily climatology
                    if 'dayofyear' in pr.coords:
                        daily_clim = pr.groupby(pr.dayofyear).mean(dim='time')
                        daily_climatologies.append(daily_clim)
                        
                        year = file_handler.extract_year_from_filename(file_path)
                        years_used.append(year)
                    
                    ds.close()
                    
                except Exception as e:
                    logger.error(f"    Failed to process file {Path(file_path).name}: {e}")
            
            if len(daily_climatologies) >= 25:  # Need at least 25 years
                logger.info(f"  Creating climate normal from {len(daily_climatologies)} years")
                
                # Combine climatologies for this 30-year period
                combined = xr.concat(daily_climatologies, dim='year')
                combined = combined.assign_coords(year=years_used)
                
                # Calculate mean across years
                climate_normal = combined.mean(dim='year')
                
                # Add metadata
                climate_normal.attrs.update({
                    'title': f'30-Year Rolling Precipitation Climate Normal (CONUS) - Target Year {target_year}',
                    'variable': 'pr',
                    'region': 'CONUS',
                    'target_year': target_year,
                    'period_used': f"{min(years_used)}-{max(years_used)}",
                    'num_years': len(years_used),
                    'creation_date': str(pd.Timestamp.now()),
                    'units': 'kg m-2 s-1',
                    'source': 'NorESM2-LM climate model',
                    'scenario': 'historical',
                    'method': '30-year rolling climate normal',
                    'processing': 'Sequential processing without Dask'
                })
                
                # Save individual file
                individual_file = output_path / f"pr_CONUS_rolling30yr_{target_year}_normal.nc"
                climate_normal.to_netcdf(individual_file)
                
                # Store for combined dataset
                all_rolling_normals.append(climate_normal)
                all_target_years.append(target_year)
                
                logger.info(f"  ✓ Saved rolling normal for {target_year} ({len(years_used)} years used)")
                
                # Clean up memory
                del combined, daily_climatologies, climate_normal
                gc.collect()
            else:
                logger.warning(f"  Insufficient data for {target_year} ({len(daily_climatologies)} years)")
        
        # Create combined dataset with all rolling normals
        if len(all_rolling_normals) > 0:
            logger.info(f"Creating combined dataset with {len(all_rolling_normals)} rolling normals")
            
            # Combine all rolling normals along a new time dimension
            combined_rolling = xr.concat(all_rolling_normals, dim='target_year')
            combined_rolling = combined_rolling.assign_coords(target_year=all_target_years)
            
            # Add global attributes
            combined_rolling.attrs.update({
                'title': 'Rolling 30-Year Precipitation Climate Normals (CONUS) 1980-2014',
                'description': 'Rolling 30-year climate normals for each target year from 1980-2014',
                'variable': 'pr',
                'region': 'CONUS',
                'target_years': f"{min(all_target_years)}-{max(all_target_years)}",
                'num_target_years': len(all_target_years),
                'window_length': '30 years',
                'creation_date': str(pd.Timestamp.now()),
                'units': 'kg m-2 s-1',
                'source': 'NorESM2-LM climate model',
                'scenario': 'historical',
                'method': 'Rolling 30-year climate normals',
                'processing': 'Sequential processing without Dask'
            })
            
            # Save combined file
            combined_file = output_path / f"pr_CONUS_rolling30yr_1980-2014_all_normals.nc"
            combined_rolling.to_netcdf(combined_file)
            
            logger.info(f"✓ Combined rolling normals saved to: {combined_file}")
            logger.info(f"  Shape: {combined_rolling.shape}")
            logger.info(f"  Target years: {min(all_target_years)}-{max(all_target_years)}")
            logger.info(f"  Number of rolling normals: {len(all_target_years)}")
            
            print(f"✓ Rolling 30-year climate normals completed successfully!")
            print(f"  Created {len(all_target_years)} rolling normals for years {min(all_target_years)}-{max(all_target_years)}")
            print(f"  Individual files: {output_path}/pr_CONUS_rolling30yr_YYYY_normal.nc")
            print(f"  Combined file: {combined_file}")
            print(f"  Combined dataset shape: {combined_rolling.shape}")
            
        else:
            logger.error("No rolling normals were successfully created")
        
    except Exception as e:
        logger.error(f"Error in rolling climate normals calculation: {e}")
        raise


def hybrid_30year_climate_normals():
    """
    Calculate rolling 30-year climate normals for SSP245 scenario (2015-2100).
    For early target years (2015-2043), combines historical and SSP245 data.
    For later target years (2044-2100), uses only SSP245 data.
    
    Example:
    - Target year 2015: historical (1986-2014) + SSP245 (2015) = 30 years
    - Target year 2020: historical (1991-2014) + SSP245 (2015-2020) = 30 years
    - Target year 2043: historical (2014) + SSP245 (2015-2043) = 30 years
    - Target year 2044: SSP245 (2015-2044) = 30 years
    - Target year 2100: SSP245 (2071-2100) = 30 years
    """
    
    data_directory = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
    output_directory = "/home/mihiarc/repos/county_climate/output/hybrid_30year_normals"
    
    logger.info("Starting hybrid 30-year climate normals calculation (Historical + SSP245)")
    logger.info(f"Data directory: {data_directory}")
    logger.info(f"Output directory: {output_directory}")
    
    # Create output directory
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize file handler
        file_handler = NorESM2FileHandler(data_directory)
        
        # Check data availability for both scenarios
        logger.info("Checking data availability for both scenarios...")
        data_availability = file_handler.validate_data_availability()
        
        # Get historical and SSP245 date ranges
        hist_start, hist_end = data_availability['pr']['historical']
        ssp245_start, ssp245_end = data_availability['pr']['ssp245']
        
        logger.info(f"Historical data: {hist_start}-{hist_end}")
        logger.info(f"SSP245 data: {ssp245_start}-{ssp245_end}")
        
        # Import necessary modules
        import xarray as xr
        from regions import REGION_BOUNDS, extract_region
        from time_util import handle_time_coordinates
        import gc
        
        # Target years (start from 2015, end at available SSP245 data)
        target_years = list(range(2015, min(2101, ssp245_end + 1)))
        logger.info(f"Calculating rolling normals for {len(target_years)} target years: {target_years[0]}-{target_years[-1]}")
        
        all_rolling_normals = []
        all_target_years = []
        
        for target_year in target_years:
            logger.info(f"Processing target year {target_year}")
            
            # Use the new hybrid file collection method
            all_files, scenario_counts = file_handler.get_hybrid_files_for_period('pr', target_year, 30)
            
            if len(all_files) < 25:  # Need at least 25 years for a meaningful climate normal
                logger.warning(f"  Only {len(all_files)} files found for target year {target_year}, skipping")
                continue
            
            hist_count = scenario_counts['historical']
            ssp245_count = scenario_counts['ssp245']
            ssp585_count = scenario_counts['ssp585']
            
            logger.info(f"  Using {hist_count} historical + {ssp245_count} SSP245 + {ssp585_count} SSP585 files")
            
            # Process files for this period
            daily_climatologies = []
            years_used = []
            scenarios_used = []
            
            for i, file_path in enumerate(all_files):
                if i % 10 == 0:  # Progress logging every 10 files
                    logger.info(f"    Processing file {i+1}/{len(all_files)}")
                
                try:
                    # Open file
                    ds = xr.open_dataset(file_path, decode_times=False)
                    
                    # Handle time coordinates
                    ds, time_method = handle_time_coordinates(ds, file_path)
                    
                    # Extract CONUS region
                    conus_bounds = REGION_BOUNDS['CONUS']
                    region_ds = extract_region(ds, conus_bounds)
                    
                    # Get precipitation variable
                    pr = region_ds.pr
                    
                    # Calculate daily climatology
                    if 'dayofyear' in pr.coords:
                        daily_clim = pr.groupby(pr.dayofyear).mean(dim='time')
                        daily_climatologies.append(daily_clim)
                        
                        year = file_handler.extract_year_from_filename(file_path)
                        years_used.append(year)
                        
                        # Determine scenario based on filename
                        if 'historical' in str(file_path):
                            scenarios_used.append('historical')
                        elif 'ssp245' in str(file_path):
                            scenarios_used.append('ssp245')
                        elif 'ssp585' in str(file_path):
                            scenarios_used.append('ssp585')
                        else:
                            scenarios_used.append('unknown')
                    
                    ds.close()
                    
                except Exception as e:
                    logger.error(f"    Failed to process file {Path(file_path).name}: {e}")
            
            if len(daily_climatologies) >= 25:  # Need at least 25 years
                logger.info(f"  Creating climate normal from {len(daily_climatologies)} years")
                
                # Count scenarios actually used (may differ slightly from file counts due to processing errors)
                final_hist_count = scenarios_used.count('historical')
                final_ssp245_count = scenarios_used.count('ssp245')
                final_ssp585_count = scenarios_used.count('ssp585')
                logger.info(f"    Successfully processed: {final_hist_count} hist + {final_ssp245_count} ssp245 + {final_ssp585_count} ssp585")
                
                # Combine climatologies for this 30-year period
                combined = xr.concat(daily_climatologies, dim='year')
                combined = combined.assign_coords(year=years_used)
                
                # Calculate mean across years
                climate_normal = combined.mean(dim='year')
                
                # Add metadata
                climate_normal.attrs.update({
                    'title': f'30-Year Hybrid Climate Normal (CONUS) - Target Year {target_year}',
                    'variable': 'pr',
                    'region': 'CONUS',
                    'target_year': target_year,
                    'period_used': f"{min(years_used)}-{max(years_used)}",
                    'num_years': len(years_used),
                    'historical_years': final_hist_count,
                    'ssp245_years': final_ssp245_count,
                    'ssp585_years': final_ssp585_count,
                    'creation_date': str(pd.Timestamp.now()),
                    'units': 'kg m-2 s-1',
                    'source': 'NorESM2-LM climate model',
                    'scenarios': 'historical + ssp245 + ssp585 (as needed)',
                    'method': '30-year hybrid rolling climate normal',
                    'processing': 'Sequential processing without Dask'
                })
                
                # Save individual file
                individual_file = output_path / f"pr_CONUS_hybrid_rolling30yr_{target_year}_normal.nc"
                climate_normal.to_netcdf(individual_file)
                
                # Store for combined dataset
                all_rolling_normals.append(climate_normal)
                all_target_years.append(target_year)
                
                logger.info(f"  ✓ Saved hybrid rolling normal for {target_year} ({len(years_used)} years total)")
                
                # Clean up memory
                del combined, daily_climatologies, climate_normal
                gc.collect()
            else:
                logger.warning(f"  Insufficient data for {target_year} ({len(daily_climatologies)} years)")
        
        # Create combined dataset with all rolling normals
        if len(all_rolling_normals) > 0:
            logger.info(f"Creating combined dataset with {len(all_rolling_normals)} rolling normals")
            
            # Combine all rolling normals along a new time dimension
            combined_rolling = xr.concat(all_rolling_normals, dim='target_year')
            combined_rolling = combined_rolling.assign_coords(target_year=all_target_years)
            
            # Add global attributes
            combined_rolling.attrs.update({
                'title': f'Hybrid 30-Year Climate Normals (CONUS) {min(all_target_years)}-{max(all_target_years)}',
                'description': f'Hybrid 30-year climate normals combining historical and SSP245 data, target years {min(all_target_years)}-{max(all_target_years)}',
                'variable': 'pr',
                'region': 'CONUS',
                'target_years': f"{min(all_target_years)}-{max(all_target_years)}",
                'num_target_years': len(all_target_years),
                'window_length': '30 years',
                'creation_date': str(pd.Timestamp.now()),
                'units': 'kg m-2 s-1',
                'source': 'NorESM2-LM climate model',
                'scenarios': 'historical + ssp245 + ssp585',
                'method': 'Hybrid rolling 30-year climate normals',
                'processing': 'Sequential processing without Dask'
            })
            
            # Save combined file
            combined_file = output_path / f"pr_CONUS_hybrid_rolling30yr_{min(all_target_years)}-{max(all_target_years)}_all_normals.nc"
            combined_rolling.to_netcdf(combined_file)
            
            logger.info(f"✓ Combined hybrid rolling normals saved to: {combined_file}")
            logger.info(f"  Shape: {combined_rolling.shape}")
            logger.info(f"  Target years: {min(all_target_years)}-{max(all_target_years)}")
            logger.info(f"  Number of rolling normals: {len(all_target_years)}")
            
            print(f"✓ Hybrid rolling 30-year climate normals completed successfully!")
            print(f"  Created {len(all_target_years)} rolling normals for years {min(all_target_years)}-{max(all_target_years)}")
            print(f"  Combines historical (1950-2014) and SSP245 (2015-2100) data as needed")
            print(f"  Individual files: {output_path}/pr_CONUS_hybrid_rolling30yr_YYYY_normal.nc")
            print(f"  Combined file: {combined_file}")
            print(f"  Combined dataset shape: {combined_rolling.shape}")
            
        else:
            logger.error("No hybrid rolling normals were successfully created")
        
    except Exception as e:
        logger.error(f"Error in hybrid rolling climate normals calculation: {e}")
        raise


if __name__ == "__main__":
    # Check command line arguments for specific processing modes
    if len(sys.argv) > 1:
        if sys.argv[1] == "noresm2":
            # Process NorESM2-LM data
            process_noresm2_data()
        elif sys.argv[1] == "example":
            # Run the example workflow
            example_usage()
        elif sys.argv[1] == "help":
            print_help()
        elif sys.argv[1] == "test":
            # Run the test with small NorESM2 subset
            test_noresm2_small()
        elif sys.argv[1] == "diagnose":
            # Run the diagnose function
            diagnose_noresm2_files()
        elif sys.argv[1] == "simple":
            # Run the simple processing function
            simple_noresm2_processing()
        elif sys.argv[1] == "rolling":
            # Run the rolling 30-year climate normals function
            rolling_30year_climate_normals()
        elif sys.argv[1] == "ssp245":
            # Run the hybrid 30-year climate normals function
            hybrid_30year_climate_normals()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use 'python run_climate_means.py help' for usage information")
    else:
        # Run the basic demonstration
        main() 