#!/usr/bin/env python3
"""
Main orchestration script for county-level climate metrics processing.

This script handles all climate data processing operations using the consolidated
climate metrics functions and parallel processing framework:
- Single file processing for testing/prototyping
- Full time series processing (1980-2100) for production
- Batch processing across multiple regions
- All climate variables (temperature, precipitation)
- Comprehensive "run-all" mode for complete dataset processing

Processing Modes:
    single      - Process one NetCDF file for testing/validation
    timeseries  - Process time series for specific variable/region combinations  
    run-all     - Process complete dataset (all variables, regions, scenarios)

Usage examples:
    # Process single file for testing
    python -m metrics.main --mode single --variable pr --file "pr_CONUS_historical_1980_30yr_normal.nc" --region CONUS
    
    # Process full time series for production
    python -m metrics.main --mode timeseries --variable pr --scenario historical --region CONUS
    
    # Process all metrics for a variable across regions
    python -m metrics.main --mode timeseries --variable tas --scenario historical --region all --metrics all
    
    # Process multiple regions  
    python -m metrics.main --mode timeseries --variable tas --scenario historical --region CONUS,AK --metrics all
    
    # COMPREHENSIVE PROCESSING - Run everything
    python -m metrics.main --mode run-all --output-dir output/production
    
    # Run-all with custom time range and resume capability
    python -m metrics.main --mode run-all --year-range 1980 2100 --run-all-resume --run-all-summary
    
    # Run-all for specific time period with detailed progress
    python -m metrics.main --mode run-all --year-range 2020 2100 --workers 32 --verbose

Complete Dataset Specifications:
    Variables: pr (precipitation), tas (mean temp), tasmin (min temp), tasmax (max temp)
    Regions: CONUS, AK (Alaska), HI (Hawaii), PRVI (Puerto Rico/VI), GU (Guam)  
    Scenarios: historical (1950-2014), ssp245 (2015-2100), ssp585 (2015-2100)
    Metrics: All available metrics per variable (annual values, extremes, percentiles)
    Time range: 1950-2100 (151 years, 3,020 total NetCDF files)
"""

import argparse
import os
import sys
import glob
from pathlib import Path
from typing import List, Optional
import pandas as pd

# Import our consolidated utilities and functions
from .utils import (
    load_us_counties, get_available_regions, get_available_metrics,
    validate_region_variable_combination, process_region_timeseries,
    get_optimal_worker_count, get_calculation_functions
)
from .core.precipitation import calculate_annual_precipitation, calculate_high_precip_days_95th
from .core.temperature import (
    calculate_annual_mean_temperature, calculate_growing_degree_days,
    calculate_high_temperature_days_90th, calculate_low_temperature_days_10th
)
from .utils.data_config import get_data_files, filter_files_by_scenario, filter_files_by_year_range, DATA_CONFIG


def setup_paths():
    """Setup paths for data and outputs."""
    base_path = Path(__file__).parent.parent.parent  # Go up from src/metrics/ to project root
    
    paths = {
        'counties_shapefile': base_path / "county_climate_metrics" / "tl_2024_us_county" / "tl_2024_us_county.shp",
        'output_base': base_path / "output",
        'climate_normals_base': Path("output")
    }
    
    return paths


def process_single_file(variable: str, file_path: str, region: str, output_dir: str, 
                       counties_shapefile: str, metrics: List[str]):
    """
    Process a single NetCDF file for testing/prototyping.
    
    Args:
        variable (str): Climate variable (pr, tas, tasmin, tasmax)
        file_path (str): Path to NetCDF file
        region (str): Geographic region
        output_dir (str): Output directory
        counties_shapefile (str): Path to county shapefile
        metrics (List[str]): List of metrics to calculate
    """
    print(f"=== SINGLE FILE PROCESSING ===")
    print(f"Variable: {variable}")
    print(f"File: {file_path}")
    print(f"Region: {region}")
    print(f"Metrics: {', '.join(metrics)}")
    print()
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NetCDF file not found: {file_path}")
    
    if not os.path.exists(counties_shapefile):
        raise FileNotFoundError(f"Counties shapefile not found: {counties_shapefile}")
    
    # Validate region-variable combination
    if not validate_region_variable_combination(region, variable):
        raise ValueError(f"Region {region} does not have {variable} data available")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load counties and filter for region
    from .utils import filter_counties_by_region, load_and_prepare_netcdf
    
    print("Loading county boundaries...")
    counties = load_us_counties(counties_shapefile)
    counties_region = filter_counties_by_region(counties, region)
    
    print(f"Loading NetCDF file: {file_path}")
    ds = load_and_prepare_netcdf(file_path, region=region)
    
    # Process each requested metric
    results = []
    for metric in metrics:
        if variable in get_calculation_functions() and metric in get_calculation_functions()[variable]:
            print(f"Calculating {metric}...")
            calc_func = get_calculation_functions()[variable][metric]
            result_df = calc_func(ds, counties_region, region)
            
            # Add metadata
            result_df['metric'] = metric
            results.append(result_df)
        else:
            print(f"Warning: Unknown metric '{metric}' for variable '{variable}'")
    
    # Combine results
    if len(results) == 1:
        final_df = results[0]
    else:
        # Merge multiple metrics
        final_df = results[0]
        for df in results[1:]:
            merge_cols = ['GEOID', 'pixel_count']
            final_df = pd.merge(final_df, df, on=merge_cols, how='outer')
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    metrics_str = '_'.join(metrics)
    output_file = os.path.join(output_dir, f"county_{variable}_{metrics_str}_{base_name}_{region}.csv")
    
    # Save results
    final_df.to_csv(output_file, index=False)
    
    # Close dataset
    ds.close()
    
    print(f"✅ Single file processing complete: {output_file}")
    return output_file


def process_timeseries(variable: str, scenario: str, regions: List[str], output_dir: str, 
                      counties_shapefile: str, climate_normals_base: str, 
                      metrics: List[str], year_range: Optional[tuple] = None,
                      max_workers: Optional[int] = None):
    """
    Process full time series (1980-2100) for production use.
    
    Args:
        variable (str): Climate variable (pr, tas, tasmin, tasmax)
        scenario (str): Climate scenario (historical, hybrid, ssp245, all)
        regions (List[str]): Geographic regions
        output_dir (str): Output directory
        counties_shapefile (str): Path to county shapefile
        climate_normals_base (str): Base path to climate normals data
        metrics (List[str]): List of metrics to calculate
        year_range (tuple, optional): (start_year, end_year) to limit processing
        max_workers (int, optional): Maximum parallel workers
    """
    print(f"=== TIME SERIES PROCESSING ===")
    print(f"Variable: {variable}")
    print(f"Scenario: {scenario}")
    print(f"Regions: {', '.join(regions)}")
    print(f"Metrics: {', '.join(metrics)}")
    if year_range:
        print(f"Year range: {year_range[0]}-{year_range[1]}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load county boundaries once
    print("Loading county boundaries...")
    counties = load_us_counties(counties_shapefile)
    
    # Process each region
    results = []
    for region in regions:
        print(f"\n=== Processing Region: {region} ===")
        
        # Validate region-variable combination
        if not validate_region_variable_combination(region, variable):
            print(f"Skipping {region} - no {variable} data available")
            continue
        
        # Get file list for this region and variable
        try:
            file_list = get_data_files(region, variable, climate_normals_base)
            
            # Filter by scenario if specified
            if scenario != "all":
                file_list = filter_files_by_scenario(file_list, scenario)
            
            # Filter by year range if specified
            if year_range:
                start_year, end_year = year_range
                file_list = filter_files_by_year_range(file_list, start_year, end_year)
            
            if not file_list:
                print(f"No files found for {region} {variable} {scenario}")
                continue
            
            # Process this region using parallel framework
            result = process_region_timeseries(
                region_key=region,
                variable=variable,
                metrics=metrics,
                counties_all=counties,
                file_list=file_list,
                output_dir=output_dir,
                max_workers=max_workers
            )
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing region {region}: {e}")
            results.append({
                'region': region,
                'variable': variable,
                'status': 'error',
                'error': str(e)
            })
    
    return results


def process_all_variables(scenarios: List[str], regions: List[str], output_dir: str,
                         counties_shapefile: str, climate_normals_base: str,
                         year_range: Optional[tuple] = None, max_workers: Optional[int] = None):
    """
    Process all climate variables for given scenarios and regions.
    
    Args:
        scenarios (List[str]): List of climate scenarios
        regions (List[str]): List of geographic regions
        output_dir (str): Output directory
        counties_shapefile (str): Path to county shapefile
        climate_normals_base (str): Base path to climate normals data
        year_range (tuple, optional): (start_year, end_year) to limit processing
        max_workers (int, optional): Maximum parallel workers
    """
    print(f"=== PROCESSING ALL VARIABLES ===")
    print(f"Scenarios: {scenarios}")
    print(f"Regions: {regions}")
    print()
    
    variables = ["pr", "tas", "tasmin", "tasmax"]
    all_results = {}
    
    for scenario in scenarios:
        for region in regions:
            for variable in variables:
                print(f"\n--- Processing {variable} for {region} {scenario} ---")
                
                # Get all available metrics for this variable
                metrics = get_available_metrics(variable)
                if not metrics:
                    print(f"No metrics available for variable {variable}")
                    continue
                
                try:
                    results = process_timeseries(
                        variable, scenario, [region], output_dir,
                        counties_shapefile, climate_normals_base, metrics,
                        year_range, max_workers
                    )
                    all_results[f"{variable}_{region}_{scenario}"] = results[0] if results else None
                    
                except Exception as e:
                    print(f"Error processing {variable} {region} {scenario}: {e}")
                    all_results[f"{variable}_{region}_{scenario}"] = {
                        'status': 'error',
                        'error': str(e)
                    }
    
    return all_results


def process_run_all(output_dir: str, counties_shapefile: str, climate_normals_base: str,
                   year_range: Optional[tuple] = None, max_workers: Optional[int] = None,
                   resume: bool = False, show_summary: bool = False):
    """
    Process complete climate dataset: all variables, regions, and scenarios.
    
    This is the production-ready "run everything" mode that processes:
    - All 4 variables: pr, tas, tasmin, tasmax  
    - All 5 regions: CONUS, AK, HI, PRVI, GU
    - All 3 scenarios: historical, hybrid, ssp245
    - Full time range: 1950-2100 (151 years) unless limited by year_range
    
    Args:
        output_dir (str): Output directory
        counties_shapefile (str): Path to county shapefile  
        climate_normals_base (str): Base path to climate normals data
        year_range (tuple, optional): (start_year, end_year) to limit processing
        max_workers (int, optional): Maximum parallel workers
        resume (bool): Skip existing output files if True
        show_summary (bool): Show comprehensive summary after completion
    """
    print("🚀 " + "="*70)
    print("🚀 COMPREHENSIVE CLIMATE METRICS PROCESSING - RUN ALL MODE")
    print("🚀 " + "="*70)
    
    # Define complete processing matrix from configuration
    variables = DATA_CONFIG['variables']
    regions = list(DATA_CONFIG['regions'].keys())
    scenarios = list(DATA_CONFIG['scenarios'].keys())
    
    print(f"📊 PROCESSING SCOPE:")
    print(f"   Variables: {len(variables)} ({', '.join(variables)})")
    print(f"   Regions: {len(regions)} ({', '.join(regions)})")
    print(f"   Scenarios: {len(scenarios)} ({', '.join(scenarios)})")
    if year_range:
        print(f"   Years: {year_range[0]}-{year_range[1]} ({year_range[1]-year_range[0]+1} years)")
    else:
        print(f"   Years: 1950-2100 (151 years)")
    print(f"   Total combinations: {len(variables) * len(regions) * len(scenarios)}")
    print(f"   Workers: {max_workers or 'auto'}")
    print(f"   Resume mode: {'✅ ON' if resume else '❌ OFF'}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tracking
    processing_results = {}
    total_combinations = len(variables) * len(regions) * len(scenarios)
    completed_count = 0
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    # Track start time
    import time
    start_time = time.time()
    
    print("🔄 Starting comprehensive processing...")
    print()
    
    # Process each combination systematically
    for var_idx, variable in enumerate(variables, 1):
        print(f"📋 VARIABLE {var_idx}/{len(variables)}: {variable.upper()}")
        print("-" * 50)
        
        for reg_idx, region in enumerate(regions, 1):
            # Check if region supports this variable
            if not validate_region_variable_combination(region, variable):
                print(f"   ⚠️  {region}: Skipping - no {variable} data available")
                for scenario in scenarios:
                    processing_results[f"{variable}_{region}_{scenario}"] = {
                        'status': 'skipped',
                        'reason': f'No {variable} data for {region}'
                    }
                    skipped_count += 1
                continue
            
            print(f"   🌍 REGION {reg_idx}/{len(regions)}: {region}")
            
            for scen_idx, scenario in enumerate(scenarios, 1):
                combination_key = f"{variable}_{region}_{scenario}"
                completed_count += 1
                
                print(f"      📅 Scenario {scen_idx}/{len(scenarios)}: {scenario} "
                      f"[{completed_count}/{total_combinations}]")
                
                # Check for existing output file if resume mode
                if resume:
                    # Generate expected output filename (match the pattern from process_timeseries)
                    expected_pattern = f"county_{variable}_*_timeseries_{region}_*.csv"
                    existing_files = glob.glob(os.path.join(output_dir, expected_pattern))
                    if existing_files:
                        print(f"         ⏩ SKIPPED - Output exists: {os.path.basename(existing_files[0])}")
                        processing_results[combination_key] = {
                            'status': 'skipped',
                            'reason': 'Output file exists (resume mode)',
                            'output_file': existing_files[0]
                        }
                        skipped_count += 1
                        continue
                
                # Get all available metrics for this variable
                metrics = get_available_metrics(variable)
                if not metrics:
                    print(f"         ❌ ERROR - No metrics available for {variable}")
                    processing_results[combination_key] = {
                        'status': 'error',
                        'error': f'No metrics available for variable {variable}'
                    }
                    error_count += 1
                    continue
                
                try:
                    # Process this combination
                    results = process_timeseries(
                        variable, scenario, [region], output_dir,
                        counties_shapefile, climate_normals_base, metrics,
                        year_range, max_workers
                    )
                    
                    if results and len(results) > 0 and results[0].get('status') == 'success':
                        print(f"         ✅ SUCCESS - {results[0].get('records', 0)} records")
                        processing_results[combination_key] = results[0]
                        success_count += 1
                    else:
                        error_msg = results[0].get('error', 'Unknown error') if results else 'No results returned'
                        print(f"         ❌ ERROR - {error_msg}")
                        processing_results[combination_key] = {
                            'status': 'error',
                            'error': error_msg
                        }
                        error_count += 1
                        
                except Exception as e:
                    print(f"         ❌ EXCEPTION - {str(e)}")
                    processing_results[combination_key] = {
                        'status': 'error', 
                        'error': str(e)
                    }
                    error_count += 1
        
        print() # Empty line between variables
    
    # Calculate processing time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Show final summary
    print("🎯 " + "="*70)
    print("🎯 COMPREHENSIVE PROCESSING COMPLETE")
    print("🎯 " + "="*70)
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"📊 Processing summary:")
    print(f"   ✅ Successful: {success_count}")
    print(f"   ⏩ Skipped: {skipped_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   📦 Total: {completed_count}")
    
    if show_summary:
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 70)
        for variable in variables:
            print(f"\n{variable.upper()}:")
            for region in regions:
                region_results = []
                for scenario in scenarios:
                    key = f"{variable}_{region}_{scenario}"
                    result = processing_results.get(key, {})
                    status = result.get('status', 'unknown')
                    if status == 'success':
                        records = result.get('records', 0)
                        region_results.append(f"{scenario}:✅({records:,})")
                    elif status == 'skipped':
                        reason = result.get('reason', '')
                        region_results.append(f"{scenario}:⏩")
                    else:
                        region_results.append(f"{scenario}:❌")
                
                print(f"  {region:6}: {' | '.join(region_results)}")
    
    # Final statistics
    success_rate = (success_count / completed_count * 100) if completed_count > 0 else 0
    print(f"\n🎯 SUCCESS RATE: {success_rate:.1f}%")
    
    if success_count > 0:
        print(f"✅ READY FOR ANALYSIS: {success_count} complete datasets generated!")
    
    return processing_results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="County-level climate metrics processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--variable", 
        choices=["pr", "tas", "tasmin", "tasmax", "all"],
        help="Climate variable to process (default: 'all' for run-all mode, required for other modes)"
    )
    
    # Processing mode
    parser.add_argument(
        "--mode",
        choices=["single", "timeseries", "run-all"],
        default="single",
        help="Processing mode: single (one file), timeseries (one variable/region), run-all (complete dataset)"
    )
    
    # Metrics selection
    parser.add_argument(
        "--metrics",
        nargs='+',
        help="Specific metrics to calculate (default: all available for variable)"
    )
    
    # Single file mode arguments
    parser.add_argument(
        "--file",
        help="Path to NetCDF file (required for single mode)"
    )
    
    # Time series mode arguments
    parser.add_argument(
        "--scenario",
        choices=["historical", "ssp245", "ssp585", "all"],
        default="historical",
        help="Climate scenario for time series mode (default: historical)"
    )
    
    parser.add_argument(
        "--year-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Year range for processing (e.g., --year-range 1980 2020)"
    )
    
    # Geographic options
    parser.add_argument(
        "--region",
        default="CONUS",
        help="Geographic region(s): comma-separated list, 'all' for all regions, or specific regions (CONUS,AK,HI,PRVI,GU)"
    )
    
    # Parallel processing
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers (default: auto)"
    )
    
    # Paths
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--counties-shapefile",
        help="Path to county shapefile (default: auto-detect)"
    )
    
    parser.add_argument(
        "--climate-data-base",
        help="Base path to climate normals data (default: auto-detect)"
    )
    
    # Verbose output
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    # Add run-all specific options
    parser.add_argument(
        "--run-all-resume",
        action="store_true", 
        help="Resume run-all mode, skipping existing output files"
    )
    
    parser.add_argument(
        "--run-all-summary",
        action="store_true",
        help="Show comprehensive summary after run-all completion"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    paths = setup_paths()
    
    # Use provided paths or defaults
    counties_shapefile = args.counties_shapefile or str(paths['counties_shapefile'])
    climate_data_base = args.climate_data_base or str(paths['climate_normals_base'])
    
    # Parse regions - handle 'all' option
    if args.region.lower() == "all":
        regions = ["CONUS", "AK", "HI", "PRVI", "GU"]
    else:
        regions = [r.strip() for r in args.region.split(",")]
    
    # Determine metrics to calculate
    if args.metrics:
        if args.metrics == ['all']:
            metrics = get_available_metrics(args.variable) if args.variable != 'all' else []
        else:
            metrics = args.metrics
    else:
        # Default: all available metrics for the variable
        metrics = get_available_metrics(args.variable) if args.variable != 'all' else []
    
    # Validate arguments
    if args.mode != "run-all" and not args.variable:
        parser.error("--variable is required for single and timeseries modes")
    
    if args.mode == "single" and not args.file:
        parser.error("--file is required for single mode")
    
    if args.mode == "timeseries" and args.file:
        parser.error("--file should not be specified for timeseries mode")
    
    if args.mode == "run-all":
        # Set default variable for run-all mode if not specified
        if not args.variable:
            args.variable = "all"
            
        # For run-all mode, override user selections with comprehensive defaults
        if args.variable != "all":
            print(f"ℹ️  Run-all mode: overriding --variable {args.variable} with 'all'")
        if args.scenario != "all":
            print(f"ℹ️  Run-all mode: overriding --scenario {args.scenario} with 'all'")
        if args.region.lower() != "all":
            print(f"ℹ️  Run-all mode: overriding --region {args.region} with 'all'")
        
        # Force all comprehensive settings for run-all mode
        args.variable = "all"
        args.scenario = "all" 
        regions = ["CONUS", "AK", "HI", "PRVI", "GU"]  # Override regions to all
        
        if args.file:
            parser.error("--file should not be specified for run-all mode")
    
    # Convert year range to tuple
    year_range = tuple(args.year_range) if args.year_range else None
    
    try:
        if args.mode == "single":
            # Single file processing
            if len(regions) > 1:
                parser.error("Single mode only supports one region")
            
            if args.variable == "all":
                parser.error("Single mode does not support --variable all")
            
            if not metrics:
                print(f"No metrics specified. Available metrics for {args.variable}: {get_available_metrics(args.variable)}")
                sys.exit(1)
            
            result = process_single_file(
                args.variable, args.file, regions[0], 
                args.output_dir, counties_shapefile, metrics
            )
            print(f"\n✅ Single file processing complete: {result}")
            
        elif args.mode == "timeseries":
            # Time series processing
            if args.variable == "all":
                # Process all variables
                scenarios = [args.scenario] if args.scenario != "all" else ["historical", "ssp245", "ssp585"]
                results = process_all_variables(
                    scenarios, regions, args.output_dir,
                    counties_shapefile, climate_data_base, year_range, args.workers
                )
                
                print(f"\n✅ All variables processing complete:")
                for key, result in results.items():
                    if result and result.get('status') == 'success':
                        print(f"  ✅ {key}: {result.get('output_file', 'Success')}")
                    else:
                        print(f"  ❌ {key}: {result.get('error', 'Failed') if result else 'No result'}")
                        
            else:
                # Process single variable
                if not metrics:
                    print(f"No metrics specified. Available metrics for {args.variable}: {get_available_metrics(args.variable)}")
                    sys.exit(1)
                
                results = process_timeseries(
                    args.variable, args.scenario, regions, args.output_dir,
                    counties_shapefile, climate_data_base, metrics, year_range, args.workers
                )
                
                print(f"\n✅ Time series processing complete:")
                for result in results:
                    if result.get('status') == 'success':
                        print(f"  ✅ {result['region']}: {result.get('output_file', 'Success')}")
                    else:
                        print(f"  ❌ {result['region']}: {result.get('error', 'Failed')}")
    
        elif args.mode == "run-all":
            # Run all mode
            results = process_run_all(
                args.output_dir, counties_shapefile, climate_data_base,
                year_range, args.workers, args.run_all_resume, args.run_all_summary
            )
            
            print(f"\n✅ Run all processing complete:")
            for key, result in results.items():
                if result and result.get('status') == 'success':
                    print(f"  ✅ {key}: {result.get('output_file', 'Success')}")
                else:
                    print(f"  ❌ {key}: {result.get('error', 'Failed') if result else 'No result'}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
