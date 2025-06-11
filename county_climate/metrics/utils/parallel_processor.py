"""
Parallel processing framework for climate data processing.

Provides multiprocessing capabilities for processing large numbers of climate
data files efficiently across multiple CPU cores.
"""

import os
import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Dict, Any, Callable, Optional
import pandas as pd
import xarray as xr
from .region_config import validate_region_variable_combination
from .netcdf_loader import load_and_prepare_netcdf


def get_calculation_functions():
    """
    Get the mapping of variables to their calculation functions.
    
    This is done dynamically to avoid circular imports.
    
    Returns:
        dict: Mapping of variables to calculation functions
    """
    # Import inside function to avoid circular imports
    from ..core.precipitation import calculate_annual_precipitation, calculate_high_precip_days_95th
    from ..core.temperature import (
        calculate_annual_mean_temperature, calculate_growing_degree_days,
        calculate_high_temperature_days_90th, calculate_low_temperature_days_10th,
        calculate_annual_min_temperature, calculate_annual_max_temperature,
        calculate_temp_days_95th_percentile, calculate_temp_days_5th_percentile,
        calculate_temp_days_1st_percentile
    )
    
    return {
        'pr': {
            'annual_precipitation': calculate_annual_precipitation,
            'high_precip_days_95th': calculate_high_precip_days_95th
        },
        'tas': {
            'annual_mean_temperature': calculate_annual_mean_temperature,
            'growing_degree_days': calculate_growing_degree_days
        },
        'tasmin': {
            'annual_min_temperature': calculate_annual_min_temperature,
            'low_temperature_days_10th': calculate_low_temperature_days_10th,
            'temp_days_5th_percentile': calculate_temp_days_5th_percentile,
            'temp_days_1st_percentile': calculate_temp_days_1st_percentile
        },
        'tasmax': {
            'annual_max_temperature': calculate_annual_max_temperature,
            'high_temperature_days_90th': calculate_high_temperature_days_90th,
            'temp_days_95th_percentile': calculate_temp_days_95th_percentile
        }
    }


# Legacy constant for backwards compatibility - will get functions dynamically
CALCULATION_FUNCTIONS = None


def process_single_file(file_info: Tuple[int, str, str], 
                       counties_region: pd.DataFrame, 
                       region_key: str, 
                       variable: str,
                       metrics: List[str]) -> Dict[str, Any]:
    """
    Process a single climate data file.
    
    Args:
        file_info (Tuple[int, str, str]): (year, scenario, filepath)
        counties_region (pd.DataFrame): County boundaries for this region
        region_key (str): Region identifier
        variable (str): Climate variable ('pr', 'tas', 'tasmin', 'tasmax')
        metrics (List[str]): List of metrics to calculate
        
    Returns:
        Dict[str, Any]: Processing result with status and data
    """
    year, scenario, filepath = file_info
    
    try:
        # Get calculation functions dynamically
        calculation_functions = get_calculation_functions()
        
        # Load and process data with region-specific CRS handling
        ds = load_and_prepare_netcdf(filepath, region=region_key)
        
        # Calculate all requested metrics
        metric_results = []
        counties_fixed_total = 0
        
        for metric in metrics:
            if variable in calculation_functions and metric in calculation_functions[variable]:
                calc_func = calculation_functions[variable][metric]
                result_df = calc_func(ds, counties_region, region_key)
                
                # Add metadata columns
                result_df['year'] = year
                result_df['scenario'] = scenario
                result_df['region'] = region_key
                result_df['variable'] = variable
                result_df['metric'] = metric
                
                metric_results.append(result_df)
                
                # Track counties fixed (if available in result)
                if 'counties_fixed' in result_df.columns:
                    counties_fixed_total += result_df['counties_fixed'].iloc[0]
            else:
                raise ValueError(f"Unknown metric '{metric}' for variable '{variable}'")
        
        # Combine all metric results
        if len(metric_results) == 1:
            final_df = metric_results[0]
        else:
            # Merge multiple metrics on GEOID
            final_df = metric_results[0]
            for df in metric_results[1:]:
                # Merge on common columns
                merge_cols = ['GEOID', 'year', 'scenario', 'region', 'variable', 'pixel_count']
                final_df = pd.merge(final_df, df, on=merge_cols, how='outer')
        
        # Close dataset
        ds.close()
        
        return {
            'status': 'success',
            'year': year,
            'scenario': scenario,
            'filepath': filepath,
            'data': final_df,
            'counties_fixed': counties_fixed_total
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'year': year,
            'scenario': scenario,
            'filepath': filepath,
            'error': str(e),
            'data': None,
            'counties_fixed': 0
        }


def process_files_parallel(file_list: List[Tuple[int, str, str]], 
                          counties_region: pd.DataFrame,
                          region_key: str,
                          variable: str,
                          metrics: List[str],
                          max_workers: Optional[int] = None,
                          progress_interval: int = 10) -> Tuple[List[pd.DataFrame], Dict[str, int]]:
    """
    Process multiple climate data files in parallel.
    
    Args:
        file_list (List[Tuple[int, str, str]]): List of (year, scenario, filepath) tuples
        counties_region (pd.DataFrame): County boundaries for this region
        region_key (str): Region identifier
        variable (str): Climate variable
        metrics (List[str]): List of metrics to calculate
        max_workers (int, optional): Maximum number of parallel workers
        progress_interval (int): Show progress every N files
        
    Returns:
        Tuple[List[pd.DataFrame], Dict[str, int]]: (successful_results, stats)
    """
    # Determine number of workers
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(file_list))
    
    print(f"  Processing {len(file_list)} files using {max_workers} parallel workers...")
    
    # Track results and statistics
    all_results = []
    stats = {
        'total_files': len(file_list),
        'successful_files': 0,
        'failed_files': 0,
        'counties_fixed': 0
    }
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_file,
        counties_region=counties_region,
        region_key=region_key,
        variable=variable,
        metrics=metrics
    )
    
    # Use multiprocessing pool
    with mp.Pool(processes=max_workers) as pool:
        # Process files and track progress
        for i, result in enumerate(pool.imap(process_func, file_list)):
            if result['status'] == 'success':
                all_results.append(result['data'])
                stats['successful_files'] += 1
                stats['counties_fixed'] += result['counties_fixed']
            else:
                print(f"    ERROR processing {result['year']} ({result['scenario']}): {result['error']}")
                stats['failed_files'] += 1
            
            # Show progress
            if (i + 1) % progress_interval == 0:
                print(f"    Progress: {i+1}/{len(file_list)} files completed "
                      f"({stats['successful_files']} success, {stats['failed_files']} failed)")
    
    # Final progress update
    print(f"    Final: {len(file_list)} files completed "
          f"({stats['successful_files']} success, {stats['failed_files']} failed)")
    
    return all_results, stats


def process_region_timeseries(region_key: str,
                             variable: str,
                             metrics: List[str],
                             counties_all: pd.DataFrame,
                             file_list: List[Tuple[int, str, str]],
                             output_dir: str,
                             max_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Process a complete time series for a region and variable.
    
    Args:
        region_key (str): Region identifier
        variable (str): Climate variable
        metrics (List[str]): List of metrics to calculate
        counties_all (pd.DataFrame): All county boundaries
        file_list (List[Tuple[int, str, str]]): Files to process
        output_dir (str): Output directory
        max_workers (int, optional): Maximum parallel workers
        
    Returns:
        Dict[str, Any]: Processing results summary
    """
    from .region_config import get_region_info, filter_counties_by_region
    
    region_config = get_region_info(region_key)
    print(f"\n=== PROCESSING REGION: {region_config['name']} ({region_key}) ===")
    print(f"Variable: {variable}")
    print(f"Metrics: {', '.join(metrics)}")
    
    # Validate region-variable combination
    if not validate_region_variable_combination(region_key, variable):
        print(f"  Skipping {region_key} - no {variable} data available")
        return {'region': region_key, 'variable': variable, 'status': 'no_data_available'}
    
    # Filter counties for this region
    counties_region = filter_counties_by_region(counties_all, region_key)
    
    if len(counties_region) == 0:
        print(f"  No counties found for region {region_key}")
        return {'region': region_key, 'variable': variable, 'status': 'no_counties'}
    
    if not file_list:
        print(f"  No files found for region {region_key} variable {variable}")
        return {'region': region_key, 'variable': variable, 'status': 'no_files'}
    
    print(f"  Found {len(file_list)} files covering years {file_list[0][0]} to {file_list[-1][0]}")
    
    # Count files by scenario
    scenarios = {}
    for year, scenario, filepath in file_list:
        scenarios[scenario] = scenarios.get(scenario, 0) + 1
    
    print(f"  Files by scenario:")
    for scenario, count in scenarios.items():
        print(f"    {scenario}: {count} files")
    
    # Process files in parallel
    all_results, stats = process_files_parallel(
        file_list, counties_region, region_key, variable, metrics, max_workers
    )
    
    if not all_results:
        print(f"  No files were successfully processed for region {region_key}!")
        return {'region': region_key, 'variable': variable, 'status': 'processing_failed'}
    
    # Combine all results
    print(f"  Combining results for region {region_key}...")
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Sort by GEOID and year
    final_df = final_df.sort_values(['GEOID', 'year'])
    
    # Generate output filename
    metrics_str = '_'.join(metrics)
    output_file = f"county_{variable}_{metrics_str}_timeseries_{region_key}_1980_2100.csv"
    output_path = os.path.join(output_dir, output_file)
    
    # Save results
    final_df.to_csv(output_path, index=False)
    
    # Calculate coverage statistics
    # Use the first metric column for coverage calculation
    metric_cols = [col for col in final_df.columns if col not in ['GEOID', 'year', 'scenario', 'region', 'variable', 'metric', 'pixel_count']]
    if metric_cols:
        coverage_pct = (final_df[metric_cols[0]].notna().sum() / len(final_df)) * 100
    else:
        coverage_pct = 0.0
    
    print(f"  === REGION {region_key} COMPLETE ===")
    print(f"  Output: {output_path}")
    print(f"  Records: {len(final_df):,}")
    print(f"  Counties: {final_df['GEOID'].nunique():,}")
    print(f"  Coverage: {coverage_pct:.1f}%")
    print(f"  Small counties fixed: {stats['counties_fixed']:,}")
    print(f"  Processing: {stats['successful_files']} successful, {stats['failed_files']} failed files")
    
    return {
        'region': region_key,
        'variable': variable,
        'status': 'success',
        'output_file': output_path,
        'records': len(final_df),
        'counties': final_df['GEOID'].nunique(),
        'coverage_pct': coverage_pct,
        'counties_fixed': stats['counties_fixed'],
        'successful_files': stats['successful_files'],
        'failed_files': stats['failed_files'],
        'dataframe': final_df
    }


def get_available_metrics(variable: str) -> List[str]:
    """
    Get available metrics for a given climate variable.
    
    Args:
        variable (str): Climate variable ('pr', 'tas', 'tasmin', 'tasmax')
        
    Returns:
        List[str]: Available metric names
    """
    calculation_functions = get_calculation_functions()
    return list(calculation_functions.get(variable, {}).keys())


def get_optimal_worker_count(file_count: int, max_workers: Optional[int] = None) -> int:
    """
    Determine optimal number of workers for parallel processing.
    
    Args:
        file_count (int): Number of files to process
        max_workers (int, optional): Maximum workers override
        
    Returns:
        int: Optimal worker count
    """
    if max_workers is not None:
        return min(max_workers, file_count, mp.cpu_count())
    
    # Use heuristic: don't use more workers than files or CPU cores
    # Also cap at reasonable number to avoid overhead
    return min(file_count, mp.cpu_count(), 16) 