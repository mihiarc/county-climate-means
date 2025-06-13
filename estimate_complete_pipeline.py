#!/usr/bin/env python3
"""
Estimate processing time and data volume for complete pipeline processing.
"""

import yaml
from pathlib import Path
from datetime import timedelta

# Configuration
VARIABLES = ["pr", "tas", "tasmax", "tasmin"]
REGIONS = ["CONUS", "AK", "HI", "PRVI", "GU"]
SCENARIOS = ["historical", "ssp245", "ssp585"]

# Years per scenario/region
YEARS = {
    "historical": {
        "CONUS": range(1950, 2015),  # 65 years
        "AK": range(1950, 2015),     # 65 years
        "HI": range(1950, 2015),     # 65 years
        "PRVI": range(1950, 2015),   # 65 years
        "GU": range(1950, 2015),     # 65 years
    },
    "ssp245": {
        "CONUS": range(2015, 2101),  # 86 years
        "AK": range(2015, 2101),     # 86 years
        "HI": range(2015, 2101),     # 86 years
        "PRVI": range(2015, 2101),   # 86 years
        "GU": range(2015, 2101),     # 86 years
    },
    "ssp585": {
        "CONUS": range(2015, 2101),  # 86 years
        "AK": range(2015, 2101),     # 86 years
        "HI": range(2015, 2101),     # 86 years
        "PRVI": range(2015, 2101),   # 86 years
        "GU": range(2015, 2101),     # 86 years
    }
}

# Counties per region (approximate)
COUNTIES = {
    "CONUS": 3109,
    "AK": 30,
    "HI": 5,
    "PRVI": 78,
    "GU": 1
}

# File sizes (approximate, in MB)
FILE_SIZES = {
    "means_nc": 5,      # Climate means NetCDF file
    "metrics_csv": 0.5,  # County metrics CSV
    "metrics_parquet": 0.3,  # County metrics Parquet
    "metrics_nc": 2,     # County metrics NetCDF
}

# Processing times (seconds per file, based on observations)
PROCESSING_TIMES = {
    "means": 0.5,      # Per year of climate means
    "metrics": 1.3,    # Per metrics file
}

def calculate_phase1_stats():
    """Calculate Phase 1 (means) statistics."""
    print("\n" + "="*80)
    print("PHASE 1: CLIMATE MEANS PROCESSING")
    print("="*80)
    
    total_files = 0
    total_size_gb = 0
    
    for variable in VARIABLES:
        var_files = 0
        for region in REGIONS:
            for scenario in SCENARIOS:
                years = len(YEARS[scenario][region])
                var_files += years
        
        total_files += var_files
        print(f"\n{variable}:")
        print(f"  Files: {var_files}")
        print(f"  Size: {var_files * FILE_SIZES['means_nc'] / 1024:.1f} GB")
    
    total_size_gb = total_files * FILE_SIZES['means_nc'] / 1024
    processing_time_hours = (total_files * PROCESSING_TIMES['means']) / 3600
    
    print(f"\nPhase 1 Totals:")
    print(f"  Total files: {total_files:,}")
    print(f"  Total size: {total_size_gb:.1f} GB")
    print(f"  Est. processing time: {processing_time_hours:.1f} hours")
    print(f"  With 6 parallel workers: {processing_time_hours/6:.1f} hours")
    
    return total_files, total_size_gb

def calculate_phase2_stats():
    """Calculate Phase 2 (metrics) statistics."""
    print("\n" + "="*80)
    print("PHASE 2: COUNTY METRICS PROCESSING")
    print("="*80)
    
    # Same number of input files as Phase 1 output
    input_files = 0
    for variable in VARIABLES:
        for region in REGIONS:
            for scenario in SCENARIOS:
                years = len(YEARS[scenario][region])
                input_files += years
    
    # Output files (3 formats per input)
    output_files = input_files * 3  # CSV, Parquet, NetCDF
    
    # Calculate sizes
    csv_size = input_files * FILE_SIZES['metrics_csv'] / 1024
    parquet_size = input_files * FILE_SIZES['metrics_parquet'] / 1024
    nc_size = input_files * FILE_SIZES['metrics_nc'] / 1024
    total_size_gb = csv_size + parquet_size + nc_size
    
    # Processing time
    processing_time_hours = (input_files * PROCESSING_TIMES['metrics']) / 3600
    
    # County statistics
    total_county_calcs = 0
    for region, county_count in COUNTIES.items():
        region_files = 0
        for variable in VARIABLES:
            for scenario in SCENARIOS:
                years = len(YEARS[scenario][region])
                region_files += years
        county_calcs = region_files * county_count
        total_county_calcs += county_calcs
        print(f"\n{region}:")
        print(f"  Counties: {county_count:,}")
        print(f"  Input files: {region_files}")
        print(f"  County calculations: {county_calcs:,}")
    
    print(f"\nPhase 2 Totals:")
    print(f"  Input files: {input_files:,}")
    print(f"  Output files: {output_files:,}")
    print(f"  Total output size: {total_size_gb:.1f} GB")
    print(f"    - CSV: {csv_size:.1f} GB")
    print(f"    - Parquet: {parquet_size:.1f} GB")
    print(f"    - NetCDF: {nc_size:.1f} GB")
    print(f"  Total county calculations: {total_county_calcs:,}")
    print(f"  Est. processing time: {processing_time_hours:.1f} hours")
    print(f"  With 8 parallel workers: {processing_time_hours/8:.1f} hours")
    
    return output_files, total_size_gb

def print_summary():
    """Print overall summary."""
    print("\n" + "="*80)
    print("COMPLETE PIPELINE SUMMARY")
    print("="*80)
    
    phase1_files, phase1_size = calculate_phase1_stats()
    phase2_files, phase2_size = calculate_phase2_stats()
    
    print("\n" + "="*80)
    print("OVERALL TOTALS")
    print("="*80)
    
    total_files = phase1_files + phase2_files
    total_size = phase1_size + phase2_size
    
    print(f"\nData Volume:")
    print(f"  Total files generated: {total_files:,}")
    print(f"  Total storage required: {total_size:.1f} GB")
    
    print(f"\nProcessing Time Estimates:")
    print(f"  Sequential: ~{(phase1_files * 0.5 + phase1_files * 1.3) / 3600:.0f} hours")
    print(f"  With parallelization: ~{((phase1_files * 0.5 / 6) + (phase1_files * 1.3 / 8)) / 3600:.0f} hours")
    print(f"  Complete pipeline: 24-48 hours (including overhead)")
    
    print(f"\nResource Requirements:")
    print(f"  Recommended RAM: 60-80 GB")
    print(f"  Recommended CPU cores: 32-48")
    print(f"  Storage space: {total_size * 1.2:.0f} GB (with 20% buffer)")

if __name__ == "__main__":
    print("Climate Processing Pipeline Estimation")
    print("=====================================")
    print("\nThis estimates processing requirements for:")
    print(f"- Variables: {', '.join(VARIABLES)}")
    print(f"- Regions: {', '.join(REGIONS)}")
    print(f"- Scenarios: {', '.join(SCENARIOS)}")
    
    print_summary()
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\n1. Use the staged configuration for more control")
    print("2. Monitor disk space - need ~100GB free")
    print("3. Run overnight or over weekend")
    print("4. Enable checkpointing for restart capability")
    print("5. Consider processing one variable at a time if memory is limited")