# Climate Means Processing Tutorial

This tutorial will guide you through processing climate data to calculate 30-year climate normals using the County Climate means module. You'll learn how to extract regional data, calculate rolling averages, and produce climate normal datasets.

## Overview

Climate normals are 30-year averages of climate variables that serve as a baseline for understanding climate patterns and changes. The County Climate means module processes NEX-GDDP-CMIP6 climate model data to generate these normals for different regions across the United States.

## Prerequisites

Before starting this tutorial, ensure you have:

- Python 3.9 or higher installed
- County Climate package installed with dependencies
- Access to NEX-GDDP-CMIP6 data files
- At least 16GB of RAM (more for parallel processing)

## Step 1: Understanding the Data Structure

The climate data is organized by:

- **Variables**: `pr` (precipitation), `tas` (temperature), `tasmax` (max temp), `tasmin` (min temp)
- **Scenarios**: `historical`, `ssp245`, `ssp585`
- **Regions**: `CONUS`, `Alaska`, `Hawaii`, `PRVI`, `Guam`
- **Time periods**: 1950-2100 (varies by scenario)

## Step 2: Basic Climate Means Processing

Let's start with a simple example of processing temperature data for a single region:

```python
from county_climate.means.core import SingleVariableProcessor
from county_climate.means.core.regions import REGIONS

# Initialize processor for temperature data
processor = SingleVariableProcessor(
    variable="tas",
    scenario="historical",
    region="CONUS",
    data_directory="/path/to/nexgddp/data",
    output_directory="./output/climate_means"
)

# Process a single year
result = processor.process_single_year(1980)
print(f"Processed {result['files_processed']} files")
print(f"Output saved to: {result['output_file']}")
```

## Step 3: Processing Multiple Years

To calculate 30-year normals, you need to process multiple years:

```python
# Process a range of years
years_to_process = range(1980, 2011)  # 1980-2010

for year in years_to_process:
    result = processor.process_single_year(year)
    print(f"Year {year}: {result['status']}")
```

## Step 4: Using the Pipeline Interface

For production workflows, use the pipeline interface that handles all the complexity:

```python
from county_climate.means.workflow import process_climate_data_workflow

# Process climate data with full workflow
results = process_climate_data_workflow(
    data_directory="/path/to/nexgddp/data",
    output_directory="./output",
    variables=["tas", "pr"],
    regions=["CONUS"],
    scenarios=["historical"],
    start_year=1980,
    end_year=2010,
    num_processes=4  # Parallel processing
)

print(f"Processed {results['total_files']} files")
print(f"Duration: {results['duration_seconds']} seconds")
```

## Step 5: Configuration-Based Processing

The recommended approach for production is using YAML configuration files:

```yaml
# climate_means_config.yaml
pipeline:
  pipeline_id: "climate-means-tutorial"
  pipeline_name: "Tutorial Climate Means Processing"
  
  stages:
    - name: "climate_means"
      stage_type: "means_processing"
      config:
        variables: ["tas", "pr"]
        regions: ["CONUS"]
        scenarios: ["historical"]
        start_year: 1980
        end_year: 2010
        num_processes: 4
        enable_rich_progress: true
```

Run with the orchestrator:

```bash
python main_orchestrated.py run --config climate_means_config.yaml
```

## Step 6: Understanding the Output

The processing creates NetCDF files with the following structure:

```python
import xarray as xr

# Load a processed file
ds = xr.open_dataset("output/data/CONUS/tas/tas_CONUS_historical_1980_climatology.nc")

# Examine the structure
print(ds)

# Variables:
# - tas_mean: 30-year average temperature
# - tas_std: Standard deviation
# - tas_min/max: Extreme values
# - valid_days: Number of valid data points

# Access the data
temperature_normals = ds.tas_mean
print(f"Shape: {temperature_normals.shape}")
print(f"Time steps: {len(ds.time)}")
```

## Step 7: Regional Processing Considerations

Different regions require special handling:

### Alaska
- Handles dateline crossing (-180/180 longitude wrap)
- Extended bounds for complete coverage

```python
processor = SingleVariableProcessor(
    variable="tas",
    scenario="historical", 
    region="Alaska",
    data_directory="/path/to/data",
    output_directory="./output"
)
```

### Hawaii/PRVI/Guam
- Smaller domains for island territories
- May have fewer grid points

```python
# Process Hawaii with appropriate bounds
processor = SingleVariableProcessor(
    variable="pr",
    scenario="ssp245",
    region="Hawaii", 
    data_directory="/path/to/data",
    output_directory="./output"
)
```

## Step 8: Parallel Processing

For faster processing, use parallel variable processing:

```python
from county_climate.means.core import ParallelVariablesProcessor

# Process all 4 variables simultaneously
processor = ParallelVariablesProcessor(
    variables=["pr", "tas", "tasmax", "tasmin"],
    scenario="historical",
    region="CONUS",
    data_directory="/path/to/data",
    output_directory="./output",
    num_processes=4  # One process per variable
)

results = processor.process_year_range(1980, 2010)
```

## Step 9: Monitoring Progress

The system provides rich progress tracking:

```python
# Enable rich progress display
processor = SingleVariableProcessor(
    variable="tas",
    scenario="historical",
    region="CONUS",
    data_directory="/path/to/data",
    output_directory="./output",
    enable_rich_progress=True  # Beautiful progress display
)

# Progress shows:
# - Files processed/remaining
# - Processing speed
# - ETA
# - System resources (CPU, Memory)
```

## Step 10: Error Handling and Recovery

The processor includes robust error handling:

```python
try:
    result = processor.process_single_year(1980)
except FileNotFoundError as e:
    print(f"Missing data files: {e}")
except Exception as e:
    print(f"Processing error: {e}")

# Check processing status
if result['status'] == 'completed':
    print("Processing successful")
else:
    print(f"Issues encountered: {result.get('errors', [])}")
```

## Advanced Topics

### Custom Time Periods

Calculate normals for different periods:

```python
# 20-year normal instead of 30-year
processor.normal_years = 20

# Process specific months
processor.months_to_process = [6, 7, 8]  # Summer only
```

### Memory Management

For large datasets or limited memory:

```python
# Configure memory limits
processor = SingleVariableProcessor(
    variable="pr",
    scenario="historical",
    region="CONUS",
    data_directory="/path/to/data",
    output_directory="./output",
    max_memory_gb=8,  # Limit to 8GB
    chunk_size="50MB"  # Process in smaller chunks
)
```

### Custom Output Formats

Export to different formats:

```python
# Save as both NetCDF and CSV
processor.output_formats = ["netcdf", "csv"]

# Custom compression
processor.compression_level = 4
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `num_processes` or process smaller regions
2. **Missing files**: Verify data directory structure matches expected pattern
3. **Slow processing**: Enable parallel processing or use SSD storage

### Performance Tips

- Use SSD storage for input data
- Allocate more processes for parallel execution
- Process variables in parallel when possible
- Monitor system resources during processing

## Next Steps

Now that you understand climate means processing:

1. Try the [County Metrics Tutorial](county-metrics.md) to aggregate data by county
2. Learn about [Data Validation](validation.md) for quality control
3. Explore [Pipeline Orchestration](pipeline.md) for complex workflows

## Example: Complete Processing Script

Here's a complete example that processes all variables for CONUS:

```python
#!/usr/bin/env python3
"""
Complete climate means processing example
"""

from county_climate.means.workflow import process_climate_data_workflow
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    config = {
        "data_directory": "/path/to/nexgddp/data",
        "output_directory": "./climate_means_output",
        "variables": ["pr", "tas", "tasmax", "tasmin"],
        "regions": ["CONUS"],
        "scenarios": ["historical", "ssp245", "ssp585"],
        "start_year": 1980,
        "end_year": 2010,
        "num_processes": 16,
        "enable_rich_progress": True
    }
    
    logger.info("Starting climate means processing")
    start_time = time.time()
    
    # Run processing
    results = process_climate_data_workflow(**config)
    
    # Report results
    duration = time.time() - start_time
    logger.info(f"Processing completed in {duration:.2f} seconds")
    logger.info(f"Total files processed: {results['total_files']}")
    logger.info(f"Output directory: {results['output_directory']}")
    
    # Check for any failures
    if results.get('failed_files'):
        logger.warning(f"Failed files: {len(results['failed_files'])}")
        for file in results['failed_files'][:5]:  # Show first 5
            logger.warning(f"  - {file}")

if __name__ == "__main__":
    main()
```

## Summary

You've learned how to:

- ✅ Process climate data for 30-year normals
- ✅ Handle different regions and their special requirements  
- ✅ Use parallel processing for better performance
- ✅ Monitor progress and handle errors
- ✅ Configure processing through YAML files

The climate means module provides the foundation for all downstream climate analysis. The processed normals can be used for county-level statistics, validation, and climate change analysis.