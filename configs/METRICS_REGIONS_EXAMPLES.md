# Running Metrics for Non-CONUS Regions

This guide provides examples of how to configure and run metrics processing for non-CONUS regions (Alaska, Hawaii, PRVI, and Guam).

## Overview

The metrics processing stage (Phase 2) can be configured to process specific regions by setting the `regions` parameter in the stage configuration. The available regions are:

- **CONUS**: Continental United States
- **AK**: Alaska
- **HI**: Hawaii  
- **PRVI**: Puerto Rico and Virgin Islands
- **GU**: Guam and Northern Mariana Islands

## Command Line Examples

### 1. Run Metrics for Alaska Only

```bash
python main_orchestrated.py run --config configs/metrics_alaska_only.yaml
```

### 2. Run Metrics for Island Territories (Hawaii, PRVI, Guam)

```bash
python main_orchestrated.py run --config configs/metrics_hawaii_prvi_guam.yaml
```

### 3. Run Metrics for a Single Region

Replace `REGION_CODE` with AK, HI, PRVI, or GU in the template:

```bash
# First, create a custom config from the template
cp configs/metrics_single_region_template.yaml configs/metrics_hawaii_only.yaml

# Edit the file to replace REGION_CODE with HI
# Then run:
python main_orchestrated.py run --config configs/metrics_hawaii_only.yaml
```

### 4. Run Complete Pipeline for All Regions

```bash
python main_orchestrated.py run --config configs/pipeline_complete_all_regions.yaml
```

## Configuration Parameters

### Key Region-Specific Settings

```yaml
stage_config:
  # Specify which regions to process
  regions: ["AK", "HI", "PRVI", "GU"]  # Can be one or multiple
  
  # Variables to process (all regions support all variables)
  variables: ["pr", "tas", "tasmax", "tasmin"]
  
  # Scenarios to process
  scenarios: ["historical", "ssp245", "ssp585"]
  
  # Region-specific options
  # For Alaska:
  handle_dateline_crossing: true  # Alaska crosses the international dateline
  
  # For small islands (HI, PRVI, GU):
  min_grid_points: 1  # Include counties with few grid points
  use_nearest_neighbor_for_small_counties: true
```

### Input Data Structure

The metrics processor expects Phase 1 outputs to be organized in one of these patterns:

```
/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/means/
├── tas/
│   ├── AK/
│   │   ├── historical/
│   │   │   └── tas_AK_historical_30yr_normal.nc
│   │   ├── ssp245/
│   │   │   └── tas_AK_ssp245_30yr_normal.nc
│   │   └── ssp585/
│   │       └── tas_AK_ssp585_30yr_normal.nc
│   ├── HI/
│   │   └── ...
│   └── ...
└── pr/
    └── ...
```

Or:

```
/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/means/
├── AK/
│   ├── tas/
│   │   └── ...
│   └── pr/
│       └── ...
├── HI/
│   └── ...
└── ...
```

### Output Structure

Metrics outputs will be organized by region:

```
/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/metrics/
├── AK/
│   ├── tas_AK_historical_30yr_normal_AK_county_metrics.csv
│   ├── tas_AK_ssp245_30yr_normal_AK_county_metrics.csv
│   └── ...
├── HI/
│   └── ...
└── ...
```

## Region-Specific Considerations

### Alaska (AK)
- **Counties**: 30 boroughs/census areas
- **Special handling**: Dateline crossing (some areas use negative longitudes < -180)
- **Grid coverage**: Large counties may have sparse grid coverage

### Hawaii (HI)
- **Counties**: 5 counties (Hawaii, Honolulu, Kalawao, Kauai, Maui)
- **Special handling**: Small island counties may have limited grid points
- **Recommendation**: Set `min_grid_points: 1` to include all counties

### Puerto Rico and Virgin Islands (PRVI)
- **Counties**: 78 municipios in PR + 3 districts in VI
- **Special handling**: Small territories with limited grid coverage
- **Recommendation**: Use lower grid point thresholds

### Guam and Northern Mariana Islands (GU)
- **Counties**: 1 territory (Guam) + 4 municipalities (Northern Mariana Islands)
- **Special handling**: Very small areas, may have minimal grid coverage
- **Recommendation**: Set `min_grid_points: 1` and consider nearest neighbor interpolation

## Complete Example: Hawaii Processing

```yaml
pipeline_id: "metrics_hawaii_complete"
pipeline_name: "Hawaii Climate Metrics Processing"
pipeline_version: "1.0.0"
environment: "production"

global_resource_limits:
  max_memory_gb: 20.0
  max_cpu_cores: 8

stages:
  - stage_id: "climate_metrics_hawaii"
    stage_type: "metrics"
    stage_name: "Hawaii County Climate Metrics"
    package_name: "county_climate.metrics.integration"
    entry_point: "metrics_stage_handler"
    trigger_type: "manual"
    
    resource_limits:
      max_memory_gb: 16.0
      max_cpu_cores: 6
    
    stage_config:
      # Input from Phase 1
      input_means_path: "/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/means"
      
      # Process Hawaii only
      regions: ["HI"]
      variables: ["pr", "tas", "tasmax", "tasmin"]
      scenarios: ["historical", "ssp245", "ssp585"]
      
      # Metrics configuration
      metrics: ["mean", "std", "min", "max", "percentiles"]
      percentiles: [10, 25, 50, 75, 90]
      
      # Hawaii-specific settings
      min_grid_points: 1  # Include small counties
      spatial_aggregation: "area_weighted_mean"
      
      # Output configuration
      output_base_path: "/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/metrics/hawaii"
      output_formats: ["csv", "netcdf4", "parquet"]
      
      # Processing parameters
      multiprocessing_workers: 4
      enable_rich_progress: true
```

## Troubleshooting

### Common Issues

1. **No files found for region**
   - Check that Phase 1 outputs exist for the specified region
   - Verify the file naming convention matches the expected pattern
   - Check the `input_means_path` configuration

2. **Counties have no data**
   - For small regions, reduce `min_grid_points` to 1
   - Check that the climate data covers the region's geographic bounds
   - Verify coordinate system alignment (should be WGS84/EPSG:4326)

3. **Memory issues with large regions**
   - Reduce `multiprocessing_workers`
   - Process regions separately rather than all at once
   - Increase `max_memory_gb` if system resources allow

### Validation

After processing, verify results:

```bash
# Check output files were created
ls -la /media/mihiarc/RPA1TB/CLIMATE_OUTPUT/metrics/AK/

# Quick validation of CSV output
head /media/mihiarc/RPA1TB/CLIMATE_OUTPUT/metrics/AK/*_county_metrics.csv

# Count processed counties
wc -l /media/mihiarc/RPA1TB/CLIMATE_OUTPUT/metrics/AK/*_county_metrics.csv
```

## Performance Tips

1. **Process regions separately** for better resource management
2. **Use fewer workers** for small regions (4-6 workers for islands)
3. **Enable rich progress** to monitor processing: `enable_rich_progress: true`
4. **Save multiple formats** if needed downstream: `output_formats: ["csv", "parquet"]`

## Next Steps

After running metrics:
1. Run validation (Phase 3) to check data quality
2. Generate visualizations for the processed regions
3. Export county summaries for analysis