# Running County Climate Pipeline From Scratch

This guide helps you set up and run the County Climate pipeline after removing all legacy code.

## Prerequisites

### 1. System Requirements
- **CPU**: 24+ cores (56 cores optimal) 
- **RAM**: 80GB+ (95GB optimal)
- **Storage**: 2TB+ fast SSD
- **OS**: Linux (tested on Ubuntu)
- **Python**: 3.8+

### 2. Climate Data Setup

The pipeline expects NEX-GDDP-CMIP6 climate data organized in this structure:

```
/path/to/climate/data/
├── pr/
│   ├── historical/
│   │   ├── pr_day_NorESM2-LM_historical_r1i1p1f1_gn_1980.nc
│   │   ├── pr_day_NorESM2-LM_historical_r1i1p1f1_gn_1981.nc
│   │   └── ... (files for each year)
│   ├── ssp245/
│   │   └── pr_day_NorESM2-LM_ssp245_r1i1p1f1_gn_2015.nc
│   └── ssp585/
├── tas/
│   ├── historical/
│   ├── ssp245/
│   └── ssp585/
├── tasmax/
│   └── ... (same structure)
└── tasmin/
    └── ... (same structure)
```

Each NetCDF file should:
- Contain daily climate data for one year
- Follow naming convention: `{variable}_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{YYYY}.nc`
- Have standard CF-compliant metadata

## Installation

### 1. Clone and Install

```bash
# Clone repository
git clone <repository-url>
cd county_climate_means

# Create virtual environment (recommended: use uv)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
uv pip install -e .[all]

# Or with standard pip:
# pip install -e .[all]
```

### 2. Install Additional Dependencies

Some dependencies may need to be installed separately:

```bash
uv pip install geopandas pyproj requests shapely rasterio
```

### 3. Set Environment Variables

Create a `.env` file or export these variables:

```bash
# Required paths
export CLIMATE_DATA_BASE="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
export CLIMATE_OUTPUT_BASE="/media/mihiarc/RPA1TB/CLIMATE_OUTPUT/organized"
export CLIMATE_OUTPUT_VERSION="v1.0"

# Optional settings
export COUNTY_CLIMATE_LOG_LEVEL="INFO"
export COUNTY_CLIMATE_MAX_WORKERS="56"
```

## Running the Pipeline

### 1. Validate Configuration

First, validate your configuration file:

```bash
python main_orchestrated.py validate --config configs/production/production_with_validation.yaml
```

### 2. Check System Status

Verify system resources and dependencies:

```bash
python main_orchestrated.py status
```

### 3. Run Full Pipeline

For production processing with all stages:

```bash
python main_orchestrated.py run --config configs/production/production_with_validation.yaml
```

### 4. Run Individual Stages

You can run stages separately:

```bash
# Stage 1: Climate Means Only
python main_orchestrated.py run --config configs/specialized/single_phase/phase1_means_only.yaml

# Stage 2: County Metrics Only (requires Stage 1 output)
python main_orchestrated.py run --config configs/specialized/single_phase/phase2_metrics_only.yaml

# Stage 3: Validation Only (requires Stage 2 output)
python main_orchestrated.py run --config configs/validation/validation_conus_all_precipitation.yaml
```

### 5. Run Specific Regions or Variables

For testing or specific processing:

```bash
# Process only CONUS region
python main_orchestrated.py run --config configs/specialized/single_region/means_conus_ssp585_flexible.yaml

# Process only Hawaii
python main_orchestrated.py run --config configs/specialized/single_region/means_hawaii_ssp585_flexible.yaml
```

## Configuration Options

Key configuration parameters in YAML files:

```yaml
# Data paths
base_data_path: "/path/to/climate/data"
base_output_path: "/path/to/output"

# Processing options
variables: ["pr", "tas", "tasmax", "tasmin"]
regions: ["CONUS", "AK", "HI", "PRVI", "GU"]
scenarios: ["historical", "ssp245", "ssp585"]

# Performance tuning
multiprocessing_workers: 32
batch_size_years: 10
max_memory_per_worker_gb: 1.8
```

## Output Structure

The pipeline creates organized outputs:

```
{CLIMATE_OUTPUT_BASE}/organized/v1.0/
├── L1_climate_means/          # 30-year climate normals
│   └── netcdf/
│       ├── historical/
│       │   ├── CONUS/
│       │   ├── AK/
│       │   └── ...
│       ├── ssp245/
│       └── ssp585/
├── L2_county_metrics/         # County-level statistics
│   └── by_region/
│       ├── CONUS/
│       │   ├── csv/
│       │   ├── parquet/
│       │   └── netcdf/
│       └── ...
└── L3_validation/             # QA/QC reports
    ├── reports/
    └── visualizations/
```

## Monitoring and Logs

### Real-time Progress
The pipeline shows progress bars and system stats during execution.

### Log Files
Logs are written to:
```
{CLIMATE_OUTPUT_BASE}/organized/logs/pipeline/
```

### Check Pipeline Status
```bash
python main_orchestrated.py status
```

## Troubleshooting

### Common Issues

1. **"Data directory not found"**
   - Verify `CLIMATE_DATA_BASE` environment variable
   - Check data directory structure matches expected format
   - Ensure data files exist for requested years

2. **"No module named 'geopandas'"**
   - Install missing dependencies: `uv pip install geopandas pyproj shapely`

3. **Memory errors**
   - Reduce `multiprocessing_workers` in configuration
   - Increase `batch_size_years` to process fewer years at once
   - Check available system memory

4. **Permission errors**
   - Ensure write permissions for output directory
   - Check disk space availability

### Debug Mode

Run with debug logging:
```bash
COUNTY_CLIMATE_LOG_LEVEL=DEBUG python main_orchestrated.py run --config <config.yaml>
```

## Testing the Pipeline

For initial testing with minimal data:

1. Create test data directory with a few sample files
2. Use test configuration: `configs/development/test_local.yaml`
3. Run: `python main_orchestrated.py run --config configs/development/test_local.yaml`

## Next Steps

After successful pipeline execution:

1. **Review outputs**:
   - Check L1 climate means NetCDF files
   - Verify L2 county metrics CSV/Parquet files
   - Review L3 validation reports

2. **Analyze results**:
   - Use validation visualizations
   - Check quality scores in reports
   - Verify spatial and temporal consistency

3. **Scale up processing**:
   - Increase worker counts for production
   - Process additional regions/scenarios
   - Enable all validation checks

## Support

For issues or questions:
- Check logs in output directory
- Review error messages and stack traces
- Ensure all dependencies are installed
- Verify data format and structure