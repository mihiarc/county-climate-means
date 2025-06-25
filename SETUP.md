# Fresh Pipeline Setup Guide

This guide helps you run the County Climate pipeline from scratch after removing all legacy code.

## Prerequisites

1. **System Requirements**:
   - CPU: 24+ cores (56 cores optimal)
   - RAM: 80GB+ (95GB optimal)
   - Storage: 2TB+ fast SSD
   - Python 3.8+

2. **Data Requirements**:
   - NEX-GDDP-CMIP6 climate data in NetCDF format
   - Data should be organized by variable/scenario

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd county_climate_means

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .[all]
```

## Configuration

1. **Set Environment Variables**:
```bash
export CLIMATE_DATA_BASE="/path/to/climate/data"
export CLIMATE_OUTPUT_BASE="/path/to/output/directory"
export CLIMATE_OUTPUT_VERSION="v1.0"
```

2. **Choose Configuration**:
   - Production: `configs/production/production_with_validation.yaml`
   - High Performance: `configs/production/production_ultra_high_performance.yaml`
   - Testing: `configs/development/test/pipeline_test_complete.yaml`

## Running the Pipeline

### Full Pipeline (All Stages)
```bash
python main_orchestrated.py run --config configs/production/production_with_validation.yaml
```

### Individual Stages
```bash
# Stage 1: Climate Means Only
python main_orchestrated.py run --config configs/specialized/single_phase/phase1_means_only.yaml

# Stage 2: County Metrics Only
python main_orchestrated.py run --config configs/specialized/single_phase/phase2_metrics_only.yaml

# Stage 3: Validation Only
python main_orchestrated.py run --config configs/validation/validation_conus_all_precipitation.yaml
```

### Specific Regions
```bash
# CONUS only
python main_orchestrated.py run --config configs/specialized/single_region/means_conus_ssp585_flexible.yaml

# Hawaii only
python main_orchestrated.py run --config configs/specialized/single_region/means_hawaii_ssp585_flexible.yaml
```

## Monitoring Progress

### Check Status
```bash
python main_orchestrated.py status
```

### View Logs
Logs are written to the output directory:
```
{CLIMATE_OUTPUT_BASE}/organized/logs/pipeline/
```

## Output Structure

All outputs follow the organized structure:
```
{CLIMATE_OUTPUT_BASE}/organized/v1.0/
├── L1_climate_means/      # 30-year climate normals
├── L2_county_metrics/     # County-level statistics
└── L3_validation/         # QA/QC reports and visualizations
```

## Troubleshooting

1. **Memory Issues**: Reduce `num_processes` in configuration
2. **Missing Data**: Check data paths in configuration
3. **Permission Errors**: Ensure write access to output directory

## Next Steps

After successful pipeline run:
1. Review validation reports in `L3_validation/reports/`
2. Check visualizations in `L3_validation/visualizations/`
3. Access county metrics in `L2_county_metrics/by_region/`