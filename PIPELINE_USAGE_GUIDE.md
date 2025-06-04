# Climate Data Processing Pipeline - Usage Guide

## 🌍 Overview

This guide covers how to use the Climate Data Processing Pipeline, a high-performance system for computing rolling 30-year climate normals from NorESM2-LM climate model data.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+
- **Memory**: 16GB+ RAM (32GB+ recommended for parallel processing)
- **Storage**: High-performance SSD storage preferred
- **CPU**: Multi-core system (6+ cores optimal for parallel processing)

### Dependencies
Install using uv (recommended):
```bash
uv pip install -r requirements.txt
```

### Data Structure
Ensure your input data follows the NorESM2-LM structure:
```
/path/to/data/NorESM2-LM/
├── pr/
│   ├── historical/
│   ├── ssp245/
│   └── ssp585/
├── tas/
├── tasmax/
└── tasmin/
```

## 🚀 Quick Start

### 1. Check System Status
```bash
python main.py status
```

### 2. Run Sequential Processing (Recommended for First Run)
```bash
python main.py sequential --input-dir /path/to/data/NorESM2-LM --output-dir output/
```

### 3. Run Parallel Processing (High Performance)
```bash
python main.py parallel --workers 6 --input-dir /path/to/data/NorESM2-LM --output-dir output/
```

## 📖 Detailed Command Reference

### Global Options
```bash
--log-level {DEBUG,INFO,WARNING,ERROR}    # Set logging level
--config-type {default,production,development,testing}  # Configuration profile
```

## 🔄 Processing Commands

### Sequential Processing
Best for: First runs, memory-constrained systems, debugging

```bash
# Basic sequential processing
python main.py sequential

# With custom paths and variables
python main.py sequential \
  --input-dir /path/to/data/NorESM2-LM \
  --output-dir /path/to/output \
  --variables pr tas \
  --regions CONUS

# Development configuration (faster testing)
python main.py sequential --config-type development
```

**Features:**
- Memory-efficient single-threaded processing
- Crash-resistant with robust error handling
- Detailed progress logging
- Ideal for initial testing and validation

### Parallel Processing
Best for: Production runs, high-performance systems

```bash
# Optimal parallel processing (6 workers)
python main.py parallel --workers 6

# Custom configuration
python main.py parallel \
  --workers 8 \
  --input-dir /path/to/data/NorESM2-LM \
  --output-dir /path/to/output \
  --variables pr tas tasmax tasmin \
  --regions CONUS

# Production configuration (conservative settings)
python main.py parallel --config-type production --workers 4
```

**Features:**
- Multi-level parallelism (Variables → Years → Files)
- Optimal 6-worker configuration (4.3x speedup)
- Process-based parallelism (avoids NetCDF threading issues)
- Memory monitoring and management
- Real-time progress tracking

## 📊 Monitoring & Status

### Real-time Progress Monitoring
```bash
# Start real-time monitoring
python main.py monitor

# Show summary and exit
python main.py monitor --summary

# Show recent log entries
python main.py monitor --log-lines 20

# Custom refresh interval
python main.py monitor --refresh-interval 10
```

### Quick Status Check
```bash
# Basic status check
python main.py status

# Shows completion percentage, recent files, variable progress
```

## ⚙️ Configuration Profiles

### Development Configuration
```bash
python main.py sequential --config-type development
```
- **Variables**: Only `pr` (for faster testing)
- **Workers**: 2
- **Log Level**: DEBUG
- **Time Periods**: Reduced ranges for quick testing

### Production Configuration
```bash
python main.py parallel --config-type production
```
- **Workers**: 4 (conservative)
- **Memory**: 3GB per worker (conservative)
- **Timeout**: 10 minutes per file
- **Retries**: 3
- **Log Level**: INFO

### Testing Configuration
```bash
python main.py sequential --config-type testing
```
- **Variables**: Only `pr`
- **Years**: 2010-2012 (minimal range)
- **Workers**: 1
- **Log Level**: DEBUG

## 🛠️ Advanced Features

### Performance Benchmarking
```bash
# Benchmark parallel vs sequential performance
python main.py benchmark --data-dir /path/to/data/NorESM2-LM --num-files 10

# Example output:
# Sequential: 15.4s
# Parallel: 3.6s  
# Speedup: 4.3x
```

### Optimization Analysis
```bash
# Run comprehensive optimization analysis
python main.py optimize --data-dir /path/to/data/NorESM2-LM

# Tests chunking strategies, memory usage, worker optimization
```

### Configuration Inspection
```bash
# Show default configuration
python main.py config

# Show production configuration
python main.py config --type production

# Show all available settings
python main.py config --type development
```

## 📁 Output Structure

The pipeline generates the following output structure:

```
output/
├── pr/
│   ├── historical/     # 1980-2014 climate normals
│   │   ├── pr_CONUS_historical_1980_30yr_normal.nc
│   │   ├── pr_CONUS_historical_1981_30yr_normal.nc
│   │   └── ...
│   ├── hybrid/         # 2015-2044 climate normals (historical + ssp245)
│   │   ├── pr_CONUS_hybrid_2015_30yr_normal.nc
│   │   └── ...
│   └── ssp245/         # 2045+ climate normals (ssp245 only)
│       ├── pr_CONUS_ssp245_2045_30yr_normal.nc
│       └── ...
├── tas/
├── tasmax/
└── tasmin/
```

## 🎯 Processing Types Explained

### Historical Normals (1980-2014)
- **Data Source**: Historical scenario only
- **Method**: 30-year rolling windows using historical data
- **Example**: Target year 2000 uses data from 1971-2000

### Hybrid Normals (2015-2044)  
- **Data Source**: Combination of historical + SSP245 scenarios
- **Method**: Seamless transition between scenarios
- **Example**: Target year 2020 might use 1991-2014 (historical) + 2015-2020 (SSP245)

### Future Normals (2045+)
- **Data Source**: SSP245 scenario only
- **Method**: 30-year rolling windows using future projections
- **Example**: Target year 2070 uses SSP245 data from 2041-2070

## 🚨 Common Issues & Solutions

### Memory Issues
```bash
# Use production config with conservative memory settings
python main.py parallel --config-type production

# Reduce number of workers
python main.py parallel --workers 2

# Use sequential processing
python main.py sequential
```

### Performance Issues
```bash
# Check optimal worker count
python main.py benchmark --data-dir /path/to/data

# Use optimization analysis
python main.py optimize --data-dir /path/to/data
```

### Missing Files/Data
```bash
# Check data availability
python main.py status

# Validate data structure
ls -la /path/to/data/NorESM2-LM/pr/historical/
```

### Process Monitoring
```bash
# Monitor in real-time
python main.py monitor

# Check recent progress
python main.py monitor --log-lines 50
```

## 📈 Performance Expectations

### System Specifications Used for Optimization
- **Hardware**: 56 CPUs, 99.8GB RAM, SSD storage
- **Optimal Workers**: 6
- **Expected Speedup**: 4.3x over sequential
- **Memory Usage**: ~24GB (6 × 4GB per worker)

### Processing Time Estimates
| Dataset | Sequential | Parallel (6 workers) | Speedup |
|---------|------------|----------------------|---------|
| 3 years | 8.8s | 3.3s | 2.7x |
| 5 years | 15.4s | 3.6s | 4.3x |
| 65 years | ~3.5 min | ~0.8 min | 4.3x |
| Full processing | ~3 hours | ~0.7 hours | 4.3x |

## 🔍 Variables & Regions

### Available Variables
- **`pr`**: Precipitation
- **`tas`**: Near-surface air temperature
- **`tasmax`**: Daily maximum near-surface air temperature  
- **`tasmin`**: Daily minimum near-surface air temperature

### Available Regions
- **`CONUS`**: Continental United States (default)
- **`AK`**: Alaska
- **`HI`**: Hawaii
- **`PRVI`**: Puerto Rico and U.S. Virgin Islands
- **`GU`**: Guam and Northern Mariana Islands

## 🎛️ Environment Variables

Override default settings using environment variables:

```bash
# Data directories
export CLIMATE_INPUT_DIR="/custom/input/path"
export CLIMATE_OUTPUT_DIR="/custom/output/path"

# Performance settings
export CLIMATE_MAX_WORKERS=8
export CLIMATE_MAX_MEMORY=6.0

# Processing scope
export CLIMATE_VARIABLES="pr,tas"
export CLIMATE_REGIONS="CONUS"

# Logging
export CLIMATE_LOG_LEVEL="DEBUG"
```

## 📝 Example Workflows

### Development Workflow
```bash
# 1. Test with development config
python main.py sequential --config-type development

# 2. Monitor progress
python main.py monitor --summary

# 3. Check results
python main.py status
```

### Production Workflow
```bash
# 1. Benchmark system
python main.py benchmark --data-dir /path/to/data

# 2. Start parallel processing
python main.py parallel --config-type production

# 3. Monitor in real-time (separate terminal)
python main.py monitor

# 4. Check final status
python main.py status
```

### Testing Workflow
```bash
# 1. Quick test run
python main.py sequential --config-type testing

# 2. Verify output structure
ls -la output/pr/historical/

# 3. Check a result file
ncdump -h output/pr/historical/pr_CONUS_historical_2010_30yr_normal.nc
```

## 🆘 Support & Troubleshooting

### Log Files
- **Main log**: `climate_processing.log`
- **Progress log**: `processing_progress.log`
- **Status file**: `processing_progress.json`

### Debug Mode
```bash
python main.py sequential --log-level DEBUG --config-type development
```

### Performance Profiling
```bash
python main.py optimize --data-dir /path/to/data
```

### System Resource Monitoring
```bash
# Monitor memory and CPU usage
python main.py monitor

# Check system status
python main.py status
```

---

## 📚 Additional Resources

- **Project Overview**: See `PROJECT_OVERVIEW.md` for technical details
- **Refactoring Summary**: See `REFACTORING_SUMMARY.md` for changes from legacy code
- **Configuration**: Use `python main.py config` to inspect settings
- **Legacy Scripts**: Available in `legacy_scripts/` for reference

For additional help, run any command with `--help` flag:
```bash
python main.py --help
python main.py parallel --help
python main.py monitor --help
``` 