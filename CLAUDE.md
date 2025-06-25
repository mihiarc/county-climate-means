# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🌍 Project Overview

County Climate is a sophisticated Python package for processing climate data and calculating 30-year climate normals from NEX-GDDP-CMIP6 climate model data. It supports county-level climate analysis across multiple U.S. regions with both means (normals) and metrics (statistics) calculations.

This project implements a high-performance climate data processing pipeline for computing rolling 30-year climate normals from NorESM2-LM climate model data. The system processes multiple climate variables across different scenarios and geographic regions, with optimized multiprocessing capabilities and comprehensive monitoring.

## 📊 Architecture

The project follows a modular, pipeline-based architecture:

- **county_climate.means**: Climate normals (30-year averages) processing
- **county_climate.metrics**: County-level climate statistics and extremes  
- **county_climate.shared**: Shared infrastructure, contracts, and orchestration

Key design patterns:
- Staged pipeline processing with Pydantic data contracts
- Configuration-driven orchestration (YAML-based)
- Multiprocessing support optimized for high-memory systems (95GB/56 cores)
- Rich progress tracking with real-time system stats

### Core Processing Modules

| Module | Purpose | Key Features |
|--------|---------|---------------|
| `climate_means.py` | Core climate calculations | Sequential processing, climate normal computation, crash resistance |
| `climate_multiprocessing.py` | High-performance processing | 6-worker optimization, 4.3x speedup, memory management |
| `process_all_climate_normals.py` | Sequential pipeline | Comprehensive processing, all variables/periods |
| `process_climate_normals_multiprocessing.py` | MP pipeline | 24-core utilization, progress tracking, resume capability |

## Common Commands

### Running the Pipeline

```bash
# Run complete pipeline with configuration
python main_orchestrated.py run --config configs/production_high_performance.yaml

# Run pipeline with validation (Phase 3)
python main_orchestrated.py run --config configs/production_with_validation.yaml

# Validate configuration
python main_orchestrated.py validate --config configs/test_integration.yaml

# Check system status
python main_orchestrated.py status

# Create sample configurations
python main_orchestrated.py create-configs --output-dir configs/

# List available configurations
python main_orchestrated.py list-configs

# Convert county boundaries to modern format
python scripts/convert_county_boundaries.py
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=county_climate

# Run specific test file
pytest tests/test_end_to_end_pipeline.py

# Run tests matching pattern
pytest -k "test_pipeline"
```

### Code Quality

```bash
# Format code
black county_climate/

# Sort imports
isort county_climate/

# Type checking
mypy county_climate/

# Linting
flake8 county_climate/
```

### Development Installation

```bash
# Install with all dependencies
pip install -e .[all]

# Development installation
pip install -e .[dev]
```

## 🔄 Pipeline Stages & Data Flow

### Stage 1: Climate Means Processing
- Calculates 30-year climate normals
- Variables: pr (precipitation), tas (temperature), tasmax, tasmin
- Regions: CONUS, Alaska, Hawaii, PRVI, Guam
- Scenarios: historical (1980-2014), ssp245, ssp585
- Periods:
  - Historical: 1980-2014 (historical data only)
  - Hybrid: 2015-2044 (historical + SSP245)
  - Future: 2045-2100 (SSP245 only)

### Stage 2: Climate Metrics
- County-level statistics (mean, std, min, max, percentiles)
- Area-weighted aggregation
- Uses modern GeoParquet format for county boundaries

### Stage 3: Validation (QA/QC)
- Comprehensive quality assurance and control
- Validators:
  - **QAQCValidator**: Data completeness, temporal/spatial consistency, logical relationships
  - **SpatialOutliersValidator**: Geographic outlier detection using IQR, Z-score methods
  - **PrecipitationValidator**: Precipitation-specific data quality checks
- Visualization suite for analysis and reporting
- Generates quality scores and validation reports

### Data Flow
1. **Input**: NEX-GDDP-CMIP6 NetCDF files by variable/scenario
2. **Phase 1**: Regional extraction → 30-year normal calculation
3. **Phase 2**: County aggregation → Metrics calculation
4. **Phase 3**: QA/QC validation → Quality reports → Visualizations
5. **Output**: NetCDF files with climate normals/metrics, CSV/Parquet exports, validation reports

## Key Files and Entry Points

- `main_orchestrated.py`: Configuration-driven pipeline orchestrator
- `run_complete_pipeline.py`: Direct pipeline execution
- `configs/`: Pipeline configuration files
- `county_climate/shared/contracts/`: Pydantic data contracts
- `county_climate/shared/orchestration/`: Pipeline orchestration logic

## Configuration System

The project uses YAML configurations with:
- Stage-specific resource allocation
- Environment variable overrides
- Multiple profiles (production, development, testing)

Example configuration structure:
```yaml
pipeline:
  stages:
    - name: "climate_means"
      config:
        num_processes: 56
        enable_rich_progress: true
```

## 📂 Output Structure

```
output/rolling_30year_climate_normals/
├── pr/
│   ├── historical/
│   │   ├── pr_CONUS_historical_YYYY_30yr_normal.nc
│   │   └── pr_CONUS_historical_1980-2014_all_normals.nc
│   ├── hybrid/
│   └── ssp245/
├── tas/
├── tasmax/
└── tasmin/
```

## County Boundaries

The project now uses modern GeoParquet format for county boundaries:
- Centralized management via `CountyBoundariesManager`
- Automatic conversion from shapefile on first use
- Optimized versions for different phases
- Located in `county_climate/data/`

## Testing Approach

- **Framework**: pytest with asyncio support
- **Key test files**:
  - `test_end_to_end_pipeline.py`: Full pipeline integration
  - `test_config_integration.py`: Configuration system
  - `test_contracts.py`: Data contracts validation

## 🚀 Performance Metrics & Optimization

### Multiprocessing Optimization Results

| Configuration | Time | Speedup | Efficiency | Throughput |
|---------------|------|---------|------------|------------|
| Sequential (1 worker) | 15.4s | 1.0x | 100% | 0.32 files/s |
| **Optimal (6 workers)** | **3.6s** | **4.3x** | **72%** | **1.41 files/s** |
| Excessive (8+ workers) | 3.6s | 4.3x | <54% | ~1.40 files/s |

### System Requirements
- **CPU**: 24+ cores recommended (56 cores optimal)
- **RAM**: 80GB+ for optimal performance (95GB recommended)
- **Storage**: High-performance SSD for NetCDF I/O
- **Python**: 3.8+ with scientific computing stack

### Key Parameters
- `MAX_CORES`: 24 (total system utilization)
- `CORES_PER_VARIABLE`: 6 (optimal from testing)
- `BATCH_SIZE_YEARS`: 2 (memory management)
- `MAX_MEMORY_PER_PROCESS_GB`: 4 (conservative limit)
- `MIN_YEARS_FOR_NORMAL`: 25 (minimum for climate normal)

## Regional Processing

Special handling for:
- Alaska: Dateline coordinate wrapping
- Hawaii/PRVI/Guam: Island-specific bounds
- CONUS: Continental US processing

## 📊 Data Sources

- **Model**: NorESM2-LM (Norwegian Earth System Model) / NEX-GDDP-CMIP6
- **Variables**: 
  - `pr`: Precipitation (kg m⁻² s⁻¹)
  - `tas`: Near-surface air temperature (K)
  - `tasmax`: Daily maximum temperature (K)
  - `tasmin`: Daily minimum temperature (K)
- **Scenarios**: 
  - `historical`: 1850-2014
  - `ssp245`: 2015-2100 (Shared Socioeconomic Pathway 2-4.5)
  - `ssp585`: 2015-2100 (Shared Socioeconomic Pathway 5-8.5)
- **Regions**: CONUS, Alaska, Hawaii, Puerto Rico & VI, Guam & CNMI

## 🎯 Project Achievements

- ✅ **High Performance**: 4.3x speedup with multiprocessing optimization
- ✅ **Comprehensive Coverage**: All 4 variables × 3 periods × 5 regions
- ✅ **Robust Processing**: Crash resistance, memory management, resume capability
- ✅ **Real-time Monitoring**: Progress tracking, performance metrics, status updates
- ✅ **Scientific Quality**: 30-year climate normals following WMO standards
- ✅ **Modular Design**: Clean separation of concerns, reusable components
- ✅ **Production Ready**: Optimized for large-scale climate data processing

## Cursor Rules

When working in this codebase:
1. Always look for root causes of errors ("Let me find the root cause...")
2. Explain logic clearly ("Here is my logic:")
3. Use teacher mode when solving problems ("Teacher mode:")