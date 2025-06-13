# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

County Climate is a sophisticated Python package for processing climate data and calculating 30-year climate normals from NEX-GDDP-CMIP6 climate model data. It supports county-level climate analysis across multiple U.S. regions with both means (normals) and metrics (statistics) calculations.

## Architecture

The project follows a modular, pipeline-based architecture:

- **county_climate.means**: Climate normals (30-year averages) processing
- **county_climate.metrics**: County-level climate statistics and extremes  
- **county_climate.shared**: Shared infrastructure, contracts, and orchestration

Key design patterns:
- Staged pipeline processing with Pydantic data contracts
- Configuration-driven orchestration (YAML-based)
- Multiprocessing support optimized for high-memory systems (95GB/56 cores)
- Rich progress tracking with real-time system stats

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

## Pipeline Stages

1. **Stage 1: Climate Means Processing**
   - Calculates 30-year climate normals
   - Variables: pr, tas, tasmax, tasmin
   - Regions: CONUS, Alaska, Hawaii, PRVI, Guam
   - Scenarios: historical, ssp245, ssp585

2. **Stage 2: Climate Metrics**
   - County-level statistics (mean, std, min, max, percentiles)
   - Area-weighted aggregation
   - Uses modern GeoParquet format for county boundaries

3. **Stage 3: Validation (QA/QC)**
   - Comprehensive quality assurance and control
   - Validators:
     - **QAQCValidator**: Data completeness, temporal/spatial consistency, logical relationships
     - **SpatialOutliersValidator**: Geographic outlier detection using IQR, Z-score methods
     - **PrecipitationValidator**: Precipitation-specific data quality checks
   - Visualization suite for analysis and reporting
   - Generates quality scores and validation reports

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

## Data Flow

1. **Input**: NEX-GDDP-CMIP6 NetCDF files by variable/scenario
2. **Phase 1**: Regional extraction → 30-year normal calculation
3. **Phase 2**: County aggregation → Metrics calculation
4. **Phase 3**: QA/QC validation → Quality reports → Visualizations
5. **Output**: NetCDF files with climate normals/metrics, CSV/Parquet exports, validation reports

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

## Performance Considerations

- Multiprocessing optimized for 56 cores/95GB RAM systems
- Memory-aware processing with configurable limits
- Rich progress tracking enabled by default (disable with `enable_rich_progress: false`)

## Regional Processing

Special handling for:
- Alaska: Dateline coordinate wrapping
- Hawaii/PRVI/Guam: Island-specific bounds
- CONUS: Continental US processing

## Cursor Rules

When working in this codebase:
1. Always look for root causes of errors ("Let me find the root cause...")
2. Explain logic clearly ("Here is my logic:")
3. Use teacher mode when solving problems ("Teacher mode:")