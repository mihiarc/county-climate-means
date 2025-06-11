# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **climate means calculation component** of a larger climate data processing pipeline. It specifically calculates 30-year climate normals (rolling averages) from NEX-GDDP-CMIP6 NorESM2-LM climate model data. The system processes basic climate variables (precipitation and temperature) across multiple U.S. regions (CONUS, Alaska, Hawaii, Puerto Rico, Guam) with optimized multiprocessing capabilities.

**Note**: This package handles climate means only. Climate metrics and extremes are calculated in a separate downstream package that uses the outputs from this system.

## Key Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt
# OR for full development setup
pip install -e .[dev]
```

### Running the System
```bash
# Main CLI entry point
python main.py --help

# Process all regions and variables
python main.py process-all

# Process specific region
python main.py process-region CONUS --variables pr tas

# Benchmark performance
python main.py benchmark --num-files 10

# Monitor progress
python main.py monitor

# Show system status
python main.py status
```

### Testing
```bash
# Run tests (based on pyproject.toml config)
pytest
pytest tests/
pytest tests/test_specific_file.py

# Run with coverage
pytest --cov=means

# Format code
black means/
isort means/

# Type checking
mypy means/
```

## Architecture Overview

This codebase represents a sophisticated, production-ready system that was successfully refactored from standalone regional scripts into a unified Python package following modern software engineering practices.

### Package Structure
- `means/` - Main Python package following clean architecture principles
  - `core/` - Business logic and domain models
    - `regional_climate_processor.py` - Unified processor with factory pattern
    - `multiprocessing_engine.py` - Advanced parallel processing with resource management
    - `regions.py` - Region definitions and coordinate handling
    - `maximum_processor.py` - Specialized maximum value processing
  - `utils/` - Shared utilities and infrastructure
    - `io_util.py` - NetCDF file handling and NorESM2 data structures
    - `rich_progress.py` - Modern terminal UI with real-time system monitoring
    - `time_util.py` - Temporal data processing utilities
  - `validation/` - Data quality and validation modules
  - `visualization/` - Plotting and data visualization capabilities
  - `config.py` - Sophisticated configuration management with dataclasses

### Architectural Patterns Used
- **Factory Pattern**: `create_regional_processor()`, `create_multiprocessing_engine()`
- **Strategy Pattern**: Different processing approaches for historical/hybrid/future periods
- **Configuration Object Pattern**: Centralized, hierarchical configuration with YAML support
- **Protocol Pattern**: `TaskProtocol` for extensible multiprocessing tasks
- **Builder Pattern**: Rich progress tracker with customizable display components

### Key Processing Flow (Climate Means Pipeline)
1. **Configuration Management**: Dataclass-based config with environment variable support
2. **NorESM2FileHandler**: NetCDF file discovery with metadata extraction and validation
3. **RegionalClimateProcessor**: Unified processor supporting all regions/variables via factory pattern
4. **MultiprocessingEngine**: Protocol-based parallel processing with automatic resource optimization
5. **RichProgressTracker**: Real-time monitoring with hierarchical task organization and system stats

**Output**: 30-year climate normal NetCDF files that serve as input to downstream climate metrics and extremes processing.

### Data Processing Types (Internal to Climate Means Stage)
- **Historical**: 1980-2014 (historical data only)
- **Hybrid**: 2015-2044 (historical + SSP245 scenario)
- **Future**: 2045-2100 (SSP245 scenario only)

**Note**: These processing types are internal to the climate means calculation. The downstream pipeline receives data indexed by final scenario (historical, ssp245, ssp585) and year, regardless of how it was processed internally.

### Supported Variables (Climate Means Only)
- `pr` - Precipitation (kg m⁻² s⁻¹)
- `tas` - Near-surface air temperature (K)  
- `tasmax` - Daily maximum temperature (K)
- `tasmin` - Daily minimum temperature (K)

**Note**: This system calculates 30-year rolling averages for these variables. Climate extremes and derived metrics are calculated in a separate downstream package.

### Supported Regions
- `CONUS` - Continental United States
- `AK` - Alaska
- `HI` - Hawaii
- `PRVI` - Puerto Rico & Virgin Islands
- `GU` - Guam & Northern Mariana Islands

## Configuration

The system uses `climate_config.yaml` for configuration. Create with:
```bash
python main.py create-config
```

Key config sections:
- `paths`: Input/output directories
- `processing`: Multiprocessing settings, memory limits
- `regions`: Regional processing parameters

## Performance Optimization

### Multiprocessing Settings
- **Optimal workers**: 6 (tested optimal for this system)
- **Memory limit**: 4GB per process
- **Batch size**: 2 years at a time
- **Expected speedup**: 4.3x over sequential processing

### Memory Management
- Automatic garbage collection between files
- Process memory monitoring with psutil
- Configurable memory limits per worker

## Development Notes

### Entry Points & API Design
- `main.py` - Primary CLI interface (external to package)
- `means.process_region()` - High-level convenience function
- `means.create_regional_processor()` - Factory function for custom configurations
- `means.core.regional_climate_processor.RegionalClimateProcessor` - Main processing class
- Legacy standalone scripts have been successfully refactored into unified approach

### Code Quality & Patterns
- **Type hints** throughout codebase for better IDE support and maintainability
- **Dataclass-based configuration** with validation and serialization
- **Protocol-based multiprocessing** for extensibility
- **Comprehensive logging** with multiple levels and file output
- **Modern Python packaging** with pyproject.toml and optional dependencies

### Error Handling Philosophy
- Always look for root cause of errors (per .cursor/rules/rootcause.mdc)
- **Graceful degradation** when optional features fail (e.g., Rich UI falls back to basic logging)
- **Resource management** with proper cleanup and memory monitoring
- **Restart functionality** - skips already processed files to enable resuming
- **Timeout handling** in multiprocessing with configurable limits

### Progress Tracking Architecture
- **Rich-based terminal UI** with real-time system monitoring
- **Hierarchical task organization** with parent/child relationships
- **Custom progress columns** for throughput, memory usage, and ETA
- **JSON status files** for programmatic monitoring and external integration
- **Processing summaries** with comprehensive timing and throughput metrics

## Data Validation

The system includes validation modules in `means/validation/`:
- Region extent validation
- Climate data alignment checks
- Output quality verification

## Common Troubleshooting

### Memory Issues
- Reduce `cores_per_variable` or `max_workers` in config
- Increase `max_memory_per_process_gb` if system has more RAM  
- Check memory usage with `python main.py status`
- Monitor real-time memory usage through Rich progress display

### Performance Issues
- Run benchmark: `python main.py benchmark`
- Check optimal worker count for your system (default 6 workers tested optimal)
- Verify SSD storage for NetCDF I/O performance
- Use `batch_size_years` to control memory vs speed tradeoff

### Configuration Issues
- Validate config with `python main.py status`
- Create sample config: `python main.py create-config`
- Check environment variables override YAML settings
- Ensure output directories have write permissions

### Data Issues
- Validate input data availability: `python main.py status`
- Check region bounds with validation modules in `means/validation/`
- Verify NetCDF file structure with NorESM2FileHandler
- Look for coordinate system issues (0-360 vs -180-180 longitude)

### Multiprocessing Issues
- Check available CPU cores and memory
- Reduce worker count if seeing timeout errors
- Verify process cleanup with system monitoring
- Use `safe_mode` in configuration for debugging