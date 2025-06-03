# Climate Data Processing - Sequential Processing

A consolidated, crash-resistant climate data processing script that combines regional operations and climate calculations using **sequential processing** for optimal reliability with NetCDF files.

## Features

- **Crash-resistant processing**: Handles memory issues gracefully with sequential approach
- **Regional climate analysis**: Supports CONUS, Alaska, Hawaii, Puerto Rico, and Guam regions
- **Sequential computing**: Optimized for NetCDF files to avoid thread-safety issues
- **30-year climate normals**: Calculates climate normals for any target year
- **Multiple climate scenarios**: Supports historical and future projection scenarios
- **Memory optimization**: Conservative chunking and memory monitoring
- **Comprehensive validation**: Data quality checks and validation

## Why Sequential Processing?

This codebase originally used Dask for distributed computing but was **migrated to sequential processing** after finding that:

1. **NetCDF/HDF5 thread-safety issues**: NetCDF libraries have known thread-safety problems
2. **I/O bottlenecks**: Climate data processing is often I/O bound rather than CPU bound
3. **Memory efficiency**: Sequential processing provides better memory management
4. **Reliability**: Fewer moving parts = fewer failure modes
5. **Simpler debugging**: Easier to trace and debug processing issues

## Installation

### Using uv (recommended)

```bash
# Install dependencies
uv pip install -r requirements.txt

# Core packages needed
uv pip install numpy xarray netcdf4 psutil
```

### Using pip

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
python run_climate_means.py
```

This runs a basic demonstration that:
- Tests file access and data validation
- Shows available regions
- Validates region definitions
- Generates example climate periods

### Full Processing Workflow

```python
python run_climate_means.py noresm2
```

This processes your climate data using sequential processing:

```python
def example_usage():
    # Configuration for sequential processing
    config = {
        'processing_type': 'sequential',     # Sequential processing approach
        'memory_conservative': True,         # Conservative memory usage
        'batch_size': 15,                   # Batch size for processing
        'max_retries': 3                    # Maximum retry attempts
    }
    
    # Define what to process
    variables = ['tas', 'tasmax', 'tasmin', 'pr']  # Climate variables
    regions = ['CONUS', 'AK', 'HI']                # US regions
    scenarios = ['historical', 'ssp245', 'ssp585'] # Climate scenarios
    
    # Set your data paths
    data_directory = "/path/to/your/climate/netcdf/files"
    output_directory = "/path/to/output/climate/normals"
    
    # Run processing
    process_climate_data_workflow(
        data_directory=data_directory,
        output_directory=output_directory,
        variables=variables,
        regions=regions,
        scenarios=scenarios,
        config=config
    )
```

## Configuration Options

### Processing Configuration

- `processing_type`: Always 'sequential' for reliability
- `memory_conservative`: Enable conservative memory management (default: True)
- `batch_size`: Number of files to process in each batch (default: 10-20)
- `max_retries`: Maximum retry attempts for failed computations (default: 3)

### Memory Management

The system uses conservative memory management:
- **Sequential file processing**: One file at a time to minimize memory usage
- **Aggressive garbage collection**: Cleanup after each file/batch
- **Safe chunking**: Conservative chunk sizes that work reliably
- **Memory monitoring**: Track and warn of memory issues

## Supported Regions

The script includes predefined regions with optimized coordinate systems:

- **CONUS**: Continental United States
- **AK**: Alaska
- **HI**: Hawaii and Islands
- **PRVI**: Puerto Rico and U.S. Virgin Islands
- **GU**: Guam and Northern Mariana Islands

## Supported Variables

The script is designed to work with standard climate variables:

- **tas**: Near-surface air temperature
- **tasmax**: Daily maximum near-surface air temperature
- **tasmin**: Daily minimum near-surface air temperature
- **pr**: Precipitation

## File Organization

The script expects NetCDF files organized with years in filenames. The `NorESM2FileHandler` class handles the NorESM2-LM dataset structure:

```
data/NorESM2-LM/
├── pr/historical/
├── pr/ssp245/
├── tas/historical/
└── ...
```

## Output

The script generates NetCDF files with 30-year climate normals:

```
{variable}_{region}_{scenario}_{target_year}_climate_normal.nc
```

Example: `tas_CONUS_historical_2014_climate_normal.nc`

Each output file includes:
- Climate normal data
- Comprehensive metadata
- Processing statistics
- Quality control information

## Memory Management

The sequential approach provides excellent memory management:

1. **One file at a time**: Process files sequentially to minimize memory usage
2. **Conservative chunking**: Use proven chunk sizes for NetCDF files
3. **Garbage collection**: Aggressive cleanup of intermediate data
4. **Memory monitoring**: Track memory usage and warn of issues
5. **Batch processing**: Process data in manageable batches

## Error Handling

The script is designed to be crash-resistant:

- **Sequential processing**: Avoid threading issues that cause crashes
- **Retry logic**: Automatically retry failed computations
- **Graceful degradation**: Handle file access and reading errors
- **Comprehensive logging**: Detailed logging for troubleshooting

## Performance

While sequential processing is slower than ideal parallel processing, it provides:

- **Reliability**: No threading issues or mysterious crashes
- **Predictability**: Consistent memory usage and processing times
- **Debugging**: Easy to track down issues when they occur
- **Stability**: Proven approach for NetCDF climate data

For typical climate processing workloads, the reliability benefits outweigh the performance costs.

## Migration from Dask

This codebase previously used Dask but was migrated to sequential processing. Key changes:

- **Removed**: All Dask cluster management and distributed computing
- **Added**: Sequential file processing with memory management  
- **Improved**: Error handling and reliability
- **Simplified**: Configuration and debugging

The core climate processing logic remains the same - only the execution strategy changed.

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `batch_size` 
2. **File not found**: Check file paths and naming conventions
3. **Coordinate issues**: Verify longitude/latitude coordinate names
4. **Time decoding errors**: The script handles this automatically with fallbacks

### Debug Mode

Enable debug logging by modifying the logging configuration:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Performance Tuning

For optimal performance:

1. **Batch size**: Adjust batch_size based on available memory
2. **File organization**: Organize files for efficient access
3. **Disk I/O**: Use fast storage for input/output operations

## Example Output

```
2024-01-15 10:30:15 - __main__ - INFO - Starting Climate Data Processing - Sequential Processing
2024-01-15 10:30:15 - __main__ - INFO - Available regions:
2024-01-15 10:30:15 - __main__ - INFO -   CONUS: CONUS
2024-01-15 10:30:15 - __main__ - INFO -   AK: Alaska
2024-01-15 10:30:15 - __main__ - INFO -   HI: Hawaii and Islands
2024-01-15 10:30:15 - __main__ - INFO -   PRVI: Puerto Rico and U.S. Virgin Islands
2024-01-15 10:30:15 - __main__ - INFO -   GU: Guam and Northern Mariana Islands
2024-01-15 10:30:15 - __main__ - INFO - Generated 35 historical periods
2024-01-15 10:30:17 - __main__ - INFO - Climate data processing completed successfully
```

## License

This script maintains the same licensing terms as the original codebase.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Enable debug logging for detailed information
3. Review the comprehensive logging output 