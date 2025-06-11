# Core Module API Reference

This document provides detailed API reference for the core modules in the `means.core` package.

## Overview

The core package contains the fundamental business logic and domain models for climate data processing:

- **`regions.py`**: Regional definitions and geographic operations
- **`regional_climate_processor.py`**: Unified regional processing engine  
- **`multiprocessing_engine.py`**: Parallel processing framework

---

## regions.py

### Constants

#### `REGION_BOUNDS`

Global dictionary defining regional boundary definitions for all supported U.S. regions.

```python
REGION_BOUNDS: Dict[str, Dict[str, Union[str, float, bool]]]
```

**Structure:**
```python
{
    'REGION_KEY': {
        'name': str,           # Human-readable region name
        'lon_min': float,      # Minimum longitude (0-360° system)
        'lon_max': float,      # Maximum longitude (0-360° system)
        'lat_min': float,      # Minimum latitude
        'lat_max': float,      # Maximum latitude
        'convert_longitudes': bool  # Whether to convert coordinates
    }
}
```

**Supported Regions:**
- `CONUS`: Continental United States (234°E-294°E, 24°N-50°N)
- `AK`: Alaska (170°E-235°E, 50°N-72°N)
- `HI`: Hawaii and Islands (181.63°E-205.20°E, 18.92°N-28.45°N)
- `PRVI`: Puerto Rico and U.S. Virgin Islands (292.03°E-295.49°E, 17.62°N-18.57°N)
- `GU`: Guam and Northern Mariana Islands (144.58°E-146.12°E, 13.18°N-20.61°N)

### Functions

#### `get_region_crs_info(region_key: str) -> Dict[str, Any]`

Get coordinate reference system information for a specific region.

**Parameters:**
- `region_key` (str): Region identifier (must be in REGION_BOUNDS)

**Returns:**
- `Dict[str, Any]`: CRS information including:
  - `crs_type`: 'epsg' or 'proj4'
  - `crs_value`: EPSG code or PROJ4 string
  - `central_longitude`: Central meridian
  - `central_latitude`: Central parallel
  - `extent`: [west, east, south, north] bounds

**Example:**
```python
from means.core.regions import get_region_crs_info

crs_info = get_region_crs_info('CONUS')
print(crs_info['crs_value'])  # 5070 (NAD83 / Conus Albers)
```

#### `convert_longitude_bounds(lon_min: float, lon_max: float, is_0_360: bool) -> Dict[str, float]`

Convert longitude bounds between 0-360° and -180°/180° coordinate systems.

**Parameters:**
- `lon_min` (float): Minimum longitude value
- `lon_max` (float): Maximum longitude value  
- `is_0_360` (bool): True if input is 0-360° system, False for -180°/180°

**Returns:**
- `Dict[str, float]`: Converted bounds with keys 'lon_min', 'lon_max'

**Example:**
```python
from means.core.regions import convert_longitude_bounds

# Convert CONUS bounds from 0-360 to -180/180 system
bounds = convert_longitude_bounds(234, 294, is_0_360=False)
print(bounds)  # {'lon_min': -126, 'lon_max': -66}
```

#### `extract_region(ds: xr.Dataset, region_bounds: Dict) -> xr.Dataset`

Extract a specific region from the dataset with coordinate system awareness.

**Parameters:**
- `ds` (xr.Dataset): Input climate dataset
- `region_bounds` (Dict): Region boundary definition from REGION_BOUNDS

**Returns:**
- `xr.Dataset`: Dataset subset to the specified region

**Features:**
- Automatic coordinate system detection (0-360° vs -180°/180°)
- Handles dateline crossing for Pacific regions
- Preserves all dataset attributes and metadata
- Validates coordinate names ('lon'/'lat' vs 'x'/'y')

**Example:**
```python
import xarray as xr
from means.core.regions import extract_region, REGION_BOUNDS

# Load climate dataset
ds = xr.open_dataset('climate_data.nc')

# Extract CONUS region
conus_bounds = REGION_BOUNDS['CONUS']
conus_data = extract_region(ds, conus_bounds)
```

#### `validate_region_bounds(region_key: str) -> bool`

Validate that a region key exists and has proper bounds definition.

**Parameters:**
- `region_key` (str): Region identifier to validate

**Returns:**
- `bool`: True if region is valid, False otherwise

**Validation Checks:**
- Region key exists in REGION_BOUNDS
- All required fields are present
- Longitude/latitude bounds are reasonable
- Special handling for dateline-crossing regions

**Example:**
```python
from means.core.regions import validate_region_bounds

if validate_region_bounds('CONUS'):
    print("CONUS region is valid")
else:
    print("Invalid region definition")
```

---

## regional_climate_processor.py

### Classes

#### `RegionalProcessingConfig`

Configuration dataclass for regional climate processing operations.

```python
@dataclass
class RegionalProcessingConfig:
    region_key: str
    variables: List[str]
    input_data_dir: Path
    output_base_dir: Path
    
    # Processing settings
    max_cores: int = 6
    cores_per_variable: int = 2
    batch_size_years: int = 2
    max_memory_per_process_gb: int = 4
    memory_check_interval: int = 10
    min_years_for_normal: int = 25
    
    # Progress tracking
    status_update_interval: int = 30
```

**Attributes:**

**Required:**
- `region_key` (str): Region identifier (must be valid in REGION_BOUNDS)
- `variables` (List[str]): Climate variables to process (e.g., ['pr', 'tas'])
- `input_data_dir` (Path): Directory containing input climate data
- `output_base_dir` (Path): Base directory for output files

**Processing Settings:**
- `max_cores` (int): Maximum CPU cores to use (default: 6)
- `cores_per_variable` (int): Cores per variable when processing (default: 2)
- `batch_size_years` (int): Years to process in each batch (default: 2)
- `max_memory_per_process_gb` (int): Memory limit per process (default: 4)
- `memory_check_interval` (int): Seconds between memory checks (default: 10)
- `min_years_for_normal` (int): Minimum years required for valid normal (default: 25)

**Progress Tracking:**
- `status_update_interval` (int): Seconds between status updates (default: 30)

**Auto-generated Properties:**
- `progress_status_file`: JSON file for progress tracking
- `progress_log_file`: Log file for progress messages
- `main_log_file`: Main processing log file

**Example:**
```python
from pathlib import Path
from means.core.regional_climate_processor import RegionalProcessingConfig

config = RegionalProcessingConfig(
    region_key='CONUS',
    variables=['pr', 'tas', 'tasmax', 'tasmin'],
    input_data_dir=Path('/data/climate'),
    output_base_dir=Path('/output/normals'),
    max_cores=8,
    batch_size_years=3
)
```

#### `RegionalClimateProcessor`

Unified processor for regional climate normals with multiprocessing support.

```python
class RegionalClimateProcessor:
    def __init__(self, config: RegionalProcessingConfig, use_rich_progress: bool = True)
```

**Parameters:**
- `config` (RegionalProcessingConfig): Processing configuration
- `use_rich_progress` (bool): Enable rich progress tracking (default: True)

**Key Methods:**

##### `process_single_file_for_climatology_safe(self, file_path: str, variable_name: str) -> Tuple[Optional[int], Optional[xr.DataArray]]`

Process a single file to extract daily climatology (multiprocessing-safe).

**Parameters:**
- `file_path` (str): Path to climate data file
- `variable_name` (str): Climate variable to extract

**Returns:**
- `Tuple[Optional[int], Optional[xr.DataArray]]`: Year and daily climatology data

**Features:**
- Conservative memory usage for multiprocessing
- Automatic time coordinate handling
- Regional extraction with coordinate system detection
- Error handling with detailed logging

##### `compute_climate_normal_safe(self, data_arrays: List, years: List[int], target_year: int) -> Optional[xr.DataArray]`

Compute climate normal from multiple data arrays.

**Parameters:**
- `data_arrays` (List): List of daily climatology data arrays
- `years` (List[int]): Years corresponding to each data array
- `target_year` (int): Target year for the climate normal

**Returns:**
- `Optional[xr.DataArray]`: Climate normal with comprehensive metadata

**Metadata Added:**
- `long_name`: Descriptive name
- `target_year`: Target year for normal
- `source_years`: Range of years used
- `number_of_years`: Count of years in calculation
- `processing_method`: Processing methodology identifier
- `region`: Region identifier

##### `process_variable_multiprocessing(self, variable: str) -> Dict`

Process a single climate variable using multiprocessing.

**Parameters:**
- `variable` (str): Climate variable to process

**Returns:**
- `Dict`: Processing results with status, timing, and file information

**Features:**
- Dynamic worker allocation based on system resources
- Memory monitoring and management
- Progress tracking with rich visual feedback
- Comprehensive error handling and retry logic

##### `process_all_variables(self) -> Dict`

Process all configured variables for the region.

**Returns:**
- `Dict`: Complete processing results for all variables

**Workflow:**
1. Validate configuration and paths
2. Set up output directories
3. Process each variable with optimal resource allocation
4. Generate comprehensive reports
5. Clean up resources

**Example:**
```python
from pathlib import Path
from means.core.regional_climate_processor import (
    RegionalProcessingConfig, RegionalClimateProcessor
)

# Create configuration
config = RegionalProcessingConfig(
    region_key='CONUS',
    variables=['pr', 'tas'],
    input_data_dir=Path('/data/climate'),
    output_base_dir=Path('/output/normals')
)

# Initialize processor
processor = RegionalClimateProcessor(config, use_rich_progress=True)

# Process all variables
results = processor.process_all_variables()
print(f"Processing completed: {results['summary']}")
```

### Factory Functions

#### `create_regional_processor(region_key: str, variables: List[str] = None, use_rich_progress: bool = True, **kwargs) -> RegionalClimateProcessor`

Factory function to create a regional processor with sensible defaults.

**Parameters:**
- `region_key` (str): Region identifier
- `variables` (List[str], optional): Variables to process (default: ['pr', 'tas', 'tasmax', 'tasmin'])
- `use_rich_progress` (bool): Enable rich progress tracking (default: True)
- `**kwargs`: Additional configuration parameters

**Returns:**
- `RegionalClimateProcessor`: Configured processor instance

#### `process_region(region_key: str, variables: List[str] = None, use_rich_progress: bool = True, **kwargs) -> Dict`

Convenience function to process a region with default settings.

**Parameters:**
- `region_key` (str): Region identifier
- `variables` (List[str], optional): Variables to process
- `use_rich_progress` (bool): Enable rich progress tracking
- `**kwargs`: Additional configuration parameters

**Returns:**
- `Dict`: Processing results

**Example:**
```python
from means.core.regional_climate_processor import process_region

# Process CONUS with default variables
results = process_region(
    'CONUS',
    variables=['pr', 'tas'],
    max_cores=8,
    batch_size_years=3
)
```

---

## multiprocessing_engine.py

### Classes

#### `MultiprocessingConfig`

Configuration for multiprocessing operations with resource management.

```python
@dataclass
class MultiprocessingConfig:
    max_workers: int
    batch_size: int
    memory_limit_gb: float
    timeout_seconds: int
    retry_attempts: int
    
    # Resource monitoring
    cpu_threshold: float = 0.8
    memory_threshold: float = 0.8
    check_interval: float = 1.0
    
    # Error handling
    max_failures: int = 10
    failure_backoff: float = 2.0
```

**Attributes:**

**Core Settings:**
- `max_workers` (int): Maximum number of worker processes
- `batch_size` (int): Items to process per batch
- `memory_limit_gb` (float): Memory limit in gigabytes
- `timeout_seconds` (int): Timeout for individual tasks
- `retry_attempts` (int): Number of retry attempts for failed tasks

**Resource Monitoring:**
- `cpu_threshold` (float): CPU usage threshold (0.0-1.0)
- `memory_threshold` (float): Memory usage threshold (0.0-1.0)
- `check_interval` (float): Seconds between resource checks

**Error Handling:**
- `max_failures` (int): Maximum failures before aborting
- `failure_backoff` (float): Exponential backoff multiplier

#### `TaskResult`

Result container for multiprocessing tasks.

```python
@dataclass
class TaskResult:
    task_id: str
    success: bool
    result: Any
    error: Optional[str]
    execution_time: float
    memory_used: float
    worker_id: int
```

**Attributes:**
- `task_id` (str): Unique identifier for the task
- `success` (bool): Whether the task completed successfully
- `result` (Any): Task result data (if successful)
- `error` (Optional[str])`: Error message (if failed)
- `execution_time` (float): Execution time in seconds
- `memory_used` (float): Peak memory usage in MB
- `worker_id` (int): ID of the worker that processed the task

#### `MultiprocessingEngine`

Advanced multiprocessing framework with resource monitoring and failure recovery.

```python
class MultiprocessingEngine:
    def __init__(self, config: MultiprocessingConfig)
```

**Key Methods:**

##### `execute_tasks(self, tasks: List[Any], task_function: Callable, progress_callback: Optional[Callable] = None) -> List[TaskResult]`

Execute a list of tasks using multiprocessing with comprehensive monitoring.

**Parameters:**
- `tasks` (List[Any]): List of tasks to execute
- `task_function` (Callable): Function to execute for each task
- `progress_callback` (Optional[Callable]): Callback for progress updates

**Returns:**
- `List[TaskResult]`: Results for all tasks

**Features:**
- Dynamic worker scaling based on system resources
- Real-time memory and CPU monitoring
- Automatic failure recovery with exponential backoff
- Comprehensive progress tracking

##### `monitor_system_resources(self) -> Dict[str, float]`

Monitor current system resource usage.

**Returns:**
- `Dict[str, float]`: Resource usage metrics including:
  - `cpu_percent`: Current CPU usage (0.0-1.0)
  - `memory_percent`: Current memory usage (0.0-1.0)
  - `available_memory_gb`: Available memory in GB
  - `disk_io_read_mb`: Disk read rate in MB/s
  - `disk_io_write_mb`: Disk write rate in MB/s

### Factory Functions

#### `create_optimized_config(max_workers: Optional[int] = None, memory_limit_gb: Optional[float] = None) -> MultiprocessingConfig`

Create an optimized multiprocessing configuration based on system resources.

**Parameters:**
- `max_workers` (Optional[int]): Override automatic worker detection
- `memory_limit_gb` (Optional[float]): Override automatic memory limit

**Returns:**
- `MultiprocessingConfig`: Optimized configuration

**Auto-detection Logic:**
- Workers: min(CPU count, memory_gb // 2)
- Memory limit: 80% of available system memory
- Batch size: Adjusted based on available memory
- Timeouts: Scaled based on expected processing time

#### `create_task_queue(tasks: List[Any], batch_size: int) -> List[List[Any]]`

Create batched task queue for efficient processing.

**Parameters:**
- `tasks` (List[Any]): Input tasks
- `batch_size` (int): Size of each batch

**Returns:**
- `List[List[Any]]`: Batched tasks

**Example:**
```python
from means.core.multiprocessing_engine import (
    MultiprocessingEngine, create_optimized_config
)

# Create optimized configuration
config = create_optimized_config(max_workers=8, memory_limit_gb=16)

# Initialize engine
engine = MultiprocessingEngine(config)

# Define task function
def process_file(file_path):
    # Your processing logic here
    return f"Processed {file_path}"

# Execute tasks
file_paths = ['/data/file1.nc', '/data/file2.nc', '/data/file3.nc']
results = engine.execute_tasks(file_paths, process_file)

# Check results
for result in results:
    if result.success:
        print(f"Task {result.task_id}: {result.result}")
    else:
        print(f"Task {result.task_id} failed: {result.error}")
```

## Error Handling

All core modules implement comprehensive error handling:

### Common Exceptions

- `ValueError`: Invalid configuration or parameters
- `FileNotFoundError`: Missing input files or directories
- `MemoryError`: Insufficient memory for processing
- `TimeoutError`: Task execution timeout
- `CoordinateSystemError`: CRS conversion failures

### Error Recovery Strategies

1. **Graceful Degradation**: Continue processing other items when one fails
2. **Retry Logic**: Automatic retry with exponential backoff
3. **Resource Management**: Reduce resource usage when limits are reached
4. **Detailed Logging**: Comprehensive error reporting and debugging information

### Best Practices

1. **Validate Inputs**: Always validate configuration and input data
2. **Monitor Resources**: Use built-in monitoring to prevent system overload
3. **Handle Failures**: Implement appropriate error handling for your use case
4. **Log Everything**: Enable comprehensive logging for debugging

## Performance Considerations

### Memory Management

- Use chunked processing for large datasets
- Monitor memory usage during processing
- Configure appropriate memory limits
- Clean up resources promptly

### CPU Optimization

- Set worker count based on system capabilities
- Balance I/O and CPU-intensive tasks
- Use batch processing for efficiency
- Monitor CPU usage and adjust accordingly

### I/O Optimization

- Minimize file opening/closing operations
- Use efficient data formats (NetCDF4 with compression)
- Consider parallel I/O for large datasets
- Cache frequently accessed data

## Maximum Performance Processing

### `MaximumPerformanceProcessor`

High-performance climate processor designed for systems with extensive resources (>90GB RAM, >50 CPU cores).

```python
class MaximumPerformanceProcessor:
    def __init__(self, max_workers=48, memory_limit_gb=80, use_rich_progress=True)
```

**Parameters:**
- `max_workers` (int): Maximum number of parallel workers (default: 48)
- `memory_limit_gb` (int): Memory limit in GB (default: 80)
- `use_rich_progress` (bool): Enable rich progress tracking (default: True)

**Key Methods:**

#### `run_maximum_processing(self, variables=None, regions=None, years_range=(1950, 2100))`

Execute maximum performance processing across all available climate data.

**Parameters:**
- `variables` (Optional[List[str]]): Climate variables to process (default: ['pr', 'tas', 'tasmax', 'tasmin'])
- `regions` (Optional[List[str]]): Regions to process (default: ['CONUS', 'AK', 'HI', 'PRVI', 'GU'])
- `years_range` (Tuple[int, int]): Year range for processing (default: (1950, 2100))

**Returns:**
- `Optional[Dict[str, Any]]`: Processing results with performance metrics

**Features:**
- Daily climatology calculation using `groupby('dayofyear').mean()`
- All variable × region × year combinations
- ProcessPoolExecutor with maximum parallelism
- Real-time progress tracking with system monitoring
- Comprehensive performance statistics

### Factory Functions

#### `create_maximum_processor(max_workers=48, memory_limit_gb=80, use_rich_progress=True)`

Create a maximum performance processor with specified settings.

**Parameters:**
- `max_workers` (int): Maximum number of parallel workers
- `memory_limit_gb` (int): Memory limit in GB  
- `use_rich_progress` (bool): Whether to use rich progress tracking

**Returns:**
- `MaximumPerformanceProcessor`: Configured processor instance

#### `run_maximum_processing(variables=None, regions=None, years_range=(1950, 2100), max_workers=48, memory_limit_gb=80, use_rich_progress=True)`

Convenience function to run maximum performance processing.

**Parameters:**
- `variables` (Optional[List[str]]): List of variables to process
- `regions` (Optional[List[str]]): List of regions to process
- `years_range` (Tuple[int, int]): Year range tuple (start, end)
- `max_workers` (int): Maximum number of workers
- `memory_limit_gb` (int): Memory limit in GB
- `use_rich_progress` (bool): Whether to use rich progress tracking

**Returns:**
- `Optional[Dict[str, Any]]`: Processing results dictionary or None if failed

### Usage Examples

#### Basic Maximum Processing

```python
from means.core import run_maximum_processing

# Process all data with default settings
results = run_maximum_processing(
    variables=['pr', 'tas', 'tasmax', 'tasmin'],
    regions=['CONUS', 'AK', 'HI', 'PRVI', 'GU'],
    years_range=(1950, 2100),
    max_workers=48,
    memory_limit_gb=80
)

if results:
    print(f"Processed {results['completed']}/{results['total_tasks']} files")
    print(f"Throughput: {results['throughput']:.1f} files/second")
    print(f"Success rate: {results['success_rate']:.1f}%")
```

#### Advanced Maximum Processing

```python
from means.core import create_maximum_processor

# Create custom processor for high-end system
processor = create_maximum_processor(
    max_workers=56,           # Use most of 64-core system
    memory_limit_gb=90,       # Use most of 128GB RAM
    use_rich_progress=True    # Enable visual progress tracking
)

# Process specific subset
results = processor.run_maximum_processing(
    variables=['pr', 'tas'],
    regions=['CONUS'],
    years_range=(2000, 2050)
)
```

#### Command Line Integration

```python
# The maximum processor can also be run as a command-line script
if __name__ == "__main__":
    from means.core.maximum_processor import main
    sys.exit(main())
```

### Output Structure

The maximum processor generates daily climatology files organized by region and variable:

```
output/data/
├── CONUS/
│   ├── pr/
│   │   ├── pr_CONUS_pr_day_NorESM2-LM_historical_r1i1p1f1_gn_1950_climatology.nc
│   │   ├── pr_CONUS_pr_day_NorESM2-LM_historical_r1i1p1f1_gn_1951_climatology.nc
│   │   └── ...
│   ├── tas/
│   ├── tasmax/
│   └── tasmin/
├── AK/
├── HI/
├── PRVI/
└── GU/
```

### Integration Benefits

- ✅ **Package Integration**: Uses means configuration system
- ✅ **Rich Progress**: Visual progress tracking with system monitoring  
- ✅ **Error Handling**: Comprehensive error capture and reporting
- ✅ **Memory Management**: Explicit cleanup and resource monitoring
- ✅ **Performance Metrics**: Detailed throughput and success rate tracking
- ✅ **Modular Design**: Clean imports and factory functions 