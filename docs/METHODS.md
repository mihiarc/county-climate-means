# Climate Means Business Logic Documentation

## Overview

The `means` package implements a comprehensive climate data processing system for calculating 30-year climate normals (averages) from NEX-GDDP-CMIP6 climate model data. This document explains the core business logic and workflows that power the system.

## Architecture Overview

```
means/
├── core/                    # Core business logic and domain models
│   ├── regions.py          # Regional definitions and geographic operations
│   ├── regional_climate_processor.py  # Unified regional processing engine
│   ├── multiprocessing_engine.py     # Parallel processing framework
│   └── __init__.py         # Core module exports
├── utils/                   # Utility modules and support functions
│   ├── io_util.py          # File I/O and data handling
│   ├── time_util.py        # Time coordinate and period calculations
│   ├── rich_progress.py    # Advanced progress tracking and monitoring
│   └── __init__.py
├── visualization/           # Data visualization and validation
│   ├── regional_visualizer.py  # Regional climate data visualization
│   └── __init__.py
├── validation/             # Data validation and quality assurance
│   ├── validate_region_extents.py     # Geographic boundary validation
│   ├── validate_climate_data_alignment.py  # Data-geography alignment checks
│   └── __init__.py
└── config.py               # Centralized configuration management
```

## Core Business Logic Components

### 1. Climate Normal Calculation Workflow

**Business Rule**: A climate normal is a 30-year average of daily climatological values for a specific target year.

**Process Flow**:
1. **Data Collection**: Gather 30 years of daily climate data ending at the target year
2. **Regional Extraction**: Extract data for specific geographic regions using coordinate-aware algorithms
3. **Daily Climatology**: Calculate daily averages (day-of-year basis) for each year
4. **Climate Normal**: Average the daily climatologies across all 30 years
5. **Metadata Enrichment**: Add comprehensive metadata for traceability

**Example**: For target year 2000, collect data from 1971-2000, calculate daily averages for each year, then average across all 30 years.

### 2. Regional Processing Architecture

#### Regional Definitions (`core/regions.py`)

**Business Purpose**: Define precise geographic boundaries for U.S. climate regions with proper coordinate reference systems.

**Supported Regions**:
- **CONUS**: Continental United States
- **AK**: Alaska 
- **HI**: Hawaii and Pacific Islands
- **PRVI**: Puerto Rico and U.S. Virgin Islands
- **GU**: Guam and Northern Mariana Islands

**Key Business Logic**:

```python
# Regional boundary definitions use 0-360° longitude system by default
REGION_BOUNDS = {
    'CONUS': {
        'name': 'CONUS',
        'lon_min': 234,   # -126°W in standard coordinates
        'lon_max': 294,   # -66°W in standard coordinates  
        'lat_min': 24.0,  # Southern Florida
        'lat_max': 50.0,  # Northern border
        'convert_longitudes': True
    }
    # ... other regions
}
```

**Coordinate System Handling**:
- **Detection**: Automatically detects 0-360° vs -180°/180° longitude systems
- **Conversion**: Seamlessly converts between coordinate systems
- **Validation**: Handles dateline crossing for Alaska and Pacific regions

#### Unified Regional Processor (`core/regional_climate_processor.py`)

**Business Purpose**: Provide a single, parameterized processor that can handle any region and climate variable combination.

**Key Components**:

1. **RegionalProcessingConfig**: Configuration dataclass that encapsulates all processing parameters
   - Region specification
   - Variable selection
   - Processing performance settings
   - Memory management parameters
   - Progress tracking configuration

2. **RegionalClimateProcessor**: Main processing engine that orchestrates the complete workflow
   - Multiprocessing coordination
   - Progress tracking and reporting
   - Error handling and recovery
   - Output file management

**Business Workflow**:
```python
# Example usage pattern
processor = create_regional_processor(
    region_key='CONUS',
    variables=['pr', 'tas', 'tasmax', 'tasmin'],
    use_rich_progress=True
)
results = processor.process_all_variables()
```

**Period Types Processed**:
- **Historical**: 1980-2014 (using historical climate data)
- **Hybrid**: 2015-2044 (mixing historical and projection data)
- **SSP2-4.5**: 2045-2100 (future projection scenario)

### 3. Multiprocessing Framework (`core/multiprocessing_engine.py`)

**Business Purpose**: Provide enterprise-grade parallel processing capabilities with resource management, progress tracking, and error handling.

#### Core Components

1. **MultiprocessingConfig**: Comprehensive configuration for parallel operations
   ```python
   @dataclass
   class MultiprocessingConfig:
       max_workers: int = 6                    # Core processing setting
       cores_per_variable: int = 2             # Resource allocation
       batch_size: int = 2                     # Memory management
       memory_per_worker_gb: float = 4.0       # Resource limits
       timeout_per_task: int = 300             # Reliability
       max_retries: int = 2                    # Error recovery
       enable_progress_tracking: bool = True   # Monitoring
   ```

2. **TaskResult**: Standardized result container for tracking task outcomes
   - Success/failure status
   - Execution metrics (time, memory usage)
   - Error information for debugging
   - Worker identification for performance analysis

3. **MultiprocessingEngine**: Main parallel processing orchestrator
   - **Resource Management**: Auto-detects optimal worker count based on system resources
   - **Progress Tracking**: Real-time monitoring with rich visual feedback
   - **Error Handling**: Comprehensive error capture and retry logic
   - **Memory Monitoring**: Prevents out-of-memory conditions

**Business Logic for Resource Optimization**:
```python
def _auto_detect_optimal_workers(self) -> int:
    """Auto-detect optimal number of workers based on system resources."""
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Conservative estimates
    max_workers_by_memory = max(1, int(memory_gb / self.memory_per_worker_gb))
    max_workers_by_cpu = max(1, cpu_count - 2)  # Leave 2 CPUs free
    
    # Use the more restrictive limit
    optimal_workers = min(6, max_workers_by_memory, max_workers_by_cpu)
    return optimal_workers
```

### 4. Configuration Management (`config.py`)

**Business Purpose**: Centralized, hierarchical configuration system supporting environment variables, config files, and programmatic overrides.

#### Configuration Hierarchy (highest to lowest priority):
1. **Programmatic overrides** (passed to functions)
2. **Environment variables** (CLIMATE_*)
3. **Configuration files** (.yaml)
4. **Default values** (hardcoded)

#### Key Configuration Classes:

1. **DataPaths**: File system layout and directory management
   ```python
   @dataclass
   class DataPaths:
       input_data_dir: str = "/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
       output_base_dir: str = "output"
       conus_output_dir: str = "output/conus_normals"
       # ... regional output directories
   ```

2. **ProcessingConfig**: Performance and processing behavior
   ```python
   @dataclass  
   class ProcessingConfig:
       max_workers: int = 4                    # Parallel processing
       batch_size: int = 15                    # Memory management
       memory_conservative: bool = True        # Resource usage
       climate_period_length: int = 30         # Business rule
   ```

3. **LoggingConfig**: Monitoring and debugging configuration

**Business Logic for Environment Integration**:
- Supports Docker containerization
- Enables CI/CD pipeline configuration
- Allows runtime configuration without code changes

### 5. File I/O and Data Handling (`utils/io_util.py`)

**Business Purpose**: Provide reliable, efficient access to climate data files with proper error handling and metadata extraction.

#### NorESM2FileHandler Class

**Expected Data Structure**:
```
data_dir/
├── pr/
│   ├── historical/
│   │   ├── pr_day_NorESM2-LM_historical_r1i1p1f1_gn_1950.nc
│   │   └── pr_day_NorESM2-LM_historical_r1i1p1f1_gn_1951.nc
│   └── ssp245/
│       ├── pr_day_NorESM2-LM_ssp245_r1i1p1f1_gn_2015.nc
│       └── pr_day_NorESM2-LM_ssp245_r1i1p1f1_gn_2016.nc
└── tas/
    ├── historical/
    └── ssp245/
```

**Key Business Methods**:

1. **get_files_for_period()**: Core data discovery method
   ```python
   def get_files_for_period(self, variable: str, scenario: str, 
                           start_year: int, end_year: int) -> List[str]:
       """Get list of files for a variable, scenario, and year range."""
   ```

2. **get_hybrid_files_for_period()**: Business logic for transition periods
   ```python
   def get_hybrid_files_for_period(self, variable: str, target_year: int, 
                                   window_years: int = 30) -> Tuple[List[str], Dict[str, int]]:
       """Get files for periods spanning historical and future scenarios."""
   ```

**Filename Pattern Recognition**:
- Extracts years from standardized filenames
- Handles version suffixes (e.g., `_v1.1.nc`)
- Validates file existence and accessibility

### 6. Time Coordinate Handling (`utils/time_util.py`)

**Business Purpose**: Handle complex time coordinate systems, period calculations, and climatology operations.

#### Key Business Logic Components:

1. **Period Generation**: Calculate 30-year periods for different scenarios
   ```python
   def generate_climate_periods(scenario: str, data_availability: Dict) -> List[Tuple[int, int, int, str]]:
       """Generate climate periods based on scenario and data availability."""
   ```

2. **Time Coordinate Handling**: Manage day-of-year coordinates for climatology
   ```python
   def handle_time_coordinates(ds: xr.Dataset, file_path: str) -> Tuple[xr.Dataset, str]:
       """Create day-of-year coordinates for daily climatology calculation."""
   ```

3. **Leap Year Management**: Standardize to 365-day year for consistency
   ```python
   def standardize_to_365_days(day_of_year: np.ndarray, year: int) -> np.ndarray:
       """Standardize day-of-year to 365 days, handling leap years."""
   ```

**Business Rules for Time Handling**:
- All climatologies use 365-day year (remove Feb 29 from leap years)
- Day-of-year coordinates enable seasonal analysis
- Support for multiple time encoding formats

### 7. Progress Tracking and Monitoring (`utils/rich_progress.py`)

**Business Purpose**: Provide enterprise-grade progress tracking with real-time monitoring, performance metrics, and visual feedback.

#### RichProgressTracker Class

**Key Features**:
- **Real-time Progress Bars**: Visual feedback for long-running operations
- **System Monitoring**: CPU, memory, and disk usage tracking
- **Performance Metrics**: Throughput calculation and ETA estimation
- **Hierarchical Tasks**: Support for nested progress tracking
- **Persistence**: Save progress state for recovery

**Business Logic for Progress Calculation**:
```python
@property
def success_rate(self) -> float:
    """Calculate success rate percentage."""
    if self.completed + self.failed == 0:
        return 100.0
    return (self.completed / (self.completed + self.failed)) * 100

@property  
def throughput(self) -> float:
    """Calculate processing throughput in items per second."""
    if not self.start_time or self.completed == 0:
        return 0.0
    elapsed = (datetime.now() - self.start_time).total_seconds()
    return self.completed / elapsed if elapsed > 0 else 0.0
```

## Data Processing Workflows

### Sequential Processing Workflow

**Use Case**: Single-threaded processing for debugging or resource-constrained environments.

**Workflow Steps**:
1. Initialize file handler and validate data availability
2. For each target year:
   - Collect 30 years of data files
   - Process each file to extract daily climatology
   - Combine climatologies into climate normal
   - Save result with comprehensive metadata

### Parallel Processing Workflow  

**Use Case**: Production processing for maximum performance.

**Workflow Steps**:
1. Initialize multiprocessing engine with resource detection
2. Generate task batches (years grouped for efficiency)
3. Distribute tasks across worker processes
4. Monitor progress and system resources
5. Collect results and handle errors
6. Generate processing summary and performance metrics

### Maximum Performance Processing Workflow

**Use Case**: High-end systems with extensive resources (>90GB RAM, >50 CPU cores) requiring maximum throughput.

**Business Purpose**: Process all available climate data with maximum parallelism for comprehensive climate analysis.

#### MaximumPerformanceProcessor (`core/maximum_processor.py`)

**Integration Story**: Originally developed as a standalone script (`maximum_processing.py`), this functionality has been fully integrated into the means package architecture while maintaining its high-performance characteristics.

**Key Features**:
- **Optimized for High-End Systems**: Designed for systems with 90+ GB RAM and 50+ CPU cores
- **Maximum Parallelism**: Up to 48+ concurrent workers processing climatology calculations
- **Comprehensive Coverage**: Processes all variables × regions × years combinations
- **Daily Climatology Focus**: Calculates `groupby('dayofyear').mean()` for seasonal analysis
- **Rich Progress Tracking**: Real-time monitoring with system resource feedback
- **Package Integration**: Uses means configuration system and standard imports

**Business Logic Workflow**:
```python
# Integrated usage pattern
from means.core import run_maximum_processing

results = run_maximum_processing(
    variables=['pr', 'tas', 'tasmax', 'tasmin'],
    regions=['CONUS', 'AK', 'HI', 'PRVI', 'GU'],
    years_range=(1950, 2100),
    max_workers=48,
    memory_limit_gb=80,
    use_rich_progress=True
)
```

**Processing Strategy**:
1. **Task Generation**: Create processing tasks for all file × region combinations
2. **Concurrent Execution**: Use `ProcessPoolExecutor` with maximum worker pool
3. **Daily Climatology**: Calculate day-of-year based climatologies for each file
4. **Regional Extraction**: Apply coordinate-aware regional boundaries
5. **Memory Management**: Explicit cleanup and garbage collection per task
6. **Performance Monitoring**: Track throughput, success rates, and system resources

**Data Processing Flow**:
```python
def process_single_file_chunk(self, args):
    file_path, variable, region_key, output_dir = args
    
    # Load and extract regional data
    ds = xr.open_dataset(file_path)
    region_bounds = REGION_BOUNDS[region_key]
    region_ds = extract_region(ds, region_bounds)
    
    # Calculate daily climatology
    var_data = region_ds[variable]
    if 'dayofyear' not in var_data.coords:
        var_data = var_data.assign_coords(dayofyear=var_data.time.dt.dayofyear)
    
    daily_clim = var_data.groupby('dayofyear').mean(dim='time')
    
    # Save with cleanup
    output_file = output_dir / f"{variable}_{region_key}_{Path(file_path).stem}_climatology.nc"
    daily_clim.to_netcdf(output_file)
    
    # Explicit memory management
    ds.close()
    del ds, region_ds, var_data, daily_clim
```

**Output Structure**:
```
output/data/
├── CONUS/
│   ├── pr/
│   │   ├── pr_CONUS_pr_day_NorESM2-LM_historical_r1i1p1f1_gn_1950_climatology.nc
│   │   └── pr_CONUS_pr_day_NorESM2-LM_historical_r1i1p1f1_gn_1951_climatology.nc
│   ├── tas/
│   ├── tasmax/
│   └── tasmin/
├── AK/
└── HI/
```

**Performance Characteristics**:
- **Throughput**: 10+ files/second on high-end systems
- **Memory Usage**: Controlled through explicit cleanup and worker limits
- **CPU Utilization**: >95% utilization across all cores
- **Success Monitoring**: Real-time failure tracking and retry logic

**Integration Benefits**:
- ✅ Uses means package configuration system
- ✅ Consistent with package import patterns
- ✅ Rich progress tracking integration
- ✅ Proper error handling and logging
- ✅ Compatible with package documentation
- ✅ Follows established code patterns

### Hybrid Period Processing

**Business Logic**: For transition periods (2015-2044), mix historical and projection data to maintain 30-year windows.

**Algorithm**:
```python
for year in range(start_year, target_year + 1):
    # Try historical first (more reliable)
    hist_files = get_files_for_period(variable, 'historical', year, year)
    if hist_files:
        all_files.extend(hist_files)
        historical_count += 1
    else:
        # Fall back to projection data
        ssp245_files = get_files_for_period(variable, 'ssp245', year, year)
        if ssp245_files:
            all_files.extend(ssp245_files)
            projection_count += 1
```

## Data Quality and Validation

### Geographic Validation

**Business Purpose**: Ensure climate data properly aligns with defined regional boundaries.

**Validation Methods**:
1. **Coordinate System Detection**: Automatically detect longitude conventions
2. **Boundary Alignment**: Verify data coverage matches regional extents  
3. **Visual Validation**: Generate maps overlaying data on regional boundaries
4. **Gap Detection**: Identify missing data or coordinate misalignments

### Processing Validation

**Quality Checks**:
- Minimum year requirements (25+ years for valid normal)
- Data completeness validation
- Coordinate consistency checking
- Output file integrity verification

## Error Handling and Recovery

### Graceful Degradation

**Business Logic**: System continues processing even when individual tasks fail.

**Error Handling Strategies**:
1. **Task-Level Isolation**: Failures in one task don't affect others
2. **Retry Logic**: Configurable retry attempts with exponential backoff
3. **Partial Results**: Save successful results even if some tasks fail
4. **Comprehensive Logging**: Detailed error information for debugging

### Resource Management

**Memory Protection**:
```python
# Monitor memory usage per worker
if process.memory_info().rss > max_memory_per_process:
    logger.warning("Worker approaching memory limit, triggering garbage collection")
    gc.collect()
```

**CPU Throttling**:
- Leave 2 CPUs free for system operations
- Dynamic worker adjustment based on system load
- Process priority management for background operation

## Output Structure and Metadata

### File Organization

**Business Rule**: Outputs organized by variable, scenario, and target year for easy discovery.

```
output/
├── {region}_normals/
│   ├── pr/
│   │   ├── historical/
│   │   │   ├── pr_{region}_historical_1980_30yr_normal.nc
│   │   │   └── pr_{region}_historical_1981_30yr_normal.nc
│   │   ├── hybrid/
│   │   └── ssp245/
│   ├── tas/
│   ├── tasmax/
│   └── tasmin/
└── visualizations/
```

### Metadata Standards

**Required Metadata**:
```python
climate_normal.attrs.update({
    'title': f'{variable.upper()} 30-Year {period_type.title()} Climate Normal ({region_key}) - Target Year {target_year}',
    'variable': variable,
    'region': region_key,
    'target_year': target_year,
    'period_type': period_type,
    'num_years': len(years_used),
    'processing_method': 'unified_regional_processor',
    'source': 'NorESM2-LM climate model',
    'method': '30-year rolling climate normal',
    'created': datetime.now().isoformat()
})
```

## Performance Optimization

### Memory Management

**Strategies**:
1. **Lazy Loading**: Open datasets without loading into memory
2. **Chunking**: Process data in memory-efficient chunks
3. **Garbage Collection**: Explicit cleanup after processing
4. **Conservative Defaults**: Memory-safe configuration options

### I/O Optimization

**Techniques**:
1. **Batch Processing**: Group related operations
2. **Compression**: Use NetCDF4 compression for outputs
3. **Caching**: Cache frequently accessed metadata
4. **Parallel I/O**: Concurrent file operations where safe

### CPU Optimization

**Approaches**:
1. **Vectorized Operations**: Use NumPy/Xarray optimized operations
2. **Process Pools**: Distribute CPU-intensive work
3. **Algorithm Selection**: Choose optimal algorithms for data size
4. **Resource Monitoring**: Adjust parallelism based on system load

## Integration Patterns

### Factory Pattern

**Usage**: Create configured processors without complex initialization.

```python
# Factory function for easy processor creation
def create_regional_processor(region_key: str, variables: List[str] = None, **kwargs):
    """Factory function to create a regional processor with configuration."""
    config = RegionalProcessingConfig(
        region_key=region_key,
        variables=variables or ['pr', 'tas', 'tasmax', 'tasmin'],
        **kwargs
    )
    return RegionalClimateProcessor(config)
```

### Command Pattern

**Usage**: Encapsulate processing operations for queuing and retry.

```python
# Tasks are encapsulated as callable objects
def process_climate_file_task(file_path: str, variable_name: str, region_key: str):
    """Multiprocessing-safe task for processing a single climate file."""
    # Implementation encapsulates all processing logic
```

### Observer Pattern

**Usage**: Progress tracking and monitoring system.

```python
# Progress tracking observers
class ProgressTracker:
    def report_task_start(self, task_id: str, worker_id: int):
        """Observer pattern for task lifecycle events."""
        
    def report_task_complete(self, task_result: TaskResult):
        """Update progress and calculate metrics."""
```

## Testing and Validation Strategies

### Unit Testing Patterns

**Core Business Logic Tests**:
- Coordinate conversion accuracy
- Period calculation correctness
- Regional boundary validation
- File discovery logic
- Metadata generation

### Integration Testing

**End-to-End Workflows**:
- Complete processing pipeline validation
- Multi-region processing consistency
- Performance regression testing
- Resource usage validation

### Data Quality Testing

**Scientific Validation**:
- Climate normal calculation accuracy
- Coordinate alignment verification
- Temporal consistency checking
- Regional coverage validation

## Deployment and Operations

### Configuration Management

**Production Deployment**:
```bash
# Environment-based configuration
export CLIMATE_INPUT_DIR="/data/climate/noresm2"
export CLIMATE_OUTPUT_DIR="/data/outputs" 
export CLIMATE_MAX_WORKERS="8"
export CLIMATE_LOG_LEVEL="INFO"
```

### Monitoring and Alerting

**Operational Metrics**:
- Processing throughput (files/hour)
- Success/failure rates
- Resource utilization (CPU, memory, disk)
- Queue depth and processing lag

### Scalability Considerations

**Horizontal Scaling**:
- Process splitting by region or variable
- Distributed processing across multiple nodes
- Shared storage requirements
- Coordination and result aggregation

This comprehensive business logic documentation provides the foundation for understanding, maintaining, and extending the climate means processing system. Each component is designed with clear separation of concerns, robust error handling, and performance optimization in mind. 