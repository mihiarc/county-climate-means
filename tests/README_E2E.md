# End-to-End Testing for Climate Means Program

This directory contains comprehensive end-to-end tests for the climate means processing workflow.

## Overview

The end-to-end tests verify the complete functionality of the climate means program, from input data ingestion through final output generation. These tests use realistic mock data and test the integration between all major components.

## Test Structure

### Main Test File: `test_climate_means_e2e.py`

Contains comprehensive end-to-end tests organized into several test classes:

#### `TestClimateWorkflowEndToEnd`
- **Purpose**: Test the complete climate processing workflow
- **Key Tests**:
  - `test_complete_workflow_execution`: Full workflow with mock NetCDF data
  - `test_workflow_with_multiple_variables_and_regions`: Multi-variable/region processing
  - `test_workflow_error_handling`: Error scenarios and recovery
  - `test_climate_normal_calculation_accuracy`: Validate output accuracy

#### `TestRunnerEndToEnd`  
- **Purpose**: Test the runner script execution paths
- **Key Tests**:
  - `test_runner_noresm2_mode_execution`: NorESM2 data processing mode
  - `test_runner_main_function_execution`: Basic demonstration mode

#### `TestDataFlowIntegration`
- **Purpose**: Test data flow through the entire pipeline
- **Key Tests**:
  - `test_file_handler_to_workflow_integration`: File discovery → processing
  - `test_region_extraction_workflow`: Geographic region extraction

#### `TestPerformanceAndScalability`
- **Purpose**: Test performance characteristics and resource usage
- **Key Tests**:
  - `test_workflow_memory_efficiency`: Memory usage patterns
  - `test_dask_integration_scaling`: Distributed computing scaling

## Running Tests

### Quick Start

```bash
# Install testing dependencies
uv add pytest pytest-cov pytest-mock

# Run all end-to-end tests
python run_e2e_tests.py

# Run fast tests only (excludes performance tests)
python run_e2e_tests.py --fast

# Run with verbose output
python run_e2e_tests.py --verbose

# Run with coverage reporting
python run_e2e_tests.py --coverage
```

### Test Categories

#### Integration Tests
```bash
python run_e2e_tests.py --integration
```
Tests marked with `@pytest.mark.integration` that verify component interactions.

#### Performance Tests
```bash
python run_e2e_tests.py --performance
```
Tests marked with `@pytest.mark.slow` that verify memory usage and scaling behavior.

#### Specific Tests
```bash
python run_e2e_tests.py --specific test_complete_workflow_execution
```
Run a specific test function by name.

### Using pytest Directly

```bash
# Run all e2e tests
pytest tests/test_climate_means_e2e.py -v

# Run only integration tests
pytest tests/test_climate_means_e2e.py -m integration -v

# Run fast tests only
pytest tests/test_climate_means_e2e.py -m "not slow" -v

# Run with coverage
pytest tests/test_climate_means_e2e.py --cov=climate_means --cov-report=html
```

## Test Data

### Mock Data Generation

The tests automatically create realistic mock NetCDF files with:

- **Temporal Coverage**: Full years with proper daily time series
- **Spatial Coverage**: Global grid covering all US regions (CONUS, AK, HI, etc.)
- **Variables**: Temperature (tas, tasmax, tasmin) and precipitation (pr)
- **Scenarios**: Historical (1980-2014) and future (2015-2050) scenarios
- **Realistic Values**: Scientifically reasonable data ranges and patterns

### Data Structure

Mock data follows the NorESM2-LM directory structure:
```
temp_data/
├── NorESM2-LM/
│   ├── tas/
│   │   ├── historical/
│   │   │   ├── tas_day_NorESM2-LM_historical_r1i1p1f1_gn_1980.nc
│   │   │   └── ...
│   │   ├── ssp245/
│   │   └── ssp585/
│   ├── tasmax/
│   ├── tasmin/
│   └── pr/
```

## Test Verification

### What Gets Tested

1. **Data Input/Output**:
   - File discovery and reading
   - NetCDF format compliance
   - Output file generation

2. **Processing Logic**:
   - Climate normal calculations (30-year means)
   - Regional extraction (CONUS, AK, HI, etc.)
   - Time coordinate handling
   - Daily climatology computation

3. **Integration**:
   - Module interactions
   - Dask distributed computing
   - Error handling and recovery
   - Memory management

4. **Results Validation**:
   - Output data ranges (temperature in reasonable K range)
   - Spatial patterns (variation across regions)
   - Metadata preservation
   - File naming conventions

### Expected Outputs

Successful tests should produce:
- NetCDF files in the output directory
- Reasonable temperature values (200-350K)
- Proper spatial variation across regions
- Correct metadata attributes
- Memory usage within expected bounds

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   uv add pytest pytest-cov numpy xarray dask netcdf4 psutil
   ```

2. **Memory Issues**:
   - Run with `--fast` to exclude memory-intensive tests
   - Reduce mock data size in `_create_mock_netcdf_file`

3. **Dask Errors**:
   - Tests use minimal Dask configurations (1-2 workers)
   - Check system resources if Dask setup fails

4. **NetCDF Issues**:
   - Ensure netcdf4 library is properly installed
   - Check file permissions in temporary directories

### Debug Mode

Run individual test functions with verbose output:
```bash
pytest tests/test_climate_means_e2e.py::TestClimateWorkflowEndToEnd::test_complete_workflow_execution -v -s
```

## Test Configuration

### Markers
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Performance/memory tests  
- `@pytest.mark.e2e`: End-to-end tests

### Fixtures
- `temp_workspace`: Creates complete mock data environment
- `temp_noresm2_structure`: Minimal NorESM2 structure for quick tests

### Configuration Files
- `pytest.ini`: Test discovery and marker configuration
- `run_e2e_tests.py`: Convenient test runner script

## Contributing

### Adding New Tests

1. Follow the existing test class structure
2. Use appropriate markers (`@pytest.mark.integration`, etc.)
3. Include proper docstrings explaining test purpose
4. Clean up temporary files in fixtures
5. Verify tests work with both mock and real data scenarios

### Test Guidelines

- Tests should be independent and not rely on external data
- Use realistic mock data that represents actual climate files
- Include both success and failure scenarios
- Verify both functionality and performance characteristics
- Document expected behavior and validation criteria

## Coverage Reports

Generate detailed coverage reports:
```bash
python run_e2e_tests.py --coverage
```

This creates:
- Terminal coverage summary
- HTML coverage report in `htmlcov/`
- Coverage data in `.coverage` file

View the HTML report by opening `htmlcov/index.html` in a browser. 