# Climate Data Processing Tests

This directory contains integration tests for the climate data processing modules.

## Test Structure

### `test_integration.py`
Comprehensive integration tests that verify the interaction between `climate_means.py` and `io_util.py` modules.

**Test Classes:**

1. **TestModuleIntegration**: Tests basic module integration
   - Import verification
   - Object identity checks
   - Constant consistency

2. **TestNorESM2FileHandlerIntegration**: Tests file handler integration
   - File handler creation from climate_means module
   - Method accessibility
   - File discovery functionality

3. **TestDatasetOperations**: Tests dataset I/O operations
   - Safe dataset opening
   - Optimized dataset opening
   - Climate result saving

4. **TestWorkflowIntegration**: Tests end-to-end workflow integration
   - Function accessibility
   - Mocked workflow testing
   - Configuration validation

5. **TestErrorHandling**: Tests error handling scenarios
   - Invalid file paths
   - Missing directories
   - Save failures

## Running Tests

### Run all integration tests:
```bash
python -m pytest tests/test_integration.py -v
```

### Run specific test class:
```bash
python -m pytest tests/test_integration.py::TestModuleIntegration -v
```

### Run specific test:
```bash
python -m pytest tests/test_integration.py::TestModuleIntegration::test_imports_successful -v
```

### Run with coverage (if coverage is installed):
```bash
python -m pytest tests/ --cov=climate_means --cov=io_util --cov-report=html
```

## Test Requirements

- pytest
- numpy
- xarray
- pathlib (built-in)
- tempfile (built-in)
- unittest.mock (built-in)

## What the Tests Verify

### Module Integration
- ✅ All I/O functions are properly imported from `io_util.py` to `climate_means.py`
- ✅ Imported objects are the same instances (not copies)
- ✅ Constants are properly shared between modules

### File Handler Integration
- ✅ `NorESM2FileHandler` can be instantiated from `climate_means.py`
- ✅ All methods are accessible and functional
- ✅ File discovery works with mock directory structures

### Dataset Operations
- ✅ Dataset opening functions work correctly
- ✅ Save functionality works with temporary files
- ✅ Error handling works for invalid inputs

### Workflow Integration
- ✅ High-level functions can use imported I/O utilities
- ✅ Configuration validation works
- ✅ Mocked workflows execute properly

## Notes

- Tests use temporary directories and files to avoid requiring real climate data
- Mock objects are used to test complex workflows without full execution
- Error conditions are tested to ensure robust behavior
- Tests focus on integration rather than unit-level functionality

## Adding New Tests

When adding new tests:

1. Follow the existing naming convention (`test_*`)
2. Use appropriate fixtures for temporary resources
3. Clean up resources after tests
4. Use mocking for external dependencies
5. Test both success and error cases 