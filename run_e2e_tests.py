#!/usr/bin/env python3
"""
End-to-End Test Runner for Climate Means Program

This script provides a convenient way to run comprehensive end-to-end tests
for the climate means processing workflow.

Usage:
    python run_e2e_tests.py                    # Run all e2e tests
    python run_e2e_tests.py --fast            # Run fast tests only
    python run_e2e_tests.py --integration     # Run integration tests only
    python run_e2e_tests.py --performance     # Run performance tests only
    python run_e2e_tests.py --verbose         # Run with verbose output
"""

import sys
import subprocess
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pytest_command(test_args, markers=None, verbose=False):
    """Run pytest with specified arguments and markers."""
    cmd = ['python', '-m', 'pytest']
    
    # Add test file
    cmd.append('tests/test_climate_means_e2e.py')
    
    # Add markers if specified
    if markers:
        for marker in markers:
            cmd.extend(['-m', marker])
    
    # Add verbosity
    if verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')
    
    # Add additional args
    cmd.extend(test_args)
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Error running pytest: {e}")
        return False


def check_dependencies():
    """Check that required dependencies are available."""
    required_modules = [
        'pytest', 'numpy', 'xarray', 'pandas', 'netCDF4', 
        'dask', 'psutil', 'tempfile', 'pathlib'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing required modules: {missing_modules}")
        logger.info("Install missing modules using: uv add <module_name>")
        return False
    
    return True


def main():
    """Main function to run end-to-end tests."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end tests for climate means program"
    )
    
    parser.add_argument(
        '--fast', 
        action='store_true',
        help='Run only fast tests (exclude slow and performance tests)'
    )
    
    parser.add_argument(
        '--integration',
        action='store_true', 
        help='Run only integration tests'
    )
    
    parser.add_argument(
        '--performance',
        action='store_true',
        help='Run only performance tests'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Run with verbose output'
    )
    
    parser.add_argument(
        '--specific',
        type=str,
        help='Run specific test function (e.g., test_complete_workflow_execution)'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run with coverage reporting'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    logger.info("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check that test file exists
    test_file = Path('tests/test_climate_means_e2e.py')
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        sys.exit(1)
    
    # Prepare pytest arguments
    pytest_args = []
    markers = []
    
    # Add coverage if requested
    if args.coverage:
        pytest_args.extend(['--cov=climate_means', '--cov=io_util', '--cov=regions', 
                           '--cov=time_util', '--cov=dask_util', '--cov=run_climate_means'])
        pytest_args.append('--cov-report=html')
        pytest_args.append('--cov-report=term-missing')
    
    # Configure test selection based on arguments
    if args.fast:
        # Exclude slow tests
        markers.append('not slow')
        logger.info("Running fast tests only...")
    
    elif args.integration:
        # Run only integration tests
        markers.append('integration')
        logger.info("Running integration tests only...")
        
    elif args.performance:
        # Run only performance tests  
        markers.append('slow')
        logger.info("Running performance tests only...")
        
    elif args.specific:
        # Run specific test
        pytest_args.extend(['-k', args.specific])
        logger.info(f"Running specific test: {args.specific}")
        
    else:
        # Run all tests
        logger.info("Running all end-to-end tests...")
    
    # Add some useful pytest options
    pytest_args.extend([
        '--tb=short',           # Short traceback format
        '--strict-markers',     # Strict marker checking
        '-ra',                  # Show summary of all results
    ])
    
    # Run the tests
    logger.info("Starting test execution...")
    success = run_pytest_command(pytest_args, markers, args.verbose)
    
    if success:
        logger.info("✓ All tests completed successfully!")
        if args.coverage:
            logger.info("Coverage report generated in htmlcov/")
    else:
        logger.error("✗ Some tests failed!")
        sys.exit(1)


def print_test_info():
    """Print information about available tests."""
    print("Available End-to-End Tests:")
    print("=" * 50)
    print()
    print("Test Classes:")
    print("- TestClimateWorkflowEndToEnd: Complete workflow testing")
    print("- TestRunnerEndToEnd: Runner script testing") 
    print("- TestDataFlowIntegration: Data flow testing")
    print("- TestPerformanceAndScalability: Performance testing")
    print()
    print("Key Test Functions:")
    print("- test_complete_workflow_execution: Full workflow with mock data")
    print("- test_workflow_with_multiple_variables_and_regions: Multi-region testing")
    print("- test_workflow_error_handling: Error scenarios")
    print("- test_climate_normal_calculation_accuracy: Result validation")
    print("- test_runner_noresm2_mode_execution: Runner script testing")
    print("- test_file_handler_to_workflow_integration: Data flow testing")
    print("- test_workflow_memory_efficiency: Memory usage testing")
    print()
    print("Test Markers:")
    print("- integration: Integration tests")
    print("- slow: Performance/memory tests")
    print()
    print("Usage Examples:")
    print("  python run_e2e_tests.py --fast")
    print("  python run_e2e_tests.py --integration --verbose")
    print("  python run_e2e_tests.py --specific test_complete_workflow_execution")
    print("  python run_e2e_tests.py --coverage")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'info':
        print_test_info()
    else:
        main() 