#!/usr/bin/env python3
"""
Test runner for climate data processing modules.

This script provides a convenient way to run all tests and optionally
generate coverage reports.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for climate data processing modules"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Generate coverage report"
    )
    parser.add_argument(
        "--html", 
        action="store_true", 
        help="Generate HTML coverage report (requires --coverage)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--test-file", 
        default="tests/",
        help="Specific test file to run (default: tests/ - all tests)"
    )
    
    args = parser.parse_args()
    
    print("🧪 Climate Data Processing Test Runner")
    print("=" * 60)
    
    # Check if test file exists
    test_file = Path(args.test_file)
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        sys.exit(1)
    
    # Build pytest command
    pytest_cmd = ["python", "-m", "pytest", str(test_file)]
    
    if args.verbose:
        pytest_cmd.append("-v")
    
    if args.coverage:
        pytest_cmd.extend([
            "--cov=climate_means", 
            "--cov=io_util",
            "--cov=run_climate_means",
            "--cov-report=term-missing"
        ])
        
        if args.html:
            pytest_cmd.append("--cov-report=html")
    
    # Run tests
    success = run_command(pytest_cmd, "Integration tests")
    
    if success:
        print("\n🎉 All tests completed successfully!")
        
        if args.coverage and args.html:
            print(f"\n📊 HTML coverage report generated in 'htmlcov/' directory")
            print("   Open 'htmlcov/index.html' in a web browser to view detailed coverage")
        
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
    
    # Additional checks
    print("\n🔍 Running additional checks...")
    
    # Check imports
    import_check = run_command(
        ["python", "-c", "import climate_means, io_util, run_climate_means; print('✅ All modules import successfully')"],
        "Import validation"
    )
    
    if not import_check:
        print("❌ Import validation failed")
        sys.exit(1)
    
    print("\n✨ All checks passed! The modules are properly integrated.")


if __name__ == "__main__":
    main() 