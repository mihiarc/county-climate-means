#!/usr/bin/env python3
"""
Run the complete climate processing pipeline with proper configuration.

This script provides a convenient way to run the full pipeline with
various options and configurations.
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime


def run_pipeline(config_file: str, dry_run: bool = False, stage: str = None):
    """Run the pipeline with the specified configuration."""
    
    cmd = [
        sys.executable,
        "main_orchestrated.py",
        "run",
        "--config", config_file,
        "--save-report"
    ]
    
    if stage:
        cmd.extend(["--stage", stage])
    
    print(f"\n{'='*80}")
    print(f"CLIMATE PROCESSING PIPELINE")
    print(f"{'='*80}")
    print(f"Configuration: {config_file}")
    print(f"Start time: {datetime.now()}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("\n[DRY RUN] Would execute the above command")
        return 0
    
    print(f"\n{'='*80}")
    print("PIPELINE EXECUTION")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Run the pipeline
        result = subprocess.run(cmd, check=True)
        
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Duration: {hours}h {minutes}m {seconds}s")
        print(f"End time: {datetime.now()}")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"PIPELINE FAILED")
        print(f"{'='*80}")
        print(f"Error code: {e.returncode}")
        print(f"Duration: {time.time() - start_time:.0f} seconds")
        return e.returncode
    except KeyboardInterrupt:
        print(f"\n\n{'='*80}")
        print(f"PIPELINE INTERRUPTED BY USER")
        print(f"{'='*80}")
        return 1


def estimate_requirements(config_name: str):
    """Show estimated requirements for the configuration."""
    
    estimates = {
        "test": {
            "files": 5,
            "storage": "< 1 GB",
            "time": "10-15 minutes",
            "memory": "16 GB",
            "description": "Test configuration with minimal data"
        },
        "staged": {
            "files": 18960,
            "storage": "~43 GB",
            "time": "48-72 hours",
            "memory": "60 GB",
            "description": "Complete processing in manageable stages"
        },
        "complete": {
            "files": 18960,
            "storage": "~43 GB", 
            "time": "24-48 hours",
            "memory": "80 GB",
            "description": "Process everything at once"
        }
    }
    
    if config_name in estimates:
        est = estimates[config_name]
        print(f"\nEstimated Requirements for '{config_name}' configuration:")
        print(f"  Description: {est['description']}")
        print(f"  Output files: {est['files']:,}")
        print(f"  Storage needed: {est['storage']}")
        print(f"  Processing time: {est['time']}")
        print(f"  Memory required: {est['memory']}")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Run the complete climate processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run test pipeline
  %(prog)s test
  
  # Run complete pipeline (staged approach)
  %(prog)s staged
  
  # Run complete pipeline (all at once)
  %(prog)s complete
  
  # Dry run to see what would be executed
  %(prog)s staged --dry-run
  
  # Run specific stage only
  %(prog)s staged --stage means_temperature_conus
"""
    )
    
    parser.add_argument(
        'config',
        choices=['test', 'staged', 'complete'],
        help='Pipeline configuration to run'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without executing'
    )
    
    parser.add_argument(
        '--stage',
        help='Run only a specific stage'
    )
    
    parser.add_argument(
        '--estimate',
        action='store_true',
        help='Show estimated requirements and exit'
    )
    
    args = parser.parse_args()
    
    # Map config names to files
    config_files = {
        'test': 'configs/pipeline_test_complete.yaml',
        'staged': 'configs/pipeline_complete_staged.yaml',
        'complete': 'configs/pipeline_complete_all_regions.yaml'
    }
    
    config_file = config_files[args.config]
    
    # Check if config file exists
    if not Path(config_file).exists():
        print(f"Error: Configuration file not found: {config_file}")
        return 1
    
    # Show estimates if requested
    if args.estimate:
        estimate_requirements(args.config)
        return 0
    
    # Show requirements
    estimate_requirements(args.config)
    
    # Confirm before running non-test pipelines
    if args.config != 'test' and not args.dry_run:
        print(f"\n{'='*80}")
        print("CONFIRMATION REQUIRED")
        print(f"{'='*80}")
        print(f"You are about to run the '{args.config}' pipeline.")
        print("This will process a large amount of data and may take many hours.")
        print("\nMake sure you have:")
        print("  - Sufficient disk space (check estimates above)")
        print("  - Stable power supply")
        print("  - No other heavy processes running")
        
        response = input("\nContinue? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Pipeline execution cancelled.")
            return 0
    
    # Run the pipeline
    return run_pipeline(config_file, args.dry_run, args.stage)


if __name__ == "__main__":
    sys.exit(main())