#!/usr/bin/env python3
"""Debug script to test the pipeline with minimal configuration."""

import subprocess
import sys

# First, let's create a minimal test config
test_config = """
pipeline_id: "test_debug"
pipeline_name: "Debug Test Pipeline"
pipeline_version: "1.0.0"
environment: "development"

base_data_path: "/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
temp_data_path: "/tmp/climate_processing"
log_path: "/tmp/test_debug.log"

global_resource_limits:
  max_memory_gb: 16.0
  max_cpu_cores: 4
  max_processing_time_hours: 1.0

global_config:
  data_source: "NEX-GDDP-CMIP6"
  model: "NorESM2-LM"

stages:
  - stage_id: "test_means"
    stage_type: "means"
    stage_name: "Test Climate Means"
    package_name: "county_climate.means.integration"
    entry_point: "means_stage_handler"
    trigger_type: "manual"
    
    resource_limits:
      max_memory_gb: 8.0
      max_cpu_cores: 2
      max_processing_time_hours: 1.0
      priority: 10
    
    retry_attempts: 1
    retry_delay_seconds: 30
    
    stage_config:
      # Just process one variable, one region, one scenario for testing
      variables: ["tas"]
      regions: ["CONUS"]
      scenarios: ["historical"]
      
      multiprocessing_workers: 2
      batch_size_years: 5
      max_memory_per_worker_gb: 3.0
      
      enable_rich_progress: true
      
      output_base_path: "/tmp/test_means_output"
      
      # Limit years for quick test
      historical_years: [2000, 2005]
"""

# Write test config
with open('/tmp/test_debug_pipeline.yaml', 'w') as f:
    f.write(test_config)

print("Created test configuration file")

# Run the pipeline
cmd = [
    sys.executable,
    "main_orchestrated.py",
    "run",
    "--config", "/tmp/test_debug_pipeline.yaml"
]

print(f"Running command: {' '.join(cmd)}")
print("-" * 80)

# Run and capture output
try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
except Exception as e:
    print(f"Error running pipeline: {e}")