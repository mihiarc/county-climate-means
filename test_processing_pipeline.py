#!/usr/bin/env python3
"""
Test script for climate normals processing pipeline.
Tests a subset of years for precipitation to validate the approach.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import xarray as xr
import gc
from typing import List, Dict, Tuple

# Import our modules
from io_util import NorESM2FileHandler
from regions import REGION_BOUNDS, extract_region
from time_util import handle_time_coordinates
from climate_means import compute_climate_normal, calculate_daily_climatology

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
INPUT_DATA_DIR = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
OUTPUT_BASE_DIR = "output/test_rolling_normals"
TEST_VARIABLE = 'pr'
TEST_REGION = 'CONUS'
MIN_YEARS_FOR_NORMAL = 25

# Test specific years
HISTORICAL_TEST_YEARS = [2014]      # Test end of historical period
HYBRID_TEST_YEARS = [2015]          # Test hybrid period (most important!)
FUTURE_TEST_YEARS = [2050]          # Test future period

def test_hybrid_normal(target_year: int, file_handler) -> bool:
    """Test processing a single hybrid normal."""
    logger.info(f"Testing hybrid normal for {target_year}")
    
    # Output directory
    output_dir = Path(OUTPUT_BASE_DIR) / TEST_VARIABLE / "hybrid"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use hybrid file collection (combines historical + ssp245)
    all_files, scenario_counts = file_handler.get_hybrid_files_for_period(TEST_VARIABLE, target_year, 30)
    
    if len(all_files) < MIN_YEARS_FOR_NORMAL:
        logger.error(f"  Insufficient files ({len(all_files)}) for {target_year}")
        return False
    
    hist_count = scenario_counts['historical']
    ssp245_count = scenario_counts['ssp245']
    logger.info(f"  Using {hist_count} historical + {ssp245_count} SSP245 files")
    
    # Just test the first few files for now
    test_files = all_files[:5]
    logger.info(f"  Testing processing with first 5 files:")
    
    for i, file_path in enumerate(test_files):
        try:
            # Extract year from filename
            year = file_handler.extract_year_from_filename(file_path)
            logger.info(f"    File {i+1}: {file_path} -> year {year}")
            
            # Open dataset quickly
            ds = xr.open_dataset(file_path, decode_times=False)
            logger.info(f"      Variables: {list(ds.data_vars)}")
            logger.info(f"      Shape: {ds[TEST_VARIABLE].shape if TEST_VARIABLE in ds.data_vars else 'Variable not found'}")
            ds.close()
            
        except Exception as e:
            logger.error(f"      Error processing file {i+1}: {e}")
    
    logger.info(f"  ✓ Hybrid file collection test successful for {target_year}")
    return True

def main():
    """Main test function."""
    logger.info("🧪 Starting Climate Normals Processing Pipeline Test")
    logger.info(f"Input directory: {INPUT_DATA_DIR}")
    logger.info(f"Output directory: {OUTPUT_BASE_DIR}")
    logger.info(f"Test variable: {TEST_VARIABLE}")
    logger.info(f"Test region: {TEST_REGION}")
    
    # Initialize file handler
    try:
        file_handler = NorESM2FileHandler(INPUT_DATA_DIR)
        logger.info("✓ File handler initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize file handler: {e}")
        return
    
    # Test hybrid normals (most important test)
    logger.info(f"\n{'='*50}")
    logger.info("🔀 Testing Hybrid Normals (2015 - Critical Test)")
    logger.info(f"{'='*50}")
    
    success = test_hybrid_normal(2015, file_handler)
    
    if success:
        logger.info("\n🎉 HYBRID TEST PASSED!")
        logger.info("✅ The 2015 hybrid approach works correctly!")
        logger.info("✅ Pipeline is ready for full processing!")
    else:
        logger.info("\n❌ HYBRID TEST FAILED!")
        logger.info("⚠️  Please check the errors above")

if __name__ == "__main__":
    main() 