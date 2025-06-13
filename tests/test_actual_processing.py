#!/usr/bin/env python3
"""
Test script to directly test the climate processing logic.
"""

import sys
import time
from pathlib import Path
import logging

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from means.core.regional_climate_processor import RegionalProcessingConfig, RegionalClimateProcessor
from means.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_direct_processing():
    """Test the processing directly with minimal configuration."""
    
    print("🧪 Testing Direct Climate Processing")
    print("=" * 50)
    
    # Get configuration
    global_config = get_config()
    
    # Create a minimal configuration for testing
    config = RegionalProcessingConfig(
        region_key='CONUS',
        variables=['pr'],
        input_data_dir=global_config.paths.input_data_dir,
        output_base_dir=Path('output/test_processing'),
        max_cores=4,
        cores_per_variable=1,
        batch_size_years=1,  # Process one year at a time
        min_years_for_normal=5  # Reduce minimum years for testing
    )
    
    print(f"📁 Input directory: {config.input_data_dir}")
    print(f"📁 Output directory: {config.output_base_dir}")
    print(f"🔧 Configuration: {config.max_cores} cores, {config.cores_per_variable} per variable")
    
    # Create processor without rich progress for clearer logging
    processor = RegionalClimateProcessor(config, use_rich_progress=False)
    
    # Test processing a single target year
    print(f"\n🚀 Starting processing...")
    start_time = time.time()
    
    try:
        results = processor.process_all_variables()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n📊 Processing Results:")
        print(f"  ⏱️  Duration: {duration:.1f} seconds")
        print(f"  📋 Results: {results}")
        
        # Check if any files were created
        output_files = list(config.output_base_dir.rglob("*.nc"))
        print(f"  📁 Output files created: {len(output_files)}")
        for f in output_files[:5]:
            print(f"    {f}")
        
        return len(output_files) > 0
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_file_processing():
    """Test processing a single file to verify the core logic."""
    
    print("\n🔬 Testing Single File Processing")
    print("=" * 40)
    
    from means.utils.io_util import NorESM2FileHandler
    from means.core.regions import REGION_BOUNDS, extract_region
    import xarray as xr
    
    try:
        # Get a test file
        handler = NorESM2FileHandler('/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM')
        files = handler.get_files_for_period('pr', 'historical', 2010, 2010)
        
        if not files:
            print("❌ No files found for testing")
            return False
        
        test_file = files[0]
        print(f"📄 Testing file: {Path(test_file).name}")
        
        # Open and process the file
        ds = xr.open_dataset(test_file)
        print(f"📊 Dataset shape: {ds.pr.shape}")
        print(f"📅 Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
        
        # Extract CONUS region
        region_bounds = REGION_BOUNDS['CONUS']
        region_ds = extract_region(ds, region_bounds)
        print(f"🗺️  CONUS region shape: {region_ds.pr.shape}")
        
        # Test climatology calculation
        if 'dayofyear' in region_ds.pr.coords:
            daily_clim = region_ds.pr.groupby(region_ds.pr.dayofyear).mean(dim='time')
            print(f"📈 Daily climatology shape: {daily_clim.shape}")
            print(f"✅ Single file processing works!")
            return True
        else:
            print("❌ No dayofyear coordinate found")
            return False
        
    except Exception as e:
        print(f"❌ Single file processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    
    print("🚀 Climate Processing Diagnostic Test")
    print("=" * 60)
    
    # Test 1: Single file processing
    test1_success = test_single_file_processing()
    
    # Test 2: Direct processing
    test2_success = test_direct_processing()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"🔬 Single file test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"🧪 Direct processing: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print("\n🎉 Core processing logic is working!")
        print("💡 The issue might be with the progress tracking or file skipping logic.")
    else:
        print("\n⚠️  There are issues with the core processing logic.")
    
    return 0 if (test1_success and test2_success) else 1


if __name__ == "__main__":
    sys.exit(main()) 