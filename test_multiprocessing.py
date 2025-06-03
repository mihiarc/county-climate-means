#!/usr/bin/env python3
"""
Quick test of multiprocessing climate data processing.
"""

import logging
import time
from pathlib import Path
from climate_multiprocessing import ClimateMultiprocessor, process_single_file_worker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_speedup_test():
    """Simple test to measure multiprocessing speedup."""
    
    data_directory = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
    
    if not Path(data_directory).exists():
        logger.error(f"Data directory not found: {data_directory}")
        return
    
    # Initialize processor
    processor = ClimateMultiprocessor(data_directory)
    
    # Get 6 test files
    test_files = processor.file_handler.get_files_for_period('pr', 'historical', 2012, 2014)[:6]
    
    if not test_files:
        logger.error("No test files found!")
        return
    
    logger.info(f"Testing with {len(test_files)} files")
    logger.info(f"System: 56 CPUs, auto-selected {processor.config.max_workers} workers")
    
    # === Test 1: Sequential Processing ===
    logger.info("\n=== Sequential Processing Test ===")
    start_time = time.time()
    
    sequential_results = []
    for i, file_path in enumerate(test_files):
        logger.info(f"Processing file {i+1}/{len(test_files)}: {Path(file_path).name}")
        result_data, exec_time, success = process_single_file_worker(file_path, 'pr', 'CONUS')
        if success:
            sequential_results.append((result_data, exec_time))
        
    sequential_total = time.time() - start_time
    sequential_avg = sequential_total / len(test_files)
    
    logger.info(f"Sequential results:")
    logger.info(f"  Total time: {sequential_total:.1f}s")
    logger.info(f"  Average per file: {sequential_avg:.1f}s")
    logger.info(f"  Successful files: {len(sequential_results)}/{len(test_files)}")
    
    # === Test 2: Multiprocessing ===
    logger.info(f"\n=== Multiprocessing Test ({processor.config.max_workers} workers) ===")
    
    # Use a unique output file
    import uuid
    output_path = f"output/multiprocessing_test/test_{uuid.uuid4().hex[:8]}.nc"
    
    result = processor.process_files_parallel(test_files, output_path, 'pr', 'CONUS')
    
    # Calculate speedup
    mp_total = result['statistics']['total_time_minutes'] * 60
    speedup = sequential_total / mp_total if mp_total > 0 else 0
    
    logger.info(f"\n=== Speedup Comparison ===")
    logger.info(f"Sequential: {sequential_total:.1f}s ({sequential_avg:.1f}s per file)")
    logger.info(f"Multiprocessing: {mp_total:.1f}s ({result['statistics']['avg_time_per_file']:.1f}s per file)")
    logger.info(f"Speedup: {speedup:.1f}x")
    logger.info(f"Efficiency: {speedup / processor.config.max_workers * 100:.1f}%")
    
    # Clean up test file
    Path(output_path).unlink(missing_ok=True)
    
    return {
        'sequential_time': sequential_total,
        'multiprocessing_time': mp_total,
        'speedup': speedup,
        'workers': processor.config.max_workers,
        'files_processed': len(test_files)
    }


def test_real_processing():
    """Test real climate processing with multiprocessing."""
    
    data_directory = "/media/mihiarc/RPA1TB/data/NorESM2-LM"
    processor = ClimateMultiprocessor(data_directory)
    
    logger.info("\n=== Real Processing Test ===")
    logger.info("Processing 3 years of historical precipitation (2012-2014)")
    
    start_time = time.time()
    output_path = processor.process_historical_precipitation_parallel(2012, 2014)
    total_time = time.time() - start_time
    
    logger.info(f"Real processing completed in {total_time:.1f}s")
    logger.info(f"Output saved to: {output_path}")
    
    # Check output file
    if Path(output_path).exists():
        file_size = Path(output_path).stat().st_size / 1e6  # MB
        logger.info(f"Output file size: {file_size:.1f} MB")
        
        # Quick validation
        import xarray as xr
        ds = xr.open_dataset(output_path)
        logger.info(f"Output shape: {dict(ds.sizes)}")
        logger.info(f"Years processed: {ds.attrs.get('years_processed', 'unknown')}")
        ds.close()
    
    return output_path


if __name__ == "__main__":
    # Run quick speedup test
    speedup_results = quick_speedup_test()
    
    if speedup_results:
        print(f"\n🚀 MULTIPROCESSING RESULTS 🚀")
        print(f"Files: {speedup_results['files_processed']}")
        print(f"Workers: {speedup_results['workers']}")
        print(f"Speedup: {speedup_results['speedup']:.1f}x")
        print(f"Sequential: {speedup_results['sequential_time']:.1f}s")
        print(f"Multiprocessing: {speedup_results['multiprocessing_time']:.1f}s")
        
        # Run real processing test if speedup looks good
        if speedup_results['speedup'] > 1.5:
            print(f"\n🎯 Speedup looks good! Running real processing test...")
            test_real_processing()
        else:
            print(f"⚠️  Speedup is less than 1.5x, multiprocessing may not be worthwhile") 