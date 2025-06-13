#!/usr/bin/env python3
"""Debug multiprocessing hang issue - fixed version"""

import sys
import time
import logging
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
import xarray as xr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)

# Define worker functions at module level for pickling
def simple_worker(x):
    print(f"Worker processing {x}")
    time.sleep(0.1)
    return x * 2

def file_worker(args):
    file_path = args['file_path']
    print(f"Worker opening file: {file_path}")
    
    try:
        with xr.open_dataset(file_path) as ds:
            shape = ds.dims
            print(f"File opened successfully, shape: {shape}")
            return {'status': 'success', 'shape': dict(shape)}
    except Exception as e:
        print(f"Error opening file: {e}")
        return {'status': 'error', 'error': str(e)}

def queue_worker(args):
    progress_queue = args.get('progress_queue')
    worker_id = args['id']
    
    print(f"Worker {worker_id} starting")
    
    # Simulate some work
    for i in range(3):
        time.sleep(0.1)
        if progress_queue:
            try:
                progress_queue.put({
                    'type': 'update',
                    'worker': worker_id,
                    'progress': i + 1
                })
                print(f"Worker {worker_id} sent progress: {i+1}")
            except Exception as e:
                print(f"Worker {worker_id} queue error: {e}")
    
    print(f"Worker {worker_id} completed")
    return {'status': 'success', 'worker': worker_id}

def test_single_variable_processor_limited():
    """Test actual processor with limited scope"""
    from county_climate.means.core.single_variable_processor import SingleVariableProcessor
    
    # Override method to limit years
    original_get_target_years = SingleVariableProcessor._get_target_years
    
    def limited_get_target_years(self):
        return [1980]  # Just one year
    
    SingleVariableProcessor._get_target_years = limited_get_target_years
    
    try:
        processor = SingleVariableProcessor(
            variable='tas',
            region_key='HI',
            scenario='historical',
            input_data_dir=Path('/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM'),
            output_base_dir=Path('/tmp/debug_hang'),
            min_years_for_normal=25
        )
        
        print("Processing single year...")
        results = processor.process_all_years()
        print(f"Results: {results['status']}")
        return results
    finally:
        # Restore original method
        SingleVariableProcessor._get_target_years = original_get_target_years

def test_actual_multiprocessing():
    """Test the actual multiprocessing setup"""
    print("\n=== Testing actual multiprocessing with limited data ===")
    
    from county_climate.means.core.single_variable_processor import process_single_variable
    
    # Create test args for just one year
    args = {
        'variable': 'tas',
        'region_key': 'HI',
        'scenario': 'historical',
        'input_data_dir': '/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM',
        'output_base_dir': '/tmp/debug_mp',
        'min_years_for_normal': 25,
        'progress_queue': None
    }
    
    print("Testing with single process first...")
    result = process_single_variable(args)
    print(f"Single process result: {result['status']}")
    
    print("\nNow testing with ProcessPoolExecutor...")
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process_single_variable, args)
        try:
            result = future.result(timeout=60)
            print(f"Multiprocess result: {result['status']}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Run focused tests"""
    print("Debugging multiprocessing hang issue...\n")
    
    # Test 1: Basic multiprocessing
    print("Test 1: Basic multiprocessing")
    print("-" * 40)
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(simple_worker, i) for i in range(4)]
        for future in as_completed(futures):
            result = future.result()
            print(f"Got result: {result}")
    print("✓ Basic multiprocessing works\n")
    
    # Test 2: File access
    print("Test 2: File access in multiprocessing")
    print("-" * 40)
    test_file = Path('/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM/tas/historical/tas_day_NorESM2-LM_historical_r1i1p1f1_gn_1980.nc')
    if test_file.exists():
        args = {'file_path': str(test_file)}
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(file_worker, args)
            result = future.result()
            print(f"File access result: {result['status']}")
        print("✓ File access works\n")
    
    # Test 3: Limited processor test
    print("Test 3: Single variable processor (limited)")
    print("-" * 40)
    try:
        test_single_variable_processor_limited()
        print("✓ Limited processor works\n")
    except Exception as e:
        print(f"✗ Limited processor failed: {e}\n")
    
    # Test 4: Actual multiprocessing
    print("Test 4: Actual multiprocessing")
    print("-" * 40)
    test_actual_multiprocessing()

if __name__ == '__main__':
    # Run with timeout
    import signal
    
    def timeout_handler(signum, frame):
        print("\n\n!!! OVERALL TEST TIMEOUT - Process is hanging !!!")
        sys.exit(1)
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(180)  # 3 minute timeout for all tests
    
    try:
        main()
        signal.alarm(0)  # Cancel timeout
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)