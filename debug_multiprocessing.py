#!/usr/bin/env python3
"""Debug multiprocessing hang issue"""

import sys
import time
import logging
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)

def timeout_handler(signum, frame):
    raise TimeoutError("Process timed out")

def test_basic_multiprocessing():
    """Test basic multiprocessing works"""
    print("\n=== Testing basic multiprocessing ===")
    
    def simple_worker(x):
        print(f"Worker processing {x}")
        time.sleep(0.1)
        return x * 2
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(simple_worker, i) for i in range(4)]
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Got result: {result}")
    
    print(f"All results: {results}")
    return True

def test_file_access():
    """Test if file access is causing issues"""
    print("\n=== Testing file access in multiprocessing ===")
    
    def file_worker(args):
        import xarray as xr
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
    
    # Test with a real climate data file
    test_file = Path('/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM/tas/historical/tas_day_NorESM2-LM_historical_r1i1p1f1_gn_1980.nc')
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return False
    
    args_list = [{'file_path': str(test_file)} for _ in range(2)]
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(file_worker, args) for args in args_list]
        for future in as_completed(futures):
            result = future.result()
            print(f"Result: {result}")
    
    return True

def test_progress_queue():
    """Test if progress queue is causing deadlock"""
    print("\n=== Testing progress queue ===")
    
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
    
    # Test without queue
    print("\nTest 1: Without progress queue")
    args_list = [{'id': i, 'progress_queue': None} for i in range(2)]
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(queue_worker, args) for args in args_list]
        for future in as_completed(futures):
            result = future.result()
            print(f"Result: {result}")
    
    # Test with queue
    print("\nTest 2: With progress queue (using Manager)")
    manager = mp.Manager()
    progress_queue = manager.Queue()
    
    args_list = [{'id': i, 'progress_queue': progress_queue} for i in range(2)]
    
    # Start queue reader in main process
    def read_queue():
        while True:
            try:
                msg = progress_queue.get(timeout=0.1)
                print(f"Main process received: {msg}")
            except:
                break
    
    import threading
    reader_thread = threading.Thread(target=read_queue)
    reader_thread.daemon = True
    reader_thread.start()
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(queue_worker, args) for args in args_list]
        for future in as_completed(futures):
            result = future.result()
            print(f"Result: {result}")
    
    time.sleep(0.5)  # Let reader thread finish
    return True

def test_actual_processor():
    """Test the actual SingleVariableProcessor"""
    print("\n=== Testing actual SingleVariableProcessor ===")
    
    # Import here to avoid issues
    from county_climate.means.core.single_variable_processor import process_single_variable
    
    # Limit processing to just 1 year
    test_args = {
        'variable': 'tas',
        'region_key': 'HI',
        'scenario': 'historical',
        'input_data_dir': '/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM',
        'output_base_dir': '/tmp/debug_test',
        'min_years_for_normal': 25,
        'progress_queue': None,  # No progress tracking
        '_test_years': [1980]  # Limit to one year
    }
    
    print(f"Processing with args: {test_args}")
    
    # Add timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout
    
    try:
        result = process_single_variable(test_args)
        print(f"Result: {result}")
        signal.alarm(0)  # Cancel timeout
        return True
    except TimeoutError:
        print("ERROR: Process timed out!")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Debugging multiprocessing hang issue...")
    
    tests = [
        ("Basic multiprocessing", test_basic_multiprocessing),
        ("File access", test_file_access),
        ("Progress queue", test_progress_queue),
        ("Actual processor", test_actual_processor),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print('='*60)
            success = test_func()
            print(f"\nResult: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()