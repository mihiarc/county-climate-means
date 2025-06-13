#!/usr/bin/env python3
"""Find exact location of hang"""

import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(processName)s - %(funcName)s - %(message)s'
)

def trace_process_single_variable():
    """Trace execution of process_single_variable with logging"""
    
    # Add tracing to key functions
    import county_climate.means.core.single_variable_processor as svp
    
    original_init = svp.SingleVariableProcessor.__init__
    original_process = svp.SingleVariableProcessor.process_all_years
    original_process_year = svp.SingleVariableProcessor._process_target_year
    
    def traced_init(self, *args, **kwargs):
        logging.info(f"TRACE: SingleVariableProcessor.__init__ called")
        return original_init(self, *args, **kwargs)
    
    def traced_process(self):
        logging.info(f"TRACE: process_all_years called")
        result = original_process(self)
        logging.info(f"TRACE: process_all_years completed")
        return result
        
    def traced_process_year(self, year):
        logging.info(f"TRACE: _process_target_year({year}) called")
        result = original_process_year(self, year)
        logging.info(f"TRACE: _process_target_year({year}) completed")
        return result
    
    # Monkey patch
    svp.SingleVariableProcessor.__init__ = traced_init
    svp.SingleVariableProcessor.process_all_years = traced_process
    svp.SingleVariableProcessor._process_target_year = traced_process_year
    
    # Now run the test
    from county_climate.means.core.single_variable_processor import process_single_variable
    
    args = {
        'variable': 'tas',
        'region_key': 'HI',
        'scenario': 'historical',
        'input_data_dir': '/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM',
        'output_base_dir': '/tmp/debug_trace',
        'min_years_for_normal': 25,
        'progress_queue': None
    }
    
    logging.info("Starting traced execution...")
    
    # Test 1: Direct call (should work)
    logging.info("=== TEST 1: Direct call ===")
    try:
        result = process_single_variable(args)
        logging.info(f"Direct call result: {result['status']}")
    except Exception as e:
        logging.error(f"Direct call failed: {e}")
    
    # Test 2: In subprocess (might hang)
    logging.info("\n=== TEST 2: In subprocess ===")
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process_single_variable, args)
        logging.info("Submitted to executor, waiting for result...")
        
        try:
            result = future.result(timeout=30)
            logging.info(f"Subprocess result: {result['status']}")
        except TimeoutError:
            logging.error("SUBPROCESS TIMED OUT - HANG DETECTED!")
            future.cancel()
        except Exception as e:
            logging.error(f"Subprocess failed: {e}")
            import traceback
            traceback.print_exc()

def test_simple_subprocess():
    """Test if basic subprocess works"""
    def simple_work():
        logging.info("Simple work started")
        time.sleep(0.5)
        logging.info("Simple work completed")
        return "done"
    
    logging.info("Testing simple subprocess...")
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(simple_work)
        result = future.result(timeout=2)
        logging.info(f"Simple subprocess result: {result}")

def test_with_imports():
    """Test subprocess with our imports"""
    def work_with_imports():
        logging.info("Importing modules in subprocess...")
        from county_climate.means.core.single_variable_processor import SingleVariableProcessor
        from county_climate.means.utils.io_util import NorESM2FileHandler
        logging.info("Imports successful")
        return "imports ok"
    
    logging.info("Testing subprocess with imports...")
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(work_with_imports)
        result = future.result(timeout=5)
        logging.info(f"Import test result: {result}")

if __name__ == '__main__':
    print("Debugging hang location...\n")
    
    # Run tests in order
    tests = [
        ("Simple subprocess", test_simple_subprocess),
        ("Subprocess with imports", test_with_imports),
        ("Traced execution", trace_process_single_variable),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        try:
            test_func()
            print(f"✓ {test_name} completed")
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()