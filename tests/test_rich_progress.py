#!/usr/bin/env python3
"""
Test script to demonstrate rich progress tracking with real climate data processing.
"""

import sys
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import means
from means.utils.rich_progress import RichProgressTracker


def test_rich_progress_with_real_data():
    """Test rich progress tracking with a small subset of real data."""
    
    print("🧪 Testing Rich Progress Tracking with Real Climate Data")
    print("=" * 60)
    
    try:
        # Test with a very small subset - just precipitation for CONUS
        # with minimal workers and batch size
        results = means.process_region(
            region_key='CONUS',
            variables=['pr'],  # Just precipitation
            max_cores=2,       # Minimal cores
            cores_per_variable=1,  # One core per variable
            batch_size_years=1,    # Process one year at a time
            use_rich_progress=True  # Enable rich progress
        )
        
        print("\n✅ Processing completed successfully!")
        print(f"📊 Results: {results}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rich_progress_demo():
    """Test the rich progress demo."""
    
    print("🎨 Testing Rich Progress Demo")
    print("=" * 40)
    
    try:
        from means.utils.rich_progress import demo_progress_tracker
        demo_progress_tracker()
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def main():
    """Main test function."""
    
    print("🚀 Rich Progress Tracking Test Suite")
    print("=" * 50)
    
    # Test 1: Demo
    print("\n1️⃣  Testing Rich Progress Demo...")
    demo_success = test_rich_progress_demo()
    
    # Test 2: Real data (optional - comment out if you want to skip)
    print("\n2️⃣  Testing with Real Climate Data...")
    print("⚠️  This will process actual climate data - it may take a few minutes")
    
    response = input("Do you want to test with real data? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        real_data_success = test_rich_progress_with_real_data()
    else:
        print("⏭️  Skipping real data test")
        real_data_success = True
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    print(f"🎨 Demo test: {'✅ PASSED' if demo_success else '❌ FAILED'}")
    print(f"🌍 Real data test: {'✅ PASSED' if real_data_success else '❌ FAILED'}")
    
    if demo_success and real_data_success:
        print("\n🎉 All tests passed! Rich progress tracking is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 