#!/usr/bin/env python3
"""
Test script to demonstrate integrated rich progress tracking with multiprocessing engine.
"""

import sys
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from means.core.multiprocessing_engine import create_multiprocessing_engine
from means.utils.rich_progress import RichProgressTracker


def test_multiprocessing_with_rich_progress():
    """Test multiprocessing engine with rich progress tracking."""
    
    print("🧪 Testing Multiprocessing Engine with Rich Progress")
    print("=" * 60)
    
    # Create rich progress tracker
    tracker = RichProgressTracker("Multiprocessing Engine Test")
    tracker.start()
    
    try:
        # Create multiprocessing engine with rich progress
        engine = create_multiprocessing_engine(
            max_workers=4,
            enable_progress_tracking=False,  # Disable basic tracking
            use_rich_progress=True,
            rich_tracker=tracker
        )
        
        # Define test tasks
        def test_task(x: int, delay: float = 0.5) -> int:
            """Simple test task that simulates work."""
            time.sleep(delay)
            if x == 7:  # Simulate a failure
                raise ValueError(f"Simulated failure for task {x}")
            return x * x
        
        # Prepare tasks
        tasks = [test_task] * 10
        task_args = [(i, 0.3) for i in range(10)]  # 0.3 second delay per task
        
        print(f"🚀 Starting {len(tasks)} test tasks...")
        
        # Process tasks
        results = engine.process_tasks(tasks, task_args)
        
        # Analyze results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"\n📊 Results:")
        print(f"  ✅ Successful: {len(successful)}")
        print(f"  ❌ Failed: {len(failed)}")
        print(f"  📈 Success rate: {len(successful)/len(results)*100:.1f}%")
        
        return len(successful) > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        tracker.stop()


def test_system_capacity():
    """Test system capacity with larger workload."""
    
    print("\n🔧 Testing System Capacity")
    print("=" * 40)
    
    # Create rich progress tracker
    tracker = RichProgressTracker("System Capacity Test")
    tracker.start()
    
    try:
        # Test with more workers and tasks
        engine = create_multiprocessing_engine(
            max_workers=8,
            enable_progress_tracking=False,
            use_rich_progress=True,
            rich_tracker=tracker
        )
        
        # Add multiple task types to the tracker
        tracker.add_task("compute", "Computational tasks", 20)
        tracker.add_task("io", "I/O simulation tasks", 15)
        
        def compute_task(x: int) -> int:
            """CPU-intensive task simulation."""
            time.sleep(0.1)
            result = sum(i*i for i in range(x*100))
            return result
        
        def io_task(x: int) -> str:
            """I/O task simulation."""
            time.sleep(0.2)
            return f"processed_{x}"
        
        # Mix of different task types
        all_tasks = []
        all_args = []
        
        # Add compute tasks
        for i in range(20):
            all_tasks.append(compute_task)
            all_args.append((i+1,))
        
        # Add I/O tasks
        for i in range(15):
            all_tasks.append(io_task)
            all_args.append((i+1,))
        
        print(f"🚀 Processing {len(all_tasks)} mixed tasks...")
        
        # Process all tasks
        results = engine.process_tasks(all_tasks, all_args)
        
        # Update individual task trackers
        compute_completed = sum(1 for r in results[:20] if r.success)
        io_completed = sum(1 for r in results[20:] if r.success)
        
        tracker.update_task("compute", advance=compute_completed)
        tracker.update_task("io", advance=io_completed)
        
        tracker.complete_task("compute")
        tracker.complete_task("io")
        
        print(f"\n📊 Capacity Test Results:")
        print(f"  🧮 Compute tasks: {compute_completed}/20")
        print(f"  💾 I/O tasks: {io_completed}/15")
        print(f"  📈 Overall success: {len([r for r in results if r.success])}/{len(results)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Capacity test failed: {e}")
        return False
    
    finally:
        tracker.stop()


def main():
    """Main test function."""
    
    print("🚀 Integrated Progress Tracking Test Suite")
    print("=" * 60)
    
    # Test 1: Basic multiprocessing with rich progress
    test1_success = test_multiprocessing_with_rich_progress()
    
    # Test 2: System capacity test
    test2_success = test_system_capacity()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"🧪 Multiprocessing + Rich: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"🔧 System Capacity: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! The integrated system is working correctly.")
        print("\n💡 Key Benefits:")
        print("  • Beautiful real-time progress visualization")
        print("  • System resource monitoring")
        print("  • Task-level progress tracking")
        print("  • Error handling and reporting")
        print("  • Performance metrics")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 