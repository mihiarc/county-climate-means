#!/usr/bin/env python3
"""Test the progress tracking directly."""

import time
from county_climate.means.utils.mp_progress import MultiprocessingProgressTracker, ProgressReporter

# Test the progress tracker
print("Testing MultiprocessingProgressTracker...")

tracker = MultiprocessingProgressTracker("Test Progress Display")
queue = tracker.start()

# Create a reporter
reporter = ProgressReporter(queue)

# Add some tasks
reporter.create_task("task1", "Processing Task 1", 10)
reporter.create_task("task2", "Processing Task 2", 20)

# Update progress
for i in range(10):
    time.sleep(0.5)
    reporter.update_task("task1", advance=1, current_item=f"Item {i+1}")
    if i % 2 == 0:
        reporter.update_task("task2", advance=2, current_item=f"Batch {i//2 + 1}")

# Complete tasks
reporter.complete_task("task1")
reporter.complete_task("task2")

time.sleep(2)

# Stop tracker
tracker.stop()

print("\nProgress tracking test completed!")