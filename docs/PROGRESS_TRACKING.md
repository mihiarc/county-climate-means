# Progress Tracking in Climate Means Processing

The climate means processing pipeline includes a sophisticated Rich-based progress tracking system that provides real-time visibility into processing status, system resource usage, and throughput metrics.

## Overview

Progress tracking is **enabled by default** and provides a beautiful, real-time dashboard showing:
- Individual processing tasks with progress bars
- Overall completion percentage  
- Processing throughput (files/second)
- System resource usage (CPU, memory)
- Failed tasks highlighted in red

## Features

### Multiprocessing-Safe Design
The progress tracking system is designed to work seamlessly with multiprocessing:
- Main process displays the Rich interface
- Worker processes communicate via a Queue
- No pickling issues with thread locks
- Graceful handling of worker failures

### Real-Time Dashboard
The dashboard updates 4 times per second and shows:
```
╭──────────────────────────────────────────────────────────────────────────────╮
│                  Climate Means Processing - 4 regions, 4 variables           │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────── Processing Tasks ──────────────────────────────╮
│ ⠋ Overall Progress        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━   45/121 0:05:23     │
│ ✓ TAS - CONUS - historical ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100/100 0:02:15     │
│ ⠙ PR - CONUS - historical  ━━━━━━━━━━━━━━━━━━━━━━━      67/100 0:01:45     │
│ ✗ TASMAX - AK - historical                               Failed              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────── System Stats ───────────╮
│  Elapsed:     5:23                 │
│  Files:       45/121               │
│  Failed:      1                    │
│  Throughput:  8.3 files/s          │
│  CPU:         78.2%                │
│  Memory:      45.3% (42.1GB/92.9GB)│
╰────────────────────────────────────╯
```

## Configuration

### Enable/Disable Progress Tracking
While enabled by default, you can control progress tracking in your configuration:

```yaml
stage_config:
  enable_rich_progress: true  # Default value
```

To disable progress tracking:
```yaml
stage_config:
  enable_rich_progress: false
```

### When to Disable
You might want to disable Rich progress tracking when:
- Running in CI/CD pipelines
- Capturing output to log files
- Running in non-interactive environments
- Debugging with verbose logging

## Implementation Details

### Progress Reporter
Worker processes use a `ProgressReporter` to send updates:
```python
progress_reporter.create_task(task_id, description, total)
progress_reporter.update_task(task_id, advance=1)
progress_reporter.complete_task(task_id)
progress_reporter.fail_task(task_id, error_msg)
```

### System Stats
The progress tracker monitors:
- CPU usage (updated every 100ms)
- Memory usage and availability
- Processing throughput
- Success/failure rates

### Graceful Degradation
If Rich cannot initialize (e.g., in non-TTY environments), the system falls back to standard logging without interrupting processing.

## Best Practices

1. **Keep It Enabled**: The default setting provides the best user experience
2. **Check Terminal Support**: Rich works best in modern terminals
3. **Monitor Resource Usage**: The system stats help identify bottlenecks
4. **Review Failed Tasks**: Failed tasks are highlighted for easy identification

## Troubleshooting

### Progress Not Showing
- Check if your terminal supports ANSI escape codes
- Ensure you're not redirecting output to a file
- Verify `enable_rich_progress: true` in configuration

### Performance Impact
The progress tracking system has minimal overhead:
- Updates are batched and rate-limited
- Queue communication is asynchronous
- System stats polling is lightweight

### Compatibility
Works with:
- Linux, macOS, Windows
- Python 3.9+
- Most modern terminal emulators
- Jupyter notebooks (with limitations)