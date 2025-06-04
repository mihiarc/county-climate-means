#!/usr/bin/env python3
"""
Progress Monitor for Climate Normals Processing Pipeline

Real-time monitoring of the multiprocessing pipeline with live updates.
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
import argparse

class ProgressMonitor:
    """Real-time progress monitoring for the climate normals pipeline."""
    
    def __init__(self, status_file="processing_progress.json", progress_log="processing_progress.log"):
        self.status_file = status_file
        self.progress_log = progress_log
    
    def load_status(self):
        """Load current status from JSON file."""
        try:
            with open(self.status_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"status": "not_started", "message": "Progress file not found"}
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid progress file"}
    
    def display_status(self, status_data):
        """Display formatted status information."""
        print("\033[2J\033[H")  # Clear screen and move to top
        print("=" * 80)
        print("🌡️  CLIMATE NORMALS PROCESSING - PROGRESS MONITOR")
        print("=" * 80)
        
        if status_data["status"] == "not_started":
            print("❌ Processing not started or progress file not found")
            return
        
        # Basic info
        print(f"📅 Started: {status_data.get('pipeline_start', 'Unknown')}")
        print(f"🔄 Status: {status_data.get('status', 'Unknown').upper()}")
        print(f"⏰ Last Update: {status_data.get('last_update', 'Unknown')}")
        print()
        
        # Overall progress
        total_completed = status_data.get('total_files_completed', 0)
        total_target = status_data.get('total_files_target', 547)
        completion_pct = status_data.get('completion_percentage', 0)
        
        print("📊 OVERALL PROGRESS")
        print("-" * 40)
        print(f"Files: {total_completed:3d}/{total_target} ({completion_pct:5.1f}%)")
        
        # Progress bar
        bar_width = 50
        filled = int(bar_width * completion_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"[{bar}]")
        print()
        
        # Variable progress
        print("🔢 PROGRESS BY VARIABLE")
        print("-" * 60)
        print(f"{'Variable':<8} {'Completed':<10} {'Target':<8} {'Progress':<10} {'Status':<12}")
        print("-" * 60)
        
        variables = status_data.get('variables', {})
        for var, info in variables.items():
            completed = info.get('completed', 0)
            target = info.get('target', 0)
            progress = (completed / target * 100) if target > 0 else 0
            status = info.get('status', 'pending')
            
            status_emoji = {
                'pending': '⏳',
                'processing': '🔄',
                'completed': '✅',
                'failed': '❌'
            }.get(status, '❓')
            
            print(f"{var.upper():<8} {completed:<10} {target:<8} {progress:>6.1f}%    {status_emoji} {status}")
        print()
        
        # Period progress
        print("📅 PROGRESS BY PERIOD")
        print("-" * 50)
        print(f"{'Period':<12} {'Completed':<10} {'Target':<8} {'Progress':<10}")
        print("-" * 50)
        
        periods = status_data.get('periods', {})
        for period, info in periods.items():
            completed = info.get('completed', 0)
            target = info.get('target', 0)
            progress = (completed / target * 100) if target > 0 else 0
            
            print(f"{period.title():<12} {completed:<10} {target:<8} {progress:>6.1f}%")
        print()
        
        # Performance metrics
        performance = status_data.get('performance', {})
        print("⚡ PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Rate: {performance.get('files_per_minute', 0):.1f} files/minute")
        print(f"Memory: {performance.get('memory_usage_gb', 0):.1f} GB")
        print(f"CPU: {performance.get('cpu_usage_percent', 0):.1f}%")
        
        eta = status_data.get('estimated_time_remaining')
        if eta:
            print(f"ETA: {eta}")
        print()
        
        # Current work
        current_work = status_data.get('current_work', {})
        active_batches = current_work.get('active_batches', [])
        recent_completed = current_work.get('recently_completed', [])
        
        if active_batches:
            print("🔄 ACTIVE BATCHES")
            print("-" * 40)
            for batch in active_batches[:5]:  # Show up to 5 active batches
                worker_id = batch.get('worker_id', '?')
                variable = batch.get('variable', '?')
                period = batch.get('period', '?')
                years = batch.get('years', [])
                year_range = f"{min(years)}-{max(years)}" if years else "?"
                print(f"Worker {worker_id}: {variable.upper()} {period} ({year_range})")
            print()
        
        if recent_completed:
            print("✅ RECENTLY COMPLETED")
            print("-" * 40)
            for completion in recent_completed[-3:]:  # Show last 3 completions
                worker_id = completion.get('worker_id', '?')
                variable = completion.get('variable', '?')
                years = completion.get('years', [])
                year_range = f"{min(years)}-{max(years)}" if years else "?"
                print(f"Worker {worker_id}: {variable.upper()} ({year_range})")
            print()
        
        print("Press Ctrl+C to exit monitor")
    
    def monitor_loop(self, refresh_interval=5):
        """Main monitoring loop with periodic updates."""
        try:
            while True:
                status_data = self.load_status()
                self.display_status(status_data)
                
                # Check if completed
                if status_data.get("status") == "completed":
                    print("\n🎉 Processing completed!")
                    break
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitor stopped by user")
    
    def show_summary(self):
        """Show a quick summary without continuous monitoring."""
        status_data = self.load_status()
        self.display_status(status_data)
    
    def tail_progress_log(self, lines=20):
        """Show recent progress log entries."""
        try:
            with open(self.progress_log, 'r') as f:
                log_lines = f.readlines()
                recent_lines = log_lines[-lines:] if len(log_lines) > lines else log_lines
                
            print("📈 RECENT PROGRESS LOG ENTRIES")
            print("-" * 60)
            for line in recent_lines:
                print(line.strip())
                
        except FileNotFoundError:
            print("❌ Progress log file not found")

def main():
    parser = argparse.ArgumentParser(description="Monitor climate normals processing progress")
    parser.add_argument("--status-file", default="processing_progress.json",
                       help="Path to status JSON file")
    parser.add_argument("--progress-log", default="processing_progress.log", 
                       help="Path to progress log file")
    parser.add_argument("--refresh", type=int, default=5,
                       help="Refresh interval in seconds")
    parser.add_argument("--summary", action="store_true",
                       help="Show summary once and exit")
    parser.add_argument("--log", type=int, metavar="LINES",
                       help="Show recent log entries and exit")
    
    args = parser.parse_args()
    
    monitor = ProgressMonitor(args.status_file, args.progress_log)
    
    if args.summary:
        monitor.show_summary()
    elif args.log:
        monitor.tail_progress_log(args.log)
    else:
        monitor.monitor_loop(args.refresh)

if __name__ == "__main__":
    main() 