#!/usr/bin/env python3
"""
Rich-based Progress Tracking for Climate Data Processing

Enhanced progress visualization using the rich library for beautiful terminal output.
Provides real-time progress bars, status displays, and performance monitoring.
"""

import time
import threading
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import psutil

from rich.console import Console
from rich.progress import (
    Progress, 
    TaskID, 
    BarColumn, 
    TextColumn, 
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    ProgressColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich import box
from rich.columns import Columns
from rich.tree import Tree


class ThroughputColumn(ProgressColumn):
    """Custom column showing processing throughput."""
    
    def render(self, task):
        if task.speed is None:
            return Text("--", style="progress.data.speed")
        return Text(f"{task.speed:.1f} files/s", style="progress.data.speed")


class MemoryColumn(ProgressColumn):
    """Custom column showing memory usage."""
    
    def render(self, task):
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            return Text(f"RAM: {memory_percent:.1f}%", style="progress.data.speed")
        except:
            return Text("RAM: --", style="progress.data.speed")


@dataclass
class ProcessingTask:
    """Represents a processing task with rich progress tracking."""
    name: str
    total: int
    completed: int = 0
    failed: int = 0
    start_time: Optional[datetime] = None
    task_id: Optional[TaskID] = None
    status: str = "pending"  # pending, running, completed, failed
    current_item: str = ""
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.completed + self.failed == 0:
            return 100.0
        return (self.completed / (self.completed + self.failed)) * 100
    
    @property
    def throughput(self) -> float:
        """Calculate processing throughput in items per second."""
        if not self.start_time or self.completed == 0:
            return 0.0
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return self.completed / elapsed if elapsed > 0 else 0.0


class RichProgressTracker:
    """
    Enhanced progress tracking using Rich library for beautiful terminal output.
    
    Features:
    - Multiple progress bars for different processing stages
    - Real-time system monitoring (CPU, memory)
    - Processing statistics and throughput
    - Hierarchical task organization
    - Live updating display
    """
    
    def __init__(self, 
                 title: str = "Climate Data Processing",
                 show_system_stats: bool = True,
                 update_interval: float = 0.5,
                 save_progress: bool = True):
        
        self.title = title
        self.show_system_stats = show_system_stats
        self.update_interval = update_interval
        self.save_progress = save_progress
        
        # Initialize rich components
        self.console = Console()
        self.tasks: Dict[str, ProcessingTask] = {}
        self.start_time = datetime.now()
        
        # Progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            ThroughputColumn(),
            TextColumn("•"),
            MemoryColumn(),
            console=self.console,
            expand=True
        )
        
        # Layout for complex display
        self.layout = Layout()
        self.setup_layout()
        
        # Live display
        self.live = None
        self.running = False
        self.update_thread = None
        
        # Progress file for persistence
        self.progress_file = Path("rich_progress.json")
    
    def setup_layout(self):
        """Setup the rich layout structure."""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=8)
        )
        
        self.layout["main"].split_row(
            Layout(name="progress", ratio=2),
            Layout(name="stats", ratio=1)
        )
    
    def start(self):
        """Start the rich progress display."""
        self.running = True
        self.live = Live(
            self.layout, 
            console=self.console, 
            refresh_per_second=1/self.update_interval,
            screen=True
        )
        self.live.start()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.console.print(f"🚀 Started {self.title}", style="bold green")
    
    def stop(self):
        """Stop the rich progress display."""
        self.running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        
        if self.live:
            self.live.stop()
        
        self._save_final_summary()
        self.console.print(f"✅ Completed {self.title}", style="bold green")
    
    def add_task(self, 
                 name: str, 
                 description: str, 
                 total: int,
                 parent: Optional[str] = None) -> str:
        """Add a new processing task."""
        
        task = ProcessingTask(
            name=name,
            total=total,
            start_time=datetime.now()
        )
        
        # Add to progress display
        task.task_id = self.progress.add_task(
            description=description,
            total=total,
            start=True
        )
        
        self.tasks[name] = task
        return name
    
    def update_task(self, 
                   name: str, 
                   advance: int = 1, 
                   current_item: str = "",
                   failed: bool = False):
        """Update task progress."""
        
        if name not in self.tasks:
            return
        
        task = self.tasks[name]
        
        if failed:
            task.failed += advance
        else:
            task.completed += advance
        
        task.current_item = current_item
        task.status = "running"
        
        # Update rich progress
        if task.task_id is not None:
            self.progress.update(
                task.task_id,
                advance=advance,
                description=f"{task.name}: {current_item}" if current_item else task.name
            )
    
    def complete_task(self, name: str, status: str = "completed"):
        """Mark a task as completed."""
        if name not in self.tasks:
            return
        
        task = self.tasks[name]
        task.status = status
        
        if task.task_id is not None:
            self.progress.update(
                task.task_id,
                completed=task.total,
                description=f"✅ {task.name} - {status}"
            )
    
    def _update_loop(self):
        """Background thread for updating the display."""
        while self.running:
            try:
                self._update_display()
                time.sleep(self.update_interval)
            except Exception as e:
                # Don't let display errors crash the processing
                pass
    
    def _update_display(self):
        """Update the rich display layout."""
        # Header
        elapsed = datetime.now() - self.start_time
        header_text = Text.assemble(
            ("🌍 ", "bold blue"),
            (self.title, "bold white"),
            (" • ", "dim"),
            (f"Elapsed: {str(elapsed).split('.')[0]}", "cyan"),
            (" • ", "dim"),
            (f"Started: {self.start_time.strftime('%H:%M:%S')}", "dim")
        )
        self.layout["header"].update(
            Panel(Align.center(header_text), box=box.ROUNDED)
        )
        
        # Progress section
        self.layout["progress"].update(self.progress)
        
        # Statistics section
        stats_table = self._create_stats_table()
        self.layout["stats"].update(
            Panel(stats_table, title="📊 Statistics", box=box.ROUNDED)
        )
        
        # Footer with system info
        if self.show_system_stats:
            footer_content = self._create_system_stats()
            self.layout["footer"].update(
                Panel(footer_content, title="💻 System Status", box=box.ROUNDED)
            )
        
        # Save progress periodically
        if self.save_progress:
            self._save_progress()
    
    def _create_stats_table(self) -> Table:
        """Create statistics table."""
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Task", style="cyan")
        table.add_column("Progress", justify="right")
        table.add_column("Success Rate", justify="right")
        table.add_column("Throughput", justify="right")
        
        for task in self.tasks.values():
            if task.status == "pending":
                continue
                
            progress_text = f"{task.completed + task.failed}/{task.total}"
            success_rate = f"{task.success_rate:.1f}%"
            throughput = f"{task.throughput:.1f}/s" if task.throughput > 0 else "--"
            
            # Color code based on success rate
            if task.success_rate >= 95:
                success_style = "green"
            elif task.success_rate >= 80:
                success_style = "yellow"
            else:
                success_style = "red"
            
            table.add_row(
                task.name,
                progress_text,
                Text(success_rate, style=success_style),
                throughput
            )
        
        return table
    
    def _create_system_stats(self) -> Table:
        """Create system statistics display."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # CPU info
            cpu_style = "green" if cpu_percent < 70 else "yellow" if cpu_percent < 90 else "red"
            table.add_row("CPU Usage", Text(f"{cpu_percent:.1f}%", style=cpu_style))
            
            # Memory info
            memory_style = "green" if memory.percent < 70 else "yellow" if memory.percent < 90 else "red"
            table.add_row("Memory", Text(f"{memory.percent:.1f}% ({memory.used/1024**3:.1f}GB)", style=memory_style))
            
            # Disk info
            disk_style = "green" if disk.percent < 80 else "yellow" if disk.percent < 95 else "red"
            table.add_row("Disk", Text(f"{disk.percent:.1f}%", style=disk_style))
            
            # Process count
            active_processes = len([t for t in self.tasks.values() if t.status == "running"])
            table.add_row("Active Tasks", str(active_processes))
            
        except Exception:
            table.add_row("System Stats", "Unavailable")
        
        return table
    
    def _save_progress(self):
        """Save current progress to file."""
        if not self.save_progress:
            return
        
        try:
            progress_data = {
                "title": self.title,
                "start_time": self.start_time.isoformat(),
                "current_time": datetime.now().isoformat(),
                "tasks": {}
            }
            
            for name, task in self.tasks.items():
                progress_data["tasks"][name] = {
                    "total": task.total,
                    "completed": task.completed,
                    "failed": task.failed,
                    "status": task.status,
                    "success_rate": task.success_rate,
                    "throughput": task.throughput,
                    "current_item": task.current_item
                }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception:
            # Don't let progress saving errors crash the processing
            pass
    
    def _save_final_summary(self):
        """Save final processing summary."""
        try:
            total_time = datetime.now() - self.start_time
            
            summary = {
                "title": self.title,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration": str(total_time),
                "tasks": {},
                "overall_stats": {
                    "total_items": sum(t.total for t in self.tasks.values()),
                    "completed_items": sum(t.completed for t in self.tasks.values()),
                    "failed_items": sum(t.failed for t in self.tasks.values()),
                    "overall_success_rate": 0.0,
                    "average_throughput": 0.0
                }
            }
            
            # Calculate overall stats
            total_processed = summary["overall_stats"]["completed_items"] + summary["overall_stats"]["failed_items"]
            if total_processed > 0:
                summary["overall_stats"]["overall_success_rate"] = (
                    summary["overall_stats"]["completed_items"] / total_processed * 100
                )
            
            if total_time.total_seconds() > 0:
                summary["overall_stats"]["average_throughput"] = (
                    summary["overall_stats"]["completed_items"] / total_time.total_seconds()
                )
            
            # Task details
            for name, task in self.tasks.items():
                summary["tasks"][name] = {
                    "total": task.total,
                    "completed": task.completed,
                    "failed": task.failed,
                    "status": task.status,
                    "success_rate": task.success_rate,
                    "throughput": task.throughput
                }
            
            # Save summary
            summary_file = Path(f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Print final summary to console
            self.console.print("\n" + "="*60)
            self.console.print("📊 PROCESSING SUMMARY", style="bold cyan", justify="center")
            self.console.print("="*60)
            self.console.print(f"⏱️  Total time: {total_time}")
            self.console.print(f"📁 Total items: {summary['overall_stats']['total_items']}")
            self.console.print(f"✅ Completed: {summary['overall_stats']['completed_items']}")
            self.console.print(f"❌ Failed: {summary['overall_stats']['failed_items']}")
            self.console.print(f"📈 Success rate: {summary['overall_stats']['overall_success_rate']:.1f}%")
            self.console.print(f"🚀 Throughput: {summary['overall_stats']['average_throughput']:.2f} items/s")
            self.console.print(f"💾 Summary saved to: {summary_file}")
            
        except Exception as e:
            self.console.print(f"⚠️  Could not save summary: {e}", style="yellow")


# Convenience functions for easy integration

def create_rich_progress_tracker(title: str = "Climate Processing", **kwargs) -> RichProgressTracker:
    """Create and return a configured rich progress tracker."""
    return RichProgressTracker(title=title, **kwargs)


def demo_progress_tracker():
    """Demonstrate the rich progress tracker."""
    import random
    
    tracker = RichProgressTracker("Demo Climate Processing")
    tracker.start()
    
    try:
        # Add some demo tasks
        tracker.add_task("precipitation", "Processing precipitation data", 50)
        tracker.add_task("temperature", "Processing temperature data", 30)
        tracker.add_task("analysis", "Running analysis", 20)
        
        # Simulate processing
        for i in range(100):
            time.sleep(0.1)
            
            # Randomly update tasks
            if random.random() < 0.3:
                tracker.update_task("precipitation", current_item=f"file_{i}.nc")
            if random.random() < 0.2:
                tracker.update_task("temperature", current_item=f"temp_{i}.nc")
            if random.random() < 0.1:
                tracker.update_task("analysis", current_item=f"analysis_{i}")
        
        # Complete tasks
        tracker.complete_task("precipitation")
        tracker.complete_task("temperature")
        tracker.complete_task("analysis")
        
        time.sleep(2)  # Show completed state
        
    finally:
        tracker.stop()


if __name__ == "__main__":
    demo_progress_tracker()