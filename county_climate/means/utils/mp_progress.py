"""
Enhanced multiprocessing-safe progress tracking with Rich.

This enhanced version fixes the progress tracking issues:
1. Supports updating task totals after creation
2. Properly tracks file counts in statistics
3. Updates progress in real-time as files are processed
"""

import time
import multiprocessing as mp
from multiprocessing import Queue, Process
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import psutil
import threading

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
)
from rich.table import Table
from rich.text import Text


@dataclass
class ProgressUpdate:
    """Progress update message for inter-process communication."""
    task_id: str
    update_type: str  # 'create', 'update', 'complete', 'fail', 'update_total'
    description: Optional[str] = None
    total: Optional[int] = None
    advance: Optional[int] = None
    completed: Optional[int] = None
    visible: Optional[bool] = None
    current_item: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None


class MultiprocessingProgressTracker:
    """
    Enhanced Rich-based progress tracker that works with multiprocessing.
    """
    
    def __init__(self, title: str = "Climate Data Processing"):
        self.title = title
        self.console = Console()
        self.queue: Optional[Queue] = None
        self.process: Optional[Process] = None
        self._manager = mp.Manager()
        self._shared_stats = self._manager.dict()
        
    def start(self) -> Queue:
        """Start the progress tracker and return the queue for updates."""
        self.queue = mp.Queue()
        
        # Initialize shared statistics
        self._shared_stats.update({
            'start_time': time.time(),
            'total_files': 0,
            'completed_files': 0,
            'failed_files': 0,
            'tasks': {}
        })
        
        self.process = mp.Process(
            target=self._run_progress_display, 
            args=(self.queue, self._shared_stats, self.title)
        )
        self.process.start()
        return self.queue
        
    def stop(self):
        """Stop the progress tracker."""
        if self.queue:
            self.queue.put(ProgressUpdate("__STOP__", "stop"))
            if self.process:
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.terminate()
                    
    def add_task(self, name: str, description: str, total: int):
        """Add a task from the main process."""
        if self.queue:
            self.queue.put(ProgressUpdate(
                task_id=name,
                update_type="create",
                description=description,
                total=total
            ))
            
    def update_task_total(self, name: str, total: int):
        """Update the total for a task."""
        if self.queue:
            self.queue.put(ProgressUpdate(
                task_id=name,
                update_type="update_total",
                total=total
            ))
            
    def complete_task(self, name: str, status: str = "completed"):
        """Mark a task as complete."""
        if self.queue:
            self.queue.put(ProgressUpdate(
                task_id=name,
                update_type="complete",
                description=status
            ))
                    
    @staticmethod
    def _run_progress_display(queue: Queue, shared_stats: dict, title: str):
        """Run the Rich progress display in a separate process."""
        console = Console()
        
        # Create progress bars with custom columns
        job_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[dim]{task.fields[current_item]}"),
            expand=True,
        )
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", size=10),
            Layout(name="footer", size=8),
        )
        
        # Header
        header_text = Text(title, style="bold white on blue", justify="center")
        layout["header"].update(Panel(header_text, style="blue"))
        
        # Task tracking
        tasks = {}
        task_stats = {}
        running = True
        
        with Live(layout, console=console, refresh_per_second=4) as live:
            while running:
                try:
                    # Process queue updates with timeout
                    update = queue.get(timeout=0.25)
                    
                    if update.task_id == "__STOP__":
                        running = False
                        break
                        
                    elif update.update_type == "create":
                        # Create new task
                        task_id = job_progress.add_task(
                            update.description,
                            total=update.total,
                            current_item=""
                        )
                        tasks[update.task_id] = task_id
                        task_stats[update.task_id] = {
                            'total': update.total,
                            'completed': 0,
                            'failed': 0
                        }
                        
                    elif update.update_type == "update_total":
                        # Update task total
                        if update.task_id in tasks:
                            job_progress.update(
                                tasks[update.task_id],
                                total=update.total
                            )
                            task_stats[update.task_id]['total'] = update.total
                            
                            # Update shared stats with new totals
                            shared_stats['total_files'] = sum(
                                s['total'] for s in task_stats.values()
                            )
                            
                    elif update.update_type == "update":
                        # Update progress
                        if update.task_id in tasks:
                            job_progress.update(
                                tasks[update.task_id],
                                advance=update.advance,
                                current_item=update.current_item or ""
                            )
                            task_stats[update.task_id]['completed'] += update.advance
                            
                            # Update shared stats
                            shared_stats['completed_files'] = sum(
                                s['completed'] for s in task_stats.values()
                            )
                            shared_stats['total_files'] = sum(
                                s['total'] for s in task_stats.values()
                            )
                            
                    elif update.update_type == "complete":
                        # Complete task
                        if update.task_id in tasks:
                            job_progress.update(
                                tasks[update.task_id],
                                completed=task_stats[update.task_id]['total'],
                                current_item=f"✓ {update.description}"
                            )
                            
                    elif update.update_type == "fail":
                        # Mark task as failed
                        if update.task_id in tasks:
                            job_progress.update(
                                tasks[update.task_id],
                                current_item=f"✗ Failed"
                            )
                            task_stats[update.task_id]['failed'] += 1
                            shared_stats['failed_files'] += 1
                    
                except:
                    # Timeout - update display
                    pass
                
                # Update statistics table
                stats_table = MultiprocessingProgressTracker._create_stats_table(shared_stats)
                
                # Update layout
                layout["main"].update(
                    Panel(job_progress, title="🌍 Processing Tasks", border_style="green")
                )
                layout["footer"].update(
                    Panel.fit(stats_table, title="📊 Statistics", border_style="blue")
                )
                
    @staticmethod
    def _create_stats_table(stats: Dict[str, Any]) -> Table:
        """Create a table with system statistics."""
        table = Table(show_header=False, box=None, expand=True)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right", style="white")
        
        # Calculate values
        elapsed = time.time() - stats.get('start_time', time.time())
        elapsed_str = f"{int(elapsed//60)}:{int(elapsed%60):02d}"
        
        completed = stats.get('completed_files', 0)
        total = stats.get('total_files', 0)
        failed = stats.get('failed_files', 0)
        
        # Calculate throughput
        throughput = completed / elapsed if elapsed > 0 and completed > 0 else 0.0
        
        # System info
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # CPU style
            cpu_style = "green" if cpu_percent < 70 else "yellow" if cpu_percent < 90 else "red"
            cpu_text = Text(f"{cpu_percent:.1f}%", style=cpu_style)
            
            # Memory style
            mem_style = "green" if memory.percent < 70 else "yellow" if memory.percent < 90 else "red"
            mem_text = Text(
                f"{memory.percent:.1f}% ({memory.used/1e9:.1f}GB/{memory.total/1e9:.1f}GB)", 
                style=mem_style
            )
        except:
            cpu_text = Text("N/A", style="dim")
            mem_text = Text("N/A", style="dim")
        
        # Add rows
        table.add_row("⏱️  Elapsed", elapsed_str)
        table.add_row("📁 Files", f"{completed}/{total}")
        
        if failed > 0:
            table.add_row("❌ Failed", Text(str(failed), style="red"))
        else:
            table.add_row("✅ Failed", "0")
            
        table.add_row("⚡ Throughput", f"{throughput:.1f} files/s")
        table.add_row("💻 CPU", cpu_text)
        table.add_row("🧠 Memory", mem_text)
        
        return table


class ProgressReporter:
    """
    Enhanced progress reporter for worker processes.
    """
    
    def __init__(self, queue: Queue):
        self.queue = queue
        
    def create_task(self, task_id: str, description: str, total: int):
        """Create a new progress task."""
        self.queue.put(ProgressUpdate(
            task_id=task_id,
            update_type="create",
            description=description,
            total=total
        ))
        
    def update_task(self, task_id: str, advance: int = 1, current_item: str = "", **kwargs):
        """Update a progress task."""
        update = ProgressUpdate(
            task_id=task_id,
            update_type="update",
            advance=advance,
            current_item=current_item
        )
        if 'description' in kwargs:
            update.description = kwargs['description']
        if 'extra_data' in kwargs:
            update.extra_data = kwargs['extra_data']
        self.queue.put(update)
        
    def update_task_total(self, task_id: str, total: int):
        """Update the total for a task."""
        self.queue.put(ProgressUpdate(
            task_id=task_id,
            update_type="update_total",
            total=total
        ))
        
    def complete_task(self, task_id: str):
        """Mark a task as complete."""
        self.queue.put(ProgressUpdate(
            task_id=task_id,
            update_type="complete"
        ))
        
    def fail_task(self, task_id: str, error_msg: str = None):
        """Mark a task as failed."""
        self.queue.put(ProgressUpdate(
            task_id=task_id,
            update_type="fail",
            description=error_msg
        ))
        
    def update_stats(self, **kwargs):
        """Update global statistics."""
        self.queue.put(ProgressUpdate(
            task_id="__STATS__",
            update_type="update",
            extra_data=kwargs
        ))