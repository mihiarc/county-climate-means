"""
Climate Data Processing Manifest System

Tracks the status of climate data processing across all pipeline stages.
Follows ETL best practices for data lineage and processing status tracking.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd


class ProcessingStatus(Enum):
    """Status of data processing for a specific task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProcessingStage(Enum):
    """Pipeline processing stages."""
    MEANS = "means"
    METRICS = "metrics"
    VALIDATION = "validation"


@dataclass
class ProcessingTask:
    """Represents a single processing task."""
    task_id: str
    stage: ProcessingStage
    region: str
    variable: str
    scenario: str
    status: ProcessingStatus
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    input_files: List[str] = None
    output_files: List[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.input_files is None:
            self.input_files = []
        if self.output_files is None:
            self.output_files = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessingManifest:
    """Manifest tracking all processing tasks."""
    manifest_version: str = "1.0"
    created_at: str = None
    updated_at: str = None
    pipeline_id: str = None
    base_output_path: str = None
    tasks: Dict[str, ProcessingTask] = None
    summary: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = {}
        if self.summary is None:
            self.summary = {}
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at


class ClimateDataManifest:
    """
    Manages the processing manifest for climate data pipeline.
    
    Provides methods to:
    - Track processing status across all pipeline stages
    - Identify what data needs processing
    - Generate status reports
    - Support resume/retry operations
    """
    
    def __init__(self, manifest_path: Path, base_output_path: Path):
        self.manifest_path = Path(manifest_path)
        self.base_output_path = Path(base_output_path)
        self.logger = logging.getLogger(__name__)
        self.manifest: Optional[ProcessingManifest] = None
        
        # Load existing manifest or create new one
        self.load_or_create_manifest()
    
    def load_or_create_manifest(self):
        """Load existing manifest or create a new one."""
        if self.manifest_path.exists():
            self.logger.info(f"Loading existing manifest from {self.manifest_path}")
            self.load_manifest()
        else:
            self.logger.info(f"Creating new manifest at {self.manifest_path}")
            self.manifest = ProcessingManifest(
                base_output_path=str(self.base_output_path),
                pipeline_id="climate_data_processing"
            )
            self.save_manifest()
    
    def load_manifest(self):
        """Load manifest from JSON file."""
        try:
            with open(self.manifest_path, 'r') as f:
                data = json.load(f)
            
            # Convert task data back to ProcessingTask objects
            tasks = {}
            for task_id, task_data in data.get('tasks', {}).items():
                task_data['stage'] = ProcessingStage(task_data['stage'])
                task_data['status'] = ProcessingStatus(task_data['status'])
                tasks[task_id] = ProcessingTask(**task_data)
            
            self.manifest = ProcessingManifest(
                manifest_version=data.get('manifest_version', '1.0'),
                created_at=data.get('created_at'),
                updated_at=data.get('updated_at'),
                pipeline_id=data.get('pipeline_id'),
                base_output_path=data.get('base_output_path'),
                tasks=tasks,
                summary=data.get('summary', {})
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load manifest: {e}")
            raise
    
    def save_manifest(self):
        """Save manifest to JSON file."""
        try:
            # Convert to dict with enum serialization
            data = asdict(self.manifest)
            
            # Convert enums to strings
            for task_id, task_data in data['tasks'].items():
                task_data['stage'] = task_data['stage'].value
                task_data['status'] = task_data['status'].value
            
            # Update timestamp
            data['updated_at'] = datetime.now(timezone.utc).isoformat()
            
            # Ensure directory exists
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.manifest_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            self.logger.debug(f"Saved manifest to {self.manifest_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save manifest: {e}")
            raise
    
    def register_task(self, 
                     region: str, 
                     variable: str, 
                     scenario: str, 
                     stage: ProcessingStage) -> str:
        """Register a new processing task."""
        task_id = f"{stage.value}_{region}_{variable}_{scenario}"
        
        if task_id not in self.manifest.tasks:
            task = ProcessingTask(
                task_id=task_id,
                stage=stage,
                region=region,
                variable=variable,
                scenario=scenario,
                status=ProcessingStatus.PENDING
            )
            self.manifest.tasks[task_id] = task
            self.save_manifest()
            self.logger.info(f"Registered new task: {task_id}")
        
        return task_id
    
    def start_task(self, task_id: str, input_files: List[str] = None):
        """Mark a task as started."""
        if task_id not in self.manifest.tasks:
            raise ValueError(f"Task {task_id} not found in manifest")
        
        task = self.manifest.tasks[task_id]
        task.status = ProcessingStatus.IN_PROGRESS
        task.start_time = datetime.now(timezone.utc).isoformat()
        
        if input_files:
            task.input_files = input_files
        
        self.save_manifest()
        self.logger.info(f"Started task: {task_id}")
    
    def complete_task(self, 
                     task_id: str, 
                     output_files: List[str] = None,
                     metadata: Dict[str, Any] = None):
        """Mark a task as completed."""
        if task_id not in self.manifest.tasks:
            raise ValueError(f"Task {task_id} not found in manifest")
        
        task = self.manifest.tasks[task_id]
        task.status = ProcessingStatus.COMPLETED
        task.end_time = datetime.now(timezone.utc).isoformat()
        
        if task.start_time:
            start = datetime.fromisoformat(task.start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(task.end_time.replace('Z', '+00:00'))
            task.duration_seconds = (end - start).total_seconds()
        
        if output_files:
            task.output_files = output_files
        
        if metadata:
            task.metadata.update(metadata)
        
        self.save_manifest()
        self.logger.info(f"Completed task: {task_id}")
    
    def fail_task(self, task_id: str, error_message: str):
        """Mark a task as failed."""
        if task_id not in self.manifest.tasks:
            raise ValueError(f"Task {task_id} not found in manifest")
        
        task = self.manifest.tasks[task_id]
        task.status = ProcessingStatus.FAILED
        task.end_time = datetime.now(timezone.utc).isoformat()
        task.error_message = error_message
        
        if task.start_time:
            start = datetime.fromisoformat(task.start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(task.end_time.replace('Z', '+00:00'))
            task.duration_seconds = (end - start).total_seconds()
        
        self.save_manifest()
        self.logger.warning(f"Failed task: {task_id} - {error_message}")
    
    def get_pending_tasks(self, 
                         stage: Optional[ProcessingStage] = None,
                         region: Optional[str] = None,
                         variable: Optional[str] = None) -> List[ProcessingTask]:
        """Get all pending tasks, optionally filtered."""
        tasks = []
        for task in self.manifest.tasks.values():
            if task.status == ProcessingStatus.PENDING:
                if stage and task.stage != stage:
                    continue
                if region and task.region != region:
                    continue
                if variable and task.variable != variable:
                    continue
                tasks.append(task)
        return tasks
    
    def get_completed_tasks(self, 
                           stage: Optional[ProcessingStage] = None) -> List[ProcessingTask]:
        """Get all completed tasks."""
        tasks = []
        for task in self.manifest.tasks.values():
            if task.status == ProcessingStatus.COMPLETED:
                if stage and task.stage != stage:
                    continue
                tasks.append(task)
        return tasks
    
    def get_failed_tasks(self) -> List[ProcessingTask]:
        """Get all failed tasks."""
        return [task for task in self.manifest.tasks.values() 
                if task.status == ProcessingStatus.FAILED]
    
    def discover_existing_outputs(self):
        """
        Discover existing output files and update manifest accordingly.
        This is useful for initializing the manifest from existing data.
        """
        self.logger.info("Discovering existing output files...")
        
        # Define expected patterns for each stage
        patterns = {
            ProcessingStage.MEANS: {
                'path_pattern': self.base_output_path / 'means' / '{variable}' / '{scenario}',
                'file_pattern': '{variable}_{region}_{scenario}_*_30yr_normal.nc'
            },
            ProcessingStage.METRICS: {
                'path_pattern': self.base_output_path / 'metrics_timeseries',
                'file_pattern': '{region}_{variable}_metrics.csv'
            },
            ProcessingStage.VALIDATION: {
                'path_pattern': self.base_output_path / 'validation',
                'file_pattern': '{region}_{variable}_validation_*.json'
            }
        }
        
        regions = ['CONUS', 'AK', 'HI', 'PRVI', 'GU']
        variables = ['pr', 'tas', 'tasmax', 'tasmin']
        scenarios = ['historical', 'ssp245', 'ssp585']
        
        for stage in ProcessingStage:
            for region in regions:
                for variable in variables:
                    for scenario in scenarios:
                        task_id = self.register_task(region, variable, scenario, stage)
                        
                        # Check if output exists
                        output_exists = self._check_output_exists(stage, region, variable, scenario)
                        
                        if output_exists:
                            self.manifest.tasks[task_id].status = ProcessingStatus.COMPLETED
                            self.logger.debug(f"Found existing output for {task_id}")
        
        self.save_manifest()
        self.logger.info("Completed output discovery")
    
    def _check_output_exists(self, stage: ProcessingStage, region: str, variable: str, scenario: str) -> bool:
        """Check if output exists for a given task."""
        try:
            if stage == ProcessingStage.MEANS:
                # For means, we need to check if all years are processed
                # Historical: 1980-2014 (35 years)
                # SSP245/SSP585: 2015-2100 (86 years)
                # Total: 121 years per scenario
                
                if scenario == 'historical':
                    # Check historical directory only
                    path = self.base_output_path / 'means' / variable / 'historical'
                    pattern = f"{variable}_{region}_historical_*_30yr_normal.nc"
                    expected_count = 35  # 1980-2014
                else:
                    # For projections, check all directories where files might be
                    # Files could be in scenario dir or hybrid dir (for transition years)
                    all_files = set()
                    
                    # Check scenario-specific directory
                    scenario_path = self.base_output_path / 'means' / variable / scenario
                    if scenario_path.exists():
                        pattern = f"{variable}_{region}_{scenario}_*_30yr_normal.nc"
                        all_files.update(scenario_path.glob(pattern))
                    
                    # Check hybrid directory (contains transition period files)
                    hybrid_path = self.base_output_path / 'means' / variable / 'hybrid'
                    if hybrid_path.exists():
                        # Hybrid files don't have scenario in the name
                        pattern = f"{variable}_{region}_*_30yr_normal.nc"
                        all_files.update(hybrid_path.glob(pattern))
                    
                    expected_count = 86  # 2015-2100
                    
                    # Extract years and count unique ones
                    years = set()
                    for f in all_files:
                        parts = f.stem.split('_')
                        # Find the year - it's followed by "30yr"
                        for i in range(len(parts) - 1):
                            if parts[i].isdigit() and i + 1 < len(parts) and parts[i + 1] == "30yr":
                                year = int(parts[i])
                                if 2015 <= year <= 2100:
                                    years.add(year)
                                break
                    
                    return len(years) == expected_count
                
                # Count files for the scenario
                if path.exists():
                    files = list(path.glob(pattern))
                    return len(files) == expected_count
                
                return False
                
            elif stage == ProcessingStage.METRICS:
                path = self.base_output_path / 'metrics_timeseries'
                file_path = path / f"{region}_{variable}_metrics.csv"
                return file_path.exists()
                
            elif stage == ProcessingStage.VALIDATION:
                path = self.base_output_path / 'validation'
                # Check for any validation output directory
                return len(list(path.glob(f"*{region.lower()}*"))) > 0
                
        except Exception as e:
            self.logger.debug(f"Error checking output for {stage.value}_{region}_{variable}_{scenario}: {e}")
            return False
        
        return False
    
    def generate_status_report(self) -> Dict[str, Any]:
        """Generate a comprehensive status report."""
        report = {
            'manifest_info': {
                'version': self.manifest.manifest_version,
                'created_at': self.manifest.created_at,
                'updated_at': self.manifest.updated_at,
                'total_tasks': len(self.manifest.tasks)
            },
            'status_summary': {},
            'stage_summary': {},
            'region_summary': {},
            'variable_summary': {},
            'failed_tasks': [],
            'completion_rates': {}
        }
        
        # Status summary
        for status in ProcessingStatus:
            count = sum(1 for task in self.manifest.tasks.values() if task.status == status)
            report['status_summary'][status.value] = count
        
        # Stage summary
        for stage in ProcessingStage:
            stage_tasks = [task for task in self.manifest.tasks.values() if task.stage == stage]
            completed = sum(1 for task in stage_tasks if task.status == ProcessingStatus.COMPLETED)
            total = len(stage_tasks)
            report['stage_summary'][stage.value] = {
                'completed': completed,
                'total': total,
                'completion_rate': completed / total if total > 0 else 0
            }
        
        # Region summary
        regions = set(task.region for task in self.manifest.tasks.values())
        for region in regions:
            region_tasks = [task for task in self.manifest.tasks.values() if task.region == region]
            completed = sum(1 for task in region_tasks if task.status == ProcessingStatus.COMPLETED)
            total = len(region_tasks)
            report['region_summary'][region] = {
                'completed': completed,
                'total': total,
                'completion_rate': completed / total if total > 0 else 0
            }
        
        # Variable summary
        variables = set(task.variable for task in self.manifest.tasks.values())
        for variable in variables:
            var_tasks = [task for task in self.manifest.tasks.values() if task.variable == variable]
            completed = sum(1 for task in var_tasks if task.status == ProcessingStatus.COMPLETED)
            total = len(var_tasks)
            report['variable_summary'][variable] = {
                'completed': completed,
                'total': total,
                'completion_rate': completed / total if total > 0 else 0
            }
        
        # Failed tasks
        report['failed_tasks'] = [
            {
                'task_id': task.task_id,
                'stage': task.stage.value,
                'region': task.region,
                'variable': task.variable,
                'scenario': task.scenario,
                'error': task.error_message
            }
            for task in self.manifest.tasks.values()
            if task.status == ProcessingStatus.FAILED
        ]
        
        # Overall completion rate
        total_tasks = len(self.manifest.tasks)
        completed_tasks = sum(1 for task in self.manifest.tasks.values() 
                            if task.status == ProcessingStatus.COMPLETED)
        report['completion_rates']['overall'] = completed_tasks / total_tasks if total_tasks > 0 else 0
        
        return report
    
    def export_status_to_csv(self, output_path: Path):
        """Export status report to CSV for spreadsheet analysis."""
        data = []
        for task in self.manifest.tasks.values():
            data.append({
                'task_id': task.task_id,
                'stage': task.stage.value,
                'region': task.region,
                'variable': task.variable,
                'scenario': task.scenario,
                'status': task.status.value,
                'start_time': task.start_time,
                'end_time': task.end_time,
                'duration_seconds': task.duration_seconds,
                'num_input_files': len(task.input_files),
                'num_output_files': len(task.output_files),
                'has_error': task.error_message is not None,
                'error_message': task.error_message
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Exported status report to {output_path}")
    
    def get_year_coverage(self, region: str, variable: str, scenario: str) -> Dict[str, Any]:
        """Get detailed year coverage for a specific region/variable/scenario combination."""
        coverage = {
            'region': region,
            'variable': variable,
            'scenario': scenario,
            'expected_years': [],
            'found_years': [],
            'missing_years': [],
            'coverage_percent': 0.0,
            'files_by_directory': {}
        }
        
        if scenario == 'historical':
            coverage['expected_years'] = list(range(1980, 2015))  # 1980-2014
            path = self.base_output_path / 'means' / variable / 'historical'
            pattern = f"{variable}_{region}_historical_*_30yr_normal.nc"
            
            if path.exists():
                files = list(path.glob(pattern))
                coverage['files_by_directory']['historical'] = len(files)
                
                for f in files:
                    parts = f.stem.split('_')
                    # Find the year - it's followed by "30yr"
                    for i in range(len(parts) - 1):
                        if parts[i].isdigit() and i + 1 < len(parts) and parts[i + 1] == "30yr":
                            year = int(parts[i])
                            if 1980 <= year <= 2014:
                                coverage['found_years'].append(year)
                            break
        
        else:  # ssp245 or ssp585
            coverage['expected_years'] = list(range(2015, 2101))  # 2015-2100
            
            # Check scenario directory
            scenario_path = self.base_output_path / 'means' / variable / scenario
            if scenario_path.exists():
                pattern = f"{variable}_{region}_{scenario}_*_30yr_normal.nc"
                files = list(scenario_path.glob(pattern))
                coverage['files_by_directory'][scenario] = len(files)
                
                for f in files:
                    parts = f.stem.split('_')
                    # Find the year - it's followed by "30yr"
                    for i in range(len(parts) - 1):
                        if parts[i].isdigit() and i + 1 < len(parts) and parts[i + 1] == "30yr":
                            year = int(parts[i])
                            if 2015 <= year <= 2100:
                                coverage['found_years'].append(year)
                            break
            
            # Check hybrid directory
            hybrid_path = self.base_output_path / 'means' / variable / 'hybrid'
            if hybrid_path.exists():
                pattern = f"{variable}_{region}_*_30yr_normal.nc"
                files = list(hybrid_path.glob(pattern))
                coverage['files_by_directory']['hybrid'] = len(files)
                
                for f in files:
                    parts = f.stem.split('_')
                    # Find the year - it's followed by "30yr"
                    for i in range(len(parts) - 1):
                        if parts[i].isdigit() and i + 1 < len(parts) and parts[i + 1] == "30yr":
                            year = int(parts[i])
                            if 2015 <= year <= 2100:
                                coverage['found_years'].append(year)
                            break
        
        # Calculate coverage
        coverage['found_years'] = sorted(list(set(coverage['found_years'])))
        coverage['missing_years'] = sorted(list(set(coverage['expected_years']) - set(coverage['found_years'])))
        coverage['coverage_percent'] = len(coverage['found_years']) / len(coverage['expected_years']) * 100 if coverage['expected_years'] else 0
        
        return coverage