"""
Configuration-driven integration system for climate data processing packages.

This module provides a flexible, declarative approach to defining how different
climate processing packages (means, metrics, etc.) integrate and communicate.
"""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator

from ..contracts.climate_data import Region, ClimateVariable, Scenario


class ProcessingStage(str, Enum):
    """Processing stages in the climate data pipeline."""
    
    MEANS = "means"
    METRICS = "metrics" 
    EXTREMES = "extremes"
    VISUALIZATION = "visualization"
    VALIDATION = "validation"


class TriggerType(str, Enum):
    """Types of triggers for pipeline execution."""
    
    MANUAL = "manual"          # User-initiated
    SCHEDULED = "scheduled"    # Time-based
    EVENT_DRIVEN = "event"     # File/data change triggered
    DEPENDENCY = "dependency"  # Triggered by upstream completion


class DataFlowType(str, Enum):
    """Types of data flow between stages."""
    
    FILE_BASED = "file"        # Files written to disk
    STREAMING = "streaming"    # In-memory data streaming  
    MESSAGE_QUEUE = "queue"    # Async message passing
    DATABASE = "database"      # Shared database


class EnvironmentType(str, Enum):
    """Deployment environments."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ResourceLimits(BaseModel):
    """Resource constraints for processing stages."""
    
    max_memory_gb: Optional[float] = Field(None, ge=0.1)
    max_cpu_cores: Optional[int] = Field(None, ge=1)
    max_processing_time_hours: Optional[float] = Field(None, ge=0.1)
    max_disk_space_gb: Optional[float] = Field(None, ge=0.1)
    priority: int = Field(1, ge=1, le=10, description="1=lowest, 10=highest priority")


class DataTransformation(BaseModel):
    """Configuration for data transformation between stages."""
    
    transformation_id: str = Field(..., description="Unique transformation identifier")
    input_format: str = Field(..., description="Expected input data format")
    output_format: str = Field(..., description="Output data format")
    transformation_script: Optional[str] = Field(None, description="Path to transformation script")
    transformation_params: Dict[str, Any] = Field(default_factory=dict)
    validation_rules: List[str] = Field(default_factory=list)


class StageConfiguration(BaseModel):
    """Configuration for a single processing stage."""
    
    stage_id: str = Field(..., description="Unique stage identifier")
    stage_type: ProcessingStage = Field(..., description="Type of processing stage")
    stage_name: str = Field(..., description="Human-readable stage name")
    package_name: str = Field(..., description="Python package implementing this stage")
    entry_point: str = Field(..., description="Function/class entry point")
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list, description="Required upstream stages")
    optional_depends_on: List[str] = Field(default_factory=list, description="Optional upstream stages")
    
    # Resources
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    
    # Execution
    trigger_type: TriggerType = Field(TriggerType.DEPENDENCY)
    parallel_execution: bool = Field(True, description="Can run in parallel with other stages")
    retry_attempts: int = Field(3, ge=0)
    retry_delay_seconds: float = Field(60.0, ge=1.0)
    
    # Input/Output
    input_data_patterns: List[str] = Field(default_factory=list)
    output_data_patterns: List[str] = Field(default_factory=list)
    data_transformations: List[DataTransformation] = Field(default_factory=list)
    
    # Configuration
    stage_config: Dict[str, Any] = Field(default_factory=dict)
    environment_overrides: Dict[EnvironmentType, Dict[str, Any]] = Field(default_factory=dict)


class DataFlowConfiguration(BaseModel):
    """Configuration for data flow between stages."""
    
    flow_id: str = Field(..., description="Unique flow identifier")
    source_stage: str = Field(..., description="Source stage ID")
    target_stage: str = Field(..., description="Target stage ID")
    flow_type: DataFlowType = Field(DataFlowType.FILE_BASED)
    
    # Data specifications
    data_contracts: List[str] = Field(default_factory=list, description="Required data contract types")
    quality_requirements: Dict[str, Any] = Field(default_factory=dict)
    
    # Flow configuration
    buffer_size: Optional[int] = Field(None, ge=1)
    batch_size: Optional[int] = Field(None, ge=1)
    timeout_seconds: Optional[float] = Field(None, ge=1.0)
    
    # Monitoring
    enable_monitoring: bool = Field(True)
    alert_on_failure: bool = Field(True)
    log_level: str = Field("INFO")


class PipelineConfiguration(BaseModel):
    """Complete pipeline configuration."""
    
    pipeline_id: str = Field(..., description="Unique pipeline identifier")
    pipeline_name: str = Field(..., description="Human-readable pipeline name")
    pipeline_version: str = Field("1.0.0", description="Pipeline configuration version")
    
    # Environment
    environment: EnvironmentType = Field(EnvironmentType.DEVELOPMENT)
    
    # Stages
    stages: List[StageConfiguration] = Field(..., min_length=1)
    data_flows: List[DataFlowConfiguration] = Field(default_factory=list)
    
    # Global settings
    global_resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    global_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Paths
    base_data_path: Path = Field(..., description="Base path for all data")
    temp_data_path: Optional[Path] = Field(None, description="Temporary data storage")
    log_path: Optional[Path] = Field(None, description="Log file location")
    
    # Monitoring
    enable_monitoring: bool = Field(True)
    monitoring_interval_seconds: int = Field(30, ge=1)
    health_check_interval_seconds: int = Field(60, ge=1)
    
    # Error handling
    global_retry_attempts: int = Field(3, ge=0)
    continue_on_stage_failure: bool = Field(False)
    
    # Metadata
    created_by: Optional[str] = Field(None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(None)
    description: Optional[str] = Field(None)
    tags: List[str] = Field(default_factory=list)
    
    @model_validator(mode='after')
    def validate_stage_dependencies(self):
        """Validate that stage dependencies are satisfied."""
        stages = self.stages
        stage_ids = {stage.stage_id for stage in stages}
        
        for stage in stages:
            # Check required dependencies exist
            for dep in stage.depends_on:
                if dep not in stage_ids:
                    raise ValueError(f"Stage {stage.stage_id} depends on {dep} which doesn't exist")
            
            # Check optional dependencies exist
            for dep in stage.optional_depends_on:
                if dep not in stage_ids:
                    raise ValueError(f"Stage {stage.stage_id} optionally depends on {dep} which doesn't exist")
        
        return self
    
    @model_validator(mode='after')
    def validate_data_flows(self):
        """Validate that data flows reference existing stages."""
        stages = self.stages
        data_flows = self.data_flows
        stage_ids = {stage.stage_id for stage in stages}
        
        for flow in data_flows:
            if flow.source_stage not in stage_ids:
                raise ValueError(f"Data flow {flow.flow_id} references non-existent source stage {flow.source_stage}")
            if flow.target_stage not in stage_ids:
                raise ValueError(f"Data flow {flow.flow_id} references non-existent target stage {flow.target_stage}")
        
        return self
    
    def get_stage_by_id(self, stage_id: str) -> Optional[StageConfiguration]:
        """Get stage configuration by ID."""
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None
    
    def get_upstream_stages(self, stage_id: str) -> List[StageConfiguration]:
        """Get all upstream stages for a given stage."""
        stage = self.get_stage_by_id(stage_id)
        if not stage:
            return []
        
        upstream = []
        for dep in stage.depends_on + stage.optional_depends_on:
            dep_stage = self.get_stage_by_id(dep)
            if dep_stage:
                upstream.append(dep_stage)
        
        return upstream
    
    def get_downstream_stages(self, stage_id: str) -> List[StageConfiguration]:
        """Get all downstream stages for a given stage."""
        downstream = []
        for stage in self.stages:
            if stage_id in stage.depends_on + stage.optional_depends_on:
                downstream.append(stage)
        return downstream
    
    def get_execution_order(self) -> List[List[str]]:
        """Get stages in execution order (topological sort)."""
        # Simple implementation - can be made more sophisticated
        remaining_stages = {stage.stage_id: set(stage.depends_on) for stage in self.stages}
        execution_order = []
        
        while remaining_stages:
            # Find stages with no dependencies
            ready_stages = [
                stage_id for stage_id, deps in remaining_stages.items() 
                if not deps
            ]
            
            if not ready_stages:
                # Circular dependency
                raise ValueError(f"Circular dependency detected in stages: {list(remaining_stages.keys())}")
            
            execution_order.append(ready_stages)
            
            # Remove ready stages and update dependencies
            for stage_id in ready_stages:
                del remaining_stages[stage_id]
            
            for deps in remaining_stages.values():
                deps -= set(ready_stages)
        
        return execution_order


class ProcessingProfile(BaseModel):
    """Predefined processing profiles for common scenarios."""
    
    profile_name: str = Field(..., description="Profile identifier")
    description: str = Field(..., description="Profile description")
    regions: List[Region] = Field(..., description="Regions to process")
    variables: List[ClimateVariable] = Field(..., description="Variables to process")
    scenarios: List[Scenario] = Field(..., description="Climate scenarios")
    year_ranges: List[tuple] = Field(..., description="Year ranges to process")
    
    # Processing options
    enable_means: bool = Field(True)
    enable_metrics: bool = Field(True)
    enable_extremes: bool = Field(False)
    enable_visualization: bool = Field(False)
    
    # Performance settings
    max_parallel_regions: int = Field(2, ge=1)
    max_parallel_variables: int = Field(4, ge=1)
    memory_per_process_gb: float = Field(4.0, ge=0.5)
    
    def to_pipeline_config(self, base_config: PipelineConfiguration) -> PipelineConfiguration:
        """Convert profile to pipeline configuration."""
        # This would generate a pipeline config based on the profile
        # Implementation would customize the base_config based on profile settings
        config = base_config.copy(deep=True)
        
        # Update global config with profile settings
        config.global_config.update({
            "profile_name": self.profile_name,
            "regions": [r.value for r in self.regions],
            "variables": [v.value for v in self.variables],
            "scenarios": [s.value for s in self.scenarios],
            "year_ranges": self.year_ranges,
            "max_parallel_regions": self.max_parallel_regions,
            "max_parallel_variables": self.max_parallel_variables,
        })
        
        # Filter stages based on profile
        active_stages = []
        for stage in config.stages:
            if stage.stage_type == ProcessingStage.MEANS and self.enable_means:
                active_stages.append(stage)
            elif stage.stage_type == ProcessingStage.METRICS and self.enable_metrics:
                active_stages.append(stage)
            elif stage.stage_type == ProcessingStage.EXTREMES and self.enable_extremes:
                active_stages.append(stage)
            elif stage.stage_type == ProcessingStage.VISUALIZATION and self.enable_visualization:
                active_stages.append(stage)
            elif stage.stage_type == ProcessingStage.VALIDATION:
                # Always include validation
                active_stages.append(stage)
        
        config.stages = active_stages
        
        return config