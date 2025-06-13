"""
Pipeline interface contracts for communication between processing packages.

These contracts define the interface for pipeline configuration, status reporting,
error handling, and health monitoring across the climate data processing system.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


class PipelineStage(str, Enum):
    """Pipeline processing stages."""
    MEANS_PROCESSING = "means_processing"
    METRICS_PROCESSING = "metrics_processing"
    VALIDATION = "validation"
    CATALOGING = "cataloging"
    ARCHIVAL = "archival"


class ProcessingMode(str, Enum):
    """Processing execution modes."""
    SEQUENTIAL = "sequential"
    MULTIPROCESSING = "multiprocessing"
    DISTRIBUTED = "distributed"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PipelineConfigContract(BaseModel):
    """Configuration contract for pipeline integration."""
    
    # Pipeline identification
    pipeline_id: str = Field(..., description="Unique pipeline identifier")
    pipeline_name: str = Field(..., description="Human-readable pipeline name")
    pipeline_version: str = Field(..., description="Pipeline software version")
    
    # Processing configuration
    processing_mode: ProcessingMode = Field(..., description="Processing execution mode")
    max_workers: int = Field(default=6, ge=1, le=128, description="Maximum worker processes")
    batch_size: int = Field(default=10, ge=1, description="Processing batch size")
    
    # Resource limits
    max_memory_per_worker_gb: float = Field(default=4.0, ge=0.1, description="Memory limit per worker")
    max_processing_time_hours: int = Field(default=24, ge=1, description="Maximum processing time")
    temp_storage_gb: float = Field(default=100.0, ge=1.0, description="Temporary storage allocation")
    
    # Quality thresholds
    min_quality_score: float = Field(default=0.95, ge=0.0, le=1.0, description="Minimum quality score")
    max_error_rate: float = Field(default=0.05, ge=0.0, le=1.0, description="Maximum acceptable error rate")
    
    # Integration settings
    upstream_pipeline: Optional[str] = Field(None, description="Upstream pipeline name")
    downstream_pipelines: List[str] = Field(default_factory=list, description="Downstream pipeline names")
    
    # Data paths
    input_data_path: str = Field(..., description="Input data directory")
    output_data_path: str = Field(..., description="Output data directory")
    catalog_path: str = Field(..., description="Data catalog file path")
    
    # Monitoring configuration
    health_check_interval_seconds: int = Field(default=60, ge=10, description="Health check interval")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pipeline_id": "climate_means_v1",
                "pipeline_name": "Climate Means Processing",
                "pipeline_version": "0.1.0",
                "processing_mode": "multiprocessing",
                "max_workers": 6,
                "batch_size": 15,
                "max_memory_per_worker_gb": 4.0,
                "min_quality_score": 0.95,
                "downstream_pipelines": ["climate_metrics"],
                "input_data_path": "/data/climate/input",
                "output_data_path": "/data/climate/output",
                "catalog_path": "/data/climate/catalog.yaml"
            }
        }


class ProcessingStatusContract(BaseModel):
    """Real-time processing status information."""
    
    # Basic status
    pipeline_id: str = Field(..., description="Pipeline identifier")
    stage: PipelineStage = Field(..., description="Current processing stage")
    status: str = Field(..., description="Current status")
    
    # Progress information
    total_tasks: int = Field(..., ge=0, description="Total number of tasks")
    completed_tasks: int = Field(..., ge=0, description="Number of completed tasks")
    failed_tasks: int = Field(..., ge=0, description="Number of failed tasks")
    active_workers: int = Field(..., ge=0, description="Currently active workers")
    
    # Timing information
    start_time: datetime = Field(..., description="Processing start time")
    last_update: datetime = Field(..., description="Last status update time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    # Performance metrics
    throughput_tasks_per_hour: float = Field(..., ge=0, description="Tasks completed per hour")
    avg_task_duration_seconds: float = Field(..., ge=0, description="Average task duration")
    memory_usage_gb: float = Field(..., ge=0, description="Current memory usage")
    cpu_usage_percent: float = Field(..., ge=0, le=100, description="Current CPU usage")
    
    # Quality metrics
    current_quality_score: float = Field(..., ge=0, le=1, description="Current average quality score")
    error_rate: float = Field(..., ge=0, le=1, description="Current error rate")
    
    # Additional context
    current_dataset: Optional[str] = Field(None, description="Currently processing dataset")
    recent_errors: List[str] = Field(default_factory=list, description="Recent error messages")
    notes: Optional[str] = Field(None, description="Additional status notes")
    
    @field_validator('completed_tasks')
    @classmethod
    def validate_completed_tasks(cls, v, info):
        if hasattr(info, 'data') and 'total_tasks' in info.data and v > info.data['total_tasks']:
            raise ValueError('completed_tasks cannot exceed total_tasks')
        return v
    
    @field_validator('failed_tasks')
    @classmethod
    def validate_failed_tasks(cls, v, info):
        if hasattr(info, 'data') and 'total_tasks' in info.data and v > info.data['total_tasks']:
            raise ValueError('failed_tasks cannot exceed total_tasks')
        return v
    
    @property
    def progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if processing is complete."""
        return self.completed_tasks == self.total_tasks


class ErrorContract(BaseModel):
    """Standardized error information."""
    
    # Error identification
    error_id: str = Field(..., description="Unique error identifier")
    pipeline_id: str = Field(..., description="Pipeline where error occurred")
    stage: PipelineStage = Field(..., description="Processing stage where error occurred")
    
    # Error details
    error_type: str = Field(..., description="Error type/class")
    error_message: str = Field(..., description="Human-readable error message")
    severity: ErrorSeverity = Field(..., description="Error severity level")
    
    # Context information
    timestamp: datetime = Field(..., description="When error occurred")
    dataset_id: Optional[str] = Field(None, description="Dataset being processed when error occurred")
    worker_id: Optional[str] = Field(None, description="Worker process ID")
    
    # Technical details
    exception_type: Optional[str] = Field(None, description="Python exception type")
    stack_trace: Optional[str] = Field(None, description="Full stack trace")
    system_info: Dict[str, Any] = Field(default_factory=dict, description="System information")
    
    # Resolution information
    is_retryable: bool = Field(default=False, description="Whether error is retryable")
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")
    resolution_status: str = Field(default="unresolved", description="Resolution status")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error_id": "err_20240115_143022_001",
                "pipeline_id": "climate_means_v1",
                "stage": "means_processing",
                "error_type": "FileNotFoundError",
                "error_message": "Input NetCDF file not found: tas_day_NorESM2-LM_historical_r1i1p1f1_gn_1995.nc",
                "severity": "medium",
                "timestamp": "2024-01-15T14:30:22Z",
                "dataset_id": "tas_CONUS_historical_1995",
                "is_retryable": True,
                "retry_count": 0,
                "system_info": {
                    "hostname": "climate-worker-01",
                    "python_version": "3.11.5",
                    "available_memory_gb": 15.2
                }
            }
        }


class DataflowContract(BaseModel):
    """Contract for data flow between pipeline stages."""
    
    # Flow identification
    flow_id: str = Field(..., description="Unique dataflow identifier")
    source_pipeline: str = Field(..., description="Source pipeline name")
    target_pipeline: str = Field(..., description="Target pipeline name")
    
    # Data information
    dataset_ids: List[str] = Field(..., description="Dataset IDs being transferred")
    data_size_gb: float = Field(..., ge=0, description="Total data size in GB")
    transfer_method: str = Field(..., description="Transfer method (file, api, queue)")
    
    # Status and timing
    status: str = Field(..., description="Transfer status")
    initiated_at: datetime = Field(..., description="Transfer initiation time")
    completed_at: Optional[datetime] = Field(None, description="Transfer completion time")
    
    # Quality information
    data_integrity_verified: bool = Field(default=False, description="Data integrity verified")
    checksum_validation: bool = Field(default=False, description="Checksum validation performed")
    
    # Metadata
    transfer_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional transfer metadata")
    
    @field_validator('dataset_ids')
    @classmethod
    def validate_dataset_ids(cls, v):
        if len(v) == 0:
            raise ValueError('At least one dataset_id must be provided')
        return v


class HealthCheckContract(BaseModel):
    """Health check results for pipeline monitoring."""
    
    # Basic health information
    pipeline_id: str = Field(..., description="Pipeline identifier")
    overall_status: HealthStatus = Field(..., description="Overall health status")
    check_timestamp: datetime = Field(..., description="Health check timestamp")
    
    # Component health
    database_status: HealthStatus = Field(..., description="Database connectivity status")
    file_system_status: HealthStatus = Field(..., description="File system access status")
    memory_status: HealthStatus = Field(..., description="Memory usage status")
    cpu_status: HealthStatus = Field(..., description="CPU usage status")
    
    # Resource metrics
    memory_usage_percent: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    cpu_usage_percent: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    disk_usage_percent: float = Field(..., ge=0, le=100, description="Disk usage percentage")
    active_connections: int = Field(..., ge=0, description="Active database connections")
    
    # Performance indicators
    recent_error_count: int = Field(..., ge=0, description="Errors in last hour")
    avg_response_time_ms: float = Field(..., ge=0, description="Average response time")
    throughput_last_hour: float = Field(..., ge=0, description="Tasks completed in last hour")
    
    # Additional details
    health_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed health information")
    recommendations: List[str] = Field(default_factory=list, description="Health improvement recommendations")
    
    @property
    def is_healthy(self) -> bool:
        """Check if pipeline is healthy."""
        return self.overall_status == HealthStatus.HEALTHY
    
    @property
    def needs_attention(self) -> bool:
        """Check if pipeline needs attention."""
        return self.overall_status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
    
    class Config:
        json_schema_extra = {
            "example": {
                "pipeline_id": "climate_means_v1",
                "overall_status": "healthy",
                "check_timestamp": "2024-01-15T15:30:00Z",
                "database_status": "healthy",
                "file_system_status": "healthy",
                "memory_status": "healthy",
                "cpu_status": "healthy",
                "memory_usage_percent": 65.2,
                "cpu_usage_percent": 45.8,
                "disk_usage_percent": 78.3,
                "active_connections": 12,
                "recent_error_count": 0,
                "avg_response_time_ms": 125.5,
                "throughput_last_hour": 145.2,
                "recommendations": []
            }
        }