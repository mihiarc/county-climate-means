"""
Pipeline orchestration engine for configuration-driven climate data processing.

This module provides the core orchestration logic that executes processing
stages based on pipeline configuration, manages dependencies, and handles
error recovery.
"""

import asyncio
import importlib
import logging
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum

from pydantic import BaseModel

from ..config.integration_config import (
    PipelineConfiguration,
    StageConfiguration,
    ProcessingStage,
    TriggerType
)
from ..contracts.pipeline_interface import (
    ProcessingStatusContract,
    ProcessingMode,
    ErrorContract
)


class ExecutionStatus(str, Enum):
    """Execution status for pipeline and stages."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class StageExecution(BaseModel):
    """Runtime execution state for a pipeline stage."""
    
    stage_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    attempt_number: int = 0
    error_message: Optional[str] = None
    output_data: Dict[str, Any] = {}
    resource_usage: Dict[str, float] = {}
    
    def start_execution(self):
        """Mark stage as started."""
        self.status = ExecutionStatus.RUNNING
        self.start_time = datetime.now(timezone.utc)
        self.attempt_number += 1
    
    def complete_execution(self, output_data: Optional[Dict[str, Any]] = None):
        """Mark stage as completed."""
        self.status = ExecutionStatus.COMPLETED
        self.end_time = datetime.now(timezone.utc)
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        if output_data:
            self.output_data.update(output_data)
    
    def fail_execution(self, error_message: str):
        """Mark stage as failed."""
        self.status = ExecutionStatus.FAILED
        self.end_time = datetime.now(timezone.utc)
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.error_message = error_message


class PipelineExecution(BaseModel):
    """Runtime execution state for entire pipeline."""
    
    pipeline_id: str
    execution_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    stage_executions: Dict[str, StageExecution] = {}
    global_context: Dict[str, Any] = {}
    
    def get_stage_execution(self, stage_id: str) -> Optional[StageExecution]:
        """Get execution state for a stage."""
        return self.stage_executions.get(stage_id)
    
    def is_stage_ready(self, stage_config: StageConfiguration) -> bool:
        """Check if stage dependencies are satisfied."""
        for dep_stage_id in stage_config.depends_on:
            dep_execution = self.get_stage_execution(dep_stage_id)
            if not dep_execution or dep_execution.status != ExecutionStatus.COMPLETED:
                return False
        return True
    
    def get_failed_stages(self) -> List[str]:
        """Get list of failed stage IDs."""
        return [
            stage_id for stage_id, execution in self.stage_executions.items()
            if execution.status == ExecutionStatus.FAILED
        ]
    
    def get_completed_stages(self) -> List[str]:
        """Get list of completed stage IDs."""
        return [
            stage_id for stage_id, execution in self.stage_executions.items()
            if execution.status == ExecutionStatus.COMPLETED
        ]


class StageExecutor:
    """Executes individual pipeline stages."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize stage executor.
        
        Args:
            logger: Logger instance for execution logging
        """
        self.logger = logger or logging.getLogger(__name__)
        self._stage_registry: Dict[str, Callable] = {}
    
    def register_stage_handler(self, stage_type: ProcessingStage, handler: Callable):
        """Register a handler function for a stage type.
        
        Args:
            stage_type: Type of processing stage
            handler: Function to handle stage execution
        """
        self._stage_registry[stage_type.value] = handler
    
    async def execute_stage(
        self,
        stage_config: StageConfiguration,
        pipeline_context: Dict[str, Any],
        stage_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single pipeline stage.
        
        Args:
            stage_config: Configuration for the stage
            pipeline_context: Global pipeline context
            stage_inputs: Input data for the stage
            
        Returns:
            Stage output data
            
        Raises:
            Exception: If stage execution fails
        """
        self.logger.info(f"Executing stage: {stage_config.stage_id}")
        
        try:
            # Get stage handler
            if stage_config.stage_type.value in self._stage_registry:
                handler = self._stage_registry[stage_config.stage_type.value]
            else:
                handler = self._load_stage_handler(stage_config)
            
            # Prepare stage context
            stage_context = {
                "stage_config": stage_config.dict(),
                "pipeline_context": pipeline_context,
                "stage_inputs": stage_inputs,
                "logger": self.logger,
            }
            
            # Execute stage (handle both sync and async handlers)
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**stage_context)
            else:
                # Run synchronous handler in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: handler(**stage_context))
            
            self.logger.info(f"Stage {stage_config.stage_id} completed successfully")
            return result or {}
            
        except Exception as e:
            self.logger.error(f"Stage {stage_config.stage_id} failed: {e}")
            raise
    
    def _load_stage_handler(self, stage_config: StageConfiguration) -> Callable:
        """Dynamically load stage handler from package."""
        try:
            module = importlib.import_module(stage_config.package_name)
            handler = getattr(module, stage_config.entry_point)
            return handler
        except (ImportError, AttributeError) as e:
            raise RuntimeError(f"Failed to load stage handler {stage_config.package_name}.{stage_config.entry_point}: {e}")


class PipelineOrchestrator:
    """Main orchestration engine for pipeline execution."""
    
    def __init__(
        self,
        config: PipelineConfiguration,
        max_concurrent_stages: int = 4,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize pipeline orchestrator.
        
        Args:
            config: Pipeline configuration
            max_concurrent_stages: Maximum number of stages to run concurrently
            logger: Logger instance
        """
        self.config = config
        self.max_concurrent_stages = max_concurrent_stages
        self.logger = logger or logging.getLogger(__name__)
        
        self.stage_executor = StageExecutor(self.logger)
        self.execution_state: Optional[PipelineExecution] = None
        
        # Event handling
        self._shutdown_event = threading.Event()
        self._pause_event = threading.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def register_stage_handler(self, stage_type: ProcessingStage, handler: Callable):
        """Register a custom handler for a stage type."""
        self.stage_executor.register_stage_handler(stage_type, handler)
    
    async def execute_pipeline(
        self,
        execution_id: Optional[str] = None,
        resume_from_failure: bool = False
    ) -> PipelineExecution:
        """Execute the complete pipeline.
        
        Args:
            execution_id: Unique execution identifier
            resume_from_failure: Whether to resume from previous failure
            
        Returns:
            Pipeline execution state
        """
        if not execution_id:
            execution_id = f"exec_{int(time.time())}"
        
        self.logger.info(f"Starting pipeline execution: {execution_id}")
        
        # Initialize execution state
        self.execution_state = PipelineExecution(
            pipeline_id=self.config.pipeline_id,
            execution_id=execution_id,
            status=ExecutionStatus.RUNNING,
            start_time=datetime.now(timezone.utc)
        )
        
        # Initialize stage executions
        for stage in self.config.stages:
            if stage.stage_id not in self.execution_state.stage_executions:
                self.execution_state.stage_executions[stage.stage_id] = StageExecution(
                    stage_id=stage.stage_id
                )
        
        try:
            # Get execution order
            execution_order = self.config.get_execution_order()
            self.logger.info(f"Execution order: {execution_order}")
            
            # Execute stages in order
            for stage_batch in execution_order:
                if self._shutdown_event.is_set():
                    self.logger.info("Pipeline execution cancelled")
                    break
                
                # Filter stages if resuming from failure
                if resume_from_failure:
                    stage_batch = [
                        stage_id for stage_id in stage_batch
                        if self.execution_state.stage_executions[stage_id].status not in 
                        [ExecutionStatus.COMPLETED, ExecutionStatus.SKIPPED]
                    ]
                
                if stage_batch:
                    await self._execute_stage_batch(stage_batch)
            
            # Determine final status
            failed_stages = self.execution_state.get_failed_stages()
            if failed_stages:
                self.execution_state.status = ExecutionStatus.FAILED
                self.logger.error(f"Pipeline failed due to stage failures: {failed_stages}")
            elif self._shutdown_event.is_set():
                self.execution_state.status = ExecutionStatus.CANCELLED
            else:
                self.execution_state.status = ExecutionStatus.COMPLETED
                self.logger.info("Pipeline execution completed successfully")
            
        except Exception as e:
            self.execution_state.status = ExecutionStatus.FAILED
            self.logger.error(f"Pipeline execution failed: {e}")
        
        finally:
            self.execution_state.end_time = datetime.now(timezone.utc)
        
        return self.execution_state
    
    async def _execute_stage_batch(self, stage_ids: List[str]):
        """Execute a batch of stages concurrently."""
        self.logger.info(f"Executing stage batch: {stage_ids}")
        
        # Create tasks for concurrent execution
        tasks = []
        for stage_id in stage_ids:
            stage_config = self.config.get_stage_by_id(stage_id)
            if stage_config and self.execution_state.is_stage_ready(stage_config):
                task = asyncio.create_task(
                    self._execute_single_stage(stage_config),
                    name=f"stage_{stage_id}"
                )
                tasks.append(task)
            else:
                # Skip stage if dependencies not met
                execution = self.execution_state.stage_executions[stage_id]
                execution.status = ExecutionStatus.SKIPPED
                self.logger.warning(f"Skipping stage {stage_id} - dependencies not met")
        
        # Wait for all tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_single_stage(self, stage_config: StageConfiguration):
        """Execute a single stage with retry logic."""
        stage_execution = self.execution_state.stage_executions[stage_config.stage_id]
        
        for attempt in range(stage_config.retry_attempts + 1):
            if self._shutdown_event.is_set():
                break
            
            # Wait for pause if requested
            while self._pause_event.is_set() and not self._shutdown_event.is_set():
                await asyncio.sleep(1)
            
            try:
                stage_execution.start_execution()
                
                # Collect stage inputs from upstream stages
                stage_inputs = self._collect_stage_inputs(stage_config)
                
                # Execute stage
                output_data = await self.stage_executor.execute_stage(
                    stage_config,
                    self.execution_state.global_context,
                    stage_inputs
                )
                
                stage_execution.complete_execution(output_data)
                break
                
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Stage {stage_config.stage_id} attempt {attempt + 1} failed: {error_msg}")
                
                if attempt < stage_config.retry_attempts:
                    # Wait before retry
                    await asyncio.sleep(stage_config.retry_delay_seconds)
                else:
                    # Final failure
                    stage_execution.fail_execution(error_msg)
                    
                    # Check if pipeline should continue
                    if not self.config.continue_on_stage_failure:
                        self.logger.error("Pipeline configured to stop on stage failure")
                        self._shutdown_event.set()
    
    def _collect_stage_inputs(self, stage_config: StageConfiguration) -> Dict[str, Any]:
        """Collect input data from upstream stages."""
        inputs = {}
        
        for dep_stage_id in stage_config.depends_on + stage_config.optional_depends_on:
            dep_execution = self.execution_state.get_stage_execution(dep_stage_id)
            if dep_execution and dep_execution.status == ExecutionStatus.COMPLETED:
                inputs[dep_stage_id] = dep_execution.output_data
        
        return inputs
    
    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self._shutdown_event.set()
    
    def pause_execution(self):
        """Pause pipeline execution."""
        self.logger.info("Pausing pipeline execution")
        self._pause_event.set()
    
    def resume_execution(self):
        """Resume pipeline execution."""
        self.logger.info("Resuming pipeline execution")
        self._pause_event.clear()
    
    def cancel_execution(self):
        """Cancel pipeline execution."""
        self.logger.info("Cancelling pipeline execution")
        self._shutdown_event.set()
    
    def get_execution_status(self) -> Optional[ProcessingStatusContract]:
        """Get current execution status as a contract."""
        if not self.execution_state:
            return None
        
        return ProcessingStatusContract(
            pipeline_id=self.execution_state.pipeline_id,
            execution_id=self.execution_state.execution_id,
            status=self.execution_state.status.value,
            start_time=self.execution_state.start_time,
            current_stage=self._get_current_stage(),
            completed_stages=len(self.execution_state.get_completed_stages()),
            total_stages=len(self.config.stages),
            processing_mode=ProcessingMode.MULTIPROCESSING,
            errors=[
                ErrorContract(
                    error_type="stage_failure",
                    error_message=execution.error_message,
                    stage_id=execution.stage_id,
                    timestamp=execution.end_time
                )
                for execution in self.execution_state.stage_executions.values()
                if execution.status == ExecutionStatus.FAILED and execution.error_message
            ]
        )
    
    def _get_current_stage(self) -> Optional[str]:
        """Get ID of currently running stage."""
        for stage_id, execution in self.execution_state.stage_executions.items():
            if execution.status == ExecutionStatus.RUNNING:
                return stage_id
        return None


class PipelineRunner:
    """High-level interface for running pipelines."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize pipeline runner.
        
        Args:
            config_path: Path to pipeline configuration file
        """
        self.config_path = config_path
        self.orchestrator: Optional[PipelineOrchestrator] = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def run_pipeline(
        self,
        config: Optional[PipelineConfiguration] = None,
        execution_id: Optional[str] = None,
        resume_from_failure: bool = False
    ) -> PipelineExecution:
        """Run pipeline synchronously.
        
        Args:
            config: Pipeline configuration (loads from file if not provided)
            execution_id: Unique execution identifier
            resume_from_failure: Whether to resume from previous failure
            
        Returns:
            Pipeline execution results
        """
        return asyncio.run(self.run_pipeline_async(config, execution_id, resume_from_failure))
    
    async def run_pipeline_async(
        self,
        config: Optional[PipelineConfiguration] = None,
        execution_id: Optional[str] = None,
        resume_from_failure: bool = False
    ) -> PipelineExecution:
        """Run pipeline asynchronously.
        
        Args:
            config: Pipeline configuration (loads from file if not provided)
            execution_id: Unique execution identifier
            resume_from_failure: Whether to resume from previous failure
            
        Returns:
            Pipeline execution results
        """
        # Load configuration if not provided
        if not config:
            if not self.config_path:
                raise ValueError("Either config or config_path must be provided")
            
            from ..config.config_loader import ConfigurationLoader
            loader = ConfigurationLoader()
            config = loader.load_pipeline_config(self.config_path)
        
        # Create and configure orchestrator
        self.orchestrator = PipelineOrchestrator(config, logger=self.logger)
        
        # Register default stage handlers if needed
        self._register_default_handlers()
        
        # Execute pipeline
        return await self.orchestrator.execute_pipeline(execution_id, resume_from_failure)
    
    def _register_default_handlers(self):
        """Register default handlers for common stage types."""
        if not self.orchestrator:
            return
        
        # These would be implemented to call the actual processing functions
        def means_handler(**context):
            """Default handler for means processing stages."""
            stage_config = context['stage_config']
            # This would call county_climate.means processing functions
            return {"status": "completed", "output_files": []}
        
        def metrics_handler(**context):
            """Default handler for metrics processing stages.""" 
            stage_config = context['stage_config']
            # This would call county_climate.metrics processing functions
            return {"status": "completed", "metrics": {}}
        
        self.orchestrator.register_stage_handler(ProcessingStage.MEANS, means_handler)
        self.orchestrator.register_stage_handler(ProcessingStage.METRICS, metrics_handler)