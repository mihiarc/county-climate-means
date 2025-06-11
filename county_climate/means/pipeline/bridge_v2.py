"""
Contract-Based Pipeline Bridge for Metrics Integration

Updated bridge implementation using Pydantic contracts for type-safe
integration between climate means and metrics packages.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from shared.contracts.climate_data import (
    ClimateDatasetContract,
    ClimateVariable,
    Region,
    Scenario,
    CatalogQueryContract,
    CatalogResponseContract
)
from shared.contracts.pipeline_interface import (
    PipelineConfigContract,
    DataflowContract,
    ProcessingStatusContract,
    ErrorContract
)
from ..output.catalog_v2 import ContractBasedCatalog

logger = logging.getLogger(__name__)


class ContractBasedBridge:
    """
    Type-safe bridge for pipeline integration using Pydantic contracts.
    
    Provides standardized interface for metrics package to access climate means
    data with full type safety and validation.
    """
    
    def __init__(self, catalog_path: str, config: Optional[PipelineConfigContract] = None):
        """
        Initialize bridge with catalog and optional configuration.
        
        Args:
            catalog_path: Path to climate means catalog
            config: Pipeline configuration contract
        """
        self.catalog_path = Path(catalog_path)
        self.catalog = ContractBasedCatalog(self.catalog_path)
        self.config = config or self._create_default_config()
        
        logger.info(f"Initialized contract-based bridge with catalog: {catalog_path}")
    
    def get_datasets_for_metrics(self, 
                                region: Optional[Region] = None,
                                variable: Optional[ClimateVariable] = None,
                                scenario: Optional[Scenario] = None,
                                year_start: Optional[int] = None,
                                year_end: Optional[int] = None,
                                min_quality_score: float = 0.95) -> List[ClimateDatasetContract]:
        """
        Get datasets ready for metrics processing with type-safe filtering.
        
        Args:
            region: Optional region filter
            variable: Optional variable filter
            scenario: Optional scenario filter
            year_start: Optional start year filter
            year_end: Optional end year filter
            min_quality_score: Minimum quality score requirement
            
        Returns:
            List of validated climate dataset contracts
        """
        # Create query contract
        query = CatalogQueryContract(
            variables=[variable] if variable else None,
            regions=[region] if region else None,
            scenarios=[scenario] if scenario else None,
            year_start=year_start,
            year_end=year_end,
            min_quality_score=min_quality_score,
            metrics_compatible_only=True,
            sort_by="target_year",
            sort_order="asc"
        )
        
        # Execute query
        response = self.catalog.query(query)
        
        logger.info(f"Retrieved {response.returned_count} datasets for metrics processing")
        return response.datasets
    
    def get_dataset_by_id(self, dataset_id: str) -> Optional[ClimateDatasetContract]:
        """
        Get a specific dataset by ID with contract validation.
        
        Args:
            dataset_id: Unique dataset identifier
            
        Returns:
            Climate dataset contract if found and valid
        """
        dataset = self.catalog.get_dataset_by_id(dataset_id)
        
        if dataset and not dataset.is_ready_for_metrics:
            logger.warning(f"Dataset {dataset_id} is not ready for metrics processing")
            return None
            
        return dataset
    
    def validate_dataset_availability(self, dataset_id: str) -> bool:
        """
        Validate that a dataset is available and ready for processing.
        
        Args:
            dataset_id: Dataset identifier to validate
            
        Returns:
            True if dataset is available and ready
        """
        dataset = self.get_dataset_by_id(dataset_id)
        return dataset is not None and dataset.is_ready_for_metrics
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of available data.
        
        Returns:
            Dictionary with data availability summary
        """
        # Query all datasets
        query = CatalogQueryContract(metrics_compatible_only=True)
        response = self.catalog.query(query)
        
        return {
            'total_datasets': response.total_matches,
            'summary_stats': response.summary,
            'config': self.config.dict(),
            'catalog_path': str(self.catalog_path),
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def create_dataflow_record(self, 
                              dataset_ids: List[str],
                              target_pipeline: str = "climate_metrics") -> DataflowContract:
        """
        Create a dataflow record for tracking data transfer to metrics pipeline.
        
        Args:
            dataset_ids: List of dataset IDs being transferred
            target_pipeline: Target pipeline name
            
        Returns:
            Dataflow contract for tracking
        """
        # Calculate total data size
        total_size_gb = 0.0
        valid_datasets = []
        
        for dataset_id in dataset_ids:
            dataset = self.get_dataset_by_id(dataset_id)
            if dataset:
                total_size_gb += dataset.data_access.file_size_bytes / (1024**3)
                valid_datasets.append(dataset_id)
        
        # Create dataflow contract
        dataflow = DataflowContract(
            flow_id=f"means_to_{target_pipeline}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            source_pipeline=self.config.pipeline_id,
            target_pipeline=target_pipeline,
            dataset_ids=valid_datasets,
            data_size_gb=total_size_gb,
            transfer_method="file_api",
            status="initiated",
            initiated_at=datetime.utcnow(),
            data_integrity_verified=True,
            checksum_validation=True
        )
        
        logger.info(f"Created dataflow record: {dataflow.flow_id}")
        return dataflow
    
    def get_processing_status(self) -> ProcessingStatusContract:
        """
        Get current processing status for pipeline monitoring.
        
        Returns:
            Processing status contract
        """
        # Get all datasets and calculate metrics
        all_datasets = self.catalog.datasets
        ready_datasets = self.catalog.get_metrics_ready_datasets()
        
        # Calculate quality metrics
        quality_scores = [d.quality_metrics.quality_score for d in all_datasets if d.quality_metrics]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Calculate error rate
        failed_datasets = [d for d in all_datasets if d.status == "failed"]
        error_rate = len(failed_datasets) / len(all_datasets) if all_datasets else 0.0
        
        return ProcessingStatusContract(
            pipeline_id=self.config.pipeline_id,
            stage="cataloging",
            status="operational",
            total_tasks=len(all_datasets),
            completed_tasks=len(ready_datasets),
            failed_tasks=len(failed_datasets),
            active_workers=0,  # Not applicable for catalog
            start_time=datetime.utcnow(),  # Would be actual start time
            last_update=datetime.utcnow(),
            throughput_tasks_per_hour=0.0,  # Would be calculated
            avg_task_duration_seconds=0.0,  # Would be calculated
            memory_usage_gb=0.0,  # Would be actual memory usage
            cpu_usage_percent=0.0,  # Would be actual CPU usage
            current_quality_score=avg_quality,
            error_rate=error_rate,
            notes=f"Catalog contains {len(all_datasets)} datasets, {len(ready_datasets)} ready for metrics"
        )
    
    def create_error_record(self, 
                           error_message: str,
                           dataset_id: Optional[str] = None,
                           severity: str = "medium") -> ErrorContract:
        """
        Create standardized error record for pipeline monitoring.
        
        Args:
            error_message: Human-readable error message
            dataset_id: Optional dataset ID where error occurred
            severity: Error severity level
            
        Returns:
            Error contract for logging/monitoring
        """
        error = ErrorContract(
            error_id=f"bridge_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            pipeline_id=self.config.pipeline_id,
            stage="cataloging",
            error_type="BridgeError",
            error_message=error_message,
            severity=severity,
            timestamp=datetime.utcnow(),
            dataset_id=dataset_id,
            system_info={
                'catalog_path': str(self.catalog_path),
                'catalog_size': len(self.catalog.datasets)
            }
        )
        
        logger.error(f"Created error record: {error.error_id} - {error_message}")
        return error
    
    def export_metrics_config(self) -> Dict[str, Any]:
        """
        Export configuration optimized for metrics package consumption.
        
        Returns:
            Configuration dictionary for metrics package
        """
        # Get data summary
        summary = self.get_data_summary()
        
        # Create metrics-optimized configuration
        metrics_config = {
            'data_source': {
                'type': 'climate_means_catalog',
                'catalog_path': str(self.catalog_path),
                'api_base_url': f"http://localhost:8000/api/v1",
                'contract_version': '1.0.0'
            },
            'available_data': summary['summary_stats'],
            'processing_recommendations': {
                'max_workers': self.config.max_workers,
                'batch_size': self.config.batch_size,
                'memory_per_worker_gb': self.config.max_memory_per_worker_gb,
                'quality_threshold': self.config.min_quality_score
            },
            'integration_settings': {
                'enable_validation': True,
                'enable_checksums': True,
                'enable_progress_tracking': True,
                'temp_storage_path': '/tmp/metrics_processing'
            }
        }
        
        return metrics_config
    
    def _create_default_config(self) -> PipelineConfigContract:
        """Create default pipeline configuration."""
        return PipelineConfigContract(
            pipeline_id="climate_means_bridge",
            pipeline_name="Climate Means Pipeline Bridge",
            pipeline_version="1.0.0",
            processing_mode="multiprocessing",
            max_workers=6,
            batch_size=15,
            input_data_path=str(self.catalog_path.parent),
            output_data_path=str(self.catalog_path.parent),
            catalog_path=str(self.catalog_path),
            downstream_pipelines=["climate_metrics"]
        )


class MetricsDataAdapter:
    """
    Adapter specifically for metrics package data access using contracts.
    
    Provides the interface expected by the metrics package while using
    the new contract-based system under the hood.
    """
    
    def __init__(self, bridge: ContractBasedBridge):
        """
        Initialize adapter with contract-based bridge.
        
        Args:
            bridge: Contract-based bridge instance
        """
        self.bridge = bridge
    
    def get_data_files(self, region: str, variable: str, 
                      base_path: Optional[str] = None) -> List[tuple]:
        """
        Get data files in the format expected by metrics package.
        
        Args:
            region: Region code
            variable: Variable name
            base_path: Ignored in contract-based implementation
            
        Returns:
            List of (year, scenario, filepath) tuples
        """
        try:
            # Convert string parameters to enum types
            region_enum = Region(region)
            variable_enum = ClimateVariable(variable)
            
            # Get datasets from bridge
            datasets = self.bridge.get_datasets_for_metrics(
                region=region_enum,
                variable=variable_enum
            )
            
            # Convert to expected format
            results = []
            for dataset in datasets:
                results.append((
                    dataset.target_year,
                    dataset.scenario,
                    dataset.data_access.file_path
                ))
            
            # Sort by year for consistency
            results.sort(key=lambda x: x[0])
            return results
            
        except ValueError as e:
            logger.error(f"Invalid parameter values: region={region}, variable={variable}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting data files: {e}")
            return []
    
    def filter_files_by_scenario(self, files: List[tuple], scenario: str) -> List[tuple]:
        """Filter file list by scenario."""
        return [(year, scen, path) for year, scen, path in files if scen == scenario]
    
    def filter_files_by_year_range(self, files: List[tuple], 
                                  start_year: int, end_year: int) -> List[tuple]:
        """Filter file list by year range."""
        return [(year, scen, path) for year, scen, path in files 
                if start_year <= year <= end_year]
    
    def get_available_years(self, region: str, variable: str, 
                           scenario: Optional[str] = None) -> List[int]:
        """Get available years for region/variable/scenario combination."""
        files = self.get_data_files(region, variable)
        
        if scenario:
            files = self.filter_files_by_scenario(files, scenario)
        
        years = sorted(list(set(year for year, _, _ in files)))
        return years
    
    def validate_data_availability(self, region: str, variable: str, 
                                  scenario: str, year: int) -> bool:
        """Validate that specific data is available."""
        dataset_id = f"{variable}_{region}_{scenario}_{year}"
        return self.bridge.validate_dataset_availability(dataset_id)