"""
Wrapper for RegionalClimateProcessor that adds progress reporting capabilities.
"""

from typing import Dict, Optional, Any
import logging

from county_climate.means.core.regional_climate_processor import (
    RegionalClimateProcessor,
    RegionalProcessingConfig
)
from county_climate.means.utils.mp_progress import ProgressReporter


class ProgressAwareClimateProcessor:
    """
    Wrapper for RegionalClimateProcessor that integrates with progress reporting.
    
    This wrapper intercepts processing calls and provides progress updates
    through a ProgressReporter instance.
    """
    
    def __init__(self, 
                 config: RegionalProcessingConfig, 
                 progress_reporter: Optional[ProgressReporter] = None):
        """
        Initialize the processor wrapper.
        
        Args:
            config: Processing configuration
            progress_reporter: Optional progress reporter for updates
        """
        self.config = config
        self.progress_reporter = progress_reporter
        self.processor = RegionalClimateProcessor(config, use_rich_progress=False)
        self.logger = logging.getLogger(f"{__name__}.{config.region_key}")
        
    def process_all_variables(self) -> Dict[str, Any]:
        """
        Process all variables with progress reporting.
        
        This method wraps the original processor's method and adds
        progress reporting at key stages.
        """
        results = {}
        
        for variable in self.config.variables:
            # Create task for this variable
            task_id = f"process_{variable}_{self.config.region_key}"
            
            if self.progress_reporter:
                self.progress_reporter.create_task(
                    task_id,
                    f"Processing {variable.upper()} for {self.config.region_key}",
                    100  # We'll update based on progress
                )
            
            try:
                # Process the variable
                self.logger.info(f"Starting processing for {variable}")
                
                # Call the original processor
                # Note: In a real implementation, we'd need to modify the processor
                # to accept callbacks or use a different approach to get granular updates
                result = self.processor.process_variable_multiprocessing(variable)
                
                results[variable] = result
                
                # Mark as complete
                if self.progress_reporter:
                    self.progress_reporter.complete_task(task_id)
                    
            except Exception as e:
                self.logger.error(f"Error processing {variable}: {e}")
                results[variable] = {'status': 'error', 'error': str(e)}
                
                # Mark as failed
                if self.progress_reporter:
                    self.progress_reporter.fail_task(task_id, str(e))
                    
        return results


def create_progress_aware_processor(
    config: RegionalProcessingConfig,
    progress_reporter: Optional[ProgressReporter] = None
) -> ProgressAwareClimateProcessor:
    """
    Factory function to create a progress-aware processor.
    
    Args:
        config: Processing configuration
        progress_reporter: Optional progress reporter
        
    Returns:
        ProgressAwareClimateProcessor instance
    """
    return ProgressAwareClimateProcessor(config, progress_reporter)