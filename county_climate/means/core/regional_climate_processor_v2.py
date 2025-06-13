#!/usr/bin/env python3
"""
Regional Climate Processor V2

Refactored version that properly parallelizes variables while processing years
sequentially to avoid NetCDF file conflicts.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from county_climate.means.core.parallel_variables_processor import ParallelVariablesProcessor
from county_climate.means.core.regions import REGION_BOUNDS
from county_climate.means.config import get_config, get_regional_output_dir


@dataclass
class RegionalProcessingConfigV2:
    """Configuration for regional climate processing V2."""
    region_key: str
    variables: List[str]
    input_data_dir: Path
    output_base_dir: Path
    
    # Processing settings
    max_workers: Optional[int] = None  # None means one worker per variable
    min_years_for_normal: int = 25
    
    # Progress tracking
    use_progress_tracking: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.region_key not in REGION_BOUNDS:
            raise ValueError(f"Invalid region key: {self.region_key}")
        
        # Convert paths to Path objects if needed
        self.input_data_dir = Path(self.input_data_dir)
        self.output_base_dir = Path(self.output_base_dir)


class RegionalClimateProcessorV2:
    """Refactored processor that parallelizes variables correctly."""
    
    def __init__(self, config: RegionalProcessingConfigV2):
        """Initialize the regional climate processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.region_key}")
        
    def process_all_scenarios(self) -> Dict:
        """Process all scenarios for the configured region and variables.
        
        Returns:
            Dictionary with results for all scenarios
        """
        self.logger.info(f"🚀 Starting regional processing for {self.config.region_key}")
        self.logger.info(f"📊 Variables: {self.config.variables}")
        
        start_time = time.time()
        results = {}
        
        # Process each scenario
        scenarios = ['historical', 'ssp245', 'ssp585']
        
        for scenario in scenarios:
            self.logger.info(f"📅 Processing {scenario} scenario")
            
            try:
                # Create processor for this scenario
                processor = ParallelVariablesProcessor(
                    region_key=self.config.region_key,
                    variables=self.config.variables,
                    scenario=scenario,
                    input_data_dir=self.config.input_data_dir,
                    output_base_dir=self.config.output_base_dir,
                    max_workers=self.config.max_workers,
                    min_years_for_normal=self.config.min_years_for_normal,
                    use_progress_tracking=self.config.use_progress_tracking
                )
                
                # Process all variables in parallel
                scenario_results = processor.process_all_variables()
                results[scenario] = scenario_results
                
            except Exception as e:
                self.logger.error(f"Error processing scenario {scenario}: {e}")
                results[scenario] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Summary
        self.logger.info(f"🎉 Regional processing completed in {total_time/60:.1f} minutes")
        
        return {
            'region': self.config.region_key,
            'processing_time_seconds': total_time,
            'scenarios': results
        }
    
    def process_single_scenario(self, scenario: str) -> Dict:
        """Process a single scenario for the configured region and variables.
        
        Args:
            scenario: Climate scenario to process
            
        Returns:
            Dictionary with results for the scenario
        """
        self.logger.info(f"📅 Processing {scenario} scenario for {self.config.region_key}")
        
        processor = ParallelVariablesProcessor(
            region_key=self.config.region_key,
            variables=self.config.variables,
            scenario=scenario,
            input_data_dir=self.config.input_data_dir,
            output_base_dir=self.config.output_base_dir,
            max_workers=self.config.max_workers,
            min_years_for_normal=self.config.min_years_for_normal,
            use_progress_tracking=self.config.use_progress_tracking
        )
        
        return processor.process_all_variables()


def create_regional_processor_v2(
    region_key: str,
    variables: Optional[List[str]] = None,
    use_progress_tracking: bool = True,
    **kwargs
) -> RegionalClimateProcessorV2:
    """Factory function to create a regional processor with configuration.
    
    Args:
        region_key: Region to process ('CONUS', 'AK', 'HI', 'PRVI', 'GU')
        variables: List of variables to process (default: all)
        use_progress_tracking: Whether to use Rich progress display
        **kwargs: Additional configuration options
        
    Returns:
        Configured RegionalClimateProcessorV2 instance
    """
    # Get global configuration
    global_config = get_config()
    
    # Default variables
    if variables is None:
        variables = ['pr', 'tas', 'tasmax', 'tasmin']
    
    config = RegionalProcessingConfigV2(
        region_key=region_key,
        variables=variables,
        input_data_dir=global_config.paths.input_data_dir,
        output_base_dir=get_regional_output_dir(region_key),
        use_progress_tracking=use_progress_tracking,
        **kwargs
    )
    
    return RegionalClimateProcessorV2(config)


def process_region_v2(
    region_key: str,
    variables: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    use_progress_tracking: bool = True,
    **kwargs
) -> Dict:
    """Convenience function to process a region with the new architecture.
    
    Args:
        region_key: Region to process
        variables: List of variables (default: all)
        scenarios: List of scenarios (default: all)
        use_progress_tracking: Whether to show progress
        **kwargs: Additional configuration
        
    Returns:
        Processing results dictionary
    """
    processor = create_regional_processor_v2(
        region_key, variables, use_progress_tracking, **kwargs
    )
    
    if scenarios and len(scenarios) == 1:
        # Process single scenario
        return processor.process_single_scenario(scenarios[0])
    else:
        # Process all scenarios
        return processor.process_all_scenarios()