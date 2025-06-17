#!/usr/bin/env python3
"""
Flexible configuration for regional climate processing.

Supports multiple climate models and configurable scenarios.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple


@dataclass
class ScenarioProcessingConfig:
    """Configuration for processing a specific scenario."""
    scenario_name: str
    year_range: Tuple[int, int]
    process_as_hybrid: bool = False
    hybrid_historical_end: Optional[int] = None  # For hybrid periods
    
    def get_years(self) -> List[int]:
        """Get list of years to process."""
        return list(range(self.year_range[0], self.year_range[1] + 1))


@dataclass
class FlexibleRegionalProcessingConfig:
    """
    Flexible configuration for regional climate processing.
    
    Supports multiple climate models and configurable scenarios.
    """
    # Core configuration
    region_key: str
    variables: List[str]
    input_data_dir: Path
    output_base_dir: Path
    
    # Climate model configuration
    model_id: str = "NorESM2-LM"
    
    # Scenario configuration - flexible instead of hardcoded
    scenarios_to_process: List[ScenarioProcessingConfig] = field(default_factory=list)
    
    # Processing settings
    max_cores: int = 6
    cores_per_variable: int = 2
    batch_size_years: int = 2
    max_memory_per_process_gb: int = 4
    memory_check_interval: int = 10
    min_years_for_normal: int = 25
    climate_normal_window: int = 30
    
    # Progress tracking
    status_update_interval: int = 30
    enable_rich_progress: bool = True
    
    def __post_init__(self):
        """Validate configuration and set defaults."""
        from county_climate.means.core.regions import REGION_BOUNDS, validate_region_bounds
        
        if self.region_key not in REGION_BOUNDS:
            raise ValueError(f"Invalid region key: {self.region_key}")
        
        if not validate_region_bounds(self.region_key):
            raise ValueError(f"Invalid region bounds for: {self.region_key}")
        
        # Convert paths to Path objects
        self.input_data_dir = Path(self.input_data_dir)
        self.output_base_dir = Path(self.output_base_dir)
        
        # Set up progress tracking files
        self.progress_status_file = f"{self.region_key.lower()}_processing_progress.json"
        self.progress_log_file = f"{self.region_key.lower()}_processing_progress.log"
        self.main_log_file = f"{self.region_key.lower()}_climate_processing.log"
        
        # If no scenarios specified, use default
        if not self.scenarios_to_process:
            self.scenarios_to_process = self._get_default_scenarios()
    
    def _get_default_scenarios(self) -> List[ScenarioProcessingConfig]:
        """Get default scenario configuration (backward compatible)."""
        return [
            ScenarioProcessingConfig(
                scenario_name="historical",
                year_range=(1980, 2014),
                process_as_hybrid=False
            ),
            ScenarioProcessingConfig(
                scenario_name="ssp245",  # Default projection
                year_range=(2015, 2044),
                process_as_hybrid=True,
                hybrid_historical_end=2014
            ),
            ScenarioProcessingConfig(
                scenario_name="ssp245",
                year_range=(2045, 2100),
                process_as_hybrid=False
            )
        ]
    
    @classmethod
    def for_scenario(cls, region_key: str, variables: List[str], 
                    input_data_dir: Path, output_base_dir: Path,
                    scenario: str, **kwargs) -> 'FlexibleRegionalProcessingConfig':
        """
        Create configuration for a specific scenario.
        
        Args:
            region_key: Region to process
            variables: Variables to process
            input_data_dir: Input data directory
            output_base_dir: Output directory
            scenario: Scenario name (e.g., 'ssp585', 'ssp245')
            **kwargs: Additional configuration parameters
        """
        # Define scenario configurations
        if scenario == "ssp585":
            scenarios = [
                ScenarioProcessingConfig(
                    scenario_name="historical",
                    year_range=(1980, 2014),
                    process_as_hybrid=False
                ),
                ScenarioProcessingConfig(
                    scenario_name="ssp585",  # Use SSP585 for hybrid
                    year_range=(2015, 2044),
                    process_as_hybrid=True,
                    hybrid_historical_end=2014
                ),
                ScenarioProcessingConfig(
                    scenario_name="ssp585",
                    year_range=(2045, 2100),
                    process_as_hybrid=False
                )
            ]
        elif scenario == "ssp245":
            scenarios = [
                ScenarioProcessingConfig(
                    scenario_name="historical",
                    year_range=(1980, 2014),
                    process_as_hybrid=False
                ),
                ScenarioProcessingConfig(
                    scenario_name="ssp245",
                    year_range=(2015, 2044),
                    process_as_hybrid=True,
                    hybrid_historical_end=2014
                ),
                ScenarioProcessingConfig(
                    scenario_name="ssp245",
                    year_range=(2045, 2100),
                    process_as_hybrid=False
                )
            ]
        elif scenario == "historical":
            scenarios = [
                ScenarioProcessingConfig(
                    scenario_name="historical",
                    year_range=(1980, 2014),
                    process_as_hybrid=False
                )
            ]
        else:
            # Generic scenario - process full range without hybrid
            scenarios = [
                ScenarioProcessingConfig(
                    scenario_name=scenario,
                    year_range=(2015, 2100),
                    process_as_hybrid=False
                )
            ]
        
        return cls(
            region_key=region_key,
            variables=variables,
            input_data_dir=input_data_dir,
            output_base_dir=output_base_dir,
            scenarios_to_process=scenarios,
            **kwargs
        )
    
    def get_output_period_type(self, scenario_config: ScenarioProcessingConfig) -> str:
        """
        Get the output period type for a scenario configuration.
        
        This determines the output directory name.
        """
        if scenario_config.process_as_hybrid:
            return "hybrid"
        elif scenario_config.scenario_name == "historical":
            return "historical"
        else:
            # For projections, use the scenario name
            return scenario_config.scenario_name