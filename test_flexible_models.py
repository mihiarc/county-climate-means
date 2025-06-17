#!/usr/bin/env python3
"""
Test script to demonstrate flexible climate model support.

Shows how the new abstract base class system allows easy support for multiple GCMs.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from county_climate.means.models import (
    get_model_handler, 
    register_model,
    list_available_models,
    ModelConfig,
    ScenarioConfig
)
from county_climate.means.models.gfdl import GFDLESM4Handler
from county_climate.means.utils.flexible_io_util import FlexibleClimateFileHandler


def test_model_registry():
    """Test the model registry functionality."""
    print("=== Testing Model Registry ===")
    
    # Register GFDL model
    register_model("GFDL-ESM4", GFDLESM4Handler)
    
    # List available models
    models = list_available_models()
    print(f"Available models: {models}")
    print()


def test_noresm2_handler():
    """Test NorESM2-LM handler."""
    print("=== Testing NorESM2-LM Handler ===")
    
    # Create handler for NorESM2-LM
    base_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
    
    try:
        handler = get_model_handler("NorESM2-LM", base_path)
        print(f"Created handler: {handler}")
        print(f"Model name: {handler.config.model_name}")
        print(f"Institution: {handler.config.institution}")
        print(f"Supported scenarios: {handler.get_supported_scenarios()}")
        
        # Test filename parsing
        test_filename = "pr_day_NorESM2-LM_historical_r1i1p1f1_gn_1985.nc"
        year = handler.extract_year_from_filename(test_filename)
        print(f"\nExtracted year from '{test_filename}': {year}")
        
        metadata = handler.extract_metadata_from_filename(test_filename)
        print(f"Extracted metadata: {metadata}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print()


def test_flexible_file_handler():
    """Test flexible file handler with different models."""
    print("=== Testing Flexible File Handler ===")
    
    # Test with NorESM2-LM (default)
    base_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
    
    try:
        handler = FlexibleClimateFileHandler(base_path, model_id="NorESM2-LM")
        print(f"Created flexible handler for {handler.model_id}")
        print(f"Model: {handler.model_name}")
        print(f"Institution: {handler.institution}")
        print(f"Supported scenarios: {handler.get_supported_scenarios()}")
        
        # Test data availability
        print("\nChecking data availability...")
        availability = handler.validate_data_availability()
        
        for variable, scenarios in availability.items():
            print(f"\n{variable}:")
            for scenario, (start_year, end_year) in scenarios.items():
                print(f"  {scenario}: {start_year}-{end_year}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print()


def test_hybrid_period_flexibility():
    """Test flexible hybrid period handling."""
    print("=== Testing Hybrid Period Flexibility ===")
    
    base_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
    
    try:
        handler = FlexibleClimateFileHandler(base_path, model_id="NorESM2-LM")
        
        # Test hybrid period with SSP245 (default)
        print("\nTesting hybrid period for 2020 with default projection:")
        files, counts = handler.get_hybrid_files_for_period("tas", 2020)
        print(f"Scenario counts: {counts}")
        print(f"Total files: {len(files)}")
        
        # Test hybrid period with SSP585 (explicit)
        print("\nTesting hybrid period for 2020 with SSP585:")
        files, counts = handler.get_hybrid_files_for_period("tas", 2020, projection_scenario="ssp585")
        print(f"Scenario counts: {counts}")
        print(f"Total files: {len(files)}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print()


def demonstrate_custom_model():
    """Demonstrate how to add a custom climate model."""
    print("=== Custom Model Configuration Example ===")
    
    # Create a custom model configuration
    custom_config = ModelConfig(
        model_name="Example Climate Model",
        model_id="EXAMPLE-CM",
        institution="Example Institute",
        ensemble_member="r1i1p1f1",
        grid_label="gr",
        variable_dir_pattern="{variable}/{scenario}",
        filename_pattern="{variable}_day_{model_id}_{scenario}_{ensemble}_{grid}_{year}.nc",
        scenarios={
            "historical": ScenarioConfig(
                name="historical",
                start_year=1950,
                end_year=2014,
                description="Historical simulation"
            ),
            "future-low": ScenarioConfig(
                name="future-low",
                start_year=2015,
                end_year=2100,
                description="Low emissions future",
                parent_scenario="historical"
            ),
            "future-high": ScenarioConfig(
                name="future-high",
                start_year=2015,
                end_year=2100,
                description="High emissions future",
                parent_scenario="historical"
            )
        }
    )
    
    print(f"Custom model: {custom_config.model_id}")
    print(f"Scenarios: {list(custom_config.scenarios.keys())}")
    print("\nThis configuration can be used to create a new handler class")
    print("that extends ClimateModelHandler for any new climate model.")


if __name__ == "__main__":
    print("Climate Model Flexibility Test\n")
    
    test_model_registry()
    test_noresm2_handler()
    test_flexible_file_handler()
    test_hybrid_period_flexibility()
    demonstrate_custom_model()
    
    print("\n✅ Test completed!")
    print("\nThe new system provides:")
    print("- Abstract base class for any climate model")
    print("- Model-specific handlers with custom logic")
    print("- Flexible scenario configuration")
    print("- Proper hybrid period handling with configurable projections")
    print("- Easy extensibility for new GCMs")