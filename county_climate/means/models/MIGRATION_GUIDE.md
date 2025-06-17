# Migration Guide: From NorESM2FileHandler to FlexibleClimateFileHandler

This guide explains how to migrate from the hardcoded `NorESM2FileHandler` to the new flexible climate model system.

## Overview

The new system provides:
- Support for multiple Global Climate Models (GCMs)
- Flexible scenario configuration (not limited to ssp245/ssp585)
- Proper hybrid period handling with configurable projections
- Easy extensibility for new climate models

## Key Changes

### 1. Import Changes

**Before:**
```python
from county_climate.means.utils.io_util import NorESM2FileHandler
```

**After:**
```python
from county_climate.means.utils.flexible_io_util import FlexibleClimateFileHandler
```

### 2. Handler Initialization

**Before:**
```python
file_handler = NorESM2FileHandler(data_directory)
```

**After:**
```python
# For NorESM2-LM (default)
file_handler = FlexibleClimateFileHandler(data_directory, model_id="NorESM2-LM")

# For other models
file_handler = FlexibleClimateFileHandler(data_directory, model_id="GFDL-ESM4")
```

### 3. Configuration Updates

Add model configuration to your config files:

```yaml
# climate_config.yaml
climate_model:
  model_id: "NorESM2-LM"  # or "GFDL-ESM4", etc.
  # Optional: override default scenarios
  scenarios:
    - historical
    - ssp126
    - ssp245
    - ssp370
    - ssp585
```

### 4. Hybrid Period Processing

**Before (hardcoded to ssp245):**
```python
# In regional_climate_processor.py line 492
ssp_files = file_handler.get_files_for_period(variable, 'ssp245', ssp_start, ssp_end)
```

**After (configurable):**
```python
# Determine which SSP scenario to use based on configuration
projection_scenario = self.config.projection_scenario  # e.g., 'ssp585'
ssp_files = file_handler.get_files_for_period(variable, projection_scenario, ssp_start, ssp_end)
```

### 5. Update Processing Configuration

Update `RegionalProcessingConfig` to include model information:

```python
@dataclass
class RegionalProcessingConfig:
    region_key: str
    variables: List[str]
    input_data_dir: Path
    output_base_dir: Path
    
    # New fields
    model_id: str = "NorESM2-LM"
    projection_scenario: str = "ssp245"  # For hybrid periods
    
    # ... rest of config
```

## Example Migration

### Original Code
```python
def process_climate_data(data_dir):
    file_handler = NorESM2FileHandler(data_dir)
    
    # Get files for SSP245 (hardcoded)
    files = file_handler.get_files_for_period('tas', 'ssp245', 2015, 2100)
    
    # Process files...
```

### Migrated Code
```python
def process_climate_data(data_dir, model_id="NorESM2-LM", scenario="ssp245"):
    file_handler = FlexibleClimateFileHandler(data_dir, model_id=model_id)
    
    # Check if scenario is supported
    if scenario not in file_handler.get_supported_scenarios():
        raise ValueError(f"Scenario {scenario} not supported by {model_id}")
    
    # Get files for any scenario
    files = file_handler.get_files_for_period('tas', scenario, 2015, 2100)
    
    # Process files...
```

## Adding Support for New Climate Models

To add a new climate model:

1. Create a new handler class:
```python
# county_climate/means/models/my_model.py
from county_climate.means.models.base import ClimateModelHandler, ModelConfig, ScenarioConfig

class MyModelHandler(ClimateModelHandler):
    def __init__(self, base_path: Path):
        config = ModelConfig(
            model_name="My Climate Model",
            model_id="MY-MODEL",
            institution="My Institute",
            # ... configure for your model
        )
        super().__init__(base_path, config)
    
    # Implement required methods...
```

2. Register the model:
```python
from county_climate.means.models import register_model
from my_model import MyModelHandler

register_model("MY-MODEL", MyModelHandler)
```

3. Use it:
```python
handler = FlexibleClimateFileHandler(data_dir, model_id="MY-MODEL")
```

## Benefits of Migration

1. **Multi-model Support**: Process data from different GCMs without code changes
2. **Flexible Scenarios**: Support all SSP scenarios (ssp126, ssp370, etc.)
3. **Correct Hybrid Periods**: Use appropriate projection scenario for hybrid calculations
4. **Future-proof**: Easy to add new models as they become available
5. **Configuration-driven**: Change models/scenarios via configuration, not code

## Testing Your Migration

Run the test script to verify your migration:
```bash
python test_flexible_models.py
```

This will test:
- Model registration and discovery
- File handling for different models
- Hybrid period processing with different scenarios
- Data availability checking