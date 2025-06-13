#!/usr/bin/env python3
"""Test processing a single file to debug the error."""

from pathlib import Path
from county_climate.means.core.regional_climate_processor import RegionalClimateProcessor, RegionalProcessingConfig

# Create minimal config
config = RegionalProcessingConfig(
    region_key="CONUS",
    variables=["tas"],
    input_data_dir=Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"),
    output_base_dir=Path("/tmp/test_output"),
    max_cores=2,
    cores_per_variable=1,
    batch_size_years=1
)

# Create processor
processor = RegionalClimateProcessor(config, use_rich_progress=False)

# Test processing a single file
test_file = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM/tas/historical/tas_day_NorESM2-LM_historical_r1i1p1f1_gn_2000.nc")

if test_file.exists():
    print(f"Processing {test_file}")
    try:
        year, climatology = processor.process_single_file_for_climatology_safe(test_file, "tas")
        if year:
            print(f"Successfully processed year {year}")
            print(f"Climatology shape: {climatology.shape if climatology is not None else 'None'}")
        else:
            print("Processing returned None")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"Test file not found: {test_file}")