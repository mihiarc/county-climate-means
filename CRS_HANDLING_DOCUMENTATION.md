# Coordinate Reference System (CRS) Handling Documentation

## Overview

This document provides comprehensive documentation on how Coordinate Reference Systems (CRS) are handled in the climate data processing system. The system processes climate data across multiple U.S. regions and handles coordinate transformations between different longitude/latitude systems.

## Architecture

### Core Components

The CRS handling is primarily implemented in three key modules:

1. **`regions.py`** - Contains CRS definitions, coordinate conversions, and regional extraction logic
2. **`climate_means.py`** - Uses CRS functionality for sequential climate data processing
3. **`climate_multiprocessing.py`** - Uses CRS functionality for parallel climate data processing

### Design Philosophy

The system is designed to:
- Handle multiple coordinate systems automatically (0-360° and -180/180° longitude)
- Support region-specific CRS definitions for accurate spatial processing
- Provide seamless coordinate conversion between different longitude conventions
- Work with both sequential and parallel processing workflows

## CRS Definitions and Regional Support

### Supported Regions

The system defines CRS information for five major U.S. regions:

```python
# From regions.py - get_region_crs_info()
SUPPORTED_REGIONS = {
    'CONUS': 'Continental United States',
    'AK': 'Alaska', 
    'HI': 'Hawaii and Islands',
    'PRVI': 'Puerto Rico and U.S. Virgin Islands',
    'GU': 'Guam and Northern Mariana Islands'
}
```

### CRS Specifications per Region

#### CONUS (Continental United States)
```python
'CONUS': {
    'crs_type': 'epsg',
    'crs_value': 5070,  # NAD83 / Conus Albers
    'central_longitude': -96,
    'central_latitude': 37.5,
    'extent': [-125, -65, 25, 50]  # West, East, South, North
}
```

#### Alaska (AK)
```python
'AK': {
    'crs_type': 'epsg',
    'crs_value': 3338,  # NAD83 / Alaska Albers
    'central_longitude': -154,
    'central_latitude': 50,
    'extent': [-170, -130, 50, 72]
}
```

#### Hawaii (HI)
```python
'HI': {
    'crs_type': 'proj4',
    'crs_value': "+proj=aea +lat_1=8 +lat_2=18 +lat_0=13 +lon_0=157 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs",
    'central_longitude': -157,
    'central_latitude': 20,
    'extent': [-178, -155, 18, 29]
}
```

#### Puerto Rico and Virgin Islands (PRVI)
```python
'PRVI': {
    'crs_type': 'epsg',
    'crs_value': 6566,  # NAD83(2011) / Puerto Rico and Virgin Islands
    'central_longitude': -66,
    'central_latitude': 18,
    'extent': [-68, -64, 17, 19]
}
```

#### Guam and Northern Mariana Islands (GU)
```python
'GU': {
    'crs_type': 'epsg',
    'crs_value': 32655,  # WGS 84 / UTM zone 55N
    'central_longitude': 147,
    'central_latitude': 13.5,
    'extent': [144, 147, 13, 21]
}
```

## Coordinate System Handling

### Longitude Convention Detection and Conversion

The system automatically detects and handles two longitude conventions:

#### 0-360° Longitude System
- Used by many global climate models
- Longitude values range from 0° to 360°
- Example: Los Angeles = 241.7°E

#### -180/180° Longitude System  
- Traditional geographic coordinate system
- Longitude values range from -180° to 180°
- Example: Los Angeles = -118.3°W

### Conversion Logic

```python
def convert_longitude_bounds(lon_min: float, lon_max: float, is_0_360: bool) -> Dict[str, float]:
    """Convert longitude bounds between 0-360 and -180-180 coordinate systems."""
    if is_0_360:
        return {'lon_min': lon_min, 'lon_max': lon_max}
    else:
        converted_min = lon_min - 360 if lon_min > 180 else lon_min
        converted_max = lon_max - 360 if lon_max > 180 else lon_max
        return {'lon_min': converted_min, 'lon_max': converted_max}
```

### Regional Boundary Definitions

Regional boundaries are defined in the 0-360° system by default:

```python
REGION_BOUNDS = {
    'CONUS': {
        'name': 'CONUS',
        'lon_min': 234,   # 234°E in 0-360 system (-126°E in -180/180)
        'lon_max': 294,   # 294°E in 0-360 system (-66°E in -180/180)
        'lat_min': 24.0,  # Extended south to fully cover Florida
        'lat_max': 50.0,  # Extended north to ensure coverage
        'convert_longitudes': True
    },
    # ... other regions
}
```

## Regional Data Extraction Process

### Core Extraction Function

The `extract_region()` function handles the complete coordinate transformation and data extraction workflow:

```python
def extract_region(ds: xr.Dataset, region_bounds: Dict) -> xr.Dataset:
    """Extract a specific region from the dataset with improved coordinate handling."""
    
    # Step 1: Detect coordinate names (lon/lat vs x/y)
    lon_name = 'lon' if 'lon' in ds.coords else 'x'
    lat_name = 'lat' if 'lat' in ds.coords else 'y'
    
    # Step 2: Analyze dataset coordinate system
    lon_min = ds[lon_name].min().item()
    lon_max = ds[lon_name].max().item()
    is_0_360 = lon_min >= 0 and lon_max > 180
    
    # Step 3: Convert region bounds to match dataset
    lon_bounds = convert_longitude_bounds(
        region_bounds['lon_min'], 
        region_bounds['lon_max'], 
        is_0_360
    )
    
    # Step 4: Handle dateline crossing cases
    if lon_bounds['lon_min'] > lon_bounds['lon_max']:
        # Special handling for regions crossing 0°/360° or -180°/180°
        region_ds = ds.where(
            ((ds[lon_name] >= lon_bounds['lon_min']) | 
             (ds[lon_name] <= lon_bounds['lon_max'])) & 
            (ds[lat_name] >= region_bounds['lat_min']) & 
            (ds[lat_name] <= region_bounds['lat_max']), 
            drop=True
        )
    else:
        # Standard rectangular region extraction
        region_ds = ds.where(
            (ds[lon_name] >= lon_bounds['lon_min']) & 
            (ds[lon_name] <= lon_bounds['lon_max']) & 
            (ds[lat_name] >= region_bounds['lat_min']) & 
            (ds[lat_name] <= region_bounds['lat_max']), 
            drop=True
        )
    
    return region_ds
```

### Coordinate System Detection Logic

```python
# Determine if we're using 0-360 or -180-180 coordinate system
lon_min = ds[lon_name].min().item()
lon_max = ds[lon_name].max().item()
is_0_360 = lon_min >= 0 and lon_max > 180
```

**Detection Rules:**
- If `lon_min >= 0` AND `lon_max > 180`: Detected as 0-360° system
- Otherwise: Detected as -180/180° system

## Integration with Processing Workflows

### Sequential Processing (climate_means.py)

The sequential processing workflow uses CRS handling in the `process_file()` function:

```python
def process_file(file_path: str, variable_name: str, region_key: str, file_handler):
    # ... file loading and validation ...
    
    # Extract region using CRS-aware extraction
    region_ds = extract_region(ds, REGION_BOUNDS[region_key])
    var = region_ds[variable_name]
    
    # ... climatology calculation ...
```

### Parallel Processing (climate_multiprocessing.py) 

The multiprocessing workflow uses identical CRS handling in the worker function:

```python
def process_single_file_worker(file_path: str, variable: str = 'pr', region: str = 'CONUS'):
    # ... dataset opening ...
    
    # Extract region using CRS-aware extraction  
    region_bounds = REGION_BOUNDS[region]
    region_ds = extract_region(ds, region_bounds)
    
    # ... processing continues ...
```

## Error Handling and Validation

### Region Validation

```python
def validate_region_bounds(region_key: str) -> bool:
    """Validate that a region key exists and has proper bounds."""
    
    # Check if region exists
    if region_key not in REGION_BOUNDS:
        logger.error(f"Unknown region key: {region_key}")
        return False
    
    # Validate required fields
    required_fields = ['name', 'lon_min', 'lon_max', 'lat_min', 'lat_max']
    missing_fields = [field for field in required_fields if field not in region]
    
    if missing_fields:
        logger.error(f"Region {region_key} missing required fields: {missing_fields}")
        return False
    
    # Validate coordinate bounds
    if region['lat_min'] >= region['lat_max']:
        logger.error(f"Region {region_key} has invalid latitude bounds")
        return False
    
    if region['lat_min'] < -90 or region['lat_max'] > 90:
        logger.error(f"Region {region_key} has latitude bounds outside valid range")
        return False
    
    return True
```

### Data Extraction Validation

The system logs warnings when no data is found within regional bounds:

```python
# Check if we have data
if region_ds[lon_name].size == 0 or region_ds[lat_name].size == 0:
    logger.warning(f"No data found within region bounds after filtering.")
    logger.warning(f"Dataset longitude range: {lon_min} to {lon_max}")
    logger.warning(f"Region bounds: {region_bounds['lon_min']} to {region_bounds['lon_max']} (original)")
    logger.warning(f"Converted bounds: {lon_bounds['lon_min']} to {lon_bounds['lon_max']}")
```

## Special Cases and Edge Conditions

### Dateline Crossing Regions

Some regions (like Alaska) cross the international dateline (180°/-180°). The system handles this by:

1. **Detection**: When `lon_bounds['lon_min'] > lon_bounds['lon_max']`
2. **Extraction**: Using OR logic instead of AND logic for longitude bounds
3. **Example**: Alaska spans from 170°E to 235°E (or -190°E to -125°E)

### Coordinate Name Variations

The system handles different coordinate naming conventions:

```python
# Check coordinate names
lon_name = 'lon' if 'lon' in ds.coords else 'x'
lat_name = 'lat' if 'lat' in ds.coords else 'y'
```

**Supported Patterns:**
- Standard: `lon`, `lat`
- Alternative: `x`, `y`

## Performance Considerations

### Memory Efficiency

- Regional extraction reduces dataset size before processing
- Coordinate conversion is done in-place when possible
- Bounds checking happens before data loading

### Processing Speed

- Coordinate detection is performed once per dataset
- Regional bounds are pre-calculated and cached
- Both sequential and parallel workflows use identical CRS logic

## Usage Examples

### Basic Regional Extraction

```python
from regions import REGION_BOUNDS, extract_region
import xarray as xr

# Load dataset
ds = xr.open_dataset('climate_data.nc')

# Extract CONUS region with automatic coordinate handling
conus_ds = extract_region(ds, REGION_BOUNDS['CONUS'])
```

### CRS Information Retrieval

```python
from regions import get_region_crs_info

# Get CRS information for CONUS
crs_info = get_region_crs_info('CONUS')
print(f"EPSG Code: {crs_info['crs_value']}")  # 5070
print(f"Central Longitude: {crs_info['central_longitude']}")  # -96
```

### Region Validation

```python
from regions import validate_region_bounds

# Validate region before processing
if validate_region_bounds('CONUS'):
    # Process the region
    pass
else:
    # Handle validation error
    pass
```

## Technical Dependencies

### Required Libraries

- **xarray**: Dataset handling and coordinate operations
- **numpy**: Numerical operations and array handling
- **logging**: Error reporting and debugging

### Coordinate System Libraries

While the system defines CRS information, it does not currently perform active coordinate transformations using libraries like:
- **Rasterio**: Not used for active CRS transformations
- **PyProj**: Referenced in Hawaii CRS definition but not actively used
- **Cartopy/Basemap**: Not used for projections

## Future Enhancements

### Potential Improvements

1. **Active CRS Transformations**: Implement actual coordinate transformations using PyProj
2. **More Regions**: Add support for additional regional definitions
3. **Projection Support**: Add support for projected coordinate systems
4. **Automatic CRS Detection**: Detect CRS from NetCDF metadata
5. **Spatial Indexing**: Implement spatial indexing for faster regional queries

### Current Limitations

1. **No Active Reprojection**: System works with geographic coordinates only
2. **Manual Region Definitions**: Regions must be manually defined
3. **Limited Validation**: Basic bounds checking but no geometric validation
4. **No CRS Metadata**: Does not read/write CRS information to output files

## Troubleshooting

### Common Issues

1. **No Data After Extraction**
   - Check longitude convention (0-360 vs -180/180)
   - Verify regional bounds are correct
   - Check for coordinate name variations

2. **Incorrect Regional Boundaries**
   - Verify region definitions in `REGION_BOUNDS`
   - Check for dateline crossing regions
   - Validate latitude bounds (-90 to 90)

3. **Coordinate Detection Failures**
   - Ensure datasets have `lon`/`lat` or `x`/`y` coordinates
   - Check for unusual coordinate naming patterns

### Debug Information

The system provides detailed logging for troubleshooting:

```python
logger.warning(f"Dataset longitude range: {lon_min} to {lon_max}")
logger.warning(f"Region bounds: {region_bounds['lon_min']} to {region_bounds['lon_max']} (original)")
logger.warning(f"Converted bounds: {lon_bounds['lon_min']} to {lon_bounds['lon_max']}")
```

## Conclusion

The CRS handling system provides robust coordinate system detection and regional data extraction capabilities. While it doesn't perform active coordinate transformations, it effectively handles the most common geographic coordinate conventions and regional boundary definitions needed for climate data processing across U.S. territories.

The system's strength lies in its automatic detection of coordinate systems and seamless integration with both sequential and parallel processing workflows, making it suitable for large-scale climate data processing operations. 