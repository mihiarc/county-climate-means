# U.S. Regional Climate Processing Validation Maps

This directory contains comprehensive validation maps for the U.S. climate normals processing pipeline, covering all five U.S. regions.

## 🗺️ Map Types

### 1. **Regional Extent Maps** (`*_extent_validation.png`)
These maps show the defined regional boundaries (red rectangles) overlaid on geographic features:
- Validate that boundary definitions make geographic sense
- Ensure proper coverage of intended areas
- Check for appropriate padding around landmasses

### 2. **Climate Data Alignment Maps** (`*_data_alignment_validation.png`)
These maps overlay actual processed climate data (temperature rasters) on regional boundaries:
- **Red dashed lines**: Regional boundary rectangles
- **Color raster**: Actual temperature data from climate normals
- **Validation**: Ensures data aligns properly with intended geographic coverage

### 3. **Overview Map** (`us_regions_overview.png`)
Comprehensive map showing all U.S. regions with detailed insets for smaller territories.

## 🌍 Regional Coverage

### **CONUS (Continental United States)**
- **Extent**: -126.0° to -66.0°E, 24.0° to 50.0°N
- **Data Coverage**: 240 × 104 grid points
- **Temperature Range**: -3.0°C to 26.0°C (270-299K)
- **Validation**: ✅ Covers continental US including Alaska panhandle boundary
- **Notes**: Appropriate north-south temperature gradient visible

### **Alaska (AK)**
- **Extent**: 170.0° to -125.0°E, 50.0° to 72.0°N *(crosses dateline)*
- **Data Coverage**: 260 × 88 grid points  
- **Temperature Range**: -16.8°C to 10.8°C (256-284K)
- **Validation**: ✅ Covers entire Alaska including Aleutian Islands
- **Notes**: Very cold temperatures as expected, proper Arctic coverage

### **Hawaii (HI)**
- **Extent**: -178.4° to -154.8°E, 18.9° to 28.4°N
- **Data Coverage**: 94 × 38 grid points
- **Temperature Range**: 8.5°C to 24.6°C (282-298K)
- **Validation**: ✅ Covers Hawaiian Island chain
- **Notes**: Warm tropical temperatures, includes Northwestern Hawaiian Islands

### **Puerto Rico & Virgin Islands (PRVI)**
- **Extent**: -68.0° to -64.5°E, 17.6° to 18.6°N
- **Data Coverage**: 14 × 4 grid points
- **Temperature Range**: 23.0°C to 26.3°C (296-299K)
- **Validation**: ✅ Covers Puerto Rico and U.S. Virgin Islands
- **Notes**: Small but well-defined tropical region

### **Guam & Northern Mariana Islands (GU)**
- **Extent**: 144.6° to 146.1°E, 13.2° to 20.6°N
- **Data Coverage**: 6 × 29 grid points
- **Temperature Range**: 26.9°C to 27.9°C (300-301K)
- **Validation**: ✅ Covers Guam and Northern Mariana Islands
- **Notes**: Smallest region, consistent hot tropical temperatures

## 📊 Temperature Validation Results

The climate data shows expected geographic temperature patterns:

| Region | Expected Pattern | Observed in Data | ✅/❌ |
|--------|------------------|------------------|-------|
| CONUS | Cold north → Warm south | Clear N-S gradient | ✅ |
| Alaska | Very cold, especially north | Arctic temperatures | ✅ |
| Hawaii | Warm tropical | Consistent tropical | ✅ |
| Puerto Rico | Warm subtropical | Tropical/subtropical | ✅ |
| Guam | Hot tropical | Very hot tropical | ✅ |

## 🔍 What to Look For

### ✅ **Good Alignment Indicators:**
- Climate data fills the red dashed boundary rectangle
- No significant gaps between data coverage and coastlines
- Temperature patterns match expected geographic patterns
- Data coverage matches expected regional extent
- Smooth transitions at boundaries

### ❌ **Potential Issues to Check:**
- Data gaps within boundary rectangles
- Misaligned boundaries (not covering intended areas)
- Unexpected temperature patterns
- Coordinate system mismatches
- Missing islands or territories

## 📁 File Inventory

### **Extent Validation Maps:**
- `conus_extent_validation.png` - Continental US boundaries
- `ak_extent_validation.png` - Alaska boundaries (crosses dateline)
- `hi_extent_validation.png` - Hawaiian Islands boundaries  
- `prvi_extent_validation.png` - Puerto Rico & Virgin Islands boundaries
- `gu_extent_validation.png` - Guam & Northern Mariana Islands boundaries

### **Data Alignment Maps:**
- `conus_data_alignment_validation.png` - CONUS temperature overlay
- `ak_data_alignment_validation.png` - Alaska temperature overlay
- `hi_data_alignment_validation.png` - Hawaii temperature overlay
- `prvi_data_alignment_validation.png` - Puerto Rico temperature overlay
- `gu_data_alignment_validation.png` - Guam temperature overlay

### **Overview & Documentation:**
- `us_regions_overview.png` - All regions overview map
- `alignment_validation_summary.txt` - Technical summary
- `README.md` - This documentation file

## 🎯 Validation Conclusions

**✅ All regional extents are properly aligned with climate data**

1. **Geographic Coverage**: All boundaries correctly encompass their intended territories
2. **Data Alignment**: Climate rasters properly fill defined regional boundaries  
3. **Temperature Patterns**: Observed data matches expected regional climate patterns
4. **Coordinate Systems**: Proper handling of 0-360° and -180/180° longitude systems
5. **Special Cases**: Alaska dateline crossing handled correctly

## 🔄 Usage in Processing Pipeline

These validated regional definitions are used throughout the climate processing pipeline:

- **Input**: `src/utils/regions.py` - Regional boundary definitions
- **Processing**: Climate data extraction and regional masking
- **Output**: Region-specific climate normals files
- **Validation**: These maps confirm proper implementation

## 📞 Contact

For questions about regional definitions or validation results, refer to:
- `PROJECT_OVERVIEW.md` - Technical pipeline documentation  
- `PIPELINE_USAGE_GUIDE.md` - Processing instructions
- `src/utils/regions.py` - Regional boundary code 