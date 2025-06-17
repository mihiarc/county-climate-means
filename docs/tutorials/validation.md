# Enhanced Validation Tutorial

This tutorial covers the enhanced validation features in County Climate, including geographic visualization, advanced pattern detection, and comprehensive reporting.

## Overview

The County Climate validation module has been enhanced with powerful features from standalone scripts:

- **Geographic Visualization**: Create choropleth maps, climate projections, and outlier maps
- **Advanced Spatial Analysis**: County profiling, persistent outlier detection, severity classification
- **Enhanced Precipitation Analysis**: Pattern detection, problem investigation, diagnostic dashboards
- **Unified Validation Suite**: Run all validators with a single interface

## Prerequisites

Install the enhanced validation dependencies:

```bash
pip install -e .[validation-enhanced]
```

This installs:
- `geopandas`: For geographic data handling
- `contextily`: For map basemaps
- `folium`: For interactive maps
- `seaborn`: For enhanced statistical plots

## Using the Validation Suite

The `ValidationSuite` provides a unified interface for all validation features:

```python
from county_climate.validation import ValidationSuite
import pandas as pd

# Load your climate metrics data
df = pd.read_csv("county_climate_metrics.csv")

# Initialize validation suite
suite = ValidationSuite(
    output_dir="validation_results"
)

# Run comprehensive validation
results = suite.run_all_validations(
    data=df,
    include_geographic=True,
    create_dashboards=True,
    export_format='html'
)

print(f"Overall quality score: {results['overall_quality_score']:.2f}")
```

## Geographic Visualization

### Creating Climate Maps

```python
from county_climate.validation.visualization import GeographicVisualizer

# Initialize geographic visualizer
geo_viz = GeographicVisualizer(
    df=climate_data,
    output_dir="maps"
)

# Create current climate maps
geo_viz.create_climate_maps(
    variables=['mean_temp_c', 'annual_precip_cm'],
    time_period="current"  # 2015-2020
)

# Create climate change projections
geo_viz.create_change_maps(
    scenario="ssp245",
    variables=['mean_temp_c', 'annual_precip_cm']
)
```

### Visualizing Spatial Outliers

```python
# From validation results
outlier_data = pd.DataFrame({
    'fips': ['12001', '12003', '12005'],
    'severity': ['extreme', 'moderate', 'mild'],
    'is_outlier': [True, True, True]
})

geo_viz.create_outlier_maps(
    outlier_data,
    title="Counties with Data Quality Issues"
)
```

## Advanced Spatial Analysis

### County Profiling

Create detailed profiles for counties with unusual patterns:

```python
from county_climate.validation.validators import EnhancedSpatialOutliersValidator

# Initialize enhanced spatial validator
spatial_validator = EnhancedSpatialOutliersValidator(
    output_dir="spatial_analysis"
)

# Validate and create profiles
result = spatial_validator.validate(climate_data)
profiles = spatial_validator.create_county_profiles(
    top_n=20,  # Top 20 outlier counties
    save_plots=True
)

# Access profile for specific county
county_profile = profiles['12345']
print(f"County: {county_profile['county_name']}")
print(f"Outlier occurrences: {county_profile['outlier_info']['total_occurrences']}")
```

### Persistent Outlier Detection

Find counties that are consistently flagged as outliers:

```python
# Identify persistent outliers
outlier_results = spatial_validator._detect_all_outliers()
persistent = spatial_validator.identify_persistent_outliers(
    outlier_results,
    min_methods=2,  # Flagged by at least 2 methods
    min_metrics=2   # Flagged for at least 2 metrics
)

# Get severity classification
severity_df = spatial_validator.classify_outlier_severity(outlier_results)
print(f"Extreme outliers: {len(severity_df[severity_df['severity'] == 'extreme'])}")
```

### Temporal Outlier Analysis

Analyze when outliers occur over time:

```python
temporal_patterns = spatial_validator.analyze_temporal_outlier_patterns()

print(f"Trending outliers: {len(temporal_patterns['trending_outliers'])}")
print(f"Outliers by decade:")
for decade, count in temporal_patterns['outliers_by_decade'].items():
    print(f"  {decade}s: {count} counties")
```

## Enhanced Precipitation Analysis

### Problem Pattern Detection

```python
from county_climate.validation.validators import EnhancedPrecipitationValidator

# Initialize enhanced precipitation validator
precip_validator = EnhancedPrecipitationValidator(
    output_dir="precipitation_analysis"
)

# Detect specific problem patterns
problems = precip_validator.detect_problem_patterns()

for pattern, df in problems.items():
    if not df.empty:
        print(f"{pattern}: {len(df)} records affected")
```

### County Investigation

Deep-dive into counties with precipitation issues:

```python
# Investigate top problem counties
investigations = precip_validator.investigate_problem_counties(
    top_n=10,
    create_plots=True
)

for county_id, investigation in investigations.items():
    print(f"\nCounty {county_id}:")
    print(f"  Problem records: {investigation['problem_records']}")
    print(f"  Problem types: {investigation['problem_types']}")
```

### Diagnostic Dashboard

Create a comprehensive 9-panel diagnostic dashboard:

```python
# Create diagnostic dashboard
precip_validator.create_diagnostic_dashboard()

# This creates:
# - Annual precipitation distribution
# - High precipitation days distribution
# - Precipitation intensity analysis
# - Problem patterns by type
# - Problems over time
# - Problems by scenario
# - State-level aggregation
# - Temperature-precipitation relationship
# - Summary statistics
```

## Comprehensive Reporting

### Executive Summary

The validation suite automatically generates executive summaries:

```python
# After running validations
suite.generate_executive_report()

# This creates:
# - executive_summary.json
# - EXECUTIVE_SUMMARY.md
```

### HTML Reports

Generate interactive HTML reports:

```python
results = suite.run_all_validations(
    data=df,
    export_format='html'  # or 'json', 'pdf', 'all'
)

# Opens in browser: validation_results/validation_report.html
```

### Custom County Reports

Generate reports for specific counties:

```python
# Get all validations for a specific county
county_id = '12345'
county_data = df[df['fips'] == county_id]

# Run focused validation
county_suite = ValidationSuite(output_dir=f"county_{county_id}_validation")
county_results = county_suite.run_all_validations(county_data)
```

## Integration with Pipeline

### Configuration-Based Validation

Add enhanced validation to your pipeline configuration:

```yaml
stages:
  - stage_id: "enhanced_validation"
    stage_type: "validation"
    stage_config:
      validators_to_run: ["qaqc", "spatial", "precipitation"]
      enhanced_features:
        geographic_maps: true
        county_profiles: true
        precipitation_diagnostics: true
      export_formats: ["json", "html"]
      top_outlier_counties: 20
```

### Programmatic Usage

```python
from county_climate.shared.orchestration import PipelineOrchestrator

# In your stage handler
def enhanced_validation_stage_handler(context):
    df = pd.read_csv(context['stage_inputs']['metrics_path'])
    
    suite = ValidationSuite(
        config=context['stage_config'],
        output_dir=context['output_dir']
    )
    
    results = suite.run_all_validations(
        data=df,
        include_geographic=True,
        create_dashboards=True
    )
    
    # Create additional analyses
    suite.create_outlier_county_profiles(top_n=20)
    suite.create_climate_change_maps()
    
    return {
        'status': 'completed',
        'validation_passed': all(r.passed for r in results.values()),
        'quality_score': np.mean([r.quality_score for r in results.values()]),
        'report_path': context['output_dir'] / 'validation_report.html'
    }
```

## Best Practices

### 1. Start with Overview
Always run the full validation suite first to get an overview:

```python
suite.run_all_validations(df)
```

### 2. Focus on Problem Areas
Use the results to identify areas needing investigation:

```python
# If spatial outliers are found
if not results['spatial'].passed:
    profiles = suite.create_outlier_county_profiles(top_n=30)
```

### 3. Create Visualizations
Visual analysis often reveals patterns not obvious in statistics:

```python
# Always create maps for spatial data
if suite._check_geographic_available():
    suite.create_climate_change_maps()
```

### 4. Document Findings
Use the reporting features to document your analysis:

```python
# Generate all report formats
results = suite.run_all_validations(
    df, 
    export_format='all'
)
```

## Troubleshooting

### Geographic Features Not Available

If you get import errors for geographic features:

```bash
# Install required packages
pip install geopandas contextily folium

# On some systems, you may need:
conda install -c conda-forge geopandas
```

### Memory Issues with Large Datasets

For large datasets, process in chunks:

```python
# Process by region
for region in df['region'].unique():
    region_df = df[df['region'] == region]
    suite.run_all_validations(region_df)
```

### Slow Map Generation

Speed up map creation by simplifying geometries:

```python
# In geographic visualizer
geo_viz.gdf['geometry'] = geo_viz.gdf['geometry'].simplify(0.01)
```

## Next Steps

- Explore the [Pipeline Orchestration Tutorial](pipeline.md) to integrate validation
- Read about [Performance Tuning](../guides/performance.md) for large datasets
- Check the [API Reference](../api/validation.md) for detailed documentation

The enhanced validation features provide comprehensive quality assurance for your climate data, ensuring reliability and identifying potential issues before they impact downstream analyses.