# Configuration Directory Structure

This directory contains configuration files for the County Climate processing pipeline.

## Directory Organization

### 📁 production/
Production-ready configurations for different use cases:
- `production_ultra_high_performance.yaml` - Optimized for 56-core systems with maximum performance
- `production_with_validation.yaml` - Includes Phase 3 QA/QC validation 
- `production_organized_output.yaml` - Uses new organized output directory structure
- `pipeline_complete_all_regions.yaml` - Processes all regions and variables
- `pipeline_parallel_variables.yaml` - V2 architecture with parallel variable processing

### 📁 development/
Development and testing configurations:
- `test/` - Test configurations with limited scope
- `experimental/` - Experimental features and architectures

### 📁 specialized/
Task-specific configurations:
- `gap_filling/` - Configs for filling gaps in specific data (e.g., SSP245 temperature extremes)
- `single_phase/` - Configs that run only one phase (e.g., metrics only)
- `single_region/` - Region-specific processing configs

### 📁 validation/
QA/QC validation configurations for different regions and scenarios

### 📁 templates/
Template configurations for creating new configs

### 📁 deprecated/
Old configurations kept for reference but no longer actively used

### 📁 docs/
Documentation related to configuration usage

## Usage

To run a pipeline with a specific configuration:

```bash
python main_orchestrated.py run --config configs/production/production_with_validation.yaml
```

## Configuration Selection Guide

- **For production runs**: Use configs in `production/`
- **For testing**: Use configs in `development/test/`
- **For specific tasks**: Check `specialized/` subdirectories
- **For validation only**: Use configs in `validation/`