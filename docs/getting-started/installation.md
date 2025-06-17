# Installation Guide

This guide will help you install County Climate and its dependencies.

## Prerequisites

Before installing County Climate, ensure you have:

- **Python 3.9+**: County Climate requires Python 3.9 or higher
- **pip**: Python package installer
- **Git**: For cloning the repository
- **C compiler**: Required for some scientific Python packages

### System Dependencies

=== "Ubuntu/Debian"

    ```bash
    sudo apt-get update
    sudo apt-get install python3-dev python3-pip git
    sudo apt-get install libgeos-dev libproj-dev libgdal-dev
    ```

=== "macOS"

    ```bash
    # Install Homebrew if not already installed
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Install dependencies
    brew install python@3.9 git
    brew install geos proj gdal
    ```

=== "Windows (WSL2)"

    ```bash
    # In WSL2 Ubuntu terminal
    sudo apt-get update
    sudo apt-get install python3-dev python3-pip git
    sudo apt-get install libgeos-dev libproj-dev libgdal-dev
    ```

## Installation Methods

### Method 1: Development Installation (Recommended)

This method is recommended for most users as it allows easy updates and modifications.

```bash
# Clone the repository
git clone https://github.com/mihiarc/county_climate_means.git
cd county_climate_means

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e .[all]
```

### Method 2: Minimal Installation

For basic functionality without optional features:

```bash
# Clone and enter directory
git clone https://github.com/mihiarc/county_climate_means.git
cd county_climate_means

# Install core dependencies only
pip install -e .
```

### Method 3: Feature-Specific Installation

Install only the features you need:

```bash
# Performance enhancements (Dask, Numba)
pip install -e .[performance]

# Geospatial functionality
pip install -e .[geospatial]

# Visualization capabilities
pip install -e .[visualization]

# Development tools (testing, linting)
pip install -e .[dev]
```

## Verifying Installation

After installation, verify everything is working:

```bash
# Check County Climate is installed
python -c "import county_climate; print(county_climate.__version__)"

# Run the help command
python main_orchestrated.py --help

# Run a simple test
python -c "from county_climate.means.core.regions import REGIONS; print(list(REGIONS.keys()))"
```

Expected output:
```
0.1.0
Usage: main_orchestrated.py [OPTIONS] COMMAND [ARGS]...
['CONUS', 'Alaska', 'Hawaii', 'PRVI', 'Guam']
```

## Setting Up Data Paths

County Climate needs to know where your climate data is stored. Set these environment variables:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export COUNTY_CLIMATE_DATA_PATH="/path/to/nexgddp/data"
export COUNTY_CLIMATE_OUTPUT_PATH="/path/to/output"
export COUNTY_CLIMATE_TEMP_PATH="/path/to/temp"

# Optional: Set resource limits
export COUNTY_CLIMATE_MAX_WORKERS=16
export COUNTY_CLIMATE_MAX_MEMORY=32
```

## Installing MkDocs (for Documentation)

To build and view the documentation locally:

```bash
# Install MkDocs and theme
pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

## Troubleshooting

### Common Issues

#### ImportError with Geospatial Libraries

If you encounter errors with `geopandas` or `cartopy`:

```bash
# Ensure system libraries are installed
sudo apt-get install libgeos-dev libproj-dev libgdal-dev

# Reinstall geospatial packages
pip install --force-reinstall geopandas cartopy
```

#### Memory Errors During Installation

Some packages require significant memory to compile:

```bash
# Install packages one at a time
pip install numpy
pip install pandas
pip install xarray
pip install -e .
```

#### Permission Errors

If you encounter permission errors:

```bash
# Use --user flag
pip install --user -e .[all]

# Or use a virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -e .[all]
```

### Getting Help

If you continue to have issues:

1. Check the [GitHub Issues](https://github.com/mihiarc/county_climate_means/issues)
2. Review the [FAQ](../guides/faq.md)
3. Open a new issue with:
   - Your operating system
   - Python version (`python --version`)
   - Full error message
   - Steps to reproduce

## Next Steps

Now that County Climate is installed:

- Continue to the [Quick Start Guide](quickstart.md)
- Read about [Configuration](configuration.md)
- Try the [Climate Means Tutorial](../tutorials/climate-means.md)