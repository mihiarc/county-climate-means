# Climate Means

Calculate climate normals (30-year averages) for climate change projections from NEX-GDDP-CMIP6 data.

## Overview

Climate Means is a Python package designed for processing climate data and calculating 30-year climate normals (averages) for climate change projections. It provides tools for working with NEX-GDDP-CMIP6 data and supports various climate variables including precipitation, temperature, and temperature extremes.

## Features

- 🌡️ **Climate Normal Calculation**: Compute 30-year rolling climate normals
- 🗺️ **Regional Processing**: Support for multiple U.S. regions (CONUS, Alaska, Hawaii, Puerto Rico, Guam)
- ⚡ **High Performance**: Multiprocessing support for large datasets
- 📊 **Multiple Variables**: Support for precipitation, temperature, and temperature extremes
- 🔄 **Flexible Scenarios**: Historical, hybrid, and future climate scenarios

## Installation

### Basic Installation
```bash
pip install climate-means
```

### With Optional Dependencies
```bash
# For enhanced performance
pip install climate-means[performance]

# For geospatial functionality
pip install climate-means[geospatial]

# For visualization capabilities
pip install climate-means[visualization]

# Install everything
pip install climate-means[all]
```

### Development Installation
```bash
git clone https://github.com/yourusername/climate-means.git
cd climate-means
pip install -e .[dev]
```

## Quick Start

```python
import climate_means as cm

# Initialize file handler
handler = cm.NorESM2FileHandler("/path/to/data")

# Process climate data
result = cm.process_climate_data_workflow(
    data_directory="/path/to/data",
    output_directory="./output",
    variables=["pr", "tas"],
    regions=["CONUS"],
    scenarios=["historical"]
)
```

## Command Line Interface

After installation, you can use the command-line tools:

```bash
# Main CLI
climate-means --help

# Process climate data
climate-process --variable pr --region CONUS

# Regional processing
climate-alaska
climate-hawaii
climate-guam
climate-prvi
```

## Supported Data

- **Data Source**: NEX-GDDP-CMIP6
- **Variables**: 
  - `pr` - Precipitation
  - `tas` - Near-surface air temperature
  - `tasmax` - Daily maximum near-surface air temperature
  - `tasmin` - Daily minimum near-surface air temperature
- **Scenarios**: Historical, SSP245, SSP585
- **Regions**: CONUS, Alaska, Hawaii, Puerto Rico & Virgin Islands, Guam & Northern Mariana Islands

## Documentation

Full documentation is available at: [https://climate-means.readthedocs.io](https://climate-means.readthedocs.io)

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{climate_means,
  title={Climate Means: Climate Normal Calculation for NEX-GDDP-CMIP6 Data},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/climate-means}
}
```

## Acknowledgments

- NEX-GDDP-CMIP6 data providers
- Climate modeling community
- Open source scientific Python ecosystem 