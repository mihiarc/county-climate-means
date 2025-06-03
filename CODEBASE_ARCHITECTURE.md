# Climate Data Processing Codebase Architecture

This document describes the architecture and data flow of the climate data processing codebase using a visual mermaid diagram.

## Overview

This is a Python-based climate data processing pipeline that calculates 30-year climate normals from NetCDF climate model data. The system has evolved from a distributed Dask-based approach to a sequential processing architecture for improved reliability with NetCDF files.

## System Architecture

```mermaid
graph TB
    %% Data Sources
    subgraph "Data Sources"
        D1[NorESM2-LM Climate Data<br/>- pr, tas, tasmax, tasmin<br/>- historical, ssp245, ssp585<br/>- NetCDF files by year]
    end

    %% Entry Points
    subgraph "Entry Points"
        E1[run_climate_means.py<br/>Main CLI entry point]
        E2[process_all_climate_normals.py<br/>Comprehensive pipeline]
        E3[run_e2e_tests.py<br/>End-to-end testing]
    end

    %% Core Processing Modules
    subgraph "Core Processing"
        C1[climate_means.py<br/>Core climate calculations<br/>- compute_climate_normal<br/>- calculate_daily_climatology<br/>- process_climate_data_workflow]
        
        C2[process_climate_normals_multiprocessing.py<br/>Multiprocessing workflows<br/>- rolling_30year_normals<br/>- hybrid_normals]
    end

    %% Utility Modules
    subgraph "Utility Modules"
        U1[io_util.py<br/>File I/O operations<br/>- NorESM2FileHandler<br/>- open_dataset_safely<br/>- save_climate_result]
        
        U2[regions.py<br/>Geographic operations<br/>- REGION_BOUNDS<br/>- extract_region<br/>- validate_region_bounds]
        
        U3[time_util.py<br/>Time handling<br/>- generate_climate_periods<br/>- handle_time_coordinates<br/>- reconstruct_time_dataarray]
        
        U4[status.py<br/>Progress monitoring<br/>- ProcessingStatus<br/>- monitor_progress]
        
        U5[optimize.py<br/>Performance optimization<br/>- memory management<br/>- chunk optimization]
    end

    %% Data Processing Flow
    subgraph "Processing Workflow"
        P1[Sequential File Processing<br/>One file at a time]
        P2[Regional Extraction<br/>CONUS, AK, HI, PRVI, GU]
        P3[Daily Climatology<br/>Calculate day-of-year means]
        P4[30-Year Climate Normals<br/>Rolling windows]
        P5[Quality Control<br/>Validation & metadata]
    end

    %% Output Products
    subgraph "Output Products"
        O1[Rolling 30-Year Normals<br/>Historical: 1980-2014<br/>Hybrid: 2015-2044<br/>Future: 2045+]
        O2[Individual NetCDF Files<br/>Variable + Region + Year]
        O3[Combined Datasets<br/>Multi-year collections]
        O4[Processing Reports<br/>Status & progress logs]
    end

    %% Testing & Validation
    subgraph "Testing Framework"
        T1[test_processing_pipeline.py<br/>Unit tests]
        T2[test_multiprocessing.py<br/>Multiprocessing tests]
        T3[pytest.ini<br/>Test configuration]
        T4[.coverage<br/>Coverage reports]
    end

    %% Configuration & Monitoring
    subgraph "Configuration"
        F1[requirements.txt<br/>Dependencies]
        F2[processing_progress.json<br/>Progress tracking]
        F3[processing_status_report.md<br/>Status reports]
        F4[MULTIPROCESSING_SETUP.md<br/>Setup documentation]
    end

    %% Data Flow Connections
    D1 --> E1
    D1 --> E2
    
    E1 --> C1
    E2 --> C1
    E2 --> C2
    
    C1 --> U1
    C1 --> U2
    C1 --> U3
    C2 --> U1
    C2 --> U2
    C2 --> U3
    
    U1 --> P1
    U2 --> P2
    U3 --> P3
    
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
    
    P5 --> O1
    P5 --> O2
    P5 --> O3
    
    U4 --> O4
    U4 --> F2
    U4 --> F3
    
    %% Testing connections
    E3 --> T1
    E3 --> T2
    T1 --> C1
    T2 --> C2
    
    %% Configuration connections
    E1 -.-> F1
    E2 -.-> F1
    C1 -.-> F4
    C2 -.-> F4
```

## Key Components

### 1. Entry Points
- **`run_climate_means.py`**: Main CLI interface with different modes (demo, noresm2, example)
- **`process_all_climate_normals.py`**: Comprehensive pipeline for processing all variables and scenarios
- **`run_e2e_tests.py`**: End-to-end testing framework

### 2. Core Processing Modules
- **`climate_means.py`**: Central processing logic with climate calculations and workflow orchestration
- **`process_climate_normals_multiprocessing.py`**: Specialized multiprocessing workflows for large-scale processing

### 3. Utility Modules
- **`io_util.py`**: File I/O operations and NorESM2 data handling
- **`regions.py`**: Geographic region definitions and extraction functions
- **`time_util.py`**: Time coordinate handling and period generation
- **`status.py`**: Progress monitoring and status reporting
- **`optimize.py`**: Performance optimization and memory management

### 4. Processing Architecture

The system follows a **sequential processing** approach that has evolved from an earlier Dask-based distributed system. Key characteristics:

- **Sequential file processing**: One NetCDF file at a time to avoid threading issues
- **Regional extraction**: Support for CONUS, Alaska, Hawaii, Puerto Rico, and Guam
- **Conservative memory management**: Aggressive garbage collection and safe chunking
- **Crash-resistant design**: Retry logic and graceful error handling

### 5. Data Flow

1. **Input**: NorESM2-LM climate model data (NetCDF files organized by variable/scenario/year)
2. **Processing**: Sequential processing with regional extraction and daily climatology calculation
3. **Computation**: 30-year rolling climate normals using historical and future scenario data
4. **Output**: NetCDF files with climate normals, metadata, and processing statistics

### 6. Output Products

The system generates three types of climate normals:
- **Historical normals** (1980-2014): Using only historical data
- **Hybrid normals** (2015-2044): Combining historical and future scenario data
- **Future normals** (2045+): Using only future scenario data

### 7. Testing Framework

Comprehensive testing includes:
- Unit tests for core processing functions
- Multiprocessing-specific tests
- End-to-end pipeline validation
- Coverage reporting with pytest

## Architecture Benefits

1. **Reliability**: Sequential processing avoids NetCDF threading issues
2. **Memory efficiency**: Conservative chunking and garbage collection
3. **Modularity**: Clear separation of concerns across utility modules
4. **Maintainability**: Comprehensive testing and documentation
5. **Flexibility**: Support for multiple regions, variables, and scenarios
6. **Monitoring**: Built-in progress tracking and status reporting

## Technology Stack

- **Python 3.12**: Core language
- **xarray**: NetCDF data manipulation
- **numpy**: Numerical computations
- **pandas**: Data analysis and time handling
- **netCDF4**: Low-level NetCDF operations
- **pytest**: Testing framework
- **uv**: Package management (as specified in user rules)

This architecture represents a mature, production-ready climate data processing system optimized for reliability and scientific reproducibility. 