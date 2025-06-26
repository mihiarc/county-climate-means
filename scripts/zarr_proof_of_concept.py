#!/usr/bin/env python3
"""
Proof of concept script demonstrating Zarr/Kerchunk capabilities for climate data processing.

This script shows:
1. Converting NetCDF to Zarr format
2. Creating Kerchunk virtual references
3. Processing climate data using Zarr
4. Performance comparison with traditional methods
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from county_climate.shared.utils.zarr_utils import (
    netcdf_to_zarr, validate_zarr_store, optimize_chunks
)
from county_climate.shared.utils.kerchunk_utils import (
    create_single_file_reference, open_kerchunk_dataset,
    benchmark_kerchunk_performance
)
from county_climate.shared.utils.dask_utils import (
    create_distributed_client, monitor_computation
)
from county_climate.means.core.zarr_climate_processor import ZarrClimateProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()


def demonstrate_netcdf_to_zarr(input_file: Path, output_dir: Path) -> Dict:
    """Demonstrate NetCDF to Zarr conversion."""
    console.print("\n[bold blue]1. NetCDF to Zarr Conversion[/bold blue]")
    
    zarr_path = output_dir / "demo.zarr"
    
    # Time the conversion
    start_time = time.time()
    
    # Open dataset to determine optimal chunks
    with xr.open_dataset(input_file) as ds:
        console.print(f"Input dataset shape: {dict(ds.dims)}")
        optimal_chunks = optimize_chunks(ds, target_chunk_size_mb=128)
        console.print(f"Optimal chunks: {optimal_chunks}")
    
    # Convert to Zarr
    with Progress() as progress:
        task = progress.add_task("Converting to Zarr...", total=100)
        
        zarr_group = netcdf_to_zarr(
            input_file,
            zarr_path,
            chunks=optimal_chunks,
            consolidated=True,
            overwrite=True
        )
        progress.update(task, completed=100)
    
    conversion_time = time.time() - start_time
    
    # Validate the Zarr store
    info = validate_zarr_store(zarr_path)
    
    # Display results
    table = Table(title="Zarr Conversion Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Conversion Time", f"{conversion_time:.2f} seconds")
    table.add_row("Output Path", str(zarr_path))
    table.add_row("Arrays", str(len(info['arrays'])))
    
    for array_name, array_info in info['arrays'].items():
        table.add_row(f"{array_name} Shape", str(array_info['shape']))
        table.add_row(f"{array_name} Chunks", str(array_info['chunks']))
        table.add_row(f"{array_name} Size", f"{array_info['size_mb']:.1f} MB")
        table.add_row(f"{array_name} Compressed", f"{array_info['compressed_size_mb']:.1f} MB")
    
    console.print(table)
    
    return {
        'zarr_path': zarr_path,
        'conversion_time': conversion_time,
        'info': info
    }


def demonstrate_kerchunk(input_file: Path, output_dir: Path) -> Dict:
    """Demonstrate Kerchunk virtual references."""
    console.print("\n[bold blue]2. Kerchunk Virtual References[/bold blue]")
    
    ref_path = output_dir / "demo_kerchunk.json"
    
    # Create reference
    start_time = time.time()
    refs = create_single_file_reference(input_file, ref_path)
    ref_creation_time = time.time() - start_time
    
    # Get reference file size
    ref_size_mb = ref_path.stat().st_size / (1024 * 1024)
    
    # Open dataset using reference
    ds_kerchunk = open_kerchunk_dataset(ref_path)
    
    # Display results
    table = Table(title="Kerchunk Reference Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Reference Creation Time", f"{ref_creation_time:.2f} seconds")
    table.add_row("Reference File Size", f"{ref_size_mb:.2f} MB")
    table.add_row("Original File Size", f"{input_file.stat().st_size / (1024**2):.1f} MB")
    table.add_row("Size Reduction", f"{(1 - ref_size_mb / (input_file.stat().st_size / (1024**2))) * 100:.1f}%")
    table.add_row("Variables", ", ".join(ds_kerchunk.data_vars))
    
    console.print(table)
    
    # Benchmark performance
    console.print("\n[yellow]Running performance benchmark...[/yellow]")
    benchmark_results = benchmark_kerchunk_performance(input_file, ref_path)
    
    bench_table = Table(title="Performance Comparison")
    bench_table.add_column("Operation", style="cyan")
    bench_table.add_column("NetCDF (s)", style="red")
    bench_table.add_column("Kerchunk (s)", style="green")
    bench_table.add_column("Speedup", style="yellow")
    
    for i in range(3):
        bench_table.add_row(
            f"Operation {i+1}",
            f"{benchmark_results[f'netcdf_op_{i}']:.2f}",
            f"{benchmark_results[f'kerchunk_op_{i}']:.2f}",
            f"{benchmark_results[f'speedup_op_{i}']:.1f}x"
        )
    
    console.print(bench_table)
    
    return {
        'ref_path': ref_path,
        'ref_size_mb': ref_size_mb,
        'benchmark_results': benchmark_results
    }


def demonstrate_zarr_processing(zarr_path: Path, output_dir: Path) -> Dict:
    """Demonstrate Zarr-based climate processing."""
    console.print("\n[bold blue]3. Zarr-based Climate Processing[/bold blue]")
    
    # Create Dask client
    client = create_distributed_client(n_workers=2, memory_limit='4GB')
    console.print(f"Dask dashboard: [link]{client.dashboard_link}[/link]")
    
    try:
        # Initialize processor
        processor = ZarrClimateProcessor(
            variable='tas',
            scenario='historical',
            region='test',
            dask_client=client,
            chunk_strategy='auto'
        )
        
        # Calculate a simple climatology
        output_path = output_dir / "climatology.zarr"
        
        start_time = time.time()
        
        # Open dataset
        ds = processor.open_zarr_dataset(zarr_path)
        
        # Calculate monthly climatology
        with Progress() as progress:
            task = progress.add_task("Calculating climatology...", total=100)
            
            # Group by month and calculate mean
            monthly_clim = ds.groupby('time.month').mean(dim='time')
            
            # Save result
            monthly_clim.to_zarr(output_path, mode='w', consolidated=True)
            progress.update(task, completed=100)
        
        processing_time = time.time() - start_time
        
        # Display results
        table = Table(title="Zarr Processing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Processing Time", f"{processing_time:.2f} seconds")
        table.add_row("Output Path", str(output_path))
        table.add_row("Output Variables", ", ".join(monthly_clim.data_vars))
        
        console.print(table)
        
        return {
            'output_path': output_path,
            'processing_time': processing_time
        }
        
    finally:
        client.close()


def create_sample_data(output_file: Path) -> None:
    """Create a sample NetCDF file for demonstration."""
    console.print("[yellow]Creating sample NetCDF file...[/yellow]")
    
    # Create synthetic climate data
    times = pd.date_range('2000-01-01', '2005-12-31', freq='D')
    lats = np.linspace(-90, 90, 180)
    lons = np.linspace(-180, 180, 360)
    
    # Create temperature data with seasonal cycle
    temp_data = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    
    for i, t in enumerate(times):
        # Seasonal cycle
        day_of_year = t.dayofyear
        seasonal = 10 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Latitude gradient
        lat_gradient = 30 * np.cos(np.deg2rad(lats))
        
        # Add to data
        temp_data[i] = seasonal + lat_gradient[:, np.newaxis] + 273.15
        
        # Add some noise
        temp_data[i] += np.random.normal(0, 2, (len(lats), len(lons)))
    
    # Create dataset
    ds = xr.Dataset(
        {
            'tas': (['time', 'lat', 'lon'], temp_data, {
                'long_name': 'Near-Surface Air Temperature',
                'units': 'K'
            })
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons
        }
    )
    
    # Add metadata
    ds.attrs['title'] = 'Sample Climate Data for Zarr Demonstration'
    ds.attrs['source'] = 'Synthetic data'
    
    # Save to NetCDF
    ds.to_netcdf(output_file, encoding={'tas': {'zlib': True, 'complevel': 4}})
    console.print(f"[green]Created sample file: {output_file}[/green]")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Zarr/Kerchunk proof of concept for climate data"
    )
    parser.add_argument(
        '--input-file',
        type=Path,
        help='Input NetCDF file (creates sample if not provided)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./zarr_demo'),
        help='Output directory for demonstrations'
    )
    parser.add_argument(
        '--skip-zarr',
        action='store_true',
        help='Skip Zarr conversion demo'
    )
    parser.add_argument(
        '--skip-kerchunk',
        action='store_true',
        help='Skip Kerchunk demo'
    )
    parser.add_argument(
        '--skip-processing',
        action='store_true',
        help='Skip processing demo'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create or use input file
    if args.input_file and args.input_file.exists():
        input_file = args.input_file
    else:
        input_file = args.output_dir / 'sample_climate_data.nc'
        if not input_file.exists():
            create_sample_data(input_file)
    
    console.print("[bold green]Climate Data Zarr/Kerchunk Proof of Concept[/bold green]")
    console.print(f"Input file: {input_file}")
    console.print(f"Output directory: {args.output_dir}")
    
    results = {}
    
    # Run demonstrations
    if not args.skip_zarr:
        results['zarr'] = demonstrate_netcdf_to_zarr(input_file, args.output_dir)
    
    if not args.skip_kerchunk:
        results['kerchunk'] = demonstrate_kerchunk(input_file, args.output_dir)
    
    if not args.skip_processing and 'zarr' in results:
        results['processing'] = demonstrate_zarr_processing(
            results['zarr']['zarr_path'],
            args.output_dir
        )
    
    # Summary
    console.print("\n[bold green]Summary[/bold green]")
    
    summary_table = Table(title="Performance Summary")
    summary_table.add_column("Method", style="cyan")
    summary_table.add_column("Time (s)", style="yellow")
    summary_table.add_column("Storage", style="green")
    
    if 'zarr' in results:
        summary_table.add_row(
            "NetCDF → Zarr",
            f"{results['zarr']['conversion_time']:.2f}",
            "Full copy"
        )
    
    if 'kerchunk' in results:
        summary_table.add_row(
            "NetCDF → Kerchunk",
            f"{results['kerchunk']['ref_size_mb']:.2f} MB reference",
            "Virtual (no copy)"
        )
    
    console.print(summary_table)
    
    # Save results
    results_file = args.output_dir / 'demonstration_results.json'
    with open(results_file, 'w') as f:
        # Convert Path objects to strings for JSON serialization
        json_results = {
            k: {kk: str(vv) if isinstance(vv, Path) else vv 
                for kk, vv in v.items()}
            for k, v in results.items()
        }
        json.dump(json_results, f, indent=2)
    
    console.print(f"\n[green]Results saved to: {results_file}[/green]")


if __name__ == '__main__':
    main()