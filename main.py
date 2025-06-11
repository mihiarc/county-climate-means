#!/usr/bin/env python3
"""
Climate Data Processing - Main Entry Point

Unified entry point for processing climate data and calculating 30-year climate normals
from NEX-GDDP-CMIP6 data. Supports both sequential and multiprocessing approaches.

Features:
- Regional climate processing for CONUS, Alaska, Hawaii, Puerto Rico, Guam
- Multiple climate variables: precipitation, temperature, temperature extremes
- Historical, hybrid, and future climate normals
- Configurable multiprocessing with automatic optimization
- Progress tracking and monitoring
- Comprehensive error handling and logging

Usage Examples:
    # Process all regions with default settings
    python main.py process-all
    
    # Process specific region with multiprocessing
    python main.py process-region CONUS --variables pr tas --max-workers 6
    
    # Process with custom configuration
    python main.py process-region AK --config-file my_config.yaml
    
    # Benchmark multiprocessing performance
    python main.py benchmark --num-files 10
    
    # Monitor processing progress
    python main.py monitor
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import our unified modules
import means
from means.config import get_config, create_sample_config
from means.utils.io_util import NorESM2FileHandler
from means.core.regions import REGION_BOUNDS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('climate_processing.log')
    ]
)
logger = logging.getLogger(__name__)


def process_region_command(args) -> bool:
    """Process a specific region using the unified regional processor."""
    logger.info(f"🌍 Processing region: {args.region}")
    logger.info(f"📊 Variables: {args.variables}")
    
    try:
        # Create regional processor with configuration
        processor = means.create_regional_processor(
            region_key=args.region,
            variables=args.variables,
            max_cores=args.max_workers,
            cores_per_variable=args.cores_per_variable,
            batch_size_years=args.batch_size,
            use_rich_progress=getattr(args, 'rich_progress', True)
        )
        
        # Process all variables for the region
        start_time = time.time()
        results = processor.process_all_variables()
        end_time = time.time()
        
        # Analyze results
        successful_vars = []
        failed_vars = []
        
        for variable, var_results in results.items():
            if var_results.get('status') == 'error':
                failed_vars.append(variable)
            else:
                successful_vars.append(variable)
        
        # Log summary
        duration = end_time - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 Regional processing completed for {args.region}")
        logger.info(f"⏱️  Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"✅ Successful variables: {successful_vars}")
        
        if failed_vars:
            logger.warning(f"❌ Failed variables: {failed_vars}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing region {args.region}: {e}")
        return False


def process_all_regions_command(args) -> bool:
    """Process all regions using the unified approach with pipeline integration."""
    logger.info("🌍 Processing all regions with integrated pipeline-aware workflow")
    
    regions = args.regions or ['CONUS', 'AK', 'HI', 'PRVI', 'GU']
    variables = args.variables or ['pr', 'tas', 'tasmax', 'tasmin']
    
    logger.info(f"📍 Regions: {regions}")
    logger.info(f"📊 Variables: {variables}")
    
    # Use pipeline-aware workflow if enabled
    if getattr(args, 'enable_pipeline_integration', True):
        try:
            from means.workflow.pipeline_workflow import PipelineAwareWorkflow
            
            # Determine downstream pipeline
            downstream_pipeline = getattr(args, 'downstream_pipeline', 'climate_extremes')
            logger.info(f"🔗 Preparing outputs for {downstream_pipeline} pipeline")
            
            # Create workflow
            workflow = PipelineAwareWorkflow()
            
            # Process with integrated workflow
            results = workflow.process_for_pipeline(
                downstream_pipeline=downstream_pipeline,
                regions=regions,
                variables=variables
            )
            
            # Report results
            summary = results.get('workflow_summary', {})
            logger.info(f"\n{'='*80}")
            logger.info(f"🎊 INTEGRATED PIPELINE WORKFLOW COMPLETED")
            logger.info(f"{'='*80}")
            logger.info(f"⏱️  Total duration: {summary.get('workflow_duration_seconds', 0):.1f} seconds")
            logger.info(f"✅ Successful regions ({summary.get('regions_successful', 0)}): {summary.get('successful_regions', [])}")
            logger.info(f"📊 Total datasets cataloged: {summary.get('total_datasets_cataloged', 0)}")
            logger.info(f"🔗 Pipeline bridge status: {summary.get('pipeline_bridge_status', 'unknown')}")
            logger.info(f"🚀 Downstream ready: {summary.get('downstream_ready', False)}")
            
            if summary.get('failed_regions'):
                logger.warning(f"❌ Failed regions ({summary.get('regions_failed', 0)}): {summary.get('failed_regions', [])}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline integration failed, falling back to legacy workflow: {e}")
            # Fall back to legacy workflow
    
    # Legacy workflow (original code)
    overall_start = time.time()
    successful_regions = []
    failed_regions = []
    
    for region in regions:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Processing region: {region}")
        logger.info(f"{'='*60}")
        
        try:
            # Use the process_region convenience function
            results = means.process_region(
                region_key=region,
                variables=variables,
                max_cores=args.max_workers,
                cores_per_variable=args.cores_per_variable,
                batch_size_years=args.batch_size,
                use_rich_progress=getattr(args, 'rich_progress', True)
            )
            
            # Check if processing was successful
            success = True
            for variable, var_results in results.items():
                if var_results.get('status') == 'error':
                    success = False
                    break
            
            if success:
                successful_regions.append(region)
                logger.info(f"✅ Successfully completed {region}")
            else:
                failed_regions.append(region)
                logger.error(f"❌ Failed to complete {region}")
                
        except Exception as e:
            failed_regions.append(region)
            logger.error(f"❌ Error processing {region}: {e}")
    
    # Final summary
    overall_duration = time.time() - overall_start
    logger.info(f"\n{'='*80}")
    logger.info(f"🎊 ALL REGIONS PROCESSING COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"⏱️  Total duration: {overall_duration:.1f} seconds ({overall_duration/60:.1f} minutes)")
    logger.info(f"✅ Successful regions ({len(successful_regions)}): {successful_regions}")
    
    if failed_regions:
        logger.warning(f"❌ Failed regions ({len(failed_regions)}): {failed_regions}")
        return False
    
    return True


def benchmark_command(args) -> bool:
    """Benchmark multiprocessing performance."""
    logger.info("🏃 Running multiprocessing performance benchmark")
    
    try:
        # Get configuration
        config = get_config()
        data_directory = str(config.paths.input_data_dir)
        
        # Run benchmark using the multiprocessing engine
        from means.core.multiprocessing_engine import benchmark_multiprocessing_performance
        results = benchmark_multiprocessing_performance(
            data_directory=data_directory,
            num_files=args.num_files,
            max_workers_to_test=args.workers_to_test
        )
        
        # Display results
        logger.info("\n📊 Benchmark Results:")
        for config_name, result in results.items():
            logger.info(f"  {config_name}: {result['total_time']:.1f}s, "
                       f"speedup: {result['speedup']:.1f}x, "
                       f"efficiency: {result['efficiency']:.1f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        return False


def monitor_command(args) -> bool:
    """Monitor processing progress."""
    logger.info("📈 Monitoring processing progress")
    
    try:
        # Look for progress files
        progress_files = list(Path('.').glob('*_processing_progress.json'))
        
        if not progress_files:
            logger.info("No active processing found")
            return True
        
        logger.info(f"Found {len(progress_files)} progress files:")
        for progress_file in progress_files:
            logger.info(f"  📄 {progress_file}")
        
        # TODO: Implement real-time monitoring
        logger.info("Real-time monitoring not yet implemented")
        return True
        
    except Exception as e:
        logger.error(f"❌ Monitoring failed: {e}")
        return False


def status_command(args) -> bool:
    """Show system status and configuration."""
    logger.info("📋 System Status and Configuration")
    
    try:
        # Get configuration
        config = get_config()
        
        logger.info(f"\n📁 Paths:")
        logger.info(f"  Input data: {config.paths.input_data_dir}")
        logger.info(f"  Output base: {config.paths.output_base_dir}")
        
        logger.info(f"\n⚙️  Processing:")
        logger.info(f"  Type: {config.processing.processing_type}")
        logger.info(f"  Max workers: {config.processing.max_workers}")
        logger.info(f"  Batch size: {config.processing.batch_size}")
        logger.info(f"  Safe mode: {config.processing.safe_mode}")
        
        # Check data availability
        try:
            file_handler = NorESM2FileHandler(str(config.paths.input_data_dir))
            availability = file_handler.validate_data_availability()
            
            logger.info(f"\n📊 Data Availability:")
            for variable, scenarios in availability.items():
                logger.info(f"  {variable}:")
                for scenario, (start, end) in scenarios.items():
                    logger.info(f"    {scenario}: {start}-{end}")
        except Exception as e:
            logger.warning(f"Could not check data availability: {e}")
        
        # Show available regions
        logger.info(f"\n🌍 Available Regions:")
        for region_key, region_info in REGION_BOUNDS.items():
            logger.info(f"  {region_key}: {region_info['name']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        return False


def visualize_command(args) -> bool:
    """Visualize climate data files."""
    logger.info(f"🗺️  Creating visualization for: {args.file_path}")
    
    try:
        # Create visualizer
        visualizer = means.RegionalVisualizer()
        
        # Check if file exists
        file_path = Path(args.file_path)
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            return False
        
        # Create visualization
        if args.simple:
            visualizer.create_simple_visualization(
                str(file_path), 
                args.output_dir, 
                args.region
            )
        else:
            visualizer.create_comprehensive_visualization(
                str(file_path), 
                args.output_dir, 
                args.region, 
                not args.no_save
            )
        
        logger.info("✅ Visualization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_config_command(args) -> bool:
    """Create a sample configuration file."""
    logger.info("📝 Creating sample configuration file")
    
    try:
        config_file = args.output or "climate_config.yaml"
        success = create_sample_config(config_file)
        
        if success:
            logger.info(f"✅ Sample configuration created: {config_file}")
            return True
        else:
            logger.error("❌ Failed to create configuration file")
            return False
            
    except Exception as e:
        logger.error(f"❌ Configuration creation failed: {e}")
        return False


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Climate Data Processing - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s process-all                           # Process all regions
  %(prog)s process-region CONUS --variables pr  # Process CONUS precipitation
  %(prog)s visualize output/file.nc --simple    # Visualize climate data
  %(prog)s benchmark --num-files 10             # Benchmark performance
  %(prog)s status                                # Show system status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process region command
    region_parser = subparsers.add_parser('process-region', help='Process a specific region')
    region_parser.add_argument('region', choices=['CONUS', 'AK', 'HI', 'PRVI', 'GU'],
                              help='Region to process')
    region_parser.add_argument('--variables', nargs='+', 
                              choices=['pr', 'tas', 'tasmax', 'tasmin'],
                              default=['pr', 'tas', 'tasmax', 'tasmin'],
                              help='Variables to process')
    region_parser.add_argument('--max-workers', type=int, default=6,
                              help='Maximum number of workers')
    region_parser.add_argument('--cores-per-variable', type=int, default=2,
                              help='Cores per variable')
    region_parser.add_argument('--batch-size', type=int, default=2,
                              help='Batch size for year processing')
    region_parser.add_argument('--no-rich-progress', dest='rich_progress', action='store_false',
                              help='Disable rich progress tracking (use simple logging instead)')
    
    # Process all regions command
    all_parser = subparsers.add_parser('process-all', help='Process all regions')
    all_parser.add_argument('--regions', nargs='+',
                           choices=['CONUS', 'AK', 'HI', 'PRVI', 'GU'],
                           help='Regions to process (default: all)')
    all_parser.add_argument('--variables', nargs='+',
                           choices=['pr', 'tas', 'tasmax', 'tasmin'],
                           default=['pr', 'tas', 'tasmax', 'tasmin'],
                           help='Variables to process')
    all_parser.add_argument('--max-workers', type=int, default=6,
                           help='Maximum number of workers')
    all_parser.add_argument('--cores-per-variable', type=int, default=2,
                           help='Cores per variable')
    all_parser.add_argument('--batch-size', type=int, default=2,
                           help='Batch size for year processing')
    all_parser.add_argument('--no-rich-progress', dest='rich_progress', action='store_false',
                           help='Disable rich progress tracking (use simple logging instead)')
    all_parser.add_argument('--downstream-pipeline', 
                           choices=['climate_extremes', 'climate_metrics'],
                           default='climate_extremes',
                           help='Target downstream pipeline for optimization')
    all_parser.add_argument('--disable-pipeline-integration', dest='enable_pipeline_integration', 
                           action='store_false',
                           help='Disable pipeline integration features')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark multiprocessing performance')
    bench_parser.add_argument('--num-files', type=int, default=10,
                             help='Number of files to use for benchmarking')
    bench_parser.add_argument('--workers-to-test', nargs='+', type=int,
                             default=[1, 2, 4, 6, 8],
                             help='Worker counts to test')
    
    # Monitor command
    subparsers.add_parser('monitor', help='Monitor processing progress')
    
    # Status command
    subparsers.add_parser('status', help='Show system status and configuration')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize climate data files')
    viz_parser.add_argument('file_path', help='Path to NetCDF file to visualize')
    viz_parser.add_argument('--output-dir', '-o', help='Output directory for saved plots')
    viz_parser.add_argument('--region', '-r', choices=list(REGION_BOUNDS.keys()),
                           help='Force specific region (auto-detect if not provided)')
    viz_parser.add_argument('--simple', '-s', action='store_true',
                           help='Create simple visualization instead of comprehensive')
    viz_parser.add_argument('--no-save', action='store_true',
                           help='Do not save plots to file')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create sample configuration file')
    config_parser.add_argument('--output', '-o', default='climate_config.yaml',
                              help='Output configuration file name')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    logger.info(f"🚀 Starting command: {args.command}")
    
    try:
        if args.command == 'process-region':
            success = process_region_command(args)
        elif args.command == 'process-all':
            success = process_all_regions_command(args)
        elif args.command == 'benchmark':
            success = benchmark_command(args)
        elif args.command == 'monitor':
            success = monitor_command(args)
        elif args.command == 'status':
            success = status_command(args)
        elif args.command == 'visualize':
            success = visualize_command(args)
        elif args.command == 'create-config':
            success = create_config_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        if success:
            logger.info(f"✅ Command '{args.command}' completed successfully")
            return 0
        else:
            logger.error(f"❌ Command '{args.command}' failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 