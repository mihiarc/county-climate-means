#!/usr/bin/env python3
"""
Main Command Line Interface for Climate Data Processing

This provides a unified entry point for all climate data processing operations:
- Sequential processing pipeline
- Parallel processing pipeline  
- Progress monitoring
- Status checking
- Optimization testing
- Configuration management
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

# Import project modules
from ..config import get_default_config, get_production_config, get_development_config, get_testing_config
from ..pipelines import SequentialPipeline
from ..monitoring import ProgressMonitor, StatusChecker
from ..utils.optimization import SafeOptimizer
from ..core import ClimateMultiprocessor, benchmark_multiprocessing_speedup


def setup_logging(level: str = "INFO"):
    """Setup logging for CLI operations."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('climate_processing.log')
        ]
    )


def cmd_sequential(args):
    """Run sequential processing pipeline."""
    print("🌍 Starting Sequential Climate Processing Pipeline")
    
    config = get_default_config()
    if args.config_type:
        if args.config_type == 'production':
            config = get_production_config()
        elif args.config_type == 'development':
            config = get_development_config()
        elif args.config_type == 'testing':
            config = get_testing_config()
    
    # Override config with CLI arguments
    if args.input_dir:
        config.input_data_dir = args.input_dir
    if args.output_dir:
        config.output_base_dir = args.output_dir
    if args.variables:
        config.variables = args.variables
    if args.regions:
        config.regions = args.regions
    if args.log_level:
        config.log_level = args.log_level
    
    setup_logging(config.log_level)
    
    try:
        pipeline = SequentialPipeline(config.input_data_dir, config.output_base_dir)
        pipeline.run(variables=config.variables, regions=config.regions)
        print("✅ Sequential processing completed successfully!")
    except Exception as e:
        print(f"❌ Sequential processing failed: {e}")
        sys.exit(1)


def cmd_parallel(args):
    """Run parallel processing pipeline."""
    print("🚀 Starting Parallel Climate Processing Pipeline")
    
    # Import parallel pipeline
    from ..pipelines.parallel import ParallelPipeline
    
    config = get_default_config()
    if args.config_type:
        if args.config_type == 'production':
            config = get_production_config()
        elif args.config_type == 'development':
            config = get_development_config()
        elif args.config_type == 'testing':
            config = get_testing_config()
    
    # Override config with CLI arguments
    if args.input_dir:
        config.input_data_dir = args.input_dir
    if args.output_dir:
        config.output_base_dir = args.output_dir
    if args.workers:
        config.max_workers = args.workers
    if args.variables:
        config.variables = args.variables
    if args.regions:
        config.regions = args.regions
    if args.log_level:
        config.log_level = args.log_level
    
    setup_logging(config.log_level)
    
    try:
        pipeline = ParallelPipeline(config)
        results = pipeline.run(
            variables=config.variables,
            regions=config.regions,
            scenarios=config.scenarios
        )
        
        print(f"✅ Parallel processing completed successfully!")
        print(f"📊 Processed {results['variables_processed']}/{len(config.variables)} variables")
        print(f"⏱️  Total time: {results['duration_seconds']:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Parallel processing failed: {e}")
        sys.exit(1)


def cmd_monitor(args):
    """Monitor processing progress."""
    print("📊 Starting Progress Monitor")
    
    monitor = ProgressMonitor(args.status_file, args.progress_log)
    
    if args.summary:
        monitor.show_summary()
    elif args.log_lines:
        monitor.tail_progress_log(args.log_lines)
    else:
        print("Starting real-time monitoring (Press Ctrl+C to exit)")
        monitor.monitor_loop(args.refresh_interval)


def cmd_status(args):
    """Quick status check."""
    print("🔍 Climate Normals Processing Status")
    
    checker = StatusChecker()
    checker.quick_status()


def cmd_optimize(args):
    """Run optimization testing."""
    print("⚡ Running Optimization Analysis")
    
    if not args.data_dir:
        print("❌ Data directory required for optimization testing")
        sys.exit(1)
    
    setup_logging(args.log_level or "INFO")
    
    try:
        optimizer = SafeOptimizer(args.data_dir)
        optimizer.run_optimization_analysis()
        print("✅ Optimization analysis completed!")
    except Exception as e:
        print(f"❌ Optimization analysis failed: {e}")
        sys.exit(1)


def cmd_benchmark(args):
    """Run multiprocessing benchmark."""
    print("🏁 Running Multiprocessing Benchmark")
    
    if not args.data_dir:
        print("❌ Data directory required for benchmarking")
        sys.exit(1)
    
    setup_logging(args.log_level or "INFO")
    
    try:
        results = benchmark_multiprocessing_speedup(args.data_dir, args.num_files)
        print("✅ Benchmark completed!")
        
        # Display results
        if 'sequential' in results and 'parallel' in results:
            seq_time = results['sequential']['time']
            par_time = results['parallel']['time']
            speedup = seq_time / par_time if par_time > 0 else 0
            
            print(f"\n📈 Benchmark Results:")
            print(f"Sequential: {seq_time:.1f}s")
            print(f"Parallel: {par_time:.1f}s")
            print(f"Speedup: {speedup:.1f}x")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)


def cmd_config(args):
    """Show configuration information."""
    print("⚙️  Climate Processing Configuration")
    
    config_type = args.config_type or 'default'
    
    if config_type == 'production':
        config = get_production_config()
    elif config_type == 'development':
        config = get_development_config()
    elif config_type == 'testing':
        config = get_testing_config()
    else:
        config = get_default_config()
    
    print(f"\nConfiguration: {config_type}")
    print("-" * 40)
    
    config_dict = config.to_dict()
    for key, value in config_dict.items():
        print(f"{key}: {value}")


def create_parser():
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="Climate Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sequential processing
  %(prog)s sequential
  
  # Run parallel processing with 8 workers
  %(prog)s parallel --workers 8
  
  # Monitor progress in real-time
  %(prog)s monitor
  
  # Quick status check
  %(prog)s status
  
  # Run optimization analysis
  %(prog)s optimize --data-dir /path/to/data
  
  # Show configuration
  %(prog)s config --type production
        """
    )
    
    # Global options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level')
    parser.add_argument('--config-type', choices=['default', 'production', 'development', 'testing'],
                       help='Configuration profile to use')
    
    # Create subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Sequential processing command
    parser_seq = subparsers.add_parser('sequential', help='Run sequential processing pipeline')
    parser_seq.add_argument('--input-dir', help='Input data directory')
    parser_seq.add_argument('--output-dir', help='Output directory')
    parser_seq.add_argument('--variables', nargs='+', choices=['pr', 'tas', 'tasmax', 'tasmin'],
                           help='Variables to process')
    parser_seq.add_argument('--regions', nargs='+', choices=['CONUS', 'AK', 'HI', 'PRVI', 'GU'],
                           help='Regions to process')
    parser_seq.set_defaults(func=cmd_sequential)
    
    # Parallel processing command
    parser_par = subparsers.add_parser('parallel', help='Run parallel processing pipeline')
    parser_par.add_argument('--input-dir', help='Input data directory')
    parser_par.add_argument('--output-dir', help='Output directory')
    parser_par.add_argument('--workers', type=int, help='Number of worker processes')
    parser_par.add_argument('--variables', nargs='+', choices=['pr', 'tas', 'tasmax', 'tasmin'],
                           help='Variables to process')
    parser_par.add_argument('--regions', nargs='+', choices=['CONUS', 'AK', 'HI', 'PRVI', 'GU'],
                           help='Regions to process')
    parser_par.set_defaults(func=cmd_parallel)
    
    # Monitor command
    parser_mon = subparsers.add_parser('monitor', help='Monitor processing progress')
    parser_mon.add_argument('--status-file', default='processing_progress.json',
                           help='Status file to monitor')
    parser_mon.add_argument('--progress-log', default='processing_progress.log',
                           help='Progress log file')
    parser_mon.add_argument('--refresh-interval', type=int, default=5,
                           help='Refresh interval in seconds')
    parser_mon.add_argument('--summary', action='store_true',
                           help='Show summary once and exit')
    parser_mon.add_argument('--log-lines', type=int, metavar='N',
                           help='Show N recent log lines and exit')
    parser_mon.set_defaults(func=cmd_monitor)
    
    # Status command
    parser_status = subparsers.add_parser('status', help='Quick status check')
    parser_status.set_defaults(func=cmd_status)
    
    # Optimize command
    parser_opt = subparsers.add_parser('optimize', help='Run optimization analysis')
    parser_opt.add_argument('--data-dir', required=True, help='Data directory for testing')
    parser_opt.set_defaults(func=cmd_optimize)
    
    # Benchmark command
    parser_bench = subparsers.add_parser('benchmark', help='Run multiprocessing benchmark')
    parser_bench.add_argument('--data-dir', required=True, help='Data directory for testing')
    parser_bench.add_argument('--num-files', type=int, default=10,
                             help='Number of files to test')
    parser_bench.set_defaults(func=cmd_benchmark)
    
    # Config command
    parser_config = subparsers.add_parser('config', help='Show configuration')
    parser_config.add_argument('--type', dest='config_type',
                              choices=['default', 'production', 'development', 'testing'],
                              help='Configuration type to show')
    parser_config.set_defaults(func=cmd_config)
    
    return parser


def main_cli():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n🛑 Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_cli() 