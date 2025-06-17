#!/usr/bin/env python3
"""
Configuration-Driven Climate Data Processing - Main Entry Point

Modern, configuration-driven entry point for processing climate data using
the new pipeline orchestration system. Supports flexible pipeline definitions,
automated dependency management, and optimized resource utilization.

Features:
- Configuration-driven pipeline execution
- Automatic stage dependency resolution
- Resource-aware processing with 95GB RAM / 56 core optimization
- Integrated means and metrics processing
- Real-time monitoring and progress tracking
- Error recovery and retry logic
- Production-ready logging and validation

Usage Examples:
    # Run with production configuration
    python main_orchestrated.py run --config configs/production_high_performance.yaml
    
    # Run specific processing profile
    python main_orchestrated.py run --profile conus_temperature_historical
    
    # Create sample configurations
    python main_orchestrated.py create-configs --output-dir configs/
    
    # Monitor pipeline execution
    python main_orchestrated.py status --execution-id exec_12345
    
    # Validate configuration
    python main_orchestrated.py validate --config configs/production_high_performance.yaml
"""

import argparse
import asyncio
import logging
import sys
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import the new configuration-driven components
from county_climate.shared.config import (
    ConfigurationLoader,
    ConfigurationManager,
    ConfigurationError,
    PipelineConfiguration,
    ProcessingStage,
)
from county_climate.shared.orchestration import (
    PipelineOrchestrator,
    PipelineRunner,
    ExecutionStatus,
)
from county_climate.shared.examples.config_examples import create_sample_config_files

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('climate_orchestrated.log')
    ]
)
logger = logging.getLogger(__name__)


async def run_pipeline_command(args) -> bool:
    """Run a climate processing pipeline using configuration-driven orchestration."""
    logger.info("🚀 Starting configuration-driven climate processing pipeline")
    
    try:
        # Load configuration
        config_manager = ConfigurationManager(config_dir=Path("configs"))
        
        if args.config:
            logger.info(f"📄 Loading configuration from: {args.config}")
            config = config_manager.load_config(args.config, args.environment)
        elif args.profile:
            logger.info(f"📋 Loading processing profile: {args.profile}")
            # This would load a profile and convert to pipeline config
            # For now, use default config
            config = config_manager.load_config("production_high_performance", args.environment)
        else:
            logger.info("📄 Using default production configuration")
            config = config_manager.load_config("production_high_performance", args.environment)
        
        # Validate configuration
        logger.info("✅ Validating pipeline configuration")
        warnings = config_manager.validate_active_config()
        if warnings:
            logger.warning(f"⚠️  Configuration warnings: {warnings}")
        
        # Create pipeline runner
        runner = PipelineRunner()
        
        # Create orchestrator manually since runner.orchestrator may be None initially
        from county_climate.shared.orchestration import PipelineOrchestrator
        orchestrator = PipelineOrchestrator(config, logger=logger)
        runner.orchestrator = orchestrator
        
        # Register stage handlers for means and metrics
        await _register_stage_handlers_direct(orchestrator)
        
        # Execute pipeline
        logger.info(f"🎯 Executing pipeline: {config.pipeline_id}")
        logger.info(f"📊 Processing stages: {[s.stage_id for s in config.stages]}")
        
        execution_id = f"exec_{int(time.time())}"
        if args.execution_id:
            execution_id = args.execution_id
        
        # Run the pipeline using the orchestrator directly (not the runner)
        start_time = time.time()
        execution = await orchestrator.execute_pipeline(
            execution_id=execution_id,
            resume_from_failure=args.resume
        )
        end_time = time.time()
        
        # Report results
        duration = end_time - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"🎊 PIPELINE EXECUTION COMPLETED")
        logger.info(f"{'='*80}")
        logger.info(f"📋 Pipeline ID: {execution.pipeline_id}")
        logger.info(f"🆔 Execution ID: {execution.execution_id}")
        logger.info(f"📊 Status: {execution.status}")
        logger.info(f"⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        # Stage summaries
        completed_stages = []
        failed_stages = []
        
        for stage_id, stage_execution in execution.stage_executions.items():
            if stage_execution.status == ExecutionStatus.COMPLETED:
                completed_stages.append(stage_id)
                logger.info(f"✅ {stage_id}: {stage_execution.duration_seconds:.1f}s")
            elif stage_execution.status == ExecutionStatus.FAILED:
                failed_stages.append(stage_id)
                logger.error(f"❌ {stage_id}: {stage_execution.error_message}")
        
        logger.info(f"\n📈 Summary:")
        logger.info(f"✅ Completed stages ({len(completed_stages)}): {completed_stages}")
        
        if failed_stages:
            logger.warning(f"❌ Failed stages ({len(failed_stages)}): {failed_stages}")
            return False
        
        # Save execution report
        if args.save_report:
            report_path = Path(f"execution_report_{execution_id}.yaml")
            await _save_execution_report(execution, config, report_path)
            logger.info(f"📄 Execution report saved: {report_path}")
        
        return execution.status == ExecutionStatus.COMPLETED
        
    except Exception as e:
        logger.error(f"💥 Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def _register_stage_handlers_direct(orchestrator):
    """Register stage handlers for the pipeline runner."""
    
    # Import handlers
    from county_climate.means.integration import means_stage_handler
    from county_climate.validation.integration import validation_stage_handler
    
    # Check if we should use special handlers
    pipeline_config = orchestrator.config
    use_v2_architecture = False
    use_flexible_architecture = False
    
    # Check if any means stage specifies a special handler
    for stage in pipeline_config.stages:
        if stage.stage_type == ProcessingStage.MEANS:
            if stage.entry_point == "means_stage_handler_v2":
                use_v2_architecture = True
                break
            elif stage.entry_point == "flexible_means_stage_handler":
                use_flexible_architecture = True
                break
    
    if use_flexible_architecture:
        logger.info("🚀 Using flexible architecture for means processing (supports SSP585)")
        try:
            from county_climate.means.integration.flexible_stage_handlers import flexible_means_stage_handler
            orchestrator.register_stage_handler(ProcessingStage.MEANS, flexible_means_stage_handler)
            logger.info("✅ Registered flexible means handler (multi-scenario support)")
        except ImportError as e:
            logger.warning(f"Flexible means handler not available, falling back to original: {e}")
            orchestrator.register_stage_handler(ProcessingStage.MEANS, means_stage_handler)
            logger.info("✅ Registered original means handler")
    elif use_v2_architecture:
        logger.info("🚀 Using V2 parallel variables architecture for means processing")
        try:
            from county_climate.means.integration.stage_handlers_v2 import means_stage_handler_v2
            orchestrator.register_stage_handler(ProcessingStage.MEANS, means_stage_handler_v2)
            logger.info("✅ Registered V2 means handler (parallel variables)")
        except ImportError as e:
            logger.warning(f"V2 means handler not available, falling back to original: {e}")
            orchestrator.register_stage_handler(ProcessingStage.MEANS, means_stage_handler)
            logger.info("✅ Registered original means handler")
    else:
        # Use original handler
        orchestrator.register_stage_handler(ProcessingStage.MEANS, means_stage_handler)
        logger.info("✅ Registered original means handler")
    
    # Register validation handler
    orchestrator.register_stage_handler(ProcessingStage.VALIDATION, validation_stage_handler)
    logger.info("✅ Registered validation handler")
    
    # Try to import metrics handler
    try:
        from county_climate.metrics.integration import metrics_stage_handler
        orchestrator.register_stage_handler(ProcessingStage.METRICS, metrics_stage_handler)
        logger.info("✅ Registered metrics handler")
    except ImportError as e:
        logger.warning(f"Metrics handler import failed: {e}")
        
        logger.error("❌ Failed to import metrics handler - metrics processing will not be available")
        logger.error("Please ensure county_climate.metrics package is properly installed")
        logger.error(f"Import error: {e}")
        raise ImportError(f"Cannot import metrics handler: {e}")
    
    logger.info("🔧 All stage handlers registered successfully")


async def _save_execution_report(execution, config: PipelineConfiguration, report_path: Path):
    """Save a detailed execution report."""
    
    report = {
        'execution_summary': {
            'pipeline_id': execution.pipeline_id,
            'execution_id': execution.execution_id,
            'status': execution.status.value,
            'start_time': execution.start_time.isoformat() if execution.start_time else None,
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'total_stages': len(config.stages),
            'completed_stages': len([s for s in execution.stage_executions.values() 
                                   if s.status == ExecutionStatus.COMPLETED]),
            'failed_stages': len([s for s in execution.stage_executions.values() 
                                if s.status == ExecutionStatus.FAILED]),
        },
        'stage_details': {},
        'configuration': config.dict(),
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    
    # Add stage execution details
    for stage_id, stage_execution in execution.stage_executions.items():
        report['stage_details'][stage_id] = {
            'status': stage_execution.status.value,
            'start_time': stage_execution.start_time.isoformat() if stage_execution.start_time else None,
            'end_time': stage_execution.end_time.isoformat() if stage_execution.end_time else None,
            'duration_seconds': stage_execution.duration_seconds,
            'attempt_number': stage_execution.attempt_number,
            'error_message': stage_execution.error_message,
            'output_data_summary': {
                k: str(v)[:100] + "..." if len(str(v)) > 100 else v
                for k, v in stage_execution.output_data.items()
            }
        }
    
    # Save report
    with open(report_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False, indent=2)


def validate_config_command(args) -> bool:
    """Validate a pipeline configuration file."""
    logger.info(f"✅ Validating configuration: {args.config}")
    
    try:
        loader = ConfigurationLoader()
        config = loader.load_pipeline_config(args.config, validate=True)
        
        logger.info(f"✅ Configuration is valid")
        logger.info(f"📋 Pipeline ID: {config.pipeline_id}")
        logger.info(f"🏷️  Pipeline Name: {config.pipeline_name}")
        logger.info(f"🌍 Environment: {config.environment}")
        logger.info(f"📊 Stages: {len(config.stages)}")
        
        # Run validation checks
        warnings = loader.validate_config(config)
        if warnings:
            logger.warning("⚠️  Configuration warnings:")
            for warning in warnings:
                logger.warning(f"   • {warning}")
        else:
            logger.info("✅ No validation warnings")
        
        # Show execution order
        execution_order = config.get_execution_order()
        logger.info(f"🔄 Execution order: {execution_order}")
        
        return True
        
    except ConfigurationError as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"💥 Validation error: {e}")
        return False


def status_command(args) -> bool:
    """Show pipeline status and system information."""
    logger.info("📋 Climate Processing System Status")
    
    try:
        # System information
        import psutil
        
        logger.info(f"\n💻 System Information:")
        logger.info(f"   Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB total, "
                   f"{psutil.virtual_memory().available / (1024**3):.1f} GB available")
        logger.info(f"   CPU cores: {psutil.cpu_count()} total, {psutil.cpu_count(logical=False)} physical")
        logger.info(f"   Load average: {psutil.getloadavg()}")
        
        # Data paths
        data_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
        if data_path.exists():
            logger.info(f"\n📁 Data Availability:")
            logger.info(f"   Input data path: {data_path}")
            
            # Check available variables
            variables = [d.name for d in data_path.iterdir() if d.is_dir() and d.name != '.']
            logger.info(f"   Available variables: {variables}")
            
            # Check scenarios for temperature
            tas_path = data_path / "tas"
            if tas_path.exists():
                scenarios = [d.name for d in tas_path.iterdir() if d.is_dir()]
                logger.info(f"   Available scenarios (tas): {scenarios}")
        else:
            logger.warning(f"❌ Data path not found: {data_path}")
        
        # Configuration availability
        config_dir = Path("configs")
        if config_dir.exists():
            configs = list(config_dir.glob("*.yaml"))
            logger.info(f"\n⚙️  Available configurations:")
            for config_file in configs:
                logger.info(f"   📄 {config_file.name}")
        else:
            logger.warning("⚠️  No configs directory found")
        
        # Check for active executions
        if args.execution_id:
            logger.info(f"\n🔍 Checking execution: {args.execution_id}")
            # This would check for active execution status
            logger.info("   Status tracking not yet implemented")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Status check failed: {e}")
        return False


def create_configs_command(args) -> bool:
    """Create sample configuration files."""
    logger.info("📝 Creating sample configuration files")
    
    try:
        output_dir = Path(args.output_dir or "configs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample configurations
        configs = create_sample_config_files()
        
        created_files = []
        for filename, config_data in configs.items():
            output_file = output_dir / filename
            with open(output_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            created_files.append(output_file)
        
        logger.info(f"✅ Created {len(created_files)} configuration files in {output_dir}/")
        for file_path in created_files:
            logger.info(f"   📄 {file_path.name}")
        
        # Create production config optimized for user's hardware
        prod_config_path = output_dir / "production_optimized.yaml"
        with open(prod_config_path, 'w') as f:
            with open("configs/production_high_performance.yaml", 'r') as prod_f:
                f.write(prod_f.read())
        logger.info(f"   📄 {prod_config_path.name} (optimized for your 95GB/56-core system)")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Configuration creation failed: {e}")
        return False


def list_configs_command(args) -> bool:
    """List available configurations and profiles."""
    logger.info("📋 Available Configurations")
    
    try:
        config_manager = ConfigurationManager(config_dir=Path("configs"))
        configs = config_manager.list_configs()
        
        if configs:
            logger.info(f"\n⚙️  Available configuration files ({len(configs)}):")
            for config_name in configs:
                logger.info(f"   📄 {config_name}")
        else:
            logger.info("   No configuration files found")
        
        # Show sample command usage
        logger.info(f"\n💡 Usage examples:")
        logger.info(f"   python {sys.argv[0]} run --config production_high_performance")
        logger.info(f"   python {sys.argv[0]} validate --config production_high_performance.yaml")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Failed to list configurations: {e}")
        return False


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Configuration-Driven Climate Data Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run --config configs/production_high_performance.yaml
  %(prog)s run --profile conus_temperature_historical  
  %(prog)s validate --config configs/production_high_performance.yaml
  %(prog)s status --execution-id exec_12345
  %(prog)s create-configs --output-dir configs/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run pipeline command
    run_parser = subparsers.add_parser('run', help='Run a climate processing pipeline')
    run_parser.add_argument('--config', '-c', help='Configuration file to use')
    run_parser.add_argument('--profile', '-p', help='Processing profile to use')
    run_parser.add_argument('--environment', '-e', choices=['development', 'testing', 'staging', 'production'],
                           default='production', help='Environment type')
    run_parser.add_argument('--execution-id', help='Custom execution ID')
    run_parser.add_argument('--resume', action='store_true', help='Resume from previous failure')
    run_parser.add_argument('--save-report', action='store_true', default=True, help='Save execution report')
    
    # Validate configuration command
    validate_parser = subparsers.add_parser('validate', help='Validate a pipeline configuration')
    validate_parser.add_argument('config', help='Configuration file to validate')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status and information')
    status_parser.add_argument('--execution-id', help='Check status of specific execution')
    
    # Create configurations command
    config_parser = subparsers.add_parser('create-configs', help='Create sample configuration files')
    config_parser.add_argument('--output-dir', '-o', default='configs', help='Output directory for configs')
    
    # List configurations command
    subparsers.add_parser('list-configs', help='List available configurations')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    logger.info(f"🚀 Starting command: {args.command}")
    
    try:
        if args.command == 'run':
            success = asyncio.run(run_pipeline_command(args))
        elif args.command == 'validate':
            success = validate_config_command(args)
        elif args.command == 'status':
            success = status_command(args)
        elif args.command == 'create-configs':
            success = create_configs_command(args)
        elif args.command == 'list-configs':
            success = list_configs_command(args)
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
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())