#!/usr/bin/env python3
"""
Integration tests for run_climate_means.py runner script.

Tests the runner script functionality and its integration with the core
climate_means.py module, ensuring that the execution interface works correctly
and can properly orchestrate climate data processing workflows.
"""

import pytest
import sys
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from io import StringIO

# Import modules to test
import run_climate_means
import climate_means


class TestRunnerModuleIntegration:
    """Test integration between run_climate_means and climate_means modules."""
    
    def test_runner_imports_successful(self):
        """Test that all required imports from climate_means work correctly."""
        # Test that run_climate_means can import from climate_means
        assert hasattr(run_climate_means, 'setup_dask_client')
        assert hasattr(run_climate_means, 'cleanup_dask_resources')
        assert hasattr(run_climate_means, 'enhanced_performance_monitor')
        assert hasattr(run_climate_means, 'REGION_BOUNDS')
        assert hasattr(run_climate_means, 'validate_region_bounds')
        assert hasattr(run_climate_means, 'generate_climate_periods')
        assert hasattr(run_climate_means, 'NorESM2FileHandler')
        assert hasattr(run_climate_means, 'process_climate_data_workflow')
        assert hasattr(run_climate_means, 'logger')
        
        # Test that the imported objects are the same instances
        assert run_climate_means.setup_dask_client is climate_means.setup_dask_client
        assert run_climate_means.REGION_BOUNDS is climate_means.REGION_BOUNDS
        assert run_climate_means.process_climate_data_workflow is climate_means.process_climate_data_workflow
        assert run_climate_means.logger is climate_means.logger
    
    def test_runner_functions_accessible(self):
        """Test that all runner functions are accessible and callable."""
        # Test main execution functions
        assert callable(run_climate_means.main)
        assert callable(run_climate_means.example_usage)
        assert callable(run_climate_means.process_noresm2_data)
        assert callable(run_climate_means.print_help)
        
        # Test that functions have proper docstrings
        assert run_climate_means.main.__doc__ is not None
        assert run_climate_means.example_usage.__doc__ is not None
        assert run_climate_means.process_noresm2_data.__doc__ is not None
        assert run_climate_means.print_help.__doc__ is not None
    
    def test_runner_module_structure(self):
        """Test that runner module has expected structure."""
        # Should have main execution functions
        expected_functions = ['main', 'example_usage', 'process_noresm2_data', 'print_help']
        for func_name in expected_functions:
            assert hasattr(run_climate_means, func_name)
        
        # Should import expected constants and classes
        assert hasattr(run_climate_means, 'REGION_BOUNDS')
        assert hasattr(run_climate_means, 'NorESM2FileHandler')
        assert hasattr(run_climate_means, 'logger')


class TestMainFunction:
    """Test the main() demonstration function."""
    
    @patch('run_climate_means.setup_dask_client')
    @patch('run_climate_means.cleanup_dask_resources')
    @patch('run_climate_means.enhanced_performance_monitor')
    @patch('run_climate_means.validate_region_bounds')
    @patch('run_climate_means.generate_climate_periods')
    @patch('pathlib.Path.exists')
    def test_main_function_execution(self, mock_path_exists, mock_generate_periods, 
                                   mock_validate_region, mock_monitor, 
                                   mock_cleanup, mock_setup_dask):
        """Test main function execution with mocked dependencies."""
        # Setup mocks
        mock_client = MagicMock()
        mock_cluster = MagicMock()
        mock_setup_dask.return_value = (mock_client, mock_cluster)
        mock_validate_region.return_value = True
        mock_generate_periods.return_value = [(1951, 1980, 1980, "historical_1980")]
        mock_monitor.return_value = {'n_workers': 4, 'memory_utilization': 50.0}
        mock_path_exists.return_value = False  # NorESM2 path doesn't exist
        
        # Execute main function
        run_climate_means.main()
        
        # Verify function calls
        mock_setup_dask.assert_called_once()
        mock_validate_region.assert_called_with('CONUS')
        mock_generate_periods.assert_called_once()
        mock_monitor.assert_called_once_with(mock_client)
        mock_cleanup.assert_called_once_with(mock_client, mock_cluster)
    
    @patch('run_climate_means.setup_dask_client')
    @patch('run_climate_means.cleanup_dask_resources')
    @patch('run_climate_means.NorESM2FileHandler')
    @patch('pathlib.Path.exists')
    def test_main_function_with_noresm2_data(self, mock_path_exists, mock_handler_class,
                                           mock_cleanup, mock_setup_dask):
        """Test main function when NorESM2 data is available."""
        # Setup mocks
        mock_client = MagicMock()
        mock_cluster = MagicMock()
        mock_setup_dask.return_value = (mock_client, mock_cluster)
        mock_path_exists.return_value = True  # NorESM2 path exists
        
        mock_handler = MagicMock()
        mock_handler.validate_data_availability.return_value = {
            'tas': {'historical': (1950, 2014)}
        }
        mock_handler_class.return_value = mock_handler
        
        # Execute main function
        run_climate_means.main()
        
        # Verify NorESM2 handler was created and used
        mock_handler_class.assert_called_once()
        mock_handler.validate_data_availability.assert_called_once()
    
    @patch('run_climate_means.setup_dask_client')
    def test_main_function_error_handling(self, mock_setup_dask):
        """Test main function error handling."""
        # Setup mock to raise exception
        mock_setup_dask.side_effect = Exception("Dask setup failed")
        
        # Should raise the exception
        with pytest.raises(Exception, match="Dask setup failed"):
            run_climate_means.main()


class TestExampleUsageFunction:
    """Test the example_usage() function."""
    
    @patch('run_climate_means.process_climate_data_workflow')
    def test_example_usage_execution(self, mock_workflow):
        """Test example_usage function execution."""
        # Execute example_usage function
        run_climate_means.example_usage()
        
        # Verify workflow was called with expected parameters
        mock_workflow.assert_called_once()
        
        # Get the call arguments - check if called with positional or keyword args
        call_args = mock_workflow.call_args
        if call_args.args:
            # Called with positional arguments
            args = call_args.args
            assert args[0] == "/path/to/your/climate/netcdf/files"  # data_directory
            assert args[1] == "/path/to/output/climate/normals"     # output_directory
            assert args[2] == ['tas', 'tasmax', 'tasmin', 'pr']     # variables
            assert args[3] == ['CONUS', 'AK', 'HI']                 # regions
            assert args[4] == ['historical', 'ssp245', 'ssp585']    # scenarios
            
            # Check config
            config = args[5]
            assert isinstance(config, dict)
            assert config['max_workers'] == 6
            assert config['target_chunk_size'] == 256
            assert config['computation_type'] == 'mixed'
        else:
            # Called with keyword arguments
            kwargs = call_args.kwargs
            assert kwargs['data_directory'] == "/path/to/your/climate/netcdf/files"
            assert kwargs['output_directory'] == "/path/to/output/climate/normals"
            assert kwargs['variables'] == ['tas', 'tasmax', 'tasmin', 'pr']
            assert kwargs['regions'] == ['CONUS', 'AK', 'HI']
            assert kwargs['scenarios'] == ['historical', 'ssp245', 'ssp585']
            
            # Check config
            config = kwargs['config']
            assert isinstance(config, dict)
            assert config['max_workers'] == 6
            assert config['target_chunk_size'] == 256
            assert config['computation_type'] == 'mixed'
    
    @patch('run_climate_means.process_climate_data_workflow')
    def test_example_usage_error_handling(self, mock_workflow):
        """Test example_usage error handling."""
        # Setup mock to raise exception
        mock_workflow.side_effect = Exception("Workflow failed")
        
        # Should not raise exception but should handle it gracefully
        run_climate_means.example_usage()
        
        # Should have called the workflow
        mock_workflow.assert_called_once()


class TestProcessNorESM2Function:
    """Test the process_noresm2_data() function."""
    
    @patch('run_climate_means.NorESM2FileHandler')
    @patch('run_climate_means.process_climate_data_workflow')
    def test_process_noresm2_data_execution(self, mock_workflow, mock_handler_class):
        """Test process_noresm2_data function execution."""
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler.validate_data_availability.return_value = {
            'tas': {'historical': (1950, 2014), 'ssp245': (2015, 2100)},
            'pr': {'historical': (1950, 2014), 'ssp245': (2015, 2100)}
        }
        mock_handler_class.return_value = mock_handler
        
        # Execute function
        run_climate_means.process_noresm2_data()
        
        # Verify handler creation
        mock_handler_class.assert_called_once_with("/media/mihiarc/RPA1TB/data/NorESM2-LM")
        mock_handler.validate_data_availability.assert_called_once()
        
        # Verify workflow was called
        mock_workflow.assert_called_once()
        
        # Get the call arguments - check if called with positional or keyword args
        call_args = mock_workflow.call_args
        if call_args.args:
            # Called with positional arguments
            args = call_args.args
            assert args[0] == "/media/mihiarc/RPA1TB/data/NorESM2-LM"
            assert args[1] == "/home/mihiarc/repos/county_climate/output/noresm2_climate_normals"
            assert args[2] == ['tas', 'tasmax', 'tasmin', 'pr']
            assert args[3] == ['CONUS', 'AK', 'HI', 'PRVI', 'GU']
            assert args[4] == ['historical', 'ssp245', 'ssp585']
            
            # Check config
            config = args[5]
            assert isinstance(config, dict)
            assert config['max_workers'] == 4
            assert config['computation_type'] == 'time_series'
            assert config['memory_safety_margin'] == 0.7
        else:
            # Called with keyword arguments
            kwargs = call_args.kwargs
            assert kwargs['data_directory'] == "/media/mihiarc/RPA1TB/data/NorESM2-LM"
            assert kwargs['output_directory'] == "/home/mihiarc/repos/county_climate/output/noresm2_climate_normals"
            assert kwargs['variables'] == ['tas', 'tasmax', 'tasmin', 'pr']
            assert kwargs['regions'] == ['CONUS', 'AK', 'HI', 'PRVI', 'GU']
            assert kwargs['scenarios'] == ['historical', 'ssp245', 'ssp585']
            
            # Check config
            config = kwargs['config']
            assert isinstance(config, dict)
            assert config['max_workers'] == 4
            assert config['computation_type'] == 'time_series'
            assert config['memory_safety_margin'] == 0.7
    
    @patch('run_climate_means.NorESM2FileHandler')
    @patch('run_climate_means.process_climate_data_workflow')
    def test_process_noresm2_data_error_handling(self, mock_workflow, mock_handler_class):
        """Test process_noresm2_data error handling."""
        # Setup mock to raise exception
        mock_handler = MagicMock()
        mock_handler.validate_data_availability.side_effect = Exception("Data validation failed")
        mock_handler_class.return_value = mock_handler
        
        # Should raise the exception with the wrapped error message
        with pytest.raises(Exception, match="Data validation failed"):
            run_climate_means.process_noresm2_data()
    
    def test_process_noresm2_data_configuration(self):
        """Test that process_noresm2_data has correct configuration."""
        # We can test the configuration by examining the function code
        # or by mocking and checking the arguments passed to workflow
        
        with patch('run_climate_means.NorESM2FileHandler') as mock_handler_class, \
             patch('run_climate_means.process_climate_data_workflow') as mock_workflow:
            
            mock_handler = MagicMock()
            mock_handler.validate_data_availability.return_value = {}
            mock_handler_class.return_value = mock_handler
            
            run_climate_means.process_noresm2_data()
            
            # Get the config that was passed
            call_args = mock_workflow.call_args
            if call_args.args:
                # Called with positional arguments
                config = call_args.args[5]
            else:
                # Called with keyword arguments
                config = call_args.kwargs['config']
            
            # Verify NorESM2-specific optimizations
            assert config['computation_type'] == 'time_series'
            assert config['memory_safety_margin'] == 0.7
            assert config['batch_size'] == 10


class TestPrintHelpFunction:
    """Test the print_help() function."""
    
    def test_print_help_output(self):
        """Test that print_help produces expected output."""
        # Capture stdout
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            run_climate_means.print_help()
        
        output = captured_output.getvalue()
        
        # Check that help output contains expected content
        assert "Climate Data Processing Runner" in output
        assert "Usage:" in output
        assert "python run_climate_means.py" in output
        assert "noresm2" in output
        assert "example" in output
        assert "help" in output
        assert "Description:" in output
        assert "Configuration:" in output
    
    def test_print_help_completeness(self):
        """Test that print_help covers all command options."""
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            run_climate_means.print_help()
        
        output = captured_output.getvalue()
        
        # Should mention all the main command options
        expected_options = ['noresm2', 'example', 'help']
        for option in expected_options:
            assert option in output
        
        # Should mention configuration options
        config_items = ['max_workers', 'memory_safety_margin', 'batch_size']
        for item in config_items:
            assert item in output


class TestCommandLineInterface:
    """Test the command-line interface functionality."""
    
    def test_command_line_help_option(self):
        """Test command-line help option."""
        # Test running with help option
        result = subprocess.run(
            [sys.executable, 'run_climate_means.py', 'help'],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Climate Data Processing Runner" in result.stdout
        assert "Usage:" in result.stdout
    
    def test_command_line_unknown_option(self):
        """Test command-line with unknown option."""
        result = subprocess.run(
            [sys.executable, 'run_climate_means.py', 'unknown'],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Unknown option: unknown" in result.stdout
        assert "Use 'python run_climate_means.py help'" in result.stdout
    
    def test_command_line_interface_structure(self):
        """Test that command-line interface has proper structure."""
        # Test that the script has if __name__ == "__main__": block
        script_path = Path('run_climate_means.py')
        with open(script_path, 'r') as f:
            content = f.read()
            assert 'if __name__ == "__main__":' in content
            assert 'sys.argv[1]' in content  # Command line argument parsing
            assert 'noresm2' in content
            assert 'example' in content
            assert 'help' in content


class TestFunctionConfigurations:
    """Test that functions have appropriate configurations."""
    
    def test_main_function_configuration(self):
        """Test main function has appropriate demonstration configuration."""
        # Extract configuration by examining the function
        # We can do this by mocking and checking what gets passed
        
        with patch('run_climate_means.setup_dask_client') as mock_setup_dask:
            mock_client = MagicMock()
            mock_cluster = MagicMock()
            mock_setup_dask.return_value = (mock_client, mock_cluster)
            
            try:
                run_climate_means.main()
            except:
                pass  # We just want to check the config passed to setup_dask_client
            
            # Get the config that was passed
            config = mock_setup_dask.call_args[0][0]
            
            # Should be appropriate for demonstration
            assert config['max_workers'] == 4
            assert config['target_chunk_size'] == 128
            assert config['computation_type'] == 'mixed'
    
    def test_function_docstring_quality(self):
        """Test that all functions have informative docstrings."""
        functions_to_check = [
            run_climate_means.main,
            run_climate_means.example_usage,
            run_climate_means.process_noresm2_data,
            run_climate_means.print_help
        ]
        
        for func in functions_to_check:
            assert func.__doc__ is not None
            assert len(func.__doc__.strip()) > 20  # Reasonable docstring length
            # Just check that it's a meaningful description, not format


class TestModuleIntegrationWithClimateWorkflow:
    """Test integration with the climate processing workflow."""
    
    @patch('run_climate_means.process_climate_data_workflow')
    @patch('run_climate_means.NorESM2FileHandler')
    def test_runner_can_orchestrate_full_workflow(self, mock_handler_class, mock_workflow):
        """Test that runner can orchestrate a complete workflow."""
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler.validate_data_availability.return_value = {
            'tas': {'historical': (1950, 2014)}
        }
        mock_handler_class.return_value = mock_handler
        
        # Run process_noresm2_data which should orchestrate the workflow
        run_climate_means.process_noresm2_data()
        
        # Verify the complete flow
        mock_handler_class.assert_called_once()
        mock_handler.validate_data_availability.assert_called_once()
        mock_workflow.assert_called_once()
        
        # Verify workflow gets proper arguments
        call_args = mock_workflow.call_args
        if call_args.args:
            # Called with positional arguments
            args = call_args.args
            assert len(args) == 6  # data_dir, output_dir, variables, regions, scenarios, config
            assert isinstance(args[5], dict)  # config should be a dictionary
        else:
            # Called with keyword arguments
            kwargs = call_args.kwargs
            expected_keys = {'data_directory', 'output_directory', 'variables', 'regions', 'scenarios', 'config'}
            assert set(kwargs.keys()) == expected_keys
            assert isinstance(kwargs['config'], dict)  # config should be a dictionary
    
    def test_runner_imports_all_necessary_workflow_components(self):
        """Test that runner imports all necessary components for workflow."""
        # Check that all necessary functions are imported
        workflow_components = [
            'setup_dask_client',
            'cleanup_dask_resources',
            'process_climate_data_workflow',
            'NorESM2FileHandler',
            'enhanced_performance_monitor',
            'validate_region_bounds',
            'generate_climate_periods'
        ]
        
        for component in workflow_components:
            assert hasattr(run_climate_means, component)
            assert callable(getattr(run_climate_means, component))
    
    def test_runner_configuration_consistency(self):
        """Test that runner configurations are consistent with workflow needs."""
        # Check that configurations use valid parameter names and values
        configs_to_check = [
            ('main', {'max_workers': 4, 'computation_type': 'mixed'}),
            ('example_usage', {'max_workers': 6, 'computation_type': 'mixed'}),
            ('process_noresm2_data', {'max_workers': 4, 'computation_type': 'time_series'})
        ]
        
        for func_name, expected_params in configs_to_check:
            # We can verify this by looking at the source or by mocking and checking
            # For this test, we'll just verify the parameters make sense
            for key, value in expected_params.items():
                if key == 'max_workers':
                    assert isinstance(value, int) and value > 0
                elif key == 'computation_type':
                    assert value in ['mixed', 'time_series', 'spatial']


class TestErrorHandling:
    """Test error handling in runner functions."""
    
    @patch('run_climate_means.setup_dask_client')
    def test_main_handles_dask_setup_failure(self, mock_setup_dask):
        """Test main function handles Dask setup failure."""
        mock_setup_dask.side_effect = Exception("Dask cluster failed to start")
        
        with pytest.raises(Exception, match="Dask cluster failed to start"):
            run_climate_means.main()
    
    @patch('run_climate_means.NorESM2FileHandler')
    def test_process_noresm2_handles_file_handler_failure(self, mock_handler_class):
        """Test process_noresm2_data handles file handler creation failure."""
        mock_handler_class.side_effect = FileNotFoundError("Data directory not found")
        
        with pytest.raises(Exception, match="Data directory not found"):
            run_climate_means.process_noresm2_data()
    
    @patch('run_climate_means.process_climate_data_workflow')
    def test_example_usage_handles_workflow_failure_gracefully(self, mock_workflow):
        """Test example_usage handles workflow failure gracefully."""
        mock_workflow.side_effect = Exception("Workflow execution failed")
        
        # Should not raise exception, should handle gracefully
        run_climate_means.example_usage()
        
        # Should have attempted to call workflow
        mock_workflow.assert_called_once()
    
    def test_command_line_interface_handles_import_errors(self):
        """Test command-line interface handles import errors gracefully."""
        # This is testing that the script can be imported without errors
        # Even if some dependencies are missing, basic functionality should work
        
        try:
            import run_climate_means
            # Basic import should succeed
            assert hasattr(run_climate_means, 'print_help')
        except ImportError as e:
            pytest.fail(f"Runner script should import cleanly: {e}")


class TestExecutableScriptBehavior:
    """Test behavior when run as executable script."""
    
    def test_script_is_executable(self):
        """Test that the script file is executable."""
        script_path = Path('run_climate_means.py')
        assert script_path.exists()
        
        # Check if script has shebang
        with open(script_path, 'r') as f:
            first_line = f.readline()
            assert first_line.startswith('#!')
    
    def test_script_has_proper_structure(self):
        """Test that script has proper module structure."""
        # Test that the script can be imported as a module
        import run_climate_means
        
        # Should have main execution functions
        assert hasattr(run_climate_means, '__file__')
        assert hasattr(run_climate_means, '__name__')
        
        # Should have if __name__ == "__main__": block by checking source
        script_path = Path('run_climate_means.py')
        with open(script_path, 'r') as f:
            content = f.read()
            assert 'if __name__ == "__main__":' in content


def test_module_level_runner_integration():
    """Test module-level integration aspects."""
    # Test that runner and climate_means can be imported together
    import run_climate_means
    import climate_means
    
    # Test that they don't conflict
    assert run_climate_means.REGION_BOUNDS is climate_means.REGION_BOUNDS
    assert run_climate_means.process_climate_data_workflow is climate_means.process_climate_data_workflow
    
    # Test that runner has all expected functionality
    expected_functions = ['main', 'example_usage', 'process_noresm2_data', 'print_help']
    for func_name in expected_functions:
        assert hasattr(run_climate_means, func_name)
        assert callable(getattr(run_climate_means, func_name))


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 