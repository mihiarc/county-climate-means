#!/usr/bin/env python3
"""
Main Entry Point for Climate Data Processing

This script provides the main entry point for the climate data processing system.
It imports and runs the CLI interface from the reorganized src package.

Usage:
    python main.py sequential
    python main.py parallel --workers 8
    python main.py monitor
    python main.py status
    python main.py config --type production
    
For help:
    python main.py --help
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the CLI
from src.cli import main_cli

if __name__ == "__main__":
    main_cli() 