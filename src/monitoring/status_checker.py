#!/usr/bin/env python3
"""
Quick Status Check for Climate Normals Processing Pipeline

Direct directory-based status checker for reliable progress tracking.
"""

import os
from pathlib import Path
from collections import defaultdict
import time


class StatusChecker:
    """Quick status checker for climate processing pipeline."""
    
    def __init__(self, output_base_dir: str = "output/rolling_30year_climate_normals"):
        self.output_base_dir = Path(output_base_dir)
    
    def count_files_by_category(self):
        """Count output files by variable, period, and region."""
        if not self.output_base_dir.exists():
            return None, 0
        
        stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        total_files = 0
        
        variables = ['pr', 'tas', 'tasmax', 'tasmin']
        periods = ['historical', 'hybrid', 'ssp245']
        regions = ['CONUS', 'Alaska', 'Hawaii', 'PuertoRico', 'Guam']
        
        for variable in variables:
            for period in periods:
                period_path = self.output_base_dir / variable / period
                if period_path.exists():
                    files = list(period_path.glob("*.nc"))
                    total_files += len(files)
                    
                    # Count by region
                    for file in files:
                        for region in regions:
                            if region in file.name:
                                stats[variable][period][region] += 1
                                break
        
        return stats, total_files

    def calculate_targets(self):
        """Calculate expected number of files based on the processing configuration."""
        # Based on your processing setup:
        # Historical: 1980-2014 (35 years) * 5 regions = 175 per variable
        # Hybrid: 2015-2044 (30 years) * 5 regions = 150 per variable  
        # SSP245: varies by variable due to data availability
        
        targets = {
            'pr': {'historical': 35*5, 'hybrid': 30*5, 'ssp245': 17*5},  # to 2076
            'tas': {'historical': 35*5, 'hybrid': 30*5, 'ssp245': 56*5},  # to 2100
            'tasmax': {'historical': 35*5, 'hybrid': 30*5, 'ssp245': 56*5},  # to 2100
            'tasmin': {'historical': 35*5, 'hybrid': 30*5, 'ssp245': 56*5}   # to 2100
        }
        
        total_target = sum(sum(periods.values()) for periods in targets.values())
        return targets, total_target

    def get_latest_files(self):
        """Get the 3 most recently created files."""
        if not self.output_base_dir.exists():
            return []
        
        all_files = []
        for nc_file in self.output_base_dir.rglob("*.nc"):
            all_files.append((nc_file.stat().st_mtime, nc_file))
        
        all_files.sort(reverse=True)
        return [str(f[1]) for f in all_files[:3]]

    def quick_status(self):
        """Display a comprehensive status based on actual output files."""
        print("🔍 Climate Normals Processing Status (Direct File Scan)")
        print("=" * 60)
        
        stats, total_files = self.count_files_by_category()
        if stats is None:
            print("❌ Output directory not found - processing may not have started")
            return
        
        targets, total_target = self.calculate_targets()
        overall_pct = (total_files / total_target * 100) if total_target > 0 else 0
        
        print(f"📊 Overall Progress: {total_files}/{total_target} files ({overall_pct:.1f}%)")
        print()
        
        # Variable breakdown
        print("🔢 Progress by Variable:")
        for variable in ['pr', 'tas', 'tasmax', 'tasmin']:
            var_completed = sum(sum(stats[variable][period].values()) for period in ['historical', 'hybrid', 'ssp245'])
            var_target = sum(targets[variable].values())
            var_pct = (var_completed / var_target * 100) if var_target > 0 else 0
            print(f"   {variable.upper()}: {var_completed}/{var_target} ({var_pct:.1f}%)")
        print()
        
        # Period breakdown
        print("📅 Progress by Period:")
        for period in ['historical', 'hybrid', 'ssp245']:
            period_completed = sum(sum(stats[variable][period].values()) for variable in ['pr', 'tas', 'tasmax', 'tasmin'])
            period_target = sum(targets[variable][period] for variable in ['pr', 'tas', 'tasmax', 'tasmin'])
            period_pct = (period_completed / period_target * 100) if period_target > 0 else 0
            print(f"   {period}: {period_completed}/{period_target} ({period_pct:.1f}%)")
        print()
        
        # Latest activity
        recent_files = self.get_latest_files()
        if recent_files:
            print("⏰ Most Recent Files:")
            for i, file_path in enumerate(recent_files, 1):
                filename = Path(file_path).name
                mtime = os.path.getmtime(file_path)
                time_str = time.strftime("%H:%M:%S", time.localtime(mtime))
                print(f"   {i}. {filename} ({time_str})")
        else:
            print("⏰ No recent files found")


# Legacy functions for backward compatibility
def count_files_by_category():
    """Count output files by variable, period, and region."""
    checker = StatusChecker()
    return checker.count_files_by_category()

def calculate_targets():
    """Calculate expected number of files based on the processing configuration."""
    checker = StatusChecker()
    return checker.calculate_targets()

def get_latest_files():
    """Get the 3 most recently created files."""
    checker = StatusChecker()
    return checker.get_latest_files()

def quick_status():
    """Display a comprehensive status based on actual output files."""
    checker = StatusChecker()
    checker.quick_status()


if __name__ == "__main__":
    quick_status() 