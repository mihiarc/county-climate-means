#!/usr/bin/env python3
"""
Convert county shapefile to GeoParquet format and set up shared data resources.

This script should be run once to initialize the county boundaries data in a modern,
efficient format that can be used by all phases of the pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from county_climate.shared.data import CountyBoundariesManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Convert county boundaries and set up data resources."""
    logger.info("Starting county boundaries conversion...")
    
    # Initialize manager
    manager = CountyBoundariesManager()
    
    # Convert shapefile to GeoParquet
    try:
        geoparquet_path = manager.ensure_geoparquet_exists(force_rebuild=True)
        logger.info(f"Successfully created GeoParquet at: {geoparquet_path}")
        
        # Test loading
        gdf = manager.load_counties()
        logger.info(f"Test load successful: {len(gdf)} counties loaded")
        
        # Show summary statistics
        logger.info("\nCounty Summary:")
        logger.info(f"Total counties: {len(gdf)}")
        
        region_counts = gdf['REGION'].value_counts()
        logger.info("\nCounties by region:")
        for region, count in region_counts.items():
            logger.info(f"  {region}: {count}")
            
        # Export optimized versions for each phase
        phase2_path = manager.data_dir / "county_boundaries_phase2.parquet"
        manager.export_for_phase('phase2', phase2_path)
        logger.info(f"Exported Phase 2 optimized file to: {phase2_path}")
        
        phase3_path = manager.data_dir / "county_boundaries_phase3.parquet"
        manager.export_for_phase('phase3', phase3_path)
        logger.info(f"Exported Phase 3 optimized file to: {phase3_path}")
        
        logger.info("\nConversion complete! County boundaries are now available in GeoParquet format.")
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()