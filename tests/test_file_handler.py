#!/usr/bin/env python3
import means
from means.utils.io_util import NorESM2FileHandler

# Test file handler
handler = NorESM2FileHandler('/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM')
files = handler.get_files_for_period('pr', 'historical', 2010, 2014)
print(f'Found {len(files)} files for 2010-2014')
for f in files[:3]:
    print(f'  {f}')

# Test target years
print(f'\nTesting target years for historical period:')
for year in [1980, 1990, 2000, 2010, 2014]:
    start_year = year - 29
    end_year = year
    files = handler.get_files_for_period('pr', 'historical', start_year, end_year)
    print(f'  Target {year}: {start_year}-{end_year} -> {len(files)} files') 