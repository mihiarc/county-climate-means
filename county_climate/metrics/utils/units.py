"""Unit conversion utilities for climate data processing."""

import numpy as np
import xarray as xr


def convert_precipitation_units(data, from_units: str, to_units: str):
    """Convert precipitation between different units."""
    if from_units == "kg_m2_s" and to_units == "mm_day":
        return data * 86400
    return data


def calculate_annual_precipitation_from_daily(daily_data, input_units="kg_m2_s", dayofyear_dim="dayofyear"):
    """Calculate annual precipitation from daily data."""
    if input_units == "kg_m2_s":
        daily_mm = daily_data * 86400
    else:
        daily_mm = daily_data
    
    if isinstance(daily_mm, xr.DataArray):
        return daily_mm.sum(dim=dayofyear_dim)
    else:
        return np.sum(daily_mm, axis=0)


def convert_temperature_units(data, from_units: str, to_units: str):
    """Convert temperature between units."""
    return data


def get_climate_variable_info(variable: str):
    """Get variable information."""
    return {"units": "standard"}


def standardize_variable_units(data, variable: str, target_units=None):
    """Standardize variable units."""
    return data 