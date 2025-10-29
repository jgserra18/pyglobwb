"""
Pytest configuration and fixtures for water balance model tests.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add pyglobwb directory to path
pyglobwb_path = Path(__file__).parent.parent / 'pyglobwb'
sys.path.insert(0, str(pyglobwb_path))

from water_balance_model import (
    SoilParameters,
    CropParameters,
    ClimateData,
    WaterBalanceModel
)
from config_manager import ConfigManager


@pytest.fixture
def sample_dates():
    """Generate sample date range for testing."""
    return pd.date_range('2020-01-01', '2020-12-31', freq='D')


@pytest.fixture
def sample_climate_data(sample_dates):
    """Generate sample climate data."""
    n_days = len(sample_dates)
    day_of_year = sample_dates.dayofyear
    
    # Simple sinusoidal pattern
    precipitation = 2.0 + 1.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    precipitation = np.maximum(0, precipitation)
    
    pet = 2.0 + 2.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + 1.0
    
    return ClimateData(
        precipitation=precipitation,
        pet=pet,
        dates=sample_dates
    )


@pytest.fixture
def sample_soil_parameters():
    """Generate sample soil parameters."""
    return SoilParameters(
        smax_base=150.0,
        reference_depth=0.6,
        rmax=10.0,
        initial_sm=75.0,
        calibration_factor=2.4
    )


@pytest.fixture
def sample_crop_parameters(sample_dates):
    """Generate sample crop parameters (Maize)."""
    n_days = len(sample_dates)
    
    # Monthly Kc for Maize
    monthly_kc = [0, 0, 0, 0.3, 0.64, 1.17, 1.2, 0.70, 0, 0, 0, 0]
    
    # Expand to daily
    kc_daily = np.zeros(n_days)
    for i, date in enumerate(sample_dates):
        month_idx = date.month - 1
        kc_daily[i] = monthly_kc[month_idx]
    
    # Simple rooting depth
    rooting_depth = np.full(n_days, 1.0)
    
    return CropParameters(
        name='Maize',
        kc_values=kc_daily,
        rooting_depth=rooting_depth,
        dates=sample_dates,
        landuse_kc=0.5
    )


@pytest.fixture
def config_manager():
    """Initialize configuration manager."""
    return ConfigManager()


@pytest.fixture
def water_balance_model(sample_soil_parameters, sample_crop_parameters, sample_climate_data):
    """Create a basic water balance model instance."""
    return WaterBalanceModel(
        soil_params=sample_soil_parameters,
        crop_params=sample_crop_parameters,
        climate_data=sample_climate_data,
        management='rainfed',
        irrigation_efficiency=1.0
    )
