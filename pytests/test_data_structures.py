"""
Test suite for data structure classes (SoilParameters, CropParameters, ClimateData).
"""

import pytest
import numpy as np
import pandas as pd
from water_balance_model import SoilParameters, CropParameters, ClimateData


class TestSoilParameters:
    """Test suite for SoilParameters dataclass."""
    
    def test_initialization_with_defaults(self):
        """Test SoilParameters initialization with default values."""
        soil = SoilParameters(
            smax_base=150.0,
            rmax=10.0
        )
        
        assert soil.smax_base == 150.0
        assert soil.reference_depth == 0.6
        assert soil.rmax == 10.0
        assert soil.initial_sm == 75.0  # 50% of smax_base
        assert soil.calibration_factor == 2.4
    
    def test_initialization_with_custom_values(self):
        """Test SoilParameters initialization with custom values."""
        soil = SoilParameters(
            smax_base=200.0,
            reference_depth=0.8,
            rmax=15.0,
            initial_sm=100.0,
            calibration_factor=3.0
        )
        
        assert soil.smax_base == 200.0
        assert soil.reference_depth == 0.8
        assert soil.rmax == 15.0
        assert soil.initial_sm == 100.0
        assert soil.calibration_factor == 3.0
    
    def test_initial_sm_default_calculation(self):
        """Test that initial_sm defaults to 50% of smax_base."""
        soil = SoilParameters(smax_base=180.0, rmax=10.0)
        assert soil.initial_sm == 90.0


class TestCropParameters:
    """Test suite for CropParameters dataclass."""
    
    def test_initialization_valid(self):
        """Test CropParameters initialization with valid data."""
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        n_days = len(dates)
        
        crop = CropParameters(
            name='TestCrop',
            kc_values=np.full(n_days, 1.0),
            rooting_depth=np.full(n_days, 0.8),
            dates=dates,
            landuse_kc=0.5
        )
        
        assert crop.name == 'TestCrop'
        assert len(crop.kc_values) == n_days
        assert len(crop.rooting_depth) == n_days
        assert len(crop.dates) == n_days
        assert crop.landuse_kc == 0.5
    
    def test_initialization_mismatched_lengths(self):
        """Test that initialization fails with mismatched array lengths."""
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        
        with pytest.raises(ValueError, match="must have the same length"):
            CropParameters(
                name='TestCrop',
                kc_values=np.full(10, 1.0),
                rooting_depth=np.full(5, 0.8),  # Wrong length
                dates=dates,
                landuse_kc=0.5
            )
    
    def test_kc_values_range(self):
        """Test that Kc values are in reasonable range."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        n_days = len(dates)
        
        crop = CropParameters(
            name='TestCrop',
            kc_values=np.linspace(0.3, 1.2, n_days),
            rooting_depth=np.full(n_days, 1.0),
            dates=dates
        )
        
        assert np.all(crop.kc_values >= 0)
        assert np.all(crop.kc_values <= 1.5)


class TestClimateData:
    """Test suite for ClimateData dataclass."""
    
    def test_initialization_valid(self):
        """Test ClimateData initialization with valid data."""
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        n_days = len(dates)
        
        climate = ClimateData(
            precipitation=np.random.uniform(0, 10, n_days),
            pet=np.random.uniform(1, 5, n_days),
            dates=dates
        )
        
        assert len(climate.precipitation) == n_days
        assert len(climate.pet) == n_days
        assert len(climate.dates) == n_days
    
    def test_initialization_mismatched_lengths(self):
        """Test that initialization fails with mismatched array lengths."""
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        
        with pytest.raises(ValueError, match="must have the same length"):
            ClimateData(
                precipitation=np.full(10, 2.0),
                pet=np.full(5, 3.0),  # Wrong length
                dates=dates
            )
    
    def test_non_negative_values(self):
        """Test that climate values are non-negative."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        n_days = len(dates)
        
        climate = ClimateData(
            precipitation=np.random.uniform(0, 10, n_days),
            pet=np.random.uniform(1, 5, n_days),
            dates=dates
        )
        
        assert np.all(climate.precipitation >= 0)
        assert np.all(climate.pet >= 0)


class TestDataStructureIntegration:
    """Integration tests for data structures."""
    
    def test_complete_model_inputs(self, sample_soil_parameters, sample_crop_parameters, 
                                   sample_climate_data):
        """Test that all data structures work together."""
        assert sample_soil_parameters.smax_base > 0
        assert len(sample_crop_parameters.kc_values) == len(sample_climate_data.dates)
        assert sample_crop_parameters.dates.equals(sample_climate_data.dates)
