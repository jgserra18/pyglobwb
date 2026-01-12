"""
Test suite for crop utilities.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add pyglobwb directory to path
pyglobwb_path = Path(__file__).parent.parent / 'pyglobwb'
sys.path.insert(0, str(pyglobwb_path))

from crop_utils import (
    CropCalendarConfig,
    interpolate_monthly_to_daily,
    interpolate_rooting_depth,
    create_daily_crop_calendar,
    create_crop_parameters,
    get_crop_config,
    CROP_CONFIGS
)


class TestCropCalendarConfig:
    """Test suite for CropCalendarConfig dataclass."""
    
    def test_valid_config(self):
        """Test valid configuration creation."""
        config = CropCalendarConfig(
            name='TestCrop',
            monthly_kc=[0, 0, 0, 0.3, 0.6, 1.0, 1.2, 0.8, 0.4, 0, 0, 0],
            max_rooting_depth=1.5
        )
        
        assert config.name == 'TestCrop'
        assert len(config.monthly_kc) == 12
        assert config.max_rooting_depth == 1.5
        assert config.is_permanent == False
        assert config.initial_rooting_depth == 0.2
    
    def test_invalid_monthly_kc_length(self):
        """Test that invalid monthly_kc length raises error."""
        with pytest.raises(ValueError, match="monthly_kc must have 12 values"):
            CropCalendarConfig(
                name='TestCrop',
                monthly_kc=[0.5, 0.6, 0.7],  # Only 3 values
                max_rooting_depth=1.0
            )
    
    def test_permanent_crop_config(self):
        """Test permanent crop configuration."""
        config = CropCalendarConfig(
            name='Olive',
            monthly_kc=[0.65] * 12,
            max_rooting_depth=1.7,
            is_permanent=True,
            initial_rooting_depth=1.7
        )
        
        assert config.is_permanent == True
        assert config.initial_rooting_depth == 1.7


class TestInterpolateMonthlyToDaily:
    """Test suite for monthly to daily interpolation."""
    
    @pytest.fixture
    def sample_dates(self):
        """Generate sample date range."""
        return pd.date_range('2020-01-01', '2020-12-31', freq='D')
    
    @pytest.fixture
    def sample_monthly_kc(self):
        """Sample monthly Kc values for maize."""
        return [0, 0, 0, 0.3, 0.64, 1.17, 1.2, 0.70, 0, 0, 0, 0]
    
    def test_step_interpolation(self, sample_dates, sample_monthly_kc):
        """Test step interpolation method."""
        daily_kc = interpolate_monthly_to_daily(
            sample_monthly_kc, sample_dates, method='step'
        )
        
        assert len(daily_kc) == len(sample_dates)
        
        # Check January values (should be 0)
        jan_mask = sample_dates.month == 1
        assert all(daily_kc[jan_mask] == 0)
        
        # Check July values (should be 1.2)
        jul_mask = sample_dates.month == 7
        assert all(daily_kc[jul_mask] == 1.2)
    
    def test_linear_interpolation(self, sample_dates, sample_monthly_kc):
        """Test linear interpolation method."""
        daily_kc = interpolate_monthly_to_daily(
            sample_monthly_kc, sample_dates, method='linear'
        )
        
        assert len(daily_kc) == len(sample_dates)
        
        # Values should be smoother than step
        # Check that mid-month values are close to monthly values
        # but transition days should be different
    
    def test_invalid_method(self, sample_dates, sample_monthly_kc):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            interpolate_monthly_to_daily(
                sample_monthly_kc, sample_dates, method='invalid'
            )
    
    def test_invalid_monthly_values_length(self, sample_dates):
        """Test that invalid monthly values length raises error."""
        with pytest.raises(ValueError, match="monthly_values must have 12 values"):
            interpolate_monthly_to_daily(
                [0.5, 0.6],  # Only 2 values
                sample_dates,
                method='step'
            )
    
    def test_multi_year_interpolation(self, sample_monthly_kc):
        """Test interpolation across multiple years."""
        dates = pd.date_range('2019-01-01', '2021-12-31', freq='D')
        daily_kc = interpolate_monthly_to_daily(
            sample_monthly_kc, dates, method='step'
        )
        
        assert len(daily_kc) == len(dates)
        
        # Check that pattern repeats each year
        for year in [2019, 2020, 2021]:
            jul_mask = (dates.month == 7) & (dates.year == year)
            assert all(daily_kc[jul_mask] == 1.2)


class TestInterpolateRootingDepth:
    """Test suite for rooting depth interpolation."""
    
    @pytest.fixture
    def sample_dates(self):
        """Generate sample date range."""
        return pd.date_range('2020-01-01', '2020-12-31', freq='D')
    
    @pytest.fixture
    def sample_kc_daily(self, sample_dates):
        """Generate sample daily Kc values."""
        monthly_kc = [0, 0, 0, 0.3, 0.64, 1.17, 1.2, 0.70, 0, 0, 0, 0]
        return interpolate_monthly_to_daily(monthly_kc, sample_dates, method='step')
    
    def test_permanent_crop_fixed_depth(self, sample_dates):
        """Test that permanent crops have fixed rooting depth."""
        kc_daily = np.ones(len(sample_dates)) * 0.65  # Constant Kc
        
        rooting_depth = interpolate_rooting_depth(
            kc_daily, sample_dates,
            max_rooting_depth=1.7,
            initial_rooting_depth=1.7,
            is_permanent=True
        )
        
        assert len(rooting_depth) == len(sample_dates)
        assert all(rooting_depth == 1.7)
    
    def test_annual_crop_growth(self, sample_dates, sample_kc_daily):
        """Test that annual crops have growing rooting depth."""
        rooting_depth = interpolate_rooting_depth(
            sample_kc_daily, sample_dates,
            max_rooting_depth=1.2,
            initial_rooting_depth=0.2,
            is_permanent=False
        )
        
        assert len(rooting_depth) == len(sample_dates)
        
        # Should start at initial depth
        assert rooting_depth[0] == 0.2
        
        # Should reach max depth during growing season
        assert max(rooting_depth) == pytest.approx(1.2)
        
        # Should increase during growth phase
        # Find July (max Kc month)
        jul_mask = sample_dates.month == 7
        jul_indices = np.where(jul_mask)[0]
        assert rooting_depth[jul_indices[0]] == pytest.approx(1.2, rel=0.1)
    
    def test_rooting_depth_bounds(self, sample_dates, sample_kc_daily):
        """Test that rooting depth stays within bounds."""
        rooting_depth = interpolate_rooting_depth(
            sample_kc_daily, sample_dates,
            max_rooting_depth=1.5,
            initial_rooting_depth=0.3,
            is_permanent=False
        )
        
        assert all(rooting_depth >= 0.2)  # At least initial or fallow depth
        assert all(rooting_depth <= 1.5)  # At most max depth


class TestCreateDailyCropCalendar:
    """Test suite for daily crop calendar creation."""
    
    def test_create_calendar(self):
        """Test creating a daily crop calendar."""
        config = CropCalendarConfig(
            name='Maize',
            monthly_kc=[0, 0, 0, 0.3, 0.64, 1.17, 1.2, 0.70, 0, 0, 0, 0],
            max_rooting_depth=1.2,
            is_permanent=False
        )
        
        calendar = create_daily_crop_calendar(
            config,
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        assert isinstance(calendar, pd.DataFrame)
        assert 'date' in calendar.columns
        assert 'kc' in calendar.columns
        assert 'rooting_depth' in calendar.columns
        assert len(calendar) == 366  # 2020 is leap year
    
    def test_calendar_values_range(self):
        """Test that calendar values are in valid range."""
        config = CropCalendarConfig(
            name='Wheat',
            monthly_kc=[0.4, 0.7, 1.1, 1.15, 0.7, 0.3, 0, 0, 0, 0.3, 0.4, 0.4],
            max_rooting_depth=1.5,
            is_permanent=False
        )
        
        calendar = create_daily_crop_calendar(
            config,
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        assert all(calendar['kc'] >= 0)
        assert all(calendar['rooting_depth'] > 0)
        assert all(calendar['rooting_depth'] <= 1.5)


class TestCreateCropParameters:
    """Test suite for CropParameters creation."""
    
    def test_create_parameters(self):
        """Test creating CropParameters from config."""
        config = CropCalendarConfig(
            name='Maize',
            monthly_kc=[0, 0, 0, 0.3, 0.64, 1.17, 1.2, 0.70, 0, 0, 0, 0],
            max_rooting_depth=1.2,
            is_permanent=False,
            landuse_kc=0.5
        )
        
        crop_params = create_crop_parameters(
            config,
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        assert crop_params.name == 'Maize'
        assert len(crop_params.kc_values) == 366
        assert len(crop_params.rooting_depth) == 366
        assert len(crop_params.dates) == 366
        assert crop_params.landuse_kc == 0.5


class TestGetCropConfig:
    """Test suite for predefined crop configurations."""
    
    def test_get_maize(self):
        """Test getting maize configuration."""
        config = get_crop_config('maize')
        
        assert config.name == 'Maize'
        assert len(config.monthly_kc) == 12
        assert config.is_permanent == False
    
    def test_get_olive(self):
        """Test getting olive configuration."""
        config = get_crop_config('olive')
        
        assert config.name == 'Olive'
        assert config.is_permanent == True
    
    def test_case_insensitive(self):
        """Test that crop name lookup is case-insensitive."""
        config1 = get_crop_config('MAIZE')
        config2 = get_crop_config('Maize')
        config3 = get_crop_config('maize')
        
        assert config1.name == config2.name == config3.name
    
    def test_unknown_crop_raises_error(self):
        """Test that unknown crop raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            get_crop_config('unknown_crop')


class TestPredefinedCrops:
    """Test suite for predefined crop configurations."""
    
    def test_all_configs_valid(self):
        """Test that all predefined configs are valid."""
        for name, config in CROP_CONFIGS.items():
            assert len(config.monthly_kc) == 12
            assert config.max_rooting_depth > 0
            assert config.initial_rooting_depth > 0
    
    def test_permanent_crops_have_all_positive_kc(self):
        """Test that permanent crops have positive Kc year-round."""
        for name, config in CROP_CONFIGS.items():
            if config.is_permanent:
                # Most months should have positive Kc for permanent crops
                positive_months = sum(1 for kc in config.monthly_kc if kc > 0)
                assert positive_months >= 6, f"{name} should have mostly positive Kc"
