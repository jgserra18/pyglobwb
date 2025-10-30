"""
Test suite for climate utilities.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add pyglobwb to path
pyglobwb_path = Path(__file__).parent.parent / 'pyglobwb'
sys.path.insert(0, str(pyglobwb_path))

from climate_utils import (
    fetch_climate_from_openmeteo,
    prepare_climate_data,
    validate_climate_data,
    calculate_aridity_index,
    get_climate_statistics,
    resample_climate_data
)
from water_balance_model import ClimateData


class TestFetchClimateFromOpenMeteo:
    """Test suite for Open-Meteo data fetching."""
    
    def test_fetch_one_year(self):
        """Test fetching one year of data."""
        # Aarhus, Denmark - 1 year
        df = fetch_climate_from_openmeteo(
            latitude=56.15,
            longitude=9.56,
            start_date="2020-01-01",
            end_date="2020-12-31",
            verbose=False
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 366  # 2020 is a leap year
        assert 'precipitation' in df.columns
        assert 'et0_fao_evapotranspiration' in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
    
    def test_fetch_short_period(self):
        """Test fetching a short period (1 month)."""
        df = fetch_climate_from_openmeteo(
            latitude=38.0,
            longitude=-1.5,
            start_date="2021-06-01",
            end_date="2021-06-30",
            verbose=False
        )
        
        assert len(df) == 30
        assert df['precipitation'].notna().all()
        assert df['et0_fao_evapotranspiration'].notna().all()
    
    def test_data_ranges(self):
        """Test that data values are in reasonable ranges."""
        df = fetch_climate_from_openmeteo(
            latitude=40.0,
            longitude=-3.0,  # Madrid, Spain
            start_date="2019-01-01",
            end_date="2019-12-31",
            verbose=False
        )
        
        # Precipitation should be non-negative
        assert (df['precipitation'] >= 0).all()
        
        # ET0 should be positive and reasonable (< 20 mm/day typically)
        assert (df['et0_fao_evapotranspiration'] > 0).all()
        assert (df['et0_fao_evapotranspiration'] < 20).all()
    
    def test_different_locations(self):
        """Test fetching data for different global locations."""
        locations = [
            (51.5, -0.1),   # London
            (-33.9, 18.4),  # Cape Town
            (35.7, 139.7),  # Tokyo
        ]
        
        for lat, lon in locations:
            df = fetch_climate_from_openmeteo(
                latitude=lat,
                longitude=lon,
                start_date="2020-01-01",
                end_date="2020-01-31",
                verbose=False
            )
            assert len(df) == 31
            assert df['precipitation'].notna().all()


class TestPrepareClimateData:
    """Test suite for climate data preparation."""
    
    def test_prepare_from_openmeteo(self):
        """Test preparing ClimateData from Open-Meteo DataFrame."""
        # Fetch data
        df = fetch_climate_from_openmeteo(
            latitude=56.15,
            longitude=9.56,
            start_date="2020-01-01",
            end_date="2020-12-31",
            verbose=False
        )
        
        # Prepare
        climate = prepare_climate_data(
            df,
            pr_column="precipitation",
            pet_column="et0_fao_evapotranspiration",
            verbose=False
        )
        
        assert isinstance(climate, ClimateData)
        assert len(climate.precipitation) == 366
        assert len(climate.pet) == 366
        assert len(climate.dates) == 366
    
    def test_prepare_with_missing_values(self):
        """Test handling of missing values."""
        # Create DataFrame with NaN
        dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
        df = pd.DataFrame({
            'precipitation': [1.0] * 30 + [np.nan],
            'et0_fao_evapotranspiration': [2.0] * 31
        }, index=dates)
        
        climate = prepare_climate_data(
            df,
            pr_column="precipitation",
            pet_column="et0_fao_evapotranspiration",
            verbose=False
        )
        
        # Missing precipitation should be filled with 0
        assert not np.any(np.isnan(climate.precipitation))
        assert climate.precipitation[-1] == 0.0


class TestValidateClimateData:
    """Test suite for climate data validation."""
    
    def test_validate_good_data(self):
        """Test validation of good data."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        climate = ClimateData(
            precipitation=np.random.gamma(2, 2, len(dates)),
            pet=3 + 2 * np.sin(2 * np.pi * dates.dayofyear / 365),
            dates=dates
        )
        
        issues = validate_climate_data(climate)
        
        assert len(issues['errors']) == 0
        assert isinstance(issues['warnings'], list)
    
    def test_validate_negative_values(self):
        """Test detection of negative values."""
        dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
        climate = ClimateData(
            precipitation=np.array([1.0] * 30 + [-1.0]),
            pet=np.ones(31) * 3.0,
            dates=dates
        )
        
        issues = validate_climate_data(climate)
        
        assert len(issues['errors']) > 0
        assert any('Negative precipitation' in err for err in issues['errors'])
    
    def test_validate_unrealistic_values(self):
        """Test detection of unrealistic values."""
        dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
        climate = ClimateData(
            precipitation=np.array([1.0] * 30 + [600.0]),  # 600mm in one day
            pet=np.ones(31) * 3.0,
            dates=dates
        )
        
        issues = validate_climate_data(climate)
        
        assert len(issues['warnings']) > 0
        assert any('Very high daily precipitation' in warn for warn in issues['warnings'])


class TestCalculateAridityIndex:
    """Test suite for aridity index calculation."""
    
    def test_aridity_index_humid(self):
        """Test aridity index for humid climate."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        climate = ClimateData(
            precipitation=np.ones(len(dates)) * 5.0,  # 5mm/day
            pet=np.ones(len(dates)) * 3.0,  # 3mm/day
            dates=dates
        )
        
        ai = calculate_aridity_index(climate)
        
        assert ai > 1.0  # Humid (P > PET)
    
    def test_aridity_index_arid(self):
        """Test aridity index for arid climate."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        climate = ClimateData(
            precipitation=np.ones(len(dates)) * 0.5,  # 0.5mm/day
            pet=np.ones(len(dates)) * 5.0,  # 5mm/day
            dates=dates
        )
        
        ai = calculate_aridity_index(climate)
        
        assert ai < 0.2  # Arid


class TestGetClimateStatistics:
    """Test suite for climate statistics."""
    
    def test_climate_statistics(self):
        """Test calculation of climate statistics."""
        # Fetch real data
        df = fetch_climate_from_openmeteo(
            latitude=38.0,
            longitude=-1.5,
            start_date="2019-01-01",
            end_date="2020-12-31",  # 2 years
            verbose=False
        )
        
        climate = prepare_climate_data(
            df,
            pr_column="precipitation",
            pet_column="et0_fao_evapotranspiration",
            verbose=False
        )
        
        stats = get_climate_statistics(climate)
        
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 2  # 2 years
        assert 'precipitation_sum' in stats.columns
        assert 'pet_sum' in stats.columns
        assert 'aridity_index' in stats.columns


class TestResampleClimateData:
    """Test suite for climate data resampling."""
    
    def test_resample_to_monthly(self):
        """Test resampling daily data to monthly."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        climate = ClimateData(
            precipitation=np.ones(len(dates)) * 2.0,
            pet=np.ones(len(dates)) * 3.0,
            dates=dates
        )
        
        monthly = resample_climate_data(climate, freq='M')
        
        assert len(monthly) == 12
        assert 'precipitation' in monthly.columns
        assert 'pet' in monthly.columns
        
        # Check that monthly sums are correct (approximately)
        # January has 31 days, so 31 * 2.0 = 62mm
        assert monthly.iloc[0]['precipitation'] == pytest.approx(62.0, rel=0.01)


class TestIntegrationWithModel:
    """Integration tests with water balance model."""
    
    def test_fetch_and_run_model(self):
        """Test complete workflow: fetch data and run model."""
        from config_manager import ConfigManager
        from water_balance_model import WaterBalanceModel, SoilParameters, CropParameters
        
        # Fetch climate data
        df = fetch_climate_from_openmeteo(
            latitude=38.0,
            longitude=-1.5,
            start_date="2020-01-01",
            end_date="2020-12-31",
            verbose=False
        )
        
        climate = prepare_climate_data(
            df,
            pr_column="precipitation",
            pet_column="et0_fao_evapotranspiration",
            verbose=False
        )
        
        # Validate
        issues = validate_climate_data(climate)
        assert len(issues['errors']) == 0
        
        # Set up model
        config = ConfigManager()
        
        kc_daily, dates = config.create_daily_kc('Maize', '2020-01-01', '2020-12-31')
        rooting_depth, _ = config.create_daily_rooting_depth(
            'Maize', '2020-01-01', '2020-12-31', 'irrigated'
        )
        
        crop = CropParameters(
            name='Maize',
            kc_values=kc_daily,
            rooting_depth=rooting_depth,
            dates=dates,
            landuse_kc=0.5
        )
        
        soil = SoilParameters(
            smax_base=150.0,
            rmax=10.0,
            calibration_factor=2.4
        )
        
        # Run model
        model = WaterBalanceModel(soil, crop, climate, 'irrigated', 0.9)
        results = model.run(spinup_iterations=10)
        
        # Check results
        assert len(results) == 366
        assert 'soil_moisture' in results.columns
        assert 'evapotranspiration' in results.columns
        assert (results['soil_moisture'] >= 0).all()
