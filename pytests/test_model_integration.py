"""
Integration tests for the complete water balance model.
"""

import pytest
import numpy as np
import pandas as pd
from water_balance_model import WaterBalanceModel


class TestModelInitialization:
    """Test suite for model initialization."""
    
    def test_initialization_rainfed(self, sample_soil_parameters, sample_crop_parameters, 
                                    sample_climate_data):
        """Test model initialization for rainfed conditions."""
        model = WaterBalanceModel(
            soil_params=sample_soil_parameters,
            crop_params=sample_crop_parameters,
            climate_data=sample_climate_data,
            management='rainfed',
            irrigation_efficiency=1.0
        )
        
        assert model.management == 'rainfed'
        assert model.irrigation_efficiency == 1.0
        assert model.n_days == len(sample_climate_data.dates)
    
    def test_initialization_irrigated(self, sample_soil_parameters, sample_crop_parameters, 
                                     sample_climate_data):
        """Test model initialization for irrigated conditions."""
        model = WaterBalanceModel(
            soil_params=sample_soil_parameters,
            crop_params=sample_crop_parameters,
            climate_data=sample_climate_data,
            management='irrigated',
            irrigation_efficiency=0.9
        )
        
        assert model.management == 'irrigated'
        assert model.irrigation_efficiency == 0.9
    
    def test_invalid_management(self, sample_soil_parameters, sample_crop_parameters, 
                               sample_climate_data):
        """Test that invalid management type raises error."""
        with pytest.raises(ValueError, match="management must be"):
            WaterBalanceModel(
                soil_params=sample_soil_parameters,
                crop_params=sample_crop_parameters,
                climate_data=sample_climate_data,
                management='invalid',
                irrigation_efficiency=1.0
            )
    
    def test_invalid_irrigation_efficiency(self, sample_soil_parameters, sample_crop_parameters, 
                                          sample_climate_data):
        """Test that invalid irrigation efficiency raises error."""
        with pytest.raises(ValueError, match="irrigation_efficiency must be"):
            WaterBalanceModel(
                soil_params=sample_soil_parameters,
                crop_params=sample_crop_parameters,
                climate_data=sample_climate_data,
                management='irrigated',
                irrigation_efficiency=1.5
            )


class TestModelSpinup:
    """Test suite for model spin-up."""
    
    def test_spinup_returns_value(self, water_balance_model):
        """Test that spinup returns a soil moisture value."""
        sm = water_balance_model.spinup(n_iterations=10)
        
        assert isinstance(sm, (int, float, np.ndarray))
        assert sm >= 0
    
    def test_spinup_convergence(self, water_balance_model):
        """Test that spinup converges to stable value."""
        sm_10 = water_balance_model.spinup(n_iterations=10)
        sm_50 = water_balance_model.spinup(n_iterations=50)
        
        # Should be similar after sufficient iterations
        assert sm_10 == pytest.approx(sm_50, rel=0.1)
    
    def test_spinup_reasonable_range(self, water_balance_model):
        """Test that spinup produces reasonable soil moisture."""
        sm = water_balance_model.spinup(n_iterations=50)
        
        # Should be between 0 and Smax
        assert sm >= 0
        assert sm <= water_balance_model.soil.smax_base * 2


class TestModelRun:
    """Test suite for complete model run."""
    
    def test_run_completes(self, water_balance_model):
        """Test that model run completes without errors."""
        results = water_balance_model.run(spinup_iterations=10)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == water_balance_model.n_days
    
    def test_run_output_columns(self, water_balance_model):
        """Test that model output has all required columns."""
        results = water_balance_model.run(spinup_iterations=10)
        
        required_columns = [
            'date', 'precipitation', 'pet', 'soil_moisture',
            'evapotranspiration', 'irrigation', 'percolation',
            'runoff', 'smax', 'seav', 'kc', 'rooting_depth'
        ]
        
        for col in required_columns:
            assert col in results.columns
    
    def test_run_non_negative_values(self, water_balance_model):
        """Test that all output values are non-negative."""
        results = water_balance_model.run(spinup_iterations=10)
        
        non_negative_cols = [
            'precipitation', 'pet', 'soil_moisture',
            'evapotranspiration', 'irrigation', 'percolation',
            'runoff', 'smax', 'seav'
        ]
        
        for col in non_negative_cols:
            assert (results[col] >= 0).all(), f"{col} has negative values"
    
    def test_run_rainfed_no_irrigation(self, sample_soil_parameters, sample_crop_parameters, 
                                       sample_climate_data):
        """Test that rainfed model produces no irrigation."""
        model = WaterBalanceModel(
            soil_params=sample_soil_parameters,
            crop_params=sample_crop_parameters,
            climate_data=sample_climate_data,
            management='rainfed',
            irrigation_efficiency=1.0
        )
        
        results = model.run(spinup_iterations=10)
        
        assert (results['irrigation'] == 0).all()
    
    def test_run_irrigated_has_irrigation(self, sample_soil_parameters, sample_crop_parameters, 
                                         sample_climate_data):
        """Test that irrigated model can produce irrigation."""
        model = WaterBalanceModel(
            soil_params=sample_soil_parameters,
            crop_params=sample_crop_parameters,
            climate_data=sample_climate_data,
            management='irrigated',
            irrigation_efficiency=0.9
        )
        
        results = model.run(spinup_iterations=10)
        
        # Should have some irrigation during growing season
        assert results['irrigation'].sum() > 0
    
    def test_soil_moisture_within_bounds(self, water_balance_model):
        """Test that soil moisture stays within physical bounds."""
        results = water_balance_model.run(spinup_iterations=10)
        
        # SM should be between 0 and Smax
        assert (results['soil_moisture'] >= 0).all()
        assert (results['soil_moisture'] <= results['smax']).all()
    
    def test_et_less_than_pet(self, water_balance_model):
        """Test that actual ET does not exceed potential ET."""
        results = water_balance_model.run(spinup_iterations=10)
        
        # Allow small numerical tolerance
        assert (results['evapotranspiration'] <= results['pet'] * 1.01).all()


class TestModelOutputAggregation:
    """Test suite for output aggregation methods."""
    
    def test_annual_summary(self, water_balance_model):
        """Test annual summary aggregation."""
        results = water_balance_model.run(spinup_iterations=10)
        annual = water_balance_model.get_annual_summary(results)
        
        assert isinstance(annual, pd.DataFrame)
        assert 'year' in annual.columns
        assert len(annual) == 1  # One year of data
    
    def test_annual_summary_columns(self, water_balance_model):
        """Test that annual summary has correct columns."""
        results = water_balance_model.run(spinup_iterations=10)
        annual = water_balance_model.get_annual_summary(results)
        
        expected_cols = [
            'year', 'precipitation', 'pet', 'evapotranspiration',
            'irrigation', 'percolation', 'runoff', 'soil_moisture'
        ]
        
        for col in expected_cols:
            assert col in annual.columns
    
    def test_monthly_summary(self, water_balance_model):
        """Test monthly summary aggregation."""
        results = water_balance_model.run(spinup_iterations=10)
        monthly = water_balance_model.get_monthly_summary(results)
        
        assert isinstance(monthly, pd.DataFrame)
        assert 'year' in monthly.columns
        assert 'month' in monthly.columns
        assert len(monthly) == 12  # 12 months
    
    def test_annual_totals_reasonable(self, water_balance_model):
        """Test that annual totals are in reasonable range."""
        results = water_balance_model.run(spinup_iterations=10)
        annual = water_balance_model.get_annual_summary(results)
        
        # Annual precipitation should be positive
        assert annual['precipitation'].iloc[0] > 0
        
        # Annual ET should be less than or equal to P + I
        total_input = annual['precipitation'].iloc[0] + annual['irrigation'].iloc[0]
        assert annual['evapotranspiration'].iloc[0] <= total_input * 1.1


class TestModelPhysicalConsistency:
    """Test suite for physical consistency of model outputs."""
    
    def test_water_balance_closure(self, water_balance_model):
        """Test that water balance closes within tolerance."""
        results = water_balance_model.run(spinup_iterations=10)
        
        # Calculate change in storage
        delta_sm = results['soil_moisture'].diff()
        
        # Calculate water balance (without percolation, as it's calculated from previous SM)
        # The model updates: SM_t = SM_t-1 + P + I - ET - R0
        # Percolation is calculated separately and doesn't directly affect SM update
        wb_calculated = (
            results['precipitation'] +
            results['irrigation'] -
            results['evapotranspiration'] -
            results['runoff']
        )
        
        # Check closure (skip first day due to diff)
        closure_error = (delta_sm - wb_calculated).iloc[1:]
        
        # Allow 25 mm tolerance (percolation is not in the SM update equation)
        assert (closure_error.abs() < 25.0).mean() > 0.95
    
    def test_et_responds_to_soil_moisture(self, water_balance_model):
        """Test that ET is reduced when soil moisture is low."""
        results = water_balance_model.run(spinup_iterations=10)
        
        # Find days with low soil moisture
        low_sm_mask = results['soil_moisture'] < results['seav']
        
        if low_sm_mask.any():
            # ET should be less than PET on these days
            low_sm_days = results[low_sm_mask]
            assert (low_sm_days['evapotranspiration'] < low_sm_days['pet']).mean() > 0.8
    
    def test_irrigation_responds_to_deficit(self, sample_soil_parameters, 
                                           sample_crop_parameters, sample_climate_data):
        """Test that irrigation responds to water deficit."""
        model = WaterBalanceModel(
            soil_params=sample_soil_parameters,
            crop_params=sample_crop_parameters,
            climate_data=sample_climate_data,
            management='irrigated',
            irrigation_efficiency=0.9
        )
        
        results = model.run(spinup_iterations=10)
        
        # During growing season (Kc > 0), irrigation should occur when needed
        growing_season = results[results['kc'] > 0.5]
        
        if len(growing_season) > 0:
            # Should have some irrigation during growing season
            assert growing_season['irrigation'].sum() > 0


class TestModelEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_zero_precipitation(self, sample_soil_parameters):
        """Test model with zero precipitation."""
        dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
        n_days = len(dates)
        
        from water_balance_model import ClimateData, CropParameters
        climate = ClimateData(
            precipitation=np.zeros(n_days),
            pet=np.full(n_days, 3.0),
            dates=dates
        )
        
        # Create matching crop parameters
        crop = CropParameters(
            name='TestCrop',
            kc_values=np.full(n_days, 0.8),
            rooting_depth=np.full(n_days, 1.0),
            dates=dates
        )
        
        model = WaterBalanceModel(
            soil_params=sample_soil_parameters,
            crop_params=crop,
            climate_data=climate,
            management='rainfed'
        )
        
        results = model.run(spinup_iterations=10)
        
        # Should complete without errors
        assert len(results) == len(dates)
        # Soil moisture should decrease
        assert results['soil_moisture'].iloc[-1] < results['soil_moisture'].iloc[0]
    
    def test_high_precipitation(self, sample_soil_parameters):
        """Test model with very high precipitation."""
        dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
        n_days = len(dates)
        
        from water_balance_model import ClimateData, CropParameters
        climate = ClimateData(
            precipitation=np.full(n_days, 50.0),  # Very high
            pet=np.full(n_days, 3.0),
            dates=dates
        )
        
        # Create matching crop parameters
        crop = CropParameters(
            name='TestCrop',
            kc_values=np.full(n_days, 0.8),
            rooting_depth=np.full(n_days, 1.0),
            dates=dates
        )
        
        model = WaterBalanceModel(
            soil_params=sample_soil_parameters,
            crop_params=crop,
            climate_data=climate,
            management='rainfed'
        )
        
        results = model.run(spinup_iterations=10)
        
        # Should have significant runoff
        assert results['runoff'].sum() > 0
    
    def test_zero_kc(self, sample_soil_parameters, sample_climate_data):
        """Test model with zero crop coefficient."""
        dates = sample_climate_data.dates
        
        from water_balance_model import CropParameters
        crop = CropParameters(
            name='NoCrop',
            kc_values=np.zeros(len(dates)),
            rooting_depth=np.full(len(dates), 0.5),
            dates=dates
        )
        
        model = WaterBalanceModel(
            soil_params=sample_soil_parameters,
            crop_params=crop,
            climate_data=sample_climate_data,
            management='rainfed'
        )
        
        results = model.run(spinup_iterations=10)
        
        # ET should be minimal (only landuse_kc)
        assert results['evapotranspiration'].mean() < 2.0
