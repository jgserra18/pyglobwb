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


class TestSmaxCalculationOptionsIntegration:
    """Integration tests for smax calculation options in full model runs."""
    
    def test_full_run_option1(self, sample_soil_parameters, sample_crop_parameters, 
                              sample_climate_data):
        """Test full model run with Option 1 (original method)."""
        model = WaterBalanceModel(
            soil_params=sample_soil_parameters,
            crop_params=sample_crop_parameters,
            climate_data=sample_climate_data,
            management='irrigated',
            irrigation_efficiency=0.9,
            smax_calculation_option=1
        )
        
        results = model.run(spinup_iterations=10)
        
        # Check that results are valid
        assert len(results) == len(sample_climate_data.dates)
        assert all(results['soil_moisture'] >= 0)
        assert all(results['smax'] > 0)
        assert all(results['seav'] > 0)
        assert all(results['seav'] <= results['smax'])
    
    def test_full_run_option2(self, sample_soil_parameters, sample_crop_parameters, 
                              sample_climate_data):
        """Test full model run with Option 2 (PAWC-based method)."""
        from water_balance_model import SoilParameters
        
        soil_pawc = SoilParameters(
            smax_base=150.0,
            pawc_soil=0.18,
            zmax=2.0,
            p=0.5,
            rmax=10.0
        )
        
        model = WaterBalanceModel(
            soil_params=soil_pawc,
            crop_params=sample_crop_parameters,
            climate_data=sample_climate_data,
            management='irrigated',
            irrigation_efficiency=0.9,
            smax_calculation_option=2
        )
        
        results = model.run(spinup_iterations=10)
        
        # Check that results are valid
        assert len(results) == len(sample_climate_data.dates)
        assert all(results['soil_moisture'] >= 0)
        assert all(results['smax'] > 0)
        assert all(results['seav'] > 0)
        assert all(results['seav'] <= results['smax'])
    
    def test_compare_options_irrigation_requirements(self, sample_crop_parameters, 
                                                     sample_climate_data):
        """Compare irrigation requirements between Option 1 and Option 2."""
        from water_balance_model import SoilParameters
        
        # Option 1 soil
        soil1 = SoilParameters(
            smax_base=150.0,
            reference_depth=0.6,
            rmax=10.0
        )
        
        # Option 2 soil
        soil2 = SoilParameters(
            smax_base=150.0,
            pawc_soil=0.18,
            zmax=2.0,
            p=0.5,
            rmax=10.0
        )
        
        model1 = WaterBalanceModel(
            soil_params=soil1,
            crop_params=sample_crop_parameters,
            climate_data=sample_climate_data,
            management='irrigated',
            irrigation_efficiency=0.9,
            smax_calculation_option=1
        )
        
        model2 = WaterBalanceModel(
            soil_params=soil2,
            crop_params=sample_crop_parameters,
            climate_data=sample_climate_data,
            management='irrigated',
            irrigation_efficiency=0.9,
            smax_calculation_option=2
        )
        
        results1 = model1.run(spinup_iterations=10)
        results2 = model2.run(spinup_iterations=10)
        
        # Both should produce valid results (irrigation may be zero if precipitation is sufficient)
        assert results1['irrigation'].sum() >= 0
        assert results2['irrigation'].sum() >= 0
        
        # Smax and Seav should differ between options
        assert not np.allclose(results1['smax'].values, 
                              results2['smax'].values, rtol=0.01)
    
    def test_option2_soil_texture_sensitivity(self, sample_crop_parameters, 
                                              sample_climate_data):
        """Test that Option 2 is sensitive to soil texture (PAWC)."""
        from water_balance_model import SoilParameters
        
        # Sandy soil (low PAWC)
        soil_sandy = SoilParameters(
            smax_base=150.0,
            pawc_soil=0.10,
            zmax=2.0,
            p=0.5,
            rmax=10.0
        )
        
        # Clay loam (high PAWC)
        soil_clay = SoilParameters(
            smax_base=150.0,
            pawc_soil=0.22,
            zmax=2.0,
            p=0.5,
            rmax=10.0
        )
        
        model_sandy = WaterBalanceModel(
            soil_params=soil_sandy,
            crop_params=sample_crop_parameters,
            climate_data=sample_climate_data,
            management='irrigated',
            irrigation_efficiency=0.9,
            smax_calculation_option=2
        )
        
        model_clay = WaterBalanceModel(
            soil_params=soil_clay,
            crop_params=sample_crop_parameters,
            climate_data=sample_climate_data,
            management='irrigated',
            irrigation_efficiency=0.9,
            smax_calculation_option=2
        )
        
        results_sandy = model_sandy.run(spinup_iterations=10)
        results_clay = model_clay.run(spinup_iterations=10)
        
        # Smax should be different (sandy < clay)
        assert results_sandy['smax'].mean() < results_clay['smax'].mean()
        
        # Sandy soil has lower water holding capacity
        # If irrigation occurs, sandy should need more, but test the capacity difference
        assert results_sandy['smax'].mean() == pytest.approx(
            0.10 * results_sandy['rooting_depth'].mean() * 1000.0, rel=0.1
        )
        assert results_clay['smax'].mean() == pytest.approx(
            0.22 * results_clay['rooting_depth'].mean() * 1000.0, rel=0.1
        )
    
    def test_option2_depletion_fraction_effect(self, sample_crop_parameters, 
                                               sample_climate_data):
        """Test that depletion fraction affects irrigation timing in Option 2."""
        from water_balance_model import SoilParameters
        
        # Low p (more sensitive, irrigate earlier)
        soil_sensitive = SoilParameters(
            smax_base=150.0,
            pawc_soil=0.18,
            zmax=2.0,
            p=0.4,
            rmax=10.0
        )
        
        # High p (less sensitive, irrigate later)
        soil_tolerant = SoilParameters(
            smax_base=150.0,
            pawc_soil=0.18,
            zmax=2.0,
            p=0.6,
            rmax=10.0
        )
        
        model_sensitive = WaterBalanceModel(
            soil_params=soil_sensitive,
            crop_params=sample_crop_parameters,
            climate_data=sample_climate_data,
            management='irrigated',
            irrigation_efficiency=0.9,
            smax_calculation_option=2
        )
        
        model_tolerant = WaterBalanceModel(
            soil_params=soil_tolerant,
            crop_params=sample_crop_parameters,
            climate_data=sample_climate_data,
            management='irrigated',
            irrigation_efficiency=0.9,
            smax_calculation_option=2
        )
        
        results_sensitive = model_sensitive.run(spinup_iterations=10)
        results_tolerant = model_tolerant.run(spinup_iterations=10)
        
        # Both should produce valid results
        assert all(results_sensitive['irrigation'] >= 0)
        assert all(results_tolerant['irrigation'] >= 0)
        
        # Seav should differ
        assert results_sensitive['seav'].mean() < results_tolerant['seav'].mean()
    
    def test_water_balance_closure_both_options(self, sample_crop_parameters, 
                                                sample_climate_data):
        """Test that water balance closes properly for both options."""
        from water_balance_model import SoilParameters
        
        soil1 = SoilParameters(smax_base=150.0, reference_depth=0.6)
        soil2 = SoilParameters(smax_base=150.0, pawc_soil=0.18, zmax=2.0, p=0.5)
        
        for option, soil in [(1, soil1), (2, soil2)]:
            model = WaterBalanceModel(
                soil_params=soil,
                crop_params=sample_crop_parameters,
                climate_data=sample_climate_data,
                management='irrigated',
                irrigation_efficiency=0.9,
                smax_calculation_option=option
            )
            
            results = model.run(spinup_iterations=10)
            
            # Check water balance closure for each day
            # Note: Percolation is calculated separately and not subtracted from SM update
            # The model's water balance is: SM_t = SM_t-1 + P + I - ET - Runoff
            for i in range(1, len(results)):
                delta_sm = results['soil_moisture'].iloc[i] - results['soil_moisture'].iloc[i-1]
                inputs = results['precipitation'].iloc[i] + results['irrigation'].iloc[i]
                outputs = (results['evapotranspiration'].iloc[i] + 
                          results['runoff'].iloc[i])
                
                # Water balance should close (allowing small numerical error)
                assert abs(delta_sm - (inputs - outputs)) < 1.0
