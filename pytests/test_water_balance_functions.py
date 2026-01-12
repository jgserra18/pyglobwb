"""
Test suite for water balance calculation functions.
"""

import pytest
import numpy as np
from water_balance_model import WaterBalanceModel, SoilParameters, CropParameters, ClimateData
import pandas as pd


class TestSoilMoistureUpdate:
    """Test suite for soil moisture update function."""
    
    def test_sm_below_smax(self, water_balance_model):
        """Test soil moisture update when WB < Smax."""
        wb = 100.0
        smax = 150.0
        
        sm = water_balance_model._update_soil_moisture(wb, smax)
        
        assert sm == 100.0
        assert sm <= smax
    
    def test_sm_exceeds_smax(self, water_balance_model):
        """Test soil moisture update when WB > Smax."""
        wb = 200.0
        smax = 150.0
        
        sm = water_balance_model._update_soil_moisture(wb, smax)
        
        assert sm == 150.0
        assert sm == smax
    
    def test_sm_negative_wb(self, water_balance_model):
        """Test soil moisture update with negative water balance."""
        wb = -10.0
        smax = 150.0
        
        sm = water_balance_model._update_soil_moisture(wb, smax)
        
        assert sm == 0.0
        assert sm >= 0


class TestEvapotranspirationUpdate:
    """Test suite for evapotranspiration update function."""
    
    def test_et_sufficient_moisture(self, water_balance_model):
        """Test ET when soil moisture is above easily available water."""
        sm_t1 = 100.0
        pet = 5.0
        smax = 150.0
        seav = 75.0
        
        et = water_balance_model._update_et(sm_t1, pet, smax, seav)
        
        assert et == pet
    
    def test_et_limited_moisture(self, water_balance_model):
        """Test ET when soil moisture is below easily available water."""
        sm_t1 = 50.0
        pet = 5.0
        smax = 150.0
        seav = 75.0
        
        et = water_balance_model._update_et(sm_t1, pet, smax, seav)
        
        expected_et = pet * (sm_t1 / seav)
        assert et == pytest.approx(expected_et)
        assert et < pet
    
    def test_et_zero_moisture(self, water_balance_model):
        """Test ET when soil moisture is zero."""
        sm_t1 = 0.0
        pet = 5.0
        smax = 150.0
        seav = 75.0
        
        et = water_balance_model._update_et(sm_t1, pet, smax, seav)
        
        assert et == 0.0
    
    def test_et_non_negative(self, water_balance_model):
        """Test that ET is always non-negative."""
        sm_t1 = 50.0
        pet = 5.0
        smax = 150.0
        seav = 75.0
        
        et = water_balance_model._update_et(sm_t1, pet, smax, seav)
        
        assert et >= 0


class TestPercolationUpdate:
    """Test suite for percolation update function."""
    
    def test_perc_above_seav(self, water_balance_model):
        """Test percolation when SM > Seav."""
        sm_t1 = 100.0
        smax = 150.0
        seav = 75.0
        
        perc = water_balance_model._update_percolation(sm_t1, smax, seav)
        
        assert perc > 0
        assert perc <= water_balance_model.soil.rmax * water_balance_model.soil.calibration_factor
    
    def test_perc_below_seav(self, water_balance_model):
        """Test percolation when SM < Seav."""
        sm_t1 = 50.0
        smax = 150.0
        seav = 75.0
        
        perc = water_balance_model._update_percolation(sm_t1, smax, seav)
        
        assert perc == 0.0
    
    def test_perc_at_seav(self, water_balance_model):
        """Test percolation when SM = Seav."""
        sm_t1 = 75.0
        smax = 150.0
        seav = 75.0
        
        perc = water_balance_model._update_percolation(sm_t1, smax, seav)
        
        assert perc == 0.0
    
    def test_perc_non_negative(self, water_balance_model):
        """Test that percolation is always non-negative."""
        sm_t1 = 100.0
        smax = 150.0
        seav = 75.0
        
        perc = water_balance_model._update_percolation(sm_t1, smax, seav)
        
        assert perc >= 0


class TestPercolationCalculation:
    """Test suite for percolation calculation with calibration factor."""
    
    def test_percolation_with_calibration_factor(self):
        """Test percolation calculation with custom calibration factor."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        soil = SoilParameters(
            smax_base=150.0,
            rmax=10.0,
            calibration_factor=2.0
        )
        crop = CropParameters(
            name='TestCrop',
            kc_values=np.ones(10),
            rooting_depth=np.ones(10) * 0.8,
            dates=dates
        )
        climate = ClimateData(
            precipitation=np.ones(10) * 5,
            pet=np.ones(10) * 3,
            dates=dates
        )
        
        model = WaterBalanceModel(
            soil_params=soil,
            crop_params=crop,
            climate_data=climate
        )
        
        # Test percolation calculation
        sm_t1 = 100.0
        smax = 150.0
        seav = 75.0
        
        perc = model._update_percolation(sm_t1, smax, seav)
        
        # Expected: (rmax * calibration_factor * (sm - seav) / (smax - seav))
        expected = 10.0 * 2.0 * (100.0 - 75.0) / (150.0 - 75.0)
        
        assert perc == pytest.approx(expected)
        assert perc > 0
    
    def test_percolation_default_calibration_factor(self):
        """Test percolation with default calibration factor (1.0)."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        soil = SoilParameters(
            smax_base=150.0,
            rmax=10.0
            # calibration_factor defaults to 1.0
        )
        crop = CropParameters(
            name='TestCrop',
            kc_values=np.ones(10),
            rooting_depth=np.ones(10) * 0.8,
            dates=dates
        )
        climate = ClimateData(
            precipitation=np.ones(10) * 5,
            pet=np.ones(10) * 3,
            dates=dates
        )
        
        model = WaterBalanceModel(
            soil_params=soil,
            crop_params=crop,
            climate_data=climate
        )
        
        sm_t1 = 100.0
        smax = 150.0
        seav = 75.0
        
        perc = model._update_percolation(sm_t1, smax, seav)
        
        # Expected: (rmax * 1.0 * (sm - seav) / (smax - seav))
        expected = 10.0 * 1.0 * (100.0 - 75.0) / (150.0 - 75.0)
        
        assert perc == pytest.approx(expected)
    
    def test_percolation_different_rmax_values(self):
        """Test percolation with different rmax values."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        
        crop = CropParameters(
            name='TestCrop',
            kc_values=np.ones(10),
            rooting_depth=np.ones(10) * 0.8,
            dates=dates
        )
        climate = ClimateData(
            precipitation=np.ones(10) * 5,
            pet=np.ones(10) * 3,
            dates=dates
        )
        
        # Low rmax
        soil_low = SoilParameters(smax_base=150.0, rmax=5.0)
        model_low = WaterBalanceModel(
            soil_params=soil_low, crop_params=crop, climate_data=climate
        )
        
        # High rmax
        soil_high = SoilParameters(smax_base=150.0, rmax=20.0)
        model_high = WaterBalanceModel(
            soil_params=soil_high, crop_params=crop, climate_data=climate
        )
        
        sm_t1 = 100.0
        smax = 150.0
        seav = 75.0
        
        perc_low = model_low._update_percolation(sm_t1, smax, seav)
        perc_high = model_high._update_percolation(sm_t1, smax, seav)
        
        # Higher rmax should produce higher percolation
        assert perc_high > perc_low
        assert perc_high == pytest.approx(perc_low * 4.0)  # 20/5 = 4


class TestRunoffUpdate:
    """Test suite for runoff update function."""
    
    def test_runoff_wb_below_smax(self, water_balance_model):
        """Test runoff when water balance < Smax."""
        sm_t1 = 100.0
        pr = 10.0
        et = 5.0
        smax = 150.0
        
        runoff = water_balance_model._update_runoff(sm_t1, pr, et, smax, irrig=0)
        
        assert runoff == 0.0
    
    def test_runoff_wb_exceeds_smax(self, water_balance_model):
        """Test runoff when water balance > Smax."""
        sm_t1 = 140.0
        pr = 30.0
        et = 5.0
        smax = 150.0
        
        runoff = water_balance_model._update_runoff(sm_t1, pr, et, smax, irrig=0)
        
        expected_wb = sm_t1 + pr - et
        expected_runoff = expected_wb - smax
        assert runoff == pytest.approx(expected_runoff)
        assert runoff > 0
    
    def test_runoff_with_irrigation(self, water_balance_model):
        """Test runoff calculation with irrigation."""
        sm_t1 = 140.0
        pr = 20.0
        et = 5.0
        smax = 150.0
        irrig = 10.0
        
        runoff = water_balance_model._update_runoff(sm_t1, pr, et, smax, irrig=irrig)
        
        expected_wb = sm_t1 + pr + irrig - et
        expected_runoff = expected_wb - smax
        assert runoff == pytest.approx(expected_runoff)
    
    def test_runoff_non_negative(self, water_balance_model):
        """Test that runoff is always non-negative."""
        sm_t1 = 100.0
        pr = 10.0
        et = 5.0
        smax = 150.0
        
        runoff = water_balance_model._update_runoff(sm_t1, pr, et, smax, irrig=0)
        
        assert runoff >= 0


class TestWaterBalanceUpdate:
    """Test suite for water balance calculation."""
    
    def test_wb_rainfed(self, water_balance_model):
        """Test water balance calculation for rainfed conditions."""
        sm_t1 = 100.0
        pr = 10.0
        et = 5.0
        runoff = 2.0
        
        wb = water_balance_model._update_water_balance(sm_t1, pr, et, runoff, irrig=0)
        
        expected_wb = sm_t1 + pr - et - runoff
        assert wb == pytest.approx(expected_wb)
    
    def test_wb_irrigated(self, water_balance_model):
        """Test water balance calculation for irrigated conditions."""
        sm_t1 = 100.0
        pr = 10.0
        et = 8.0
        runoff = 2.0
        irrig = 5.0
        
        wb = water_balance_model._update_water_balance(sm_t1, pr, et, runoff, irrig=irrig)
        
        expected_wb = sm_t1 + pr + irrig - et - runoff
        assert wb == pytest.approx(expected_wb)
    
    def test_wb_conservation(self, water_balance_model):
        """Test water balance conservation principle."""
        sm_t1 = 100.0
        pr = 10.0
        et = 5.0
        runoff = 3.0
        irrig = 2.0
        
        wb = water_balance_model._update_water_balance(sm_t1, pr, et, runoff, irrig=irrig)
        
        # Inputs - Outputs should equal change in storage
        inputs = pr + irrig
        outputs = et + runoff
        delta_storage = wb - sm_t1
        
        assert delta_storage == pytest.approx(inputs - outputs)


class TestDynamicSoilParameters:
    """Test suite for dynamic Smax and Seav calculation."""
    
    def test_smax_calculation_option1(self, water_balance_model):
        """Test dynamic Smax calculation based on rooting depth (Option 1)."""
        day_idx = 0
        
        smax, seav = water_balance_model._calculate_dynamic_smax_seav(day_idx)
        
        expected_smax = (water_balance_model.soil.smax_base / 
                        water_balance_model.soil.reference_depth * 
                        water_balance_model.crop.rooting_depth[day_idx])
        
        assert smax == pytest.approx(expected_smax)
    
    def test_seav_calculation_option1(self, water_balance_model):
        """Test that Seav is 50% of Smax (Option 1)."""
        day_idx = 0
        
        smax, seav = water_balance_model._calculate_dynamic_smax_seav(day_idx)
        
        assert seav == pytest.approx(smax * 0.5)
    
    def test_smax_varies_with_rooting_depth(self, water_balance_model):
        """Test that Smax changes with rooting depth."""
        # Modify rooting depth for testing
        water_balance_model.crop.rooting_depth[0] = 0.5
        water_balance_model.crop.rooting_depth[1] = 1.5
        
        smax_0, _ = water_balance_model._calculate_dynamic_smax_seav(0)
        smax_1, _ = water_balance_model._calculate_dynamic_smax_seav(1)
        
        assert smax_1 > smax_0


class TestSmaxCalculationOptions:
    """Test suite for different Smax/Seav calculation options."""
    
    def test_option1_original_method(self):
        """Test Option 1: Original smax_base scaling method."""
        # Setup
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        soil = SoilParameters(
            smax_base=150.0,
            reference_depth=0.6,
            rmax=10.0
        )
        crop = CropParameters(
            name='TestCrop',
            kc_values=np.ones(10),
            rooting_depth=np.linspace(0.3, 0.9, 10),
            dates=dates
        )
        climate = ClimateData(
            precipitation=np.ones(10) * 5,
            pet=np.ones(10) * 3,
            dates=dates
        )
        
        model = WaterBalanceModel(
            soil_params=soil,
            crop_params=crop,
            climate_data=climate,
            smax_calculation_option=1
        )
        
        # Test calculation
        smax, seav = model._calculate_dynamic_smax_seav(5)
        
        expected_smax = (150.0 / 0.6) * crop.rooting_depth[5]
        expected_seav = expected_smax * 0.5
        
        assert smax == pytest.approx(expected_smax)
        assert seav == pytest.approx(expected_seav)
    
    def test_option2_pawc_method(self):
        """Test Option 2: PAWC-based method."""
        # Setup
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        soil = SoilParameters(
            smax_base=150.0,  # Not used in option 2
            pawc_soil=0.18,
            zmax=2.0,
            p=0.5,
            rmax=10.0
        )
        crop = CropParameters(
            name='TestCrop',
            kc_values=np.ones(10),
            rooting_depth=np.linspace(0.3, 0.9, 10),
            dates=dates
        )
        climate = ClimateData(
            precipitation=np.ones(10) * 5,
            pet=np.ones(10) * 3,
            dates=dates
        )
        
        model = WaterBalanceModel(
            soil_params=soil,
            crop_params=crop,
            climate_data=climate,
            smax_calculation_option=2
        )
        
        # Test calculation
        day_idx = 5
        smax, seav = model._calculate_dynamic_smax_seav(day_idx)
        
        zr_eff = min(crop.rooting_depth[day_idx], soil.zmax)
        expected_smax = soil.pawc_soil * zr_eff * 1000.0
        expected_seav = soil.p * expected_smax
        
        assert smax == pytest.approx(expected_smax)
        assert seav == pytest.approx(expected_seav)
    
    def test_option2_depth_limitation(self):
        """Test that Option 2 limits rooting depth to zmax."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        soil = SoilParameters(
            smax_base=150.0,
            pawc_soil=0.18,
            zmax=1.0,  # Limit to 1m
            p=0.5
        )
        crop = CropParameters(
            name='TestCrop',
            kc_values=np.ones(10),
            rooting_depth=np.array([0.5, 0.8, 1.2, 1.5, 2.0, 2.0, 2.0, 2.0, 1.5, 1.0]),
            dates=dates
        )
        climate = ClimateData(
            precipitation=np.ones(10) * 5,
            pet=np.ones(10) * 3,
            dates=dates
        )
        
        model = WaterBalanceModel(
            soil_params=soil,
            crop_params=crop,
            climate_data=climate,
            smax_calculation_option=2
        )
        
        # Test with rooting depth > zmax
        smax_deep, _ = model._calculate_dynamic_smax_seav(4)  # rooting_depth = 2.0m
        
        # Should be limited to zmax
        expected_smax = soil.pawc_soil * soil.zmax * 1000.0
        assert smax_deep == pytest.approx(expected_smax)
    
    def test_option2_variable_depletion_fraction(self):
        """Test Option 2 with different depletion fractions."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        
        # Test with p=0.4 (more sensitive crop)
        soil_sensitive = SoilParameters(
            smax_base=150.0,
            pawc_soil=0.18,
            zmax=2.0,
            p=0.4
        )
        
        # Test with p=0.6 (less sensitive crop)
        soil_tolerant = SoilParameters(
            smax_base=150.0,
            pawc_soil=0.18,
            zmax=2.0,
            p=0.6
        )
        
        crop = CropParameters(
            name='TestCrop',
            kc_values=np.ones(10),
            rooting_depth=np.ones(10) * 0.8,
            dates=dates
        )
        climate = ClimateData(
            precipitation=np.ones(10) * 5,
            pet=np.ones(10) * 3,
            dates=dates
        )
        
        model_sensitive = WaterBalanceModel(
            soil_params=soil_sensitive,
            crop_params=crop,
            climate_data=climate,
            smax_calculation_option=2
        )
        
        model_tolerant = WaterBalanceModel(
            soil_params=soil_tolerant,
            crop_params=crop,
            climate_data=climate,
            smax_calculation_option=2
        )
        
        smax_s, seav_s = model_sensitive._calculate_dynamic_smax_seav(0)
        smax_t, seav_t = model_tolerant._calculate_dynamic_smax_seav(0)
        
        # Smax should be the same
        assert smax_s == pytest.approx(smax_t)
        
        # Seav should differ based on p
        assert seav_s == pytest.approx(smax_s * 0.4)
        assert seav_t == pytest.approx(smax_t * 0.6)
        assert seav_t > seav_s
    
    def test_invalid_option_raises_error(self):
        """Test that invalid smax_calculation_option raises ValueError."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        soil = SoilParameters(smax_base=150.0)
        crop = CropParameters(
            name='TestCrop',
            kc_values=np.ones(10),
            rooting_depth=np.ones(10) * 0.8,
            dates=dates
        )
        climate = ClimateData(
            precipitation=np.ones(10) * 5,
            pet=np.ones(10) * 3,
            dates=dates
        )
        
        with pytest.raises(ValueError, match="smax_calculation_option must be 1 or 2"):
            WaterBalanceModel(
                soil_params=soil,
                crop_params=crop,
                climate_data=climate,
                smax_calculation_option=3
            )
    
    def test_option1_and_option2_produce_different_results(self):
        """Test that Option 1 and Option 2 produce different Smax/Seav values."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        
        soil1 = SoilParameters(
            smax_base=150.0,
            reference_depth=0.6
        )
        
        soil2 = SoilParameters(
            smax_base=150.0,
            pawc_soil=0.18,
            zmax=2.0,
            p=0.5
        )
        
        crop = CropParameters(
            name='TestCrop',
            kc_values=np.ones(10),
            rooting_depth=np.ones(10) * 0.8,
            dates=dates
        )
        climate = ClimateData(
            precipitation=np.ones(10) * 5,
            pet=np.ones(10) * 3,
            dates=dates
        )
        
        model1 = WaterBalanceModel(
            soil_params=soil1,
            crop_params=crop,
            climate_data=climate,
            smax_calculation_option=1
        )
        
        model2 = WaterBalanceModel(
            soil_params=soil2,
            crop_params=crop,
            climate_data=climate,
            smax_calculation_option=2
        )
        
        smax1, seav1 = model1._calculate_dynamic_smax_seav(0)
        smax2, seav2 = model2._calculate_dynamic_smax_seav(0)
        
        # Results should be different
        assert smax1 != pytest.approx(smax2)
        assert seav1 != pytest.approx(seav2)
    
    def test_option2_with_different_soil_textures(self):
        """Test Option 2 with different PAWC values representing soil textures."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        
        # Sandy soil (low PAWC)
        soil_sandy = SoilParameters(
            smax_base=150.0,
            pawc_soil=0.10,
            zmax=2.0,
            p=0.5
        )
        
        # Loam soil (medium PAWC)
        soil_loam = SoilParameters(
            smax_base=150.0,
            pawc_soil=0.18,
            zmax=2.0,
            p=0.5
        )
        
        # Clay loam (high PAWC)
        soil_clay = SoilParameters(
            smax_base=150.0,
            pawc_soil=0.22,
            zmax=2.0,
            p=0.5
        )
        
        crop = CropParameters(
            name='TestCrop',
            kc_values=np.ones(10),
            rooting_depth=np.ones(10) * 1.0,
            dates=dates
        )
        climate = ClimateData(
            precipitation=np.ones(10) * 5,
            pet=np.ones(10) * 3,
            dates=dates
        )
        
        model_sandy = WaterBalanceModel(
            soil_params=soil_sandy, crop_params=crop,
            climate_data=climate, smax_calculation_option=2
        )
        model_loam = WaterBalanceModel(
            soil_params=soil_loam, crop_params=crop,
            climate_data=climate, smax_calculation_option=2
        )
        model_clay = WaterBalanceModel(
            soil_params=soil_clay, crop_params=crop,
            climate_data=climate, smax_calculation_option=2
        )
        
        smax_sandy, _ = model_sandy._calculate_dynamic_smax_seav(0)
        smax_loam, _ = model_loam._calculate_dynamic_smax_seav(0)
        smax_clay, _ = model_clay._calculate_dynamic_smax_seav(0)
        
        # Smax should increase with PAWC
        assert smax_sandy < smax_loam < smax_clay
        assert smax_sandy == pytest.approx(100.0)  # 0.10 * 1.0 * 1000
        assert smax_loam == pytest.approx(180.0)   # 0.18 * 1.0 * 1000
        assert smax_clay == pytest.approx(220.0)   # 0.22 * 1.0 * 1000


class TestWaterBalanceClosure:
    """Test water balance closure and conservation."""
    
    def test_mass_balance_single_step(self, water_balance_model):
        """Test mass balance for a single time step."""
        # Initial conditions
        sm_t1 = 100.0
        pr = 10.0
        pet = 5.0
        smax = 150.0
        seav = 75.0
        
        # Calculate components in correct order
        et = water_balance_model._update_et(sm_t1, pet, smax, seav)
        runoff = water_balance_model._update_runoff(sm_t1, pr, et, smax, irrig=0)
        wb = water_balance_model._update_water_balance(sm_t1, pr, et, runoff, irrig=0)
        sm_t = water_balance_model._update_soil_moisture(wb, smax)
        
        # Check water balance: SM_t = SM_t-1 + P - ET - R0
        # (percolation is separate and happens from current SM)
        delta_sm = sm_t - sm_t1
        mass_balance = pr - et - runoff
        
        # Allow small numerical error
        assert delta_sm == pytest.approx(mass_balance, abs=0.1)
