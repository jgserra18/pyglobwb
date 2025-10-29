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
    
    def test_smax_calculation(self, water_balance_model):
        """Test dynamic Smax calculation based on rooting depth."""
        day_idx = 0
        
        smax, seav = water_balance_model._calculate_dynamic_smax_seav(day_idx)
        
        expected_smax = (water_balance_model.soil.smax_base / 
                        water_balance_model.soil.reference_depth * 
                        water_balance_model.crop.rooting_depth[day_idx])
        
        assert smax == pytest.approx(expected_smax)
    
    def test_seav_calculation(self, water_balance_model):
        """Test that Seav is 50% of Smax."""
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
