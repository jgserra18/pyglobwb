"""
Test suite for configuration manager and YAML loading.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestConfigManagerInitialization:
    """Test suite for ConfigManager initialization."""
    
    def test_initialization_default_path(self, config_manager):
        """Test ConfigManager initialization with default path."""
        assert config_manager.config_dir.exists()
        assert config_manager.config_dir.is_dir()
    
    def test_initialization_custom_path(self):
        """Test ConfigManager initialization with custom path."""
        from config_manager import ConfigManager
        
        config_dir = Path(__file__).parent.parent / 'config'
        config = ConfigManager(str(config_dir))
        
        assert config.config_dir == config_dir
    
    def test_initialization_invalid_path(self):
        """Test that invalid path raises error."""
        from config_manager import ConfigManager
        
        with pytest.raises(FileNotFoundError):
            ConfigManager('/invalid/path/to/config')


class TestCropKcLoading:
    """Test suite for crop Kc configuration loading."""
    
    def test_load_crop_kc(self, config_manager):
        """Test loading crop Kc configuration."""
        crop_kc = config_manager.crop_kc
        
        assert 'crops' in crop_kc
        assert isinstance(crop_kc['crops'], dict)
        assert len(crop_kc['crops']) > 0
    
    def test_get_crop_kc_valid(self, config_manager):
        """Test getting Kc for valid crop."""
        maize_kc = config_manager.get_crop_kc('Maize')
        
        assert 'kc_monthly' in maize_kc
        assert 'order' in maize_kc
        assert 'is_permanent' in maize_kc
        assert len(maize_kc['kc_monthly']) == 12
    
    def test_get_crop_kc_invalid(self, config_manager):
        """Test that invalid crop raises error."""
        with pytest.raises(ValueError, match="not found"):
            config_manager.get_crop_kc('InvalidCrop')
    
    def test_list_crops(self, config_manager):
        """Test listing available crops."""
        crops = config_manager.list_crops()
        
        assert isinstance(crops, list)
        assert len(crops) > 0
        assert 'Maize' in crops
        assert 'Wheat' in crops
    
    def test_kc_values_range(self, config_manager):
        """Test that all Kc values are in valid range."""
        crops = config_manager.list_crops()
        
        for crop in crops:
            kc_config = config_manager.get_crop_kc(crop)
            kc_values = kc_config['kc_monthly']
            
            assert all(0 <= kc <= 1.5 for kc in kc_values), f"{crop} has invalid Kc values"


class TestRootingDepthLoading:
    """Test suite for rooting depth configuration loading."""
    
    def test_load_rooting_depth(self, config_manager):
        """Test loading rooting depth configuration."""
        rooting_depth = config_manager.rooting_depth
        
        assert 'crops' in rooting_depth
        assert isinstance(rooting_depth['crops'], dict)
    
    def test_get_rooting_depth_irrigated(self, config_manager):
        """Test getting rooting depth for irrigated management."""
        depth = config_manager.get_rooting_depth('Maize', 'irrigated')
        
        assert isinstance(depth, (int, float))
        assert depth > 0
        assert depth < 5.0  # Reasonable upper bound
    
    def test_get_rooting_depth_rainfed(self, config_manager):
        """Test getting rooting depth for rainfed management."""
        depth = config_manager.get_rooting_depth('Maize', 'rainfed')
        
        assert isinstance(depth, (int, float))
        assert depth > 0
    
    def test_rainfed_deeper_than_irrigated(self, config_manager):
        """Test that rainfed rooting depth is typically deeper."""
        crops = config_manager.list_crops()
        
        deeper_count = 0
        for crop in crops:
            try:
                irrig = config_manager.get_rooting_depth(crop, 'irrigated')
                rainfed = config_manager.get_rooting_depth(crop, 'rainfed')
                if rainfed >= irrig:
                    deeper_count += 1
            except ValueError:
                pass
        
        # Most crops should have rainfed >= irrigated
        assert deeper_count > len(crops) * 0.8
    
    def test_get_rooting_depth_invalid_management(self, config_manager):
        """Test that invalid management type raises error."""
        with pytest.raises(ValueError, match="must be"):
            config_manager.get_rooting_depth('Maize', 'invalid')


class TestIrrigationEfficiencyLoading:
    """Test suite for irrigation efficiency configuration loading."""
    
    def test_load_irrigation_efficiency(self, config_manager):
        """Test loading irrigation efficiency configuration."""
        irrig_eff = config_manager.irrigation_efficiency
        
        assert 'irrigation_systems' in irrig_eff
        assert isinstance(irrig_eff['irrigation_systems'], dict)
    
    def test_get_irrigation_efficiency_valid(self, config_manager):
        """Test getting efficiency for valid system."""
        efficiency = config_manager.get_irrigation_efficiency('Drip')
        
        assert isinstance(efficiency, (int, float))
        assert 0 < efficiency <= 1.0
    
    def test_get_irrigation_efficiency_invalid(self, config_manager):
        """Test that invalid system raises error."""
        with pytest.raises(ValueError, match="not found"):
            config_manager.get_irrigation_efficiency('InvalidSystem')
    
    def test_list_irrigation_systems(self, config_manager):
        """Test listing available irrigation systems."""
        systems = config_manager.list_irrigation_systems()
        
        assert isinstance(systems, list)
        assert len(systems) > 0
        assert 'Drip' in systems
        assert 'Sprinkler' in systems
    
    def test_efficiency_values_range(self, config_manager):
        """Test that all efficiency values are in valid range."""
        systems = config_manager.list_irrigation_systems()
        
        for system in systems:
            eff = config_manager.get_irrigation_efficiency(system)
            assert 0 < eff <= 1.0, f"{system} has invalid efficiency"


class TestDailyKcCreation:
    """Test suite for daily Kc creation."""
    
    def test_create_daily_kc(self, config_manager):
        """Test creating daily Kc values."""
        kc_daily, dates = config_manager.create_daily_kc(
            'Maize', '2020-01-01', '2020-12-31'
        )
        
        assert isinstance(kc_daily, np.ndarray)
        assert isinstance(dates, pd.DatetimeIndex)
        assert len(kc_daily) == len(dates)
        assert len(dates) == 366  # 2020 is leap year
    
    def test_daily_kc_values_range(self, config_manager):
        """Test that daily Kc values are in valid range."""
        kc_daily, _ = config_manager.create_daily_kc(
            'Maize', '2020-01-01', '2020-12-31'
        )
        
        assert np.all(kc_daily >= 0)
        assert np.all(kc_daily <= 1.5)
    
    def test_daily_kc_matches_monthly(self, config_manager):
        """Test that daily Kc matches monthly configuration."""
        crop_config = config_manager.get_crop_kc('Maize')
        monthly_kc = crop_config['kc_monthly']
        
        kc_daily, dates = config_manager.create_daily_kc(
            'Maize', '2020-01-01', '2020-01-31'
        )
        
        # All January days should have January Kc
        assert np.all(kc_daily == monthly_kc[0])


class TestDailyRootingDepthCreation:
    """Test suite for daily rooting depth creation."""
    
    def test_create_daily_rooting_depth(self, config_manager):
        """Test creating daily rooting depth values."""
        depth_daily, dates = config_manager.create_daily_rooting_depth(
            'Maize', '2020-01-01', '2020-12-31', 'irrigated'
        )
        
        assert isinstance(depth_daily, np.ndarray)
        assert isinstance(dates, pd.DatetimeIndex)
        assert len(depth_daily) == len(dates)
    
    def test_permanent_crop_constant_depth(self, config_manager):
        """Test that permanent crops have constant rooting depth."""
        depth_daily, _ = config_manager.create_daily_rooting_depth(
            'Olive', '2020-01-01', '2020-12-31', 'irrigated'
        )
        
        # Should be constant
        assert np.all(depth_daily == depth_daily[0])
    
    def test_annual_crop_variable_depth(self, config_manager):
        """Test that annual crops have variable rooting depth."""
        depth_daily, _ = config_manager.create_daily_rooting_depth(
            'Maize', '2020-01-01', '2020-12-31', 'irrigated'
        )
        
        # Should vary during growing season
        assert depth_daily.min() < depth_daily.max()
    
    def test_rooting_depth_reaches_maximum(self, config_manager):
        """Test that rooting depth reaches configured maximum."""
        max_depth = config_manager.get_rooting_depth('Maize', 'irrigated')
        depth_daily, _ = config_manager.create_daily_rooting_depth(
            'Maize', '2020-01-01', '2020-12-31', 'irrigated'
        )
        
        assert depth_daily.max() == pytest.approx(max_depth, rel=0.01)
    
    def test_rooting_depth_non_negative(self, config_manager):
        """Test that rooting depth is always non-negative."""
        depth_daily, _ = config_manager.create_daily_rooting_depth(
            'Maize', '2020-01-01', '2020-12-31', 'irrigated'
        )
        
        assert np.all(depth_daily >= 0)
    
    def test_interpolation_methods(self, config_manager):
        """Test different interpolation methods."""
        depth_linear, _ = config_manager.create_daily_rooting_depth(
            'Maize', '2020-01-01', '2020-12-31', 'irrigated', 'linear'
        )
        
        depth_sigmoid, _ = config_manager.create_daily_rooting_depth(
            'Maize', '2020-01-01', '2020-12-31', 'irrigated', 'sigmoid'
        )
        
        # Both should reach same maximum
        assert depth_linear.max() == pytest.approx(depth_sigmoid.max(), rel=0.01)
        
        # Sigmoid should be smoother (less linear)
        assert not np.array_equal(depth_linear, depth_sigmoid)


class TestDynamicConfigurationUpdate:
    """Test suite for dynamic configuration updates."""
    
    def test_update_crop_kc(self, config_manager):
        """Test updating crop Kc configuration."""
        custom_kc = [0.3, 0.4, 0.6, 0.9, 1.1, 1.2, 1.2, 1.0, 0.7, 0.5, 0.3, 0.3]
        
        config_manager.update_crop_kc(
            crop_name='TestCrop',
            monthly_kc=custom_kc,
            is_permanent=False
        )
        
        # Should be able to retrieve it
        crop_config = config_manager.get_crop_kc('TestCrop')
        assert crop_config['kc_monthly'] == custom_kc
        assert crop_config['is_permanent'] is False
    
    def test_update_crop_kc_invalid_length(self, config_manager):
        """Test that invalid Kc length raises error."""
        with pytest.raises(ValueError, match="12 values"):
            config_manager.update_crop_kc(
                crop_name='TestCrop',
                monthly_kc=[0.5, 0.6, 0.7],  # Only 3 values
                is_permanent=False
            )
    
    def test_update_rooting_depth(self, config_manager):
        """Test updating rooting depth configuration."""
        config_manager.update_rooting_depth(
            crop_name='TestCrop',
            irrigated=0.8,
            rainfed=1.2
        )
        
        # Should be able to retrieve it
        irrig_depth = config_manager.get_rooting_depth('TestCrop', 'irrigated')
        rainfed_depth = config_manager.get_rooting_depth('TestCrop', 'rainfed')
        
        assert irrig_depth == 0.8
        assert rainfed_depth == 1.2
    
    def test_update_rooting_depth_invalid_values(self, config_manager):
        """Test that invalid rooting depth raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            config_manager.update_rooting_depth(
                crop_name='TestCrop',
                irrigated=-0.5,
                rainfed=1.0
            )


class TestConfigurationExport:
    """Test suite for configuration export functionality."""
    
    def test_export_crop_summary(self, config_manager, tmp_path):
        """Test exporting crop summary."""
        output_file = tmp_path / 'crop_summary.csv'
        
        summary = config_manager.export_crop_summary(str(output_file))
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0
        assert output_file.exists()
    
    def test_crop_summary_columns(self, config_manager):
        """Test that crop summary has expected columns."""
        summary = config_manager.export_crop_summary()
        
        expected_cols = [
            'Crop', 'Is_Permanent', 'Kc_Min', 'Kc_Max', 'Kc_Mean',
            'Rooting_Depth_Irrigated_m', 'Rooting_Depth_Rainfed_m'
        ]
        
        for col in expected_cols:
            assert col in summary.columns
    
    def test_crop_summary_values(self, config_manager):
        """Test that crop summary values are reasonable."""
        summary = config_manager.export_crop_summary()
        
        # Kc values should be in range
        assert (summary['Kc_Min'] >= 0).all()
        assert (summary['Kc_Max'] <= 1.5).all()
        assert (summary['Kc_Mean'] >= summary['Kc_Min']).all()
        assert (summary['Kc_Mean'] <= summary['Kc_Max']).all()
