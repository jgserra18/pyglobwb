"""
Configuration Manager for Water Balance Model

This module provides functions to load and manage YAML configuration files
for crop parameters, rooting depths, and irrigation efficiencies.
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings


class ConfigManager:
    """
    Manager for loading and accessing model configuration from YAML files.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Path to configuration directory. If None, uses default location.
        """
        if config_dir is None:
            # Default to config directory relative to this file
            self.config_dir = Path(__file__).parent.parent / 'config'
        else:
            self.config_dir = Path(config_dir)
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")
        
        # Load all configurations
        self._crop_kc = None
        self._rooting_depth = None
        self._irrigation_efficiency = None
    
    @property
    def crop_kc(self) -> Dict:
        """Load crop Kc configuration (lazy loading)."""
        if self._crop_kc is None:
            config_file = self.config_dir / 'crop_kc.yaml'
            with open(config_file, 'r') as f:
                self._crop_kc = yaml.safe_load(f)
        return self._crop_kc
    
    @property
    def rooting_depth(self) -> Dict:
        """Load rooting depth configuration (lazy loading)."""
        if self._rooting_depth is None:
            config_file = self.config_dir / 'rooting_depth.yaml'
            with open(config_file, 'r') as f:
                self._rooting_depth = yaml.safe_load(f)
        return self._rooting_depth
    
    @property
    def irrigation_efficiency(self) -> Dict:
        """Load irrigation efficiency configuration (lazy loading)."""
        if self._irrigation_efficiency is None:
            config_file = self.config_dir / 'irrigation_efficiency.yaml'
            with open(config_file, 'r') as f:
                self._irrigation_efficiency = yaml.safe_load(f)
        return self._irrigation_efficiency
    
    def get_crop_kc(self, crop_name: str) -> Dict:
        """
        Get crop Kc parameters.
        
        Args:
            crop_name: Name of the crop
            
        Returns:
            Dictionary with kc_monthly, order, and is_permanent
        """
        crops = self.crop_kc['crops']
        
        if crop_name not in crops:
            available = list(crops.keys())
            raise ValueError(f"Crop '{crop_name}' not found. Available crops: {available}")
        
        return crops[crop_name]
    
    def get_rooting_depth(self, crop_name: str, management: str = 'irrigated') -> float:
        """
        Get rooting depth for a crop.
        
        Args:
            crop_name: Name of the crop
            management: 'irrigated' or 'rainfed'
            
        Returns:
            Maximum rooting depth in meters
        """
        crops = self.rooting_depth['crops']
        
        if crop_name not in crops:
            available = list(crops.keys())
            raise ValueError(f"Crop '{crop_name}' not found. Available crops: {available}")
        
        if management.lower() not in ['irrigated', 'rainfed']:
            raise ValueError("management must be 'irrigated' or 'rainfed'")
        
        return crops[crop_name][management.lower()]
    
    def get_irrigation_efficiency(self, system: str) -> float:
        """
        Get irrigation efficiency for a system.
        
        Args:
            system: Name of irrigation system
            
        Returns:
            Efficiency value (0-1)
        """
        systems = self.irrigation_efficiency['irrigation_systems']
        
        if system not in systems:
            available = list(systems.keys())
            raise ValueError(f"Irrigation system '{system}' not found. Available systems: {available}")
        
        return systems[system]['efficiency']
    
    def list_crops(self) -> List[str]:
        """Get list of available crops."""
        return list(self.crop_kc['crops'].keys())
    
    def list_irrigation_systems(self) -> List[str]:
        """Get list of available irrigation systems."""
        return list(self.irrigation_efficiency['irrigation_systems'].keys())
    
    def create_daily_kc(
        self,
        crop_name: str,
        start_date: str,
        end_date: str
    ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Create daily Kc values from monthly configuration.
        
        Args:
            crop_name: Name of the crop
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Tuple of (daily_kc_array, dates)
        """
        crop_config = self.get_crop_kc(crop_name)
        monthly_kc = crop_config['kc_monthly']
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Expand monthly Kc to daily
        kc_daily = np.zeros(n_days)
        for i, date in enumerate(dates):
            month_idx = date.month - 1
            kc_daily[i] = monthly_kc[month_idx]
        
        return kc_daily, dates
    
    def create_daily_rooting_depth(
        self,
        crop_name: str,
        start_date: str,
        end_date: str,
        management: str = 'irrigated',
        interpolation_method: str = 'linear'
    ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Create daily rooting depth values with interpolation based on crop calendar.
        
        For annual crops: interpolates from 0.2m to max depth during growing season
        For permanent crops: constant at maximum depth
        
        Args:
            crop_name: Name of the crop
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            management: 'irrigated' or 'rainfed'
            interpolation_method: 'linear' or 'sigmoid'
            
        Returns:
            Tuple of (daily_rooting_depth_array, dates)
        """
        crop_config = self.get_crop_kc(crop_name)
        max_depth = self.get_rooting_depth(crop_name, management)
        is_permanent = crop_config['is_permanent']
        
        # Get daily Kc to determine growing season
        kc_daily, dates = self.create_daily_kc(crop_name, start_date, end_date)
        n_days = len(dates)
        
        rooting_depth = np.zeros(n_days)
        
        if is_permanent:
            # Permanent crops have constant maximum rooting depth
            rooting_depth[:] = max_depth
        else:
            # Annual crops: interpolate during growing season
            # Process each year separately
            for year in dates.year.unique():
                year_mask = dates.year == year
                year_indices = np.where(year_mask)[0]
                year_kc = kc_daily[year_mask]
                
                # Find growing season (where Kc > 0)
                growing_mask = year_kc > 0
                
                if not growing_mask.any():
                    # No growing season this year
                    rooting_depth[year_indices] = 0.2
                    continue
                
                # Find first and last day of growing season
                growing_days = np.where(growing_mask)[0]
                start_grow = growing_days[0]
                end_grow = growing_days[-1]
                
                # Find peak Kc day (maximum root development)
                peak_day = start_grow + np.argmax(year_kc[start_grow:end_grow+1])
                
                # Before growing season: minimal depth
                if start_grow > 0:
                    rooting_depth[year_indices[:start_grow]] = 0.2
                
                # Interpolate from start to peak
                if interpolation_method == 'linear':
                    n_develop = peak_day - start_grow + 1
                    if n_develop > 0:
                        rooting_depth[year_indices[start_grow:peak_day+1]] = np.linspace(
                            0.2, max_depth, n_develop
                        )
                elif interpolation_method == 'sigmoid':
                    # Sigmoid interpolation for more realistic root growth
                    n_develop = peak_day - start_grow + 1
                    if n_develop > 0:
                        x = np.linspace(-3, 3, n_develop)
                        sigmoid = 1 / (1 + np.exp(-x))
                        rooting_depth[year_indices[start_grow:peak_day+1]] = (
                            0.2 + (max_depth - 0.2) * sigmoid
                        )
                
                # After peak: constant at maximum
                rooting_depth[year_indices[peak_day:end_grow+1]] = max_depth
                
                # After growing season: back to minimal
                if end_grow < len(year_indices) - 1:
                    rooting_depth[year_indices[end_grow+1:]] = 0.2
        
        return rooting_depth, dates
    
    def update_crop_kc(
        self,
        crop_name: str,
        monthly_kc: List[float],
        is_permanent: bool = False,
        order: Optional[List[int]] = None
    ) -> None:
        """
        Dynamically update or add crop Kc configuration.
        
        Args:
            crop_name: Name of the crop
            monthly_kc: List of 12 monthly Kc values
            is_permanent: Whether crop is permanent
            order: Optional custom month order (1-12)
        """
        if len(monthly_kc) != 12:
            raise ValueError("monthly_kc must have exactly 12 values")
        
        if order is None:
            order = list(range(1, 13))
        
        if len(order) != 12:
            raise ValueError("order must have exactly 12 values")
        
        # Update in-memory configuration
        if 'crops' not in self.crop_kc:
            self.crop_kc['crops'] = {}
        
        self.crop_kc['crops'][crop_name] = {
            'kc_monthly': monthly_kc,
            'order': order,
            'is_permanent': is_permanent
        }
        
        print(f"Updated Kc configuration for crop: {crop_name}")
    
    def update_rooting_depth(
        self,
        crop_name: str,
        irrigated: float,
        rainfed: float
    ) -> None:
        """
        Dynamically update or add rooting depth configuration.
        
        Args:
            crop_name: Name of the crop
            irrigated: Maximum rooting depth for irrigated (m)
            rainfed: Maximum rooting depth for rainfed (m)
        """
        if irrigated <= 0 or rainfed <= 0:
            raise ValueError("Rooting depths must be positive")
        
        # Update in-memory configuration
        if 'crops' not in self.rooting_depth:
            self.rooting_depth['crops'] = {}
        
        self.rooting_depth['crops'][crop_name] = {
            'irrigated': irrigated,
            'rainfed': rainfed
        }
        
        print(f"Updated rooting depth configuration for crop: {crop_name}")
    
    def save_configurations(self) -> None:
        """
        Save current in-memory configurations back to YAML files.
        """
        # Save crop Kc
        if self._crop_kc is not None:
            config_file = self.config_dir / 'crop_kc.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(self._crop_kc, f, default_flow_style=False, sort_keys=False)
            print(f"Saved crop Kc configuration to {config_file}")
        
        # Save rooting depth
        if self._rooting_depth is not None:
            config_file = self.config_dir / 'rooting_depth.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(self._rooting_depth, f, default_flow_style=False, sort_keys=False)
            print(f"Saved rooting depth configuration to {config_file}")
        
        # Save irrigation efficiency
        if self._irrigation_efficiency is not None:
            config_file = self.config_dir / 'irrigation_efficiency.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(self._irrigation_efficiency, f, default_flow_style=False, sort_keys=False)
            print(f"Saved irrigation efficiency configuration to {config_file}")
    
    def export_crop_summary(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Export summary of all crop configurations.
        
        Args:
            output_file: Optional path to save CSV file
            
        Returns:
            DataFrame with crop summary
        """
        crops = self.list_crops()
        
        summary_data = []
        for crop in crops:
            kc_config = self.get_crop_kc(crop)
            
            try:
                irrig_depth = self.get_rooting_depth(crop, 'irrigated')
                rainfed_depth = self.get_rooting_depth(crop, 'rainfed')
            except ValueError:
                irrig_depth = None
                rainfed_depth = None
            
            summary_data.append({
                'Crop': crop,
                'Is_Permanent': kc_config['is_permanent'],
                'Kc_Min': min(kc_config['kc_monthly']),
                'Kc_Max': max(kc_config['kc_monthly']),
                'Kc_Mean': np.mean(kc_config['kc_monthly']),
                'Rooting_Depth_Irrigated_m': irrig_depth,
                'Rooting_Depth_Rainfed_m': rainfed_depth
            })
        
        df = pd.DataFrame(summary_data)
        
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Exported crop summary to {output_file}")
        
        return df


def load_config(config_dir: Optional[str] = None) -> ConfigManager:
    """
    Convenience function to load configuration manager.
    
    Args:
        config_dir: Path to configuration directory
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_dir)


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("Configuration Manager - Example Usage")
    print("=" * 60)
    
    # Load configuration
    config = ConfigManager()
    
    # List available crops
    print("\nAvailable crops:")
    crops = config.list_crops()
    for i, crop in enumerate(crops, 1):
        print(f"  {i}. {crop}")
    
    # List irrigation systems
    print("\nAvailable irrigation systems:")
    systems = config.list_irrigation_systems()
    for system in systems:
        eff = config.get_irrigation_efficiency(system)
        print(f"  - {system}: {eff:.0%}")
    
    # Example: Get Maize parameters
    print("\n" + "=" * 60)
    print("Example: Maize Configuration")
    print("=" * 60)
    
    crop_name = 'Maize'
    kc_config = config.get_crop_kc(crop_name)
    print(f"\nMonthly Kc values: {kc_config['kc_monthly']}")
    print(f"Is permanent crop: {kc_config['is_permanent']}")
    
    irrig_depth = config.get_rooting_depth(crop_name, 'irrigated')
    rainfed_depth = config.get_rooting_depth(crop_name, 'rainfed')
    print(f"Rooting depth (irrigated): {irrig_depth} m")
    print(f"Rooting depth (rainfed): {rainfed_depth} m")
    
    # Create daily values
    kc_daily, dates = config.create_daily_kc(crop_name, '2020-01-01', '2020-12-31')
    print(f"\nCreated {len(kc_daily)} daily Kc values")
    print(f"Kc range: {kc_daily.min():.3f} - {kc_daily.max():.3f}")
    
    depth_daily, _ = config.create_daily_rooting_depth(
        crop_name, '2020-01-01', '2020-12-31', 'irrigated'
    )
    print(f"Rooting depth range: {depth_daily.min():.3f} - {depth_daily.max():.3f} m")
    
    # Example: Update configuration dynamically
    print("\n" + "=" * 60)
    print("Example: Dynamic Configuration Update")
    print("=" * 60)
    
    # Add a new crop
    config.update_crop_kc(
        crop_name='Custom_Crop',
        monthly_kc=[0.3, 0.4, 0.6, 0.9, 1.1, 1.2, 1.2, 1.0, 0.7, 0.5, 0.3, 0.3],
        is_permanent=False
    )
    
    config.update_rooting_depth(
        crop_name='Custom_Crop',
        irrigated=0.8,
        rainfed=1.2
    )
    
    print("\nCustom crop added successfully!")
    print(f"Total crops: {len(config.list_crops())}")
    
    # Export summary
    print("\n" + "=" * 60)
    print("Exporting Crop Summary")
    print("=" * 60)
    summary = config.export_crop_summary()
    print(summary.head(10))
