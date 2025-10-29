"""
Example usage of the Vertical Water Balance Model

This script demonstrates how to use the water_balance_model module
with synthetic and real data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .water_balance_model import (
    WaterBalanceModel,
    SoilParameters,
    CropParameters,
    ClimateData,
    create_crop_parameters_from_monthly_kc,
    get_irrigation_efficiency
)


def example_1_synthetic_data():
    """
    Example 1: Run model with synthetic climate data.
    """
    print("=" * 60)
    print("Example 1: Synthetic Data - Maize Crop")
    print("=" * 60)
    
    # Define simulation period
    start_date = '2015-01-01'
    end_date = '2019-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Create synthetic climate data
    # Simple sinusoidal pattern with some noise
    day_of_year = dates.dayofyear
    
    # Precipitation: higher in winter, lower in summer
    pr_base = 3.0 + 2.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    pr_noise = np.random.gamma(2, 0.5, n_days)
    precipitation = pr_base * pr_noise
    
    # PET: lower in winter, higher in summer
    pet = 2.0 + 2.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + 1.5
    
    climate = ClimateData(
        precipitation=precipitation,
        pet=pet,
        dates=dates
    )
    
    # Define soil parameters
    soil = SoilParameters(
        smax_base=150.0,  # mm at 0.6m depth
        reference_depth=0.6,  # m
        rmax=10.0,  # mm/day
        initial_sm=75.0,  # mm
        calibration_factor=2.4
    )
    
    # Define crop parameters for maize
    # Monthly Kc values (Jan-Dec)
    monthly_kc_maize = [0, 0, 0, 0.3, 0.64, 1.17, 1.2, 0.70, 0, 0, 0, 0]
    
    crop = create_crop_parameters_from_monthly_kc(
        crop_name='Maize',
        monthly_kc=monthly_kc_maize,
        rooting_depth_max=1.2,  # m
        start_date=start_date,
        end_date=end_date,
        landuse_kc=0.5,
        is_permanent_crop=False
    )
    
    # Run rainfed scenario
    print("\nRunning RAINFED scenario...")
    model_rainfed = WaterBalanceModel(
        soil_params=soil,
        crop_params=crop,
        climate_data=climate,
        management='rainfed'
    )
    results_rainfed = model_rainfed.run(spinup_iterations=50)
    
    # Run irrigated scenario with drip irrigation
    print("Running IRRIGATED scenario (drip irrigation)...")
    model_irrigated = WaterBalanceModel(
        soil_params=soil,
        crop_params=crop,
        climate_data=climate,
        management='irrigated',
        irrigation_efficiency=get_irrigation_efficiency('drip')
    )
    results_irrigated = model_irrigated.run(spinup_iterations=50)
    
    # Calculate annual summaries
    annual_rainfed = model_rainfed.get_annual_summary(results_rainfed)
    annual_irrigated = model_irrigated.get_annual_summary(results_irrigated)
    
    print("\n" + "=" * 60)
    print("ANNUAL SUMMARY - RAINFED")
    print("=" * 60)
    print(annual_rainfed.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("ANNUAL SUMMARY - IRRIGATED")
    print("=" * 60)
    print(annual_irrigated.to_string(index=False))
    
    # Plot comparison
    plot_comparison(results_rainfed, results_irrigated, 2017)
    
    return results_rainfed, results_irrigated


def example_2_wheat_crop():
    """
    Example 2: Winter wheat with different irrigation systems.
    """
    print("\n" + "=" * 60)
    print("Example 2: Wheat Crop - Irrigation System Comparison")
    print("=" * 60)
    
    # Define simulation period
    start_date = '2015-01-01'
    end_date = '2019-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Create synthetic climate data
    day_of_year = dates.dayofyear
    pr_base = 2.5 + 1.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    pr_noise = np.random.gamma(2, 0.5, n_days)
    precipitation = pr_base * pr_noise
    pet = 1.5 + 2.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + 1.0
    
    climate = ClimateData(
        precipitation=precipitation,
        pet=pet,
        dates=dates
    )
    
    # Soil parameters
    soil = SoilParameters(
        smax_base=120.0,
        reference_depth=0.6,
        rmax=8.0,
        initial_sm=60.0,
        calibration_factor=2.4
    )
    
    # Wheat crop parameters
    # Monthly Kc values (Jan-Dec) - winter wheat
    monthly_kc_wheat = [0.4, 0.5, 0.7, 1.15, 1.15, 0.4, 0, 0, 0, 0, 0.3, 0.4]
    
    crop = create_crop_parameters_from_monthly_kc(
        crop_name='Wheat',
        monthly_kc=monthly_kc_wheat,
        rooting_depth_max=1.0,
        start_date=start_date,
        end_date=end_date,
        landuse_kc=0.5,
        is_permanent_crop=False
    )
    
    # Compare different irrigation systems
    irrigation_systems = ['rainfed', 'drip', 'sprinkler', 'traditional']
    results_dict = {}
    
    for system in irrigation_systems:
        print(f"\nRunning {system.upper()} scenario...")
        
        if system == 'rainfed':
            management = 'rainfed'
            efficiency = 1.0
        else:
            management = 'irrigated'
            efficiency = get_irrigation_efficiency(system)
        
        model = WaterBalanceModel(
            soil_params=soil,
            crop_params=crop,
            climate_data=climate,
            management=management,
            irrigation_efficiency=efficiency
        )
        
        results = model.run(spinup_iterations=50)
        annual = model.get_annual_summary(results)
        results_dict[system] = annual
    
    # Print comparison
    print("\n" + "=" * 60)
    print("IRRIGATION SYSTEM COMPARISON (Average Annual Values)")
    print("=" * 60)
    
    comparison = pd.DataFrame({
        system: {
            'Precipitation (mm)': data['precipitation'].mean(),
            'PET (mm)': data['pet'].mean(),
            'ET (mm)': data['evapotranspiration'].mean(),
            'Irrigation (mm)': data['irrigation'].mean(),
            'Percolation (mm)': data['percolation'].mean(),
            'Runoff (mm)': data['runoff'].mean(),
            'Avg Soil Moisture (mm)': data['soil_moisture'].mean()
        }
        for system, data in results_dict.items()
    }).T
    
    print(comparison.round(1))
    
    return results_dict


def example_3_permanent_crop():
    """
    Example 3: Permanent crop (olive) with constant rooting depth.
    """
    print("\n" + "=" * 60)
    print("Example 3: Olive Tree (Permanent Crop)")
    print("=" * 60)
    
    # Define simulation period
    start_date = '2015-01-01'
    end_date = '2019-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Mediterranean climate pattern
    day_of_year = dates.dayofyear
    # Dry summer, wet winter
    pr_base = 4.0 - 3.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    pr_noise = np.random.gamma(1.5, 0.8, n_days)
    precipitation = pr_base * pr_noise
    precipitation = np.maximum(0, precipitation)
    
    # High summer PET
    pet = 3.0 + 3.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + 1.0
    
    climate = ClimateData(
        precipitation=precipitation,
        pet=pet,
        dates=dates
    )
    
    # Soil parameters
    soil = SoilParameters(
        smax_base=180.0,  # Higher storage for deep-rooted olive
        reference_depth=0.6,
        rmax=12.0,
        initial_sm=90.0,
        calibration_factor=2.4
    )
    
    # Olive crop parameters
    # Monthly Kc values (Jan-Dec)
    monthly_kc_olive = [0.5, 0.5, 0.6, 0.65, 0.7, 0.7, 0.7, 0.7, 0.65, 0.6, 0.55, 0.5]
    
    crop = create_crop_parameters_from_monthly_kc(
        crop_name='Olive',
        monthly_kc=monthly_kc_olive,
        rooting_depth_max=1.5,  # Deep rooting
        start_date=start_date,
        end_date=end_date,
        landuse_kc=0.5,
        is_permanent_crop=True  # Constant rooting depth
    )
    
    # Run irrigated scenario
    print("\nRunning IRRIGATED scenario (drip irrigation)...")
    model = WaterBalanceModel(
        soil_params=soil,
        crop_params=crop,
        climate_data=climate,
        management='irrigated',
        irrigation_efficiency=get_irrigation_efficiency('drip')
    )
    
    results = model.run(spinup_iterations=50)
    annual = model.get_annual_summary(results)
    monthly = model.get_monthly_summary(results)
    
    print("\n" + "=" * 60)
    print("ANNUAL SUMMARY")
    print("=" * 60)
    print(annual.to_string(index=False))
    
    # Plot seasonal pattern
    plot_seasonal_pattern(monthly)
    
    return results, annual, monthly


def plot_comparison(results_rainfed, results_irrigated, year):
    """
    Plot comparison between rainfed and irrigated scenarios for a specific year.
    """
    # Filter for specific year
    mask_rf = results_rainfed['date'].dt.year == year
    mask_ir = results_irrigated['date'].dt.year == year
    
    rf = results_rainfed[mask_rf].copy()
    ir = results_irrigated[mask_ir].copy()
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle(f'Water Balance Comparison - Year {year}', fontsize=14, fontweight='bold')
    
    # Soil moisture
    axes[0].plot(rf['date'], rf['soil_moisture'], label='Rainfed', color='brown')
    axes[0].plot(ir['date'], ir['soil_moisture'], label='Irrigated', color='blue')
    axes[0].set_ylabel('Soil Moisture (mm)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ET
    axes[1].plot(rf['date'], rf['evapotranspiration'], label='Rainfed', color='brown')
    axes[1].plot(ir['date'], ir['evapotranspiration'], label='Irrigated', color='blue')
    axes[1].set_ylabel('ET (mm/day)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Irrigation
    axes[2].bar(ir['date'], ir['irrigation'], label='Irrigation', color='cyan', alpha=0.6)
    axes[2].set_ylabel('Irrigation (mm/day)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Water inputs and outputs
    axes[3].bar(rf['date'], rf['precipitation'], label='Precipitation', color='blue', alpha=0.4)
    axes[3].plot(rf['date'], rf['runoff'], label='Runoff (Rainfed)', color='red', linewidth=2)
    axes[3].plot(ir['date'], ir['runoff'], label='Runoff (Irrigated)', color='orange', linewidth=2)
    axes[3].set_ylabel('Water (mm/day)')
    axes[3].set_xlabel('Date')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('water_balance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'water_balance_comparison.png'")
    plt.close()


def plot_seasonal_pattern(monthly_results):
    """
    Plot seasonal patterns of water balance components.
    """
    # Average by month across all years
    monthly_avg = monthly_results.groupby('month').mean()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Seasonal Water Balance Patterns', fontsize=14, fontweight='bold')
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Precipitation and PET
    axes[0, 0].bar(months, monthly_avg['precipitation'], label='Precipitation', alpha=0.6)
    axes[0, 0].plot(months, monthly_avg['pet'], 'r-o', label='PET', linewidth=2)
    axes[0, 0].set_ylabel('Water (mm/month)')
    axes[0, 0].set_title('Climate Forcing')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # ET and Irrigation
    axes[0, 1].plot(months, monthly_avg['evapotranspiration'], 'g-o', label='ET', linewidth=2)
    axes[0, 1].plot(months, monthly_avg['irrigation'], 'b-s', label='Irrigation', linewidth=2)
    axes[0, 1].set_ylabel('Water (mm/month)')
    axes[0, 1].set_title('Water Use')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Soil moisture
    axes[1, 0].plot(months, monthly_avg['soil_moisture'], 'brown', marker='o', linewidth=2)
    axes[1, 0].set_ylabel('Soil Moisture (mm)')
    axes[1, 0].set_title('Soil Water Storage')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Losses
    axes[1, 1].plot(months, monthly_avg['percolation'], 'purple', marker='o', label='Percolation', linewidth=2)
    axes[1, 1].plot(months, monthly_avg['runoff'], 'red', marker='s', label='Runoff', linewidth=2)
    axes[1, 1].set_ylabel('Water (mm/month)')
    axes[1, 1].set_title('Water Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('seasonal_patterns.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as 'seasonal_patterns.png'")
    plt.close()


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run examples
    print("\n" + "=" * 60)
    print("VERTICAL WATER BALANCE MODEL - EXAMPLES")
    print("=" * 60)
    
    # Example 1: Maize with rainfed vs irrigated
    results_rf, results_ir = example_1_synthetic_data()
    
    # Example 2: Wheat with different irrigation systems
    wheat_results = example_2_wheat_crop()
    
    # Example 3: Permanent crop (olive)
    olive_results, olive_annual, olive_monthly = example_3_permanent_crop()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - water_balance_comparison.png")
    print("  - seasonal_patterns.png")
