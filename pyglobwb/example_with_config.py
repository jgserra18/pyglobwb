"""
Example: Using Water Balance Model with YAML Configuration

This script demonstrates how to use the ConfigManager to load crop parameters
from YAML files and run the water balance model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config_manager import ConfigManager
from .water_balance_model import (
    WaterBalanceModel,
    SoilParameters,
    CropParameters,
    ClimateData
)


def example_using_yaml_config():
    """
    Example: Load crop parameters from YAML and run model.
    """
    print("=" * 70)
    print("Water Balance Model with YAML Configuration")
    print("=" * 70)
    
    # Initialize configuration manager
    config = ConfigManager()
    
    # Define simulation period
    start_date = '2015-01-01'
    end_date = '2019-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Generate synthetic climate data
    print("\nGenerating synthetic climate data...")
    day_of_year = dates.dayofyear
    pr_base = 2.5 + 1.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    pr_noise = np.random.gamma(2, 0.5, n_days)
    precipitation = pr_base * pr_noise
    pet = 2.0 + 2.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + 1.0
    
    climate = ClimateData(
        precipitation=precipitation,
        pet=pet,
        dates=dates
    )
    
    # Define soil parameters
    soil = SoilParameters(
        smax_base=150.0,
        reference_depth=0.6,
        rmax=10.0,
        initial_sm=75.0,
        calibration_factor=2.4
    )
    
    # Select crop and management
    crop_name = 'Maize'
    management = 'irrigated'
    irrigation_system = 'Drip'
    
    print(f"\nCrop: {crop_name}")
    print(f"Management: {management}")
    print(f"Irrigation system: {irrigation_system}")
    
    # Load crop parameters from YAML
    print("\nLoading crop parameters from YAML configuration...")
    kc_daily, _ = config.create_daily_kc(crop_name, start_date, end_date)
    rooting_depth, _ = config.create_daily_rooting_depth(
        crop_name, start_date, end_date, management, interpolation_method='linear'
    )
    
    kc_config = config.get_crop_kc(crop_name)
    irrigation_efficiency = config.get_irrigation_efficiency(irrigation_system)
    
    print(f"  - Kc range: {kc_daily.min():.3f} - {kc_daily.max():.3f}")
    print(f"  - Rooting depth range: {rooting_depth.min():.2f} - {rooting_depth.max():.2f} m")
    print(f"  - Is permanent crop: {kc_config['is_permanent']}")
    print(f"  - Irrigation efficiency: {irrigation_efficiency:.0%}")
    
    # Create CropParameters object
    crop = CropParameters(
        name=crop_name,
        kc_values=kc_daily,
        rooting_depth=rooting_depth,
        dates=dates,
        landuse_kc=0.5
    )
    
    # Run model
    print("\nRunning water balance model...")
    model = WaterBalanceModel(
        soil_params=soil,
        crop_params=crop,
        climate_data=climate,
        management=management,
        irrigation_efficiency=irrigation_efficiency
    )
    
    results = model.run(spinup_iterations=50)
    
    # Get summaries
    annual = model.get_annual_summary(results)
    
    print("\n" + "=" * 70)
    print("ANNUAL SUMMARY")
    print("=" * 70)
    print(annual.to_string(index=False))
    
    # Calculate water balance statistics
    total_pr = results['precipitation'].sum()
    total_et = results['evapotranspiration'].sum()
    total_irrig = results['irrigation'].sum()
    total_perc = results['percolation'].sum()
    total_runoff = results['runoff'].sum()
    
    print("\n" + "=" * 70)
    print("TOTAL WATER BALANCE (5 years)")
    print("=" * 70)
    print(f"Precipitation:        {total_pr:8.1f} mm")
    print(f"Irrigation:           {total_irrig:8.1f} mm")
    print(f"Evapotranspiration:   {total_et:8.1f} mm")
    print(f"Percolation:          {total_perc:8.1f} mm")
    print(f"Runoff:               {total_runoff:8.1f} mm")
    print(f"{'-' * 70}")
    print(f"Input:                {total_pr + total_irrig:8.1f} mm")
    print(f"Output:               {total_et + total_perc + total_runoff:8.1f} mm")
    
    return results, config


def compare_multiple_crops():
    """
    Example: Compare water requirements for multiple crops.
    """
    print("\n" + "=" * 70)
    print("Comparing Multiple Crops")
    print("=" * 70)
    
    # Initialize configuration
    config = ConfigManager()
    
    # Define simulation period
    start_date = '2018-01-01'
    end_date = '2018-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Generate climate data
    day_of_year = dates.dayofyear
    pr_base = 2.5 + 1.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    pr_noise = np.random.gamma(2, 0.5, n_days)
    precipitation = pr_base * pr_noise
    pet = 2.0 + 2.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + 1.0
    
    climate = ClimateData(precipitation=precipitation, pet=pet, dates=dates)
    
    # Soil parameters
    soil = SoilParameters(
        smax_base=150.0,
        reference_depth=0.6,
        rmax=10.0,
        calibration_factor=2.4
    )
    
    # Crops to compare
    crops_to_compare = ['Maize', 'Wheat', 'Tomato', 'Olive']
    management = 'irrigated'
    irrigation_system = 'Drip'
    irrigation_efficiency = config.get_irrigation_efficiency(irrigation_system)
    
    results_dict = {}
    
    for crop_name in crops_to_compare:
        print(f"\nRunning model for {crop_name}...")
        
        # Load crop parameters
        kc_daily, _ = config.create_daily_kc(crop_name, start_date, end_date)
        rooting_depth, _ = config.create_daily_rooting_depth(
            crop_name, start_date, end_date, management
        )
        
        crop = CropParameters(
            name=crop_name,
            kc_values=kc_daily,
            rooting_depth=rooting_depth,
            dates=dates,
            landuse_kc=0.5
        )
        
        # Run model
        model = WaterBalanceModel(soil, crop, climate, management, irrigation_efficiency)
        results = model.run(spinup_iterations=30)
        results_dict[crop_name] = results
    
    # Create comparison table
    print("\n" + "=" * 70)
    print("CROP WATER REQUIREMENTS COMPARISON (Annual Totals)")
    print("=" * 70)
    
    comparison = pd.DataFrame({
        crop: {
            'Precipitation (mm)': data['precipitation'].sum(),
            'ET (mm)': data['evapotranspiration'].sum(),
            'Irrigation (mm)': data['irrigation'].sum(),
            'Percolation (mm)': data['percolation'].sum(),
            'Runoff (mm)': data['runoff'].sum(),
            'Avg Soil Moisture (mm)': data['soil_moisture'].mean()
        }
        for crop, data in results_dict.items()
    }).T
    
    print(comparison.round(1))
    
    # Plot comparison
    plot_crop_comparison(results_dict)
    
    return results_dict


def example_custom_crop():
    """
    Example: Add a custom crop dynamically and run model.
    """
    print("\n" + "=" * 70)
    print("Example: Custom Crop Configuration")
    print("=" * 70)
    
    # Initialize configuration
    config = ConfigManager()
    
    # Add custom crop
    custom_crop_name = 'Custom_Vegetable'
    print(f"\nAdding custom crop: {custom_crop_name}")
    
    # Define custom monthly Kc (growing season May-September)
    custom_kc = [0.0, 0.0, 0.0, 0.0, 0.5, 0.8, 1.1, 1.0, 0.6, 0.0, 0.0, 0.0]
    
    config.update_crop_kc(
        crop_name=custom_crop_name,
        monthly_kc=custom_kc,
        is_permanent=False
    )
    
    config.update_rooting_depth(
        crop_name=custom_crop_name,
        irrigated=0.6,
        rainfed=0.9
    )
    
    print(f"Custom crop added successfully!")
    print(f"Monthly Kc: {custom_kc}")
    
    # Run model with custom crop
    start_date = '2018-01-01'
    end_date = '2018-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Climate data
    day_of_year = dates.dayofyear
    pr = 2.5 + 1.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    pr = pr * np.random.gamma(2, 0.5, len(dates))
    pet = 2.0 + 2.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + 1.0
    
    climate = ClimateData(precipitation=pr, pet=pet, dates=dates)
    
    soil = SoilParameters(smax_base=150.0, rmax=10.0, calibration_factor=2.4)
    
    # Load custom crop parameters
    kc_daily, _ = config.create_daily_kc(custom_crop_name, start_date, end_date)
    rooting_depth, _ = config.create_daily_rooting_depth(
        custom_crop_name, start_date, end_date, 'irrigated'
    )
    
    crop = CropParameters(
        name=custom_crop_name,
        kc_values=kc_daily,
        rooting_depth=rooting_depth,
        dates=dates,
        landuse_kc=0.5
    )
    
    # Run model
    print("\nRunning model with custom crop...")
    model = WaterBalanceModel(
        soil, crop, climate, 'irrigated',
        config.get_irrigation_efficiency('Drip')
    )
    results = model.run()
    
    annual = model.get_annual_summary(results)
    print("\n" + "=" * 70)
    print("CUSTOM CROP RESULTS")
    print("=" * 70)
    print(annual.to_string(index=False))
    
    return results


def plot_crop_comparison(results_dict):
    """
    Plot comparison of multiple crops.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Crop Water Requirements Comparison', fontsize=14, fontweight='bold')
    
    crops = list(results_dict.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(crops)))
    
    # Annual totals
    annual_data = {
        crop: {
            'ET': data['evapotranspiration'].sum(),
            'Irrigation': data['irrigation'].sum(),
            'Percolation': data['percolation'].sum(),
            'Runoff': data['runoff'].sum()
        }
        for crop, data in results_dict.items()
    }
    
    # Plot 1: Annual water use
    ax = axes[0, 0]
    x = np.arange(len(crops))
    width = 0.35
    
    et_vals = [annual_data[c]['ET'] for c in crops]
    irrig_vals = [annual_data[c]['Irrigation'] for c in crops]
    
    ax.bar(x - width/2, et_vals, width, label='ET', alpha=0.8)
    ax.bar(x + width/2, irrig_vals, width, label='Irrigation', alpha=0.8)
    ax.set_ylabel('Water (mm/year)')
    ax.set_title('Annual Water Use')
    ax.set_xticks(x)
    ax.set_xticklabels(crops, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Water losses
    ax = axes[0, 1]
    perc_vals = [annual_data[c]['Percolation'] for c in crops]
    runoff_vals = [annual_data[c]['Runoff'] for c in crops]
    
    ax.bar(x - width/2, perc_vals, width, label='Percolation', alpha=0.8)
    ax.bar(x + width/2, runoff_vals, width, label='Runoff', alpha=0.8)
    ax.set_ylabel('Water (mm/year)')
    ax.set_title('Water Losses')
    ax.set_xticks(x)
    ax.set_xticklabels(crops, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Daily ET patterns (first 180 days)
    ax = axes[1, 0]
    for crop, color in zip(crops, colors):
        data = results_dict[crop]
        ax.plot(data['date'][:180], data['evapotranspiration'][:180],
               label=crop, color=color, linewidth=2)
    ax.set_ylabel('ET (mm/day)')
    ax.set_title('Daily ET Pattern (First 6 months)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Soil moisture patterns
    ax = axes[1, 1]
    for crop, color in zip(crops, colors):
        data = results_dict[crop]
        ax.plot(data['date'][:180], data['soil_moisture'][:180],
               label=crop, color=color, linewidth=2)
    ax.set_ylabel('Soil Moisture (mm)')
    ax.set_title('Soil Moisture Pattern (First 6 months)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('crop_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'crop_comparison.png'")
    plt.close()


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("WATER BALANCE MODEL WITH YAML CONFIGURATION - EXAMPLES")
    print("=" * 70)
    
    # Example 1: Basic usage with YAML config
    results1, config = example_using_yaml_config()
    
    # Example 2: Compare multiple crops
    results2 = compare_multiple_crops()
    
    # Example 3: Custom crop
    results3 = example_custom_crop()
    
    # Export crop summary
    print("\n" + "=" * 70)
    print("Exporting Crop Configuration Summary")
    print("=" * 70)
    summary = config.export_crop_summary('crop_summary.csv')
    print(f"\nCrop summary exported to 'crop_summary.csv'")
    print(f"Total crops configured: {len(summary)}")
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)
