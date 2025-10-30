"""
Example: Water Balance Model with Real Climate Data

This example demonstrates:
1. Fetching real climate data from OpenMeteo using climatePy
2. Running irrigated maize simulation with default parameters
3. Using 50-year spinup for equilibration
4. Plotting comprehensive water balance results

Location: Eastern Spain (38°N, 1.5°W)
Period: 2015-2021 (daily)
Crop: Irrigated Maize
"""

import sys
from pathlib import Path

# Add pyglobwb to path
pyglobwb_path = Path(__file__).parent.parent / 'pyglobwb'
sys.path.insert(0, str(pyglobwb_path))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config_manager import ConfigManager
from water_balance_model import (
    WaterBalanceModel,
    SoilParameters,
    CropParameters
)
from climate_utils import (
    fetch_climate_from_openmeteo,
    prepare_climate_data,
    validate_climate_data,
    calculate_aridity_index
)

# Note: fetch_climate_from_openmeteo now uses Open-Meteo ERA5 API (daily data, no API key needed)




def setup_maize_parameters(config, start_date, end_date):
    """
    Set up maize crop parameters from YAML configuration.
    
    Args:
        config: ConfigManager instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        CropParameters object
    """
    crop_name = 'Maize'
    management = 'irrigated'
    
    print("\n" + "=" * 70)
    print("Loading Maize Parameters from Configuration")
    print("=" * 70)
    
    # Load from YAML
    kc_daily, dates = config.create_daily_kc(crop_name, start_date, end_date)
    rooting_depth, _ = config.create_daily_rooting_depth(
        crop_name, start_date, end_date, management, interpolation_method='linear'
    )
    
    kc_config = config.get_crop_kc(crop_name)
    
    print(f"Crop: {crop_name}")
    print(f"Management: {management}")
    print(f"Kc range: {kc_daily.min():.3f} - {kc_daily.max():.3f}")
    print(f"Rooting depth range: {rooting_depth.min():.2f} - {rooting_depth.max():.2f} m")
    print(f"Is permanent crop: {kc_config['is_permanent']}")
    
    crop = CropParameters(
        name=crop_name,
        kc_values=kc_daily,
        rooting_depth=rooting_depth,
        dates=dates,
        landuse_kc=0.5
    )
    
    return crop


def setup_soil_parameters():
    """
    Set up soil parameters for Eastern Spain.
    
    Returns:
        SoilParameters object
    """
    print("\n" + "=" * 70)
    print("Soil Parameters (Mediterranean loam)")
    print("=" * 70)
    
    soil = SoilParameters(
        smax_base=150.0,      # mm at 0.6m depth
        reference_depth=0.6,  # m
        rmax=10.0,            # mm/day
        initial_sm=75.0,      # mm (50% of smax_base)
        calibration_factor=2.4  # Mediterranean region
    )
    
    print(f"Maximum soil moisture (base): {soil.smax_base} mm")
    print(f"Reference depth: {soil.reference_depth} m")
    print(f"Maximum percolation rate: {soil.rmax} mm/day")
    print(f"Initial soil moisture: {soil.initial_sm} mm")
    print(f"Calibration factor: {soil.calibration_factor}")
    
    return soil


def run_simulation(soil, crop, climate, config, spinup_iterations=50):
    """
    Run water balance simulation.
    
    Args:
        soil: SoilParameters object
        crop: CropParameters object
        climate: ClimateData object
        config: ConfigManager instance
        spinup_iterations: Number of spinup iterations
        
    Returns:
        DataFrame with simulation results
    """
    print("\n" + "=" * 70)
    print("Running Water Balance Model")
    print("=" * 70)
    
    # Get irrigation efficiency for drip system
    irrigation_system = 'Drip'
    irrigation_efficiency = config.get_irrigation_efficiency(irrigation_system)
    
    print(f"Irrigation system: {irrigation_system}")
    print(f"Irrigation efficiency: {irrigation_efficiency:.0%}")
    print(f"Spinup iterations: {spinup_iterations}")
    
    # Create and run model
    model = WaterBalanceModel(
        soil_params=soil,
        crop_params=crop,
        climate_data=climate,
        management='irrigated',
        irrigation_efficiency=irrigation_efficiency
    )
    
    print("\nRunning simulation...")
    results = model.run(spinup_iterations=spinup_iterations)
    
    print("Simulation complete!")
    
    # Get summaries
    annual = model.get_annual_summary(results)
    
    print("\n" + "=" * 70)
    print("ANNUAL SUMMARY")
    print("=" * 70)
    print(annual.to_string(index=False))
    
    # Calculate total water balance
    total_pr = results['precipitation'].sum()
    total_et = results['evapotranspiration'].sum()
    total_irrig = results['irrigation'].sum()
    total_perc = results['percolation'].sum()
    total_runoff = results['runoff'].sum()
    
    print("\n" + "=" * 70)
    print(f"TOTAL WATER BALANCE ({len(results['date'].dt.year.unique())} years)")
    print("=" * 70)
    print(f"Precipitation:        {total_pr:8.1f} mm")
    print(f"Irrigation:           {total_irrig:8.1f} mm")
    print(f"Evapotranspiration:   {total_et:8.1f} mm")
    print(f"Percolation:          {total_perc:8.1f} mm")
    print(f"Runoff:               {total_runoff:8.1f} mm")
    print(f"{'-' * 70}")
    print(f"Input:                {total_pr + total_irrig:8.1f} mm")
    print(f"Output:               {total_et + total_perc + total_runoff:8.1f} mm")
    print(f"Balance error:        {(total_pr + total_irrig) - (total_et + total_perc + total_runoff):8.1f} mm")
    
    return results, annual


def plot_results(results, annual, output_dir='usage'):
    """
    Create comprehensive plots of water balance results.
    
    Args:
        results: Daily results DataFrame
        annual: Annual summary DataFrame
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 70)
    print("Creating Plots")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Plot 1: Time series for one year (2018)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('Water Balance Components - 2018', fontsize=16, fontweight='bold')
    
    # Filter for 2018
    mask_2018 = results['date'].dt.year == 2018
    data_2018 = results[mask_2018].copy()
    
    # Soil moisture
    axes[0].plot(data_2018['date'], data_2018['soil_moisture'], 'brown', linewidth=2)
    axes[0].plot(data_2018['date'], data_2018['smax'], 'k--', alpha=0.5, label='Smax')
    axes[0].plot(data_2018['date'], data_2018['seav'], 'r--', alpha=0.5, label='Seav')
    axes[0].set_ylabel('Soil Moisture (mm)', fontsize=11)
    axes[0].set_title('Soil Moisture Storage', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # ET and PET
    axes[1].plot(data_2018['date'], data_2018['pet'], 'r-', alpha=0.6, label='PET', linewidth=1.5)
    axes[1].plot(data_2018['date'], data_2018['evapotranspiration'], 'g-', label='Actual ET', linewidth=2)
    axes[1].set_ylabel('ET (mm/day)', fontsize=11)
    axes[1].set_title('Evapotranspiration', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Water inputs
    axes[2].bar(data_2018['date'], data_2018['precipitation'], label='Precipitation', 
                color='blue', alpha=0.6, width=1)
    axes[2].bar(data_2018['date'], data_2018['irrigation'], label='Irrigation', 
                color='cyan', alpha=0.8, width=1, bottom=data_2018['precipitation'])
    axes[2].set_ylabel('Water Input (mm/day)', fontsize=11)
    axes[2].set_title('Water Inputs', fontsize=12)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    # Water losses
    axes[3].plot(data_2018['date'], data_2018['percolation'], 'purple', 
                label='Percolation', linewidth=2)
    axes[3].plot(data_2018['date'], data_2018['runoff'], 'red', 
                label='Runoff', linewidth=2)
    axes[3].set_ylabel('Water Loss (mm/day)', fontsize=11)
    axes[3].set_xlabel('Date', fontsize=11)
    axes[3].set_title('Water Losses', fontsize=12)
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot1_path = output_path / 'water_balance_2018.png'
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot1_path}")
    plt.close()
    
    # Plot 2: Annual summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Annual Water Balance Summary (2015-2021)', fontsize=16, fontweight='bold')
    
    years = annual['year'].values
    x = np.arange(len(years))
    width = 0.35
    
    # Annual water inputs
    ax = axes[0, 0]
    ax.bar(x - width/2, annual['precipitation'], width, label='Precipitation', alpha=0.8)
    ax.bar(x + width/2, annual['irrigation'], width, label='Irrigation', alpha=0.8)
    ax.set_ylabel('Water (mm/year)', fontsize=11)
    ax.set_title('Annual Water Inputs', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annual ET
    ax = axes[0, 1]
    ax.bar(years, annual['evapotranspiration'], color='green', alpha=0.7)
    ax.plot(years, annual['pet'], 'ro-', label='PET', linewidth=2, markersize=6)
    ax.set_ylabel('ET (mm/year)', fontsize=11)
    ax.set_title('Annual Evapotranspiration', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annual water losses
    ax = axes[1, 0]
    ax.bar(x - width/2, annual['percolation'], width, label='Percolation', alpha=0.8)
    ax.bar(x + width/2, annual['runoff'], width, label='Runoff', alpha=0.8)
    ax.set_ylabel('Water (mm/year)', fontsize=11)
    ax.set_title('Annual Water Losses', fontsize=12)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Average soil moisture
    ax = axes[1, 1]
    ax.bar(years, annual['soil_moisture'], color='brown', alpha=0.7)
    ax.set_ylabel('Soil Moisture (mm)', fontsize=11)
    ax.set_title('Average Annual Soil Moisture', fontsize=12)
    ax.set_xlabel('Year', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot2_path = output_path / 'annual_summary.png'
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot2_path}")
    plt.close()
    
    # Plot 3: Monthly patterns (averaged across all years)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Average Monthly Water Balance Patterns', fontsize=16, fontweight='bold')
    
    # Calculate monthly averages
    results['month'] = results['date'].dt.month
    monthly_avg = results.groupby('month').mean()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Precipitation and PET
    ax = axes[0, 0]
    ax.bar(months, monthly_avg['precipitation'], alpha=0.6, label='Precipitation')
    ax.plot(months, monthly_avg['pet'], 'r-o', label='PET', linewidth=2)
    ax.set_ylabel('Water (mm/month)', fontsize=11)
    ax.set_title('Climate Forcing', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    # ET and Irrigation
    ax = axes[0, 1]
    ax.plot(months, monthly_avg['evapotranspiration'], 'g-o', 
            label='ET', linewidth=2, markersize=6)
    ax.plot(months, monthly_avg['irrigation'], 'b-s', 
            label='Irrigation', linewidth=2, markersize=6)
    ax.set_ylabel('Water (mm/month)', fontsize=11)
    ax.set_title('Water Use', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    # Soil moisture
    ax = axes[1, 0]
    ax.plot(months, monthly_avg['soil_moisture'], 'brown', 
            marker='o', linewidth=2, markersize=6)
    ax.set_ylabel('Soil Moisture (mm)', fontsize=11)
    ax.set_title('Soil Water Storage', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    # Losses
    ax = axes[1, 1]
    ax.plot(months, monthly_avg['percolation'], 'purple', 
            marker='o', label='Percolation', linewidth=2, markersize=6)
    ax.plot(months, monthly_avg['runoff'], 'red', 
            marker='s', label='Runoff', linewidth=2, markersize=6)
    ax.set_ylabel('Water (mm/month)', fontsize=11)
    ax.set_title('Water Losses', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot3_path = output_path / 'monthly_patterns.png'
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot3_path}")
    plt.close()
    
    print("\nAll plots saved successfully!")


def main():
    """Main execution function."""
    # Configuration
    latitude = 38.0   # 38°N
    longitude = -1.5  # 1.5°W (negative for West)
    start_date = "2015-01-01"
    end_date = "2021-12-31"
    spinup_iterations = 50
    
    print("\n" + "=" * 70)
    print("WATER BALANCE MODEL - REAL CLIMATE DATA EXAMPLE")
    print("=" * 70)
    print("Location: Eastern Spain (38°N, 1.5°W)")
    print("Period: 2015-2021")
    print("Crop: Irrigated Maize")
    print("Spinup: 50 iterations")
    print("=" * 70)
    
    # Step 1: Fetch climate data
    climate_df = fetch_climate_from_openmeteo(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        verbose=True
    )
    
    # Step 2: Prepare climate data for model
    climate = prepare_climate_data(
        climate_df,
        pr_column="precipitation",
        pet_column="et0_fao_evapotranspiration",
        verbose=True
    )
    
    # Validate climate data
    issues = validate_climate_data(climate)
    if issues['errors']:
        print("\n⚠ ERRORS found in climate data:")
        for error in issues['errors']:
            print(f"  - {error}")
    if issues['warnings']:
        print("\n⚠ WARNINGS:")
        for warning in issues['warnings']:
            print(f"  - {warning}")
    
    # Calculate aridity index
    aridity_index = calculate_aridity_index(climate)
    print(f"\nAridity Index (P/PET): {aridity_index:.3f}")
    if aridity_index < 0.20:
        climate_class = "Arid"
    elif aridity_index < 0.50:
        climate_class = "Semi-arid"
    elif aridity_index < 0.65:
        climate_class = "Dry sub-humid"
    else:
        climate_class = "Humid"
    print(f"Climate classification: {climate_class}")
    
    # Step 3: Load configuration and set up parameters
    config = ConfigManager()
    crop = setup_maize_parameters(config, start_date, end_date)
    soil = setup_soil_parameters()
    
    # Step 4: Run simulation
    results, annual = run_simulation(soil, crop, climate, config, spinup_iterations)
    
    # Step 5: Plot results
    plot_results(results, annual)
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated files in 'usage/' directory:")
    print("  - water_balance_2018.png")
    print("  - annual_summary.png")
    print("  - monthly_patterns.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
