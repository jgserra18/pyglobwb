"""
Utility functions for loading and processing data for the water balance model.

This module provides helper functions to:
- Load climate data from various formats
- Process crop parameters
- Handle spatial data (rasters, shapefiles)
- Export model results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings


def load_climate_csv(
    filepath: str,
    date_column: str = 'date',
    pr_column: str = 'precipitation',
    pet_column: str = 'pet',
    date_format: Optional[str] = None
) -> pd.DataFrame:
    """
    Load climate data from CSV file.
    
    Args:
        filepath: Path to CSV file
        date_column: Name of date column
        pr_column: Name of precipitation column
        pet_column: Name of PET column
        date_format: Date format string (e.g., '%Y-%m-%d')
    
    Returns:
        DataFrame with standardized column names
    """
    df = pd.read_csv(filepath)
    
    # Parse dates
    if date_format:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    else:
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Standardize column names
    df = df.rename(columns={
        date_column: 'date',
        pr_column: 'precipitation',
        pet_column: 'pet'
    })
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Check for missing values
    if df[['precipitation', 'pet']].isnull().any().any():
        warnings.warn("Climate data contains missing values")
    
    return df[['date', 'precipitation', 'pet']]


def load_climate_netcdf(
    filepath: str,
    lat: float,
    lon: float,
    pr_var: str = 'pr',
    pet_var: str = 'pet',
    time_var: str = 'time'
) -> pd.DataFrame:
    """
    Load climate data from NetCDF file for a specific location.
    
    Requires: xarray, netCDF4
    
    Args:
        filepath: Path to NetCDF file
        lat: Latitude of location
        lon: Longitude of location
        pr_var: Name of precipitation variable
        pet_var: Name of PET variable
        time_var: Name of time variable
    
    Returns:
        DataFrame with climate data
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray is required for NetCDF support. Install with: pip install xarray netCDF4")
    
    # Open dataset
    ds = xr.open_dataset(filepath)
    
    # Extract point data
    point = ds.sel(lat=lat, lon=lon, method='nearest')
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': pd.to_datetime(point[time_var].values),
        'precipitation': point[pr_var].values,
        'pet': point[pet_var].values
    })
    
    return df


def generate_synthetic_climate(
    start_date: str,
    end_date: str,
    mean_annual_pr: float = 800.0,
    mean_annual_pet: float = 1200.0,
    pr_seasonality: float = 0.5,
    pet_seasonality: float = 0.6,
    pr_variability: float = 0.8,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic climate data with seasonal patterns.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        mean_annual_pr: Mean annual precipitation (mm)
        mean_annual_pet: Mean annual PET (mm)
        pr_seasonality: Precipitation seasonality (0-1)
        pet_seasonality: PET seasonality (0-1)
        pr_variability: Precipitation day-to-day variability
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic climate data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    day_of_year = dates.dayofyear
    
    # Daily mean precipitation
    pr_daily_mean = mean_annual_pr / 365.25
    
    # Seasonal pattern (peak in winter, low in summer for Mediterranean)
    pr_seasonal = pr_daily_mean * (1 + pr_seasonality * np.sin(2 * np.pi * (day_of_year - 80) / 365))
    
    # Add variability using gamma distribution
    pr = pr_seasonal * np.random.gamma(1/pr_variability, pr_variability, n_days)
    pr = np.maximum(0, pr)
    
    # PET seasonal pattern (low in winter, high in summer)
    pet_daily_mean = mean_annual_pet / 365.25
    pet = pet_daily_mean * (1 + pet_seasonality * np.sin(2 * np.pi * (day_of_year - 80) / 365))
    pet = np.maximum(0.5, pet)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'precipitation': pr,
        'pet': pet
    })
    
    return df


def aggregate_to_monthly(
    daily_data: pd.DataFrame,
    date_column: str = 'date',
    agg_funcs: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Aggregate daily data to monthly.
    
    Args:
        daily_data: DataFrame with daily data
        date_column: Name of date column
        agg_funcs: Dictionary of {column: aggregation_function}
                  Default: sum for fluxes, mean for states
    
    Returns:
        Monthly aggregated DataFrame
    """
    df = daily_data.copy()
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    
    if agg_funcs is None:
        # Default aggregation
        agg_funcs = {
            col: 'sum' if col not in ['soil_moisture', 'smax', 'seav', 'kc', 'rooting_depth']
            else 'mean'
            for col in df.columns
            if col not in ['date', 'year', 'month']
        }
    
    monthly = df.groupby(['year', 'month']).agg(agg_funcs).reset_index()
    
    return monthly


def aggregate_to_annual(
    daily_data: pd.DataFrame,
    date_column: str = 'date',
    water_year: bool = False,
    water_year_start_month: int = 10,
    agg_funcs: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Aggregate daily data to annual.
    
    Args:
        daily_data: DataFrame with daily data
        date_column: Name of date column
        water_year: Use water year instead of calendar year
        water_year_start_month: Starting month of water year (1-12)
        agg_funcs: Dictionary of {column: aggregation_function}
    
    Returns:
        Annual aggregated DataFrame
    """
    df = daily_data.copy()
    
    if water_year:
        # Calculate water year
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['year'] = df.apply(
            lambda row: row['year'] + 1 if row['month'] >= water_year_start_month else row['year'],
            axis=1
        )
    else:
        df['year'] = df[date_column].dt.year
    
    if agg_funcs is None:
        # Default aggregation
        agg_funcs = {
            col: 'sum' if col not in ['soil_moisture', 'smax', 'seav', 'kc', 'rooting_depth']
            else 'mean'
            for col in df.columns
            if col not in ['date', 'year', 'month']
        }
    
    annual = df.groupby('year').agg(agg_funcs).reset_index()
    
    return annual


def calculate_water_balance_closure(
    results: pd.DataFrame,
    tolerance: float = 0.1
) -> pd.DataFrame:
    """
    Calculate water balance closure error.
    
    Water balance: ΔS = P + I - ET - R0 - Perc
    
    Args:
        results: Model output DataFrame
        tolerance: Acceptable error (mm)
    
    Returns:
        DataFrame with closure statistics
    """
    # Calculate change in soil moisture
    delta_sm = results['soil_moisture'].diff()
    
    # Calculate water balance
    wb_calculated = (
        results['precipitation'] +
        results['irrigation'] -
        results['evapotranspiration'] -
        results['runoff'] -
        results['percolation']
    )
    
    # Closure error
    error = delta_sm - wb_calculated
    
    # Statistics
    stats = pd.DataFrame({
        'metric': ['mean_error', 'max_error', 'rmse', 'pct_within_tolerance'],
        'value': [
            error.mean(),
            error.abs().max(),
            np.sqrt((error**2).mean()),
            (error.abs() <= tolerance).mean() * 100
        ]
    })
    
    return stats


def export_results_csv(
    results: pd.DataFrame,
    output_path: str,
    include_metadata: bool = True
) -> None:
    """
    Export model results to CSV file.
    
    Args:
        results: Model output DataFrame
        output_path: Path for output CSV file
        include_metadata: Include metadata header
    """
    if include_metadata:
        # Write metadata as comments
        with open(output_path, 'w') as f:
            f.write(f"# Water Balance Model Results\n")
            f.write(f"# Generated: {pd.Timestamp.now()}\n")
            f.write(f"# Period: {results['date'].min()} to {results['date'].max()}\n")
            f.write(f"# Number of days: {len(results)}\n")
            f.write("#\n")
        
        # Append data
        results.to_csv(output_path, mode='a', index=False)
    else:
        results.to_csv(output_path, index=False)
    
    print(f"Results exported to: {output_path}")


def export_summary_excel(
    results: pd.DataFrame,
    output_path: str,
    model_config: Optional[Dict] = None
) -> None:
    """
    Export model results and summaries to Excel file with multiple sheets.
    
    Requires: openpyxl
    
    Args:
        results: Model output DataFrame
        output_path: Path for output Excel file
        model_config: Optional model configuration dictionary
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required for Excel export. Install with: pip install openpyxl")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Daily results
        results.to_excel(writer, sheet_name='Daily', index=False)
        
        # Monthly summary
        monthly = aggregate_to_monthly(results)
        monthly.to_excel(writer, sheet_name='Monthly', index=False)
        
        # Annual summary
        annual = aggregate_to_annual(results)
        annual.to_excel(writer, sheet_name='Annual', index=False)
        
        # Statistics
        stats = results[['precipitation', 'pet', 'evapotranspiration', 
                        'irrigation', 'percolation', 'runoff', 'soil_moisture']].describe()
        stats.to_excel(writer, sheet_name='Statistics')
        
        # Water balance closure
        closure = calculate_water_balance_closure(results)
        closure.to_excel(writer, sheet_name='Closure', index=False)
        
        # Configuration (if provided)
        if model_config:
            config_df = pd.DataFrame([
                {'Parameter': k, 'Value': str(v)}
                for k, v in model_config.items()
            ])
            config_df.to_excel(writer, sheet_name='Configuration', index=False)
    
    print(f"Summary exported to: {output_path}")


def create_comparison_table(
    results_dict: Dict[str, pd.DataFrame],
    metric: str = 'annual_sum'
) -> pd.DataFrame:
    """
    Create comparison table from multiple model runs.
    
    Args:
        results_dict: Dictionary of {scenario_name: results_dataframe}
        metric: 'annual_sum', 'annual_mean', or 'total'
    
    Returns:
        Comparison DataFrame
    """
    comparison = {}
    
    for scenario, results in results_dict.items():
        if metric == 'annual_sum':
            annual = aggregate_to_annual(results)
            comparison[scenario] = annual.mean()
        elif metric == 'annual_mean':
            annual = aggregate_to_annual(results)
            comparison[scenario] = annual.mean()
        elif metric == 'total':
            comparison[scenario] = results.sum()
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    df = pd.DataFrame(comparison).T
    
    # Select relevant columns
    cols = ['precipitation', 'pet', 'evapotranspiration', 'irrigation', 
            'percolation', 'runoff', 'soil_moisture']
    df = df[[c for c in cols if c in df.columns]]
    
    return df


def validate_input_data(
    climate_data: pd.DataFrame,
    check_negative: bool = True,
    check_extreme: bool = True,
    pr_max: float = 300.0,
    pet_max: float = 20.0
) -> Dict[str, List[str]]:
    """
    Validate input climate data.
    
    Args:
        climate_data: DataFrame with climate data
        check_negative: Check for negative values
        check_extreme: Check for extreme values
        pr_max: Maximum reasonable daily precipitation (mm)
        pet_max: Maximum reasonable daily PET (mm)
    
    Returns:
        Dictionary of validation issues
    """
    issues = {
        'errors': [],
        'warnings': []
    }
    
    # Check for required columns
    required_cols = ['date', 'precipitation', 'pet']
    missing_cols = [col for col in required_cols if col not in climate_data.columns]
    if missing_cols:
        issues['errors'].append(f"Missing required columns: {missing_cols}")
        return issues
    
    # Check for missing values
    null_counts = climate_data[['precipitation', 'pet']].isnull().sum()
    if null_counts.any():
        issues['warnings'].append(f"Missing values found: {null_counts.to_dict()}")
    
    # Check for negative values
    if check_negative:
        if (climate_data['precipitation'] < 0).any():
            n_neg = (climate_data['precipitation'] < 0).sum()
            issues['errors'].append(f"Negative precipitation values found: {n_neg} days")
        
        if (climate_data['pet'] < 0).any():
            n_neg = (climate_data['pet'] < 0).sum()
            issues['errors'].append(f"Negative PET values found: {n_neg} days")
    
    # Check for extreme values
    if check_extreme:
        if (climate_data['precipitation'] > pr_max).any():
            n_extreme = (climate_data['precipitation'] > pr_max).sum()
            max_val = climate_data['precipitation'].max()
            issues['warnings'].append(
                f"Extreme precipitation values: {n_extreme} days > {pr_max} mm (max: {max_val:.1f} mm)"
            )
        
        if (climate_data['pet'] > pet_max).any():
            n_extreme = (climate_data['pet'] > pet_max).sum()
            max_val = climate_data['pet'].max()
            issues['warnings'].append(
                f"Extreme PET values: {n_extreme} days > {pet_max} mm (max: {max_val:.1f} mm)"
            )
    
    # Check date continuity
    date_diff = climate_data['date'].diff()
    if (date_diff[1:] != pd.Timedelta(days=1)).any():
        issues['warnings'].append("Non-continuous dates detected")
    
    return issues


if __name__ == '__main__':
    # Example usage
    print("Data Utilities Module")
    print("=" * 60)
    
    # Generate synthetic data
    print("\nGenerating synthetic climate data...")
    climate = generate_synthetic_climate(
        start_date='2015-01-01',
        end_date='2019-12-31',
        mean_annual_pr=600,
        mean_annual_pet=1200,
        seed=42
    )
    
    print(f"Generated {len(climate)} days of data")
    print(f"Mean annual precipitation: {climate['precipitation'].sum() / 5:.1f} mm")
    print(f"Mean annual PET: {climate['pet'].sum() / 5:.1f} mm")
    
    # Validate data
    print("\nValidating climate data...")
    issues = validate_input_data(climate)
    
    if issues['errors']:
        print("Errors found:")
        for error in issues['errors']:
            print(f"  - {error}")
    else:
        print("No errors found")
    
    if issues['warnings']:
        print("Warnings:")
        for warning in issues['warnings']:
            print(f"  - {warning}")
    else:
        print("No warnings")
    
    # Export example
    print("\nExporting to CSV...")
    export_results_csv(climate, 'example_climate.csv')
    
    print("\nDone!")
