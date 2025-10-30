"""
Climate Data Utilities

Functions for fetching and preparing climate data from various sources
for use with the water balance model.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import requests

try:
    from .water_balance_model import ClimateData
except ImportError:
    from water_balance_model import ClimateData


def fetch_climate_from_openmeteo(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch daily climate data from Open-Meteo ERA5 archive.
    
    Uses the Open-Meteo ERA5 historical weather API to fetch daily precipitation
    and FAO-56 reference evapotranspiration. No API key required.
    
    Coverage: Global, 1940-present
    Resolution: ~25km (ERA5 reanalysis)
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        verbose: Print progress information
        
    Returns:
        DataFrame with daily precipitation and ET0 (mm/day)
        
    Raises:
        requests.HTTPError: If API request fails
    """
    if verbose:
        print("=" * 70)
        print("Fetching Daily Climate Data from Open-Meteo ERA5")
        print("=" * 70)
        print(f"Location: {latitude}°N, {abs(longitude)}°{'W' if longitude < 0 else 'E'}")
        print(f"Period: {start_date} to {end_date}")
        print("Source: ERA5 reanalysis (~25km resolution)")
    
    # Build API URL
    url = (
        "https://archive-api.open-meteo.com/v1/era5?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=precipitation_sum,et0_fao_evapotranspiration"
        "&timezone=UTC"
    )
    
    if verbose:
        print("\nFetching data from Open-Meteo API...")
    
    # Make request
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch data from Open-Meteo: {e}")
    
    # Parse JSON response
    data = response.json()
    daily = data["daily"]
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "precipitation": daily["precipitation_sum"],
        "et0_fao_evapotranspiration": daily["et0_fao_evapotranspiration"]
    })
    
    df.set_index("date", inplace=True)
    
    if verbose:
        print(f"Data retrieved: {len(df)} days")
        print("\nFirst few rows:")
        print(df.head())
        print(f"\nPrecipitation range: {df['precipitation'].min():.2f} - {df['precipitation'].max():.2f} mm/day")
        print(f"ET0 range: {df['et0_fao_evapotranspiration'].min():.2f} - {df['et0_fao_evapotranspiration'].max():.2f} mm/day")
    
    return df


def fetch_climate_from_terraclim(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch climate data from TerraClimate using climatePy.
    
    TerraClimate provides global gridded climate data at ~4km resolution.
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        verbose: Print progress information
        
    Returns:
        DataFrame with precipitation and PET data
        
    Raises:
        ImportError: If climatePy is not installed
    """
    try:
        import climatePy
        from shapely.geometry import Point
    except ImportError:
        raise ImportError(
            "climatePy is required for fetching climate data. "
            "Install it with: pip install climatePy"
        )
    
    if verbose:
        print("=" * 70)
        print("Fetching Climate Data from TerraClimate")
        print("=" * 70)
        print(f"Location: {latitude}°N, {abs(longitude)}°{'W' if longitude < 0 else 'E'}")
        print(f"Period: {start_date} to {end_date}")
    
    # Create point geometry for location
    point = Point(longitude, latitude)
    
    # Fetch precipitation and PET data
    # TerraClimate variables: 'ppt' (precipitation), 'pet' (potential evapotranspiration)
    data = climatePy.getTerraClim(
        AOI=point,
        varname=['ppt', 'pet'],
        startDate=start_date,
        endDate=end_date,
        verbose=verbose
    )
    
    # Convert to DataFrame
    df_list = []
    for var_name, xr_data in data.items():
        # Extract values for the point
        df_var = xr_data.to_dataframe().reset_index()
        df_var = df_var.rename(columns={var_name: var_name})
        df_list.append(df_var[['time', var_name]])
    
    # Merge precipitation and PET
    df = df_list[0].merge(df_list[1], on='time')
    df = df.rename(columns={
        'time': 'date',
        'ppt': 'precipitation',
        'pet': 'pet'
    })
    df.set_index('date', inplace=True)
    
    if verbose:
        print(f"\nData retrieved: {len(df)} days")
        print("\nFirst few rows:")
        print(df.head())
    
    return df


def fetch_climate_from_nasa_power(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch climate data from NASA POWER using climatePy.
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        verbose: Print progress information
        
    Returns:
        DataFrame with precipitation and PET data
        
    Raises:
        ImportError: If climatePy is not installed
    """
    try:
        from climatePy import ClimateClient
    except ImportError:
        raise ImportError(
            "climatePy is required for fetching climate data. "
            "Install it with: pip install climatePy"
        )
    
    if verbose:
        print("=" * 70)
        print("Fetching Climate Data from NASA POWER")
        print("=" * 70)
        print(f"Location: {latitude}°N, {abs(longitude)}°{'W' if longitude < 0 else 'E'}")
        print(f"Period: {start_date} to {end_date}")
    
    # Initialize climate client
    cl = ClimateClient(source="nasa_power")
    
    # Fetch data
    data = cl.get_data(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        variables=["precipitation", "et0_fao_evapotranspiration"]
    )
    
    if verbose:
        print(f"\nData retrieved: {len(data)} days")
        print("\nFirst few rows:")
        print(data.head())
    
    return data


def prepare_climate_data(
    climate_df: pd.DataFrame,
    pr_column: str = "precipitation",
    pet_column: str = "et0_fao_evapotranspiration",
    date_column: Optional[str] = None,
    verbose: bool = True
) -> ClimateData:
    """
    Convert climate DataFrame to ClimateData object.
    
    Args:
        climate_df: DataFrame with climate data
        pr_column: Name of precipitation column (mm)
        pet_column: Name of PET column (mm)
        date_column: Name of date column (if None, uses index)
        verbose: Print summary statistics
        
    Returns:
        ClimateData object ready for water balance model
    """
    # Extract dates
    if date_column is not None:
        dates = pd.to_datetime(climate_df[date_column])
    else:
        dates = pd.to_datetime(climate_df.index)
    
    # Extract precipitation and PET
    precipitation = climate_df[pr_column].values
    pet = climate_df[pet_column].values
    
    # Handle missing values
    if np.any(np.isnan(precipitation)):
        n_missing = np.sum(np.isnan(precipitation))
        print(f"Warning: {n_missing} missing precipitation values, filling with 0")
        precipitation = np.nan_to_num(precipitation, nan=0.0)
    
    if np.any(np.isnan(pet)):
        n_missing = np.sum(np.isnan(pet))
        print(f"Warning: {n_missing} missing PET values, filling with mean")
        pet = np.nan_to_num(pet, nan=np.nanmean(pet))
    
    # Create ClimateData object
    climate = ClimateData(
        precipitation=precipitation,
        pet=pet,
        dates=dates
    )
    
    if verbose:
        print("\n" + "=" * 70)
        print("Climate Data Summary")
        print("=" * 70)
        print(f"Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
        print(f"Number of days: {len(dates)}")
        print(f"\nPrecipitation:")
        print(f"  Total: {precipitation.sum():.1f} mm")
        print(f"  Mean daily: {precipitation.mean():.2f} mm/day")
        print(f"  Max daily: {precipitation.max():.2f} mm/day")
        print(f"\nPotential ET:")
        print(f"  Total: {pet.sum():.1f} mm")
        print(f"  Mean daily: {pet.mean():.2f} mm/day")
        print(f"  Max daily: {pet.max():.2f} mm/day")
    
    return climate


def validate_climate_data(climate: ClimateData) -> Dict[str, list]:
    """
    Validate climate data for common issues.
    
    Args:
        climate: ClimateData object
        
    Returns:
        Dictionary with 'errors' and 'warnings' lists
    """
    issues = {'errors': [], 'warnings': []}
    
    # Check for negative values
    if np.any(climate.precipitation < 0):
        issues['errors'].append("Negative precipitation values detected")
    
    if np.any(climate.pet < 0):
        issues['errors'].append("Negative PET values detected")
    
    # Check for unrealistic values
    if np.any(climate.precipitation > 500):
        issues['warnings'].append(
            f"Very high daily precipitation detected (max: {climate.precipitation.max():.1f} mm)"
        )
    
    if np.any(climate.pet > 20):
        issues['warnings'].append(
            f"Very high daily PET detected (max: {climate.pet.max():.1f} mm)"
        )
    
    # Check for missing data (NaN)
    if np.any(np.isnan(climate.precipitation)):
        issues['errors'].append("Missing precipitation values (NaN) detected")
    
    if np.any(np.isnan(climate.pet)):
        issues['errors'].append("Missing PET values (NaN) detected")
    
    # Check date continuity
    date_diff = np.diff(climate.dates.values).astype('timedelta64[D]').astype(int)
    if not np.all(date_diff == 1):
        gaps = np.sum(date_diff != 1)
        issues['warnings'].append(f"Date sequence has {gaps} gaps (non-consecutive days)")
    
    return issues


def calculate_aridity_index(climate: ClimateData) -> float:
    """
    Calculate aridity index (P/PET ratio).
    
    Args:
        climate: ClimateData object
        
    Returns:
        Aridity index (annual precipitation / annual PET)
        
    Notes:
        - AI < 0.05: Hyper-arid
        - 0.05 ≤ AI < 0.20: Arid
        - 0.20 ≤ AI < 0.50: Semi-arid
        - 0.50 ≤ AI < 0.65: Dry sub-humid
        - AI ≥ 0.65: Humid
    """
    total_pr = climate.precipitation.sum()
    total_pet = climate.pet.sum()
    
    if total_pet == 0:
        return np.inf
    
    return total_pr / total_pet


def get_climate_statistics(climate: ClimateData) -> pd.DataFrame:
    """
    Calculate comprehensive climate statistics.
    
    Args:
        climate: ClimateData object
        
    Returns:
        DataFrame with annual climate statistics
    """
    # Create DataFrame
    df = pd.DataFrame({
        'date': climate.dates,
        'precipitation': climate.precipitation,
        'pet': climate.pet
    })
    
    df['year'] = df['date'].dt.year
    
    # Annual statistics
    annual_stats = df.groupby('year').agg({
        'precipitation': ['sum', 'mean', 'max', 'std'],
        'pet': ['sum', 'mean', 'max', 'std']
    }).round(2)
    
    # Flatten column names
    annual_stats.columns = ['_'.join(col).strip() for col in annual_stats.columns.values]
    
    # Add aridity index
    annual_stats['aridity_index'] = (
        annual_stats['precipitation_sum'] / annual_stats['pet_sum']
    ).round(3)
    
    return annual_stats.reset_index()


def resample_climate_data(
    climate: ClimateData,
    freq: str = 'M'
) -> pd.DataFrame:
    """
    Resample climate data to different temporal frequency.
    
    Args:
        climate: ClimateData object
        freq: Pandas frequency string ('M' for monthly, 'Y' for yearly, etc.)
        
    Returns:
        Resampled DataFrame
    """
    df = pd.DataFrame({
        'date': climate.dates,
        'precipitation': climate.precipitation,
        'pet': climate.pet
    })
    
    df.set_index('date', inplace=True)
    
    # Resample (sum for precipitation and PET)
    resampled = df.resample(freq).sum()
    
    return resampled.reset_index()
