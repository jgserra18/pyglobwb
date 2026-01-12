"""
Crop Utilities for Water Balance Model

Utilities for creating daily crop calendars from monthly or stage-based data.
Handles interpolation of crop coefficients (Kc) and rooting depths.

Key Functions:
- interpolate_monthly_to_daily: Convert monthly Kc values to daily
- interpolate_rooting_depth: Interpolate rooting depth based on crop growth
- create_crop_parameters: Create CropParameters from monthly/stage data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class CropCalendarConfig:
    """
    Configuration for crop calendar generation.
    
    Attributes:
        name: Crop name
        monthly_kc: Monthly Kc values (12 values, Jan-Dec)
        max_rooting_depth: Maximum rooting depth (m)
        is_permanent: Whether crop is permanent (fixed rooting depth) or annual
        initial_rooting_depth: Initial rooting depth for annual crops (m)
        landuse_kc: Land use Kc for rainfed conditions (default 0.5)
    """
    name: str
    monthly_kc: List[float]
    max_rooting_depth: float
    is_permanent: bool = False
    initial_rooting_depth: float = 0.2
    landuse_kc: float = 0.5
    
    def __post_init__(self):
        if len(self.monthly_kc) != 12:
            raise ValueError("monthly_kc must have 12 values (one per month)")


def interpolate_monthly_to_daily(
    monthly_values: List[float],
    dates: pd.DatetimeIndex,
    method: str = 'step'
) -> np.ndarray:
    """
    Interpolate monthly values to daily values.
    
    Args:
        monthly_values: List of 12 monthly values (Jan-Dec)
        dates: DatetimeIndex for the output daily values
        method: Interpolation method ('step' or 'linear')
            - 'step': Each day gets the value of its month (no smoothing)
            - 'linear': Linear interpolation between month midpoints
    
    Returns:
        Array of daily values matching the dates
    """
    if len(monthly_values) != 12:
        raise ValueError("monthly_values must have 12 values (one per month)")
    
    daily_values = np.zeros(len(dates))
    
    if method == 'step':
        # Simple step function: each day gets its month's value
        for i, date in enumerate(dates):
            month_idx = date.month - 1  # 0-indexed
            daily_values[i] = monthly_values[month_idx]
    
    elif method == 'linear':
        # Linear interpolation between month midpoints
        # Create midpoint values for each month across years
        for i, date in enumerate(dates):
            month = date.month
            year = date.year
            day = date.day
            
            # Days in current month
            if month == 12:
                days_in_month = 31
            else:
                next_month = pd.Timestamp(year=year, month=month+1, day=1)
                days_in_month = (next_month - pd.Timestamp(year=year, month=month, day=1)).days
            
            midpoint = days_in_month / 2
            
            # Current and adjacent month values
            curr_val = monthly_values[month - 1]
            
            if day <= midpoint:
                # Interpolate with previous month
                prev_month = month - 1 if month > 1 else 12
                prev_val = monthly_values[prev_month - 1]
                
                # Previous month days
                if prev_month == 12:
                    prev_days = 31
                else:
                    prev_days = (pd.Timestamp(year=year if month > 1 else year-1, month=prev_month+1 if prev_month < 12 else 1, day=1) - 
                                pd.Timestamp(year=year if month > 1 else year-1, month=prev_month, day=1)).days
                
                # Distance from previous midpoint to current day
                dist = (prev_days - prev_days/2) + day
                total_dist = (prev_days - prev_days/2) + midpoint
                
                weight = dist / total_dist
                daily_values[i] = prev_val + weight * (curr_val - prev_val)
            else:
                # Interpolate with next month
                next_month = month + 1 if month < 12 else 1
                next_val = monthly_values[next_month - 1]
                
                # Distance from current midpoint to current day
                dist = day - midpoint
                
                # Next month midpoint
                if next_month == 12:
                    next_days = 31
                else:
                    next_year = year if month < 12 else year + 1
                    next_days = (pd.Timestamp(year=next_year, month=next_month+1 if next_month < 12 else 1, day=1) - 
                                pd.Timestamp(year=next_year, month=next_month, day=1)).days
                
                total_dist = (days_in_month - midpoint) + next_days/2
                
                weight = dist / total_dist
                daily_values[i] = curr_val + weight * (next_val - curr_val)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'step' or 'linear'")
    
    return daily_values


def interpolate_rooting_depth(
    kc_daily: np.ndarray,
    dates: pd.DatetimeIndex,
    max_rooting_depth: float,
    initial_rooting_depth: float = 0.2,
    is_permanent: bool = False
) -> np.ndarray:
    """
    Interpolate rooting depth based on crop growth cycle.
    
    For permanent crops: Fixed rooting depth throughout the year.
    For annual crops: Rooting depth increases from initial to max during growth,
                     then stays at max until harvest.
    
    Args:
        kc_daily: Daily Kc values (used to determine growth stages)
        dates: DatetimeIndex for the output
        max_rooting_depth: Maximum rooting depth (m)
        initial_rooting_depth: Initial rooting depth at planting (m)
        is_permanent: If True, use fixed max_rooting_depth
    
    Returns:
        Array of daily rooting depths (m)
    """
    rooting_depth = np.zeros(len(dates))
    
    if is_permanent:
        # Permanent crops have fixed rooting depth
        rooting_depth[:] = max_rooting_depth
        return rooting_depth
    
    # For annual crops, interpolate based on growth cycle
    # Process year by year
    years = dates.year.unique()
    
    for year in years:
        year_mask = dates.year == year
        year_indices = np.where(year_mask)[0]
        
        if len(year_indices) == 0:
            continue
        
        year_kc = kc_daily[year_indices]
        
        # Find growing season (where Kc > 0)
        growing_mask = year_kc > 0
        
        if not growing_mask.any():
            # No growing season this year, set to initial depth
            rooting_depth[year_indices] = initial_rooting_depth
            continue
        
        # Find the day when Kc reaches maximum (end of root growth)
        max_kc_idx = np.argmax(year_kc)
        
        # Find first day of growing season
        growing_indices = np.where(growing_mask)[0]
        first_growing_day = growing_indices[0]
        
        # Before growing season: initial rooting depth
        rooting_depth[year_indices[:first_growing_day]] = initial_rooting_depth
        
        # During root growth phase (from planting to max Kc)
        if max_kc_idx > first_growing_day:
            growth_days = max_kc_idx - first_growing_day + 1
            growth_indices = year_indices[first_growing_day:max_kc_idx + 1]
            rooting_depth[growth_indices] = np.linspace(
                initial_rooting_depth, 
                max_rooting_depth, 
                growth_days
            )
        else:
            rooting_depth[year_indices[first_growing_day]] = max_rooting_depth
        
        # After max Kc: maintain max rooting depth until end of growing season
        if max_kc_idx + 1 < len(year_indices):
            rooting_depth[year_indices[max_kc_idx + 1:]] = max_rooting_depth
        
        # After growing season ends: back to initial (or keep max, depending on interpretation)
        # Here we keep max until end of year for simplicity
        last_growing_day = growing_indices[-1]
        if last_growing_day + 1 < len(year_indices):
            # After harvest, could reset to initial or keep max
            # Keeping initial for fallow period
            rooting_depth[year_indices[last_growing_day + 1:]] = initial_rooting_depth
    
    return rooting_depth


def create_daily_crop_calendar(
    config: CropCalendarConfig,
    start_date: str,
    end_date: str,
    kc_interpolation: str = 'step'
) -> pd.DataFrame:
    """
    Create a daily crop calendar with Kc and rooting depth.
    
    Args:
        config: CropCalendarConfig with crop parameters
        start_date: Start date string (e.g., '2020-01-01')
        end_date: End date string (e.g., '2020-12-31')
        kc_interpolation: Method for Kc interpolation ('step' or 'linear')
    
    Returns:
        DataFrame with columns: date, kc, rooting_depth
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Interpolate Kc
    kc_daily = interpolate_monthly_to_daily(
        config.monthly_kc, 
        dates, 
        method=kc_interpolation
    )
    
    # Interpolate rooting depth
    rooting_depth = interpolate_rooting_depth(
        kc_daily,
        dates,
        config.max_rooting_depth,
        config.initial_rooting_depth,
        config.is_permanent
    )
    
    return pd.DataFrame({
        'date': dates,
        'kc': kc_daily,
        'rooting_depth': rooting_depth
    })


def create_crop_parameters(
    config: CropCalendarConfig,
    start_date: str,
    end_date: str,
    kc_interpolation: str = 'step'
):
    """
    Create CropParameters object from configuration.
    
    Args:
        config: CropCalendarConfig with crop parameters
        start_date: Start date string
        end_date: End date string
        kc_interpolation: Method for Kc interpolation
    
    Returns:
        CropParameters object ready for WaterBalanceModel
    """
    from water_balance_model import CropParameters
    
    calendar = create_daily_crop_calendar(
        config, start_date, end_date, kc_interpolation
    )
    
    return CropParameters(
        name=config.name,
        kc_values=calendar['kc'].values,
        rooting_depth=calendar['rooting_depth'].values,
        dates=pd.DatetimeIndex(calendar['date']),
        landuse_kc=config.landuse_kc
    )


# Predefined crop configurations (examples)
CROP_CONFIGS = {
    'maize': CropCalendarConfig(
        name='Maize',
        monthly_kc=[0, 0, 0, 0.3, 0.64, 1.17, 1.2, 0.70, 0, 0, 0, 0],
        max_rooting_depth=1.2,
        is_permanent=False,
        initial_rooting_depth=0.2
    ),
    'wheat': CropCalendarConfig(
        name='Wheat',
        monthly_kc=[0.4, 0.7, 1.1, 1.15, 0.7, 0.3, 0, 0, 0, 0.3, 0.4, 0.4],
        max_rooting_depth=1.5,
        is_permanent=False,
        initial_rooting_depth=0.2
    ),
    'olive': CropCalendarConfig(
        name='Olive',
        monthly_kc=[0.65, 0.65, 0.65, 0.60, 0.55, 0.50, 0.50, 0.50, 0.55, 0.60, 0.65, 0.65],
        max_rooting_depth=1.7,
        is_permanent=True,
        initial_rooting_depth=1.7
    ),
    'vineyard': CropCalendarConfig(
        name='Vineyard',
        monthly_kc=[0, 0, 0.3, 0.5, 0.7, 0.85, 0.85, 0.7, 0.5, 0.3, 0, 0],
        max_rooting_depth=1.5,
        is_permanent=True,
        initial_rooting_depth=1.5
    )
}


def get_crop_config(crop_name: str) -> CropCalendarConfig:
    """
    Get predefined crop configuration by name.
    
    Args:
        crop_name: Crop name (case-insensitive)
    
    Returns:
        CropCalendarConfig for the crop
    
    Raises:
        KeyError: If crop not found
    """
    key = crop_name.lower()
    if key not in CROP_CONFIGS:
        available = ', '.join(CROP_CONFIGS.keys())
        raise KeyError(f"Crop '{crop_name}' not found. Available: {available}")
    return CROP_CONFIGS[key]


def load_crop_config_from_csv(
    kc_csv_path: str,
    rooting_depth_csv_path: str,
    crop_name: str,
    management: str = 'Irrigated'
) -> CropCalendarConfig:
    """
    Load crop configuration from CSV files.
    
    Expected CSV formats:
    - kc_csv: columns [crop, month, Kc] or [crop, Jan, Feb, ..., Dec]
    - rooting_depth_csv: columns [Crop, Irrigated, Rainfed]
    
    Args:
        kc_csv_path: Path to monthly Kc CSV file
        rooting_depth_csv_path: Path to rooting depth CSV file
        crop_name: Name of the crop to load
        management: 'Irrigated' or 'Rainfed' for rooting depth
    
    Returns:
        CropCalendarConfig for the crop
    """
    # Load Kc data
    kc_df = pd.read_csv(kc_csv_path)
    crop_kc = kc_df[kc_df['crop'].str.lower() == crop_name.lower()]
    
    if len(crop_kc) == 0:
        raise ValueError(f"Crop '{crop_name}' not found in {kc_csv_path}")
    
    # Extract monthly Kc values
    if 'month' in crop_kc.columns:
        # Long format: crop, month, Kc
        monthly_kc = crop_kc.sort_values('month')['Kc'].tolist()
    else:
        # Wide format: crop, Jan, Feb, ..., Dec
        month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_kc = crop_kc[month_cols].values.flatten().tolist()
    
    # Load rooting depth
    rd_df = pd.read_csv(rooting_depth_csv_path)
    crop_rd = rd_df[rd_df['Crop'].str.lower() == crop_name.lower()]
    
    if len(crop_rd) == 0:
        raise ValueError(f"Crop '{crop_name}' not found in {rooting_depth_csv_path}")
    
    max_rooting_depth = float(crop_rd[management].values[0])
    
    # Determine if permanent crop (all Kc values > 0)
    is_permanent = all(kc > 0 for kc in monthly_kc)
    
    return CropCalendarConfig(
        name=crop_name,
        monthly_kc=monthly_kc,
        max_rooting_depth=max_rooting_depth,
        is_permanent=is_permanent,
        initial_rooting_depth=0.2 if not is_permanent else max_rooting_depth
    )
