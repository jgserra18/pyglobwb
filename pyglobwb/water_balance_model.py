"""
Vertical Water Balance Model (GLOBWAT-based)

A generalized implementation of the vertical water balance model for crop water use estimation.
This model simulates daily soil moisture dynamics, evapotranspiration, irrigation requirements,
percolation, and runoff.

Key Components:
- Soil moisture tracking with dynamic rooting depth
- Evapotranspiration calculation based on crop coefficients
- Irrigation requirement estimation
- Deep percolation (aquifer recharge)
- Surface runoff

References:
- GLOBWAT model framework
- FAO-56 methodology for crop water requirements
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class SoilParameters:
    """
    Soil hydraulic parameters for water balance calculations.
    
    Attributes:
        smax_base: Base maximum soil moisture storage (mm) at reference depth
        reference_depth: Reference rooting depth (m) for smax_base, default 0.6m
        rmax: Maximum percolation rate (mm/year) - will be converted to daily rate internally
        initial_sm: Initial soil moisture (mm)
        calibration_factor: Regional calibration factor for percolation (default 2.4 for Spain)
    """
    smax_base: float
    reference_depth: float = 0.6
    rmax: float = 10.0
    initial_sm: Optional[float] = None
    calibration_factor: float = 2.4
    
    def __post_init__(self):
        if self.initial_sm is None:
            self.initial_sm = self.smax_base * 0.5


@dataclass
class CropParameters:
    """
    Crop-specific parameters for water balance calculations.
    
    Attributes:
        name: Crop name
        kc_values: Daily crop coefficients (Kc) array
        rooting_depth: Daily rooting depth (m) array
        dates: Corresponding dates for kc_values and rooting_depth
        landuse_kc: Land use Kc for rainfed conditions (default 0.5)
    """
    name: str
    kc_values: np.ndarray
    rooting_depth: np.ndarray
    dates: pd.DatetimeIndex
    landuse_kc: float = 0.5
    
    def __post_init__(self):
        if len(self.kc_values) != len(self.rooting_depth) != len(self.dates):
            raise ValueError("kc_values, rooting_depth, and dates must have the same length")


@dataclass
class ClimateData:
    """
    Climate forcing data for water balance model.
    
    Attributes:
        precipitation: Daily precipitation (mm)
        pet: Daily potential evapotranspiration (mm)
        dates: Corresponding dates
    """
    precipitation: np.ndarray
    pet: np.ndarray
    dates: pd.DatetimeIndex
    
    def __post_init__(self):
        if len(self.precipitation) != len(self.pet) != len(self.dates):
            raise ValueError("precipitation, pet, and dates must have the same length")


class WaterBalanceModel:
    """
    Vertical water balance model for crop water use estimation.
    
    This model simulates daily soil moisture dynamics including:
    - Soil moisture storage
    - Actual evapotranspiration
    - Irrigation requirements
    - Deep percolation
    - Surface runoff
    """
    
    def __init__(
        self,
        soil_params: SoilParameters,
        crop_params: CropParameters,
        climate_data: ClimateData,
        management: str = "rainfed",
        irrigation_efficiency: float = 1.0
    ):
        """
        Initialize the water balance model.
        
        Args:
            soil_params: Soil hydraulic parameters
            crop_params: Crop-specific parameters
            climate_data: Climate forcing data
            management: Management type ("rainfed" or "irrigated")
            irrigation_efficiency: Irrigation system efficiency (0-1)
        """
        self.soil = soil_params
        self.crop = crop_params
        self.climate = climate_data
        self.management = management.lower()
        self.irrigation_efficiency = irrigation_efficiency
        
        # Validate inputs
        self._validate_inputs()
        
        # Initialize storage arrays
        self.n_days = len(climate_data.dates)
        self._initialize_storage()
        
    def _validate_inputs(self):
        """Validate model inputs."""
        if self.management not in ["rainfed", "irrigated"]:
            raise ValueError("management must be 'rainfed' or 'irrigated'")
        
        if not 0 < self.irrigation_efficiency <= 1:
            raise ValueError("irrigation_efficiency must be between 0 and 1")
        
        # Check date alignment
        if not self.climate.dates.equals(self.crop.dates):
            warnings.warn("Climate and crop dates do not match. Ensure proper alignment.")
    
    def _initialize_storage(self):
        """Initialize arrays to store model outputs."""
        self.results = {
            'soil_moisture': np.zeros(self.n_days),
            'evapotranspiration': np.zeros(self.n_days),
            'irrigation': np.zeros(self.n_days),
            'percolation': np.zeros(self.n_days),
            'runoff': np.zeros(self.n_days),
            'smax': np.zeros(self.n_days),
            'seav': np.zeros(self.n_days)
        }
    
    def _calculate_dynamic_smax_seav(self, day_idx: int) -> Tuple[float, float]:
        """
        Calculate dynamic maximum soil moisture and easily available water.
        
        Smax varies with crop rooting depth. Seav is typically 50% of Smax.
        
        Args:
            day_idx: Day index
            
        Returns:
            Tuple of (smax, seav) in mm
        """
        effective_root_depth = self.crop.rooting_depth[day_idx]
        smax = (self.soil.smax_base / self.soil.reference_depth) * effective_root_depth
        seav = smax * 0.5
        
        return smax, seav
    
    def _update_et(
        self,
        sm_t1: float,
        pet: float,
        smax: float,
        seav: float
    ) -> float:
        """
        Update evapotranspiration based on soil moisture availability.
        
        When SM < Seav, ET is reduced proportionally to soil moisture.
        
        Args:
            sm_t1: Soil moisture at previous time step (mm)
            pet: Potential evapotranspiration (mm)
            smax: Maximum soil moisture (mm)
            seav: Easily available soil moisture (mm)
            
        Returns:
            Actual evapotranspiration (mm)
        """
        if sm_t1 < seav:
            et = pet * (sm_t1 / seav)
        else:
            et = pet
        
        return max(0, et)
    
    def _update_percolation(
        self,
        sm_t1: float,
        smax: float,
        seav: float
    ) -> float:
        """
        Update deep percolation (aquifer recharge).
        
        Percolation occurs when SM > Seav, proportional to excess moisture.
        Note: Rmax is in mm/year, so divide by 365 for daily rate.
        
        Args:
            sm_t1: Soil moisture at previous time step (mm)
            smax: Maximum soil moisture (mm)
            seav: Easily available soil moisture (mm)
            
        Returns:
            Deep percolation (mm/day)
        """
        if sm_t1 < seav:
            perc = 0
        else:
            # Rmax is annual rate (mm/year), convert to daily (mm/day)
            rmax_daily = self.soil.rmax / 365.0
            perc = (rmax_daily * self.soil.calibration_factor * 
                   (sm_t1 - seav) / (smax - seav))
        
        return max(0, perc)
    
    def _update_runoff(
        self,
        sm_t1: float,
        pr: float,
        et: float,
        smax: float,
        irrig: float = 0
    ) -> float:
        """
        Update surface runoff.
        
        Runoff occurs when water balance exceeds maximum soil moisture.
        
        Args:
            sm_t1: Soil moisture at previous time step (mm)
            pr: Precipitation (mm)
            et: Evapotranspiration (mm)
            smax: Maximum soil moisture (mm)
            irrig: Irrigation (mm)
            
        Returns:
            Surface runoff (mm)
        """
        wb = sm_t1 + pr + irrig - et
        
        if wb < smax:
            runoff = 0
        else:
            runoff = wb - smax
        
        return max(0, runoff)
    
    def _update_water_balance(
        self,
        sm_t1: float,
        pr: float,
        et: float,
        runoff: float,
        irrig: float = 0
    ) -> float:
        """
        Update water balance.
        
        Args:
            sm_t1: Soil moisture at previous time step (mm)
            pr: Precipitation (mm)
            et: Evapotranspiration (mm)
            runoff: Surface runoff (mm)
            irrig: Irrigation (mm)
            
        Returns:
            Water balance (mm)
        """
        wb = sm_t1 + pr + irrig - et - runoff
        return wb
    
    def _update_soil_moisture(
        self,
        wb: float,
        smax: float
    ) -> float:
        """
        Update soil moisture from water balance.
        
        Args:
            wb: Water balance (mm)
            smax: Maximum soil moisture (mm)
            
        Returns:
            Updated soil moisture (mm)
        """
        sm = min(wb, smax)
        sm = max(0, sm)
        
        return sm
    
    def spinup(self, n_iterations: int = 50) -> float:
        """
        Spin up soil moisture by repeating the first year.
        
        This ensures the model starts from an equilibrium state.
        
        Args:
            n_iterations: Number of iterations to spin up
            
        Returns:
            Equilibrium soil moisture (mm)
        """
        # Get first year indices
        first_year = self.climate.dates.year[0]
        first_year_mask = self.climate.dates.year == first_year
        first_year_indices = np.where(first_year_mask)[0]
        
        sm = self.soil.initial_sm
        
        for _ in range(n_iterations):
            for idx in first_year_indices:
                # Get dynamic soil parameters
                smax, seav = self._calculate_dynamic_smax_seav(idx)
                
                # Calculate rainfed ET
                et_rain = self.climate.pet[idx] * self.crop.landuse_kc
                et_rain = self._update_et(sm, et_rain, smax, seav)
                
                # Calculate percolation
                perc = self._update_percolation(sm, smax, seav)
                
                # Calculate runoff
                runoff = self._update_runoff(
                    sm, self.climate.precipitation[idx], et_rain, smax, irrig=0
                )
                
                # Update water balance
                wb = self._update_water_balance(
                    sm, self.climate.precipitation[idx], et_rain, runoff, irrig=0
                )
                
                # Update soil moisture
                sm = self._update_soil_moisture(wb, smax)
        
        return sm
    
    def run(self, spinup_iterations: int = 50) -> pd.DataFrame:
        """
        Run the water balance model.
        
        Args:
            spinup_iterations: Number of iterations for model spinup
            
        Returns:
            DataFrame with daily water balance components
        """
        # Spin up soil moisture
        sm = self.spinup(n_iterations=spinup_iterations)
        
        # Main simulation loop
        for day in range(self.n_days):
            # Get dynamic soil parameters
            smax, seav = self._calculate_dynamic_smax_seav(day)
            self.results['smax'][day] = smax
            self.results['seav'][day] = seav
            
            # Calculate crop ET demand
            crop_et = self.climate.pet[day] * self.crop.kc_values[day]
            
            # Calculate rainfed ET
            et_rain = self.climate.pet[day] * self.crop.landuse_kc
            et_rain = self._update_et(sm, et_rain, smax, seav)
            
            # Calculate irrigation requirement
            if self.management == "rainfed":
                irrig = 0
                et_actual = et_rain
            else:
                # Net irrigation requirement (NIR)
                nir = max(0, crop_et - et_rain)
                # Gross irrigation requirement (GIR)
                irrig = nir / self.irrigation_efficiency
                et_actual = crop_et
            
            # Calculate percolation
            perc = self._update_percolation(sm, smax, seav)
            
            # Calculate runoff
            runoff = self._update_runoff(
                sm, self.climate.precipitation[day], et_actual, smax, irrig
            )
            
            # Update water balance
            wb = self._update_water_balance(
                sm, self.climate.precipitation[day], et_actual, runoff, irrig
            )
            
            # Update soil moisture
            sm = self._update_soil_moisture(wb, smax)
            
            # Store results
            self.results['soil_moisture'][day] = sm
            self.results['evapotranspiration'][day] = et_actual
            self.results['irrigation'][day] = irrig
            self.results['percolation'][day] = perc
            self.results['runoff'][day] = runoff
        
        # Create output DataFrame
        output = pd.DataFrame({
            'date': self.climate.dates,
            'precipitation': self.climate.precipitation,
            'pet': self.climate.pet,
            'soil_moisture': self.results['soil_moisture'],
            'evapotranspiration': self.results['evapotranspiration'],
            'irrigation': self.results['irrigation'],
            'percolation': self.results['percolation'],
            'runoff': self.results['runoff'],
            'smax': self.results['smax'],
            'seav': self.results['seav'],
            'kc': self.crop.kc_values,
            'rooting_depth': self.crop.rooting_depth
        })
        
        return output
    
    def get_annual_summary(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily results to annual totals.
        
        Args:
            results: DataFrame from run() method
            
        Returns:
            DataFrame with annual water balance components
        """
        results['year'] = results['date'].dt.year
        
        annual = results.groupby('year').agg({
            'precipitation': 'sum',
            'pet': 'sum',
            'evapotranspiration': 'sum',
            'irrigation': 'sum',
            'percolation': 'sum',
            'runoff': 'sum',
            'soil_moisture': 'mean'
        }).reset_index()
        
        return annual
    
    def get_monthly_summary(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily results to monthly totals.
        
        Args:
            results: DataFrame from run() method
            
        Returns:
            DataFrame with monthly water balance components
        """
        results['year'] = results['date'].dt.year
        results['month'] = results['date'].dt.month
        
        monthly = results.groupby(['year', 'month']).agg({
            'precipitation': 'sum',
            'pet': 'sum',
            'evapotranspiration': 'sum',
            'irrigation': 'sum',
            'percolation': 'sum',
            'runoff': 'sum',
            'soil_moisture': 'mean'
        }).reset_index()
        
        return monthly


def create_crop_parameters_from_monthly_kc(
    crop_name: str,
    monthly_kc: List[float],
    rooting_depth_max: float,
    start_date: str,
    end_date: str,
    landuse_kc: float = 0.5,
    is_permanent_crop: bool = False
) -> CropParameters:
    """
    Create crop parameters from monthly Kc values.
    
    Args:
        crop_name: Name of the crop
        monthly_kc: List of 12 monthly Kc values (Jan-Dec)
        rooting_depth_max: Maximum rooting depth (m)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        landuse_kc: Land use Kc for rainfed conditions
        is_permanent_crop: If True, rooting depth is constant at maximum
        
    Returns:
        CropParameters object
    """
    if len(monthly_kc) != 12:
        raise ValueError("monthly_kc must have 12 values")
    
    # Create daily date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Expand monthly Kc to daily
    kc_daily = np.zeros(n_days)
    for i, date in enumerate(dates):
        month_idx = date.month - 1
        kc_daily[i] = monthly_kc[month_idx]
    
    # Create rooting depth array
    if is_permanent_crop:
        # Permanent crops have constant maximum rooting depth
        rooting_depth = np.full(n_days, rooting_depth_max)
    else:
        # Annual crops: interpolate rooting depth during growing season
        rooting_depth = np.zeros(n_days)
        
        # Process each year separately
        for year in dates.year.unique():
            year_mask = dates.year == year
            year_indices = np.where(year_mask)[0]
            year_kc = kc_daily[year_mask]
            
            # Find peak Kc day (end of root development)
            peak_day_in_year = np.argmax(year_kc)
            peak_day_global = year_indices[peak_day_in_year]
            
            # Interpolate from 0.2m to max during root development
            if peak_day_in_year > 0:
                rooting_depth[year_indices[:peak_day_in_year+1]] = np.linspace(
                    0.2, rooting_depth_max, peak_day_in_year + 1
                )
            
            # Constant max after peak
            rooting_depth[year_indices[peak_day_in_year:]] = rooting_depth_max
    
    return CropParameters(
        name=crop_name,
        kc_values=kc_daily,
        rooting_depth=rooting_depth,
        dates=dates,
        landuse_kc=landuse_kc
    )


def get_irrigation_efficiency(irrigation_system: str) -> float:
    """
    Get typical irrigation efficiency for different systems.
    
    Args:
        irrigation_system: Type of irrigation system
        
    Returns:
        Irrigation efficiency (0-1)
    """
    efficiencies = {
        'drip': 0.90,
        'sprinkler': 0.75,
        'traditional': 0.60,
        'flooded': 0.60,
        'rainfed': 1.00
    }
    
    system_lower = irrigation_system.lower()
    if system_lower not in efficiencies:
        raise ValueError(f"Unknown irrigation system: {irrigation_system}")
    
    return efficiencies[system_lower]
