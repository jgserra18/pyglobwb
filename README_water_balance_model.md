# Vertical Water Balance Model

A generalized Python implementation of a vertical water balance model for crop water use estimation, based on the GLOBWAT framework.

## Overview

This model simulates daily soil moisture dynamics and water balance components for agricultural systems, including:

- **Soil moisture tracking** with dynamic rooting depth
- **Evapotranspiration** (actual and potential)
- **Irrigation requirements** (net and gross)
- **Deep percolation** (aquifer recharge)
- **Surface runoff**

The model is designed to be flexible and applicable to various crops, climates, and management scenarios.

## Key Features

- ✅ **Generalized inputs** - No region-specific dependencies
- ✅ **Multiple crop types** - Annual and permanent crops
- ✅ **Irrigation systems** - Drip, sprinkler, traditional, flooded, or rainfed
- ✅ **Dynamic rooting depth** - Adjusts soil water storage capacity
- ✅ **Spin-up capability** - Ensures equilibrium initial conditions
- ✅ **Flexible outputs** - Daily, monthly, and annual summaries

## Model Structure

### Core Components

1. **SoilParameters**: Hydraulic properties
   - Maximum soil moisture storage (Smax)
   - Maximum percolation rate (Rmax)
   - Initial soil moisture
   - Calibration factor

2. **CropParameters**: Crop-specific characteristics
   - Daily crop coefficients (Kc)
   - Rooting depth progression
   - Land use Kc for rainfed conditions

3. **ClimateData**: Meteorological forcing
   - Daily precipitation
   - Daily potential evapotranspiration (PET)

4. **WaterBalanceModel**: Main simulation engine
   - Integrates all components
   - Runs daily time-step simulation
   - Produces comprehensive outputs

## Water Balance Equations

### Soil Moisture Update

```
SM(t) = min(WB(t), Smax(t))
```

where:
```
WB(t) = SM(t-1) + P(t) + I(t) - ET(t) - R0(t)
```

### Evapotranspiration

```
ET(t) = PET(t) * Kc(t)                    [if SM(t-1) >= Seav]
ET(t) = PET(t) * Kc(t) * SM(t-1)/Seav    [if SM(t-1) < Seav]
```

### Irrigation Requirement

```
NIR(t) = max(0, ET_crop(t) - ET_rain(t))  [Net Irrigation Requirement]
GIR(t) = NIR(t) / efficiency              [Gross Irrigation Requirement]
```

### Deep Percolation

```
Perc(t) = 0                                           [if SM(t-1) < Seav]
Perc(t) = Rmax * CF * (SM(t-1) - Seav)/(Smax - Seav) [if SM(t-1) >= Seav]
```

where CF is the calibration factor (default 2.4)

### Surface Runoff

```
R0(t) = 0                [if WB(t) < Smax]
R0(t) = WB(t) - Smax     [if WB(t) >= Smax]
```

### Dynamic Soil Parameters

```
Smax(t) = Smax_base * Zr(t) / Zr_ref
Seav(t) = 0.5 * Smax(t)
```

where:
- Zr(t) is the current rooting depth
- Zr_ref is the reference depth (typically 0.6 m)
- Seav is the easily available water (50% of Smax)

## Installation

### Requirements

```bash
pip install numpy pandas matplotlib
```

### Files

- `water_balance_model.py` - Core model implementation
- `example_usage.py` - Demonstration scripts
- `README_water_balance_model.md` - This documentation

## Quick Start

### Basic Example

```python
import numpy as np
import pandas as pd
from water_balance_model import (
    WaterBalanceModel,
    SoilParameters,
    CropParameters,
    ClimateData,
    create_crop_parameters_from_monthly_kc,
    get_irrigation_efficiency
)

# Define simulation period
dates = pd.date_range(start='2015-01-01', end='2019-12-31', freq='D')

# Create climate data
climate = ClimateData(
    precipitation=np.random.gamma(2, 2, len(dates)),  # mm/day
    pet=3.0 + 2.0 * np.sin(2 * np.pi * dates.dayofyear / 365),  # mm/day
    dates=dates
)

# Define soil parameters
soil = SoilParameters(
    smax_base=150.0,      # mm at reference depth
    reference_depth=0.6,  # m
    rmax=10.0,            # mm/day
    initial_sm=75.0,      # mm
    calibration_factor=2.4
)

# Define crop parameters (Maize example)
monthly_kc = [0, 0, 0, 0.3, 0.64, 1.17, 1.2, 0.70, 0, 0, 0, 0]
crop = create_crop_parameters_from_monthly_kc(
    crop_name='Maize',
    monthly_kc=monthly_kc,
    rooting_depth_max=1.2,  # m
    start_date='2015-01-01',
    end_date='2019-12-31',
    is_permanent_crop=False
)

# Create and run model
model = WaterBalanceModel(
    soil_params=soil,
    crop_params=crop,
    climate_data=climate,
    management='irrigated',
    irrigation_efficiency=get_irrigation_efficiency('drip')
)

# Run simulation
results = model.run(spinup_iterations=50)

# Get annual summary
annual = model.get_annual_summary(results)
print(annual)
```

## Usage Examples

The `example_usage.py` file contains three detailed examples:

### Example 1: Maize (Annual Crop)
- Compares rainfed vs. irrigated scenarios
- Demonstrates dynamic rooting depth
- Shows irrigation requirement calculation

### Example 2: Wheat (Winter Crop)
- Compares different irrigation systems
- Evaluates irrigation efficiency impacts
- Demonstrates seasonal patterns

### Example 3: Olive (Permanent Crop)
- Shows constant rooting depth implementation
- Mediterranean climate pattern
- Deep-rooted crop behavior

Run all examples:
```bash
python example_usage.py
```

## Input Data Requirements

### Climate Data

| Variable | Unit | Frequency | Description |
|----------|------|-----------|-------------|
| Precipitation | mm | Daily | Total rainfall |
| PET | mm | Daily | Potential evapotranspiration |

**Sources**: 
- E-OBS for precipitation
- FAO-56 Penman-Monteith for PET
- ERA5 reanalysis
- Local weather stations

### Crop Parameters

| Parameter | Unit | Description |
|-----------|------|-------------|
| Kc | - | Crop coefficient (monthly or daily) |
| Zr_max | m | Maximum rooting depth |
| Land use Kc | - | Background Kc for rainfed ET |

**Sources**:
- FAO-56 crop coefficient tables
- Regional agricultural databases
- Field measurements

### Soil Parameters

| Parameter | Unit | Description |
|-----------|------|-------------|
| Smax_base | mm | Max soil moisture at reference depth |
| Rmax | mm/day | Maximum percolation rate |
| Reference depth | m | Depth for Smax_base (typically 0.6m) |

**Estimation**:
```
Smax_base = (θ_fc - θ_wp) × Zr_ref × 1000
```

where:
- θ_fc = Field capacity (m³/m³)
- θ_wp = Wilting point (m³/m³)
- Zr_ref = Reference depth (m)

**Sources**:
- Soil texture databases (e.g., SoilGrids)
- Pedotransfer functions
- Field measurements

## Output Variables

### Daily Outputs

| Variable | Unit | Description |
|----------|------|-------------|
| soil_moisture | mm | Current soil water storage |
| evapotranspiration | mm | Actual ET |
| irrigation | mm | Gross irrigation applied |
| percolation | mm | Deep drainage |
| runoff | mm | Surface runoff |
| smax | mm | Maximum soil moisture capacity |
| seav | mm | Easily available water threshold |

### Aggregated Outputs

- **Monthly**: Sum of fluxes, mean soil moisture
- **Annual**: Sum of fluxes, mean soil moisture
- **Growing season**: Filtered by Kc > 0

## Irrigation Systems

The model supports different irrigation efficiencies:

| System | Efficiency | Description |
|--------|-----------|-------------|
| Drip | 0.90 | Localized, high efficiency |
| Sprinkler | 0.75 | Overhead application |
| Traditional | 0.60 | Surface/furrow irrigation |
| Flooded | 0.60 | Basin/border irrigation |
| Rainfed | 1.00 | No irrigation |

Custom efficiencies can be specified directly.

## Model Calibration

### Key Calibration Parameters

1. **Calibration Factor (CF)**: Adjusts percolation rate
   - Default: 2.4 (calibrated for Spain)
   - Range: 1.0 - 5.0
   - Higher values → more percolation

2. **Smax_base**: Controls soil water storage
   - Depends on soil texture and depth
   - Typical range: 50-250 mm

3. **Rmax**: Maximum percolation rate
   - Depends on soil hydraulic conductivity
   - Typical range: 5-20 mm/day

### Calibration Approach

1. **Observed data**: Soil moisture, ET, or irrigation records
2. **Objective function**: RMSE, NSE, or bias
3. **Parameters**: Adjust CF, Smax_base, Rmax
4. **Validation**: Split-sample or cross-validation

## Model Assumptions

1. **Single soil layer**: Vertically integrated
2. **No lateral flow**: 1D vertical water movement
3. **Uniform soil properties**: Spatially homogeneous
4. **No groundwater interaction**: Deep percolation is lost
5. **Instantaneous irrigation**: Applied at start of day
6. **No interception**: Precipitation reaches soil directly

## Limitations

- Does not simulate:
  - Soil salinity
  - Nutrient dynamics
  - Crop growth/yield
  - Groundwater table effects
  - Capillary rise
  - Snow accumulation/melt

## References

1. **GLOBWAT Model**: Hoogeveen et al. (2015). Global Water Satisfaction and Shortage in Agriculture.

2. **FAO-56**: Allen et al. (1998). Crop evapotranspiration - Guidelines for computing crop water requirements. FAO Irrigation and drainage paper 56.

3. **Percolation Calibration**: Siebert et al. (2010). Groundwater use for irrigation - a global inventory. Hydrology and Earth System Sciences.

## Citation

If you use this model in your research, please cite:

```
[Your citation information here]
```

## License

[Specify your license]

## Contact

[Your contact information]

## Changelog

### Version 1.0.0 (2024)
- Initial release
- Core water balance functions
- Support for annual and permanent crops
- Multiple irrigation systems
- Example scripts and documentation
