# YAML Configuration Guide

This guide explains how to use the YAML configuration system for the water balance model.

## Overview

The model uses three YAML configuration files:
1. **`crop_kc.yaml`** - Crop coefficients and growing season information
2. **`rooting_depth.yaml`** - Maximum rooting depths for irrigated and rainfed conditions
3. **`irrigation_efficiency.yaml`** - Irrigation system efficiencies

## Configuration Files

### 1. Crop Kc Configuration (`crop_kc.yaml`)

Defines monthly crop coefficients for each crop.

**Structure:**
```yaml
crops:
  Maize:
    kc_monthly: [0.0, 0.0, 0.0, 0.3, 0.638, 1.166, 1.2, 0.698, 0.0, 0.0, 0.0, 0.0]
    order: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    is_permanent: false
```

**Fields:**
- `kc_monthly`: List of 12 monthly Kc values (January to December)
- `order`: Month order for growing season (1=Jan, 12=Dec)
- `is_permanent`: `true` for perennial crops, `false` for annual crops

**Permanent crops** (trees, perennials): Constant rooting depth year-round
**Annual crops**: Rooting depth interpolates during growing season

### 2. Rooting Depth Configuration (`rooting_depth.yaml`)

Defines maximum rooting depths for different management types.

**Structure:**
```yaml
crops:
  Maize:
    irrigated: 1.0  # meters
    rainfed: 1.7    # meters
```

**Fields:**
- `irrigated`: Maximum rooting depth under irrigation (m)
- `rainfed`: Maximum rooting depth under rainfed conditions (m)

**Note:** Rainfed crops typically develop deeper roots to access more water.

### 3. Irrigation Efficiency Configuration (`irrigation_efficiency.yaml`)

Defines efficiency for different irrigation systems.

**Structure:**
```yaml
irrigation_systems:
  Drip:
    efficiency: 0.90
    description: "Drip/trickle irrigation - localized, high efficiency"
    typical_application_rate: "2-4 mm/hr"
```

**Available systems:**
- `Drip`: 90% efficiency
- `Sprinkler`: 75% efficiency
- `Traditional`: 60% efficiency
- `Flooded`: 60% efficiency
- `Rainfed`: 100% (no irrigation)

## Using the Configuration Manager

### Basic Usage

```python
from config_manager import ConfigManager

# Load configuration
config = ConfigManager()

# List available crops
crops = config.list_crops()
print(crops)

# Get crop Kc parameters
maize_kc = config.get_crop_kc('Maize')
print(maize_kc['kc_monthly'])

# Get rooting depth
depth = config.get_rooting_depth('Maize', 'irrigated')
print(f"Rooting depth: {depth} m")

# Get irrigation efficiency
efficiency = config.get_irrigation_efficiency('Drip')
print(f"Efficiency: {efficiency:.0%}")
```

### Creating Daily Values

The configuration manager can automatically convert monthly values to daily:

```python
# Create daily Kc values
kc_daily, dates = config.create_daily_kc(
    crop_name='Maize',
    start_date='2020-01-01',
    end_date='2020-12-31'
)

# Create daily rooting depth with interpolation
rooting_depth, dates = config.create_daily_rooting_depth(
    crop_name='Maize',
    start_date='2020-01-01',
    end_date='2020-12-31',
    management='irrigated',
    interpolation_method='linear'  # or 'sigmoid'
)
```

### Rooting Depth Interpolation

For **annual crops**, rooting depth is interpolated during the growing season:

1. **Before growing season** (Kc = 0): Minimal depth (0.2 m)
2. **Root development** (Kc increasing): Linear interpolation from 0.2 m to max depth
3. **Peak growth** (Kc = max): Constant at maximum depth
4. **After growing season** (Kc = 0): Back to minimal depth

For **permanent crops**, rooting depth is constant at maximum throughout the year.

**Interpolation methods:**
- `linear`: Straight line from 0.2 m to max depth
- `sigmoid`: S-curve for more realistic root growth

### Dynamic Configuration Updates

You can add or modify crops at runtime:

```python
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

# Save changes to YAML files
config.save_configurations()
```

## Integration with Water Balance Model

### Complete Example

```python
from config_manager import ConfigManager
from water_balance_model import (
    WaterBalanceModel, SoilParameters, CropParameters, ClimateData
)
import pandas as pd
import numpy as np

# Load configuration
config = ConfigManager()

# Define simulation period
dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')

# Create climate data (example)
climate = ClimateData(
    precipitation=np.random.gamma(2, 2, len(dates)),
    pet=3 + 2 * np.sin(2 * np.pi * dates.dayofyear / 365),
    dates=dates
)

# Define soil
soil = SoilParameters(
    smax_base=150.0,
    rmax=10.0,
    calibration_factor=2.4
)

# Load crop parameters from YAML
crop_name = 'Maize'
management = 'irrigated'

kc_daily, _ = config.create_daily_kc(crop_name, '2020-01-01', '2020-12-31')
rooting_depth, _ = config.create_daily_rooting_depth(
    crop_name, '2020-01-01', '2020-12-31', management
)

crop = CropParameters(
    name=crop_name,
    kc_values=kc_daily,
    rooting_depth=rooting_depth,
    dates=dates,
    landuse_kc=0.5
)

# Get irrigation efficiency
irrigation_efficiency = config.get_irrigation_efficiency('Drip')

# Run model
model = WaterBalanceModel(soil, crop, climate, management, irrigation_efficiency)
results = model.run()

# Get summary
annual = model.get_annual_summary(results)
print(annual)
```

## Available Crops

The configuration includes 32 crops:

**Field Crops:**
- Alfalfa, Barley, Wheat, Maize, Rice, Sorghum, Cottonseed, Rape

**Vegetables:**
- Tomato, Potato, Onion, Lettuce, Beans, Chickpea, Lentils
- Faba bean/Broad bean, Sunflower seed, Cabbage/Broccoli, Cauliflower
- Vegetables (other)

**Fruit Trees:**
- Olive, Orange, Citrus, Almonds, Apple, Peach, Plum
- Grapevine, Avocado, Kiwifruit, Blueberries

## Adding New Crops

### Method 1: Edit YAML Files Directly

Edit `config/crop_kc.yaml`:
```yaml
crops:
  NewCrop:
    kc_monthly: [0.4, 0.5, 0.7, 0.9, 1.1, 1.2, 1.1, 0.9, 0.7, 0.5, 0.4, 0.4]
    order: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    is_permanent: false
```

Edit `config/rooting_depth.yaml`:
```yaml
crops:
  NewCrop:
    irrigated: 0.8
    rainfed: 1.2
```

### Method 2: Use Python API

```python
config = ConfigManager()

config.update_crop_kc(
    crop_name='NewCrop',
    monthly_kc=[0.4, 0.5, 0.7, 0.9, 1.1, 1.2, 1.1, 0.9, 0.7, 0.5, 0.4, 0.4],
    is_permanent=False
)

config.update_rooting_depth(
    crop_name='NewCrop',
    irrigated=0.8,
    rainfed=1.2
)

config.save_configurations()
```

## Month Order for Winter Crops

Some crops (wheat, barley, chickpea, lentils, rape) have growing seasons that span calendar years. Use the `order` field to specify the correct sequence:

```yaml
Wheat:
  kc_monthly: [0.848, 0.943, 1.038, 1.130, 1.150, 0.648, 0.0, 0.0, 0.0, 0.0, 0.7, 0.748]
  order: [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Starts in November
  is_permanent: false
```

This means:
- November (month 11): Kc = 0.7
- December (month 12): Kc = 0.748
- January (month 1): Kc = 0.848
- etc.

## Exporting Configuration Summary

```python
config = ConfigManager()

# Export summary to CSV
summary = config.export_crop_summary('crop_summary.csv')
print(summary)
```

Output includes:
- Crop name
- Is permanent crop
- Kc min, max, mean
- Rooting depths (irrigated and rainfed)

## Tips and Best Practices

1. **Validate Kc values**: Should be between 0 and 1.5 (typically 0-1.2)
2. **Check rooting depths**: Rainfed should be ≥ irrigated
3. **Permanent crops**: Set `is_permanent: true` for trees and perennials
4. **Growing season**: Set Kc = 0 for months outside growing season
5. **Backup configs**: Keep backup copies before making changes
6. **Test changes**: Run model with new crops to verify results

## Troubleshooting

### Crop not found
```python
# Check available crops
config = ConfigManager()
print(config.list_crops())
```

### Invalid Kc values
- Ensure 12 values in `kc_monthly`
- Values should be 0-1.5
- Use 0 for non-growing months

### Rooting depth issues
- Check both irrigated and rainfed are defined
- Values should be in meters (0.3-2.0 typical range)
- Rainfed ≥ irrigated

### YAML syntax errors
- Use proper indentation (2 spaces)
- Quote strings with special characters
- Check for missing colons or brackets

## Example Scripts

See these example scripts for complete usage:
- `example_with_config.py` - Basic usage with YAML configs
- `example_usage.py` - Original examples with hardcoded values
- `config_manager.py` - Configuration manager source code

## References

- FAO-56: Crop evapotranspiration guidelines
- GLOBWAT: Global water balance model
- Crop coefficients: Regional agricultural databases
