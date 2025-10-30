# Quick Start Guide - Water Balance Model

Get started with the vertical water balance model in 5 minutes.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Simple Example with Synthetic Data

```python
import numpy as np
import pandas as pd
from water_balance_model import (
    WaterBalanceModel, SoilParameters, ClimateData,
    create_crop_parameters_from_monthly_kc,
    get_irrigation_efficiency
)

# Create date range
dates = pd.date_range('2015-01-01', '2019-12-31', freq='D')

# Generate simple climate data
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

# Define crop (Maize)
crop = create_crop_parameters_from_monthly_kc(
    crop_name='Maize',
    monthly_kc=[0, 0, 0, 0.3, 0.64, 1.17, 1.2, 0.70, 0, 0, 0, 0],
    rooting_depth_max=1.2,
    start_date='2015-01-01',
    end_date='2019-12-31'
)

# Run model
model = WaterBalanceModel(soil, crop, climate, 'irrigated', 
                         get_irrigation_efficiency('drip'))
results = model.run()

# View results
print(results.head())
print("\nAnnual Summary:")
print(model.get_annual_summary(results))
```

### 2. Using YAML Configuration

See `example_with_config.py` and `CONFIG_GUIDE.md` for a complete example using the YAML files in the `config/` directory via `ConfigManager`.

### 3. Loading Your Own Data

```python
from data_utils import load_climate_csv, validate_input_data

# Load climate data
climate_df = load_climate_csv(
    'your_climate_data.csv',
    date_column='date',
    pr_column='rainfall',
    pet_column='evapotranspiration'
)

# Validate data
issues = validate_input_data(climate_df)
if issues['errors']:
    print("Fix these errors:", issues['errors'])
else:
    print("Data is valid!")

# Create ClimateData object
climate = ClimateData(
    precipitation=climate_df['precipitation'].values,
    pet=climate_df['pet'].values,
    dates=pd.DatetimeIndex(climate_df['date'])
)
```

### 4. Comparing Scenarios

```python
# Run multiple scenarios
scenarios = {}

for system in ['rainfed', 'drip', 'sprinkler']:
    management = 'rainfed' if system == 'rainfed' else 'irrigated'
    efficiency = get_irrigation_efficiency(system)
    
    model = WaterBalanceModel(soil, crop, climate, management, efficiency)
    scenarios[system] = model.run()

# Compare results
from data_utils import create_comparison_table
comparison = create_comparison_table(scenarios, metric='annual_sum')
print(comparison)
```

### 5. Exporting Results

```python
from data_utils import export_results_csv, export_summary_excel

# Export to CSV
export_results_csv(results, 'output_daily.csv')

# Export to Excel with multiple sheets
export_summary_excel(results, 'output_summary.xlsx', 
                    model_config={'crop': 'Maize', 'soil': 'Loam'})
```

## Run Complete Examples

```bash
# Run all examples with plots
python example_usage.py
```

This will generate:
- `water_balance_comparison.png` - Rainfed vs irrigated comparison
- `seasonal_patterns.png` - Seasonal water balance patterns

## Available Crops

Run this to see all available crops:

```python
from config_manager import ConfigManager
print("Available crops:", ConfigManager().list_crops())
```

**Common crops**: maize, wheat, barley, rice, tomato, potato, olive, almond, grapevine, orange, apple

## Key Parameters to Adjust

### Soil Parameters
- `smax_base`: Water holding capacity (50-250 mm typical)
- `rmax`: Percolation rate (3-20 mm/day)
- `calibration_factor`: Regional adjustment (1.5-3.0)

### Crop Parameters
- `monthly_kc`: Crop coefficients (0-1.2)
- `rooting_depth_max`: Maximum root depth (0.3-2.0 m)
- `is_permanent_crop`: True for trees, False for annuals

### Irrigation
- `management`: 'rainfed' or 'irrigated'
- `irrigation_efficiency`: 0.6 (traditional) to 0.9 (drip)

## Troubleshooting

### Issue: Negative soil moisture
**Solution**: Increase `smax_base` or reduce `rmax`

### Issue: Too much runoff
**Solution**: Increase `smax_base` or check precipitation data

### Issue: Unrealistic irrigation amounts
**Solution**: Check `irrigation_efficiency` and crop Kc values

### Issue: Model runs slowly
**Solution**: Reduce `spinup_iterations` from 50 to 20-30

## Next Steps

1. Read the full documentation: `README_water_balance_model.md`
2. Explore YAML configuration: `CONFIG_GUIDE.md`
3. Check data utilities: `data_utils.py`
4. Review example scripts: `example_usage.py`, `example_with_config.py`

## Support

For questions or issues, refer to the main README or check the inline documentation in the source code.
