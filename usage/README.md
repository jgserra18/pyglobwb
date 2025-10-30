# Usage Examples

This folder contains practical examples demonstrating how to use the water balance model with real-world data.

## Examples

### `example_real_climate_data.py`

Complete workflow using real climate data from OpenMeteo for Eastern Spain.

**Features:**
- Fetches climate data using `climatePy` library
- Loads crop parameters from YAML configuration
- Runs irrigated maize simulation with 50-year spinup
- Generates comprehensive plots

**Location:** Eastern Spain (38°N, 1.5°W)  
**Period:** 2015-2021 (7 years, daily)  
**Crop:** Irrigated Maize  
**Irrigation:** Drip system (90% efficiency)

**Requirements:**
```bash
pip install climatePy
```

**Run:**
```bash
python usage/example_real_climate_data.py
```

**Output:**
- `water_balance_2018.png` - Detailed time series for 2018
- `annual_summary.png` - 7-year annual totals
- `monthly_patterns.png` - Average seasonal patterns

## Key Utilities Used

### Climate Data (`climate_utils.py`)
- `fetch_climate_from_openmeteo()` - Fetch from OpenMeteo API
- `fetch_climate_from_nasa_power()` - Fetch from NASA POWER
- `prepare_climate_data()` - Convert to model format
- `validate_climate_data()` - Check for issues
- `calculate_aridity_index()` - Compute P/PET ratio

### Configuration (`config_manager.py`)
- `ConfigManager()` - Load YAML configurations
- `create_daily_kc()` - Generate daily crop coefficients
- `create_daily_rooting_depth()` - Interpolate rooting depth
- `get_irrigation_efficiency()` - Get system efficiency

### Model (`water_balance_model.py`)
- `WaterBalanceModel()` - Main model class
- `run(spinup_iterations=50)` - Execute simulation
- `get_annual_summary()` - Annual aggregation
- `get_monthly_summary()` - Monthly aggregation

## Customization

### Change Location
```python
latitude = 40.0   # Your latitude
longitude = -3.0  # Your longitude (negative for West)
```

### Change Crop
```python
crop_name = 'Wheat'  # See config/crop_kc.yaml for available crops
```

### Change Irrigation System
```python
irrigation_system = 'Sprinkler'  # Options: Drip, Sprinkler, Traditional, etc.
```

### Adjust Spinup
```python
spinup_iterations = 30  # Reduce for faster runs, increase for equilibration
```

### Change Climate Source
```python
from climate_utils import fetch_climate_from_nasa_power

climate_df = fetch_climate_from_nasa_power(
    latitude=latitude,
    longitude=longitude,
    start_date=start_date,
    end_date=end_date
)
```

## Tips

1. **First run**: Start with a short period (1-2 years) to test setup
2. **Spinup**: Use 30-50 iterations for multi-year runs
3. **Validation**: Always check `validate_climate_data()` output
4. **Aridity**: Review aridity index to understand water stress
5. **Plots**: Customize plots in the script for your needs

## Support

- See `CONFIG_GUIDE.md` for YAML configuration details
- See `QUICKSTART.md` for basic model usage
- See `README_water_balance_model.md` for model documentation
