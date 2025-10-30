# PyGlobWB - Vertical Water Balance Model

[![Tests](https://img.shields.io/badge/tests-104%20passing-brightgreen)](https://github.com/jgserra18/pyglobwb)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A Python implementation of a vertical water balance model for agricultural water management and crop water use estimation. Based on the GLOBWAT framework and FAO-56 methodology.

## ✨ Features

### Core Model
- **Daily water balance simulation** with dynamic soil moisture tracking
- **Rainfed and irrigated** management scenarios
- **Multiple irrigation systems** (drip, sprinkler, traditional, etc.)
- **Dynamic rooting depth** for annual and permanent crops
- **Spinup capability** for equilibrated initial conditions

### Climate Data Integration
- **Fetch real climate data** from Open-Meteo ERA5 (global, 1940-present)
- **No API key required** - free and open access
- **Daily precipitation and ET0** (FAO-56 reference evapotranspiration)
- **Built-in validation** and quality checks

### Configuration
- **YAML-based crop database** with 30+ crops (maize, wheat, olive, etc.)
- **Configurable parameters** for soil, irrigation, and regional calibration
- **Easy customization** - add your own crops via YAML or Python API

### Analysis & Visualization
- **Annual and monthly summaries** of water balance components
- **Aridity index calculation** for climate classification
- **Comprehensive plotting** utilities
- **Export to CSV/Excel** for further analysis

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/jgserra18/pyglobwb.git
cd pyglobwb

# (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- numpy, pandas, matplotlib
- pyyaml (for configuration)
- requests (for climate data fetching)

## 🚀 Quick Start

### Example 1: Real Climate Data (Recommended)

Fetch real climate data and run a water balance simulation:

```bash
python usage/example_real_climate_data.py
```

This example:
- Fetches 7 years of daily climate data from Open-Meteo ERA5
- Runs irrigated maize simulation for Eastern Spain
- Generates comprehensive plots and statistics
- **No API key needed!**

**Output:**
- `usage/water_balance_2018.png` - Detailed time series
- `usage/annual_summary.png` - Multi-year comparison  
- `usage/monthly_patterns.png` - Seasonal patterns

### Example 2: Custom Simulation

```python
from pyglobwb import (
    WaterBalanceModel,
    SoilParameters,
    CropParameters,
    ConfigManager,
    fetch_climate_from_openmeteo,
    prepare_climate_data
)

# Fetch climate data
climate_df = fetch_climate_from_openmeteo(
    latitude=38.0,
    longitude=-1.5,
    start_date="2020-01-01",
    end_date="2020-12-31"
)

climate = prepare_climate_data(climate_df)

# Load crop parameters from YAML
config = ConfigManager()
kc_daily, dates = config.create_daily_kc('Maize', '2020-01-01', '2020-12-31')
rooting_depth, _ = config.create_daily_rooting_depth(
    'Maize', '2020-01-01', '2020-12-31', 'irrigated'
)

crop = CropParameters(
    name='Maize',
    kc_values=kc_daily,
    rooting_depth=rooting_depth,
    dates=dates,
    landuse_kc=0.5
)

# Define soil
soil = SoilParameters(
    smax_base=150.0,  # mm
    rmax=10.0,        # mm/day
    calibration_factor=2.4
)

# Run model
model = WaterBalanceModel(
    soil_params=soil,
    crop_params=crop,
    climate_data=climate,
    management='irrigated',
    irrigation_efficiency=0.90  # Drip irrigation
)

results = model.run(spinup_iterations=50)

# Get summaries
annual = model.get_annual_summary(results)
print(annual)
```

## 📚 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get started in 5 minutes
- **[Configuration Guide](CONFIG_GUIDE.md)** - YAML configuration details
- **[Usage Examples](usage/README.md)** - Real-world examples
- **[Model Documentation](README_water_balance_model.md)** - Technical details

## 🌍 Available Crops

**Field Crops:** Alfalfa, Barley, Wheat, Maize, Rice, Sorghum, Cotton, Rape  
**Vegetables:** Tomato, Potato, Onion, Lettuce, Beans, Chickpea, Lentils, Cabbage, Cauliflower  
**Fruit Trees:** Olive, Orange, Citrus, Almonds, Apple, Peach, Plum, Grapevine, Avocado, Kiwifruit, Blueberries

See [`config/crop_kc.yaml`](config/crop_kc.yaml) for complete list with parameters.

## ⚙️ Configuration

All configuration is stored in YAML files in the `config/` directory:

- **`crop_kc.yaml`** - Monthly crop coefficients for 30+ crops
- **`rooting_depth.yaml`** - Maximum rooting depths (irrigated/rainfed)
- **`irrigation_efficiency.yaml`** - Irrigation system efficiencies

### Adding Custom Crops

**Option 1: Edit YAML directly**
```yaml
# config/crop_kc.yaml
crops:
  MyCustomCrop:
    kc_monthly: [0.3, 0.4, 0.6, 0.9, 1.1, 1.2, 1.2, 1.0, 0.7, 0.5, 0.3, 0.3]
    order: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    is_permanent: false
```

**Option 2: Use Python API**
```python
config = ConfigManager()
config.update_crop_kc(
    crop_name='MyCustomCrop',
    monthly_kc=[0.3, 0.4, 0.6, 0.9, 1.1, 1.2, 1.2, 1.0, 0.7, 0.5, 0.3, 0.3],
    is_permanent=False
)
config.update_rooting_depth(
    crop_name='MyCustomCrop',
    irrigated=0.8,
    rainfed=1.2
)
config.save_configurations()
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest -v

# Run specific test modules
python -m pytest pytests/test_climate_utils.py -v
python -m pytest pytests/test_config_manager.py -v

# Run with coverage
python -m pytest --cov=pyglobwb --cov-report=html
```

**Test Coverage:**
- 104 tests passing
- Core model functions
- Configuration management
- Climate data fetching and validation
- Integration tests

## 🌐 Climate Data Sources

### Open-Meteo ERA5 (Recommended)
- **Coverage:** Global, 1940-present
- **Resolution:** ~25km (ERA5 reanalysis)
- **Variables:** Daily precipitation, FAO-56 ET0
- **API Key:** Not required
- **Function:** `fetch_climate_from_openmeteo()`

### Optional: TerraClimate, GridMET, NASA POWER
Install `climatePy` for additional data sources:
```bash
pip install climatePy
```

## 📊 Model Outputs

The model provides daily, monthly, and annual summaries:

**Daily outputs:**
- Soil moisture (mm)
- Actual evapotranspiration (mm/day)
- Irrigation requirement (mm/day)
- Deep percolation (mm/day)
- Surface runoff (mm/day)
- Rooting depth (m)
- Crop coefficient (Kc)

**Aggregated summaries:**
- Annual water balance totals
- Monthly patterns and statistics
- Aridity index and climate classification

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgements

- Based on the **GLOBWAT** framework for global water balance modeling
- Follows **FAO-56** methodology for crop water requirements
- Climate data from **Open-Meteo** and **ERA5** reanalysis
- Ported and adapted from Spain GlobWat scripts

## 📖 Citation

If you use this model in your research, please cite:

```bibtex
@software{pyglobwb2024,
  title = {PyGlobWB: Vertical Water Balance Model for Agricultural Water Management},
  author = {Serra, J.G.},
  year = {2024},
  url = {https://github.com/jgserra18/pyglobwb}
}
```

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on [GitHub](https://github.com/jgserra18/pyglobwb/issues)
- Check the [documentation](CONFIG_GUIDE.md)

---

**Made with ❤️ for sustainable water management**
