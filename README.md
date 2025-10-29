# PyGlobWB - Vertical Water Balance Model

PyGlobWB is a Python implementation of a vertical water balance model for agricultural water management. It supports rainfed and irrigated systems, multiple irrigation efficiencies, and can be driven by synthetic or real climate data.

## Features

- Daily vertical water balance simulation
- Support for rainfed and irrigated management
- Configurable irrigation efficiency (drip, sprinkler, traditional)
- YAML-based crop parameters (Kc), rooting depth, and irrigation efficiency
- Example scripts and a comprehensive test suite

## Installation

```bash
# Clone the repository
git clone https://github.com/jgserra18/pyglobwb.git
cd pyglobwb

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the examples to generate demonstration outputs:

```bash
python -m pyglobwb.example_usage
```

or using YAML configuration:

```bash
python -m pyglobwb.example_with_config
```

Outputs include:
- `water_balance_comparison.png`
- `seasonal_patterns.png`

## Usage in Your Code

```python
from pyglobwb import (
    WaterBalanceModel,
    SoilParameters,
    CropParameters,
    ClimateData,
    ConfigManager,
    create_crop_parameters_from_monthly_kc,
    get_irrigation_efficiency,
)
```

## Configuration

YAML files in `config/` define:
- `crop_kc.yaml`: monthly Kc for crops
- `rooting_depth.yaml`: typical rooting depths
- `irrigation_efficiency.yaml`: efficiencies by irrigation system

You can also load them programmatically:

```python
from pyglobwb import ConfigManager
config = ConfigManager()  # looks in package config by default
```

## Running Tests

```bash
python -m pytest -v
```

CI runs tests on push/PR via GitHub Actions.

## License

MIT License (add a LICENSE file as needed)

## Acknowledgements

Ported and adapted from the Spain GlobWat scripts.
