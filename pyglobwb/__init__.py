"""
pyglobwb - Vertical Water Balance Model

A Python implementation of the GLOBWAT vertical water balance model
for crop water use estimation.
"""

from .water_balance_model import (
    WaterBalanceModel,
    SoilParameters,
    CropParameters,
    ClimateData,
    create_crop_parameters_from_monthly_kc,
    get_irrigation_efficiency
)

from .config_manager import ConfigManager, load_config

from .data_utils import (
    load_climate_csv,
    export_results_csv,
    export_summary_excel,
    create_comparison_table,
    validate_input_data
)

from .climate_utils import (
    fetch_climate_from_openmeteo,
    fetch_climate_from_nasa_power,
    prepare_climate_data,
    validate_climate_data,
    calculate_aridity_index,
    get_climate_statistics,
    resample_climate_data
)

__version__ = '0.1.0'

__all__ = [
    'WaterBalanceModel',
    'SoilParameters',
    'CropParameters',
    'ClimateData',
    'create_crop_parameters_from_monthly_kc',
    'get_irrigation_efficiency',
    'ConfigManager',
    'load_config',
    'load_climate_csv',
    'export_results_csv',
    'export_summary_excel',
    'create_comparison_table',
    'validate_input_data',
    'fetch_climate_from_openmeteo',
    'fetch_climate_from_nasa_power',
    'prepare_climate_data',
    'validate_climate_data',
    'calculate_aridity_index',
    'get_climate_statistics',
    'resample_climate_data',
]
