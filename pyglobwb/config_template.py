"""
Configuration template for water balance model simulations.

This file provides example configurations for common crops and scenarios.
Copy and modify as needed for your specific application.
"""

from typing import Dict, List

# ============================================================================
# CROP COEFFICIENT DATABASE
# ============================================================================
# Monthly Kc values (Jan-Dec) for various crops
# Source: FAO-56 and regional adaptations

CROP_KC_DATABASE: Dict[str, List[float]] = {
    # Field crops
    'maize': [0, 0, 0, 0.3, 0.64, 1.17, 1.2, 0.70, 0, 0, 0, 0],
    'wheat': [0.4, 0.5, 0.7, 1.15, 1.15, 0.4, 0, 0, 0, 0, 0.3, 0.4],
    'barley': [0.3, 0.4, 0.7, 1.15, 1.15, 0.4, 0, 0, 0, 0, 0, 0.3],
    'rice': [0, 0, 0, 1.05, 1.2, 1.2, 1.05, 0.7, 0, 0, 0, 0],
    'sorghum': [0, 0, 0, 0.3, 0.55, 1.0, 1.0, 0.55, 0, 0, 0, 0],
    'sunflower': [0, 0, 0, 0.35, 0.7, 1.15, 1.15, 0.7, 0.35, 0, 0, 0],
    'cotton': [0, 0, 0, 0.35, 0.7, 1.15, 1.15, 0.7, 0.5, 0, 0, 0],
    
    # Vegetables
    'tomato': [0, 0, 0, 0.6, 1.15, 1.15, 1.15, 0.8, 0.6, 0, 0, 0],
    'potato': [0, 0, 0.5, 0.75, 1.15, 1.15, 0.95, 0.75, 0, 0, 0, 0],
    'onion': [0, 0, 0.5, 0.7, 1.05, 1.05, 0.85, 0.75, 0, 0, 0, 0],
    'lettuce': [0.7, 0.7, 1.0, 1.0, 0.95, 0, 0, 0, 0, 0.7, 0.7, 0.7],
    'cabbage': [0, 0, 0.7, 1.05, 1.05, 1.05, 0.95, 0, 0, 0, 0, 0],
    'beans': [0, 0, 0, 0.5, 0.75, 1.05, 1.05, 0.9, 0.85, 0, 0, 0],
    'chickpea': [0, 0, 0.4, 0.7, 1.0, 1.0, 0.7, 0.4, 0, 0, 0, 0],
    'lentils': [0, 0, 0.4, 0.7, 1.1, 1.1, 0.7, 0.4, 0, 0, 0, 0],
    
    # Fruit trees (permanent crops)
    'olive': [0.5, 0.5, 0.6, 0.65, 0.7, 0.7, 0.7, 0.7, 0.65, 0.6, 0.55, 0.5],
    'almond': [0.4, 0.4, 0.5, 0.6, 0.7, 0.7, 0.7, 0.65, 0.6, 0.5, 0.45, 0.4],
    'grapevine': [0.3, 0.3, 0.4, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
    'orange': [0.7, 0.7, 0.65, 0.65, 0.65, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
    'apple': [0.45, 0.45, 0.6, 0.8, 0.95, 0.95, 0.95, 0.9, 0.75, 0.6, 0.5, 0.45],
    'peach': [0.45, 0.45, 0.55, 0.8, 0.95, 0.95, 0.9, 0.85, 0.7, 0.6, 0.5, 0.45],
    'avocado': [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
    'kiwifruit': [0.4, 0.4, 0.5, 0.7, 1.0, 1.05, 1.05, 1.0, 0.8, 0.6, 0.5, 0.4],
    
    # Forages
    'alfalfa': [0.4, 0.5, 0.7, 0.95, 1.05, 1.05, 1.05, 1.0, 0.95, 0.85, 0.6, 0.4],
}

# ============================================================================
# ROOTING DEPTH DATABASE
# ============================================================================
# Maximum rooting depths (m) for various crops
# Format: {crop: {'irrigated': depth_m, 'rainfed': depth_m}}

ROOTING_DEPTH_DATABASE: Dict[str, Dict[str, float]] = {
    # Field crops
    'maize': {'irrigated': 1.2, 'rainfed': 1.5},
    'wheat': {'irrigated': 1.0, 'rainfed': 1.2},
    'barley': {'irrigated': 1.0, 'rainfed': 1.2},
    'rice': {'irrigated': 0.5, 'rainfed': 0.6},
    'sorghum': {'irrigated': 1.2, 'rainfed': 1.5},
    'sunflower': {'irrigated': 1.0, 'rainfed': 1.5},
    'cotton': {'irrigated': 1.2, 'rainfed': 1.5},
    
    # Vegetables
    'tomato': {'irrigated': 0.7, 'rainfed': 1.0},
    'potato': {'irrigated': 0.5, 'rainfed': 0.6},
    'onion': {'irrigated': 0.4, 'rainfed': 0.5},
    'lettuce': {'irrigated': 0.3, 'rainfed': 0.4},
    'cabbage': {'irrigated': 0.5, 'rainfed': 0.6},
    'beans': {'irrigated': 0.6, 'rainfed': 0.8},
    'chickpea': {'irrigated': 0.8, 'rainfed': 1.0},
    'lentils': {'irrigated': 0.8, 'rainfed': 1.0},
    
    # Fruit trees (permanent crops)
    'olive': {'irrigated': 1.5, 'rainfed': 2.0},
    'almond': {'irrigated': 1.5, 'rainfed': 2.0},
    'grapevine': {'irrigated': 1.2, 'rainfed': 1.5},
    'orange': {'irrigated': 1.2, 'rainfed': 1.5},
    'apple': {'irrigated': 1.2, 'rainfed': 1.5},
    'peach': {'irrigated': 1.2, 'rainfed': 1.5},
    'avocado': {'irrigated': 1.0, 'rainfed': 1.2},
    'kiwifruit': {'irrigated': 1.0, 'rainfed': 1.2},
    
    # Forages
    'alfalfa': {'irrigated': 1.5, 'rainfed': 2.0},
}

# ============================================================================
# PERMANENT CROPS
# ============================================================================
# Crops with constant rooting depth throughout the year

PERMANENT_CROPS: List[str] = [
    'olive', 'almond', 'grapevine', 'orange', 'apple', 'peach',
    'avocado', 'kiwifruit', 'alfalfa'
]

# ============================================================================
# SOIL TYPE PARAMETERS
# ============================================================================
# Typical soil parameters for different textures
# Smax_base is calculated as: (θ_fc - θ_wp) × 0.6m × 1000

SOIL_TYPE_DATABASE: Dict[str, Dict[str, float]] = {
    'sand': {
        'smax_base': 60.0,   # mm at 0.6m depth
        'rmax': 20.0,        # mm/day (high percolation)
        'field_capacity': 0.15,
        'wilting_point': 0.05
    },
    'loamy_sand': {
        'smax_base': 80.0,
        'rmax': 15.0,
        'field_capacity': 0.20,
        'wilting_point': 0.07
    },
    'sandy_loam': {
        'smax_base': 100.0,
        'rmax': 12.0,
        'field_capacity': 0.25,
        'wilting_point': 0.08
    },
    'loam': {
        'smax_base': 130.0,
        'rmax': 10.0,
        'field_capacity': 0.30,
        'wilting_point': 0.09
    },
    'silt_loam': {
        'smax_base': 150.0,
        'rmax': 8.0,
        'field_capacity': 0.33,
        'wilting_point': 0.08
    },
    'sandy_clay_loam': {
        'smax_base': 140.0,
        'rmax': 7.0,
        'field_capacity': 0.31,
        'wilting_point': 0.12
    },
    'clay_loam': {
        'smax_base': 160.0,
        'rmax': 6.0,
        'field_capacity': 0.35,
        'wilting_point': 0.13
    },
    'silty_clay_loam': {
        'smax_base': 170.0,
        'rmax': 5.0,
        'field_capacity': 0.37,
        'wilting_point': 0.13
    },
    'sandy_clay': {
        'smax_base': 150.0,
        'rmax': 5.0,
        'field_capacity': 0.34,
        'wilting_point': 0.17
    },
    'silty_clay': {
        'smax_base': 180.0,
        'rmax': 4.0,
        'field_capacity': 0.39,
        'wilting_point': 0.17
    },
    'clay': {
        'smax_base': 190.0,
        'rmax': 3.0,
        'field_capacity': 0.40,
        'wilting_point': 0.20
    }
}

# ============================================================================
# IRRIGATION SYSTEM EFFICIENCIES
# ============================================================================

IRRIGATION_EFFICIENCY: Dict[str, float] = {
    'drip': 0.90,
    'micro_sprinkler': 0.85,
    'sprinkler': 0.75,
    'center_pivot': 0.80,
    'traditional': 0.60,
    'surface': 0.60,
    'furrow': 0.60,
    'border': 0.65,
    'basin': 0.70,
    'flooded': 0.60,
    'rainfed': 1.00
}

# ============================================================================
# REGIONAL CALIBRATION FACTORS
# ============================================================================
# Percolation calibration factors for different regions

CALIBRATION_FACTORS: Dict[str, float] = {
    'spain': 2.4,
    'mediterranean': 2.4,
    'temperate': 2.0,
    'tropical': 1.5,
    'arid': 3.0,
    'semi_arid': 2.5,
    'default': 2.4
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_crop_config(crop_name: str, management: str = 'irrigated') -> Dict:
    """
    Get complete crop configuration.
    
    Args:
        crop_name: Name of the crop (lowercase)
        management: 'irrigated' or 'rainfed'
    
    Returns:
        Dictionary with crop parameters
    """
    crop_lower = crop_name.lower()
    
    if crop_lower not in CROP_KC_DATABASE:
        raise ValueError(f"Crop '{crop_name}' not found in database")
    
    return {
        'name': crop_name,
        'monthly_kc': CROP_KC_DATABASE[crop_lower],
        'rooting_depth_max': ROOTING_DEPTH_DATABASE[crop_lower][management],
        'is_permanent': crop_lower in PERMANENT_CROPS,
        'landuse_kc': 0.5
    }


def get_soil_config(soil_type: str) -> Dict:
    """
    Get soil configuration.
    
    Args:
        soil_type: Soil texture type
    
    Returns:
        Dictionary with soil parameters
    """
    soil_lower = soil_type.lower()
    
    if soil_lower not in SOIL_TYPE_DATABASE:
        raise ValueError(f"Soil type '{soil_type}' not found in database")
    
    config = SOIL_TYPE_DATABASE[soil_lower].copy()
    config['reference_depth'] = 0.6
    config['initial_sm'] = config['smax_base'] * 0.5
    
    return config


def get_irrigation_config(system: str) -> Dict:
    """
    Get irrigation system configuration.
    
    Args:
        system: Irrigation system type
    
    Returns:
        Dictionary with irrigation parameters
    """
    system_lower = system.lower()
    
    if system_lower not in IRRIGATION_EFFICIENCY:
        raise ValueError(f"Irrigation system '{system}' not found in database")
    
    return {
        'system': system,
        'efficiency': IRRIGATION_EFFICIENCY[system_lower],
        'management': 'rainfed' if system_lower == 'rainfed' else 'irrigated'
    }


def get_calibration_factor(region: str = 'default') -> float:
    """
    Get regional calibration factor.
    
    Args:
        region: Region name
    
    Returns:
        Calibration factor
    """
    region_lower = region.lower()
    return CALIBRATION_FACTORS.get(region_lower, CALIBRATION_FACTORS['default'])


# ============================================================================
# EXAMPLE CONFIGURATIONS
# ============================================================================

def example_maize_irrigated():
    """Example configuration for irrigated maize."""
    return {
        'crop': get_crop_config('maize', 'irrigated'),
        'soil': get_soil_config('loam'),
        'irrigation': get_irrigation_config('drip'),
        'calibration_factor': get_calibration_factor('mediterranean')
    }


def example_wheat_rainfed():
    """Example configuration for rainfed wheat."""
    return {
        'crop': get_crop_config('wheat', 'rainfed'),
        'soil': get_soil_config('clay_loam'),
        'irrigation': get_irrigation_config('rainfed'),
        'calibration_factor': get_calibration_factor('temperate')
    }


def example_olive_irrigated():
    """Example configuration for irrigated olive."""
    return {
        'crop': get_crop_config('olive', 'irrigated'),
        'soil': get_soil_config('sandy_loam'),
        'irrigation': get_irrigation_config('drip'),
        'calibration_factor': get_calibration_factor('mediterranean')
    }


if __name__ == '__main__':
    # Print available crops
    print("Available crops:")
    for crop in sorted(CROP_KC_DATABASE.keys()):
        is_perm = " (permanent)" if crop in PERMANENT_CROPS else ""
        print(f"  - {crop}{is_perm}")
    
    print("\nAvailable soil types:")
    for soil in sorted(SOIL_TYPE_DATABASE.keys()):
        print(f"  - {soil}")
    
    print("\nAvailable irrigation systems:")
    for system in sorted(IRRIGATION_EFFICIENCY.keys()):
        eff = IRRIGATION_EFFICIENCY[system]
        print(f"  - {system}: {eff:.0%} efficiency")
    
    # Example usage
    print("\n" + "="*60)
    print("Example: Irrigated Maize Configuration")
    print("="*60)
    config = example_maize_irrigated()
    import json
    print(json.dumps(config, indent=2))
