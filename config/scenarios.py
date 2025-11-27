"""
Pre-defined climate scenarios from NGRIP ice core data.

Provides easy access to standard Late Pleistocene periods.
"""

from dataclasses import dataclass
from typing import Dict

@dataclass
class Scenario:
    """Climate scenario definition"""
    name: str
    eta: float
    eta_std: float
    age_range: tuple
    amplitude: float = 40000.0  # Default pulse amplitude [kg]

# Standard scenarios from NGRIP data
SCENARIOS: Dict[str, Scenario] = {
    'HS1': Scenario('Heinrich Stadial 1', eta=1.093, eta_std=0.096, age_range=(15.0, 18.0)),
    'LGM': Scenario('LGM Core', eta=1.242, eta_std=0.206, age_range=(20.0, 26.0)),
    'PreLGM': Scenario('Pre-LGM', eta=1.403, eta_std=0.204, age_range=(28.0, 32.0)),
    'YD': Scenario('Younger Dryas', eta=1.734, eta_std=0.225, age_range=(11.67, 13.0)),
    'BA': Scenario('Bølling-Allerød', eta=1.919, eta_std=0.242, age_range=(14.0, 15.0)),
}

def get_scenario(name: str) -> Scenario:
    """Get scenario by name or abbreviation"""
    return SCENARIOS[name.upper()]