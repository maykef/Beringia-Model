"""
Pre-defined climate scenarios from NGRIP ice core data.

Provides easy access to standard Late Pleistocene periods.
η values are loaded dynamically from the CSV file.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path


@dataclass
class Scenario:
    """Climate scenario definition"""
    name: str
    eta: float
    eta_std: float
    age_range: tuple
    amplitude: float = 40000.0  # Default pulse amplitude [kg]


# Age ranges for standard periods (ka BP)
PERIOD_DEFINITIONS = {
    'HS1': ('Heinrich Stadial 1', (15.0, 18.0)),
    'LGM': ('LGM Core', (20.0, 26.0)),
    'PreLGM': ('Pre-LGM', (28.0, 32.0)),
    'YD': ('Younger Dryas', (11.67, 13.0)),
    'BA': ('Bølling-Allerød', (14.0, 15.0)),
}

# Cache for loaded scenarios
_SCENARIOS_CACHE: Optional[Dict[str, Scenario]] = None


def _load_scenarios_from_csv() -> Dict[str, Scenario]:
    """Load η values dynamically from NGRIP CSV file."""
    import pandas as pd

    # Find CSV file
    data_path = Path(__file__).parent.parent / 'data' / 'NGRIP_eta_real_31_32ka.csv'

    if not data_path.exists():
        raise FileNotFoundError(f"NGRIP data not found at {data_path}")

    df = pd.read_csv(data_path)

    # Drop rows with NaN eta values
    df = df.dropna(subset=['eta'])

    scenarios = {}
    for key, (name, age_range) in PERIOD_DEFINITIONS.items():
        start, end = age_range
        mask = (df['Age_ka'] >= start) & (df['Age_ka'] <= end)
        period_data = df[mask]

        if len(period_data) == 0:
            print(f"Warning: No data for {name} ({start}-{end} ka BP)")
            continue

        eta_mean = float(period_data['eta'].mean())
        eta_std = float(period_data['eta'].std())

        scenarios[key] = Scenario(
            name=name,
            eta=eta_mean,
            eta_std=eta_std,
            age_range=age_range
        )

    return scenarios


def get_scenarios() -> Dict[str, Scenario]:
    """Get all scenarios, loading from CSV if needed."""
    global _SCENARIOS_CACHE

    if _SCENARIOS_CACHE is None:
        _SCENARIOS_CACHE = _load_scenarios_from_csv()

    return _SCENARIOS_CACHE


def get_scenario(name: str) -> Scenario:
    """Get scenario by name or abbreviation."""
    scenarios = get_scenarios()
    return scenarios[name.upper()]


# For backwards compatibility with code that imports SCENARIOS directly
def __getattr__(name):
    if name == 'SCENARIOS':
        return get_scenarios()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
