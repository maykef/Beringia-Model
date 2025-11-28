"""
Climate forcing module: NGRIP ice core data integration.

Handles:
- Loading NGRIP Ca²⁺ time series
- Reading pre-computed climate irregularity parameter η from CSV
- Defining standard paleoclimate periods
- Generating stochastic resource pulses (Poisson process)

The climate irregularity parameter η modulates ecosystem dynamics:
- Higher η → fewer, more variable resource pulses
- Lower η → more frequent, predictable pulses

References:
- Rasmussen et al. (2014) - NGRIP GICC05 chronology
- This study - η calibration from Ca²⁺ variability

Author: [Your name]
Date: November 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.parameters import ClimateParams


@dataclass
class ClimatePeriod:
    """
    Named climate period with η statistics.

    Attributes:
        name: Period name (e.g., "Heinrich Stadial 1")
        age_range: (start_ka, end_ka) in thousands of years BP
        eta: Mean climate irregularity parameter
        eta_std: Standard deviation of η
        ca_mean: Mean Ca²⁺ concentration [ppb]
        ca_cv: Coefficient of variation of Ca²⁺
        n_datapoints: Number of data points in period
    """
    name: str
    age_range: Tuple[float, float]
    eta: float
    eta_std: float
    ca_mean: float
    ca_cv: float
    n_datapoints: int

    def __repr__(self) -> str:
        return (f"ClimatePeriod('{self.name}', {self.age_range[0]:.1f}-{self.age_range[1]:.1f} ka BP, "
                f"η={self.eta:.3f}±{self.eta_std:.3f}, n={self.n_datapoints})")


class NGRIPData:
    """
    NGRIP ice core data handler.

    Loads and processes Ca²⁺ concentration data with pre-computed
    climate irregularity parameter η for different periods.

    IMPORTANT: η values must be pre-computed in the CSV file.
    This class does NOT compute η on-the-fly.
    """

    # Standard Late Pleistocene periods (ka BP)
    STANDARD_PERIODS = {
        'Younger Dryas': (11.67, 13.0),
        'Bølling-Allerød': (14.0, 15.0),
        'Heinrich Stadial 1': (15.0, 18.0),
        'LGM Core': (20.0, 26.0),
        'Pre-LGM': (28.0, 32.0),
    }

    def __init__(self, csv_path: Optional[str] = None):
        """
        Load NGRIP data from CSV.

        Expected columns:
            - Age_ka: Age in thousands of years before present
            - Ca_ppb: Calcium concentration [ppb]
            - CV_Ca: Coefficient of variation (pre-computed)
            - eta: Climate irregularity parameter (pre-computed)

        Args:
            csv_path: Path to NGRIP CSV file. If None, expects data
                     to be loaded manually via load_data()
        """
        self.df: Optional[pd.DataFrame] = None
        self.periods_computed: bool = False
        self.period_cache: Dict[str, ClimatePeriod] = {}

        if csv_path is not None:
            self.load_data(csv_path)

    def load_data(self, csv_path: str) -> None:
        """
        Load NGRIP data from CSV file.

        Args:
            csv_path: Path to CSV file

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns missing
        """
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"NGRIP data file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self._validate_data()
        self.periods_computed = False
        self.period_cache = {}

        print(f"✓ Loaded NGRIP data: {len(self.df)} records from {csv_path}")

    def _validate_data(self) -> None:
        """
        Validate loaded data integrity.

        Raises:
            ValueError: If data validation fails
        """
        if self.df is None:
            raise ValueError("No data loaded")

        # Check required columns
        required_cols = ['Age_ka', 'Ca_ppb']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check for empty dataset
        if len(self.df) == 0:
            raise ValueError("Dataset is empty")

        # Check for negative ages
        if (self.df['Age_ka'] < 0).any():
            raise ValueError("Negative ages found in dataset")

        # Require η column to be present in CSV - DO NOT compute on-the-fly
        if 'eta' not in self.df.columns:
            raise ValueError(
                "η column not found in CSV. "
                "Pre-computed η values must be provided in the data file. "
                "Use the separate η computation script to generate these values."
            )

        # Drop rows with NaN η values
        n_before = len(self.df)
        self.df = self.df.dropna(subset=['eta'])
        n_dropped = n_before - len(self.df)
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} rows with NaN η values")

        # Validate η values
        if (self.df['eta'] <= 0).any():
            raise ValueError("Non-positive η values found")

        print(f"  η range: [{self.df['eta'].min():.3f}, {self.df['eta'].max():.3f}]")

    def get_period_eta(self, period_name: str) -> ClimatePeriod:
        """
        Extract η statistics for named climate period.

        Args:
            period_name: Name of period (must be in STANDARD_PERIODS)

        Returns:
            ClimatePeriod object with statistics

        Raises:
            ValueError: If period name unknown or no data in range
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Check cache first
        if period_name in self.period_cache:
            return self.period_cache[period_name]

        # Validate period name
        if period_name not in self.STANDARD_PERIODS:
            raise ValueError(
                f"Unknown period: {period_name}. "
                f"Available: {list(self.STANDARD_PERIODS.keys())}"
            )

        # Extract data for period
        start, end = self.STANDARD_PERIODS[period_name]
        mask = (self.df['Age_ka'] >= start) & (self.df['Age_ka'] <= end)
        period_data = self.df[mask]

        if len(period_data) == 0:
            raise ValueError(
                f"No data for {period_name} ({start:.1f}-{end:.1f} ka BP). "
                f"Data range: {self.df['Age_ka'].min():.1f}-{self.df['Age_ka'].max():.1f} ka BP"
            )

        # Compute statistics
        period = ClimatePeriod(
            name=period_name,
            age_range=(start, end),
            eta=float(period_data['eta'].mean()),
            eta_std=float(period_data['eta'].std()),
            ca_mean=float(period_data['Ca_ppb'].mean()),
            ca_cv=float(period_data['CV_Ca'].mean()) if 'CV_Ca' in period_data.columns else 0.0,
            n_datapoints=len(period_data)
        )

        # Cache result
        self.period_cache[period_name] = period

        return period

    def get_all_periods(self) -> Dict[str, ClimatePeriod]:
        """
        Get η statistics for all defined periods.

        Returns:
            Dictionary mapping period names to ClimatePeriod objects
        """
        return {name: self.get_period_eta(name)
                for name in self.STANDARD_PERIODS.keys()}

    def get_eta_timeseries(self,
                           age_min: Optional[float] = None,
                           age_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get continuous η time series.

        Args:
            age_min: Minimum age [ka BP] (None = full range)
            age_max: Maximum age [ka BP] (None = full range)

        Returns:
            (ages, eta_values) arrays
        """
        if self.df is None:
            raise ValueError("No data loaded")

        # Apply age filter if specified
        if age_min is not None or age_max is not None:
            mask = np.ones(len(self.df), dtype=bool)
            if age_min is not None:
                mask &= (self.df['Age_ka'] >= age_min)
            if age_max is not None:
                mask &= (self.df['Age_ka'] <= age_max)
            data = self.df[mask]
        else:
            data = self.df

        return data['Age_ka'].values, data['eta'].values

    def summary(self) -> str:
        """
        Generate summary of all climate periods.

        Returns:
            Formatted summary string
        """
        if self.df is None:
            return "No data loaded"

        lines = [
            "=" * 80,
            "NGRIP CLIMATE PERIODS SUMMARY",
            "=" * 80,
            f"Data range: {self.df['Age_ka'].min():.1f}-{self.df['Age_ka'].max():.1f} ka BP",
            f"Total records: {len(self.df):,}",
            "",
            f"{'Period':<25} {'Age (ka BP)':<15} {'η':<12} {'Ca²⁺ (ppb)':<12} {'n':<8}",
            "-" * 80,
        ]

        try:
            periods = self.get_all_periods()
            for name, period in periods.items():
                age_str = f"{period.age_range[0]:.1f}-{period.age_range[1]:.1f}"
                eta_str = f"{period.eta:.3f}±{period.eta_std:.3f}"
                ca_str = f"{period.ca_mean:.1f}"
                lines.append(
                    f"{name:<25} {age_str:<15} {eta_str:<12} {ca_str:<12} {period.n_datapoints:<8}"
                )
        except Exception as e:
            lines.append(f"Error computing periods: {e}")

        lines.append("=" * 80)
        return "\n".join(lines)


class ResourcePulseGenerator:
    """
    Stochastic resource pulse generator.

    Implements Poisson point process with Gaussian amplitude
    modulation, climate-modulated by η parameter.

    Pulse dynamics:
    - Frequency: λ_eff = λ_base / η  (higher η → fewer pulses)
    - Variance: σ_eff = σ_base × η   (higher η → more variable)

    Physical interpretation:
    Pulses represent seasonal resource windfalls (e.g., caribou migrations,
    salmon runs, seasonal plant growth) that are less frequent and more
    unpredictable under irregular climate (high η).
    """

    def __init__(self,
                 climate_params: ClimateParams,
                 rng: Optional[np.random.RandomState] = None):
        """
        Initialize pulse generator.

        Args:
            climate_params: ClimateParams instance with λ, σ, duration
            rng: Random state for reproducibility (None = new random state)
        """
        self.params = climate_params
        self.rng = rng if rng is not None else np.random.RandomState()

        # Track pulse history
        self.pulse_times: List[float] = []
        self.pulse_amplitudes: List[float] = []

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset random state and pulse history.

        Args:
            seed: Random seed (None = don't reset seed)
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.pulse_times = []
        self.pulse_amplitudes = []

    def generate(self,
                 t: float,
                 dt: float,
                 eta: float,
                 amplitude: float) -> float:
        """
        Generate resource pulse at time t.

        Uses Poisson arrival process with climate-modulated rate.
        When pulse occurs, amplitude drawn from Gaussian with
        climate-modulated variance.

        Args:
            t: Current time [years]
            dt: Timestep [years]
            eta: Climate irregularity parameter [dimensionless]
            amplitude: Base pulse amplitude [kg]

        Returns:
            Pulse rate [kg/year] (0 if no pulse, amplitude/duration if pulse)
        """
        # Effective rates modulated by η
        lambda_eff = self.params.effective_frequency(eta)
        sigma_eff = self.params.effective_variance(eta)

        # Poisson arrival probability in timestep dt
        # P(pulse in dt) = λ_eff × dt for small dt
        arrival_prob = lambda_eff * dt

        if self.rng.random() < arrival_prob:
            # Pulse occurs - sample amplitude from Gaussian
            pulse_amp = amplitude * (1.0 + sigma_eff * self.rng.randn())
            pulse_amp = max(0.0, pulse_amp)  # Ensure non-negative

            # Record pulse
            self.pulse_times.append(t)
            self.pulse_amplitudes.append(pulse_amp)

            # Convert to rate (pulse distributed over duration)
            return pulse_amp / self.params.pulse_duration

        return 0.0

    def get_statistics(self, t_start: float, t_end: float) -> Dict[str, float]:
        """
        Compute pulse statistics over time interval.

        Args:
            t_start: Start time [years]
            t_end: End time [years]

        Returns:
            Dictionary with statistics:
                - n_pulses: Number of pulses
                - mean_amplitude: Mean pulse amplitude [kg]
                - std_amplitude: Std pulse amplitude [kg]
                - frequency: Realized frequency [1/year]
        """
        # Filter pulses in time window
        mask = [(t >= t_start and t <= t_end)
                for t in self.pulse_times]
        amplitudes = [a for a, m in zip(self.pulse_amplitudes, mask) if m]

        n_pulses = len(amplitudes)
        duration = t_end - t_start

        if n_pulses == 0:
            return {
                'n_pulses': 0,
                'mean_amplitude': 0.0,
                'std_amplitude': 0.0,
                'frequency': 0.0
            }

        return {
            'n_pulses': n_pulses,
            'mean_amplitude': float(np.mean(amplitudes)),
            'std_amplitude': float(np.std(amplitudes)),
            'frequency': n_pulses / duration
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing climate forcing module...\n")

    # Test 1: Load NGRIP data
    print("=" * 80)
    print("TEST 1: NGRIP Data Loading (REAL DATA)")
    print("=" * 80)

    # Use real NGRIP data
    data_path = Path(__file__).parent.parent / 'data' / 'NGRIP_eta_real_31_32ka.csv'

    if not data_path.exists():
        print(f"ERROR: NGRIP data not found at {data_path}")
        print("Please ensure NGRIP_eta_real_31_32ka.csv is in data/ directory")
        import sys
        sys.exit(1)

    ngrip = NGRIPData(str(data_path))
    print(ngrip.summary())

    # Test 2: Extract period statistics
    print("\n" + "=" * 80)
    print("TEST 2: Period Statistics")
    print("=" * 80)

    hs1 = ngrip.get_period_eta('Heinrich Stadial 1')
    print(f"\n{hs1}")
    print(f"  Expected pulses/year: {1.2 / hs1.eta:.2f}")
    print(f"  Expected CV: {0.20 * hs1.eta:.2%}")

    lgm = ngrip.get_period_eta('LGM Core')
    print(f"\n{lgm}")
    print(f"  Expected pulses/year: {1.2 / lgm.eta:.2f}")
    print(f"  Expected CV: {0.20 * lgm.eta:.2%}")

    # Test 3: Generate stochastic pulses
    print("\n" + "=" * 80)
    print("TEST 3: Stochastic Pulse Generation")
    print("=" * 80)

    pulse_gen = ResourcePulseGenerator(
        ClimateParams(),
        rng=np.random.RandomState(42)
    )

    # Generate 100 years of pulses for HS1
    dt = 0.05  # years
    times = np.arange(0, 100, dt)
    pulses_hs1 = []

    for t in times:
        pulse = pulse_gen.generate(t, dt, hs1.eta, 40000.0)
        pulses_hs1.append(pulse)

    stats_hs1 = pulse_gen.get_statistics(0, 100)
    print(f"\nHeinrich Stadial 1 (η={hs1.eta:.3f}):")
    print(f"  Number of pulses: {stats_hs1['n_pulses']}")
    print(f"  Realized frequency: {stats_hs1['frequency']:.2f} /year")
    print(f"  Expected frequency: {1.2 / hs1.eta:.2f} /year")
    print(f"  Mean amplitude: {stats_hs1['mean_amplitude']/1000:.1f} tonnes")
    print(f"  Std amplitude: {stats_hs1['std_amplitude']/1000:.1f} tonnes")

    # Generate for LGM
    pulse_gen.reset(seed=42)
    pulses_lgm = []

    for t in times:
        pulse = pulse_gen.generate(t, dt, lgm.eta, 40000.0)
        pulses_lgm.append(pulse)

    stats_lgm = pulse_gen.get_statistics(0, 100)
    print(f"\nLGM Core (η={lgm.eta:.3f}):")
    print(f"  Number of pulses: {stats_lgm['n_pulses']}")
    print(f"  Realized frequency: {stats_lgm['frequency']:.2f} /year")
    print(f"  Expected frequency: {1.2 / lgm.eta:.2f} /year")
    print(f"  Mean amplitude: {stats_lgm['mean_amplitude']/1000:.1f} tonnes")
    print(f"  Std amplitude: {stats_lgm['std_amplitude']/1000:.1f} tonnes")

    print("\n" + "=" * 80)
    print("✓ All tests passed")
    print("=" * 80)
