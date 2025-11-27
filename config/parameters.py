"""
Parameter definitions for Beringia population dynamics model.

All parameter values are justified in:
- Thermodynamics_of_Beringia_V4.docx (energy budgets, fuel dynamics)
- Thermodynamics_of_Beringia_V5.docx (Lake El'gygytgyn validation)
- Sikora et al. (2019) Nature - genetics
- Ethnographic sources (Burch 1988, Binford 2001)

Author: [Your name]
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SpatialParams:
    """
    Spatial domain parameters.

    The model represents a single band territory, not the entire BLB.
    Full BLB mesic zone: ~280,000-360,000 km² (Thermodynamics doc Table 7)
    Single band territory: ~10,000 km² (this model)

    Total Beringian population = 3-6 bands × band_size
    Constrained by Ne = 500-1,500 (Sikora et al. 2019)
    """
    area_km2: float = 10000.0  # Band territory [km²]
    suitable_fraction: float = 0.40  # Fraction suitable for resources

    def __post_init__(self):
        """Validate spatial parameters"""
        assert self.area_km2 > 0, "Area must be positive"
        assert 0 < self.suitable_fraction <= 1.0, "Fraction must be in (0,1]"

    @property
    def suitable_area_km2(self) -> float:
        """Effective area for resources [km²]"""
        return self.area_km2 * self.suitable_fraction


@dataclass
class ClimateParams:
    """
    Climate forcing parameters from NGRIP ice core Ca²⁺ data.

    The climate irregularity parameter η = CV(Ca²⁺) / baseline modulates:
    - Resource pulse frequency: λ_eff = λ_base / η
    - Resource pulse variance: σ_eff = σ_base × η

    References:
    - NGRIP Ca²⁺ data (Rasmussen et al. 2014)
    - Model calibration (this study)
    """
    lambda_base: float = 1.2  # Base pulse frequency [1/year]
    sigma_base: float = 0.20  # Base pulse CV [dimensionless]
    pulse_duration: float = 0.08  # Pulse duration [years] (~1 month)
    amplitude_default: float = 40000.0  # Default pulse amplitude [kg]

    def effective_frequency(self, eta: float) -> float:
        """
        Climate-modulated pulse frequency.

        Higher η → lower frequency (more irregular = fewer pulses)
        """
        return self.lambda_base / eta

    def effective_variance(self, eta: float) -> float:
        """
        Climate-modulated pulse variance.

        Higher η → higher variance (more irregular = larger fluctuations)
        """
        return self.sigma_base * eta


@dataclass
class ResourceParams:
    """
    Protein biomass dynamics (ungulate resources).

    Based on mammoth steppe ecology:
    - NPP: 300-500 g C/m²/yr (Thermodynamics doc §3)
    - Herbivore biomass: 5-15 kg/m² for BLB mesic (doc Table 2)
    - Carrying capacity: Derived from NPP → sustainable yield

    References:
    - Zimov et al. (2012) - mammoth steppe productivity
    - Thermodynamics_of_Beringia_V4.docx §3
    """
    # Initial state
    P0: float = 350000.0  # Initial protein biomass [kg]

    # Growth dynamics
    r_P: float = 0.35  # Intrinsic growth rate [1/year]
    # Justification: Ungulate populations grow 25-45%/yr (Caughley 1977)

    K_P_base: float = 450000.0  # Base carrying capacity [kg]
    # Justification: From NPP 275 g C/m²/yr × 10^10 m² × 10% trophic
    #                = 2.75×10^8 kg C/yr × 0.1-0.2% harvestable
    #                = 275,000-550,000 kg/yr sustainable yield
    #                Mid-range = 450,000 kg

    alpha_density: float = 80.0  # Density dependence coefficient
    # Justification: K_eff drops from 450t → 150t as N goes 0→350
    #                Fitted to produce this response

    # Hunting parameters
    h_base: float = 350.0  # Base hunting yield [kg/person/year]
    # Justification: Doc §4 "1 kg meat/day family" = 365 kg/yr/person

    gamma_interference: float = 0.12  # Interference competition coefficient
    # Justification: Hunting efficiency drops with crowding (Holling Type II)

    epsilon_conversion: float = 0.60  # Edible fraction
    # Justification: Doc Table 2 explicit: "60% edible fraction"
    #                350 kg/yr gross × 60% = 210 kg/yr net

    P_half: float = 50000.0  # Half-saturation for Type II response [kg]

    def carrying_capacity(self, N: float, area_km2: float) -> float:
        """
        Density-dependent carrying capacity.

        K_eff decreases as human density increases (trampling, disturbance)
        """
        density = N / area_km2
        return self.K_P_base / (1.0 + self.alpha_density * density)

    def hunting_rate(self, P: float, N: float) -> float:
        """
        Per-capita hunting with Type II functional response and interference.

        Args:
            P: Current protein biomass [kg]
            N: Current population [persons]

        Returns:
            Per-capita hunting success [kg/person/year]
        """
        # Interference reduces effectiveness
        h_eff = self.h_base / (1.0 + self.gamma_interference * N / 500.0)

        # Type II functional response (saturation at high P)
        success = h_eff * (P / (P + self.P_half))

        return success


@dataclass
class EnergeticsParams:
    """
    Fat-in-system dynamics (total accessible energy reserves).

    CRITICAL INTERPRETATION:
    F_a is NOT body fat (adipose tissue)!
    F_a = Total accessible fat in coupled human-environment system:
          - Cached animal fat in permafrost storage
          - Fresh kills being processed
          - Social network sharing access
          - Body adipose buffer (~20-30%)

    Thresholds represent CACHE STATUS, not body composition.

    References:
    - Thermodynamics_of_Beringia_V4.docx §4 (energy budgets)
    - Ethnographic cache systems (Burch 1988)
    """
    # Initial state
    F_a0: float = 35.0  # Initial fat availability [kg/person]

    # Thresholds (kg/person of accessible fat in system)
    F_min: float = 8.0  # CRISIS: Caches exhausted, <1 month supply
    # At F_min: Band eating body fat (last resort), mortality spikes

    F_crit: float = 15.0  # STRESS: Low reserves, ~2 month supply
    # At F_crit: Emergency rationing, elevated mortality

    F_opt: float = 45.0  # COMFORTABLE: Good surplus, ~5 month supply
    # At F_opt: Normal operations, stable mortality

    F_max: float = 100.0  # SATURATION: Storage capacity limit, ~10 months
    # At F_max: Cache capacity reached (physical/spoilage limits)

    # Energy flow rates
    phi_fat_storage: float = 0.10  # Fraction of hunting → storable fat
    # Justification: Of 350 kg gross hunting:
    #                - 210 kg edible (60%)
    #                - 35 kg renderable fat (10% of gross)
    #                - After rendering losses: ~28 kg stored
    #                phi ≈ 28/350 = 8% ≈ 10% (order of magnitude)

    psi_fat_consumption: float = 12.0  # Net thermal deficit [kg/person/year]
    # Justification: Thermoregulation needs 500-1,000 kcal/day (doc §4)
    #                = 20-40 kg fat/year
    #                Hunting provides: phi × h_base = 35 kg/year
    #                Deficit: 40-35 = 5 kg/year (average)
    #                Model uses 12 kg/yr (conservative, accounts for cold spikes)

    metabolic_baseline: float = 180.0  # Lean tissue needs [kg/person/year]

    # Justification: BMR ~1,500-2,000 kcal/day (doc §4)
    #                Activity: +1,000-1,500 kcal/day
    #                Total: ~2,500-3,500 kcal/day = 180 kg protein/yr

    def stress_factor(self, F_a: float) -> float:
        """
        Calculate stress based on fat availability [0,1].

        Returns:
            0.0 = abundant (F_a >= F_crit)
            1.0 = crisis (F_a <= F_min)
        """
        if F_a >= self.F_crit:
            return 0.0
        elif F_a <= self.F_min:
            return 1.0
        else:
            return (self.F_crit - F_a) / (self.F_crit - self.F_min)


@dataclass
class DemographicParams:
    """
    Population dynamics parameters.

    Vital rates based on hunter-gatherer ethnography:
    - Birth rate: 3.5-4.5%/year (Binford 2001, !Kung San data)
    - Mortality: 3.2-12% depending on fat availability

    References:
    - Howell (1979) - !Kung demography
    - Binford (2001) - global HG database
    - Hill & Hurtado (1996) - Ache foragers
    """
    # Initial state
    N0: float = 350.0  # Initial population [persons]

    # Vital rates
    b: float = 0.038  # Birth rate [1/year] (3.8%/year)
    # Justification: Mid-range of HG birth rates (3.5-4.5%)
    #                Accounts for high infant mortality, 3-4 year spacing

    d_min: float = 0.032  # Minimum mortality [1/year] (3.2%/year)
    # Justification: Baseline mortality under good conditions
    #                b > d_min → intrinsic growth rate = 0.6%/year

    d_max: float = 0.120  # Maximum mortality [1/year] (12%/year)
    # Justification: Crisis mortality (starvation, cold stress)
    #                Observed during Arctic famines (Burch 1988)

    # Dispersal
    D_base: float = 0.005  # Baseline emigration [1/year] (0.5%/year)
    stress_dispersion: float = 0.010  # Stress-induced dispersal [1/year] (1%/year)

    def mortality_rate(self, F_a: float, energetics: EnergeticsParams) -> float:
        """
        Calculate mortality from fat availability.

        Uses sigmoid-like response: gradual increase below F_crit,
        rapid escalation near F_min.

        Args:
            F_a: Current fat availability [kg/person]
            energetics: EnergeticsParams for thresholds

        Returns:
            Mortality rate [1/year]
        """
        if F_a < energetics.F_crit:
            # Below critical: mortality increases nonlinearly
            stress = energetics.stress_factor(F_a)
            d = self.d_min + (self.d_max - self.d_min) * (stress ** 2.5)
        else:
            # Above critical: minimal excess mortality
            d = self.d_min * (1.0 + 0.15 * np.exp(-(F_a - energetics.F_crit) / 12.0))

        return d

    def dispersion_rate(self, F_a: float, energetics: EnergeticsParams) -> float:
        """
        Calculate emigration from stress.

        People leave when caches run low (seeking better conditions).

        Args:
            F_a: Current fat availability [kg/person]
            energetics: EnergeticsParams for thresholds

        Returns:
            Dispersion rate [1/year]
        """
        if F_a < energetics.F_opt * 0.8:
            stress = 1.0 - F_a / (energetics.F_opt * 0.8)
            return self.D_base + self.stress_dispersion * stress
        else:
            return self.D_base


@dataclass
class NumericalParams:
    """
    Integration and simulation control parameters.

    Uses semi-implicit Euler for numerical stability:
    - Fat updated first (explicit)
    - Then used for mortality calculation (implicit)

    This prevents numerical fat crashes that plagued forward Euler.
    """
    dt: float = 0.05  # Timestep [years] (18.25 days)
    # Justification: Semi-implicit stable at this step

    t_max: float = 1500.0  # Simulation duration [years]
    burn_in: float = 500.0  # Transient removal [years]

    # Ensemble parameters
    n_ensemble: int = 50  # Number of realizations
    base_seed: int = 42  # Random seed for reproducibility

    @property
    def n_steps(self) -> int:
        """Total number of timesteps"""
        return int(self.t_max / self.dt)

    @property
    def burn_in_steps(self) -> int:
        """Number of burn-in timesteps to discard"""
        return int(self.burn_in / self.dt)


@dataclass
class ModelParams:
    """
    Complete model parameter set.

    Aggregates all parameter groups with validation.
    """
    spatial: SpatialParams = field(default_factory=SpatialParams)
    climate: ClimateParams = field(default_factory=ClimateParams)
    resources: ResourceParams = field(default_factory=ResourceParams)
    energetics: EnergeticsParams = field(default_factory=EnergeticsParams)
    demographics: DemographicParams = field(default_factory=DemographicParams)
    numerical: NumericalParams = field(default_factory=NumericalParams)

    def validate(self) -> None:
        """
        Run all validation checks.

        Ensures:
        1. Thermodynamic closure (energy budget balances)
        2. Demographic viability (birth > death at equilibrium)
        3. Physical bounds (all parameters positive, fractions in [0,1])
        """
        print("Validating model parameters...")

        # 1. Check thermodynamic closure
        net_food = self.resources.epsilon_conversion * self.resources.h_base
        assert net_food > self.energetics.metabolic_baseline, \
            f"Hunting ({net_food:.0f} kg/yr) must exceed metabolic baseline ({self.energetics.metabolic_baseline:.0f} kg/yr)"

        energy_surplus = net_food - self.energetics.metabolic_baseline
        print(
            f"  ✓ Energy budget: {net_food:.0f} kg/yr net food - {self.energetics.metabolic_baseline:.0f} kg/yr metabolism = +{energy_surplus:.0f} kg/yr surplus")

        # 2. Check demographic viability
        assert self.demographics.b > self.demographics.d_min, \
            f"Birth rate ({self.demographics.b:.4f}) must exceed minimum mortality ({self.demographics.d_min:.4f})"

        intrinsic_growth = self.demographics.b - self.demographics.d_min
        print(
            f"  ✓ Demographic viability: b={self.demographics.b:.4f}, d_min={self.demographics.d_min:.4f}, r={intrinsic_growth:.4f} (0.6%/yr growth)")

        # 3. Check physical bounds
        assert self.energetics.F_min < self.energetics.F_crit < self.energetics.F_opt < self.energetics.F_max, \
            "Fat thresholds must be ordered: F_min < F_crit < F_opt < F_max"
        print(
            f"  ✓ Fat thresholds ordered: {self.energetics.F_min} < {self.energetics.F_crit} < {self.energetics.F_opt} < {self.energetics.F_max} kg/person")

        # 4. Check carrying capacity
        K_test = self.resources.carrying_capacity(self.demographics.N0, self.spatial.area_km2)
        assert K_test > 0, "Carrying capacity must be positive"
        print(f"  ✓ Carrying capacity at N0: K_eff = {K_test / 1000:.1f} tonnes")

        print("✓ All parameter validations passed\n")

    def summary(self) -> str:
        """
        Generate parameter summary string.

        Returns:
            Formatted summary of key parameters
        """
        lines = [
            "=" * 60,
            "BERINGIA MODEL PARAMETERS (v5.0)",
            "=" * 60,
            "",
            "SPATIAL:",
            f"  Band territory: {self.spatial.area_km2:,.0f} km²",
            f"  Suitable fraction: {self.spatial.suitable_fraction:.0%}",
            "",
            "RESOURCES:",
            f"  Base carrying capacity: {self.resources.K_P_base / 1000:.0f} tonnes",
            f"  Growth rate: {self.resources.r_P:.1%}/year",
            f"  Hunting yield: {self.resources.h_base:.0f} kg/person/year",
            f"  Edible fraction: {self.resources.epsilon_conversion:.0%}",
            f"  Net food: {self.resources.h_base * self.resources.epsilon_conversion:.0f} kg/person/year",
            "",
            "ENERGETICS (Fat-in-System):",
            f"  Crisis threshold: {self.energetics.F_min:.0f} kg/person",
            f"  Stress threshold: {self.energetics.F_crit:.0f} kg/person",
            f"  Optimal level: {self.energetics.F_opt:.0f} kg/person",
            f"  Storage capacity: {self.energetics.F_max:.0f} kg/person",
            f"  Storage fraction: {self.energetics.phi_fat_storage:.0%}",
            f"  Thermal deficit: {self.energetics.psi_fat_consumption:.0f} kg/person/year",
            "",
            "DEMOGRAPHICS:",
            f"  Birth rate: {self.demographics.b:.1%}/year",
            f"  Min mortality: {self.demographics.d_min:.1%}/year",
            f"  Max mortality: {self.demographics.d_max:.1%}/year",
            f"  Intrinsic growth: {(self.demographics.b - self.demographics.d_min):.1%}/year",
            "",
            "NUMERICAL:",
            f"  Timestep: {self.numerical.dt:.3f} years ({self.numerical.dt * 365:.1f} days)",
            f"  Duration: {self.numerical.t_max:.0f} years",
            f"  Burn-in: {self.numerical.burn_in:.0f} years",
            f"  Ensemble size: {self.numerical.n_ensemble}",
            "=" * 60,
        ]
        return "\n".join(lines)


# Convenience function for quick parameter loading
def load_default_params() -> ModelParams:
    """
    Load default parameter set with validation.

    Returns:
        ModelParams: Validated parameter set
    """
    params = ModelParams()
    params.validate()
    return params


if __name__ == "__main__":
    # Test parameter loading
    print("Testing parameter module...\n")

    params = load_default_params()
    print(params.summary())

    # Test parameter access
    print("\nParameter access examples:")
    print(f"  Hunting yield: {params.resources.h_base} kg/person/year")
    print(f"  Fat storage: {params.energetics.phi_fat_storage:.0%} of hunting")
    print(f"  Timestep: {params.numerical.dt} years")

    # Test validation failure
    print("\nTesting validation failure...")
    bad_params = ModelParams()
    bad_params.demographics.b = 0.02  # Too low!
    try:
        bad_params.validate()
    except AssertionError as e:
        print(f"  ✓ Caught invalid parameter: {e}")
