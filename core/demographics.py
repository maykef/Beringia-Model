"""
Demographics module: Population dynamics (births, deaths, dispersal).

Implements:
- Vital rates (births, deaths)
- Stress-dependent mortality
- Dispersal/emigration
- Population change dynamics

Mortality increases nonlinearly when fat availability drops below
critical threshold (cache depletion → hypothermia, starvation).

References:
- Howell (1979) - !Kung San demography
- Binford (2001) - Global hunter-gatherer database
- Hill & Hurtado (1996) - Ache foragers
- Burch (1988) - Arctic population crises

Author: [Your name]
Date: November 2025
"""

import numpy as np
from typing import Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config.parameters import DemographicParams, EnergeticsParams


class PopulationDynamics:
    """
    Population dynamics with stress-dependent mortality.

    State variable:
        N: Population size [persons]

    Dynamics:
        dN/dt = N × (b - d(F_a) - D(F_a))

    Where:
        b = constant birth rate
        d(F_a) = stress-dependent mortality
        D(F_a) = stress-dependent dispersal
    """

    def __init__(self,
                 demographic_params: DemographicParams,
                 energetics_params: EnergeticsParams):
        """
        Initialize population dynamics.

        Args:
            demographic_params: DemographicParams instance
            energetics_params: EnergeticsParams instance (for stress calculation)
        """
        self.params = demographic_params
        self.energetics = energetics_params

    def mortality_rate(self, F_a: float) -> float:
        """
        Calculate mortality rate from fat availability.

        Uses nonlinear response with power law:
            d(F_a) = d_min + (d_max - d_min) × stress^2.5

        Where stress = (F_crit - F_a) / (F_crit - F_min) for F_a < F_crit

        Power of 2.5 creates:
        - Gradual increase as F_a drops below F_crit
        - Rapid escalation near F_min (crisis)
        - Smooth transition (no discontinuities)

        Physical mechanisms:
        - Hypothermia (inadequate fat burning)
        - Starvation (protein-energy malnutrition)
        - Disease susceptibility (immune suppression)
        - Accidents (weakness, impaired judgment)

        Args:
            F_a: Current fat availability [kg/person]

        Returns:
            Mortality rate [1/year]
        """
        if F_a < self.energetics.F_crit:
            # Below critical threshold - stress-dependent mortality
            stress = self.energetics.stress_factor(F_a)
            d = self.params.d_min + (self.params.d_max - self.params.d_min) * (stress ** 2.5)
        else:
            # Above critical - baseline mortality with slight health benefit
            excess = F_a - self.energetics.F_crit
            health_benefit = 0.15 * np.exp(-excess / 12.0)
            d = self.params.d_min * (1.0 + health_benefit)

        return d

    def dispersion_rate(self, F_a: float) -> float:
        """
        Calculate emigration rate from stress.

        People leave when caches run low (seeking better conditions).

        Formula:
            D(F_a) = D_base                           if F_a >= 0.8 × F_opt
                   = D_base + D_stress × (1 - F_a/threshold)  if F_a < threshold

        Physical interpretation:
        - D_base: Normal movement (exploration, mate-seeking, conflict)
        - D_stress: Stress-induced dispersal (resource scarcity, mortality risk)

        Threshold at 0.8 × F_opt means dispersal begins BEFORE crisis,
        allowing proactive population adjustment.

        Args:
            F_a: Current fat availability [kg/person]

        Returns:
            Dispersion rate [1/year]
        """
        threshold = self.energetics.F_opt * 0.8

        if F_a < threshold:
            stress_level = 1.0 - (F_a / threshold)
            D = self.params.D_base + self.params.stress_dispersion * stress_level
        else:
            D = self.params.D_base

        return D

    def birth_rate(self) -> float:
        """
        Return constant birth rate.

        In this model, births are NOT stress-dependent because:
        1. Birth intervals span years (lag effect)
        2. Mortality acts faster than fertility suppression
        3. Empirical data shows births relatively stable in HG populations

        Could be extended to include:
        - Age structure effects
        - Nutritional fertility suppression
        - Postpartum amenorrhea duration

        Returns:
            Birth rate [1/year]
        """
        return self.params.b

    def net_growth_rate(self, F_a: float) -> float:
        """
        Calculate net population growth rate.

        Formula:
            r(F_a) = b - d(F_a) - D(F_a)

        Positive r → population growth
        Negative r → population decline

        Args:
            F_a: Current fat availability [kg/person]

        Returns:
            Net growth rate [1/year]
        """
        b = self.birth_rate()
        d = self.mortality_rate(F_a)
        D = self.dispersion_rate(F_a)

        r = b - d - D

        return r

    def derivative(self, N: float, F_a: float) -> float:
        """
        Complete dN/dt.

        Formula:
            dN/dt = N × r(F_a)
                  = N × (b - d(F_a) - D(F_a))

        Args:
            N: Current population [persons]
            F_a: Current fat availability [kg/person]

        Returns:
            dN/dt [persons/year]
        """
        if N <= 0:
            return 0.0

        r = self.net_growth_rate(F_a)
        dN_dt = N * r

        return dN_dt

    def time_to_extinction(self, N: float, F_a: float) -> float:
        """
        Estimate time to extinction at current mortality rate.

        Uses exponential decline approximation:
            N(t) = N0 × exp(r × t)

        Solving for N(t) = 1 (extinction):
            t_ext = ln(1/N0) / r = -ln(N0) / r

        Args:
            N: Current population [persons]
            F_a: Current fat availability [kg/person]

        Returns:
            Time to extinction [years] (np.inf if growing)
        """
        r = self.net_growth_rate(F_a)

        if r >= 0:
            return np.inf  # Population stable or growing

        if N <= 1:
            return 0.0  # Already extinct

        t_ext = -np.log(N) / r

        return t_ext

    def equilibrium_population(self, F_a: float) -> float:
        """
        Calculate equilibrium population for given fat availability.

        At equilibrium: dN/dt = 0 → r(F_a) = 0

        Useful for:
        - Understanding carrying capacity
        - Validation
        - Initial conditions

        Args:
            F_a: Fat availability [kg/person]

        Returns:
            Equilibrium population [persons] (0 if r < 0 at all N)
        """
        r = self.net_growth_rate(F_a)

        if r > 0:
            return np.inf  # Population would grow without resource limits
        elif r < 0:
            return 0.0  # Population declining, no stable equilibrium
        else:
            return self.params.N0  # Exactly at replacement

    def summary(self, N: float, F_a: float) -> dict:
        """
        Compute all demographic metrics for current state.

        Args:
            N: Current population [persons]
            F_a: Current fat availability [kg/person]

        Returns:
            Dictionary with metrics
        """
        b = self.birth_rate()
        d = self.mortality_rate(F_a)
        D = self.dispersion_rate(F_a)
        r = self.net_growth_rate(F_a)
        dN_dt = self.derivative(N, F_a)

        # Time to extinction (if declining)
        if r < 0 and N > 1:
            t_ext = self.time_to_extinction(N, F_a)
        else:
            t_ext = np.inf

        return {
            'N': N,
            'F_a': F_a,
            'birth_rate': b,
            'mortality_rate': d,
            'dispersion_rate': D,
            'net_growth_rate': r,
            'dN_dt': dN_dt,
            'doubling_time_yr': np.log(2) / r if r > 0 else np.inf,
            'time_to_extinction_yr': t_ext,
            'intrinsic_growth': self.params.b - self.params.d_min,
        }


# Testing and validation
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING DEMOGRAPHICS MODULE")
    print("=" * 80)

    from config.parameters import DemographicParams, EnergeticsParams

    # Load parameters
    demographic_params = DemographicParams()
    energetics_params = EnergeticsParams()

    dynamics = PopulationDynamics(demographic_params, energetics_params)

    # Test 1: Mortality response to fat availability
    print("\nTEST 1: Mortality Response to Fat Availability")
    print("-" * 80)

    fat_levels = [5, 8, 10, 12, 15, 20, 30, 45, 60, 80, 100]
    print(f"{'F_a (kg)':<12} {'Stress':<10} {'d (% /yr)':<12} {'Life Exp (yr)':<15}")
    print("-" * 80)

    for F_a in fat_levels:
        stress = energetics_params.stress_factor(F_a)
        d = dynamics.mortality_rate(F_a)
        life_exp = 1.0 / d if d > 0 else np.inf

        print(f"{F_a:<12} {stress:<10.2f} {d * 100:<12.2f} {life_exp:<15.1f}")

    # Test 2: Dispersal response
    print("\n\nTEST 2: Dispersal Response to Fat Availability")
    print("-" * 80)

    print(f"{'F_a (kg)':<12} {'D (% /yr)':<15} {'Status':<20}")
    print("-" * 80)

    for F_a in [5, 10, 20, 30, 36, 40, 45, 60]:
        D = dynamics.dispersion_rate(F_a)

        if F_a < energetics_params.F_opt * 0.8:
            status = "STRESS DISPERSAL"
        else:
            status = "BASELINE ONLY"

        print(f"{F_a:<12} {D * 100:<15.2f} {status:<20}")

    # Test 3: Net growth rate across fat gradient
    print("\n\nTEST 3: Net Growth Rate vs Fat Availability")
    print("-" * 80)

    print(f"{'F_a (kg)':<12} {'b':<10} {'d':<10} {'D':<10} {'r (% /yr)':<12} {'Outcome':<20}")
    print("-" * 80)

    for F_a in [5, 8, 10, 15, 20, 30, 45, 60]:
        b = dynamics.birth_rate()
        d = dynamics.mortality_rate(F_a)
        D = dynamics.dispersion_rate(F_a)
        r = dynamics.net_growth_rate(F_a)

        if r > 0:
            outcome = f"GROWING (+{r * 100:.2f}%)"
        elif r < 0:
            outcome = f"DECLINING ({r * 100:.2f}%)"
        else:
            outcome = "EQUILIBRIUM"

        print(f"{F_a:<12} {b:<10.4f} {d:<10.4f} {D:<10.4f} {r * 100:<12.2f} {outcome:<20}")

    # Test 4: Population trajectories
    print("\n\nTEST 4: Population Trajectories (N=300)")
    print("-" * 80)

    N_test = 300
    scenarios = [
        ("Abundant", 60),
        ("Optimal", 45),
        ("Comfortable", 30),
        ("Stressed", 12),
        ("Crisis", 8),
    ]

    print(f"{'Scenario':<15} {'F_a':<8} {'r (% /yr)':<12} {'dN/dt':<10} {'T_double':<15} {'T_extinct':<15}")
    print("-" * 80)

    for name, F_a in scenarios:
        r = dynamics.net_growth_rate(F_a)
        dN_dt = dynamics.derivative(N_test, F_a)

        if r > 0:
            t_double = np.log(2) / r
            t_double_str = f"{t_double:.1f} yr"
            t_ext_str = "∞"
        else:
            t_double_str = "N/A"
            t_ext = dynamics.time_to_extinction(N_test, F_a)
            t_ext_str = f"{t_ext:.1f} yr" if t_ext < 1000 else "∞"

        print(f"{name:<15} {F_a:<8} {r * 100:<12.2f} {dN_dt:<10.1f} {t_double_str:<15} {t_ext_str:<15}")

    # Test 5: Complete summary
    print("\n\nTEST 5: Complete Summary (N=350, F_a=45)")
    print("-" * 80)

    N_summary = 350
    F_a_summary = 45

    summary = dynamics.summary(N_summary, F_a_summary)

    print(f"  Population: {summary['N']:.0f} persons")
    print(f"  Fat availability: {summary['F_a']:.1f} kg/person")
    print(f"\n  Vital rates:")
    print(f"    Birth rate: {summary['birth_rate']:.4f} ({summary['birth_rate'] * 100:.2f}% /yr)")
    print(f"    Mortality rate: {summary['mortality_rate']:.4f} ({summary['mortality_rate'] * 100:.2f}% /yr)")
    print(f"    Dispersion rate: {summary['dispersion_rate']:.4f} ({summary['dispersion_rate'] * 100:.2f}% /yr)")
    print(f"\n  Population change:")
    print(f"    Net growth rate: {summary['net_growth_rate']:.4f} ({summary['net_growth_rate'] * 100:.2f}% /yr)")
    print(f"    dN/dt: {summary['dN_dt']:+.2f} persons/year")

    if summary['doubling_time_yr'] < np.inf:
        print(f"    Doubling time: {summary['doubling_time_yr']:.1f} years")
    else:
        print(f"    Doubling time: ∞ (declining)")

    if summary['time_to_extinction_yr'] < np.inf:
        print(f"    Time to extinction: {summary['time_to_extinction_yr']:.1f} years")
    else:
        print(f"    Time to extinction: ∞ (stable/growing)")

    print(
        f"\n  Intrinsic growth (stress-free): {summary['intrinsic_growth']:.4f} ({summary['intrinsic_growth'] * 100:.2f}% /yr)")

    print("\n" + "=" * 80)
    print("✓ All demographics tests complete")
    print("=" * 80)