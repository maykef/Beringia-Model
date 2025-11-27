"""
Resource dynamics module: Ungulate protein biomass.

Implements:
- Logistic growth with density-dependent carrying capacity
- Type II functional response (hunting saturation)
- Interference competition among hunters
- Fat gain from hunting success

Based on mammoth steppe ecology:
- NPP: 300-500 g C/m²/yr (Thermodynamics doc §3)
- Herbivore biomass: 5-15 kg/m² for BLB mesic
- Sustainable yield: 275,000-550,000 kg/yr from NPP conversion

References:
- Zimov et al. (2012) - Mammoth steppe productivity
- Caughley (1977) - Ungulate population dynamics
- Thermodynamics_of_Beringia_V4.docx §3

Author: [Your name]
Date: November 2025
"""

import numpy as np
from typing import Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config.parameters import ResourceParams, SpatialParams


class ProteinDynamics:
    """
    Ungulate protein biomass dynamics.

    Implements logistic growth with density-dependent carrying capacity
    and Type II functional response hunting.

    State variable:
        P: Protein biomass [kg] (ungulate standing stock)

    Dynamics:
        dP/dt = r_P × P × (1 - P/K_eff(N)) - H(P, N) + Pulses

    Where:
        K_eff(N) = Density-dependent carrying capacity
        H(P, N) = Hunting with Type II response + interference
    """

    def __init__(self,
                 resource_params: ResourceParams,
                 spatial_params: SpatialParams):
        """
        Initialize protein dynamics.

        Args:
            resource_params: ResourceParams instance
            spatial_params: SpatialParams instance
        """
        self.params = resource_params
        self.spatial = spatial_params

    def carrying_capacity(self, N: float) -> float:
        """
        Calculate density-dependent carrying capacity.

        K_eff decreases as human density increases due to:
        - Trampling and disturbance
        - Habitat modification
        - Increased predation pressure

        Formula:
            K_eff(N) = K_base / (1 + α × ρ)

        Where:
            ρ = N / Area [persons/km²]
            α = density dependence coefficient

        Args:
            N: Current population [persons]

        Returns:
            Effective carrying capacity [kg]
        """
        if N <= 0:
            return self.params.K_P_base

        density = N / self.spatial.area_km2
        K_eff = self.params.K_P_base / (1.0 + self.params.alpha_density * density)

        return K_eff

    def growth_rate(self, P: float, N: float) -> float:
        """
        Logistic growth with density-dependent K.

        Formula:
            dP/dt|_growth = r_P × P × (1 - P/K_eff(N))

        Features:
        - Exponential growth at low P
        - Saturates at K_eff
        - K_eff decreases with human density

        Args:
            P: Current protein biomass [kg]
            N: Current population [persons]

        Returns:
            Growth rate [kg/year]
        """
        if P <= 0:
            return 0.0

        K_eff = self.carrying_capacity(N)
        growth = self.params.r_P * P * (1.0 - P / K_eff)

        return growth

    def hunting_success(self, P: float, N: float) -> float:
        """
        Per-capita hunting yield with Type II response and interference.

        Type II functional response (Holling 1959):
            h(P) = h_base × P / (P + P_half)

        Captures saturation at high resource density (handling time limit).

        Interference competition:
            h_eff = h_base / (1 + γ × N/N_ref)

        As hunter density increases, efficiency drops (competition for space).

        Args:
            P: Current protein biomass [kg]
            N: Current population [persons]

        Returns:
            Per-capita hunting success [kg/person/year]
        """
        if P <= 0 or N <= 0:
            return 0.0

        # Interference reduces base hunting rate
        h_eff = self.params.h_base / (1.0 + self.params.gamma_interference * N / 500.0)

        # Type II functional response (saturation)
        success = h_eff * (P / (P + self.params.P_half))

        return success

    def total_harvest(self, P: float, N: float) -> float:
        """
        Total harvest rate (all hunters combined).

        Formula:
            H = ε × h(P) × N

        Where:
            ε = conversion efficiency (edible fraction)
            h(P) = per-capita hunting success
            N = number of hunters

        Args:
            P: Current protein biomass [kg]
            N: Current population [persons]

        Returns:
            Total harvest rate [kg/year]
        """
        if N <= 0:
            return 0.0

        success_per_capita = self.hunting_success(P, N)
        harvest = self.params.epsilon_conversion * success_per_capita * N

        return harvest

    def fat_gain_rate(self, P: float, N: float, phi_fat_storage: float = 0.10) -> float:
        """
        Per-capita fat gain from hunting.

        A fraction φ of hunting yield goes to storable fat
        (rendering animal fat for caching).

        Formula:
            dF_a/dt|_hunting = φ × h(P) × (P/(P+50000))

        Args:
            P: Current protein biomass [kg]
            N: Current population [persons]
            phi_fat_storage: Fat storage fraction (from EnergeticsParams)

        Returns:
            Fat gain per person [kg/person/year]
        """
        if P <= 0:
            return 0.0

        # Hunting success (NOT multiplied by epsilon - this is gross yield)
        hunting = self.hunting_success(P, N)

        # Resource availability factor (Type II again for fat extraction)
        resource_factor = P / (P + 50000.0)

        # Fat storage
        fat_gain = phi_fat_storage * hunting * resource_factor

        return fat_gain

    def derivative(self,
                   P: float,
                   N: float,
                   pulse: float = 0.0) -> float:
        """
        Complete dP/dt including all processes.

        Formula:
            dP/dt = Growth - Harvest + Pulse
                  = r_P × P × (1 - P/K_eff) - ε × h(P) × N + Pulse

        Args:
            P: Current protein biomass [kg]
            N: Current population [persons]
            pulse: Stochastic pulse rate [kg/year]

        Returns:
            dP/dt [kg/year]
        """
        growth = self.growth_rate(P, N)
        harvest = self.total_harvest(P, N)

        dP_dt = growth - harvest + pulse

        return dP_dt

    def equilibrium(self, N: float, pulse_rate: float = 0.0) -> float:
        """
        Find equilibrium protein biomass for given population.

        Solves: dP/dt = 0

        Useful for:
        - Initial conditions
        - Stability analysis
        - Validating parameter choices

        Args:
            N: Population [persons]
            pulse_rate: Average pulse input [kg/year]

        Returns:
            Equilibrium P [kg]
        """
        # Use simple iterative solver
        P = self.params.K_P_base * 0.5  # Initial guess

        for _ in range(100):
            dP = self.derivative(P, N, pulse_rate)
            P += 0.01 * dP  # Small step toward equilibrium

            if abs(dP) < 1.0:  # Converged
                break

        return max(0, P)

    def summary(self, P: float, N: float) -> dict:
        """
        Compute all relevant metrics for current state.

        Useful for:
        - Debugging
        - Monitoring
        - Analysis

        Args:
            P: Current protein biomass [kg]
            N: Current population [persons]

        Returns:
            Dictionary with metrics
        """
        K_eff = self.carrying_capacity(N)
        growth = self.growth_rate(P, N)
        harvest = self.total_harvest(P, N)
        hunting_pc = self.hunting_success(P, N)
        fat_gain = self.fat_gain_rate(P, N)

        return {
            'P_kg': P,
            'P_tonnes': P / 1000,
            'K_eff_kg': K_eff,
            'K_eff_tonnes': K_eff / 1000,
            'utilization': P / K_eff if K_eff > 0 else 0,
            'growth_rate_kg_yr': growth,
            'harvest_rate_kg_yr': harvest,
            'hunting_per_capita_kg_yr': hunting_pc,
            'net_food_per_capita_kg_yr': hunting_pc * self.params.epsilon_conversion,
            'fat_gain_per_capita_kg_yr': fat_gain,
            'net_dP_dt': growth - harvest,
        }


# Testing and validation
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING RESOURCE DYNAMICS MODULE")
    print("=" * 80)

    from config.parameters import ResourceParams, SpatialParams

    # Load parameters
    resource_params = ResourceParams()
    spatial_params = SpatialParams()

    dynamics = ProteinDynamics(resource_params, spatial_params)

    # Test 1: Carrying capacity dependence
    print("\nTEST 1: Density-Dependent Carrying Capacity")
    print("-" * 80)
    populations = [0, 100, 200, 300, 400, 500]
    print(f"{'N (persons)':<15} {'K_eff (tonnes)':<20} {'Reduction':<15}")
    print("-" * 80)
    K_base = resource_params.K_P_base / 1000
    for N in populations:
        K_eff = dynamics.carrying_capacity(N) / 1000
        reduction = (1 - K_eff / K_base) * 100 if K_base > 0 else 0
        print(f"{N:<15} {K_eff:<20.1f} {reduction:<15.1f}%")

    # Test 2: Type II functional response
    print("\n\nTEST 2: Type II Functional Response (Hunting)")
    print("-" * 80)
    N_test = 300
    protein_levels = [10000, 50000, 100000, 200000, 400000]
    print(f"{'P (tonnes)':<15} {'h (kg/person/yr)':<20} {'% of h_base':<15}")
    print("-" * 80)
    for P in protein_levels:
        h = dynamics.hunting_success(P, N_test)
        pct = (h / resource_params.h_base) * 100
        print(f"{P / 1000:<15.0f} {h:<20.1f} {pct:<15.1f}%")

    # Test 3: Equilibrium at different population sizes
    print("\n\nTEST 3: Equilibrium Analysis")
    print("-" * 80)
    print(f"{'N (persons)':<15} {'P_eq (tonnes)':<20} {'Growth':<15} {'Harvest':<15}")
    print("-" * 80)
    for N in [200, 300, 400, 500]:
        P_eq = dynamics.equilibrium(N)
        growth = dynamics.growth_rate(P_eq, N)
        harvest = dynamics.total_harvest(P_eq, N)
        print(f"{N:<15} {P_eq / 1000:<20.1f} {growth / 1000:<15.1f} {harvest / 1000:<15.1f}")

    # Test 4: Complete state summary
    print("\n\nTEST 4: State Summary (N=350, P=350 tonnes)")
    print("-" * 80)
    P_test = 350000
    N_test = 350
    summary = dynamics.summary(P_test, N_test)

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:<30}: {value:.2f}")
        else:
            print(f"  {key:<30}: {value}")

    # Test 5: Energy budget validation
    print("\n\nTEST 5: Energy Budget Validation")
    print("-" * 80)
    net_food = summary['net_food_per_capita_kg_yr']
    metabolic_need = 180.0  # kg/person/yr from parameters
    surplus = net_food - metabolic_need

    print(f"  Net food per capita: {net_food:.1f} kg/yr")
    print(f"  Metabolic baseline: {metabolic_need:.1f} kg/yr")
    print(f"  Energy surplus: {surplus:.1f} kg/yr ({surplus / metabolic_need * 100:.1f}%)")

    if surplus > 0:
        print("  ✓ Energy budget POSITIVE (sustainable)")
    else:
        print("  ✗ Energy budget NEGATIVE (unsustainable)")

    # Test 6: Fat gain validation
    print("\n\nTEST 6: Fat Gain Validation")
    print("-" * 80)
    fat_gain = summary['fat_gain_per_capita_kg_yr']
    thermal_deficit = 12.0  # kg/person/yr from parameters
    net_fat = fat_gain - thermal_deficit

    print(f"  Fat gain from hunting: {fat_gain:.1f} kg/person/yr")
    print(f"  Thermal deficit: {thermal_deficit:.1f} kg/person/yr")
    print(f"  Net fat balance: {net_fat:.1f} kg/person/yr")

    if net_fat > 0:
        print("  ✓ Fat balance POSITIVE (caches accumulate)")
    else:
        print(f"  ⚠ Fat balance NEGATIVE (caches deplete by {abs(net_fat):.1f} kg/yr)")

    print("\n" + "=" * 80)
    print("✓ All resource dynamics tests complete")
    print("=" * 80)