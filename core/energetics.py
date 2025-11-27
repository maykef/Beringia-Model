"""
Energetics module: Fat-in-system dynamics.

Implements:
- Fat-in-system accumulation/depletion
- Thermal deficit consumption
- Stress-dependent mortality
- Energy budget tracking

CRITICAL: F_a is NOT body fat (adipose tissue)!
F_a represents TOTAL FAT AVAILABILITY in the coupled human-environment system:
  - Cached animal fat in permafrost storage
  - Fresh kills being processed
  - Social network sharing access
  - Body adipose buffer (~20-30%)

Thresholds represent CACHE STATUS, not individual body composition.

References:
- Thermodynamics_of_Beringia_V4.docx §4 (energy budgets)
- Burch (1988) - Inuit cache systems
- Pontzer et al. (2012) - Hunter-gatherer energetics

Author: [Your name]
Date: November 2025
"""

import numpy as np
from typing import Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config.parameters import EnergeticsParams


class FatDynamics:
    """
    Fat-in-system dynamics (total accessible energy reserves).

    State variable:
        F_a: Fat availability [kg/person] (system-level, not just adipose)

    Dynamics:
        dF_a/dt = Fat_gain(hunting) - ψ(thermal deficit) - Losses(mortality)

    Where:
        Fat_gain = φ × hunting_success (from resources module)
        ψ = thermal deficit (base rate)
        Losses = Fat lost when people die/emigrate
    """

    def __init__(self, energetics_params: EnergeticsParams):
        """
        Initialize fat dynamics.

        Args:
            energetics_params: EnergeticsParams instance
        """
        self.params = energetics_params

    def stress_factor(self, F_a: float) -> float:
        """
        Calculate stress from fat availability [0, 1].

        Stress increases nonlinearly as caches deplete:
        - F_a >= F_crit: stress = 0 (comfortable)
        - F_crit > F_a > F_min: stress = (F_crit - F_a)/(F_crit - F_min)
        - F_a <= F_min: stress = 1 (crisis)

        Args:
            F_a: Current fat availability [kg/person]

        Returns:
            Stress factor [0, 1]
        """
        return self.params.stress_factor(F_a)

    def consumption_rate(self, F_a: float, stress_multiplier: float = 1.0) -> float:
        """
        Fat consumption rate (thermal deficit).

        Base consumption (ψ) represents thermal needs beyond what
        hunting provides in lean tissue. Can increase under stress
        (e.g., during fuel shortage, requiring more body fat burning).

        Formula:
            Consumption = ψ × stress_multiplier

        Args:
            F_a: Current fat availability [kg/person]
            stress_multiplier: Additional consumption under stress (default 1.0)

        Returns:
            Consumption rate [kg/person/year]
        """
        base_consumption = self.params.psi_fat_consumption

        # Can be modified by external stressors (e.g., fuel shortage)
        consumption = base_consumption * stress_multiplier

        return consumption

    def derivative(self,
                   F_a: float,
                   fat_gain: float,
                   stress_multiplier: float = 1.0) -> float:
        """
        Complete dF_a/dt (per capita, excluding mortality losses).

        Formula:
            dF_a/dt = fat_gain - consumption

        Note: This is PER CAPITA. Population-level losses from mortality
        are handled in the integrator (people taking their F_a with them).

        Args:
            F_a: Current fat availability [kg/person]
            fat_gain: Fat gain from hunting [kg/person/year]
            stress_multiplier: Stress factor (default 1.0)

        Returns:
            dF_a/dt [kg/person/year]
        """
        consumption = self.consumption_rate(F_a, stress_multiplier)

        dF_a_dt = fat_gain - consumption

        return dF_a_dt

    def enforce_bounds(self, F_a: float, dF_a: float, dt: float) -> Tuple[float, float]:
        """
        Enforce physical bounds on fat reserves.

        Energy-conserving constraints:
        - F_min ≤ F_a ≤ F_max
        - If dF_a would violate bounds, reduce rate proportionally

        This prevents numerical crashes while preserving energy conservation.

        Args:
            F_a: Current fat availability [kg/person]
            dF_a: Proposed change [kg/person]
            dt: Timestep [years]

        Returns:
            (F_a_new, dF_a_constrained)
        """
        # Proposed new value
        F_a_proposed = F_a + dF_a * dt

        # Check upper bound
        if F_a_proposed > self.params.F_max:
            # Limit rate to reach F_max exactly
            dF_a_constrained = (self.params.F_max - F_a) / dt
            F_a_new = self.params.F_max

        # Check lower bound
        elif F_a_proposed < self.params.F_min:
            # Limit rate to reach F_min exactly
            dF_a_constrained = (self.params.F_min - F_a) / dt
            F_a_new = self.params.F_min

        else:
            # Within bounds - no constraint needed
            dF_a_constrained = dF_a
            F_a_new = F_a_proposed

        return F_a_new, dF_a_constrained

    def cache_status(self, F_a: float) -> str:
        """
        Classify cache status based on F_a level.

        Returns human-readable status for monitoring/debugging.

        Args:
            F_a: Current fat availability [kg/person]

        Returns:
            Status string
        """
        if F_a >= self.params.F_opt:
            return "ABUNDANT"
        elif F_a >= self.params.F_crit:
            return "COMFORTABLE"
        elif F_a >= self.params.F_min:
            return "STRESSED"
        else:
            return "CRISIS"

    def time_to_depletion(self, F_a: float, consumption_rate: float) -> float:
        """
        Estimate time until caches reach F_min at current consumption.

        Useful for:
        - Early warning of crisis
        - Decision-making (dispersal timing)
        - Analysis

        Args:
            F_a: Current fat availability [kg/person]
            consumption_rate: Net consumption [kg/person/year]

        Returns:
            Time to F_min [years] (np.inf if consumption <= 0)
        """
        if consumption_rate <= 0:
            return np.inf  # Caches accumulating

        available_above_min = F_a - self.params.F_min

        if available_above_min <= 0:
            return 0.0  # Already at minimum

        time = available_above_min / consumption_rate

        return time

    def energy_budget(self,
                      F_a: float,
                      fat_gain: float,
                      net_food: float) -> dict:
        """
        Compute complete energy budget.

        Tracks all energy flows:
        - Input: Net food (protein) + Fat gain (lipid)
        - Output: Metabolic baseline + Thermal deficit
        - Balance: Surplus/deficit

        Args:
            F_a: Current fat availability [kg/person]
            fat_gain: Fat gain from hunting [kg/person/year]
            net_food: Net protein intake [kg/person/year]

        Returns:
            Dictionary with budget components
        """
        # Inputs
        energy_in = net_food + fat_gain  # Total food energy

        # Outputs
        metabolic = self.params.metabolic_baseline
        thermal = self.consumption_rate(F_a)
        energy_out = metabolic + thermal

        # Balance
        balance = energy_in - energy_out

        # Convert to kcal for reference (4 kcal/g for both protein and fat, roughly)
        kcal_per_kg = 4000

        return {
            'net_food_kg_yr': net_food,
            'fat_gain_kg_yr': fat_gain,
            'energy_in_kg_yr': energy_in,
            'energy_in_kcal_day': energy_in * kcal_per_kg / 365,
            'metabolic_kg_yr': metabolic,
            'thermal_deficit_kg_yr': thermal,
            'energy_out_kg_yr': energy_out,
            'energy_out_kcal_day': energy_out * kcal_per_kg / 365,
            'balance_kg_yr': balance,
            'balance_kcal_day': balance * kcal_per_kg / 365,
            'surplus_pct': (balance / energy_out) * 100 if energy_out > 0 else 0,
        }

    def summary(self, F_a: float, fat_gain: float = 0.0) -> dict:
        """
        Compute all relevant metrics for current state.

        Args:
            F_a: Current fat availability [kg/person]
            fat_gain: Fat gain from hunting [kg/person/year]

        Returns:
            Dictionary with metrics
        """
        stress = self.stress_factor(F_a)
        consumption = self.consumption_rate(F_a)
        net_change = fat_gain - consumption
        status = self.cache_status(F_a)

        # Time to crisis
        if net_change < 0:
            time_to_min = self.time_to_depletion(F_a, -net_change)
        else:
            time_to_min = np.inf

        return {
            'F_a_kg': F_a,
            'status': status,
            'stress_factor': stress,
            'fat_gain_kg_yr': fat_gain,
            'consumption_kg_yr': consumption,
            'net_change_kg_yr': net_change,
            'time_to_crisis_yr': time_to_min,
            'F_min_threshold': self.params.F_min,
            'F_crit_threshold': self.params.F_crit,
            'F_opt_threshold': self.params.F_opt,
            'F_max_capacity': self.params.F_max,
        }


# Testing and validation
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING ENERGETICS MODULE")
    print("=" * 80)

    from config.parameters import EnergeticsParams

    # Load parameters
    energetics_params = EnergeticsParams()
    dynamics = FatDynamics(energetics_params)

    # Test 1: Stress response
    print("\nTEST 1: Stress Factor vs Fat Availability")
    print("-" * 80)
    fat_levels = [5, 8, 10, 15, 20, 30, 45, 60, 80, 100]
    print(f"{'F_a (kg)':<12} {'Status':<15} {'Stress':<12} {'Time to Crisis':<20}")
    print("-" * 80)

    for F_a in fat_levels:
        stress = dynamics.stress_factor(F_a)
        status = dynamics.cache_status(F_a)
        consumption = dynamics.consumption_rate(F_a)

        # Assume no fat gain (worst case)
        time_to_crisis = dynamics.time_to_depletion(F_a, consumption)
        time_str = f"{time_to_crisis:.1f} yr" if time_to_crisis < 100 else "∞"

        print(f"{F_a:<12} {status:<15} {stress:<12.2f} {time_str:<20}")

    # Test 2: Fat dynamics under different scenarios
    print("\n\nTEST 2: Fat Dynamics Scenarios")
    print("-" * 80)

    scenarios = [
        ("Good hunting", 45, 35.0, 1.0),
        ("Average hunting", 45, 25.0, 1.0),
        ("Poor hunting", 45, 15.0, 1.0),
        ("Good + fuel stress", 45, 35.0, 1.5),
        ("Crisis (low F_a)", 10, 15.0, 1.0),
    ]

    print(f"{'Scenario':<25} {'F_a':<8} {'Fat gain':<12} {'Stress×':<10} {'dF_a/dt':<12} {'Outcome':<15}")
    print("-" * 80)

    for name, F_a, fat_gain, stress_mult in scenarios:
        dF_a = dynamics.derivative(F_a, fat_gain, stress_mult)

        if dF_a > 0:
            outcome = f"+{dF_a:.1f} kg/yr"
        else:
            outcome = f"{dF_a:.1f} kg/yr"

        print(f"{name:<25} {F_a:<8} {fat_gain:<12.1f} {stress_mult:<10.1f} {dF_a:<12.1f} {outcome:<15}")

    # Test 3: Energy budget
    print("\n\nTEST 3: Energy Budget Analysis")
    print("-" * 80)

    F_a_test = 45  # kg/person (comfortable)
    fat_gain_test = 25.0  # kg/person/yr
    net_food_test = 170.0  # kg/person/yr (from resources at equilibrium)

    budget = dynamics.energy_budget(F_a_test, fat_gain_test, net_food_test)

    print(f"  Fat availability: {F_a_test} kg/person ({dynamics.cache_status(F_a_test)})")
    print(f"\n  INPUTS:")
    print(f"    Net food (protein): {budget['net_food_kg_yr']:.1f} kg/yr")
    print(f"    Fat gain (hunting): {budget['fat_gain_kg_yr']:.1f} kg/yr")
    print(f"    Total energy IN: {budget['energy_in_kg_yr']:.1f} kg/yr ({budget['energy_in_kcal_day']:.0f} kcal/day)")

    print(f"\n  OUTPUTS:")
    print(f"    Metabolic baseline: {budget['metabolic_kg_yr']:.1f} kg/yr")
    print(f"    Thermal deficit: {budget['thermal_deficit_kg_yr']:.1f} kg/yr")
    print(
        f"    Total energy OUT: {budget['energy_out_kg_yr']:.1f} kg/yr ({budget['energy_out_kcal_day']:.0f} kcal/day)")

    print(f"\n  BALANCE:")
    print(f"    Net: {budget['balance_kg_yr']:+.1f} kg/yr ({budget['balance_kcal_day']:+.0f} kcal/day)")
    print(f"    Surplus: {budget['surplus_pct']:+.1f}%")

    if budget['balance_kg_yr'] > 0:
        print(f"    ✓ POSITIVE energy balance (sustainable)")
    else:
        print(f"    ✗ NEGATIVE energy balance (unsustainable)")

    # Test 4: Bounds enforcement
    print("\n\nTEST 4: Bounds Enforcement (Energy Conservation)")
    print("-" * 80)

    dt = 0.05  # years (timestep)

    test_cases = [
        ("Near F_max", 95, 150, dt),  # Would exceed F_max
        ("Near F_min", 10, -50, dt),  # Would go below F_min
        ("Within bounds", 50, 20, dt),  # Normal operation
    ]

    print(f"{'Case':<20} {'F_a':<8} {'dF_a':<12} {'F_a_new':<12} {'Constrained?':<15}")
    print("-" * 80)

    for name, F_a, dF_a, timestep in test_cases:
        F_a_new, dF_a_const = dynamics.enforce_bounds(F_a, dF_a, timestep)
        constrained = "YES" if abs(dF_a_const - dF_a) > 0.01 else "NO"

        print(f"{name:<20} {F_a:<8.1f} {dF_a:<12.1f} {F_a_new:<12.1f} {constrained:<15}")

    # Test 5: Complete state summary
    print("\n\nTEST 5: Complete State Summary")
    print("-" * 80)

    F_a_summary = 30  # kg/person (stressed)
    fat_gain_summary = 20.0  # kg/person/yr

    summary = dynamics.summary(F_a_summary, fat_gain_summary)

    print(f"  Fat availability: {summary['F_a_kg']:.1f} kg/person")
    print(f"  Status: {summary['status']}")
    print(f"  Stress factor: {summary['stress_factor']:.2f}")
    print(f"\n  Fat dynamics:")
    print(f"    Gain: {summary['fat_gain_kg_yr']:.1f} kg/yr")
    print(f"    Consumption: {summary['consumption_kg_yr']:.1f} kg/yr")
    print(f"    Net change: {summary['net_change_kg_yr']:+.1f} kg/yr")

    if summary['time_to_crisis_yr'] < np.inf:
        print(f"    Time to crisis: {summary['time_to_crisis_yr']:.1f} years")
    else:
        print(f"    Time to crisis: ∞ (accumulating)")

    print(f"\n  Thresholds:")
    print(f"    F_min (crisis): {summary['F_min_threshold']:.1f} kg")
    print(f"    F_crit (stress): {summary['F_crit_threshold']:.1f} kg")
    print(f"    F_opt (comfortable): {summary['F_opt_threshold']:.1f} kg")
    print(f"    F_max (capacity): {summary['F_max_capacity']:.1f} kg")

    print("\n" + "=" * 80)
    print("✓ All energetics tests complete")
    print("=" * 80)