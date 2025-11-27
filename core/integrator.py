"""
Integrator module: Semi-implicit Euler integration for coupled dynamics.

Ties together all model components:
- Climate forcing (stochastic pulses)
- Resources (protein biomass)
- Energetics (fat-in-system)
- Demographics (population)

Uses semi-implicit method for numerical stability:
1. Update fat explicitly (F_a_new from current state)
2. Use F_a_new for mortality calculation (implicit)
3. Update population with new mortality
4. Update resources with new population

This prevents numerical crashes from fat-mortality coupling.

Author: [Your name]
Date: November 2025
"""

import numpy as np
from typing import Dict, Tuple, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.parameters import ModelParams
from core.climate_forcing import ResourcePulseGenerator
from core.resources import ProteinDynamics
from core.energetics import FatDynamics
from core.demographics import PopulationDynamics


class ModelIntegrator:
    """
    Semi-implicit Euler integrator for coupled Beringia model.

    State vector:
        [P, F_a, N]

    Where:
        P: Protein biomass [kg]
        F_a: Fat availability [kg/person]
        N: Population [persons]
    """

    def __init__(self, params: ModelParams):
        """
        Initialize integrator with all dynamics modules.

        Args:
            params: Complete ModelParams instance
        """
        self.params = params

        # Initialize all dynamics modules
        self.pulse_gen = ResourcePulseGenerator(
            params.climate,
            rng=np.random.RandomState(params.numerical.base_seed)
        )

        self.resources = ProteinDynamics(
            params.resources,
            params.spatial
        )

        self.energetics = FatDynamics(params.energetics)

        self.demographics = PopulationDynamics(
            params.demographics,
            params.energetics
        )

    def step(self,
             state: Tuple[float, float, float],
             t: float,
             eta: float,
             amplitude: float) -> Tuple[float, float, float]:
        """
        Advance system by one timestep using semi-implicit Euler.

        Order of operations (CRITICAL for stability):
        1. Generate climate pulse
        2. Calculate fat gain from current resources
        3. Update fat (EXPLICIT)
        4. Calculate mortality from NEW fat (IMPLICIT)
        5. Update population with new mortality
        6. Update resources with new population

        Args:
            state: Current [P, F_a, N]
            t: Current time [years]
            eta: Climate irregularity parameter
            amplitude: Pulse amplitude [kg]

        Returns:
            New state [P_new, F_a_new, N_new]
        """
        P, F_a, N = state
        dt = self.params.numerical.dt

        # 1. Generate stochastic pulse
        pulse = self.pulse_gen.generate(t, dt, eta, amplitude)

        # 2. Calculate fat gain from hunting
        fat_gain = self.resources.fat_gain_rate(
            P, N,
            phi_fat_storage=self.params.energetics.phi_fat_storage
        )

        # 3. Update fat (EXPLICIT step)
        dF_a = self.energetics.derivative(F_a, fat_gain, stress_multiplier=1.0)
        F_a_new, dF_a_constrained = self.energetics.enforce_bounds(F_a, dF_a, dt)

        # 4. Calculate demographic rates using NEW fat (IMPLICIT)
        mortality = self.demographics.mortality_rate(F_a_new)
        dispersion = self.demographics.dispersion_rate(F_a_new)
        birth = self.demographics.birth_rate()

        # 5. Update population
        net_rate = birth - mortality - dispersion
        dN = N * net_rate * dt
        N_new = max(1.0, N + dN)  # Enforce minimum population

        # 6. Update resources (using NEW population for harvest)
        dP = self.resources.derivative(P, N_new, pulse)
        P_new = max(0.0, P + dP * dt)

        return (P_new, F_a_new, N_new)

    def simulate(self,
                 eta: float,
                 amplitude: float,
                 initial_state: Optional[Tuple[float, float, float]] = None,
                 seed: Optional[int] = None) -> Dict:
        """
        Run complete simulation.

        Args:
            eta: Climate irregularity parameter
            amplitude: Pulse amplitude [kg]
            initial_state: Optional [P0, F_a0, N0] (uses params if None)
            seed: Random seed (uses params if None)

        Returns:
            Dictionary with time series and statistics
        """
        # Set random seed
        if seed is not None:
            self.pulse_gen.reset(seed)

        # Initial state
        if initial_state is None:
            P0 = self.params.resources.P0
            F_a0 = self.params.energetics.F_a0
            N0 = self.params.demographics.N0
        else:
            P0, F_a0, N0 = initial_state

        # Time array
        n_steps = self.params.numerical.n_steps
        dt = self.params.numerical.dt
        times = np.arange(n_steps) * dt

        # Storage arrays
        P_arr = np.zeros(n_steps)
        F_a_arr = np.zeros(n_steps)
        N_arr = np.zeros(n_steps)

        # Set initial conditions
        state = (P0, F_a0, N0)
        P_arr[0], F_a_arr[0], N_arr[0] = state

        # Integration loop
        for i in range(1, n_steps):
            state = self.step(state, times[i], eta, amplitude)
            P_arr[i], F_a_arr[i], N_arr[i] = state

        # Remove burn-in
        burn_in_steps = self.params.numerical.burn_in_steps
        times_analysis = times[burn_in_steps:]
        P_analysis = P_arr[burn_in_steps:]
        F_a_analysis = F_a_arr[burn_in_steps:]
        N_analysis = N_arr[burn_in_steps:]

        # Compute statistics
        stats = self._compute_statistics(
            P_analysis, F_a_analysis, N_analysis
        )

        return {
            'times_full': times,
            'P_full': P_arr,
            'F_a_full': F_a_arr,
            'N_full': N_arr,
            'times': times_analysis,
            'P': P_analysis,
            'F_a': F_a_analysis,
            'N': N_analysis,
            'stats': stats,
            'params': {
                'eta': eta,
                'amplitude': amplitude,
                'dt': dt,
                'burn_in': self.params.numerical.burn_in,
            }
        }

    def _compute_statistics(self,
                            P: np.ndarray,
                            F_a: np.ndarray,
                            N: np.ndarray) -> Dict:
        """
        Compute summary statistics from time series.

        Args:
            P, F_a, N: Time series arrays (post burn-in)

        Returns:
            Dictionary with statistics
        """
        # Percentiles
        percentiles = [5, 25, 50, 75, 95]

        stats = {
            # Protein
            'P_mean': np.mean(P),
            'P_std': np.std(P),
            'P_cv': np.std(P) / np.mean(P) if np.mean(P) > 0 else 0,
            'P_min': np.min(P),
            'P_max': np.max(P),
            'P_percentiles': {p: np.percentile(P, p) for p in percentiles},

            # Fat
            'F_a_mean': np.mean(F_a),
            'F_a_std': np.std(F_a),
            'F_a_cv': np.std(F_a) / np.mean(F_a) if np.mean(F_a) > 0 else 0,
            'F_a_min': np.min(F_a),
            'F_a_max': np.max(F_a),
            'F_a_percentiles': {p: np.percentile(F_a, p) for p in percentiles},

            # Population
            'N_mean': np.mean(N),
            'N_std': np.std(N),
            'N_cv': np.std(N) / np.mean(N) if np.mean(N) > 0 else 0,
            'N_min': np.min(N),
            'N_max': np.max(N),
            'N_percentiles': {p: np.percentile(N, p) for p in percentiles},

            # Critical events
            'time_below_F_crit': np.sum(F_a < self.params.energetics.F_crit) / len(F_a) * 100,
            'time_below_F_min': np.sum(F_a < self.params.energetics.F_min) / len(F_a) * 100,
            'N_below_100': np.sum(N < 100) / len(N) * 100,
        }

        return stats

    def ensemble(self,
                 eta: float,
                 amplitude: float,
                 n_runs: Optional[int] = None,
                 base_seed: Optional[int] = None) -> Dict:
        """
        Run ensemble of simulations with different random seeds.

        Args:
            eta: Climate irregularity parameter
            amplitude: Pulse amplitude [kg]
            n_runs: Number of ensemble members (uses params if None)
            base_seed: Base random seed (uses params if None)

        Returns:
            Dictionary with ensemble statistics
        """
        if n_runs is None:
            n_runs = self.params.numerical.n_ensemble

        if base_seed is None:
            base_seed = self.params.numerical.base_seed

        # Storage for ensemble
        all_results = []

        print(f"Running ensemble: {n_runs} realizations...")
        for i in range(n_runs):
            seed = base_seed + i
            result = self.simulate(eta, amplitude, seed=seed)
            all_results.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{n_runs}")

        # Aggregate statistics
        ensemble_stats = self._aggregate_ensemble(all_results)

        return {
            'individual_runs': all_results,
            'ensemble_stats': ensemble_stats,
            'params': {
                'eta': eta,
                'amplitude': amplitude,
                'n_runs': n_runs,
            }
        }

    def _aggregate_ensemble(self, results: list) -> Dict:
        """
        Aggregate statistics across ensemble members.

        Args:
            results: List of individual simulation results

        Returns:
            Dictionary with ensemble statistics
        """
        # Extract mean values from each run
        P_means = [r['stats']['P_mean'] for r in results]
        F_a_means = [r['stats']['F_a_mean'] for r in results]
        N_means = [r['stats']['N_mean'] for r in results]

        return {
            'P_ensemble_mean': np.mean(P_means),
            'P_ensemble_std': np.std(P_means),
            'F_a_ensemble_mean': np.mean(F_a_means),
            'F_a_ensemble_std': np.std(F_a_means),
            'N_ensemble_mean': np.mean(N_means),
            'N_ensemble_std': np.std(N_means),
            'N_ensemble_cv': np.std(N_means) / np.mean(N_means) if np.mean(N_means) > 0 else 0,
        }


# Testing
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING INTEGRATOR MODULE")
    print("=" * 80)

    from config.parameters import ModelParams

    # Load parameters
    params = ModelParams()
    params.validate()

    integrator = ModelIntegrator(params)

    # Test 1: Single step
    print("\nTEST 1: Single Integration Step")
    print("-" * 80)

    state0 = (350000, 35, 350)  # P, F_a, N
    t = 0.0
    eta = 1.09
    amplitude = 40000

    print(f"Initial state:")
    print(f"  P = {state0[0] / 1000:.1f} tonnes")
    print(f"  F_a = {state0[1]:.1f} kg/person")
    print(f"  N = {state0[2]:.0f} persons")

    state1 = integrator.step(state0, t, eta, amplitude)

    print(f"\nAfter one step (dt={params.numerical.dt} yr):")
    print(f"  P = {state1[0] / 1000:.1f} tonnes (Δ = {(state1[0] - state0[0]) / 1000:+.1f})")
    print(f"  F_a = {state1[1]:.1f} kg/person (Δ = {state1[1] - state0[1]:+.1f})")
    print(f"  N = {state1[2]:.0f} persons (Δ = {state1[2] - state0[2]:+.0f})")

    # Test 2: Short simulation
    print("\n\nTEST 2: Short Simulation (100 years)")
    print("-" * 80)

    # Temporarily reduce duration
    params.numerical.t_max = 100
    params.numerical.burn_in = 20

    result = integrator.simulate(eta=1.09, amplitude=40000, seed=42)

    stats = result['stats']
    print(f"\nPopulation statistics:")
    print(f"  Mean: {stats['N_mean']:.1f} persons")
    print(f"  Std: {stats['N_std']:.1f} persons")
    print(f"  CV: {stats['N_cv']:.1%}")
    print(f"  Range: [{stats['N_min']:.0f}, {stats['N_max']:.0f}]")

    print(f"\nFat statistics:")
    print(f"  Mean: {stats['F_a_mean']:.1f} kg/person")
    print(f"  Std: {stats['F_a_std']:.1f} kg/person")
    print(f"  Time below F_crit: {stats['time_below_F_crit']:.1f}%")

    print(f"\nProtein statistics:")
    print(f"  Mean: {stats['P_mean'] / 1000:.1f} tonnes")
    print(f"  CV: {stats['P_cv']:.1%}")

    print("\n" + "=" * 80)
    print("✓ All integrator tests complete")
    print("=" * 80)