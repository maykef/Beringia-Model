#!/usr/bin/env python3
"""
Run Beringia population dynamics simulation.

Complete working simulation of Late Pleistocene hunter-gatherer
population dynamics in Beringia with climate forcing from NGRIP ice core.

Usage:
    python scripts/run_simulation.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parameters import ModelParams
from core.climate_forcing import NGRIPData
from core.integrator import ModelIntegrator


def main():
    print("=" * 80)
    print("BERINGIA POPULATION DYNAMICS MODEL")
    print("=" * 80)

    # 1. Load parameters
    print("\n1. Loading parameters...")
    params = ModelParams()
    params.validate()
    print(params.summary())

    # 2. Load climate data
    print("\n2. Loading NGRIP climate data...")
    data_path = Path(__file__).parent.parent / 'data' / 'NGRIP_eta_real_31_32ka.csv'

    if not data_path.exists():
        print(f"ERROR: NGRIP data not found at {data_path}")
        return 1

    ngrip = NGRIPData(str(data_path))

    # Get Heinrich Stadial 1 (optimal stability period)
    hs1 = ngrip.get_period_eta('Heinrich Stadial 1')
    print(f"\nRunning {hs1.name} scenario:")
    print(f"  η = {hs1.eta:.3f} ± {hs1.eta_std:.3f}")
    print(f"  Age range: {hs1.age_range[0]:.1f}-{hs1.age_range[1]:.1f} ka BP")
    print(f"  Expected pulse frequency: {1.2 / hs1.eta:.2f} /year")

    # 3. Initialize integrator
    print("\n3. Initializing integrator...")
    integrator = ModelIntegrator(params)

    # 4. Run simulation
    print(f"\n4. Running simulation...")
    print(f"   Duration: {params.numerical.t_max} years")
    print(f"   Burn-in: {params.numerical.burn_in} years")
    print(f"   Timestep: {params.numerical.dt} years")

    result = integrator.simulate(
        eta=hs1.eta,
        amplitude=params.climate.amplitude_default,
        seed=params.numerical.base_seed
    )

    # 5. Display results
    print("\n5. Results:")
    print("=" * 80)

    stats = result['stats']

    print(f"\nPOPULATION:")
    print(f"  Mean: {stats['N_mean']:.1f} ± {stats['N_std']:.1f} persons")
    print(f"  CV: {stats['N_cv']:.1%}")
    print(f"  Range: [{stats['N_min']:.0f}, {stats['N_max']:.0f}]")
    print(f"  5th-95th percentile: [{stats['N_percentiles'][5]:.0f}, {stats['N_percentiles'][95]:.0f}]")

    print(f"\nFAT AVAILABILITY:")
    print(f"  Mean: {stats['F_a_mean']:.1f} ± {stats['F_a_std']:.1f} kg/person")
    print(f"  CV: {stats['F_a_cv']:.1%}")
    print(f"  Range: [{stats['F_a_min']:.1f}, {stats['F_a_max']:.1f}]")
    print(f"  Time below F_crit ({params.energetics.F_crit} kg): {stats['time_below_F_crit']:.1f}%")
    print(f"  Time below F_min ({params.energetics.F_min} kg): {stats['time_below_F_min']:.1f}%")

    print(f"\nPROTEIN BIOMASS:")
    print(f"  Mean: {stats['P_mean'] / 1000:.1f} ± {stats['P_std'] / 1000:.1f} tonnes")
    print(f"  CV: {stats['P_cv']:.1%}")
    print(f"  Range: [{stats['P_min'] / 1000:.1f}, {stats['P_max'] / 1000:.1f}]")

    # 6. Create visualization
    print("\n6. Creating visualization...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Beringia Population Dynamics: {hs1.name} (η={hs1.eta:.3f})',
                 fontsize=14, fontweight='bold')

    times = result['times']

    # Population
    ax = axes[0]
    ax.plot(times, result['N'], 'b-', alpha=0.7, linewidth=1)
    n_mean = stats['N_mean']
    ax.axhline(n_mean, color='r', linestyle='--', alpha=0.5, label=f'Mean = {n_mean:.0f}')
    ax.fill_between(times,
                    stats['N_percentiles'][5],
                    stats['N_percentiles'][95],
                    alpha=0.2, color='blue', label='5th-95th percentile')
    ax.set_ylabel('Population (persons)', fontsize=11)
    ax.set_title('Population Dynamics', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Fat availability
    ax = axes[1]
    ax.plot(times, result['F_a'], 'g-', alpha=0.7, linewidth=1)
    f_crit = params.energetics.F_crit
    f_opt = params.energetics.F_opt
    f_mean = stats['F_a_mean']
    ax.axhline(f_crit, color='orange', linestyle='--', alpha=0.5, label=f'F_crit = {f_crit} kg')
    ax.axhline(f_opt, color='green', linestyle='--', alpha=0.5, label=f'F_opt = {f_opt} kg')
    ax.axhline(f_mean, color='r', linestyle='--', alpha=0.5, label=f'Mean = {f_mean:.1f} kg')
    ax.set_ylabel('Fat Availability (kg/person)', fontsize=11)
    ax.set_title('Fat-in-System Dynamics', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Protein biomass
    ax = axes[2]
    ax.plot(times, result['P'] / 1000, 'brown', alpha=0.7, linewidth=1)
    p_mean = stats['P_mean'] / 1000
    ax.axhline(p_mean, color='r', linestyle='--', alpha=0.5, label=f'Mean = {p_mean:.0f} tonnes')
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Protein Biomass (tonnes)', fontsize=11)
    ax.set_title('Resource Dynamics', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'simulation_HS1.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    print("\n" + "=" * 80)
    print("✓ Simulation complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())