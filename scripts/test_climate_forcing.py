#!/usr/bin/env python3
"""
Test script for climate forcing module.

Loads real NGRIP data and generates stochastic resource pulses
for all climate periods.

Usage:
    python scripts/test_climate_forcing.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parameters import ClimateParams
from core.climate_forcing import NGRIPData, ResourcePulseGenerator


def main():
    print("=" * 80)
    print("BERINGIA MODEL: Climate Forcing Test")
    print("=" * 80)

    # 1. Load real NGRIP data
    print("\n1. Loading NGRIP ice core data...")
    data_path = Path(__file__).parent.parent / 'data' / 'NGRIP_eta_real_31_32ka.csv'

    if not data_path.exists():
        print(f"ERROR: NGRIP data not found at {data_path}")
        print("Please ensure NGRIP_eta_real_31_32ka.csv is in data/ directory")
        return 1

    ngrip = NGRIPData(str(data_path))
    print(ngrip.summary())

    # 2. Set up pulse generator
    print("\n2. Setting up pulse generator...")
    climate_params = ClimateParams()
    pulse_gen = ResourcePulseGenerator(climate_params, rng=np.random.RandomState(42))

    # 3. Generate pulses for each period
    print("\n3. Generating stochastic pulses (100 years)...")
    print("=" * 80)

    dt = 0.05  # years
    t_max = 100.0
    times = np.arange(0, t_max, dt)
    amplitude = 40000.0  # kg

    periods = ngrip.get_all_periods()
    results = {}

    for period_name, period in periods.items():
        print(f"\n{period_name} (η={period.eta:.3f}):")

        # Reset pulse generator
        pulse_gen.reset(seed=42)

        # Generate pulses
        pulses = []
        for t in times:
            pulse = pulse_gen.generate(t, dt, period.eta, amplitude)
            pulses.append(pulse)

        # Get statistics
        stats = pulse_gen.get_statistics(0, t_max)

        print(f"  Pulses generated: {stats['n_pulses']}")
        print(f"  Realized frequency: {stats['frequency']:.2f} /year")
        print(f"  Expected frequency: {1.2 / period.eta:.2f} /year")
        print(f"  Mean amplitude: {stats['mean_amplitude'] / 1000:.1f} tonnes")
        print(f"  Std amplitude: {stats['std_amplitude'] / 1000:.1f} tonnes")
        print(f"  CV: {stats['std_amplitude'] / stats['mean_amplitude']:.1%}")

        results[period_name] = {
            'period': period,
            'times': times,
            'pulses': np.array(pulses),
            'stats': stats
        }

    # 4. Create visualization
    print("\n4. Creating visualization...")
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))
    fig.suptitle('Stochastic Resource Pulses Across Climate Periods', fontsize=14, fontweight='bold')

    for i, (period_name, data) in enumerate(results.items()):
        ax = axes[i]
        period = data['period']

        # Plot pulses
        ax.plot(data['times'], data['pulses'] / 1000, 'b-', alpha=0.6, linewidth=0.5)

        # Mark pulse events
        pulse_times = data['times'][data['pulses'] > 0]
        pulse_values = data['pulses'][data['pulses'] > 0] / 1000
        ax.scatter(pulse_times, pulse_values, c='red', s=10, alpha=0.8, zorder=5)

        # Labels
        ax.set_ylabel('Resource Pulse\n(tonnes/year)', fontsize=10)
        ax.set_title(
            f"{period_name} (η={period.eta:.3f}, {data['stats']['n_pulses']} pulses)",
            fontsize=11, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)

        # Add frequency annotation
        freq_expected = 1.2 / period.eta
        freq_realized = data['stats']['frequency']
        ax.text(
            0.98, 0.95,
            f"Expected: {freq_expected:.2f}/yr\nRealized: {freq_realized:.2f}/yr",
            transform=ax.transAxes,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9
        )

        if i == 4:
            ax.set_xlabel('Time (years)', fontsize=10)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'climate_forcing_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    # 5. Summary comparison
    print("\n5. Summary Comparison:")
    print("=" * 80)
    print(f"{'Period':<25} {'η':<10} {'Pulses/yr':<12} {'CV':<10} {'Δ Freq':<10}")
    print("-" * 80)

    for period_name, data in results.items():
        period = data['period']
        stats = data['stats']
        freq_expected = 1.2 / period.eta
        freq_realized = stats['frequency']
        cv = stats['std_amplitude'] / stats['mean_amplitude'] if stats['mean_amplitude'] > 0 else 0
        delta_freq = ((freq_realized - freq_expected) / freq_expected) * 100

        print(f"{period_name:<25} {period.eta:<10.3f} {freq_realized:<12.2f} {cv:<10.1%} {delta_freq:+.1f}%")

    print("=" * 80)
    print("\n✓ Climate forcing test complete!")
    print(f"Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())