"""Time series visualization."""

import matplotlib.pyplot as plt

def plot_timeseries(result, save_path=None):
    """Plot standard 3-panel time series."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    times = result['times']

    # Population
    axes[0].plot(times, result['N'], 'b-', alpha=0.7)
    axes[0].set_ylabel('Population (persons)')
    axes[0].set_title('Population Dynamics')
    axes[0].grid(True, alpha=0.3)

    # Fat
    axes[1].plot(times, result['F_a'], 'g-', alpha=0.7)
    axes[1].set_ylabel('Fat (kg/person)')
    axes[1].set_title('Fat Availability')
    axes[1].grid(True, alpha=0.3)

    # Protein
    axes[2].plot(times, result['P']/1000, 'brown', alpha=0.7)
    axes[2].set_ylabel('Protein (tonnes)')
    axes[2].set_xlabel('Time (years)')
    axes[2].set_title('Resource Dynamics')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


if __name__ == "__main__":
    """Demo: Time series visualization for HS1"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config.parameters import ModelParams
    from config.scenarios import get_scenario
    from core.integrator import ModelIntegrator

    print("Running HS1 simulation for time series visualization...")
    params = ModelParams()
    integrator = ModelIntegrator(params)
    scenario = get_scenario('HS1')

    result = integrator.simulate(scenario.eta, scenario.amplitude, seed=42)

    print(f"Creating time series plot...")
    print(f"  Duration: {len(result['times'])} timesteps")
    print(f"  Population: {result['stats']['N_mean']:.0f} ± {result['stats']['N_std']:.0f}")
    print(f"  Fat: {result['stats']['F_a_mean']:.1f} ± {result['stats']['F_a_std']:.1f} kg")

    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'timeseries_HS1.png'

    plot_timeseries(result, str(output_path))
    print(f"Saved: {output_path}")