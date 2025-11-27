"""Bifurcation analysis tools."""

import numpy as np

def parameter_sweep(integrator, param_name, param_values, amplitude=40000, seed=42):
    """
    Sweep parameter and collect statistics.

    Args:
        integrator: ModelIntegrator instance
        param_name: 'eta' or other parameter
        param_values: Array of values to sweep

    Returns:
        results: Dict with statistics for each value
    """
    results = {
        'param_values': param_values,
        'N_mean': [],
        'N_std': [],
        'N_min': [],
        'N_max': [],
        'F_a_mean': [],
        'P_mean': [],
    }

    for val in param_values:
        if param_name == 'eta':
            result = integrator.simulate(val, amplitude, seed=seed)
        else:
            raise ValueError(f"Parameter {param_name} not supported")

        stats = result['stats']
        results['N_mean'].append(stats['N_mean'])
        results['N_std'].append(stats['N_std'])
        results['N_min'].append(stats['N_min'])
        results['N_max'].append(stats['N_max'])
        results['F_a_mean'].append(stats['F_a_mean'])
        results['P_mean'].append(stats['P_mean'])

    return results


if __name__ == "__main__":
    """Demo: η parameter sweep (bifurcation diagram)"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config.parameters import ModelParams
    from core.integrator import ModelIntegrator
    import matplotlib.pyplot as plt

    print("Running bifurcation analysis (η sweep)...")
    params = ModelParams()
    params.numerical.t_max = 500  # Shorter for sweep
    params.numerical.burn_in = 200
    integrator = ModelIntegrator(params)

    eta_values = np.linspace(0.8, 2.1, 20)
    print(f"Sweeping η from {eta_values[0]:.2f} to {eta_values[-1]:.2f} ({len(eta_values)} points)")

    results = parameter_sweep(integrator, 'eta', eta_values, amplitude=40000, seed=42)

    # Plot bifurcation diagram
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Population bifurcation
    ax = axes[0]
    ax.plot(results['param_values'], results['N_mean'], 'b-', linewidth=2, label='Mean')
    ax.fill_between(results['param_values'],
                     results['N_min'],
                     results['N_max'],
                     alpha=0.3, label='Range')
    ax.axvline(1.09, color='g', linestyle='--', alpha=0.5, label='HS1')
    ax.axvline(1.92, color='r', linestyle='--', alpha=0.5, label='B-A')
    ax.set_ylabel('Population (persons)', fontsize=11)
    ax.set_title('Population Bifurcation Diagram', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fat bifurcation
    ax = axes[1]
    ax.plot(results['param_values'], results['F_a_mean'], 'g-', linewidth=2, label='Mean Fat')
    ax.axvline(1.09, color='g', linestyle='--', alpha=0.5, label='HS1')
    ax.axvline(1.92, color='r', linestyle='--', alpha=0.5, label='B-A')
    ax.set_xlabel('Climate irregularity (η)', fontsize=11)
    ax.set_ylabel('Fat (kg/person)', fontsize=11)
    ax.set_title('Fat Bifurcation Diagram', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'bifurcation_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")

    # Print key findings
    print("\nKey Findings:")
    print(f"  At η=1.09 (HS1): N={results['N_mean'][np.argmin(np.abs(np.array(results['param_values'])-1.09))]:.0f} persons")
    print(f"  At η=1.92 (B-A): N={results['N_mean'][np.argmin(np.abs(np.array(results['param_values'])-1.92))]:.0f} persons")
    print(f"  Population decline: {(1 - results['N_mean'][-1]/results['N_mean'][0])*100:.0f}% from low to high η")