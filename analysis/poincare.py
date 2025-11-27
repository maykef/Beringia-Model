"""Poincaré section analysis."""

import numpy as np

def poincare_section(P, F_a, N, plane='P', value=None):
    """
    Extract Poincaré section at specified plane.

    Args:
        P, F_a, N: Time series
        plane: 'P', 'F_a', or 'N'
        value: Plane value (median if None)

    Returns:
        section_data: Points where trajectory crosses plane
    """
    if plane == 'P':
        var = P
        others = (F_a, N)
    elif plane == 'F_a':
        var = F_a
        others = (P, N)
    else:
        var = N
        others = (P, F_a)

    if value is None:
        value = np.median(var)

    # Find crossings
    crossings = []
    for i in range(len(var) - 1):
        if (var[i] <= value <= var[i+1]) or (var[i+1] <= value <= var[i]):
            # Linear interpolation
            frac = (value - var[i]) / (var[i+1] - var[i] + 1e-10)
            point = [o[i] + frac * (o[i+1] - o[i]) for o in others]
            crossings.append(point)

    return np.array(crossings)


if __name__ == "__main__":
    """Demo: Compute Poincaré section for HS1"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config.parameters import ModelParams
    from config.scenarios import get_scenario
    from core.integrator import ModelIntegrator
    import matplotlib.pyplot as plt

    print("Running HS1 simulation for Poincaré section...")
    params = ModelParams()
    integrator = ModelIntegrator(params)
    scenario = get_scenario('HS1')

    result = integrator.simulate(scenario.eta, scenario.amplitude, seed=42)

    print("Computing Poincaré section at median protein...")
    section = poincare_section(result['P'], result['F_a'], result['N'], plane='P')

    print(f"Found {len(section)} crossing points")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(section[:, 0], section[:, 1], c=np.arange(len(section)),
               cmap='viridis', s=20, alpha=0.6)
    ax.set_xlabel('Fat (kg/person)')
    ax.set_ylabel('Population (persons)')
    ax.set_title(f'Poincaré Section (P = median): {scenario.name}')
    ax.grid(True, alpha=0.3)

    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'poincare_HS1.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")