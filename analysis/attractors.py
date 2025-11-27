"""Quasi-potential landscape analysis."""

import numpy as np
from scipy.stats import gaussian_kde

def compute_quasi_potential(F_a, N, bins=50):
    """
    Compute quasi-potential landscape V(F_a, N) = -log(P(F_a, N)).

    Returns:
        F_edges, N_edges, V (2D array)
    """
    # Create 2D histogram
    H, F_edges, N_edges = np.histogram2d(F_a, N, bins=bins)

    # Convert to probability
    P = H / H.sum()
    P[P == 0] = 1e-10  # Avoid log(0)

    # Quasi-potential
    V = -np.log(P)
    V = V - V.min()  # Normalize

    return F_edges, N_edges, V.T

def find_attractors(F_a, N, V, threshold=2.0):
    """Find local minima (attractors) in quasi-potential."""
    from scipy.ndimage import minimum_filter

    # Local minima
    local_min = (V == minimum_filter(V, size=3))

    # Above threshold
    significant = (V < threshold)

    attractors = local_min & significant
    return attractors


if __name__ == "__main__":
    """Demo: Compute quasi-potential for HS1 simulation"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config.parameters import ModelParams
    from config.scenarios import get_scenario
    from core.integrator import ModelIntegrator
    import matplotlib.pyplot as plt

    print("Running HS1 simulation for quasi-potential analysis...")
    params = ModelParams()
    integrator = ModelIntegrator(params)
    scenario = get_scenario('HS1')

    result = integrator.simulate(scenario.eta, scenario.amplitude, seed=42)

    print("Computing quasi-potential landscape...")
    F_edges, N_edges, V = compute_quasi_potential(result['F_a'], result['N'], bins=40)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.contourf(F_edges[:-1], N_edges[:-1], V, levels=20, cmap='viridis')
    ax.contour(F_edges[:-1], N_edges[:-1], V, levels=10, colors='white', alpha=0.3)
    plt.colorbar(c, ax=ax, label='Quasi-potential V')
    ax.set_xlabel('Fat (kg/person)')
    ax.set_ylabel('Population (persons)')
    ax.set_title(f'Quasi-Potential Landscape: {scenario.name}')

    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'attractor_HS1.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
