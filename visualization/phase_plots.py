"""Phase space visualization."""

import matplotlib.pyplot as plt
import numpy as np

def plot_phase_portrait(result, save_path=None):
    """2D phase portraits."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # F_a vs N
    axes[0].plot(result['F_a'], result['N'], 'b-', alpha=0.5, linewidth=0.5)
    axes[0].set_xlabel('Fat (kg/person)')
    axes[0].set_ylabel('Population (persons)')
    axes[0].set_title('Fat-Population Phase Space')
    axes[0].grid(True, alpha=0.3)

    # P vs N
    axes[1].plot(result['P']/1000, result['N'], 'g-', alpha=0.5, linewidth=0.5)
    axes[1].set_xlabel('Protein (tonnes)')
    axes[1].set_ylabel('Population (persons)')
    axes[1].set_title('Resource-Population Phase Space')
    axes[1].grid(True, alpha=0.3)

    # P vs F_a
    axes[2].plot(result['P']/1000, result['F_a'], 'r-', alpha=0.5, linewidth=0.5)
    axes[2].set_xlabel('Protein (tonnes)')
    axes[2].set_ylabel('Fat (kg/person)')
    axes[2].set_title('Resource-Fat Phase Space')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes

def plot_attractor(F_a, N, V, save_path=None):
    """Plot quasi-potential landscape."""
    fig, ax = plt.subplots(figsize=(10, 8))

    c = ax.contourf(F_a, N, V, levels=20, cmap='viridis')
    ax.contour(F_a, N, V, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    plt.colorbar(c, ax=ax, label='Quasi-potential V')
    ax.set_xlabel('Fat (kg/person)')
    ax.set_ylabel('Population (persons)')
    ax.set_title('Quasi-Potential Landscape')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


if __name__ == "__main__":
    """Demo: Phase portraits for HS1"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config.parameters import ModelParams
    from config.scenarios import get_scenario
    from core.integrator import ModelIntegrator

    print("Running HS1 simulation for phase space visualization...")
    params = ModelParams()
    integrator = ModelIntegrator(params)
    scenario = get_scenario('HS1')

    result = integrator.simulate(scenario.eta, scenario.amplitude, seed=42)

    print(f"Creating phase portraits...")

    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'phase_portraits_HS1.png'

    plot_phase_portrait(result, str(output_path))
    print(f"Saved: {output_path}")

    # Also create 3D trajectory plot
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(result['times'])))

    # Plot trajectory
    for i in range(len(result['times']) - 1):
        ax.plot(result['P'][i:i+2]/1000,
                result['F_a'][i:i+2],
                result['N'][i:i+2],
                color=colors[i], alpha=0.6, linewidth=0.8)

    ax.set_xlabel('Protein (tonnes)', fontsize=11)
    ax.set_ylabel('Fat (kg/person)', fontsize=11)
    ax.set_zlabel('Population (persons)', fontsize=11)
    ax.set_title(f'3D Phase Space Trajectory: {scenario.name}', fontsize=12, fontweight='bold')

    output_path_3d = output_dir / 'phase_3d_HS1.png'
    plt.savefig(output_path_3d, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path_3d}")
