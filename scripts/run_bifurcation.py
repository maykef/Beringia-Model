#!/usr/bin/env python3
"""Run bifurcation analysis (η sweep)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parameters import ModelParams
from core.integrator import ModelIntegrator
import matplotlib.pyplot as plt
import numpy as np

params = ModelParams()
params.numerical.t_max = 500  # Shorter for sweep
params.numerical.burn_in = 200
integrator = ModelIntegrator(params)

eta_values = np.linspace(0.8, 2.1, 15)
N_mins, N_maxs, N_means = [], [], []

print("Running bifurcation sweep...")
for eta in eta_values:
    result = integrator.simulate(eta, 40000, seed=42)
    N_means.append(result['stats']['N_mean'])
    N_mins.append(result['stats']['N_min'])
    N_maxs.append(result['stats']['N_max'])
    print(f"  η={eta:.2f}: N={N_means[-1]:.0f} [{N_mins[-1]:.0f}, {N_maxs[-1]:.0f}]")

# Plot bifurcation diagram
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(eta_values, N_means, 'b-', linewidth=2, label='Mean')
ax.fill_between(eta_values, N_mins, N_maxs, alpha=0.3, label='Range')
ax.axvline(1.09, color='g', linestyle='--', alpha=0.5, label='HS1')
ax.axvline(1.92, color='r', linestyle='--', alpha=0.5, label='B-A')
ax.set_xlabel('Climate irregularity (η)')
ax.set_ylabel('Population (persons)')
ax.set_title('Bifurcation Diagram: Population vs η')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

output_dir = Path(__file__).parent.parent / 'output'
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'bifurcation.png'
plt.savefig(output_path, dpi=150)
print(f"\nSaved: {output_path}")