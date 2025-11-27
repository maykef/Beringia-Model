#!/usr/bin/env python3
"""Run baseline Heinrich Stadial 1 simulation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parameters import ModelParams
from config.scenarios import get_scenario
from core.integrator import ModelIntegrator
import matplotlib.pyplot as plt

params = ModelParams()
params.validate()

scenario = get_scenario('HS1')
integrator = ModelIntegrator(params)

print(f"Running {scenario.name} (η={scenario.eta:.3f})")
result = integrator.simulate(scenario.eta, scenario.amplitude, seed=42)

print(f"\nResults:")
print(f"  Population: {result['stats']['N_mean']:.0f} ± {result['stats']['N_std']:.0f}")
print(f"  Fat: {result['stats']['F_a_mean']:.1f} ± {result['stats']['F_a_std']:.1f} kg")

# Quick plot
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].plot(result['times'], result['N'])
axes[0].set_ylabel('Population')
axes[1].plot(result['times'], result['F_a'])
axes[1].set_ylabel('Fat (kg)')
axes[2].plot(result['times'], result['P']/1000)
axes[2].set_ylabel('Protein (tonnes)')
axes[2].set_xlabel('Time (years)')
plt.tight_layout()

# Create output directory and save
output_dir = Path(__file__).parent.parent / 'output'
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'baseline_HS1.png'
plt.savefig(output_path, dpi=150)
print(f"\nSaved: {output_path}")
