#!/usr/bin/env python3
"""Run ensemble analysis across all climate periods."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parameters import ModelParams
from config.scenarios import SCENARIOS
from core.integrator import ModelIntegrator
import matplotlib.pyplot as plt
import numpy as np

params = ModelParams()
params.numerical.n_ensemble = 10  # Quick test
integrator = ModelIntegrator(params)

results = {}
for name, scenario in SCENARIOS.items():
    print(f"\nRunning {scenario.name}...")
    ensemble = integrator.ensemble(scenario.eta, scenario.amplitude, n_runs=10)
    results[name] = ensemble

# Plot ensemble means
fig, ax = plt.subplots(figsize=(10, 6))
names = [SCENARIOS[k].name for k in results.keys()]
etas = [SCENARIOS[k].eta for k in results.keys()]
N_means = [results[k]['ensemble_stats']['N_ensemble_mean'] for k in results.keys()]
N_stds = [results[k]['ensemble_stats']['N_ensemble_std'] for k in results.keys()]

ax.errorbar(etas, N_means, yerr=N_stds, fmt='o-', capsize=5)
for i, name in enumerate(names):
    ax.annotate(name, (etas[i], N_means[i]), xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('Climate irregularity (η)')
ax.set_ylabel('Mean population (persons)')
ax.set_title('Population vs Climate Irregularity (Ensemble)')
ax.grid(True, alpha=0.3)
plt.tight_layout()

output_dir = Path(__file__).parent.parent / 'output'
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'ensemble_summary.png'
plt.savefig(output_path, dpi=150)
print(f"\nSaved: {output_path}")