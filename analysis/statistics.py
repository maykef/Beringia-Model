"""Statistical analysis of ensemble simulations."""

import numpy as np

def ensemble_statistics(results_list):
    """
    Compute statistics across ensemble members.

    Args:
        results_list: List of simulation results

    Returns:
        stats: Aggregated statistics
    """
    N_means = [r['stats']['N_mean'] for r in results_list]
    F_a_means = [r['stats']['F_a_mean'] for r in results_list]

    return {
        'N_ensemble_mean': np.mean(N_means),
        'N_ensemble_std': np.std(N_means),
        'N_ensemble_cv': np.std(N_means) / np.mean(N_means),
        'F_a_ensemble_mean': np.mean(F_a_means),
        'F_a_ensemble_std': np.std(F_a_means),
    }

def confidence_intervals(data, confidence=0.95):
    """Compute confidence intervals."""
    alpha = 1 - confidence
    lower = np.percentile(data, alpha/2 * 100)
    upper = np.percentile(data, (1 - alpha/2) * 100)
    return lower, upper


if __name__ == "__main__":
    """Demo: Ensemble statistics for HS1"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config.parameters import ModelParams
    from config.scenarios import get_scenario
    from core.integrator import ModelIntegrator
    import matplotlib.pyplot as plt

    print("Running ensemble for HS1 (n=20 realizations)...")
    params = ModelParams()
    integrator = ModelIntegrator(params)
    scenario = get_scenario('HS1')

    # Run ensemble
    ensemble = integrator.ensemble(scenario.eta, scenario.amplitude, n_runs=20, base_seed=42)

    # Extract all runs
    all_N_means = [r['stats']['N_mean'] for r in ensemble['individual_runs']]
    all_F_a_means = [r['stats']['F_a_mean'] for r in ensemble['individual_runs']]

    # Compute statistics
    stats = ensemble_statistics(ensemble['individual_runs'])
    N_ci_lower, N_ci_upper = confidence_intervals(all_N_means)
    F_ci_lower, F_ci_upper = confidence_intervals(all_F_a_means)

    print(f"\nEnsemble Statistics:")
    print(f"  Population:")
    print(f"    Mean: {stats['N_ensemble_mean']:.1f} ± {stats['N_ensemble_std']:.1f} persons")
    print(f"    CV: {stats['N_ensemble_cv']:.1%}")
    print(f"    95% CI: [{N_ci_lower:.0f}, {N_ci_upper:.0f}]")
    print(f"  Fat:")
    print(f"    Mean: {stats['F_a_ensemble_mean']:.1f} ± {stats['F_a_ensemble_std']:.1f} kg")
    print(f"    95% CI: [{F_ci_lower:.1f}, {F_ci_upper:.1f}]")

    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(all_N_means, bins=15, alpha=0.7, edgecolor='black')
    axes[0].axvline(stats['N_ensemble_mean'], color='r', linestyle='--', linewidth=2, label='Mean')
    axes[0].axvline(N_ci_lower, color='orange', linestyle=':', label='95% CI')
    axes[0].axvline(N_ci_upper, color='orange', linestyle=':')
    axes[0].set_xlabel('Mean Population (persons)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Population Distribution Across Ensemble')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(all_F_a_means, bins=15, alpha=0.7, edgecolor='black', color='green')
    axes[1].axvline(stats['F_a_ensemble_mean'], color='r', linestyle='--', linewidth=2, label='Mean')
    axes[1].axvline(F_ci_lower, color='orange', linestyle=':', label='95% CI')
    axes[1].axvline(F_ci_upper, color='orange', linestyle=':')
    axes[1].set_xlabel('Mean Fat (kg/person)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Fat Distribution Across Ensemble')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'ensemble_statistics.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")