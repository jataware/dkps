#!/usr/bin/env python
"""
Experiment S1: Coverage Interaction

Question: How does estimator error depend on alpha (fraction of shared queries)?

Design: m_total=1000, alpha in {0.0, 0.05, 0.1, 0.2, 0.5, 1.0}, 50 seeds.
For each (alpha, seed): compute paired-only, unpaired-only, and combined
distance matrices; MSE vs ground truth.

Output: results/s1_coverage_interaction.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_generation import generate_synthetic_models, generate_query_projections, sample_responses
from dkps import DataKernelPerspectiveSpace


def distance_mse(D_est, D_gt):
    """MSE between upper-triangular entries of two distance matrices."""
    idx = np.triu_indices_from(D_gt, k=1)
    # Normalize both to unit median for fair comparison
    gt_vals = D_gt[idx]
    est_vals = D_est[idx]
    gt_med = np.median(gt_vals)
    est_med = np.median(est_vals)
    if gt_med > 0:
        gt_vals = gt_vals / gt_med
    if est_med > 0:
        est_vals = est_vals / est_med
    return float(np.mean((est_vals - gt_vals) ** 2))


def run_s1(n_models=20, p=50, m_total=1000, n_seeds=5, alphas=None):
    if alphas is None:
        alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    n_projections = m_total * 3  # large pool of projections
    results = []

    for seed in tqdm(range(n_seeds), desc='S1 seeds'):
        mus, gt_dist = generate_synthetic_models(n_models=n_models, p=p, seed=seed)
        projections = generate_query_projections(n_projections, p, seed=seed + 10000)

        for alpha in alphas:
            df = sample_responses(mus, projections, m_total, alpha, seed=seed + 20000)

            # Combined estimator
            dkps_combined = DataKernelPerspectiveSpace(coverage_correction=False)
            emb_combined = dkps_combined.fit_transform(df, return_dict=False)
            D_combined = dkps_combined._last_dist_matrix if hasattr(dkps_combined, '_last_dist_matrix') else None

            # We need the raw distance matrices, so compute them directly
            model_names, query_sets, shared_queries_dict, alpha_matrix = dkps_combined._partition_queries(df)
            n = len(model_names)

            # Paired distances
            if alpha > 0:
                D_paired = dkps_combined._compute_paired_distances(df, model_names, shared_queries_dict)
                D_paired_norm = dkps_combined._normalize_to_unit_median(D_paired)
                mse_paired = distance_mse(D_paired, gt_dist)
            else:
                D_paired_norm = None
                mse_paired = np.nan

            # Unpaired distances
            D_unpaired = dkps_combined._compute_unpaired_distances(df, model_names)
            D_unpaired_norm = dkps_combined._normalize_to_unit_median(D_unpaired)
            mse_unpaired = distance_mse(D_unpaired, gt_dist)

            # Combined distances
            eff_alpha = alpha_matrix.copy()
            D_comb = np.zeros((n, n))
            for i in range(n):
                for k in range(i + 1, n):
                    a = eff_alpha[i, k]
                    d = 0.0
                    if D_paired_norm is not None and not np.isnan(D_paired_norm[i, k]):
                        d += a * D_paired_norm[i, k]
                    else:
                        a = 0.0
                    d += (1.0 - a) * D_unpaired_norm[i, k]
                    D_comb[i, k] = D_comb[k, i] = d
            mse_combined = distance_mse(D_comb, gt_dist)

            results.append({'alpha': alpha, 'seed': seed, 'method': 'paired', 'mse': mse_paired})
            results.append({'alpha': alpha, 'seed': seed, 'method': 'unpaired', 'mse': mse_unpaired})
            results.append({'alpha': alpha, 'seed': seed, 'method': 'combined', 'mse': mse_combined})

    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/s1_coverage_interaction.csv', index=False)
    print(f'S1 results saved to results/s1_coverage_interaction.csv')
    print(results_df.groupby(['alpha', 'method'])['mse'].mean().unstack())
    return results_df


def plot_s1(results_df=None, csv_path='results/s1_coverage_interaction.csv'):
    if results_df is None:
        results_df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    method_styles = {
        'paired':   {'color': '#1f77b4', 'marker': 's', 'label': 'Paired only'},
        'unpaired': {'color': '#ff7f0e', 'marker': '^', 'label': 'Unpaired only'},
        'combined': {'color': '#2ca02c', 'marker': 'o', 'label': 'Combined'},
    }

    for method, style in method_styles.items():
        sub = results_df[results_df['method'] == method]
        grouped = sub.groupby('alpha')['mse']
        mean = grouped.mean()
        sem = grouped.sem()
        ax.errorbar(
            mean.index, mean.values, yerr=sem.values,
            marker=style['marker'], color=style['color'],
            label=style['label'], capsize=3, linewidth=1.5, markersize=6,
        )

    ax.set_xlabel(r'$\alpha$ (fraction of shared queries)', fontsize=12)
    ax.set_ylabel('MSE (normalized distances)', fontsize=12)
    ax.set_title('S1: Estimator error vs. shared query fraction', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('results/s1_coverage_interaction.png', dpi=150)
    fig.savefig('results/s1_coverage_interaction.pdf')
    plt.close(fig)
    print('S1 plots saved to results/s1_coverage_interaction.{png,pdf}')


if __name__ == '__main__':
    df = run_s1()
    plot_s1(df)
