#!/usr/bin/env python
"""
Experiment S4: Exchange Rate

Question: Does empirical exchange rate match 1/epsilon^2?

Design: m_p in {10, 25, 50, 100}. For each, binary search for smallest m_u
where combined MSE <= paired-only MSE with m_p.

Output: results/s4_exchange_rate.csv
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
    gt_vals = D_gt[idx]
    est_vals = D_est[idx]
    gt_med = np.median(gt_vals)
    est_med = np.median(est_vals)
    if gt_med > 0:
        gt_vals = gt_vals / gt_med
    if est_med > 0:
        est_vals = est_vals / est_med
    return float(np.mean((est_vals - gt_vals) ** 2))


def compute_mse_for_config(mus, projections, m_p, m_u, n_seeds, n_models, p):
    """Compute average MSE for a given (m_p, m_u) configuration."""
    dkps_obj = DataKernelPerspectiveSpace(coverage_correction=False)
    mses = []

    for seed in range(n_seeds):
        _, gt_dist = generate_synthetic_models(n_models=n_models, p=p, seed=seed)
        # Use the same mus but regenerate to get consistent seeds
        rng_mus = np.random.default_rng(seed)
        mus_s = rng_mus.standard_normal((n_models, p))
        gt_dist_s = np.zeros((n_models, n_models))
        for i in range(n_models):
            for k in range(i + 1, n_models):
                d = np.linalg.norm(mus_s[i] - mus_s[k])
                gt_dist_s[i, k] = gt_dist_s[k, i] = d

        m_total = m_p + m_u
        alpha = m_p / m_total if m_total > 0 else 1.0

        df = sample_responses(mus_s, projections, m_total, alpha, seed=seed + 20000)

        model_names, _, shared_queries_dict, alpha_matrix = dkps_obj._partition_queries(df)
        n = len(model_names)

        D_paired = dkps_obj._compute_paired_distances(df, model_names, shared_queries_dict)
        D_unpaired = dkps_obj._compute_unpaired_distances(df, model_names)
        D_paired_norm = dkps_obj._normalize_to_unit_median(D_paired)
        D_unpaired_norm = dkps_obj._normalize_to_unit_median(D_unpaired)

        D_comb = np.zeros((n, n))
        for i in range(n):
            for k in range(i + 1, n):
                a = alpha_matrix[i, k]
                d = a * D_paired_norm[i, k] + (1 - a) * D_unpaired_norm[i, k]
                D_comb[i, k] = D_comb[k, i] = d

        mses.append(distance_mse(D_comb, gt_dist_s))

    return np.mean(mses)


def run_s4(n_models=20, p=50, n_seeds=20, m_p_values=None):
    if m_p_values is None:
        m_p_values = [10, 25, 50, 100]

    n_projections = 5000
    projections = generate_query_projections(n_projections, p, seed=42)

    results = []

    for m_p in tqdm(m_p_values, desc='S4 m_p values'):
        # Baseline: paired-only MSE with m_p shared queries
        dkps_obj = DataKernelPerspectiveSpace(coverage_correction=False)
        baseline_mses = []
        for seed in range(n_seeds):
            mus_s, gt_dist_s = generate_synthetic_models(n_models=n_models, p=p, seed=seed)
            df_baseline = sample_responses(mus_s, projections, m_p, alpha=1.0, seed=seed + 20000)
            model_names, _, shared_queries_dict, _ = dkps_obj._partition_queries(df_baseline)
            D_paired = dkps_obj._compute_paired_distances(df_baseline, model_names, shared_queries_dict)
            baseline_mses.append(distance_mse(D_paired, gt_dist_s))
        baseline_mse = np.mean(baseline_mses)

        # Binary search for smallest m_u where combined MSE <= baseline
        lo, hi = 0, 2000
        best_m_u = hi

        while lo <= hi:
            mid = (lo + hi) // 2
            if mid == 0:
                lo = 1
                continue
            mse = compute_mse_for_config(None, projections, m_p, mid, n_seeds, n_models, p)
            if mse <= baseline_mse:
                best_m_u = mid
                hi = mid - 1
            else:
                lo = mid + 1

        # Compute epsilon = sqrt(m_p / m_total) where m_total = m_p + best_m_u
        m_total = m_p + best_m_u
        epsilon = np.sqrt(m_p / m_total) if m_total > 0 else 1.0
        empirical_rate = best_m_u / m_p if m_p > 0 else np.nan
        theoretical_rate = 1.0 / (epsilon ** 2) if epsilon > 0 else np.nan

        results.append({
            'm_p': m_p,
            'm_u': best_m_u,
            'epsilon': epsilon,
            'empirical_rate': empirical_rate,
            'theoretical_rate': theoretical_rate,
            'baseline_mse': baseline_mse,
        })
        print(f'm_p={m_p}, m_u={best_m_u}, epsilon={epsilon:.3f}, '
              f'empirical_rate={empirical_rate:.2f}, theoretical_rate={theoretical_rate:.2f}')

    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/s4_exchange_rate.csv', index=False)
    print(f'S4 results saved to results/s4_exchange_rate.csv')
    print(results_df)
    return results_df


def plot_s4(results_df=None, csv_path='results/s4_exchange_rate.csv'):
    if results_df is None:
        results_df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left panel: empirical vs theoretical exchange rate
    ax = axes[0]
    ax.plot(results_df['m_p'], results_df['empirical_rate'],
            'o-', color='#2ca02c', label='Empirical $m_u / m_p$', markersize=8, linewidth=1.5)
    ax.plot(results_df['m_p'], results_df['theoretical_rate'],
            's--', color='#d62728', label=r'Theoretical $1/\varepsilon^2$', markersize=8, linewidth=1.5)
    ax.set_xlabel('$m_p$ (paired queries)', fontsize=12)
    ax.set_ylabel('Exchange rate', fontsize=12)
    ax.set_title('S4: Unpaired-to-paired exchange rate', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right panel: m_u vs m_p
    ax = axes[1]
    ax.plot(results_df['m_p'], results_df['m_u'],
            'o-', color='#1f77b4', markersize=8, linewidth=1.5, label='$m_u$ needed')
    # Overlay reference lines for different rates
    m_p_range = np.linspace(results_df['m_p'].min(), results_df['m_p'].max(), 50)
    for rate, ls in [(5, ':'), (10, '--'), (20, '-.')]:
        ax.plot(m_p_range, rate * m_p_range, ls, color='gray', alpha=0.5,
                label=f'{rate}:1 ratio')
    ax.set_xlabel('$m_p$ (paired queries)', fontsize=12)
    ax.set_ylabel('$m_u$ (unpaired queries needed)', fontsize=12)
    ax.set_title('S4: Unpaired queries to match paired MSE', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('results/s4_exchange_rate.png', dpi=150)
    fig.savefig('results/s4_exchange_rate.pdf')
    plt.close(fig)
    print('S4 plots saved to results/s4_exchange_rate.{png,pdf}')


if __name__ == '__main__':
    df = run_s4()
    plot_s4(df)
