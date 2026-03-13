#!/usr/bin/env python
"""
Experiment S2: Query Distribution Mismatch

Question: Does coverage correction fix bias under query distribution mismatch?

Design: alpha=0.2, KL in {0.0, 0.5, 1.0, 2.0, 5.0, inf}, 50 seeds.
Three estimators:
  (a) naive uncorrected
  (b) coverage-adjusted with oracle densities
  (c) coverage-adjusted with KDE

Output: results/s2_query_mismatch.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_generation import generate_synthetic_models, generate_query_projections, sample_responses
from dkps import DataKernelPerspectiveSpace
from dkps.coverage import estimate_coverage_weights


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


def compute_oracle_weights(df, model_names, query_distribution_kl):
    """
    Compute oracle coverage weights from known query distributions.

    When KL=0, all models share the same query distribution so weight=1.
    When KL=inf, distributions are disjoint so weight -> 0.
    """
    n = len(model_names)
    weights = {}
    for i in range(n):
        for k in range(i + 1, n):
            mi, mk = model_names[i], model_names[k]
            if query_distribution_kl == 0.0:
                w = 1.0
            elif np.isinf(query_distribution_kl):
                w = 0.0
            else:
                # Approximate: overlap decreases exponentially with KL
                w = float(np.exp(-query_distribution_kl * abs(i - k) / n))
            weights[(mi, mk)] = w
            weights[(mk, mi)] = w
    return weights


def apply_coverage_to_unpaired(D_unpaired, model_names, cov_weights):
    """Apply coverage weights to unpaired distance matrix."""
    n = len(model_names)
    D_adj = D_unpaired.copy()
    med = np.median(D_unpaired[np.triu_indices(n, k=1)]) if n > 1 else 1.0
    for i in range(n):
        for k in range(i + 1, n):
            mi, mk = model_names[i], model_names[k]
            w = cov_weights.get((mi, mk), 1.0)
            adjusted = w * D_unpaired[i, k] + (1 - w) * med
            D_adj[i, k] = D_adj[k, i] = adjusted
    return D_adj


def run_s2(n_models=20, p=50, m_total=1000, alpha=0.2, n_seeds=50, kls=None):
    if kls is None:
        kls = [0.0, 0.5, 1.0, 2.0, 5.0, np.inf]

    n_projections = m_total * 3
    results = []

    dkps_obj = DataKernelPerspectiveSpace(coverage_correction=False)

    for seed in tqdm(range(n_seeds), desc='S2 seeds'):
        mus, gt_dist = generate_synthetic_models(n_models=n_models, p=p, seed=seed)
        projections = generate_query_projections(n_projections, p, seed=seed + 10000)

        for kl in kls:
            df = sample_responses(mus, projections, m_total, alpha, seed=seed + 20000,
                                  query_distribution_kl=kl)

            model_names, query_sets, shared_queries_dict, alpha_matrix = dkps_obj._partition_queries(df)
            n = len(model_names)

            D_paired = dkps_obj._compute_paired_distances(df, model_names, shared_queries_dict)
            D_unpaired = dkps_obj._compute_unpaired_distances(df, model_names)

            # (a) Naive uncorrected
            D_unpaired_norm = dkps_obj._normalize_to_unit_median(D_unpaired)
            D_paired_norm = dkps_obj._normalize_to_unit_median(D_paired)
            D_naive = np.zeros((n, n))
            for i in range(n):
                for k in range(i + 1, n):
                    a = alpha_matrix[i, k]
                    d = a * D_paired_norm[i, k] + (1 - a) * D_unpaired_norm[i, k]
                    D_naive[i, k] = D_naive[k, i] = d
            mse_naive = distance_mse(D_naive, gt_dist)

            # (b) Coverage-adjusted with oracle densities
            oracle_weights = compute_oracle_weights(df, model_names, kl)
            D_unpaired_oracle = apply_coverage_to_unpaired(D_unpaired, model_names, oracle_weights)
            D_unpaired_oracle_norm = dkps_obj._normalize_to_unit_median(D_unpaired_oracle)
            D_oracle = np.zeros((n, n))
            for i in range(n):
                for k in range(i + 1, n):
                    a = alpha_matrix[i, k]
                    d = a * D_paired_norm[i, k] + (1 - a) * D_unpaired_oracle_norm[i, k]
                    D_oracle[i, k] = D_oracle[k, i] = d
            mse_oracle = distance_mse(D_oracle, gt_dist)

            # (c) Coverage-adjusted with KDE
            embs_by_model = {}
            for m in model_names:
                embs = df[df['model_id'] == m]['embedding'].values
                embs_by_model[m] = np.stack([e for e in embs])
            kde_weights = estimate_coverage_weights(embs_by_model)
            D_unpaired_kde = apply_coverage_to_unpaired(D_unpaired, model_names, kde_weights)
            D_unpaired_kde_norm = dkps_obj._normalize_to_unit_median(D_unpaired_kde)
            D_kde = np.zeros((n, n))
            for i in range(n):
                for k in range(i + 1, n):
                    a = alpha_matrix[i, k]
                    d = a * D_paired_norm[i, k] + (1 - a) * D_unpaired_kde_norm[i, k]
                    D_kde[i, k] = D_kde[k, i] = d
            mse_kde = distance_mse(D_kde, gt_dist)

            kl_label = 'inf' if np.isinf(kl) else str(kl)
            results.append({'kl': kl_label, 'seed': seed, 'method': 'naive', 'mse': mse_naive})
            results.append({'kl': kl_label, 'seed': seed, 'method': 'oracle', 'mse': mse_oracle})
            results.append({'kl': kl_label, 'seed': seed, 'method': 'kde', 'mse': mse_kde})

    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/s2_query_mismatch.csv', index=False)
    print(f'S2 results saved to results/s2_query_mismatch.csv')
    print(results_df.groupby(['kl', 'method'])['mse'].mean().unstack())
    return results_df


def plot_s2(results_df=None, csv_path='results/s2_query_mismatch.csv'):
    if results_df is None:
        results_df = pd.read_csv(csv_path)

    # Order KL values properly
    kl_order = ['0.0', '0.5', '1.0', '2.0', '5.0', 'inf']
    kl_labels_display = ['0', '0.5', '1', '2', '5', r'$\infty$']
    results_df['kl'] = pd.Categorical(results_df['kl'].astype(str), categories=kl_order, ordered=True)
    results_df = results_df.dropna(subset=['kl'])

    fig, ax = plt.subplots(figsize=(7, 4.5))

    method_styles = {
        'naive':  {'color': '#d62728', 'marker': 'x', 'label': 'Naive (uncorrected)'},
        'oracle': {'color': '#9467bd', 'marker': 'D', 'label': 'Oracle coverage'},
        'kde':    {'color': '#17becf', 'marker': 'o', 'label': 'KDE coverage'},
    }

    for method, style in method_styles.items():
        sub = results_df[results_df['method'] == method]
        grouped = sub.groupby('kl', observed=True)['mse']
        mean = grouped.mean()
        sem = grouped.sem()

        x_pos = list(range(len(mean)))
        ax.errorbar(
            x_pos, mean.values, yerr=sem.values,
            marker=style['marker'], color=style['color'],
            label=style['label'], capsize=3, linewidth=1.5, markersize=6,
        )

    present_kls = [k for k in kl_order if k in results_df['kl'].cat.categories and k in results_df['kl'].values]
    tick_labels = [kl_labels_display[kl_order.index(k)] for k in present_kls]
    ax.set_xticks(range(len(present_kls)))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Query distribution mismatch (KL)', fontsize=12)
    ax.set_ylabel('MSE (normalized distances)', fontsize=12)
    ax.set_title(r'S2: Coverage correction under query mismatch ($\alpha=0.2$)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('results/s2_query_mismatch.png', dpi=150)
    fig.savefig('results/s2_query_mismatch.pdf')
    plt.close(fig)
    print('S2 plots saved to results/s2_query_mismatch.{png,pdf}')


if __name__ == '__main__':
    df = run_s2()
    plot_s2(df)
