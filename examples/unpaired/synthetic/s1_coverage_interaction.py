#!/usr/bin/env python
"""
Experiment S1: Coverage Interaction

Question: How does estimator error depend on alpha (fraction of shared queries)?

Design: m_total=100, alpha in {0.0, 0.05, 0.1, 0.2, 0.5, 1.0}, 100 seeds.
For each (alpha, seed): run DKPS on shared-only, unshared-only, and all data;
compute MSE and Spearman correlation of both internal distance matrix and CMDS
output distances vs ground truth.

Output: results/s1_coverage_interaction.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
from data_generation import generate_synthetic_models, sample_responses
from dkps import DataKernelPerspectiveSpace


def distance_mse(D_est, D_gt):
    """MSE between upper-triangular entries of estimated vs ground truth distance matrices."""
    idx = np.triu_indices_from(D_gt, k=1)
    gt_vals = D_gt[idx]
    est_vals = D_est[idx]
    # Normalize both to unit median so scale doesn't matter
    gt_med = np.median(gt_vals)
    est_med = np.median(est_vals)
    if gt_med > 0:
        gt_vals = gt_vals / gt_med
    if est_med > 0:
        est_vals = est_vals / est_med
    return float(np.mean((est_vals - gt_vals) ** 2))


def distance_spearman(D_est, D_gt):
    """Spearman correlation between upper-triangular entries of estimated vs ground truth distance matrices."""
    idx = np.triu_indices_from(D_gt, k=1)
    gt_vals = D_gt[idx]
    est_vals = D_est[idx]
    r, _ = spearmanr(est_vals, gt_vals)
    return float(r)


def _eval_distances(D_est, D_gt):
    """Compute both MSE and Spearman for a pair of distance matrices."""
    return distance_mse(D_est, D_gt), distance_spearman(D_est, D_gt)


def _run_seed(seed, n_models, p, m_total, alphas):
    """Run all alphas for a single seed. Returns list of result dicts."""
    mus, gt_dist = generate_synthetic_models(n_models=n_models, p=p, seed=seed)
    results = []

    for alpha in alphas:
        df = sample_responses(mus, m_total, alpha, seed=seed + 20000)
        is_shared = df['query_id'].str.startswith('shared')

        dkps = DataKernelPerspectiveSpace(n_components_cmds=8)

        # Paired only (shared queries)
        if alpha > 0:
            emb_paired = dkps.fit_transform(df[is_shared], return_dict=False)
            mse_paired_raw, rho_paired_raw = _eval_distances(dkps.dist_matrix_, gt_dist)
            mse_paired_cmds, rho_paired_cmds = _eval_distances(squareform(pdist(emb_paired)), gt_dist)
        else:
            mse_paired_raw = rho_paired_raw = np.nan
            mse_paired_cmds = rho_paired_cmds = np.nan

        # Unpaired only (private queries)
        if alpha < 1:
            emb_unpaired = dkps.fit_transform(df[~is_shared], return_dict=False)
            mse_unpaired_raw, rho_unpaired_raw = _eval_distances(dkps.dist_matrix_, gt_dist)
            mse_unpaired_cmds, rho_unpaired_cmds = _eval_distances(squareform(pdist(emb_unpaired)), gt_dist)
        else:
            mse_unpaired_raw = rho_unpaired_raw = np.nan
            mse_unpaired_cmds = rho_unpaired_cmds = np.nan

        # Combined (all data)
        emb_combined = dkps.fit_transform(df, return_dict=False)
        mse_combined_raw, rho_combined_raw = _eval_distances(dkps.dist_matrix_, gt_dist)
        mse_combined_cmds, rho_combined_cmds = _eval_distances(squareform(pdist(emb_combined)), gt_dist)

        for metric, vals_p, vals_u, vals_c in [
            ('raw',
             (mse_paired_raw, rho_paired_raw),
             (mse_unpaired_raw, rho_unpaired_raw),
             (mse_combined_raw, rho_combined_raw)),
            ('cmds',
             (mse_paired_cmds, rho_paired_cmds),
             (mse_unpaired_cmds, rho_unpaired_cmds),
             (mse_combined_cmds, rho_combined_cmds)),
        ]:
            for method, (mse, rho) in [('paired', vals_p), ('unpaired', vals_u), ('combined', vals_c)]:
                results.append({
                    'alpha': alpha, 'seed': seed, 'metric': metric,
                    'method': method, 'mse': mse, 'spearman': rho,
                })

    return results


def run_s1(n_models=20, p=50, m_total=100, n_seeds=100, alphas=None, n_workers=None):
    if alphas is None:
        alphas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

    if n_workers is None:
        n_workers = min(n_seeds, os.cpu_count() or 1)

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_run_seed, seed, n_models, p, m_total, alphas): seed
            for seed in range(n_seeds)
        }
        for future in tqdm(as_completed(futures), total=n_seeds, desc='S1 seeds'):
            results.extend(future.result())

    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/s1_coverage_interaction.csv', index=False)
    print(f'S1 results saved to results/s1_coverage_interaction.csv')
    for metric in ['raw', 'cmds']:
        print(f'\n{metric}:')
        sub = results_df[results_df['metric'] == metric]
        print('  MSE:')
        print(sub.groupby(['alpha', 'method'])['mse'].mean().unstack())
        print('  Spearman:')
        print(sub.groupby(['alpha', 'method'])['spearman'].mean().unstack())
    return results_df


def plot_s1(results_df=None, csv_path='results/s1_coverage_interaction.csv'):
    if results_df is None:
        results_df = pd.read_csv(csv_path)

    method_styles = {
        'paired':   {'color': '#1f77b4', 'marker': 's', 'label': 'Paired only'},
        'unpaired': {'color': '#ff7f0e', 'marker': '^', 'label': 'Unpaired only'},
        'combined': {'color': '#2ca02c', 'marker': 'o', 'label': 'Combined'},
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for col, metric, metric_title in [
        (0, 'raw',  'Internal distance matrix'),
        (1, 'cmds', 'CMDS output distances'),
    ]:
        sub_metric = results_df[results_df['metric'] == metric]

        for row, stat, stat_label, use_log in [
            (0, 'spearman', 'Spearman correlation', False),
            (1, 'mse', 'MSE (median-normalized)', True),
        ]:
            ax = axes[row, col]
            for method, style in method_styles.items():
                sub = sub_metric[sub_metric['method'] == method]
                grouped = sub.groupby('alpha')[stat]
                mean = grouped.mean()
                sem = grouped.sem()
                ax.errorbar(
                    mean.index, mean.values, yerr=sem.values,
                    marker=style['marker'], color=style['color'],
                    label=style['label'], capsize=3, linewidth=1.5, markersize=6,
                )

            ax.set_xlabel(r'$\alpha$ (fraction of shared queries)', fontsize=12)
            ax.set_ylabel(stat_label, fontsize=12)
            ax.set_title(f'{metric_title}', fontsize=13)
            ax.legend(fontsize=10)
            if use_log:
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
