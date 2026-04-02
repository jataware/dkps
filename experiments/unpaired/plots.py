import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'matplotlib'))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ESTIMATOR_COLORS = {
    'paired_only': '#0f766e',
    'unpaired_only': '#c2410c',
    'combined': '#1d4ed8',
    'linear_mmd': '#6b21a8',
    'coverage_adjusted_linear': '#0f766e',
    'coverage_adjusted_rbf': '#b91c1c',
    'oracle_queries': '#0f766e',
    'noisy_queries': '#c2410c',
    'embedding_pca': '#1d4ed8',
}

BLOCK0_STUDIES = [
    ('alpha', 'paired fraction alpha'),
    ('latent_noise', 'latent noise std (1/s)'),
    ('obs_noise', 'observation noise std (1/t)'),
    ('extra_noise_dims', 'extra observed noise dims'),
]


def _aggregate(df, group_cols, value_col):
    grouped = (
        df.groupby(group_cols, dropna=False)[value_col]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    grouped['sem'] = grouped['std'].fillna(0.0) / np.sqrt(grouped['count'].clip(lower=1))
    return grouped


def _finalize_figure(fig):
    fig.tight_layout()
    return fig


def plot_s0_results(df):
    has_spearman = 'spearman' in df.columns
    metrics = [('mse', 'off-diagonal MSE')]
    if has_spearman:
        metrics.append(('spearman', 'Spearman correlation'))

    coverage_modes = list(df['coverage_mode'].drop_duplicates())
    m_totals = sorted(df['m_total'].drop_duplicates())

    n_rows = len(metrics) * len(coverage_modes)
    n_cols = len(m_totals)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3.8 * n_rows),
        sharex=True,
        squeeze=False,
    )

    for metric_idx, (metric, ylabel) in enumerate(metrics):
        summary = _aggregate(df, ['coverage_mode', 'm_total', 'alpha_requested', 'estimator'], metric)
        for cov_idx, coverage_mode in enumerate(coverage_modes):
            row_idx = metric_idx * len(coverage_modes) + cov_idx
            for col_idx, m_total in enumerate(m_totals):
                ax = axes[row_idx][col_idx]
                subset = summary[
                    (summary['coverage_mode'] == coverage_mode) &
                    (summary['m_total'] == m_total)
                ]
                for estimator in subset['estimator'].drop_duplicates():
                    curve = subset[subset['estimator'] == estimator].sort_values('alpha_requested')
                    color = ESTIMATOR_COLORS.get(estimator, None)
                    ax.plot(curve['alpha_requested'], curve['mean'], marker='o', linewidth=1.5, color=color, label=estimator)
                    ax.fill_between(
                        curve['alpha_requested'],
                        curve['mean'] - curve['sem'],
                        curve['mean'] + curve['sem'],
                        color=color,
                        alpha=0.18,
                    )
                ax.set_title(f'{coverage_mode}, m={m_total}')
                ax.set_xlabel('paired fraction alpha')
                if col_idx == 0:
                    ax.set_ylabel(ylabel)
                ax.legend(fontsize=8, frameon=False)
                ax.grid(alpha=0.25, linewidth=0.6)

    return _finalize_figure(fig)


def plot_s1_results(summary_df):
    summary = _aggregate(summary_df.dropna(subset=['exchange_rate']), ['coverage_mode', 'epsilon', 'm_p'], 'exchange_rate')
    theory = (
        summary_df[['coverage_mode', 'epsilon', 'theory_exchange_rate']]
        .drop_duplicates()
        .sort_values(['coverage_mode', 'epsilon'])
    )
    coverage_modes = list(summary_df['coverage_mode'].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(coverage_modes),
        figsize=(5.2 * len(coverage_modes), 4.0),
        sharey=True,
        squeeze=False,
    )

    for idx, coverage_mode in enumerate(coverage_modes):
        ax = axes[0][idx]
        subset = summary[summary['coverage_mode'] == coverage_mode]
        for m_p in sorted(subset['m_p'].drop_duplicates()):
            curve = subset[subset['m_p'] == m_p].sort_values('epsilon')
            ax.plot(curve['epsilon'], curve['mean'], marker='o', linewidth=2, label=f'm_p={m_p}')
            ax.fill_between(
                curve['epsilon'],
                curve['mean'] - curve['sem'],
                curve['mean'] + curve['sem'],
                alpha=0.18,
            )

        theory_curve = theory[theory['coverage_mode'] == coverage_mode].sort_values('epsilon')
        if len(theory_curve):
            ax.plot(
                theory_curve['epsilon'],
                theory_curve['theory_exchange_rate'],
                linestyle='--',
                color='black',
                linewidth=1.8,
                label='theory 1/epsilon^2',
            )
        ax.set_title(f'{coverage_mode}')
        ax.set_xlabel('query overlap epsilon')
        if idx == 0:
            ax.set_ylabel('exchange rate m_u / m_p')
        ax.set_yscale('log')
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25, linewidth=0.6)

    return _finalize_figure(fig)


def plot_s2_results(df):
    summary = _aggregate(df, ['coverage_mode', 'alpha_requested', 'd_sep', 'estimator'], 'mse')
    coverage_modes = list(summary['coverage_mode'].drop_duplicates())
    alphas = sorted(summary['alpha_requested'].drop_duplicates())
    fig, axes = plt.subplots(
        len(coverage_modes),
        len(alphas),
        figsize=(4.7 * len(alphas), 3.8 * len(coverage_modes)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row_idx, coverage_mode in enumerate(coverage_modes):
        for col_idx, alpha in enumerate(alphas):
            ax = axes[row_idx][col_idx]
            subset = summary[
                (summary['coverage_mode'] == coverage_mode) &
                (summary['alpha_requested'] == alpha)
            ]
            for estimator in subset['estimator'].drop_duplicates():
                curve = subset[subset['estimator'] == estimator].sort_values('d_sep')
                color = ESTIMATOR_COLORS.get(estimator, None)
                ax.plot(curve['d_sep'], curve['mean'], marker='o', linewidth=2, color=color, label=estimator)
                ax.fill_between(
                    curve['d_sep'],
                    curve['mean'] - curve['sem'],
                    curve['mean'] + curve['sem'],
                    color=color,
                    alpha=0.18,
                )
            ax.set_title(f'{coverage_mode}, alpha={alpha:g}')
            ax.set_xlabel('component separation d')
            if col_idx == 0:
                ax.set_ylabel('off-diagonal MSE')
            ax.legend(fontsize=8, frameon=False)
            ax.grid(alpha=0.25, linewidth=0.6)

    return _finalize_figure(fig)


def plot_s3_results(df):
    has_spearman = 'spearman' in df.columns
    metrics = [('mse', 'off-diagonal MSE')]
    if has_spearman:
        metrics.append(('spearman', 'Spearman correlation'))

    coverage_modes = list(df['coverage_mode'].drop_duplicates())

    n_rows = len(metrics)
    n_cols = len(coverage_modes)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 3.8 * n_rows),
        sharex=True,
        squeeze=False,
    )

    for metric_idx, (metric, ylabel) in enumerate(metrics):
        summary = _aggregate(df, ['coverage_mode', 'noise_level', 'estimator'], metric)
        for cov_idx, coverage_mode in enumerate(coverage_modes):
            ax = axes[metric_idx][cov_idx]
            subset = summary[summary['coverage_mode'] == coverage_mode]
            for estimator in subset['estimator'].drop_duplicates():
                curve = subset[subset['estimator'] == estimator].sort_values('noise_level')
                color = ESTIMATOR_COLORS.get(estimator, None)
                ax.plot(curve['noise_level'], curve['mean'], marker='o', linewidth=1.5, color=color, label=estimator)
                ax.fill_between(
                    curve['noise_level'],
                    curve['mean'] - curve['sem'],
                    curve['mean'] + curve['sem'],
                    color=color,
                    alpha=0.18,
                )
            ax.set_title(f'{coverage_mode}')
            ax.set_xlabel('inversion noise level')
            if cov_idx == 0:
                ax.set_ylabel(ylabel)
            ax.legend(fontsize=8, frameon=False)
            ax.grid(alpha=0.25, linewidth=0.6)

    return _finalize_figure(fig)


def save_block1_plots(output_dir, s0_df=None, s1_summary_df=None, s2_df=None, s3_df=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    if s0_df is not None and len(s0_df):
        fig = plot_s0_results(s0_df)
        path = output_dir / 's0_mse_vs_alpha.png'
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)

    if s1_summary_df is not None and len(s1_summary_df):
        fig = plot_s1_results(s1_summary_df)
        path = output_dir / 's1_exchange_rate.png'
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)

    if s2_df is not None and len(s2_df):
        fig = plot_s2_results(s2_df)
        path = output_dir / 's2_mse_vs_d_sep.png'
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)

    if s3_df is not None and len(s3_df):
        fig = plot_s3_results(s3_df)
        path = output_dir / 's3_mse_vs_noise.png'
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)

    return saved


def plot_block0_results(df, metric):
    summary = _aggregate(df, ['study', 'x_value', 'estimator'], metric)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), squeeze=False)

    for ax, (study, xlabel) in zip(axes.ravel(), BLOCK0_STUDIES):
        subset = summary[summary['study'] == study]
        for estimator in subset['estimator'].drop_duplicates():
            curve = subset[subset['estimator'] == estimator].sort_values('x_value')
            color = ESTIMATOR_COLORS.get(estimator, None)
            ax.plot(curve['x_value'], curve['mean'], marker='o', linewidth=2, color=color, label=estimator)
            ax.fill_between(
                curve['x_value'],
                curve['mean'] - curve['sem'],
                curve['mean'] + curve['sem'],
                color=color,
                alpha=0.18,
            )
        ax.set_title(study.replace('_', ' '))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric.replace('_', ' '))
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25, linewidth=0.6)

    return _finalize_figure(fig)


def save_block0_plots(output_dir, df):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for metric, filename in (
        ('mse', 'block0_mse.png'),
        ('max_abs_error', 'block0_max_abs_error.png'),
    ):
        fig = plot_block0_results(df, metric)
        path = output_dir / filename
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)

    return saved
