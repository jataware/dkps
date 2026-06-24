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
    'rbf_paired': '#0f766e',
    'rbf_unpaired': '#c2410c',
    'rbf_combined': '#1d4ed8',
}

ESTIMATOR_LABELS = {
    'rbf_paired': 'paired',
    'rbf_unpaired': 'unpaired',
    'rbf_combined': 'combined',
}

ESTIMATOR_ORDER = ['rbf_paired', 'rbf_unpaired', 'rbf_combined']

_BASELINE_STYLE = {}

NQ_STYLES = {3: '--', 10: '-', 50: ':'}


def _aggregate(df, group_cols, value_col):
    grouped = (
        df.groupby(group_cols, dropna=False)[value_col]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    grouped['sem'] = grouped['std'].fillna(0.0) / np.sqrt(grouped['count'].clip(lower=1))
    return grouped


def _plot_standard_panel(ax, df, x_col, xlabel, log_x=False, log_y=False):
    """Standard panel: 3 estimators, x vs MAE."""
    summary = _aggregate(df, [x_col, 'estimator'], 'mae')
    for est in ESTIMATOR_ORDER:
        if est not in summary['estimator'].values:
            continue
        curve = summary[summary['estimator'] == est].sort_values(x_col)
        c = ESTIMATOR_COLORS[est]
        label = ESTIMATOR_LABELS[est]
        ls = _BASELINE_STYLE.get(est, '-')
        ax.plot(curve[x_col], curve['mean'], marker='o', lw=1.5, color=c, label=label, linestyle=ls)
        ax.fill_between(curve[x_col], curve['mean'] - curve['sem'],
                        curve['mean'] + curve['sem'], color=c, alpha=0.18)
    ax.set_xlabel(xlabel)
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, lw=0.6)


def _plot_noise_x_queries(ax, df):
    """Interaction panel: response noise x queries per task (combined only)."""
    nq_values = sorted(df['n_queries_per_task'].unique())
    summary = _aggregate(df, ['response_noise', 'n_queries_per_task'], 'mae')

    c = ESTIMATOR_COLORS['rbf_combined']
    for nq in nq_values:
        curve = summary[summary['n_queries_per_task'] == nq].sort_values('response_noise')
        ls = NQ_STYLES.get(nq, '-')
        ax.plot(curve['response_noise'], curve['mean'], marker='o', lw=1.5,
                color=c, linestyle=ls, label=f'nq={nq}')
        ax.fill_between(curve['response_noise'], curve['mean'] - curve['sem'],
                        curve['mean'] + curve['sem'], color=c, alpha=0.10)

    ax.set_xlabel('response noise')
    ax.set_xscale('log')
    ax.legend(fontsize=7, frameon=False, title='combined', title_fontsize=7)
    ax.grid(alpha=0.25, lw=0.6)


# Panel specs: (experiment_name, x_col, xlabel, log_x, log_y, custom_plot_fn)
PANELS = [
    ('n_models',        'n_models',           'number of models',              False, False, None),
    ('n_tasks',         'n_tasks',            'number of tasks',               False, False, None),
    ('task_parity',     'obs_prob',           'task observation probability',   False, False, None),
    ('query_sparsity',  'query_obs_prob',     'query observation probability',  False, False, None),
    ('task_spread',     'task_spread',        'task spread',                    False, True,  None),
    ('noise_x_queries', None,                 None,                            False, False, _plot_noise_x_queries),
]


def plot_figure(results, nrows=2, ncols=3, figsize=(14, 7.5)):
    """Create the main 2x3 figure."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    panel_idx = 0
    for row in range(nrows):
        for col in range(ncols):
            if panel_idx >= len(PANELS):
                axes[row][col].set_visible(False)
                panel_idx += 1
                continue

            exp_name, x_col, xlabel, log_x, log_y, custom_fn = PANELS[panel_idx]
            ax = axes[row][col]

            if exp_name not in results or results[exp_name] is None:
                ax.set_visible(False)
                panel_idx += 1
                continue

            df = results[exp_name]

            if custom_fn is not None:
                custom_fn(ax, df)
            else:
                _plot_standard_panel(ax, df, x_col, xlabel, log_x, log_y)

            if col == 0:
                ax.set_ylabel('MAE')
            ax.set_title(f'({chr(97 + panel_idx)})', fontsize=10, loc='left', fontweight='bold')

            panel_idx += 1

    fig.tight_layout()
    return fig


def save_figure(output_dir, results, filename='fig_synthetic'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_figure(results)
    for ext in ('pdf', 'png'):
        fig.savefig(output_dir / f'{filename}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
