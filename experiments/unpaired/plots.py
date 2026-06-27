import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'matplotlib'))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd


# the full PKPS (combined, uses all data) is red like PKPS elsewhere; the paired/unpaired
# ablations get their own colours (teal/orange) so they never collide with a method's colour
# (blue = matrix completion, green = DKPS, purple = IRT in the other figures).
ESTIMATOR_COLORS = {'sample': '#777777', 'dkps': '#8EBA42', 'pkps': '#E24A33'}
ESTIMATOR_LABELS = {'sample': 'sample score', 'dkps': 'DKPS', 'pkps': 'PKPS'}
ESTIMATOR_ORDER = ['sample', 'dkps', 'pkps']


def _aggregate(df, group_cols, value_col):
    grouped = (
        df.groupby(group_cols, dropna=False)[value_col]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    grouped['sem'] = grouped['std'].fillna(0.0) / np.sqrt(grouped['count'].clip(lower=1))
    return grouped


def _errbar(ax, x, mean, sem, color, label, ls='-'):
    ax.errorbar(x, mean, yerr=sem, marker='o', ms=4, lw=1.5, color=color, ls=ls,
                label=label, capsize=2, elinewidth=0.8)


def _plot_standard_panel(ax, df, x_col, xlabel):
    """DKPS vs PKPS, x vs MAE (no per-panel legend; the figure has one shared legend)."""
    summary = _aggregate(df, [x_col, 'estimator'], 'mae')
    for est in ESTIMATOR_ORDER:
        if est not in summary['estimator'].values:
            continue
        curve = summary[summary['estimator'] == est].sort_values(x_col)
        _errbar(ax, curve[x_col], curve['mean'], curve['sem'],
                ESTIMATOR_COLORS[est], ESTIMATOR_LABELS[est])
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25, lw=0.6)


def _plot_rho(ax, df, xlabel):
    """Cross-model overlap rho: DKPS (delta) collapses as rho->0, PKPS (rbf) holds."""
    summary = _aggregate(df, ['rho', 'estimator'], 'mae')
    for est in ESTIMATOR_ORDER:
        curve = summary[summary['estimator'] == est].sort_values('rho')
        _errbar(ax, curve['rho'], curve['mean'], curve['sem'],
                ESTIMATOR_COLORS[est], ESTIMATOR_LABELS[est])
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25, lw=0.6)


def _plot_query_efficiency(ax, df, xlabel):
    """MAE vs per-cell query budget for DKPS and PKPS at paired (rho=1) / unpaired (rho=0).
    Method by colour (shared legend); overlap by line style, noted inline."""
    rhos = sorted(df['rho'].unique())
    ls_map = {min(rhos): '--', max(rhos): '-'}      # dashed = unpaired, solid = paired
    # sample score is rho-independent (it ignores queries): a single reference line
    s = _aggregate(df[df['estimator'] == 'sample'], ['budget'], 'mae').sort_values('budget')
    _errbar(ax, s['budget'], s['mean'], s['sem'], ESTIMATOR_COLORS['sample'], None, '-')
    summary = _aggregate(df, ['budget', 'rho', 'estimator'], 'mae')
    for est in ('dkps', 'pkps'):
        for rho in rhos:
            curve = summary[(summary['estimator'] == est) & (summary['rho'] == rho)].sort_values('budget')
            _errbar(ax, curve['budget'], curve['mean'], curve['sem'],
                    ESTIMATOR_COLORS[est], None, ls_map.get(rho, '-'))
    ax.set_xlabel(xlabel)
    ax.set_xscale('log', base=2)
    ax.text(0.97, 0.97, 'solid: paired ($\\rho{=}1$)\ndashed: unpaired ($\\rho{=}0$)',
            transform=ax.transAxes, ha='right', va='top', fontsize=7, color='#444')
    ax.grid(alpha=0.25, lw=0.6)


# Panel specs: (experiment, x_col, xlabel, custom_fn, fixed-parameter title)
PANELS = [
    ('n_models', 'n_models', r'number of models $n$', None,
     r'$T{=}20,\ M_{ij}{=}10,\ p_{\mathrm{task}}{=}0.3,\ p_{\mathrm{query}}{=}0.8$'),
    ('n_tasks', 'n_tasks', r'number of tasks $T$', None,
     r'$n{=}100,\ M_{ij}{=}10,\ p_{\mathrm{task}}{=}0.3,\ p_{\mathrm{query}}{=}0.8$'),
    ('task_parity', 'obs_prob', r'task obs. prob. $p_{\mathrm{task}}$', None,
     r'$n{=}100,\ T{=}20,\ M_{ij}{=}10,\ p_{\mathrm{query}}{=}0.8$'),
    ('query_sparsity', 'query_obs_prob', r'query obs. prob. $p_{\mathrm{query}}$', None,
     r'$n{=}100,\ T{=}20,\ M_{ij}{=}10,\ p_{\mathrm{task}}{=}0.3$'),
    ('rho', None, r"query overlap $\rho=m_{ii'}/M_{ij}$", _plot_rho,
     r'$n{=}100,\ T{=}20,\ M_{ij}{=}10,\ p_{\mathrm{task}}{=}0.3$'),
    ('query_efficiency', None, r'queries per cell $M_{ij}$', _plot_query_efficiency,
     r'$n{=}60,\ T{=}12,\ p_{\mathrm{task}}{=}1$'),
]


def plot_figure(results, nrows=2, ncols=3, figsize=(14, 7.6)):
    """Create the main 2x3 figure with one shared legend and per-panel fixed-parameter titles."""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharey=True)

    for idx, (exp_name, x_col, xlabel, custom_fn, fixed) in enumerate(PANELS):
        ax = axes[idx // ncols][idx % ncols]
        if exp_name not in results or results[exp_name] is None:
            ax.set_visible(False)
            continue
        df = results[exp_name]
        if custom_fn is not None:
            custom_fn(ax, df, xlabel)
        else:
            _plot_standard_panel(ax, df, x_col, xlabel)
        if idx % ncols == 0:
            ax.set_ylabel('score MAE')
        ax.set_title(fixed, fontsize=8.5)
        ax.set_title(f'({chr(97 + idx)})', loc='left', fontweight='bold', fontsize=11)

    handles = [Line2D([0], [0], color=ESTIMATOR_COLORS[e], marker='o', lw=1.5,
                      label=ESTIMATOR_LABELS[e]) for e in ESTIMATOR_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.005), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def save_figure(output_dir, results, filename='fig_synthetic'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_figure(results)
    for ext in ('pdf', 'png'):
        fig.savefig(output_dir / f'{filename}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
