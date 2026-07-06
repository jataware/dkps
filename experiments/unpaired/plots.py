import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'matplotlib'))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({                              # clean modern style: white, subtle grid, despined
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#b8b8b8', 'axes.linewidth': 0.9,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': '#e9e9e9', 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': '#555', 'ytick.color': '#555',
    'axes.labelcolor': '#222', 'axes.titlecolor': '#111', 'text.color': '#222', 'font.size': 11,
})
import numpy as np
import pandas as pd


# the full PKPS (combined, uses all data) is red like PKPS elsewhere; the paired/unpaired
# ablations get their own colours (teal/orange) so they never collide with a method's colour
# (blue = matrix completion, green = DKPS, purple = IRT in the other figures).
ESTIMATOR_COLORS = {'sample': '#777777', 'dkps': '#8EBA42', 'pkps': '#E24A33'}
ESTIMATOR_LABELS = {'sample': 'Sample score', 'dkps': 'DKPS', 'pkps': 'PKPS'}
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
    ax.errorbar(x, mean, yerr=sem, marker='o', ms=4, lw=2.6, color=color, ls=ls,
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
            transform=ax.transAxes, ha='right', va='top', fontsize=9.8, color='#444')
    ax.grid(alpha=0.25, lw=0.6)


# Panel specs: (experiment, x_col, xlabel, custom_fn, fixed-parameter title)
PANELS = [
    ('n_models', 'n_models', r'number of models $n$', None,
     r'$T{=}20,\ M_{ij}{=}10,\ p_{\mathrm{task}}{=}0.3,\ p_{\mathrm{query}}{=}0.8$'),
    ('n_tasks', 'n_tasks', r'number of tasks $T$', None,
     r'$n{=}100,\ M_{ij}{=}10,\ p_{\mathrm{task}}{=}0.3,\ p_{\mathrm{query}}{=}0.8$'),
    ('task_parity', 'obs_prob', r'task coverage $p_{\mathrm{task}}$', None,
     r'$n{=}100,\ T{=}20,\ M_{ij}{=}10,\ p_{\mathrm{query}}{=}0.8$'),
    ('query_sparsity', 'query_obs_prob', r'query depth $p_{\mathrm{query}}$', None,
     r'$n{=}100,\ T{=}20,\ M_{ij}{=}10,\ p_{\mathrm{task}}{=}0.3$'),
    ('rho', None, r"query overlap $\rho=m_{ii'}/M_{ij}$", _plot_rho,
     r'$n{=}100,\ T{=}20,\ M_{ij}{=}10,\ p_{\mathrm{task}}{=}0.3$'),
    ('query_efficiency', None, r'queries per cell $M_{ij}$', _plot_query_efficiency,
     r'$n{=}60,\ T{=}12,\ p_{\mathrm{task}}{=}1$'),
]


def plot_figure(results, nrows=2, ncols=3, figsize=(11.5, 4.9)):
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
            ax.set_ylabel('score MAE', fontsize=17.5)
        ax.set_xlabel(ax.get_xlabel(), fontsize=17.5)
        ax.tick_params(labelsize=14.0)
        # single left-aligned title (bold letter + fixed params): the 4-parameter
        # subtitles are too wide to sit under a corner letter without colliding
        ax.set_title(f'$\\bf{{({chr(97 + idx)})}}$  {fixed}', loc='left', fontsize=11.0)

    axes[0][0].set_ylim(top=2.5)  # shared y; cap so panel (f)'s high-noise tail doesn't stretch all panels
    handles = [Line2D([0], [0], color=ESTIMATOR_COLORS[e], marker='o', lw=2.6,
                      label=ESTIMATOR_LABELS[e]) for e in ESTIMATOR_ORDER]
    fig.tight_layout()
    # legend flush above the top row's panel titles, suptitle flush above the legend,
    # both centred on the panel span (canvas centre is skewed left by the shared ylabel)
    fig.canvas.draw()
    r, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    flat = [ax for row in axes for ax in row if ax.get_visible()]
    xc = 0.5 * (min(ax.get_position().x0 for ax in flat) +
                max(ax.get_position().x1 for ax in flat))
    ytop = max(ax.get_tightbbox(r).transformed(inv).y1 for ax in flat)
    lg = fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False,
                    bbox_to_anchor=(xc, ytop + 0.010), fontsize=16.8)
    fig.canvas.draw()
    fig.suptitle('Synthetic study', fontsize=21.0, fontweight='bold',
                 x=xc, y=lg.get_window_extent(r).transformed(inv).y1 + 0.012, va='bottom')
    return fig


def save_figure(output_dir, results, filename='fig_synthetic'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_figure(results)
    for ext in ('pdf', 'png'):
        fig.savefig(output_dir / f'{filename}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
