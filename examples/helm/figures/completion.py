#!/usr/bin/env python
"""Main real-data figure 2 -- matrix completion on the heterogeneous 18-task suite. Observe a
fraction of cells (each with a fraction p_query of its queries), predict the MISSING cells.
Low-rank matrix completion vs the PKPS embedding vs the ensemble; MAE over MISSING cells,
+/- 1 SEM. Colour = method; line style = cohort size (solid n=93, dashed n=10). Four levers:
task coverage, query depth p_query, number of tasks T, and cohort n."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
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
from matplotlib.lines import Line2D

D = Path('results-pkps-unified')
COL = {'mc': '#348ABC', 'pkps': '#E24A33', 'ens': '#8C564B'}
LBL = {'mc': 'matrix completion', 'pkps': 'PKPS', 'ens': 'ensemble'}
LW = {'mc': 2.6, 'pkps': 2.6, 'ens': 3.6}
NLS = {93: '-', 10: '--'}                       # cohort -> line style
# (sweep, xcol, xlabel, fixed-title, line-family?)
PANELS = [('coverage', 'p_task', r'task coverage $p_{\mathrm{task}}$', r'$p_{\mathrm{query}}{=}0.5,\ T{=}18$', True),
          ('p_query', 'p_query', r'query depth $p_{\mathrm{query}}$',  r'$p_{\mathrm{task}}{=}0.5,\ T{=}18$', True),
          ('n_tasks', 'n_tasks', r'number of tasks $T$',              r'$p_{\mathrm{task}}{=}0.5,\ p_{\mathrm{query}}{=}0.5$', True),
          ('n_models', 'n_models', r'number of models $n$',           r'$p_{\mathrm{task}}{=}0.5,\ p_{\mathrm{query}}{=}0.5,\ T{=}18$', False)]


def panel(ax, sweep, xcol, xlabel, fixed, family):
    df = pd.read_csv(D / f'completion_suite_{sweep}.csv')
    for meth in ('mc', 'pkps', 'ens'):
        if family:
            for n, ls in NLS.items():
                sub = df[df['n_models'] == n]
                g = sub.groupby(xcol)[meth].agg(['mean', 'sem'])
                ax.errorbar(g.index.values, g['mean'], yerr=g['sem'], marker='o', ms=3,
                            lw=LW[meth], color=COL[meth], ls=ls, capsize=2, elinewidth=0.7)
        else:
            g = df.groupby(xcol)[meth].agg(['mean', 'sem'])
            ax.errorbar(g.index.values, g['mean'], yerr=g['sem'], marker='o', ms=3,
                        lw=LW[meth], color=COL[meth], ls='-', capsize=2, elinewidth=0.7)
    ax.set_xlabel(xlabel, fontsize=15)


fig, axes = plt.subplots(1, 4, figsize=(11.5, 4.2), sharey=True)
for ax, spec, letter in zip(axes, PANELS, 'abcd'):
    panel(ax, *spec)
    # single left-aligned title (bold letter + fixed params) -- avoids the
    # centered-subtitle / corner-letter collision in these narrow 4-up panels
    ax.set_title(f'$\\bf{{({letter})}}$  {spec[3]}', loc='left', fontsize=12.8)
    ax.tick_params(labelsize=13)
axes[0].set_ylabel('MAE on missing cells', fontsize=15)
meth_h = [Line2D([0], [0], color=COL[m], lw=LW[m], marker='o', ms=3, label=LBL[m]) for m in ('mc', 'pkps', 'ens')]
n_h = [Line2D([0], [0], color='#444', ls='-', label='$n{=}93$'),
       Line2D([0], [0], color='#444', ls='--', label='$n{=}10$')]
fig.legend(handles=meth_h + n_h, loc='upper center', ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 0.97), fontsize=13.5)
fig.suptitle('Matrix completion', fontsize=18, fontweight='bold', y=1.03)
fig.tight_layout(rect=[0, 0, 1, 0.88])
for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-unified/fig_suite_completion.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-unified/fig_suite_completion.png')
