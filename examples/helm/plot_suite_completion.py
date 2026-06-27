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
plt.style.use('ggplot')
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

D = Path('results-pkps-unified')
COL = {'mc': '#348ABC', 'pkps': '#E24A33', 'ens': '#8C564B'}
LBL = {'mc': 'matrix completion', 'pkps': 'PKPS', 'ens': 'ensemble'}
LW = {'mc': 1.6, 'pkps': 1.6, 'ens': 2.4}
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
    ax.set_xlabel(xlabel)
    ax.set_title(fixed, fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.6)


fig, axes = plt.subplots(1, 4, figsize=(16, 4.0), sharey=True)
for ax, spec, letter in zip(axes, PANELS, 'abcd'):
    panel(ax, *spec)
    ax.set_title(f'({letter})', loc='left', fontweight='bold', fontsize=11)
axes[0].set_ylabel('MAE on missing cells')
meth_h = [Line2D([0], [0], color=COL[m], lw=LW[m], marker='o', ms=3, label=LBL[m]) for m in ('mc', 'pkps', 'ens')]
n_h = [Line2D([0], [0], color='#444', ls='-', label='$n{=}93$'),
       Line2D([0], [0], color='#444', ls='--', label='$n{=}10$')]
fig.legend(handles=meth_h + n_h, loc='upper center', ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 1.03), fontsize=9.5)
fig.suptitle('Matrix completion: the embedding rescues the small-cohort regime ($n{=}10$, dashed); '
             'at full cohort ($n{=}93$, solid) it ties low-rank completion', fontsize=11, y=1.10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-unified/fig_suite_completion.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-unified/fig_suite_completion.png')
