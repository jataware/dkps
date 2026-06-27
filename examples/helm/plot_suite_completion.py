#!/usr/bin/env python
"""Main real-data figure 2 -- matrix completion on the heterogeneous 18-task suite. A fraction
of cells is evaluated in full; predict the MISSING cells. Low-rank matrix completion vs the
PKPS embedding vs their ensemble; MAE over MISSING cells, +/- 1 SEM over seeds. Three levers:
cohort n, task coverage p_task, number of tasks T."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

D = Path('results-pkps-unified')
# colour = method; line style = score-only (dashed: matrix completion) vs response embedding
# (solid: PKPS); the ensemble is solid and thicker.
STYLE = {'mc':   dict(color='#348ABC', ls='--', label='matrix completion'),
         'pkps': dict(color='#E24A33', ls='-',  label='PKPS (embedding)'),
         'ens':  dict(color='#8C564B', ls='-',  label='ensemble', lw=2.6)}
ORDER = ['mc', 'pkps', 'ens']
PANELS = [('n_models', 'n_models', r'number of models $n$',
           r'$p_{\mathrm{task}}{=}0.5,\ T{=}18,\ p_{\mathrm{query}}{=}0.8$'),
          ('coverage', 'p_task',   r'task coverage $p_{\mathrm{task}}$',
           r'$n{=}93,\ T{=}18,\ p_{\mathrm{query}}{=}0.8$'),
          ('n_tasks',  'n_tasks',  r'number of tasks $T$',
           r'$n{=}93,\ p_{\mathrm{task}}{=}0.5,\ p_{\mathrm{query}}{=}0.8$')]


def panel(ax, sweep, xcol, xlabel, fixed):
    df = pd.read_csv(D / f'completion_suite_{sweep}.csv')
    g = df.groupby(xcol)[ORDER].agg(['mean', 'sem'])
    x = g.index.values
    for m in ORDER:
        st = STYLE[m]
        ax.errorbar(x, g[(m, 'mean')], yerr=g[(m, 'sem')], marker='o', ms=4,
                    lw=st.get('lw', 1.6), color=st['color'], ls=st['ls'], label=st['label'],
                    capsize=2, elinewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(fixed, fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)


fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (sweep, xcol, xlabel, fixed), letter in zip(axes, PANELS, 'abc'):
    panel(ax, sweep, xcol, xlabel, fixed)
    ax.set_title(f'({letter})', loc='left', fontweight='bold', fontsize=11)
axes[0].set_ylabel('MAE vs true score (missing cells)')
handles = [Line2D([0], [0], color=STYLE[m]['color'], ls=STYLE[m]['ls'],
                  lw=STYLE[m].get('lw', 1.6), marker='o', ms=4, label=STYLE[m]['label']) for m in ORDER]
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.02), fontsize=9.5)
fig.suptitle('Matrix completion on the heterogeneous suite: the embedding rescues the '
             'scarce-data regimes; the ensemble tracks the best everywhere', fontsize=11, y=1.07)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-unified/fig_suite_completion.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-unified/fig_suite_completion.png')
