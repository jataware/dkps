#!/usr/bin/env python
"""Main real-data figure 1 -- query efficiency on the heterogeneous 18-task suite.
Each cell is observed with m noisy queries; predict its true (full-budget) score. Sample =
the m-query mean; DKPS/PKPS denoise via the perspective; IRT is the item-response baseline;
ensemble = sample (+) PKPS. Three observation levers; MAE over observed cells, +/- 1 SEM."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

D = Path('results-pkps-rd1')
# colour = method; line style = score-only (dashed: sample, IRT) vs response embedding
# (solid: DKPS, PKPS); the ensemble is solid and thicker.
STYLE = {'sample': dict(color='#777777', ls='--', label='sample'),
         'irt':    dict(color='#988ED5', ls='--', label='IRT'),
         'dkps':   dict(color='#8EBA42', ls='-',  label='DKPS'),
         'pkps':   dict(color='#E24A33', ls='-',  label='PKPS'),
         'ens':    dict(color='#8C564B', ls='-',  label='ensemble', lw=2.6)}
ORDER = ['sample', 'irt', 'dkps', 'pkps', 'ens']
PANELS = [('budget',   'm',       r'queries per cell $m$',           r'$n{=}93,\ p_{\mathrm{task}}{=}1$', True),
          ('n_models', 'n_models', r'number of models $n$',          r'$m{=}2,\ p_{\mathrm{task}}{=}1$',   False),
          ('coverage', 'p_task',  r'task coverage $p_{\mathrm{task}}$', r'$m{=}2,\ n{=}93$',              False)]


def panel(ax, sweep, xcol, xlabel, fixed, logx):
    df = pd.read_csv(D / f'rd1_suite_{sweep}.csv')
    # mean over tasks per (x, seed, method), then mean +/- SEM over seeds
    per_seed = df.groupby([xcol, 'seed', 'method'])['mae'].mean().reset_index()
    g = per_seed.groupby([xcol, 'method'])['mae'].agg(['mean', 'sem'])
    for m in ORDER:
        if m not in df['method'].values:
            continue
        st = STYLE[m]
        sub = g.xs(m, level='method').sort_index()
        ax.errorbar(sub.index.values, sub['mean'], yerr=sub['sem'], marker='o', ms=4,
                    lw=st.get('lw', 1.6), color=st['color'], ls=st['ls'], label=st['label'],
                    capsize=2, elinewidth=0.8)
    if logx:
        ax.set_xscale('log', base=2)
    ax.set_xlabel(xlabel)
    ax.set_title(fixed, fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)


fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, (sweep, xcol, xlabel, fixed, logx), letter in zip(axes, PANELS, 'abc'):
    panel(ax, sweep, xcol, xlabel, fixed, logx)
    ax.set_title(f'({letter})', loc='left', fontweight='bold', fontsize=11)
axes[0].set_ylabel('MAE vs true score (observed cells)')
handles = [Line2D([0], [0], color=STYLE[m]['color'], ls=STYLE[m]['ls'],
                  lw=STYLE[m].get('lw', 1.6), marker='o', ms=4, label=STYLE[m]['label']) for m in ORDER]
fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 1.02), fontsize=9.5)
fig.suptitle('Query efficiency on the heterogeneous suite: PKPS denoises few-query samples, '
             'paired or not, where DKPS and IRT cannot', fontsize=11, y=1.07)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-rd1/fig_suite_query_efficiency.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-rd1/fig_suite_query_efficiency.png')
