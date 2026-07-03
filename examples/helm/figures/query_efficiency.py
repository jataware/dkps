#!/usr/bin/env python
"""Main real-data figure 1 -- query efficiency on the heterogeneous 18-task suite.
Each cell is observed with m noisy queries; predict its true (full-budget) score. Sample =
the m-query mean; DKPS/PKPS denoise via the perspective; IRT is the item-response baseline;
ensemble = sample (+) PKPS. Three observation levers; MAE over observed cells, +/- 1 SEM."""
import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument('--suite', choices=['helm', 'eee'], default='helm')
args = ap.parse_args()
# suite-specific: results dir, full cohort size n, output stem
CFG = {'helm': dict(dir='results-pkps-rd1', n=93, stem='fig_suite_query_efficiency'),
       'eee':  dict(dir='results-eee-rd1',  n=45, stem='fig_eee_query_efficiency')}[args.suite]
D = Path(CFG['dir'])
# colour = method; line style = score-only (dashed: sample, IRT) vs response embedding
# (solid: DKPS, PKPS); the ensemble is solid and thicker.
STYLE = {'sample': dict(color='#777777', ls='--', label='sample'),
         'irt':    dict(color='#988ED5', ls='--', label='IRT'),
         'dkps':   dict(color='#8EBA42', ls='-',  label='DKPS'),
         'pkps':   dict(color='#E24A33', ls='-',  label='PKPS'),
         'ens':    dict(color='#8C564B', ls='-',  label='ensemble', lw=3.6)}
ORDER = ['sample', 'irt', 'dkps', 'pkps', 'ens']
N = CFG['n']
PANELS = [('budget',   'm',       r'queries per cell $m$',           rf'$n{{=}}{N},\ p_{{\mathrm{{task}}}}{{=}}1$', True),
          ('n_models', 'n_models', r'number of models $n$',          r'$m{=}2,\ p_{\mathrm{task}}{=}1$',   False),
          ('coverage', 'p_task',  r'task coverage $p_{\mathrm{task}}$', rf'$m{{=}}2,\ n{{=}}{N}$',              False)]


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
                    lw=st.get('lw', 2.6), color=st['color'], ls=st['ls'], label=st['label'],
                    capsize=2, elinewidth=0.8)
    if logx:
        ax.set_xscale('log', base=2)
    ax.set_xlabel(xlabel, fontsize=15.6)
    ax.set_title(fixed, fontsize=13.8)


fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
for ax, (sweep, xcol, xlabel, fixed, logx), letter in zip(axes, PANELS, 'abc'):
    panel(ax, sweep, xcol, xlabel, fixed, logx)
    ax.set_title(f'({letter})', loc='left', fontweight='bold', fontsize=16.2)
    ax.tick_params(labelsize=13.1)
axes[0].set_ylabel('MAE vs. true score', fontsize=15.6)
handles = [Line2D([0], [0], color=STYLE[m]['color'], ls=STYLE[m]['ls'],
                  lw=STYLE[m].get('lw', 2.6), marker='o', ms=4, label=STYLE[m]['label']) for m in ORDER]
fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 0.95), fontsize=13.8)
fig.suptitle('Query efficiency', fontsize=17.5, fontweight='bold', y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.87])
for ext in ('png', 'pdf'):
    fig.savefig(D / f'{CFG["stem"]}.{ext}', dpi=200, bbox_inches='tight')
print(f'wrote {D}/{CFG["stem"]}.png')
