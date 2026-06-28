#!/usr/bin/env python
"""Supplementary: the pairing cliff is robust to the per-cell budget. Same experiment as the
headline (slide rho from paired to unpaired) repeated at budgets M_ij in {2,4,8}; DKPS and IRT
collapse as rho->0 at every budget, PKPS holds. MAE vs full-eval score, +/-1 SEM over seeds."""
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
import pandas as pd
from matplotlib.lines import Line2D

STYLE = {'sample': dict(color='#777777', ls='--', label='sample'),
         'irt':    dict(color='#988ED5', ls='--', label='IRT'),
         'dkps':   dict(color='#8EBA42', ls='-',  label='DKPS'),
         'pkps':   dict(color='#E24A33', ls='-',  label='PKPS'),
         'ens':    dict(color='#8C564B', ls='-',  label='ensemble', lw=3.6)}
ORDER = ['sample', 'irt', 'dkps', 'pkps', 'ens']
BUDGETS = [2, 4, 8]

df = pd.read_csv('results-pkps-rd1/rd1_suite_pairing.csv')
fig, axes = plt.subplots(1, len(BUDGETS), figsize=(13, 4.2), sharey=True)
for ax, b in zip(axes, BUDGETS):
    sub = df[df['m'] == b].copy(); sub['rho'] = sub['n_paired'] / b
    per_seed = sub.groupby(['rho', 'seed', 'method'])['mae'].mean().reset_index()
    g = per_seed.groupby(['rho', 'method'])['mae'].agg(['mean', 'sem'])
    for m in ORDER:
        if m not in sub['method'].values:
            continue
        st = STYLE[m]; s = g.xs(m, level='method').sort_index()
        ax.errorbar(s.index.values, s['mean'], yerr=s['sem'], marker='o', ms=4,
                    lw=st.get('lw', 2.6), color=st['color'], ls=st['ls'], capsize=3, elinewidth=0.9)
    ax.set_xlabel(r"fraction paired  $\rho = m_{ii'}/M_{ij}$", fontsize=12.5)
    ax.set_title(f'budget $M_{{ij}}{{=}}{b}$ queries/cell', fontsize=11.5)
    ax.tick_params(labelsize=10.5)
axes[0].set_ylabel('MAE vs. full-eval score', fontsize=12.5)
handles = [Line2D([0], [0], color=STYLE[m]['color'], ls=STYLE[m]['ls'], lw=STYLE[m].get('lw', 2.6),
                  marker='o', ms=4, label=STYLE[m]['label']) for m in ORDER]
fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.95), fontsize=11)
fig.suptitle('Robustness to the per-cell budget', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.87])
for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-rd1/fig_suite_pairing_robust.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-rd1/fig_suite_pairing_robust.png')
