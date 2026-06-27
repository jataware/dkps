#!/usr/bin/env python
"""HEADLINE real-data figure -- the pairing cliff on the heterogeneous 18-task suite. At a
fixed per-cell budget, slide the queries from fully paired (every model answers the same
ones) to fully unpaired (disjoint per model). DKPS (identity query kernel) and IRT need the
shared queries and collapse as they vanish; PKPS bridges similar queries and holds. MAE vs
the full-eval score, +/- 1 SEM over seeds."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd

BUDGET = 4
STYLE = {'sample': dict(color='#777777', ls='--', label='sample'),
         'irt':    dict(color='#988ED5', ls='--', label='IRT'),
         'dkps':   dict(color='#8EBA42', ls='-',  label='DKPS'),
         'pkps':   dict(color='#E24A33', ls='-',  label='PKPS'),
         'ens':    dict(color='#8C564B', ls='-',  label='ensemble', lw=3.6)}
ORDER = ['sample', 'irt', 'dkps', 'pkps', 'ens']

df = pd.read_csv('results-pkps-rd1/rd1_suite_pairing.csv')
df['rho'] = df['n_paired'] / BUDGET
per_seed = df.groupby(['rho', 'seed', 'method'])['mae'].mean().reset_index()
g = per_seed.groupby(['rho', 'method'])['mae'].agg(['mean', 'sem'])

fig, ax = plt.subplots(figsize=(6.2, 4.6))
for m in ORDER:
    st = STYLE[m]
    sub = g.xs(m, level='method').sort_index()
    ax.errorbar(sub.index.values, sub['mean'], yerr=sub['sem'], marker='o', ms=5,
                lw=st.get('lw', 2.6), color=st['color'], ls=st['ls'], label=st['label'],
                capsize=3, elinewidth=0.9)
ax.set_xlabel(r"fraction of paired queries  $\rho = m_{ii'}/M_{ij}$")
ax.set_ylabel('MAE vs full-eval score')
ax.set_title(f'Unpaired evaluation (suite, budget $M_{{ij}}{{=}}{BUDGET}$ queries/cell):\n'
             'DKPS and IRT collapse as queries unpair; PKPS holds', fontsize=11)
ax.legend(frameon=False, fontsize=9.5, ncol=2)
ax.grid(alpha=0.25, lw=0.6)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-rd1/fig_suite_pairing.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-rd1/fig_suite_pairing.png')
