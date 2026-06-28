#!/usr/bin/env python
"""Conditional (per-dataset / per-model) analysis of the unpaired headline. Reads the per-cell
dump from `helm_rd1_suite.py --sweep breakdown` (budget 4, pairing 0=unpaired and 4=paired)
and localizes WHERE the cliff bites. (The per-model PKPS-vs-DKPS scatter lives in the headline
figure.)

  (a) per-dataset MAE, unpaired (rho=0): which domains DKPS/IRT lose and PKPS holds.
  (b) per-model cliff depth (unpaired - paired MAE): DKPS jumps, PKPS barely moves.
"""
import sys
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

CSV = sys.argv[1] if len(sys.argv) > 1 else 'results-pkps-rd1/rd1_suite_breakdown.csv'
BUDGET = 4
COL = {'sample': '#777777', 'irt': '#988ED5', 'dkps': '#8EBA42', 'pkps': '#E24A33', 'ens': '#8C564B'}
LBL = {'sample': 'sample', 'irt': 'IRT', 'dkps': 'DKPS', 'pkps': 'PKPS', 'ens': 'ensemble'}

df = pd.read_csv(CSV)
un = df[df['n_paired'] == 0]
pa = df[df['n_paired'] == BUDGET]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

# (a) per-dataset MAE, unpaired ------------------------------------------------
order = ['sample', 'irt', 'dkps', 'pkps', 'ens']
datasets = sorted(un['dataset'].unique())
ps = un.groupby(['dataset', 'method', 'seed'])['abs_err'].mean().reset_index()
g = ps.groupby(['dataset', 'method'])['abs_err'].agg(['mean', 'sem'])
ax = axes[0]
w = 0.16
for k, meth in enumerate(order):
    xs, ms, es = [], [], []
    for d in datasets:
        if (d, meth) in g.index:
            xs.append(datasets.index(d) + (k - 2) * w)
            ms.append(g.loc[(d, meth), 'mean']); es.append(g.loc[(d, meth), 'sem'])
    ax.bar(xs, ms, width=w, yerr=es, color=COL[meth], label=LBL[meth], capsize=2,
           error_kw=dict(elinewidth=0.7))
ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, rotation=20, ha='right', fontsize=8.5)
ax.set_ylabel('MAE vs full-eval score')
ax.set_title('(a) per dataset, unpaired ($\\rho{=}0$)', fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5, ncol=2)

# (b) per-model cliff depth: unpaired - paired MAE -----------------------------
def gap(d):
    u = un.groupby(['model', 'method'])['abs_err'].mean().unstack()[d]
    p = pa.groupby(['model', 'method'])['abs_err'].mean().unstack()[d]
    return (u - p).dropna()
ax = axes[1]
jit = np.random.default_rng(0)
for x, meth in [(0, 'dkps'), (1, 'pkps')]:
    vals = gap(meth)
    ax.scatter(np.full(len(vals), x) + jit.normal(0, 0.05, len(vals)), vals,
               s=24, c=COL[meth], edgecolors='white', lw=0.4, alpha=0.85, zorder=3)
    ax.hlines(vals.mean(), x - 0.26, x + 0.26, color=COL[meth], lw=3, zorder=4)
    ax.text(x + 0.30, vals.mean(), f'mean {vals.mean():+.2f}', ha='left', va='center',
            fontsize=9, color=COL[meth], fontweight='bold', zorder=5)
ax.axhline(0, color='#bbb', lw=0.9)
ax.set_xlim(-0.5, 1.5); ax.set_xticks([0, 1]); ax.set_xticklabels(['DKPS', 'PKPS'])
ax.set_ylabel('MAE increase when unpaired\n(unpaired $-$ paired, per model)')
ax.set_title('(b) per-model cliff depth (lower = robust)', fontsize=10.5)

fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-rd1/fig_suite_breakdown.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-rd1/fig_suite_breakdown.png')
