#!/usr/bin/env python
"""Conditional (per-task / per-model) analysis of the unpaired headline. Reads the per-cell
dump from `helm_rd1_suite.py --sweep breakdown` (budget 4, pairing levels 0=unpaired and
4=paired) and shows WHERE the pairing cliff bites and where PKPS recovers it.

  (a) per-dataset MAE, unpaired (rho=0): which domains DKPS/IRT lose and PKPS holds.
  (b) per-model PKPS vs DKPS MAE, unpaired: every model below the diagonal -> PKPS wins.
  (c) per-model PKPS recovery: unpaired-minus-paired MAE for DKPS vs PKPS (smaller = robust).
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

CSV = sys.argv[1] if len(sys.argv) > 1 else 'results-pkps-rd1/rd1_suite_breakdown.csv'
BUDGET = 4
COL = {'sample': '#777777', 'irt': '#988ED5', 'dkps': '#8EBA42', 'pkps': '#E24A33', 'ens': '#8C564B'}
LBL = {'sample': 'sample', 'irt': 'IRT', 'dkps': 'DKPS', 'pkps': 'PKPS', 'ens': 'ensemble'}

df = pd.read_csv(CSV)
un = df[df['n_paired'] == 0]                      # unpaired (rho=0)
pa = df[df['n_paired'] == BUDGET]                 # paired   (rho=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

# (a) per-dataset MAE, unpaired -------------------------------------------------
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
ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, rotation=20, ha='right', fontsize=8)
ax.set_ylabel('MAE vs full-eval score'); ax.set_title('(a) per dataset, unpaired ($\\rho{=}0$)', fontsize=10)
ax.legend(frameon=False, fontsize=8, ncol=2)

# (b) per-model PKPS vs DKPS, unpaired -----------------------------------------
pm = un.groupby(['model', 'method'])['abs_err'].mean().unstack()
ax = axes[1]
lim = float(np.nanmax([pm['dkps'].max(), pm['pkps'].max()])) * 1.08
ax.plot([0, lim], [0, lim], ls='--', color='#94a3b8', lw=1.2)
ax.scatter(pm['pkps'], pm['dkps'], s=26, c=COL['pkps'], edgecolors='white', lw=0.5, zorder=3)
ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal')
ax.set_xlabel('PKPS MAE'); ax.set_ylabel('DKPS MAE')
frac = float((pm['dkps'] > pm['pkps']).mean())
ax.set_title(f'(b) per model, unpaired: PKPS<DKPS for {frac:.0%}', fontsize=10)

# (c) per-model robustness: unpaired - paired MAE (smaller = holds up) ----------
def gap(d):
    u = un.groupby(['model', 'method'])['abs_err'].mean().unstack()[d]
    p = pa.groupby(['model', 'method'])['abs_err'].mean().unstack()[d]
    return (u - p).dropna()
ax = axes[2]
for meth in ('dkps', 'pkps'):
    vals = gap(meth)
    ax.scatter(np.full(len(vals), 0 if meth == 'dkps' else 1) + np.random.default_rng(0).normal(0, 0.05, len(vals)),
               vals, s=22, c=COL[meth], edgecolors='white', lw=0.4, alpha=0.8)
    ax.hlines(vals.mean(), (0 if meth == 'dkps' else 1) - 0.25, (0 if meth == 'dkps' else 1) + 0.25,
              color=COL[meth], lw=2.5)
ax.axhline(0, color='#bbb', lw=0.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(['DKPS', 'PKPS'])
ax.set_ylabel('MAE increase when unpaired\n(unpaired $-$ paired)')
ax.set_title('(c) per-model cliff depth (lower = robust)', fontsize=10)

fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-rd1/fig_suite_breakdown.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-rd1/fig_suite_breakdown.png')
