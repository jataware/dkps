#!/usr/bin/env python
"""SECONDARY query-efficiency conditional figure (4 ridgeline panels). Per-cell denoising win
Delta = sample MAE - ensemble MAE (>0 = the embedding-augmented ensemble beats the raw sample),
stacked by dataset, by budget, by cohort, and by task coverage. Mean marker per row. Reads the
per-cell dumps from _dump_qe_cells.py (budget/dataset) and _dump_qe_levers.py (cohort/coverage)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({                              # clean modern style: white, subtle grid, despined
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': '#b8b8b8',
    'axes.linewidth': 0.9, 'axes.grid': True, 'axes.axisbelow': True, 'grid.color': '#ececec',
    'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'xtick.color': '#555', 'ytick.color': '#555',
    'axes.labelcolor': '#222', 'axes.titlecolor': '#111', 'text.color': '#222', 'font.size': 11})
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

XL = (-0.12, 0.34)
XLAB = 'sample $-$ ensemble (per cell)'
DS = [('MATH', 'math'), ('MedQA', 'med_qa'), ('LegalB', 'legalbench'), ('WMT', 'wmt_14')]


def ridge(ax, groups, title):
    xs = np.linspace(*XL, 300)
    for i, (lab, v) in enumerate(groups[::-1]):
        v = np.asarray(v, float); v = v[np.isfinite(v)]; off = i * 1.0
        if len(v) > 10:
            dd = gaussian_kde(v)(xs); dd = dd / dd.max() * 0.82
            ax.fill_between(xs, off, off + dd, where=xs >= 0, color='#E24A33', alpha=0.6, lw=0)
            ax.fill_between(xs, off, off + dd, where=xs < 0, color='#bbb', alpha=0.55, lw=0)
            ax.plot(xs, off + dd, color='#444', lw=0.7)
            ax.plot([v.mean()], [off + 0.02], 'v', color='#111', ms=5)
        ax.text(XL[0] - 0.02 * (XL[1] - XL[0]), off + 0.12, lab, fontsize=9, ha='right', va='bottom')
    ax.axvline(0, color='#888', lw=0.7); ax.set_yticks([]); ax.set_xlim(*XL)
    ax.spines['left'].set_visible(False); ax.set_xlabel(XLAB, fontsize=10)
    ax.set_title(title, fontsize=10.5)


def piv(path, idx):
    d = pd.read_csv(path)
    p = d.pivot_table(index=idx, columns='method', values='abs_err').reset_index()
    p['d'] = p['sample'] - p['ens']
    return p


qb = piv('results-pkps-rd1/qe_cells.csv', ['m', 'seed', 'model', 'task', 'dataset'])
ql = piv('results-pkps-rd1/qe_cells_levers.csv', ['n_models', 'p_task', 'seed', 'model', 'task', 'dataset'])

fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))
ridge(ax[0], [(n, qb[(qb.m == 2) & (qb.dataset == d)]['d'].values) for n, d in DS],
      '(a) by dataset\n$m{=}2,\\ n{=}93,\\ p_\\mathrm{task}{=}1$')
ridge(ax[1], [(f'$m{{=}}{m}$', qb[qb.m == m]['d'].values) for m in [1, 2, 4, 8]],
      '(b) by budget\n$n{=}93,\\ p_\\mathrm{task}{=}1$')
ridge(ax[2], [(f'$n{{=}}{n}$', ql[(ql.p_task == 1.0) & (ql.n_models == n)]['d'].values) for n in [10, 40, 93]],
      '(c) by cohort\n$m{=}2,\\ p_\\mathrm{task}{=}1$')
ridge(ax[3], [(f'$p_\\mathrm{{task}}{{=}}{p}$', ql[(ql.n_models == 93) & (ql.p_task == p)]['d'].values) for p in [0.2, 0.5, 0.9]],
      '(d) by task coverage\n$m{=}2,\\ n{=}93$')
fig.suptitle('Query efficiency: where the denoising win comes from ($\\Delta>0\\to$ win)',
             fontsize=13.5, fontweight='bold', y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.92])
for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-rd1/fig_qe_conditional.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-rd1/fig_qe_conditional.png')
