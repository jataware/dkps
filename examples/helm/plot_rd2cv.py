#!/usr/bin/env python
"""RD2 figure (unified CV): PKPS-combined vs matrix completion vs ensemble, swept
over task-observation probability and cohort size."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {
    'matcomplete': dict(color='#7c3aed', ls='--', label='matrix completion'),
    'combined':    dict(color='#1d4ed8', ls='-', label='PKPS'),
    'ensemble':    dict(color='#15803d', ls='-', label='ensemble', lw=2.4),
}
ORDER = ['matcomplete', 'combined', 'ensemble']


def panel(ax, df, xcol, xlabel):
    g = df.groupby([xcol, 'method'])['mae'].agg(['mean', 'std', 'count']).reset_index()
    g['sem'] = g['std'].fillna(0) / np.sqrt(g['count'].clip(lower=1))
    for meth in ORDER:
        c = g[g['method'] == meth].sort_values(xcol)
        if c.empty:
            continue
        st = STYLE[meth]
        ax.plot(c[xcol], c['mean'], marker='o', lw=st.get('lw', 1.7),
                color=st['color'], ls=st['ls'], label=st['label'])
        ax.fill_between(c[xcol], c['mean'] - c['sem'], c['mean'] + c['sem'],
                        color=st['color'], alpha=0.12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('held-out (model, task) MAE')
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='math')
    ap.add_argument('--dir', default='results-pkps-rd2cv')
    ap.add_argument('--out', default='results-pkps-rd2cv/fig_rd2_math')
    args = ap.parse_args()
    d = Path(args.dir)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    tp = d / f'rd2_{args.dataset}_task_parity.csv'
    nm = d / f'rd2_{args.dataset}_n_models.csv'
    if tp.exists():
        panel(axes[0], pd.read_csv(tp), 'obs_prob', 'task-observation probability')
        axes[0].set_title('(a) sparsity', loc='left', fontsize=10)
    if nm.exists():
        panel(axes[1], pd.read_csv(nm), 'n_models', 'number of models')
        axes[1].set_title('(b) cohort size', loc='left', fontsize=10)
    fig.suptitle(f'RD2 missing tasks ({args.dataset.upper()}): PKPS vs matrix completion',
                 fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}.{ext}', dpi=200, bbox_inches='tight')
    print(f'wrote {args.out}.png')


if __name__ == '__main__':
    main()
