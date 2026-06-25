#!/usr/bin/env python
"""RD1 figure: query budget (paired) + unpaired panels, sample/IRT/DKPS/PKPS/ens."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {
    'sample': dict(color='#6b7280', ls='--', label='sample'),
    'irt':    dict(color='#c2410c', ls='--', label='IRT'),
    'dkps':   dict(color='#0f766e', ls='-', label='DKPS (delta)'),
    'pkps':   dict(color='#1d4ed8', ls='-', label='PKPS (adaptive)'),
    'ens':    dict(color='#15803d', ls='-', label='ensemble', lw=2.4),
}
ORDER = ['sample', 'irt', 'dkps', 'pkps', 'ens']


def panel(ax, df, xcol, xlabel, logx=False):
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
    ax.set_ylabel('benchmark-score MAE')
    if logx:
        ax.set_xscale('log')
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='math')
    ap.add_argument('--dir', default='results-pkps-rd1')
    ap.add_argument('--out', default='results-pkps-rd1/fig_rd1_math')
    args = ap.parse_args()
    d = Path(args.dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    a = d / f'rd1_{args.dataset}_budget.csv'
    b = d / f'rd1_{args.dataset}_pairing.csv'
    c = d / f'rd1_{args.dataset}_n_models.csv'
    if a.exists():
        panel(axes[0], pd.read_csv(a), 'n_total', 'query budget (paired)', logx=True)
        axes[0].set_title('(a) query efficiency', loc='left', fontsize=10)
    if b.exists():
        panel(axes[1], pd.read_csv(b), 'n_paired', 'number of paired queries (total=16)')
        axes[1].set_title('(b) pairing cliff', loc='left', fontsize=10)
    if c.exists():
        panel(axes[2], pd.read_csv(c), 'n_models', 'number of models')
        axes[2].set_title('(c) number of models', loc='left', fontsize=10)
    fig.suptitle(f'Unpaired and low-budget evaluation ({args.dataset.upper()}): PKPS vs DKPS / IRT / sample',
                 fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}.{ext}', dpi=200, bbox_inches='tight')
    print(f'wrote {args.out}.png')


if __name__ == '__main__':
    main()
