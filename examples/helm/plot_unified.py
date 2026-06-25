#!/usr/bin/env python
"""Unified RD1/RD2: full-matrix MAE vs queries-per-cell. One ensemble of sample
(observed), MC (missing) and PKPS (everywhere) beats the best score-only predictor
across the realistic regime, on math, translation, and the heterogeneous suite."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {'mc_all': dict(color='#7c3aed', ls=':', label='matrix completion'),
         'pkps_all': dict(color='#1d4ed8', ls=':', label='PKPS'),
         'score_all': dict(color='#6b7280', ls='--', label='score only (sample / MC)'),
         'ens_all': dict(color='#15803d', ls='-', label='ensemble (sample $\\oplus$ MC $\\oplus$ PKPS)', lw=2.4)}
ORDER = ['mc_all', 'pkps_all', 'score_all', 'ens_all']
DL = {'math': 'MATH', 'wmt_14': 'WMT', 'suite': 'SUITE'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['math', 'wmt_14', 'suite'])
    ap.add_argument('--dir', default='results-pkps-unified')
    ap.add_argument('--out', default='results-pkps-unified/fig_unified')
    args = ap.parse_args()
    d = Path(args.dir)
    n = len(args.datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.2), squeeze=False)
    for j, ds in enumerate(args.datasets):
        ax = axes[0][j]
        f = d / f'unified_{ds}_queries.csv'
        if not f.exists():
            continue
        df = pd.read_csv(f)
        g = df.groupby('q')[ORDER].mean()
        for m in ORDER:
            st = STYLE[m]
            ax.plot(g.index, g[m], marker='o', lw=st.get('lw', 1.6), color=st['color'],
                    ls=st['ls'], label=st['label'])
        ax.set_xscale('log', base=2)
        ax.set_xlabel('queries per observed cell  (noise $\\leftarrow$)')
        if j == 0:
            ax.set_ylabel('full-matrix MAE (denoise + complete)')
        ax.set_title(DL.get(ds, ds), loc='left', fontsize=11)
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle('Unified estimation of the full score matrix: one sample$\\oplus$MC$\\oplus$PKPS '
                 'ensemble beats the best score-only predictor, by more as queries shrink', fontsize=11)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}.{ext}', dpi=200, bbox_inches='tight')
    print(f'wrote {args.out}.png')


if __name__ == '__main__':
    main()
