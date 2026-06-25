#!/usr/bin/env python
"""RD2 under evaluation noise: held-out MAE vs queries-per-observed-cell, for
MATH / WMT / SUITE. The PKPS ensemble overtakes matrix completion as cells get
noisier (fewer queries) -- even on low-rank matrices where MC wins noiselessly."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {'matcomplete': dict(color='#7c3aed', ls='--', label='matrix completion'),
         'combined': dict(color='#1d4ed8', ls='-', label='PKPS'),
         'ensemble': dict(color='#15803d', ls='-', label='ensemble', lw=2.4)}
ORDER = ['matcomplete', 'combined', 'ensemble']
DL = {'math': 'MATH', 'wmt_14': 'WMT', 'suite': 'SUITE'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['math', 'wmt_14', 'suite'])
    ap.add_argument('--dir', default='results-pkps-rd2cv')
    ap.add_argument('--out', default='results-pkps-rd2cv/fig_rd2_noise')
    args = ap.parse_args()
    d = Path(args.dir)
    n = len(args.datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.2), squeeze=False)
    for j, ds in enumerate(args.datasets):
        ax = axes[0][j]
        f = d / f'rd2_{ds}_query_obs.csv'
        if not f.exists():
            continue
        df = pd.read_csv(f)
        g = df.groupby(['query_obs', 'method'])['mae'].agg(['mean', 'std', 'count']).reset_index()
        g['sem'] = g['std'].fillna(0) / np.sqrt(g['count'].clip(lower=1))
        for m in ORDER:
            c = g[g['method'] == m].sort_values('query_obs')
            st = STYLE[m]
            ax.plot(c['query_obs'], c['mean'], marker='o', lw=st.get('lw', 1.7),
                    color=st['color'], ls=st['ls'], label=st['label'])
            ax.fill_between(c['query_obs'], c['mean'] - c['sem'], c['mean'] + c['sem'],
                            color=st['color'], alpha=0.12)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('queries per observed cell  (noise $\\leftarrow$)')
        if j == 0:
            ax.set_ylabel('held-out (model, task) MAE')
        ax.set_title(f'{DL.get(ds, ds)}', loc='left', fontsize=11)
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('RD2 under evaluation noise: the PKPS ensemble overtakes matrix completion '
                 'as cells get noisier', fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}.{ext}', dpi=200, bbox_inches='tight')
    print(f'wrote {args.out}.png')


if __name__ == '__main__':
    main()
