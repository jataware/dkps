#!/usr/bin/env python
"""Three-lever view of the unified problem (one dataset): full-matrix MAE vs queries
per cell (noise), task coverage, and cohort size. One sample-MC-PKPS ensemble beats
the best score-only predictor across every lever."""
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
         'ens_all': dict(color='#15803d', ls='-', label='ensemble (sample$\\oplus$MC$\\oplus$PKPS)', lw=2.4)}
ORDER = ['mc_all', 'pkps_all', 'score_all', 'ens_all']
PANELS = [('queries', 'q', 'queries per cell  ($0$ = complete $\\to$ denoise)', 'cat'),
          ('coverage', 'p', 'task coverage', False),
          ('n_models', 'n_models', 'number of models', False)]
DL = {'math': 'MATH', 'wmt_14': 'WMT', 'suite': 'SUITE'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='suite')
    ap.add_argument('--dir', default='results-pkps-unified')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or f'{args.dir}/fig_unified_levers_{args.dataset}'
    d = Path(args.dir)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for ax, (sweep, xcol, xlabel, logx) in zip(axes, PANELS):
        f = d / f'unified_{args.dataset}_{sweep}.csv'
        if not f.exists():
            continue
        g = pd.read_csv(f).groupby(xcol)[ORDER].mean()
        cat = (logx == 'cat')                      # categorical x so q=0 fits on the axis
        xs = np.arange(len(g)) if cat else g.index.values
        for m in ORDER:
            st = STYLE[m]
            x_m, y_m = xs, g[m].values
            # at q=0 there is no sample -- the score-only predictor is just MC there,
            # so don't draw the sample/score line at q=0 (start it at q=1).
            if cat and m == 'score_all' and g.index[0] == 0:
                x_m, y_m = xs[1:], y_m[1:]
            ax.plot(x_m, y_m, marker='o', lw=st.get('lw', 1.6), color=st['color'],
                    ls=st['ls'], label=st['label'])
        if cat:
            ax.set_xticks(xs); ax.set_xticklabels([int(v) for v in g.index])
            ax.axvline(0.5, color='#94a3b8', lw=0.8, ls=':')   # completion | denoise boundary
        elif logx:
            ax.set_xscale('log', base=2)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel('full-matrix MAE (denoise + complete)')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f'Unified estimation of the full score matrix ({DL.get(args.dataset, args.dataset)}): '
                 f'one ensemble beats the best score-only predictor across all three levers', fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=200, bbox_inches='tight')
    print(f'wrote {out}.png')


if __name__ == '__main__':
    main()
