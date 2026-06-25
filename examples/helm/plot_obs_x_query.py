#!/usr/bin/env python
"""2D sweep plot: MAE vs obs_prob, one line per query_obs_prob, panel per method."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

PANELS = [('rbf_paired', 'paired'), ('rbf_unpaired', 'unpaired'),
          ('rbf_combined', 'combined'), ('matcomplete', 'matrix completion')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results-doublekernel-google/obs_x_query.csv')
    ap.add_argument('--out', default='results-doublekernel-google/fig_helm_obs_x_query')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    g = (df.groupby(['estimator', 'query_obs_prob', 'obs_prob'])['mae']
         .mean().reset_index())
    qvals = sorted(df['query_obs_prob'].unique())
    colors = {q: cm.viridis(i / max(1, len(qvals) - 1)) for i, q in enumerate(qvals)}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), squeeze=False)
    ymin = g['mae'].min() * 0.95
    ymax = g['mae'].max() * 1.03
    for ax, (est, title) in zip(axes.flat, PANELS):
        sub = g[g['estimator'] == est]
        for q in qvals:
            c = sub[sub['query_obs_prob'] == q].sort_values('obs_prob')
            ax.plot(c['obs_prob'], c['mae'], marker='o', lw=1.7,
                    color=colors[q], label=f'q={q:g}')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('task observation probability')
        ax.set_ylabel('held-out score MAE')
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(frameon=False, fontsize=8, title='query obs', title_fontsize=8)
    fig.suptitle('HELM MATH: obs_prob x query_obs_prob (observed = sample score; '
                 'MAE vs full score)', fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}.{ext}', dpi=200, bbox_inches='tight')
    print(f'wrote {args.out}.png')


if __name__ == '__main__':
    main()
