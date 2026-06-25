#!/usr/bin/env python
"""Plot a HELM DoubleKernelDKPS sweep (task_parity / query_sparsity / n_models).

Auto-detects the swept x-column (the one among obs_prob/query_obs_prob/n_models
that varies). Use --combined for a 1x3 panel over multiple CSVs.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {'rbf_paired': '#0f766e', 'rbf_unpaired': '#c2410c', 'rbf_combined': '#1d4ed8',
          'matcomplete': '#7c3aed'}
LABELS = {'rbf_paired': 'PKPS (paired k_Q)', 'rbf_unpaired': 'PKPS (unpaired k_Q)',
          'rbf_combined': 'PKPS', 'matcomplete': 'matrix completion'}
ORDER = ['rbf_paired', 'rbf_unpaired', 'rbf_combined', 'matcomplete']
DASHED = {'matcomplete'}
XLABELS = {
    'obs_prob': 'task observation probability',
    'query_obs_prob': 'query observation probability',
    'n_models': 'number of models',
}
CANDIDATES = ['obs_prob', 'query_obs_prob', 'n_models']


def detect_x(df):
    varying = [c for c in CANDIDATES if c in df.columns and df[c].nunique() > 1]
    if not varying:
        # fall back to whichever candidate exists
        for c in CANDIDATES:
            if c in df.columns:
                return c
        raise ValueError('no x-column found')
    return varying[0]


def plot_panel(ax, df, x_col):
    g = df.groupby([x_col, 'estimator'])['mae'].agg(['mean', 'std', 'count']).reset_index()
    g['sem'] = g['std'].fillna(0.0) / np.sqrt(g['count'].clip(lower=1))
    for est in ORDER:
        c = g[g['estimator'] == est].sort_values(x_col)
        if c.empty:
            continue
        ax.plot(c[x_col], c['mean'], marker='o', lw=1.8, color=COLORS[est],
                label=LABELS[est], linestyle='--' if est in DASHED else '-')
        ax.fill_between(c[x_col], c['mean'] - c['sem'], c['mean'] + c['sem'],
                        color=COLORS[est], alpha=0.18)
    ax.set_xlabel(XLABELS.get(x_col, x_col))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default=None)
    args = ap.parse_args()

    csvs = [Path(c) for c in args.csv]
    n = len(csvs)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.2), squeeze=False)
    for j, c in enumerate(csvs):
        df = pd.read_csv(c)
        x_col = detect_x(df)
        ax = axes[0][j]
        plot_panel(ax, df, x_col)
        if j == 0:
            ax.set_ylabel('held-out score MAE')
        ax.set_title(f'({chr(97 + j)}) {XLABELS.get(x_col, x_col)}', fontsize=10, loc='left')
    if args.title:
        fig.suptitle(args.title, fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}.{ext}', dpi=200, bbox_inches='tight')
    print(f'wrote {args.out}.png')


if __name__ == '__main__':
    main()
