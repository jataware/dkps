#!/usr/bin/env python
"""Joint RD1 figure: query-efficient evaluation across the heterogeneous suite.
(a) suite-wide MAE vs query budget; (b) per-dataset MAE at a fixed budget."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {
    'sample': dict(color='#6b7280', ls='--', label='sample mean'),
    'irt':    dict(color='#c2410c', ls=':', label='IRT'),
    'pkps':   dict(color='#1d4ed8', ls='-', label='PKPS'),
    'ens':    dict(color='#15803d', ls='-', label='PKPS + sample', lw=2.4),
}
ORDER = ['sample', 'irt', 'pkps', 'ens']
DLABEL = {'math': 'MATH', 'wmt_14': 'WMT', 'med_qa': 'MedQA', 'legalbench': 'LegalBench'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results-pkps-rd1/rd1_suite_budget.csv')
    ap.add_argument('--fixed_m', type=int, default=8)
    ap.add_argument('--out', default='results-pkps-rd1/fig_rd1_suite')
    args = ap.parse_args()
    df = pd.read_csv(args.csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1, 1.15]))

    # (a) suite-wide efficiency curve
    g = df.groupby(['m', 'method'])['mae'].agg(['mean', 'std', 'count']).reset_index()
    g['sem'] = g['std'].fillna(0) / np.sqrt(g['count'].clip(lower=1))
    ax = axes[0]
    for meth in ORDER:
        c = g[g['method'] == meth].sort_values('m')
        if c.empty:
            continue
        st = STYLE[meth]
        ax.plot(c['m'], c['mean'], marker='o', lw=st.get('lw', 1.7), color=st['color'],
                ls=st['ls'], label=st['label'])
        ax.fill_between(c['m'], c['mean'] - c['sem'], c['mean'] + c['sem'], color=st['color'], alpha=0.12)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('queries per task (budget)')
    ax.set_ylabel('full-score MAE (suite-wide)')
    ax.set_title('(a) efficient evaluation across the suite', loc='left', fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    # focus on the sample/PKPS/ensemble region; IRT sits well above (needs dense pairing)
    irt_mean = g[g['method'] == 'irt']['mean']
    top = 0.22
    ax.set_ylim(0, top)
    if len(irt_mean):
        ax.annotate(f'IRT ≈ {irt_mean.mean():.2f} (off-scale:\nneeds dense paired data)',
                    xy=(1, top), xytext=(1.1, top * 0.93), fontsize=7.5, color='#c2410c', va='top')
    ax.legend(frameon=False, fontsize=8, loc='upper right')

    # (b) per-dataset at fixed budget
    ax = axes[1]
    sub = df[df['m'] == args.fixed_m]
    by = sub.groupby(['dataset', 'method'])['mae'].mean().unstack()
    dsets = [d for d in ['math', 'wmt_14', 'med_qa', 'legalbench'] if d in by.index]
    meths = [m for m in ['sample', 'pkps', 'ens'] if m in by.columns]   # IRT off-scale; omit here
    x = np.arange(len(dsets))
    w = 0.8 / len(meths)
    for j, meth in enumerate(meths):
        ax.bar(x + (j - (len(meths) - 1) / 2) * w, by.loc[dsets, meth], w,
               color=STYLE[meth]['color'], label=STYLE[meth]['label'])
    ax.set_xticks(x)
    ax.set_xticklabels([DLABEL.get(d, d) for d in dsets])
    ax.set_ylabel('full-score MAE')
    ax.set_title(f'(b) per-dataset (budget = {args.fixed_m} queries/task)', loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis='y', alpha=0.25, lw=0.6)

    fig.suptitle('Joint RD1 — query-efficient evaluation of a heterogeneous benchmark suite '
                 '(MATH + WMT + MedQA + LegalBench)', fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}.{ext}', dpi=200, bbox_inches='tight')
    print(f'wrote {args.out}.png')


if __name__ == '__main__':
    main()
