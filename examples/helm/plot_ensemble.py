#!/usr/bin/env python
"""Plot the DKPS / matrix-completion / ensemble sweep (combined vs MC vs blend)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

STYLE = {
    'combined':    dict(color='#1d4ed8', ls='-',  label='PKPS combined'),
    'matcomplete': dict(color='#7c3aed', ls='--', label='matrix completion'),
    'ensemble':    dict(color='#15803d', ls='-',  label='ensemble', lw=2.6),
}
ORDER = ['combined', 'matcomplete', 'ensemble']
XLABEL = {'obs_prob': 'task observation probability', 'n_models': 'number of models'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results-doublekernel-google/ensemble_sweep.csv')
    ap.add_argument('--out', default='results-doublekernel-google/fig_helm_ensemble')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    sweeps = [s for s in ['obs_prob', 'n_models'] if s in df['sweep'].unique()]
    fig, axes = plt.subplots(1, len(sweeps), figsize=(5.2 * len(sweeps), 4.3), squeeze=False)
    for j, sw in enumerate(sweeps):
        ax = axes[0][j]
        sub = df[df['sweep'] == sw]
        for est in ORDER:
            c = sub[sub['estimator'] == est].sort_values('x')
            st = STYLE[est]
            ax.plot(c['x'], c['mae'], marker='o', lw=st.get('lw', 1.8),
                    color=st['color'], ls=st['ls'], label=st['label'])
        ax.set_xlabel(XLABEL.get(sw, sw))
        if j == 0:
            ax.set_ylabel('held-out score MAE')
        ax.set_title(f'({chr(97 + j)}) {XLABEL.get(sw, sw)}', fontsize=10, loc='left')
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
    fig.suptitle('HELM MATH: PKPS + matrix-completion ensemble (alpha tuned on held-out seeds)',
                 fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}.{ext}', dpi=200, bbox_inches='tight')
    print(f'wrote {args.out}.png')


if __name__ == '__main__':
    main()
