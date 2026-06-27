#!/usr/bin/env python
"""Main real-data figure 1 -- matrix completion on the heterogeneous 18-task suite.
Two completion levers (task coverage, cohort size); the item-level embedding (PKPS) vs the
score-only matrix completion and sample baselines, and the ensemble that combines them.
Full-matrix MAE vs the true score, error bars = +/- 1 SEM over seeds."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd

D = Path('results-pkps-unified')
# colour = method, line style = information used (solid = response embedding, dashed = score-only;
# the ensemble is solid and thicker).
STYLE = {'pkps_all':  dict(color='#E24A33', ls='-',  label='PKPS (embedding)'),
         'mc_all':    dict(color='#348ABC', ls='--', label='matrix completion'),
         'score_all': dict(color='#777777', ls='--', label='sample / score only'),
         'ens_all':   dict(color='#8C564B', ls='-',  label='ensemble', lw=2.6)}
ORDER = ['score_all', 'mc_all', 'pkps_all', 'ens_all']
PANELS = [('coverage', 'p', 'task coverage'), ('n_models', 'n_models', 'number of models')]


def panel(ax, sweep, xcol, xlabel):
    df = pd.read_csv(D / f'unified_suite_{sweep}.csv')
    g = df.groupby(xcol)[ORDER].agg(['mean', 'sem'])
    x = g.index.values
    for m in ORDER:
        st = STYLE[m]
        ax.errorbar(x, g[(m, 'mean')], yerr=g[(m, 'sem')], marker='o', ms=4,
                    lw=st.get('lw', 1.6), color=st['color'], ls=st['ls'], label=st['label'],
                    capsize=2, elinewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25, lw=0.6)


fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
for ax, (sweep, xcol, xlabel) in zip(axes, PANELS):
    panel(ax, sweep, xcol, xlabel)
axes[0].set_ylabel('full-matrix MAE')
axes[0].set_title('(a)', loc='left', fontweight='bold', fontsize=11)
axes[1].set_title('(b)', loc='left', fontweight='bold', fontsize=11)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.02), fontsize=9.5)
fig.suptitle('Matrix completion on the heterogeneous suite: the embedding ensemble beats '
             'score-only completion across coverage and cohort', fontsize=11, y=1.08)
fig.tight_layout(rect=[0, 0, 1, 0.97])
for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-unified/fig_suite_completion.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-unified/fig_suite_completion.png')
