#!/usr/bin/env python
"""Descriptive figure: the product kernel under paired vs unpaired evaluation.

For two models A, B, the PKPS affinity is

    A_AB = sum_{j,l} k_Q(q_j, q_l) k_R(x_Aj, x_Bl) / sum_{j,l} k_Q(q_j, q_l),

i.e. every (A-response j, B-response l) pair contributes, weighted by the query
kernel k_Q. We plot that weight matrix over the two models' answered queries.

  - DKPS uses the query-identity kernel k_Q = delta: ONLY pairs on the same query
    survive (the diagonal). Under paired data the diagonal is full; under unpaired
    data it is empty -> DKPS discards essentially every response.
  - PKPS uses an RBF k_Q: responses to *similar* queries bridge the two models, so
    the whole matrix contributes even when no query is shared.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

rng = np.random.default_rng(1)
m = 12                         # queries answered per model
sigma = 1.1                    # RBF query bandwidth (in coordinate units)

# A shared 1-D "query space"; queries live at integer-ish coordinates.
# Paired: both models answer the SAME queries. Unpaired: disjoint query sets that
# nonetheless cover a similar region (so similar queries exist to bridge).
paired_c = np.sort(rng.uniform(0, m, m))
qa_paired, qb_paired = paired_c, paired_c.copy()
qa_unpaired = np.sort(rng.uniform(0, m, m))
qb_unpaired = np.sort(rng.uniform(0, m, m))

CMAP = LinearSegmentedColormap.from_list('kq', ['#f8fafc', '#bfdbfe', '#1d4ed8'])
SAME_TOL = 1e-9


def kq_rbf(ca, cb):
    return np.exp(-(ca[:, None] - cb[None, :]) ** 2 / (2 * sigma ** 2))


def kq_delta(ca, cb):
    return (np.abs(ca[:, None] - cb[None, :]) < SAME_TOL).astype(float)


def draw(ax, W, title, label, ok):
    ax.imshow(W, cmap=CMAP, vmin=0, vmax=1, origin='lower', aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor('#cbd5e1')
    if title:
        ax.set_title(title, fontsize=10)
    ax.text(0.5, -0.09, label, transform=ax.transAxes, ha='center', va='top',
            fontsize=9, color='#15803d' if ok else '#b91c1c')


T_DELTA = 'DKPS  ·  query-identity kernel $k_Q=\\delta$'
T_RBF = 'PKPS  ·  query-similarity kernel $k_Q=$ RBF'
fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.6))

# DKPS (delta): the unit of information is a *shared query* (a diagonal match).
n_shared_p = int((kq_delta(qa_paired, qb_paired) > 0.5).sum())
n_shared_u = int((kq_delta(qa_unpaired, qb_unpaired) > 0.5).sum())
# PKPS (rbf): every response pair contributes (weighted) -> count effective weight.
draw(axes[0, 0], kq_delta(qa_paired, qb_paired), T_DELTA,
     f'{n_shared_p} of {m} queries shared  ·  comparison supported', ok=True)
draw(axes[0, 1], kq_rbf(qa_paired, qb_paired), T_RBF,
     f'all {m}×{m} response pairs contribute', ok=True)
draw(axes[1, 0], kq_delta(qa_unpaired, qb_unpaired), '',
     f'{n_shared_u} of {m} queries shared  ·  no comparison possible', ok=False)
draw(axes[1, 1], kq_rbf(qa_unpaired, qb_unpaired), '',
     f'all {m}×{m} response pairs contribute, bridged by similarity', ok=True)

# row labels
axes[0, 0].set_ylabel('Paired evaluation\n(models answer the same queries)', fontsize=10)
axes[1, 0].set_ylabel('Unpaired evaluation\n(models answer different queries)', fontsize=10)
# axis meaning (shared, only need it once)
for ax in (axes[1, 0], axes[1, 1]):
    ax.set_xlabel("model B's answered queries", fontsize=9)
axes[0, 0].annotate("model A's answered queries", xy=(-0.18, 0.5), xycoords='axes fraction',
                    rotation=90, ha='center', va='center', fontsize=9, color='#334155')

fig.suptitle('Paired DKPS discards unpaired responses; PKPS keeps them via query similarity',
             fontsize=12, y=0.98)
fig.text(0.5, 0.005,
         'Each cell = one (model-A response, model-B response) pair, shaded by the query-kernel weight it '
         'contributes to the model affinity $A_{AB}$.\nWhen evaluations are unpaired the identity kernel has '
         'no diagonal left and the comparison collapses (red); the RBF kernel still bridges responses to '
         'similar queries (green).',
         ha='center', fontsize=8.5, color='#475569')
fig.tight_layout(rect=[0.02, 0.04, 1, 0.96])
for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-rd1/fig_concept.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-rd1/fig_concept.png')
