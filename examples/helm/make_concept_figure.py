#!/usr/bin/env python
"""Descriptive figure: from unpaired answered-query sets to the product kernel.

Two models i (red) and i' (blue) each answer their OWN subset of a shared query pool.
The PKPS affinity weights every (i-response, i'-response) pair by the query kernel:

    A_{i i'} = sum_{j,l} k_Q(q_j, q_l) k_R(x_ij, x_{i'l}) / sum_{j,l} k_Q(q_j, q_l).

  LEFT  (two stacked views of the SAME queries):
    top    -- queries as dots in embedding space (colour = which model answered; nearby = similar).
    bottom -- the same queries as a row of blocks: model i (red) / model i' (blue) answered subsets.
  CENTER : DKPS, k_Q=delta -> purple only on the diagonal where i and i' answered the SAME
           query (the overlap). Almost everything is discarded.
  RIGHT  : PKPS, k_Q=RBF  -> a dense band; every query pair contributes, weighted by query
           similarity, so responses to *similar* (not identical) queries are used.

red (model i) x blue (model i') -> purple = the product kernel.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

N = 12
RED, BLUE, PURPLE, GRAY = '#dc2626', '#2563eb', '#7c3aed', '#cbd5e1'
EMPTY = '#eef2f7'
PCMAP = LinearSegmentedColormap.from_list('pk', ['#ffffff', '#ddd6fe', PURPLE])

# which model answered each of the N queries (unpaired: mostly one model, a few both/neither).
Si = np.array([0, 1, 3, 4, 6, 7, 10])      # model i  (red)
Sip = np.array([1, 2, 4, 7, 8, 9, 11])     # model i' (blue)
inI = np.zeros(N, bool); inI[Si] = True
inP = np.zeros(N, bool); inP[Sip] = True
overlap = np.where(inI & inP)[0]
# per-query label: 0 i-only, 1 i'-only, 2 both, 3 neither
lab = np.where(inI & inP, 2, np.where(inI, 0, np.where(inP, 1, 3)))

# queries live in a 2-D embedding; the kernel matrices below are computed FROM these
# positions over a single UNIVERSAL ordering of the queries, shared by both axes.
rngL = np.random.default_rng(4)
P = np.concatenate([c + rngL.normal(0, 0.3, (4, 2))
                    for c in ([0, 0], [2.7, 0.2], [1.3, 2.4])])     # 3 tight clusters x 4 = 12 queries
t = (P - P.mean(0)) @ np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]   # 1-D order
# order the union of the two models' answered queries by the 1-D embedding projection and
# index BOTH axes by this single ordering. The PKPS query kernel is then a symmetric Gram
# matrix; DKPS's identity kernel is active only on the diagonal, at the shared queries.
U = np.union1d(Si, Sip)
U = U[np.argsort(t[U])]
sharedU = np.isin(U, overlap)
D2 = ((P[U][:, None] - P[U][None, :]) ** 2).sum(-1)
Krbf = np.exp(-D2 / (2 * 1.7 ** 2))            # PKPS: symmetric query-similarity kernel
Kdkps = np.diag(sharedU.astype(float))         # DKPS: identity, active only at shared queries

# ---- example perspective spaces (4th column) --------------------------------
# A small model population: two groups with opposite response styles. Each model answers an
# UNPAIRED random subset of a shared query pool, so pairwise query overlap is small. The
# product kernel (RBF k_Q) bridges responses to *similar* queries and recovers the two
# groups; the DKPS limit (delta k_Q) sees only the tiny shared overlap and collapses.
rngE = np.random.default_rng(8)
NQ, FRAC, NG = 24, 0.30, 6                        # pool size, answered fraction, models/group
qE = np.concatenate([c + rngE.normal(0, 0.35, (NQ // 2, 2)) for c in ([0, 0], [2.6, 1.8])])
grp = np.array([0] * NG + [1] * NG)
NM = len(grp)
# the group signal is WEAK per query (+/-0.6) and swamped by per-query noise (1.0): no single
# query tells the groups apart, only the average over many does. DKPS, restricted to the few
# shared queries, cannot average enough and collapses; PKPS averages over all similar queries.
sty = np.array([[0.6, 0.0], [-0.6, 0.0]])         # group response styles
ansE = [np.sort(rngE.choice(NQ, max(2, int(FRAC * NQ)), replace=False)) for _ in range(NM)]
Xe = {(m, q): sty[grp[m]] + 0.18 * (qE[q] - qE.mean(0)) + rngE.normal(0, 1.0, 2)
      for m in range(NM) for q in ansE[m]}

def _affinity(delta, sig=0.8):
    A = np.zeros((NM, NM))
    for a in range(NM):
        for b in range(NM):
            num = den = 0.0
            for qa in ansE[a]:
                for qb in ansE[b]:
                    kq = (1.0 if qa == qb else 0.0) if delta else \
                        np.exp(-((qE[qa] - qE[qb]) ** 2).sum() / (2 * sig ** 2))
                    num += kq * (Xe[(a, qa)] @ Xe[(b, qb)]); den += kq
            A[a, b] = num / den if den > 0 else 0.0
    return A

def _mds2(A):                                     # classical MDS to 2-D from an affinity
    d2 = np.maximum(np.diag(A)[:, None] + np.diag(A)[None, :] - 2 * A, 0)
    J = np.eye(NM) - 1.0 / NM
    w, V = np.linalg.eigh(-0.5 * J @ d2 @ J)
    idx = np.argsort(w)[::-1][:2]
    Z = V[:, idx] * np.sqrt(np.maximum(w[idx], 0))
    return Z / (np.abs(Z).max() + 1e-9)

Zdk, Zpk = _mds2(_affinity(True)), _mds2(_affinity(False))

fig = plt.figure(figsize=(16.4, 4.7))
gs = GridSpec(2, 4, width_ratios=[0.82, 1, 1, 0.92], height_ratios=[0.58, 1.42],
              wspace=0.28, hspace=0.05)
axBlk = fig.add_subplot(gs[0, 0])
axEmb = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[:, 1])
axR = fig.add_subplot(gs[:, 2])
axDK = fig.add_subplot(gs[0, 3])
axPK = fig.add_subplot(gs[1, 3])

# ---- LEFT TOP: queries as dots in embedding space ----------------------------
for k, col, sz, z in [(3, GRAY, 42, 1), (0, RED, 105, 3), (1, BLUE, 105, 3), (2, PURPLE, 125, 4)]:
    m = lab == k
    axEmb.scatter(P[m, 0], P[m, 1], s=sz, c=col, edgecolors='white', lw=1.2, zorder=z)
axEmb.set_aspect('equal'); axEmb.axis('off')
axEmb.set_title('query embedding  (nearby = similar)', fontsize=10.5, pad=8)

# ---- LEFT BOTTOM: same queries as a row of answered blocks --------------------
for j in range(N):
    axBlk.add_patch(Rectangle((j, 1), 1, 1, facecolor=RED if inI[j] else EMPTY,
                              edgecolor='white', lw=1.5))
    axBlk.add_patch(Rectangle((j, 0), 1, 1, facecolor=BLUE if inP[j] else EMPTY,
                              edgecolor='white', lw=1.5))
axBlk.set_xlim(-0.3, N); axBlk.set_ylim(-0.4, 2.4)
axBlk.set_aspect('equal'); axBlk.set_anchor('C'); axBlk.axis('off')
axBlk.text(-0.45, 1.5, 'model $i$', ha='right', va='center', fontsize=11, color=RED)
axBlk.text(-0.45, 0.5, "model $i'$", ha='right', va='center', fontsize=11, color=BLUE)
# ---- CENTER & RIGHT: the kernel-weight matrices ------------------------------
for ax, W in [(axC, Kdkps), (axR, Krbf)]:
    im = ax.imshow(W, cmap=PCMAP, vmin=0, vmax=1, origin='upper', aspect='equal')
    ax.set_xticks(np.arange(-.5, W.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, W.shape[0], 1), minor=True)
    ax.grid(which='minor', color='#eef1f6', lw=0.5)
    ax.tick_params(which='minor', length=0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor('#d7dde6')
    ax.set_xlabel('query index', fontsize=10, color='#475569')
axC.set_ylabel('query index', fontsize=10, color='#475569')

# ---- 4th COLUMN: the resulting perspective spaces (each dot = one model) ------
for ax, Z in [(axDK, Zdk), (axPK, Zpk)]:
    for g, col in [(0, RED), (1, BLUE)]:
        m = grp == g
        ax.scatter(Z[m, 0], Z[m, 1], s=66, c=col, edgecolors='white', lw=1.0, zorder=3)
    ax.set_xlim(-1.28, 1.28); ax.set_ylim(-1.28, 1.28)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor('#d7dde6')
axDK.set_xlabel('models collapse', fontsize=9.5, color='#475569', labelpad=2)
axPK.set_xlabel('models separate', fontsize=9.5, color='#475569', labelpad=2)

cax = make_axes_locatable(axR).append_axes('right', size='4.5%', pad=0.08)
cb = fig.colorbar(im, cax=cax)
cb.set_label('query-kernel weight  $k_Q(q_j,q_l)$', fontsize=9.5)
cb.set_ticks([0, 1])
# steal the same sliver from the DKPS axes so both matrices render at identical size
spacer = make_axes_locatable(axC).append_axes('right', size='4.5%', pad=0.08)
spacer.axis('off')

fig.suptitle('PKPS can use all available information',
             fontsize=13.5, fontweight='bold', y=0.965)

# place the (a)/(b)/(c) panel titles at a common figure-y. The panels have different box
# tops (the matrices are letterbox-centered), so per-axes titles can't line up -- figure-
# coordinate text over each panel centre does.
fig.canvas.draw()
# stack the two perspective panels as equal squares spanning the matrices' vertical extent
pr = axR.get_position(); pcol = axDK.get_position()
gap = 0.16
h = (pr.height - gap) / 2
axDK.set_position([pcol.x0, pr.y1 - h, pcol.width, h])     # top = DKPS
axPK.set_position([pcol.x0, pr.y0, pcol.width, h])         # bottom = PKPS

Y = max(ax.get_position().y1 for ax in (axBlk, axC, axR)) + 0.012
for ax, t in [(axBlk, r'$\bf{(a)}$  answered queries'),
              (axC, r'$\bf{(b)}$  DKPS  ·  $k_Q=\delta$'),
              (axR, r'$\bf{(c)}$  PKPS  ·  $k_Q=$ RBF'),
              (axDK, r'$\bf{(d)}$  DKPS perspective')]:
    p = ax.get_position()
    fig.text((p.x0 + p.x1) / 2, Y, t, ha='center', va='bottom', fontsize=12)
pp = axPK.get_position()
fig.text((pp.x0 + pp.x1) / 2, pp.y1 + 0.012, r'$\bf{(e)}$  PKPS perspective',
         ha='center', va='bottom', fontsize=12)

# size the embedding to fill the lower-left region, and put a shared legend at the bottom
# (fixed in figure coords) so the left column bottoms out level with the matrices' labels.
pe = axEmb.get_position(); pm = axC.get_position()
axEmb.set_position([pe.x0, 0.235, pe.width, pe.height * 0.74])
xc = (axBlk.get_position().x0 + axBlk.get_position().x1) / 2
fig.legend(handles=[Line2D([0], [0], marker='s', ls='', mfc=RED, mec='white', ms=9, label='model $i$'),
                    Line2D([0], [0], marker='s', ls='', mfc=BLUE, mec='white', ms=9, label="model $i'$"),
                    Line2D([0], [0], marker='o', ls='', mfc=PURPLE, mec='white', ms=9, label='both')],
           loc='lower center', bbox_to_anchor=(xc, 0.125), bbox_transform=fig.transFigure,
           ncol=3, frameon=False, fontsize=9.5, handletextpad=0.2, columnspacing=1.0)

# faint rule grouping the data column (a) apart from the two method panels (b, c)
pc = axC.get_position()
sep_x = (axEmb.get_position().x1 + pc.x0) / 2
fig.add_artist(Line2D([sep_x, sep_x], [pc.y0, pc.y1], transform=fig.transFigure,
                      color='#dce1e8', lw=1.2, zorder=0))

for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-rd1/fig_concept.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-rd1/fig_concept.png')
