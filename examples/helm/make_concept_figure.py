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

# ---- benchmark-score prediction example (4th column) ------------------------
# A small model population with a latent ability that sets each model's true full-benchmark
# score. Each model answers an UNPAIRED random subset of a shared query pool, so pairwise
# overlap is small, and the ability signal is WEAK per query (swamped by per-query noise):
# only the average over many queries reveals it. We embed the models in each perspective
# space and predict every model's score by leave-one-out regression onto the embedding.
# PKPS bridges responses to *similar* queries and predicts well; the DKPS limit sees only
# the tiny shared overlap, cannot average enough, and its prediction degrades.
rngE = np.random.default_rng(4)
NQ, FRAC, NM = 24, 0.30, 12                        # pool size, answered fraction, n models
qE = np.concatenate([c + rngE.normal(0, 0.35, (NQ // 2, 2)) for c in ([0, 0], [2.6, 1.8])])
abil = np.sort(rngE.uniform(-1.5, 1.5, NM))        # latent model ability
yE = 0.5 + 0.5 * np.tanh(1.2 * abil)               # true full-benchmark score in [0, 1]
edir = np.array([1.0, 0.0])
ansE = [np.sort(rngE.choice(NQ, max(2, int(FRAC * NQ)), replace=False)) for _ in range(NM)]
Xe = {(m, q): abil[m] * edir + 0.18 * (qE[q] - qE.mean(0)) + rngE.normal(0, 1.1, 2)
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

def _mds2(A):                                      # classical MDS to 2-D from an affinity
    d2 = np.maximum(np.diag(A)[:, None] + np.diag(A)[None, :] - 2 * A, 0)
    J = np.eye(NM) - 1.0 / NM
    w, V = np.linalg.eigh(-0.5 * J @ d2 @ J)
    idx = np.argsort(w)[::-1][:2]
    return V[:, idx] * np.sqrt(np.maximum(w[idx], 0))

def _loo_predict(Z):                               # predict each score from the others' embeddings
    Zb = np.c_[np.ones(NM), Z]
    pred = np.zeros(NM)
    for m in range(NM):
        tr = np.arange(NM) != m
        beta, *_ = np.linalg.lstsq(Zb[tr], yE[tr], rcond=None)
        pred[m] = np.clip(Zb[m] @ beta, 0, 1)
    return pred

Zdk, Zpk = _mds2(_affinity(True)), _mds2(_affinity(False))
pred_dk, pred_pk = _loo_predict(Zdk), _loo_predict(Zpk)
mae_dk, mae_pk = np.mean(np.abs(pred_dk - yE)), np.mean(np.abs(pred_pk - yE))

fig = plt.figure(figsize=(18.0, 4.7))
gs = GridSpec(2, 5, width_ratios=[0.82, 1, 1, 0.6, 0.6], height_ratios=[0.58, 1.42],
              wspace=0.28, hspace=0.05)
axBlk = fig.add_subplot(gs[0, 0])
axEmb = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[:, 1])
axR = fig.add_subplot(gs[:, 2])
axDKe = fig.add_subplot(gs[0, 3]); axDKp = fig.add_subplot(gs[0, 4])     # DKPS embed | pred
axPKe = fig.add_subplot(gs[1, 3]); axPKp = fig.add_subplot(gs[1, 4])     # PKPS embed | pred

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

# ---- RIGHT 2x2 BLOCK: perspective embedding (coloured by score) + LOO prediction
SCM = 'viridis'
for axe, axp, Z, pred, mae in [(axDKe, axDKp, Zdk, pred_dk, mae_dk),
                               (axPKe, axPKp, Zpk, pred_pk, mae_pk)]:
    # MDS 2-D model representations (one dot per model), coloured by true benchmark score
    sc = axe.scatter(Z[:, 0], Z[:, 1], c=yE, cmap=SCM, vmin=0, vmax=1,
                     s=46, edgecolors='white', lw=0.8, zorder=3)
    axe.set_aspect('equal'); axe.set_xticks([]); axe.set_yticks([])
    for s in axe.spines.values():
        s.set_edgecolor('#d7dde6')
    # predict each model's score by LOO regression onto its 2-D representation
    axp.plot([0, 1], [0, 1], ls='--', color='#94a3b8', lw=1.2, zorder=1)
    axp.scatter(yE, pred, c=yE, cmap=SCM, vmin=0, vmax=1, s=40, edgecolors='white', lw=0.8, zorder=3)
    axp.set_xlim(-0.05, 1.05); axp.set_ylim(-0.05, 1.05); axp.set_aspect('equal')
    axp.set_xticks([0, 1]); axp.set_yticks([0, 1]); axp.tick_params(labelsize=8, color='#d7dde6')
    for s in axp.spines.values():
        s.set_edgecolor('#d7dde6')
    axp.text(0.06, 0.94, f'MAE {mae:.2f}', transform=axp.transAxes, ha='left', va='top',
             fontsize=9, color='#334155')
axPKe.set_xlabel('perspective space', fontsize=9, color='#475569', labelpad=3)
axPKp.set_xlabel('true score', fontsize=9, color='#475569', labelpad=1)
axDKp.set_ylabel('predicted', fontsize=9, color='#475569', labelpad=1)
axPKp.set_ylabel('predicted', fontsize=9, color='#475569', labelpad=1)

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
# arrange the right 2x2 block (DKPS row / PKPS row) x (embed | pred) as equal squares
# spanning the matrices' vertical extent
pr = axR.get_position()
gap_v = 0.18
h = (pr.height - gap_v) / 2
ce, cp = axDKe.get_position(), axDKp.get_position()
for ax, x0, w, y0 in [(axDKe, ce.x0, ce.width, pr.y1 - h), (axDKp, cp.x0, cp.width, pr.y1 - h),
                      (axPKe, ce.x0, ce.width, pr.y0),     (axPKp, cp.x0, cp.width, pr.y0)]:
    ax.set_position([x0, y0, w, h])

# slim colour key (true benchmark score) for the embedding + prediction dots
cpp = axDKp.get_position()
scax = fig.add_axes([cpp.x1 + 0.010, pr.y0, 0.007, pr.height])
scb = fig.colorbar(sc, cax=scax); scb.set_label('true score', fontsize=8.5)
scb.set_ticks([0, 1]); scb.ax.tick_params(labelsize=7)

Y = max(ax.get_position().y1 for ax in (axBlk, axC, axR)) + 0.012
for ax, t in [(axBlk, r'$\bf{(a)}$  answered queries'),
              (axC, r'$\bf{(b)}$  DKPS  ·  $k_Q=\delta$'),
              (axR, r'$\bf{(c)}$  PKPS  ·  $k_Q=$ RBF'),
              (axDKe, r'$\bf{(d)}$  DKPS embed'),
              (axDKp, r"$\bf{(d')}$  DKPS pred")]:
    p = ax.get_position()
    fig.text((p.x0 + p.x1) / 2, Y, t, ha='center', va='bottom', fontsize=11)
for ax, t in [(axPKe, r'$\bf{(e)}$  PKPS embed'), (axPKp, r"$\bf{(e')}$  PKPS pred")]:
    p = ax.get_position()
    fig.text((p.x0 + p.x1) / 2, p.y1 + 0.012, t, ha='center', va='bottom', fontsize=11)

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
