#!/usr/bin/env python
"""HEADLINE real-data figure -- the pairing cliff on the heterogeneous 18-task suite.
LEFT: at a fixed per-cell budget, slide queries from fully paired to fully unpaired; DKPS and
IRT collapse, PKPS holds (zoomed to the working band -- IRT runs off-scale and is annotated).
RIGHT: the same unpaired (rho=0) regime per model -- PKPS beats DKPS for most models. MAE vs
the full-eval score, +/-1 SEM over seeds (left); one point per model (right)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BUDGET = 4
STYLE = {'sample': dict(color='#777777', ls='--', label='sample'),
         'irt':    dict(color='#988ED5', ls='--', label='IRT'),
         'dkps':   dict(color='#8EBA42', ls='-',  label='DKPS'),
         'pkps':   dict(color='#E24A33', ls='-',  label='PKPS'),
         'ens':    dict(color='#8C564B', ls='-',  label='ensemble', lw=3.6)}
ORDER = ['sample', 'dkps', 'pkps', 'ens']        # IRT is off-scale -> annotated, not drawn

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.6))

# ---- LEFT: the cliff, zoomed to the working band -----------------------------
df = pd.read_csv('results-pkps-rd1/rd1_suite_pairing.csv')
df = df[df['m'] == BUDGET].copy()
df['rho'] = df['n_paired'] / BUDGET
per_seed = df.groupby(['rho', 'seed', 'method'])['mae'].mean().reset_index()
g = per_seed.groupby(['rho', 'method'])['mae'].agg(['mean', 'sem'])
for m in ORDER:
    st = STYLE[m]; s = g.xs(m, level='method').sort_index()
    axL.errorbar(s.index.values, s['mean'], yerr=s['sem'], marker='o', ms=5,
                 lw=st.get('lw', 2.6), color=st['color'], ls=st['ls'], label=st['label'],
                 capsize=3, elinewidth=0.9)
irt = g.xs('irt', level='method')['mean']
axL.set_ylim(0.082, 0.188)
axL.text(0.30, 0.185, f"IRT off-scale ({irt.loc[1.0]:.2f}$\\to${irt.loc[0.0]:.2f})",
         ha='center', va='top', fontsize=8.5, color=STYLE['irt']['color'], fontweight='bold')
axL.set_xlabel(r"fraction of paired queries  $\rho = m_{ii'}/M_{ij}$")
axL.set_ylabel('MAE vs full-eval score')
axL.set_title('(a) the cliff: PKPS holds as queries unpair', fontsize=10.5)
axL.legend(frameon=False, fontsize=8.5, ncol=2, loc='upper right')
axL.grid(alpha=0.25, lw=0.6)

# ---- RIGHT: per-model PKPS vs DKPS, unpaired ---------------------------------
bd = pd.read_csv('results-pkps-rd1/rd1_suite_breakdown.csv')
pm = bd[bd['n_paired'] == 0].groupby(['model', 'method'])['abs_err'].mean().unstack()
lim = float(np.nanmax([pm['dkps'].max(), pm['pkps'].max()])) * 1.06
axR.fill_between([0, lim], [0, lim], lim, color=STYLE['pkps']['color'], alpha=0.06, zorder=0)
axR.plot([0, lim], [0, lim], ls='--', color='#94a3b8', lw=1.3, zorder=1)
axR.scatter(pm['pkps'], pm['dkps'], s=30, c=STYLE['pkps']['color'], edgecolors='white',
            lw=0.5, zorder=3)
frac = float((pm['dkps'] > pm['pkps']).mean())
axR.text(0.04 * lim, 0.96 * lim, 'PKPS better', color=STYLE['pkps']['color'], fontsize=9,
         va='top', fontweight='bold')
axR.set_xlim(0, lim); axR.set_ylim(0, lim); axR.set_aspect('equal')
axR.set_xlabel('PKPS MAE (per model)'); axR.set_ylabel('DKPS MAE (per model)')
axR.set_title(f'(b) unpaired, per model: PKPS $<$ DKPS for {frac:.0%}', fontsize=10.5)
axR.grid(alpha=0.25, lw=0.6)

fig.suptitle(f'Unpaired evaluation on the suite (budget $M_{{ij}}{{=}}{BUDGET}$ queries/cell): '
             'PKPS holds where DKPS and IRT collapse', fontsize=11.5, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-rd1/fig_suite_pairing.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-rd1/fig_suite_pairing.png')
