"""Fig: QUENCH pair (rebuilt, data-driven).

Left:  error vs probe budget m (figures/pipeline_final.json) -- sample score,
       trace-end slice, qubric geometry, fused geometry, honest ensemble.
Right: error vs reference-pool size n (figures/quench_n.json) at m=5 and
       m=20, geometry and ensemble.

Writes figures/fig4_quench.png.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, SURFACE, GRID = '#0A1638', '#f4f7fc', '#DFE6F2'
NAVY, AMBER, SLATE = '#2E5CA6', '#D97706', '#6D93D6'

pf = json.load(open('figures/pipeline_final.json'))
qs = json.load(open('figures/qspec_fig_data_pkps.json'))
qn = json.load(open('figures/quench_n.json'))


def predict_mean_mae(labels='data/leaderboard/verified_labels.json',
                     judge_dir='data/judge/structured-qspec', fallback=0.1343):
    """Baseline: always guess the population mean resolve rate (mean over the
    107 systems of their true full-500 rate, 0.553), scored against each
    system's true rate like every other curve. Constant in m."""
    if not (os.path.exists(labels) and os.path.isdir(judge_dir)):
        return fallback
    L = json.load(open(labels))
    systems = sorted(s for s in os.listdir(judge_dir) if 'resolved' in L.get(s, {}))
    y = np.array([len(L[s]['resolved']) / 500 for s in systems])
    return float(np.abs(y.mean() - y).mean())


# outcome-only baselines (scripts/outcome_baselines.py); optional
ob = json.load(open('figures/outcome_baselines.json')) if os.path.exists('figures/outcome_baselines.json') else None
MEAN_MAE = ob['constant_mean_mae'] if ob else predict_mean_mae()

SHOW_N_PANEL = False      # right panel (error vs reference-pool size) commented out for now
if SHOW_N_PANEL:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
else:
    fig, ax1 = plt.subplots(figsize=(11.0, 5.0))
    ax2 = None
fig.patch.set_facecolor(SURFACE)
fig.suptitle('Query-efficient benchmarking from probe traces '
             '(leave-one-model-out, true labels)', fontsize=12 if SHOW_N_PANEL else 10.5,
             fontweight='bold', color=INK, y=0.98)

m = pf['budgets']
ax1.axhline(MEAN_MAE, color='#8a1c1c', lw=1.4, ls=(0, (2, 2)),
            label=f'guess population mean, 0.553 (MAE {MEAN_MAE:.3f})')
ax1.plot(m, pf['sample'], '--', color=INK, lw=1.6, label='sample score')
if ob:
    RED = '#8a1c1c'
    ax1.plot(ob['m'], ob['count_lookup'], '-.', color=RED, lw=1.6,
             label='average score of references with the same n correct')
    for key, ls, lab in (('irt_2pl_random', '-', '2PL IRT, random probes'),
                         ('irt_2pl_informative', '--', '2PL IRT, pre-computed informative probes'),
                         ('irt_2pl_adaptive', ':', '2PL IRT, adaptive probes')):
        ax1.plot(ob['m'], ob[key], ls, color='#c2410c', lw=1.8, label=lab)
ax1.plot(m, qs['tail|large'], color=SLATE, lw=1.8, label='trace end (8K tok)')
ax1.plot(m, qs['qspec|large'], color='#9db8e8', lw=1.8, label='qubric')
ax1.plot(m, pf['dkps_eh'], color=NAVY, lw=2.0, label='qubric + trace-end')
ax1.plot(m, pf['ens_eh'], color=AMBER, lw=2.2, label='ensemble vs sample')
ax1.set_title('QUENCH(m): error vs probe budget --- '
              f'~{qn.get("n_ref", 106)} reference agents', fontsize=9.5,
              color=INK)
ax1.set_xlabel('probe instances run on the new agent (m)', fontsize=9.5)
ax1.set_ylabel('MAE of predicted resolve rate', fontsize=9.5)
ax1.set_xticks(m)
ax1.set_ylim(0.0, 0.15)
ax1.legend(fontsize=8, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))

if SHOW_N_PANEL:
    ng = qn['n_grid']
    styles = {'5': ('--', 1.6), '20': ('-', 2.0)}
    for b in map(str, qn['budgets']):
        ls, lw = styles[b]
        ax2.plot(ng, qn['curves'][b]['dkps'], ls, color=NAVY, lw=lw,
                 label=f'geometry, m={b}')
        ax2.plot(ng, qn['curves'][b]['ens'], ls, color=AMBER, lw=lw,
                 label=f'ensemble, m={b}')
    for b, y in (('m=5', pf['sample'][3]), ('m=20', pf['sample'][5])):
        ax2.axhline(y, color=INK, lw=0.9, ls=':', alpha=0.5)
        ax2.text(ng[-1], y + 0.003, f'sample score, {b}', fontsize=7,
                 color=INK, ha='right', alpha=0.7)
    ax2.set_title('QUENCH(n): error vs reference pool -- qubric + trace-end',
                  fontsize=9.5, color=INK)
    ax2.set_xlabel('reference agents in the cache (n)', fontsize=9.5)
    ax2.set_xticks(ng)
    ax2.legend(fontsize=8, frameon=False, ncols=2)

for ax in ([ax1, ax2] if SHOW_N_PANEL else [ax1]):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(labelsize=8.5, color=GRID)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig('figures/fig4_quench.png', dpi=200, facecolor=SURFACE)
print('wrote figures/fig4_quench.png')
