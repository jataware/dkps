"""Fig: sensitivity radars across seven embedders (rebuilt, data-driven).

Reads figures/radar_all_data_v2.json + figures/family_metric.json. Spokes are
oriented desirable-OUTSIDE: content spokes (Task, Behavior) plot normalized
fidelity; authorship spokes (Identity, Model Family, Harness) plot normalized
invariance, so chance sits at the rim for them (dashed ring shows chance on
every spoke). Identity uses the aggregated test to match the heatmap.

Writes figures/radar_all.png.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK, SURFACE, GRID = '#0A1638', '#f4f7fc', '#DFE6F2'
NAVY, AMBER = '#2E5CA6', '#D97706'

d = json.load(open('figures/radar_all_data_v2.json'))
fam = json.load(open('figures/family_metric.json'))
CH = d['chance']
CH['family'] = fam['chance_family']

SPOKES = [('behavior', 'Behavior', 'keep'), ('task', 'Task', 'keep'),
          ('scaf', 'Harness', 'inv'), ('family', 'Model\nFamily', 'inv'),
          ('provagg', 'Identity', 'inv')]
BEH_MAX = 0.05          # behavior deltas are small; fixed scale across panels


def coords(vals):
    out = []
    for key, _, kind in SPOKES:
        v, ch = vals[key], CH[key]
        if key == 'behavior':
            r = np.clip(v / BEH_MAX, 0, 1)
        elif kind == 'keep':
            r = np.clip((v - ch) / (1 - ch), 0, 1)
        else:
            r = np.clip(1 - (v - ch) / (1 - ch), 0, 1)
        out.append(r)
    return out


def chance_coords():
    return [0 if kind == 'keep' else 1 for _, _, kind in SPOKES]


panels = list(d['panels'].items())
for name, p in panels:
    p['raw']['family'] = fam['panels'][name]['raw']
    p['qubric']['family'] = fam['panels'][name]['qubric']

ang = np.linspace(0, 2 * np.pi, len(SPOKES), endpoint=False)
fig = plt.figure(figsize=(16.0, 8.4))
fig.patch.set_facecolor(SURFACE)
pos = [(0.045 + i * 0.245, 0.50) for i in range(4)] + \
      [(0.17 + i * 0.245, 0.045) for i in range(3)]

for (name, p), (x0, y0) in zip(panels, pos):
    ax = fig.add_axes([x0, y0, 0.165, 0.36], projection='polar')
    ax.set_facecolor(SURFACE)
    ax.set_rorigin(-0.35)
    ax.set_ylim(0, 1.0)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    for series, color in (('raw', NAVY), ('qubric', AMBER)):
        r = coords(p[series])
        aa = np.concatenate([ang, ang[:1]])
        rr = np.array(r + r[:1])
        ax.plot(aa, rr, color=color, lw=2)
        ax.fill(aa, rr, color=color, alpha=0.18)
    cc = chance_coords()
    ax.plot(np.concatenate([ang, ang[:1]]), np.array(cc + cc[:1]), '--',
            color=INK, lw=1.4, alpha=0.55)
    ax.set_xticks(ang)
    ax.set_xticklabels([lab for _, lab, _ in SPOKES], fontsize=8, color=INK)
    ax.set_yticks([])
    ax.grid(color=GRID, lw=0.8)
    ax.spines['polar'].set_color(GRID)
    ax.set_title(name, fontsize=10, color=INK, pad=14)

fig.text(0.5, 0.965,
         'Sensitivity profiles across embedding models --- rim = ideal: '
         'faithful to content (Task, Behavior), invariant to authorship '
         '(Identity, Model Family, Harness)',
         ha='center', fontsize=11.5, fontweight='bold', color=INK)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=NAVY, lw=2, label='raw trace embedding'),
                    Line2D([], [], color=AMBER, lw=2, label='qubric'),
                    Line2D([], [], color=INK, lw=1.4, ls='--', alpha=0.55,
                           label='chance-level')],
           loc='center', bbox_to_anchor=(0.92, 0.22), frameon=False,
           fontsize=10)
fig.savefig('figures/radar_all.png', dpi=200, facecolor=SURFACE)
print('wrote figures/radar_all.png')
