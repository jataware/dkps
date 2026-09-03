"""Fig: QUENCH by trace representation (does the judge prompt matter?).

Reads figures/quench_constructions.json (scripts/quench_constructions.py) and
figures/outcome_baselines.json (scripts/outcome_baselines.py). Three panels,
shared y-axis:
  1. single representations, geometry alone
  2. each judge construction fused with trace-end (trace-end alone for reference)
  3. the panel-2 representations blended with the correctness-count lookup,
     with the count lookup and 2PL IRT (random probes) as trace-free references
  4. the panel-2 representations blended with the raw sample score (the
     paper's ensemble), with the sample score alone for reference
Writes figures/quench_constructions.png.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, SURFACE, GRID = '#0A1638', '#f4f7fc', '#DFE6F2'
RED = '#8a1c1c'
STYLE = {   # representation -> (color, linestyle)
    'trace-end': ('#2E5CA6', '-'), 'trace-start': ('#9db8e8', '-'), 'trace-end+start': ('#6D93D6', '-'),
    'blob': ('#7a7a7a', '-'), 'generic': ('#2a9d8f', '-'), 'verdict': ('#8a5cf6', '-'), 'qubric': ('#D97706', '-'),
    'blob+trace-end': ('#7a7a7a', '--'), 'generic+trace-end': ('#2a9d8f', '--'),
    'verdict+trace-end': ('#8a5cf6', '--'), 'qubric+trace-end': ('#D97706', '--'),
}
LABEL = {'generic': 'common rubric', 'generic+trace-end': 'common rubric + trace-end'}
PANELS = [
    ('geometry', ['trace-end', 'trace-start', 'trace-end+start', 'blob', 'generic', 'verdict', 'qubric'],
     'representation alone (paired DKPS, kNN k=3)'),
    ('geometry', ['trace-end', 'blob+trace-end', 'generic+trace-end', 'verdict+trace-end', 'qubric+trace-end'],
     'judge construction fused with trace-end'),
    ('geometry_plus_count', ['trace-end', 'blob+trace-end', 'generic+trace-end', 'verdict+trace-end', 'qubric+trace-end'],
     'fused, then blended with "same n correct" lookup'),
    ('geometry_plus_sample', ['trace-end', 'blob+trace-end', 'generic+trace-end', 'verdict+trace-end', 'qubric+trace-end'],
     'fused, then blended with raw sample score'),
]

d = json.load(open('figures/quench_constructions.json'))
ob = json.load(open('figures/outcome_baselines.json')) if os.path.exists('figures/outcome_baselines.json') else None

fig, axes = plt.subplots(1, 4, figsize=(21.0, 5.0), sharey=True)
fig.patch.set_facecolor(SURFACE)
for ax, (key, names, title) in zip(axes, PANELS):
    for name in names:
        if name not in d[key]:
            continue
        c, ls = STYLE[name]
        ax.plot(d['m'], d[key][name], ls, color=c, lw=2.2 if name == 'qubric' or name.startswith('qubric') else 1.8,
                label=LABEL.get(name, name))
    if ob:
        ax.axhline(ob['constant_mean_mae'], color=RED, lw=1.2, ls=(0, (2, 2)), label='guess population mean')
        if key == 'geometry_plus_count':
            ax.plot(ob['m'], ob['count_lookup'], '-.', color=RED, lw=1.4, label='same n correct lookup alone')
            ax.plot(ob['m'], ob['irt_2pl_random'], ':', color=RED, lw=1.6, label='2PL IRT, random probes')
        if key == 'geometry_plus_sample' and 'sample_score' in d:
            ax.plot(d['m'], d['sample_score'], '--', color=INK, lw=1.4, label='sample score alone')
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel('probe instances (m)', fontsize=9.5)
    ax.set_xticks(d['m']); ax.set_ylim(0.0, 0.15)
    ax.set_facecolor(SURFACE); ax.grid(color=GRID, lw=0.8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(labelsize=8.5, color=GRID)
    ax.legend(fontsize=7.5, frameon=False, loc='lower left')
axes[0].set_ylabel('MAE of predicted resolve rate', fontsize=9.5)
fig.suptitle(f"QUENCH by trace representation ({d.get('embedder', '')}, {d.get('protocol', '')}, 107 systems x q20)",
             fontsize=11, fontweight='bold', color=INK, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig('figures/quench_constructions.png', dpi=200, facecolor=SURFACE)
print('wrote figures/quench_constructions.png')
