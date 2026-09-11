"""The familiar QUENCH figure, rebuilt only from audited nested-pool results.

python scripts/quench_constructions.py
python scripts/fig_quench.py
Use --reference-comparison for the separate 20-vs-500 reference experiment.
Historical PKPS curves with unavailable generating code are not mixed into
this paired-DKPS figure. Original inputs/figures are archived separately.
"""
import json
from pathlib import Path
import sys

if '--reference-comparison' in sys.argv:
    from fig_irt_reference_comparison import main
    main()
    raise SystemExit(0)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, SURFACE, GRID = '#0A1638', '#f4f7fc', '#DFE6F2'


def load_corrected():
    path = Path('figures/quench_constructions_v2.json')
    d = json.loads(path.read_text())
    if d.get('protocol_version') != 'nested-pools-v2':
        raise ValueError('The figure requires audited nested-pool predictions')
    if len(d['records']) != len(set(d['groups'])):
        raise ValueError('Refusing to plot an incomplete group holdout evaluation')
    return d


def style(ax, m):
    ax.set_xlabel('probe instances run on the new agent (m)', fontsize=9.5)
    ax.set_ylabel('MAE of predicted resolve rate', fontsize=9.5)
    ax.set_xticks(m); ax.set_facecolor(SURFACE); ax.grid(color=GRID, lw=.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(labelsize=8.5, color=GRID)


def main():
    d = load_corrected(); m = d['m']
    fig, ax = plt.subplots(figsize=(12, 5.4)); fig.patch.set_facecolor(SURFACE)
    ax.axhline(d['constant_mean_mae'], color='#8a1c1c', lw=1.4, ls=(0, (2, 2)),
               label=f"guess reference-pool mean (MAE {d['constant_mean_mae']:.3f})")
    ax.plot(m, d['sample_score'], '--', color=INK, lw=1.6, label='sample score')
    ax.plot(m, d['count_lookup'], '-.', color='#8a1c1c', lw=1.6,
            label='average score of references with the same n correct')
    for key, ls, label in [('irt_2pl_random', '-', '2PL IRT, random probes'),
                           ('irt_2pl_informative', '--', '2PL IRT, fixed informative probes'),
                           ('irt_2pl_adaptive', ':', '2PL IRT, adaptive probes')]:
        ax.plot(m, d[key], ls, color='#c2410c', lw=1.8, label=label)
    for rep, color, label in [('trace-end', '#6D93D6', 'trace end (8K tokens)'),
                              ('qubric', '#9db8e8', 'qubric'),
                              ('qubric+trace-end', '#2E5CA6', 'qubric + trace-end')]:
        ax.plot(m, d['geometry'][rep], color=color, lw=1.8, label=label)
    ax.plot(m, d['geometry_plus_sample']['qubric+trace-end'], color='#D97706', lw=2.2,
            label='qubric + trace-end blended with sample score (nested tuning)')
    for rep, color, label in [('qubric+trace-end', '#0f5132', 'qubric + trace-end'),
                              ('generic', '#2a9d8f', 'common rubric')]:
        ax.plot(m, d['irt_adaptive_trace_prior'][rep], color=color, lw=2,
                label=f'2PL adaptive, {label} geometry as prior')
    ax.plot(m, d['irt_random_trace_prior']['generic'], '--', color='#2a9d8f', lw=1.8,
            label='2PL random, common rubric geometry as prior')
    nref = [len(r['train']) for r in d['records'] for _ in r['test']]
    fig.suptitle('Query-efficient benchmarking from probe traces — corrected evaluation',
                 fontsize=12, color=INK, fontweight='bold', y=.98)
    ax.set_title(f"Paired DKPS; {len(d['systems'])} systems × q20; "
                 f"{min(nref)}–{max(nref)} references; outer model-group exclusion", fontsize=9.5)
    style(ax, m); ax.set_ylim(0, .15)
    ax.legend(fontsize=8, frameon=False, loc='center left', bbox_to_anchor=(1.01, .5))
    fig.text(.02, .025, 'Centering and calibration use permitted references only. '
             f"Sample-score MAE at m=1 is {d['sample_score'][0]:.3f} (above the shown range).",
             fontsize=8, color=INK)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig('figures/fig4_quench.png', dpi=200, facecolor=SURFACE)
    fig.savefig('figures/fig4_quench.pdf', facecolor=SURFACE)
    plt.close(fig)
    print('wrote figures/fig4_quench.png and .pdf')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, rep, policy in zip(axes, ['generic', 'qubric+trace-end', 'generic'],
                               ['random', 'random', 'adaptive']):
        key = f'{policy}:prior:{rep}'
        a = d['audit'][key]
        ax.plot(m, a['old_reproduction_mae'], '--', color='#8a1c1c', label='old dependency reproduced')
        if policy == 'random':
            ax.plot(m, d['all_curves'][f'score_mask_only:prior:{rep}'], ':', color='#D97706',
                    label='score masks fixed; global centering retained')
        ax.plot(m, a['corrected_mae'], '-', color='#2a9d8f', label='nested pools + reference-only centering')
        ax.set_title(f"{'Common rubric' if rep == 'generic' else 'Qubric + trace-end'} prior, {policy}", fontsize=10)
        style(ax, m); ax.set_ylim(bottom=0); ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle('Leakage audit: matched systems, probe draws and model settings', fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig('figures/quench_leakage_audit.png', dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print('wrote figures/quench_leakage_audit.png')


if __name__ == '__main__': main()
