"""The only repository-specific adapter; export cached embeddings without APIs.

Usage: python experiments/linear_probes/export_repo.py --out data/linear_probes
The portable runner needs only the exported manifest, CSV, and NPZs.
"""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np


SECTIONS = ('understanding', 'localization', 'reproduction', 'editing', 'verification', 'final_state')
UNKNOWN = {'', 'None', 'null', 'Multiple', 'Undisclosed', '4x Scaled', 'TTS(Bo8)', 'TTS(Bo16)'}


def metadata_scalar(record, field):
    """Read the cache's simple, two-space-indented scalar tags, not general YAML.

    Reject duplicate fields and YAML collections/block strings. The exported
    CSV is the auditable label source; colleagues should supply curated labels.
    """
    values = re.findall(r'^  ' + re.escape(field) + r':[^\S\n]*(.*)$',
                        record.get('metadata_yaml', ''), re.MULTILINE)
    if len(values) > 1:
        raise ValueError(f'Ambiguous metadata field: {field}')
    value = values[0].strip() if values else ''
    if value.startswith(('>', '|', '[', '{')):
        raise ValueError(f'Non-scalar metadata field: {field}')
    if value.startswith('"'):
        value = json.loads(value)
    elif value.startswith("'") and value.endswith("'"):
        value = value[1:-1].replace("''", "'")
    return '' if value in UNKNOWN else value


def legacy_panel(root, labels):
    """Reconstruct and check the orders explicitly used by old unlabeled caches."""
    ref = root / 'data/judge/structured-qspec'
    systems = sorted(p.name for p in ref.iterdir() if p.is_dir() and 'resolved' in labels.get(p.name, {}))
    queries = sorted(p.stem for p in (ref / systems[0]).glob('*.json'))
    # judge_matrix.py additionally intersected these directories.
    dirs = ['gpt-4o-mini', 'gpt-5.4-mini', 'structured-fixed', 'structured-qspec',
            'structured-questions', 'structured-qspec-gpt-5.4-nano',
            'structured-qspec-openai_gpt-oss-120b', 'structured-qspec-openai_gpt-oss-20b',
            'structured-qspec-deepseek_deepseek-chat-v3.1']
    matrix_systems = [s for s in systems if len(list((ref / s).iterdir())) == len(queries)
                      and all((root / 'data/judge' / d / s).is_dir() for d in dirs)]
    if matrix_systems != systems:
        raise ValueError('Legacy matrix and pillars caches do not have the same reconstructed panel')
    return systems, queries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'manifest.json').exists():
        raise ValueError('Export already exists; use a new directory to preserve it')
    labels = json.loads((root / 'data/leaderboard/verified_labels.json').read_text())
    matrix = root / 'data/judge/matrix_emb'
    with np.load(matrix / 'generic-gpt-5.4-mini.openai_text-embedding-3-small.npz') as z:
        systems, queries = z['systems'].tolist(), z['q20'].tolist()
    old_systems, old_queries = legacy_panel(root, labels)
    if systems != old_systems or queries != old_queries:
        raise ValueError('Labeled and reconstructed legacy cache orders disagree')
    m, q = len(systems), len(queries)
    rows, valid, missing = [], [], []
    for i, system in enumerate(systems):
        record = labels[system]
        model = metadata_scalar(record, 'model_display')
        for j, task in enumerate(queries):
            row_id = f'{system}/{task}'
            # Shared cohort: reject traces with wholly missing/unparseable judge descriptions.
            for directory in ('structured-fixed', 'structured-qspec'):
                path = root / 'data/judge' / directory / system / f'{task}.json'
                try:
                    description = json.loads(path.read_text())
                    if isinstance(description, list) and description:
                        description = description[0]
                    if not isinstance(description, dict) or not any(str(description.get(s, '') or '').strip() for s in SECTIONS):
                        raise ValueError('Empty extraction')
                except (OSError, ValueError):
                    missing.append(dict(row_id=row_id, construction=directory))
                    break
            else:
                rows.append(dict(row_id=row_id, system=system, task=task, model=model,
                                 vendor=metadata_scalar(record, 'model_org'),
                                 harness=metadata_scalar(record, 'agent'),
                                 model_group=model or f'unknown:{system}',
                                 outcome=str(int(task in record['resolved']))))
                valid.append(i * q + j)
    if not rows:
        raise ValueError('No complete rows')
    with (args.out / 'metadata.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    row_ids = np.array([r['row_id'] for r in rows])
    manifest = dict(metadata='metadata.csv', representations={}, targets={
        'system': dict(label='system', group='task', description='Exact submission identity on unseen tasks'),
        'model': dict(label='model', group='task', description='Reported model_display on unseen tasks; unknown/mixed excluded'),
        'vendor': dict(label='vendor', group='task', description='Reported model_org on unseen tasks; a coarse lineage proxy'),
        'harness': dict(label='harness', group='task', description='Reported agent tag on unseen tasks; no keyword inference'),
        'task': dict(label='task', group='model_group', description='Task identity on held-out model groups'),
        'outcome': dict(label='outcome', group='model_group', description='Resolved/not resolved on held-out model groups'),
    }, export_notes=dict(systems=m, tasks=q, retained_rows=len(rows), excluded=missing,
                         judge='gpt-5.4-mini', legacy_order='Reconstructed from generating code and checked against labeled OpenAI caches',
                         raw='Concatenated head and tail; not full-trace embeddings'))

    def save(name, x):
        destination = name + '.npz'
        if len(x) != m * q:
            raise ValueError(f'{name}: unexpected row count')
        np.savez_compressed(args.out / destination, X=x[valid].astype(np.float32), row_id=row_ids)
        manifest['representations'][name] = destination
        print(name, x[valid].shape, flush=True)

    for short, tag in [('openai', 'openai_text-embedding-3-small'),
                       ('nomic', 'nomic-ai_nomic-embed-text-v1.5')]:
        for construction in ('generic', 'qubric'):
            path = matrix / f'{construction}-gpt-5.4-mini.{tag}.npz'
            with np.load(path) as z:
                e = z['E']
                if 'systems' in z:
                    si = [z['systems'].tolist().index(s) for s in systems]
                    qi = [z['q20'].tolist().index(t) for t in queries]
                    e = e[si][:, qi]
                if e.shape[:2] != (m, q):
                    raise ValueError(f'{path}: wrong panel dimensions')
                save(f'{short}_{construction}', e.reshape(m * q, *e.shape[2:]))
        if short == 'openai':
            raw = []
            for system in systems:
                for task in queries:
                    with np.load(root / '.dkps_cache_lb' / system / f'{task}.1a49d97e.npz') as z:
                        raw.append(np.stack([z['head'], z['tail']]))
            save('openai_raw', np.array(raw))
        else:
            with np.load(root / 'data/judge/pillars_emb_structured-qspec_nomic-ai_nomic-embed-text-v1.5.npz') as z:
                with np.load(matrix / f'qubric-gpt-5.4-mini.{tag}.npz') as other:
                    np.testing.assert_allclose(z['Xq'], other['E'].reshape(m * q, -1), atol=1e-6)
                save('nomic_raw', np.stack([z['H'], z['T']], axis=1))
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {len(rows)} paired rows; {len(missing)} excluded')


if __name__ == '__main__':
    main()
