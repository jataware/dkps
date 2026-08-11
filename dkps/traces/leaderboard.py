"""Loader for SWE-bench leaderboard submissions (github.com/swe-bench/experiments).

Trajectory formats vary per scaffold (SWE-agent .traj dicts, OpenHands/TRAE
chat lists, sonar block lists, Amazon Q plain text, ...). This loader is
format-agnostic: it renders any trajectory to ordered plain text and wraps it
in a single-step Trace, which is exactly what the whole-trace channel needs --
our results show that representation matches structured channels for score
prediction. Structured per-scaffold parsers can be added later where cheap.

Expected layout (as synced from s3://swe-bench-submissions):
    root/<submission>/trajs/<instance_id>.<ext>
    root/<submission>/all_preds.jsonl      # patches: {instance_id, model_patch}
plus the labels file harvested from the git repo (results/results.json per
submission) as {submission: {resolved: [...], metadata_yaml: ...}}.
"""
from __future__ import annotations

import json
import os
import re
from glob import glob

from .schema import Step, Trace

# dict keys rendered first, in this order, when present; gives chat-style and
# step-style formats a stable, readable rendering
_PRIORITY_KEYS = ('role', 'thought', 'content', 'text', 'action', 'command',
                  'observation', 'output', 'blocks')
_SKIP_KEYS = {'token_usage_prompt', 'token_usage_completion', 'token_usage_total',
              'llm_exec_time', 'step_idx', 'done', 'additional_kwargs', 'info',
              'timestamp', 'id'}


def render_text(obj, depth=0):
    """Render an arbitrary trajectory object to ordered plain text."""
    if obj is None:
        return ''
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, list):
        return '\n'.join(t for t in (render_text(x, depth + 1) for x in obj) if t)
    if isinstance(obj, dict):
        parts = []
        seen = set()
        for k in _PRIORITY_KEYS:
            if k in obj:
                seen.add(k)
                t = render_text(obj[k], depth + 1)
                if t:
                    parts.append(f'{k}: {t}' if k in ('role', 'action') else t)
        for k, v in obj.items():
            if k not in seen and k not in _SKIP_KEYS:
                t = render_text(v, depth + 1)
                if t:
                    parts.append(t)
        return '\n'.join(parts)
    return ''


def _extract_query_id(basename):
    m = re.search(r'([A-Za-z][A-Za-z0-9.\-]*__[A-Za-z0-9_.\-]+?-\d+)', basename)
    return m.group(1) if m else re.sub(
        r'\.(traj|json|txt|md|log|jsonl|yaml)$', '', basename)


def _render_file(path):
    raw = open(path, errors='replace').read()
    try:
        return render_text(json.loads(raw))
    except (json.JSONDecodeError, ValueError):
        return raw


def load_leaderboard_trajectory(path, model_id='', query_id=None):
    """One trajectory (file, or directory of files) -> single-step Trace with
    the rendered text. Some scaffolds store trajs/<instance>/<several files>;
    those are rendered in filename order and concatenated."""
    if query_id is None:
        query_id = _extract_query_id(os.path.basename(os.path.normpath(path)))
    if os.path.isdir(path):
        parts = []
        for f in sorted(glob(os.path.join(path, '**', '*'), recursive=True)):
            if os.path.isfile(f):
                parts.append(f'### {os.path.relpath(f, path)}\n{_render_file(f)}')
        text = '\n'.join(parts)
    else:
        text = _render_file(path)
    tr = Trace(model_id=model_id, query_id=query_id, replicate=0,
               steps=[Step(index=0, assistant_text=text)])
    tr.metadata['path'] = path
    tr.metadata['rendered_chars'] = len(text)
    return tr


def load_leaderboard_submission(sub_dir, model_id=None, labels=None):
    """All trajectories of one submission -> list[Trace].

    Attaches patches from all_preds.jsonl as final_output, and resolved labels
    (from the harvested git results) into metadata['resolved'].
    """
    if model_id is None:
        model_id = os.path.basename(os.path.normpath(sub_dir))
    patches = {}
    preds_path = os.path.join(sub_dir, 'all_preds.jsonl')
    if os.path.exists(preds_path):
        for line in open(preds_path, errors='replace'):
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
                patches[p.get('instance_id')] = p.get('model_patch') or ''
            except json.JSONDecodeError:
                continue
    resolved = set(labels.get(model_id, {}).get('resolved', [])) if labels else None

    traces = []
    for path in sorted(glob(os.path.join(sub_dir, 'trajs', '*'))):
        tr = load_leaderboard_trajectory(path, model_id=model_id)
        tr.final_output = patches.get(tr.query_id)
        if resolved is not None:
            tr.metadata['resolved'] = tr.query_id in resolved
        traces.append(tr)
    return traces


def load_leaderboard_corpus(root, labels_path=None, min_trajs=400):
    """All submissions under root -> list[Trace]. Skips submissions with fewer
    than min_trajs trajectory files (partial/broken syncs)."""
    labels = json.load(open(labels_path)) if labels_path else None
    traces = []
    skipped = []
    for sub in sorted(os.listdir(root)):
        sub_dir = os.path.join(root, sub)
        n = len(glob(os.path.join(sub_dir, 'trajs', '*')))
        if n < min_trajs:
            skipped.append((sub, n))
            continue
        traces.extend(load_leaderboard_submission(sub_dir, labels=labels))
    if skipped:
        print(f'load_leaderboard_corpus: skipped {len(skipped)} submissions '
              f'with < {min_trajs} trajs: {[s for s, _ in skipped[:5]]}...')
    return traces
