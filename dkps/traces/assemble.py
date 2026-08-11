"""Assemble per-trace vectors into the rectangular arrays DKPS consumes, and
helpers for the distance-combination mode.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from graspologic.embed import ClassicalMDS
from scipy.spatial.distance import pdist, squareform


def build_dkps_input(traces, embeddings, missing='impute_model_mean',
                     queries=None, models=None):
    """Arrange per-trace embedding rows into {model: (n_queries, n_replicates, d)}.

    embeddings: (len(traces), d) array, rows aligned with traces.
    missing: how to handle (model, query) cells with fewer replicates than the max:
      - 'impute_model_mean': fill the slot with the mean of that cell's available
        replicates (equivalent to averaging available replicates under DKPS's
        default mean-collapse, so it is unbiased there)
      - 'drop_replicate': use the minimum replicate count across all cells
      - 'error': raise

    Returns ({model: array}, report); report lists imputed/dropped slots.
    """
    embeddings = np.asarray(embeddings)
    by_key = {}
    for tr, e in zip(traces, embeddings):
        by_key[(tr.model_id, tr.query_id, tr.replicate)] = e

    if models is None:
        models = sorted({tr.model_id for tr in traces})
    if queries is None:
        queries = sorted({tr.query_id for tr in traces})

    cell_reps = defaultdict(list)
    for m, q, r in by_key:
        cell_reps[(m, q)].append(r)
    for m in models:
        for q in queries:
            if not cell_reps[(m, q)]:
                raise ValueError(f'model {m} has no replicates for query {q}')
            cell_reps[(m, q)].sort()

    counts = [len(v) for v in cell_reps.values()]
    n_reps = min(counts) if missing == 'drop_replicate' else max(counts)

    d = embeddings.shape[1]
    report = {'imputed': [], 'dropped': [], 'n_replicates': n_reps}
    out = {}
    for m in models:
        arr = np.zeros((len(queries), n_reps, d))
        for qi, q in enumerate(queries):
            have = cell_reps[(m, q)]
            for ri in range(n_reps):
                if ri < len(have):
                    arr[qi, ri] = by_key[(m, q, have[ri])]
                elif missing == 'impute_model_mean':
                    arr[qi, ri] = np.mean([by_key[(m, q, r)] for r in have], axis=0)
                    report['imputed'].append((m, q, ri))
                else:
                    raise ValueError(f'missing replicate {ri} for ({m}, {q})')
            report['dropped'].extend((m, q, r) for r in have[n_reps:])
        out[m] = arr
    return out, report


def attach_labels(traces, labels):
    """Join outcome labels (e.g. SWE-bench resolved) onto traces.

    labels: {(model_id, query_id, replicate): value}. Stored in
    trace.metadata['resolved']. Returns count of matched traces.
    """
    n = 0
    for tr in traces:
        if tr.key in labels:
            tr.metadata['resolved'] = labels[tr.key]
            n += 1
    return n


def model_distance_matrix(X, metric='euclidean'):
    """The distance computation DataKernelPerspectiveSpace uses internally:
    mean over replicates, flatten, pdist / sqrt(n_queries).

    X: {model: (n_queries, n_replicates, d)}. Returns (models, dist_matrix).
    """
    models = list(X.keys())
    collapsed = np.array([np.mean(np.asarray(X[m]), axis=1).ravel() for m in models])
    n_queries = next(iter(X.values())).shape[0]
    dist = squareform(pdist(collapsed, metric=metric)) / np.sqrt(n_queries)
    return models, dist


def combine_channel_distances(channel_inputs, weights=None, normalize='median',
                              n_components_cmds=None, n_elbows_cmds=2):
    """Distance-combination mode: per-channel model distance matrices, each
    normalized then combined as a weighted sum, embedded with CMDS.

    channel_inputs: {channel_name: {model: (n_q, n_r, d_ch)}}.
    normalize: 'median' (divide each channel's distances by their median
    off-diagonal value, making weights interpretable) or 'none'.

    Returns ({model: coords}, combined_dist, {channel: dist}).
    """
    weights = weights or {}
    per_channel = {}
    models = None
    for name, X in channel_inputs.items():
        ms, dist = model_distance_matrix(X)
        if models is None:
            models = ms
        elif ms != models:
            raise ValueError('channel inputs have inconsistent model sets')
        if normalize == 'median':
            off = dist[np.triu_indices_from(dist, k=1)]
            med = np.median(off)
            if med > 0:
                dist = dist / med
        per_channel[name] = dist

    total_w = sum(weights.get(name, 1.0) for name in per_channel)
    combined = sum(weights.get(name, 1.0) * dist for name, dist in per_channel.items())
    combined = combined / total_w
    coords = ClassicalMDS(n_components_cmds, n_elbows_cmds).fit_transform(combined)
    return dict(zip(models, coords)), combined, per_channel
