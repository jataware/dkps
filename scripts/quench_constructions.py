"""QUENCH curves for every trace representation under one protocol (paired
DKPS: per-task median centering + L2, concatenate the m probe tasks, kNN k=3
over leave-one-LLM-out references; OpenAI text-embedding-3-small throughout).

Representations
  trace-end        last 8K tokens of the rendered trace      (.dkps_cache_lb)
  trace-start      first 8K tokens                           (.dkps_cache_lb)
  trace-end+start  both slices, concatenated
  blob | generic | verdict | qubric   judge descriptions     (data/judge/matrix_emb, see embed_judge_texts.py)
  <judge>+trace-end                    judge sections fused with the tail slice at unit RMS

For each representation: geometry alone, geometry blended (honest per-target
alpha) with the correctness-count lookup from outcome_baselines.py, and geometry
blended the same way with the raw sample score (the paper's ensemble).
The CLI now delegates to nested_quench.py and writes quench_constructions_v2.json.\nThe representation helpers below retain transductive centering for older callers.

Optional stage (--irt-blend): combine a representation with 2PL IRT under
adaptive probe selection, two ways --
  geometry_plus_irt_adaptive : honest alpha blend of the geometry kNN (on the
                               adaptively chosen probes) with the 2PL prediction
  irt_adaptive_trace_prior   : one model -- the geometry prediction sets the
                               prior mean on ability (via a regression fitted on
                               the references), adaptive probes update it.
  irt_random_trace_prior     : the same model with random probes instead.

Usage: python scripts/quench_constructions.py [--reps a,b,...] [--draws 40] [--k 3] [--list]
       python scripts/quench_constructions.py --irt-blend --reps trace-end,qubric+trace-end
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from outcome_baselines import ItemModel, count_lookup, load_panel  # noqa: E402

EMB_TAG = 'openai_text-embedding-3-small'
HEADTAIL_CFG = '1a49d97e'        # sha1('openai/text-embedding-3-small|headtail8000')[:8]
JUDGE_CELLS = {'blob': 'blob-gpt-5.4-mini', 'generic': 'generic-gpt-5.4-mini',
               'verdict': 'verdict-gpt-5.4-mini', 'qubric': 'qubric-gpt-5.4-mini'}


# ------------------------------------------------------ representations ----
def center_by_task(X, task, n_sections):
    """Per (task, section): subtract the median over systems, L2-normalise the
    residual; sections concatenated back. X is (n, n_sections * d)."""
    S = X.reshape(len(X), n_sections, -1)
    out = np.zeros_like(S)
    for q in np.unique(task):
        sel = task == q
        block = S[sel] - np.median(S[sel], axis=0, keepdims=True)
        out[sel] = block / np.maximum(np.linalg.norm(block, axis=-1, keepdims=True), 1e-9)
    return out.reshape(len(X), -1)


def unit_rms(X):
    return X / np.sqrt((X ** 2).mean())


def build_representations(systems, q20):
    """name -> (M*Q, d) centred matrix, rows ordered system-major."""
    task = np.tile(np.arange(len(q20)), len(systems))
    head, tail = [], []
    for s in systems:
        for q in q20:
            z = np.load(f'.dkps_cache_lb/{s}/{q}.{HEADTAIL_CFG}.npz')
            head.append(z['head']); tail.append(z['tail'])
    head, tail = np.array(head, np.float32), np.array(tail, np.float32)
    reps = {'trace-end': center_by_task(tail, task, 1),
            'trace-start': center_by_task(head, task, 1),
            'trace-end+start': center_by_task(np.concatenate([tail, head], 1), task, 2)}
    for name, cell in JUDGE_CELLS.items():
        p = f'data/judge/matrix_emb/{cell}.{EMB_TAG}.npz'
        if not os.path.exists(p):
            print(f'note: {p} missing (run scripts/embed_judge_texts.py); skipping {name}')
            continue
        E = np.load(p)['E']                                   # (M, Q, k, d)
        reps[name] = center_by_task(E.reshape(len(task), -1), task, E.shape[2])
        reps[name + '+trace-end'] = np.concatenate([unit_rms(reps[name]), unit_rms(reps['trace-end'])], 1)
    return reps


# ------------------------------------------------------------ predictors ----
def knn_predict(X, cols, y, allowed, k):
    """Paired kNN: rows = systems on the probe tasks `cols`; each target is
    predicted from its k nearest allowed references (inverse-distance weights)."""
    M = len(allowed)
    Xc = X.reshape(M, -1, X.shape[1])[:, cols].reshape(M, -1)
    sq = (Xc ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * Xc @ Xc.T, 0))
    pred = np.zeros(M)
    for i in range(M):
        idx = np.where(allowed[i])[0]
        nn = idx[np.argsort(D[i, idx])[:k]]
        w = 1 / (D[i, nn] + 1e-9)
        pred[i] = w @ y[nn] / w.sum()
    return pred


def honest_blend(p_a, p_b, y, allowed, alphas=np.linspace(0, 1, 11),
                 reference_predictions=None):
    """Blend using explicitly outer-pool cross-fitted reference predictions.

    reference_predictions=(A, B), with A[i,j] and B[i,j] predictions of
    reference j fitted without outer group i or j's inner validation group.
    Global leave-one-out vectors are insufficient and are rejected.
    """
    if reference_predictions is None:
        raise ValueError('Outer-pool cross-fitted reference_predictions are required')
    ra, rb = map(np.asarray, reference_predictions)
    if ra.shape != allowed.shape or rb.shape != allowed.shape:
        raise ValueError('Reference prediction matrices must match the exclusion matrix')
    out = np.zeros(len(y))
    for i in range(len(y)):
        r = allowed[i]
        if r[i] or not (np.isfinite(ra[i, r]).all() and np.isfinite(rb[i, r]).all()):
            raise ValueError('Invalid outer-pool reference predictions')
        errors = [np.abs(a * ra[i, r] + (1 - a) * rb[i, r] - y[r]).mean()
                  for a in alphas]
        a = alphas[int(np.argmin(errors))]
        out[i] = a * p_a[i] + (1 - a) * p_b[i]
    return out


def main():
    # One maintained evaluator computes all stages with nested reference pools.
    from nested_quench import main as run_nested
    run_nested()


if __name__ == '__main__':
    main()
