"""QUENCH-style results from qubric representations.

Uses dkps.traces.qubric for the representation (cached graded traces ->
embed_graded -> consensus_center) and reproduces the core query-efficient
benchmarking result: leave-one-LLM-out prediction error of the true Verified
resolve rate vs probe budget m, for geometry, sample score, and the honest
ensemble.

Usage:
  python scripts/quench.py [--judge-dir data/judge/structured-qspec]
                           [--embed-model nomic-ai/nomic-embed-text-v1.5]
Writes figures/quench_core.json.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dkps.traces.qubric import (DEFAULT_SECTIONS, consensus_center,
                                embed_graded)  # noqa: E402


def load_labels(path='data/leaderboard/verified_labels.json'):
    return json.load(open(path))


def model_tag(labels, s):
    m = re.search(r'^\s+model_display:\s*(.*)$',
                  labels[s].get('metadata_yaml', ''), re.M)
    return m.group(1).strip().strip('"\'') if m else None


def load_graded(judge_dir, labels):
    """-> systems, queries, graded[(i,j)] section dicts."""
    systems = sorted(s for s in os.listdir(judge_dir)
                     if 'resolved' in labels.get(s, {}))
    queries = sorted(f[:-5] for f in
                     os.listdir(os.path.join(judge_dir, systems[0])))
    graded = []
    for s in systems:
        for q in queries:
            try:
                d = json.loads(open(os.path.join(judge_dir, s, f'{q}.json')).read())
                if isinstance(d, list) and d:
                    d = d[0]
                if not isinstance(d, dict):
                    d = {}
            except (json.JSONDecodeError, FileNotFoundError):
                d = {}
            graded.append(d)
    return systems, queries, graded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge-dir', default='data/judge/structured-qspec')
    ap.add_argument('--labels', default='data/leaderboard/verified_labels.json')
    ap.add_argument('--embed-model', default='nomic-ai/nomic-embed-text-v1.5')
    ap.add_argument('--ms', default='1,2,3,5,10,20')
    ap.add_argument('--draws', type=int, default=40)
    ap.add_argument('--out', default='figures/quench_core.json')
    args = ap.parse_args()

    labels = load_labels(args.labels)
    systems, queries, graded = load_graded(args.judge_dir, labels)
    M, Q = len(systems), len(queries)
    y = np.array([len(labels[s]['resolved']) / 500 for s in systems])
    B = np.array([[q in set(labels[s]['resolved']) for q in queries]
                  for s in systems], float)
    print(f'{M} systems x {Q} instances from {args.judge_dir}')

    cache = os.path.join('data/judge',
                         f'quench_emb_{os.path.basename(args.judge_dir)}_'
                         f'{args.embed_model.replace("/", "_")}.npz')
    if os.path.exists(cache):
        X = np.load(cache)['X']
    else:
        X = embed_graded(graded, None, args.embed_model)
        np.savez_compressed(cache, X=X)
    inst = np.tile(np.arange(Q), M)
    Xc = consensus_center(X, inst).reshape(M, Q, -1)

    tags = {s: model_tag(labels, s) for s in systems}
    allowed = np.array([[j != i and not (tags[systems[i]]
                        and tags[systems[j]] == tags[systems[i]])
                        for j in range(M)] for i in range(M)])

    def knn_pred(D, i, k=3):
        idx = np.where(allowed[i])[0]
        nn = idx[np.argsort(D[i][idx])[:k]]
        w = 1 / (D[i][nn] + 1e-12)
        return float(np.dot(w, y[nn]) / w.sum())

    def geometry_mae(cols):
        D = squareform(pdist(Xc[:, cols, :].reshape(M, -1)))
        preds = np.array([knn_pred(D, i) for i in range(M)])
        return preds, float(np.abs(preds - y).mean())

    def honest_alpha(i, g_pred, s_pred, cols, alphas=np.linspace(0, 1, 11)):
        """Pick alpha for target i from allowed references only."""
        idx = np.where(allowed[i])[0]
        D = squareform(pdist(Xc[:, cols, :].reshape(M, -1)))
        gs = np.array([knn_pred(D, j) for j in idx])
        ss = B[idx][:, cols].mean(1)
        errs = [np.abs(a * ss + (1 - a) * gs - y[idx]).mean() for a in alphas]
        a = alphas[int(np.argmin(errs))]
        return a * s_pred + (1 - a) * g_pred

    rng = np.random.default_rng(0)
    ms = [int(x) for x in args.ms.split(',')]
    out = {'judge_dir': args.judge_dir, 'embed_model': args.embed_model,
           'n_systems': M, 'n_instances': Q, 'curves': {}}
    for m in ms:
        gs, ss, es = [], [], []
        for _ in range(args.draws):
            cols = rng.choice(Q, m, replace=False)
            gp, gm = geometry_mae(cols)
            sp = B[:, cols].mean(1)
            ep = np.array([honest_alpha(i, gp[i], sp[i], cols)
                           for i in range(M)])
            gs.append(gm)
            ss.append(float(np.abs(sp - y).mean()))
            es.append(float(np.abs(ep - y).mean()))
        out['curves'][m] = {'geometry': float(np.mean(gs)),
                            'sample': float(np.mean(ss)),
                            'ensemble': float(np.mean(es)),
                            'geometry_std': float(np.std(gs)),
                            'ensemble_std': float(np.std(es))}
        c = out['curves'][m]
        print(f"m={m:2d}  geometry {c['geometry']:.4f}  "
              f"sample {c['sample']:.4f}  ensemble {c['ensemble']:.4f}")
    json.dump(out, open(args.out, 'w'), indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
