"""Pillar (property) metrics for trace representations, via dkps.traces.qubric.

Computes, for raw head/tail and qubric representations, the six properties of
the write-up: Task fidelity, Behavioral fidelity, Identity (trace + agg),
Model-family invariance, Harness invariance -- on the leaderboard corpus --
plus Stability on the replicate small cohort (delegated to
figures/stability_column.json if present).

Tag heuristics (documented, may differ marginally from hand-curated tags):
  vendor  from model_display (first known-vendor keyword)
  harness from submission name (known scaffold keyword list)

Usage:
  python scripts/pillars.py [--judge-dir data/judge/structured-qspec]
                            [--embed-model nomic-ai/nomic-embed-text-v1.5]
Writes figures/pillars.json (consumed by radar/heatmap figure scripts).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dkps.traces.qubric import consensus_center, embed_graded  # noqa: E402
from quench import load_graded, load_labels, model_tag  # noqa: E402

VENDORS = ('claude', 'anthropic', 'gpt', 'openai', 'o1', 'o3', 'o4', 'gemini',
           'qwen', 'deepseek', 'glm', 'llama', 'mistral', 'devstral', 'kimi',
           'moonshot', 'nova', 'grok', 'doubao', 'seed', 'skywork', 'lingma')
SCAFFOLDS = ('sweagent', 'swe-agent', 'openhands', 'agentless', 'autocoderover',
             'moatless', 'composio', 'marscode', 'lingma', 'gru', 'masai',
             'codeact', 'tools', 'epam', 'solver', 'aime', 'blackbox',
             'devlo', 'emergent', 'nemotron', 'trae', 'refact', 'zencoder')


def vendor_tag(labels, s):
    md = (model_tag(labels, s) or '').lower()
    for v in VENDORS:
        if v in md:
            return v
    return None


def harness_tag(s):
    low = s.lower()
    for h in SCAFFOLDS:
        if h in low:
            return h
    return None


def nn_hit_rate(X, group, mask_same_instance=None, inst=None):
    """P(nearest neighbor shares `group`), excluding self; optionally
    restrict candidates to OTHER instances (cross-task) or same instance."""
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    if mask_same_instance is not None and inst is not None:
        same = inst[:, None] == inst[None, :]
        D[same if mask_same_instance == 'exclude' else ~same] = np.inf
    nn = D.argmin(1)
    ok = np.array(group) != None  # noqa: E711
    g = np.asarray(group)
    return float((g[nn[ok]] == g[ok]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge-dir', default='data/judge/structured-qspec')
    ap.add_argument('--labels', default='data/leaderboard/verified_labels.json')
    ap.add_argument('--embed-model', default='nomic-ai/nomic-embed-text-v1.5')
    ap.add_argument('--trace-texts', default='data/judge/trace_texts')
    ap.add_argument('--out', default='figures/pillars.json')
    args = ap.parse_args()

    labels = load_labels(args.labels)
    systems, queries, graded = load_graded(args.judge_dir, labels)
    M, Q = len(systems), len(queries)
    B = np.array([[q in set(labels[s]['resolved']) for q in queries]
                  for s in systems], float)
    print(f'{M} systems x {Q} instances')

    # ---- representations ---------------------------------------------------
    emb_tag = args.embed_model.replace('/', '_')
    cache = f'data/judge/pillars_emb_{os.path.basename(args.judge_dir)}_{emb_tag}.npz'
    if os.path.exists(cache):
        z = np.load(cache)
        Xq, Hr, Tr = z['Xq'], z['H'], z['T']
    else:
        Xq = embed_graded(graded, None, args.embed_model)
        from dkps.traces.embedder import make_sentence_transformer_embed_fn
        embed = make_sentence_transformer_embed_fn(model_name=args.embed_model)
        heads, tails = [], []
        for s in systems:
            for q in queries:
                t = open(os.path.join(args.trace_texts, s, f'{q}.txt')).read()
                heads.append(t[:32_000])
                tails.append(t[-32_000:])
        Hr = np.asarray(embed(heads), dtype=np.float32)
        Tr = np.asarray(embed(tails), dtype=np.float32)
        np.savez_compressed(cache, Xq=Xq, H=Hr, T=Tr)
    inst = np.tile(np.arange(Q), M)
    sysid = np.repeat(np.arange(M), Q)
    reps = {
        'raw': np.concatenate([Hr, Tr], axis=1),
        'qubric': consensus_center(Xq, inst),
    }

    vend = [vendor_tag(labels, systems[i]) for i in sysid]
    harn = [harness_tag(systems[i]) for i in sysid]
    outcome = B[sysid, inst]

    out = {'embed_model': args.embed_model, 'judge_dir': args.judge_dir,
           'chance': {'task': 1 / Q,
                      'identity': 1 / M,
                      'family': None, 'harness': None},
           'reps': {}}
    from scipy.spatial.distance import pdist, squareform
    for name, X in reps.items():
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        # task: NN restricted to OTHER systems; hit = same instance
        D = squareform(pdist(Xn))
        np.fill_diagonal(D, np.inf)
        D_task = D.copy()
        D_task[sysid[:, None] == sysid[None, :]] = np.inf
        task = float((inst[D_task.argmin(1)] == inst).mean())
        # identity (trace): NN across OTHER instances; hit = same system
        D_id = D.copy()
        D_id[inst[:, None] == inst[None, :]] = np.inf
        nn_id = D_id.argmin(1)
        ident = float((sysid[nn_id] == sysid).mean())
        # family/harness: NN restricted to same instance (within-task)
        D_who = D.copy()
        D_who[inst[:, None] != inst[None, :]] = np.inf
        nn = D_who.argmin(1)
        v = np.asarray(vend); h = np.asarray(harn)
        vok = v != None  # noqa: E711
        hok = h != None  # noqa: E711
        fam = float((v[nn[vok]] == v[vok]).mean())
        har = float((h[nn[hok]] == h[hok]).mean())
        # behavior: within-instance same-outcome AUC
        aucs = []
        for qi in range(Q):
            sel = np.where(inst == qi)[0]
            o = outcome[sel]
            if o.min() == o.max():
                continue
            Dq = D[np.ix_(sel, sel)]
            same = o[:, None] == o[None, :]
            iu = np.triu_indices(len(sel), 1)
            ds, dd = Dq[iu][same[iu]], Dq[iu][~same[iu]]
            aucs.append(float((ds[:, None] < dd[None, :]).mean()))
        behavior = float(np.mean(aucs)) - 0.5
        # identity aggregated: split-half retrieval over instances
        rng = np.random.default_rng(0)
        hits = 0
        for _ in range(20):
            perm = rng.permutation(Q)
            a, b = perm[:Q // 2], perm[Q // 2:]
            Xa = Xn.reshape(M, Q, -1)[:, a].mean(1)
            Xb = Xn.reshape(M, Q, -1)[:, b].mean(1)
            Dab = ((Xa[:, None] - Xb[None]) ** 2).sum(-1)
            hits += (Dab.argmin(1) == np.arange(M)).mean()
        ident_agg = float(hits / 20)
        out['reps'][name] = dict(task=task, behavior=behavior,
                                 identity_trace=ident, identity_agg=ident_agg,
                                 family=fam, harness=har)
        print(f"{name:8s} task {task:.3f}  behavior {behavior:+.3f}  "
              f"id {ident:.3f}/{ident_agg:.3f}  fam {fam:.3f}  harness {har:.3f}")

    # chance for family/harness: base rates of sharing the tag within instance
    v = np.asarray(vend); h = np.asarray(harn)
    def base(gr):
        g = gr[gr != None]  # noqa: E711
        _, c = np.unique(g, return_counts=True)
        return float(((c / c.sum()) ** 2).sum())
    out['chance']['family'] = base(v)
    out['chance']['harness'] = base(h)
    if os.path.exists('figures/stability_column.json'):
        out['stability'] = json.load(open('figures/stability_column.json'))
    json.dump(out, open(args.out, 'w'), indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
