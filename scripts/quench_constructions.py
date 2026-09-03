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
Writes / merges into figures/quench_constructions.json (rows keyed by name).

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


def honest_blend(p_a, p_b, y, allowed, alphas=np.linspace(0, 1, 11)):
    """alpha * p_a + (1 - alpha) * p_b, alpha chosen per target to minimise the
    error over that target's allowed references only."""
    out = np.zeros(len(y))
    for i in range(len(y)):
        r = allowed[i]
        errs = [np.abs(a * p_a[r] + (1 - a) * p_b[r] - y[r]).mean() for a in alphas]
        a = alphas[int(np.argmin(errs))]
        out[i] = a * p_a[i] + (1 - a) * p_b[i]
    return out


# ---------------------------------------------- geometry x adaptive IRT ----
def irt_adaptive_stage(reps, ms, y, B, allowed, args):
    """For each representation: combine its geometry with 2PL IRT under
    adaptive probe selection (see module docstring). Merges two rows per
    representation into the output JSON."""
    M, Q = B.shape
    models = [ItemModel(B[allowed[i]], y[allowed[i]], args.ridge_a) for i in range(M)]
    paths = [models[i].adaptive_path(B[i])[0] for i in range(M)]          # per-target item order
    out = json.load(open(args.out)) if os.path.exists(args.out) else {'m': ms}
    out.setdefault('geometry_plus_irt_adaptive', {}); out.setdefault('irt_adaptive_trace_prior', {}); out.setdefault('irt_random_trace_prior', {})
    rng = np.random.default_rng(0)
    rand_draws = {m: ([np.arange(Q)] if m == Q else [np.array([q]) for q in range(Q)] if m == 1
                      else [rng.choice(Q, m, replace=False) for _ in range(args.draws)]) for m in ms}

    def trace_prior_predict(mdl, i, g, cols):
        """Prior mean/width from the reference regression of ability on the
        geometry prediction g; posterior after the target's outcomes on cols."""
        r = allowed[i]
        slope, intercept = np.polyfit(g[r], mdl.theta, 1)
        resid_sd = float(np.std(mdl.theta - (intercept + slope * g[r])))
        return mdl.predict(cols, B[i, cols], mu=intercept + slope * g[i], sd=resid_sd)
    out['irt_adaptive'] = out.get('irt_adaptive') or [float(np.mean([abs(models[i].predict(paths[i][:m], B[i, paths[i][:m]]) - y[i]) for i in range(M)])) for m in ms]
    print(f"{'representation':18s} " + ' '.join(f'{"m=%d" % m:>7s}' for m in ms) + '   (adaptive blend | adaptive trace-prior | random trace-prior)')
    for name, X in reps.items():
        e_blend = {m: [] for m in ms}; e_prior = {m: [] for m in ms}
        for i in range(M):
            mdl, r = models[i], allowed[i]
            for m in ms:
                cols = np.array(paths[i][:m])
                g = knn_predict(X, cols, y, allowed, args.k)                 # geometry for everyone on target i's probes
                p_irt = np.array([models[j].predict(cols, B[j, cols]) for j in range(M)])
                e_blend[m].append(abs(honest_blend(p_irt, g, y, allowed)[i] - y[i]))
                e_prior[m].append(abs(trace_prior_predict(mdl, i, g, cols) - y[i]))
        e_rand = {m: [] for m in ms}                                           # random probes, trace prior
        for m in ms:
            for cols in rand_draws[m]:
                g = knn_predict(X, cols, y, allowed, args.k)
                e_rand[m].append(np.mean([abs(trace_prior_predict(models[i], i, g, cols) - y[i]) for i in range(M)]))
        out['geometry_plus_irt_adaptive'][name] = [float(np.mean(e_blend[m])) for m in ms]
        out['irt_adaptive_trace_prior'][name] = [float(np.mean(e_prior[m])) for m in ms]
        out['irt_random_trace_prior'][name] = [float(np.mean(e_rand[m])) for m in ms]
        json.dump(out, open(args.out, 'w'), indent=1)
        print(f'{name:18s} ' + ' '.join(f"{a:7.4f}" for a in out['geometry_plus_irt_adaptive'][name]) + '  |  '
              + ' '.join(f"{a:7.4f}" for a in out['irt_adaptive_trace_prior'][name]) + '  |  '
              + ' '.join(f"{a:7.4f}" for a in out['irt_random_trace_prior'][name]))
    print('wrote', args.out)


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reps', default='all')
    ap.add_argument('--ms', default='1,2,3,5,10,20')
    ap.add_argument('--draws', type=int, default=40)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--out', default='figures/quench_constructions.json')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--irt-blend', action='store_true', help='run the IRT-adaptive combination stage instead of the sweep')
    ap.add_argument('--ridge-a', type=float, default=10.0)
    args = ap.parse_args()

    systems, q20, y, B, allowed = load_panel()
    M, Q = B.shape
    reps = build_representations(systems, q20)
    if args.list:
        print('\n'.join(reps)); return
    if args.reps != 'all':
        want = [r.strip() for r in args.reps.split(',')]
        missing = [r for r in want if r not in reps]
        if missing:
            sys.exit(f'unknown representations {missing}; use --list')
        reps = {r: reps[r] for r in want}

    ms = [int(x) for x in args.ms.split(',')]
    if args.irt_blend:
        irt_adaptive_stage(reps, ms, y, B, allowed, args)
        return
    rng = np.random.default_rng(0)
    draws = {m: ([np.arange(Q)] if m == Q else [np.array([q]) for q in range(Q)] if m == 1
                 else [rng.choice(Q, m, replace=False) for _ in range(args.draws)]) for m in ms}
    count_preds = {m: [np.array([count_lookup(B, y, allowed, i, cols) for i in range(M)])
                       for cols in draws[m]] for m in ms}

    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    out['sample_score'] = [float(np.mean([np.abs(B[:, cols].mean(1) - y).mean() for cols in draws[m]])) for m in ms]
    out.update({'m': ms, 'k': args.k, 'draws': args.draws, 'embedder': EMB_TAG, 'protocol': 'paired DKPS kNN, leave-one-LLM-out'})
    for key in ('geometry', 'geometry_plus_count', 'geometry_plus_sample'):
        out.setdefault(key, {})
    print(f"{'representation':18s} " + ' '.join(f'{"m=%d" % m:>7s}' for m in ms) + '   | + count lookup: ' + ' '.join(f'{"m=%d" % m:>6s}' for m in ms))
    for name, X in reps.items():
        geo, geo_count, geo_sample = [], [], []
        for m in ms:
            e_g, e_gc, e_gs = [], [], []
            for cols, pc in zip(draws[m], count_preds[m]):
                pg = knn_predict(X, cols, y, allowed, args.k)
                ps = B[:, cols].mean(1)                                   # raw sample score on the probes
                e_g.append(np.abs(pg - y).mean())
                e_gc.append(np.abs(honest_blend(pc, pg, y, allowed) - y).mean())
                e_gs.append(np.abs(honest_blend(ps, pg, y, allowed) - y).mean())
            geo.append(float(np.mean(e_g))); geo_count.append(float(np.mean(e_gc))); geo_sample.append(float(np.mean(e_gs)))
        out['geometry'][name] = geo; out['geometry_plus_count'][name] = geo_count; out['geometry_plus_sample'][name] = geo_sample
        json.dump(out, open(args.out, 'w'), indent=1)              # checkpoint after every representation
        print(f'{name:18s} ' + ' '.join(f'{v:7.4f}' for v in geo) + '   |                  ' + ' '.join(f'{v:6.4f}' for v in geo_count))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
