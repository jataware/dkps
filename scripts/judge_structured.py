"""Structured judge extraction: LLM extracts per-rubric-section descriptions
from each trace; sections are embedded separately and concatenated (aligned
section-to-section comparison across systems).

Variants:
  A (--variant fixed): fixed 6-section SWE-bench rubric.
  B (--variant qspec): per-instance rubric -- an LLM first writes, from the
    problem statement, what each section role specifically means for that
    instance; the extractor works against that rubric. Rubric cached per
    instance.

Cache: data/judge/structured-<variant>/<system>/<instance>.json
Usage: python scripts/judge_structured.py --variant fixed|qspec
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from glob import glob
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dkps.traces import load_dotenv, make_openai_embed_fn
from dkps.traces.leaderboard import _extract_query_id
import dkps.traces.leaderboard as lb

SECTIONS = ['understanding', 'localization', 'reproduction',
            'editing', 'verification', 'final_state']

EXTRACT_PROMPT = """Below is the (truncated) execution trace of a software-engineering agent.
Extract a factual description of the agent's behavior for EACH of the following
sections. 30-60 words per section. Describe behavior only -- do not restate the
problem. If a section did not occur, write exactly what is missing (e.g. "never
ran any test after editing").

{rubric}

Respond with ONLY a JSON object with keys {keys}, each a string.
"""

FIXED_RUBRIC = """- understanding: how the agent oriented itself before acting
- localization: how it searched/navigated to find the relevant code
- reproduction: whether/how it reproduced the issue before fixing
- editing: what it changed (files, nature and size of the edit)
- verification: whether it re-ran tests or a repro script after editing, and the OBSERVED outcome (passing/failing/errors)
- final_state: how the run ended (clean finish, step/cost limit, submitted unverified)"""

QSPEC_PROMPT = """Here is a repository issue that a software-engineering agent will be asked to fix:

{problem}

Write an instance-specific rubric for judging an agent's trace on THIS issue:
for each section below, one sentence saying what specifically to look for in
this instance (name the likely relevant modules/files/tests if inferable).

Sections: understanding, localization, reproduction, editing, verification, final_state.
Respond with ONLY a JSON object with those keys, each a one-sentence string."""

QUESTIONS_PROMPT = """Here is a repository issue that a software-engineering agent will be asked to fix:

{problem}

Write six instance-specific YES/NO questions for judging an agent's trace on
THIS issue, one per section below. Each question should name the specific
thing to check for this instance (likely modules/files/tests if inferable).

Sections: understanding, localization, reproduction, editing, verification, final_state.
Respond with ONLY a JSON object with those keys, each a single question string."""

QUESTIONS_EXTRACT = """Below is the (truncated) execution trace of a software-engineering agent.
Answer EACH question about the agent's behavior. Start each answer with YES,
NO, or PARTIALLY, followed by one sentence of concrete evidence from the trace
(or what is missing). Judge behavior only; do not speculate.

{rubric}

Respond with ONLY a JSON object with keys {keys}, each a string.
"""

_wanted_g = None


def _load_one(args):
    sub_dir, wanted = args
    ts = lb.load_leaderboard_submission(sub_dir, labels=None)
    out = {}
    for t in ts:
        if t.query_id in wanted:
            txt = t.steps[0].assistant_text or ''
            out[t.query_id] = txt[:40_000] + ('\n...[omitted]...\n' + txt[-20_000:]
                                              if len(txt) > 60_000 else '')
    return os.path.basename(sub_dir), out


def chat(url, key, model, content, max_tokens=2000):
    body = {'model': model, 'messages': [{'role': 'user', 'content': content}],
            'max_completion_tokens': max_tokens,
            'response_format': {'type': 'json_object'}}
    delay = 2.0
    for _ in range(10):
        r = requests.post(url, json=body,
                          headers={'Authorization': f'Bearer {key}'}, timeout=300)
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content'] or '{}'
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(max(delay, float(r.headers.get('retry-after', 0) or 0)))
            delay = min(delay * 2, 60)
            continue
        raise RuntimeError(f'{r.status_code}: {r.text[:200]}')
    raise RuntimeError('chat retries exhausted')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=('fixed', 'qspec', 'questions'), default='fixed')
    ap.add_argument('--judge-model', default='gpt-5.4-mini')
    ap.add_argument('--root', default='data/leaderboard/verified')
    ap.add_argument('--labels', default='data/leaderboard/verified_labels.json')
    ap.add_argument('--cache-dir', default='.dkps_cache_lb')
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    load_dotenv()
    key = os.environ['OPENAI_API_KEY']
    url = (os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
           ).rstrip('/') + '/chat/completions'

    labels = json.load(open(args.labels))
    all_systems = [s for s in sorted(os.listdir(args.root))
                   if len(glob(os.path.join(args.root, s, 'trajs', '*'))) >= 480
                   and 'resolved' in labels.get(s, {})]
    per_sys = [{_extract_query_id(os.path.basename(os.path.normpath(p)))
                for p in glob(os.path.join(args.root, s, 'trajs', '*'))}
               for s in all_systems]
    q418 = sorted(set.intersection(*per_sys))
    rng0 = np.random.default_rng(0)
    q150 = sorted(rng0.choice(q418, 150, replace=False))
    rng2 = np.random.default_rng(1)
    q20 = sorted(rng2.choice(q150, 20, replace=False))
    ccfg = hashlib.sha1('openai/text-embedding-3-small|chunks4000'.encode()).hexdigest()[:8]
    systems = [s for s in all_systems
               if all(os.path.exists(f'{args.cache_dir}/{s}/{q}.{ccfg}.npz') for q in q20)]
    y = np.array([len(labels[s]['resolved']) / 500 for s in systems])
    M, Q = len(systems), len(q20)
    print(f'{M} systems x {Q} instances, variant={args.variant}')

    # per-instance rubrics for qspec / questions
    rubrics = {}
    if args.variant in ('qspec', 'questions'):
        rdir = ('data/judge/qspec_rubrics' if args.variant == 'qspec'
                else 'data/judge/qspec_rubrics_questions')
        rprompt = QSPEC_PROMPT if args.variant == 'qspec' else QUESTIONS_PROMPT
        os.makedirs(rdir, exist_ok=True)
        from datasets import load_dataset
        stmts = {r['instance_id']: r['problem_statement'][:20_000]
                 for r in load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
                 if r['instance_id'] in set(q20)}
        for q in q20:
            p = os.path.join(rdir, f'{q}.json')
            if not os.path.exists(p):
                out = chat(url, key, args.judge_model,
                           rprompt.format(problem=stmts[q]), 1200)
                open(p, 'w').write(out)
            rubrics[q] = json.loads(open(p).read())
        print('instance rubrics ready')

    jdir = f'data/judge/structured-{args.variant}'

    def jpath(s, q):
        return os.path.join(jdir, s, f'{q}.json')

    todo = [(s, q) for s in systems for q in q20 if not os.path.exists(jpath(s, q))]
    if todo:
        need = sorted({s for s, _ in todo})
        with Pool(16) as pool:
            texts = dict(pool.map(_load_one, [(os.path.join(args.root, s), set(q20))
                                              for s in need]))

        def extract(pair):
            s, q = pair
            if args.variant == 'fixed':
                base = EXTRACT_PROMPT.format(rubric=FIXED_RUBRIC, keys=SECTIONS)
            elif args.variant == 'qspec':
                rub = '\n'.join(f'- {k}: {rubrics[q].get(k, "")}' for k in SECTIONS)
                base = EXTRACT_PROMPT.format(rubric=rub, keys=SECTIONS)
            else:
                rub = '\n'.join(f'- {k}: {rubrics[q].get(k, "")}' for k in SECTIONS)
                base = QUESTIONS_EXTRACT.format(rubric=rub, keys=SECTIONS)
            content = base + '\n\n=== TRACE ===\n' + texts[s].get(q, '(empty trace)')
            out = chat(url, key, args.judge_model, content)
            os.makedirs(os.path.dirname(jpath(s, q)), exist_ok=True)
            open(jpath(s, q), 'w').write(out)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(tqdm(ex.map(extract, todo), total=len(todo), desc='extract'))

    # parse + embed sections, evaluate
    texts6 = np.empty((M, Q, len(SECTIONS)), dtype=object)
    n_bad = 0
    for i, s in enumerate(systems):
        for j, q in enumerate(q20):
            try:
                d = json.loads(open(jpath(s, q)).read())
            except json.JSONDecodeError:
                d = {}
                n_bad += 1
            for k, sec in enumerate(SECTIONS):
                texts6[i, j, k] = str(d.get(sec, '') or ' ')
    print(f'parse failures: {n_bad}')
    embed_fn = make_openai_embed_fn()
    flat = [texts6[i, j, k] for i in range(M) for j in range(Q)
            for k in range(len(SECTIONS))]
    E = embed_fn(flat).reshape(M, Q, len(SECTIONS), -1).astype(np.float32)

    def tagf(s, k):
        m = re.search(rf'^\s+{k}:\s*(.*)$', labels[s].get('metadata_yaml', ''), re.M)
        return m.group(1).strip().strip('"\'') if m else None
    model_tag = {s: tagf(s, 'model_display') for s in systems}
    allowed = np.array([[jj != i and not (model_tag[systems[i]]
                        and model_tag[systems[jj]] == model_tag[systems[i]])
                        for jj in range(M)] for i in range(M)])

    def knn_eval(Xv):
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(Xv.reshape(M, -1)))
        preds = []
        for i in range(M):
            idx = np.where(allowed[i])[0]
            nn = idx[np.argsort(D[i][idx])[:3]]
            w = 1 / (D[i][nn] + 1e-12)
            preds.append(np.dot(w, y[nn]) / w.sum())
        preds = np.array(preds)
        return np.abs(preds - y).mean(), spearmanr(preds, y).statistic

    def ridge_eval(Xv, lams=(1.0, 10.0, 100.0)):
        feats = Xv.reshape(M, -1)
        best = (np.inf, 0)
        for lam in lams:
            preds = []
            for i in range(M):
                tr = np.where(allowed[i])[0]
                A = feats[tr]; b = y[tr]
                mu = A.mean(0); Ac = A - mu; bm = b.mean()
                G = Ac @ Ac.T + lam * np.eye(len(tr))
                al = np.linalg.solve(G, b - bm)
                preds.append(bm + (feats[i] - mu) @ (Ac.T @ al))
            preds = np.array(preds)
            mae = np.abs(preds - y).mean()
            if mae < best[0]:
                best = (mae, spearmanr(preds, y).statistic)
        return best

    def norm(A):
        return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-9)

    def rms(A):
        r = np.sqrt((A ** 2).mean())
        return A / max(r, 1e-9)

    def report(name, Xv):
        km, kr = knn_eval(Xv)
        rm, rr = ridge_eval(Xv)
        print(f'{name:42s} knn: {km:.4f}/{kr:.3f}  ridge: {rm:.4f}/{rr:.3f}')

    hcfg = hashlib.sha1('openai/text-embedding-3-small|headtail8000'.encode()).hexdigest()[:8]
    X20 = np.zeros((M, Q, 3072), np.float32)
    for i, s in enumerate(systems):
        for j, q in enumerate(q20):
            with np.load(f'{args.cache_dir}/{s}/{q}.{hcfg}.npz') as z:
                X20[i, j] = np.concatenate([z['head'], z['tail']])
    REF = norm(X20 - np.median(X20, axis=0, keepdims=True))

    # per-section centering (median over systems, per instance per section)
    Es = E - np.median(E, axis=0, keepdims=True)
    Es = Es / np.maximum(np.linalg.norm(Es, axis=-1, keepdims=True), 1e-9)
    SECREP = Es.reshape(M, Q, -1)

    report('NAIVE head+tail centered', REF)
    report(f'sections concat ({args.variant})', norm(E.reshape(M, Q, -1)))
    report(f'sections concat centered ({args.variant})', SECREP)
    report('NAIVE + sections', np.concatenate([rms(REF), rms(SECREP)], -1))


if __name__ == '__main__':
    main()
