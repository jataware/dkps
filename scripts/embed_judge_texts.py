"""Embed the cached judge texts (blob, generic rubric, verdict, qubric) with
OpenAI text-embedding-3-small -- the embedder behind the paper's figures -- and
store them in the judge-matrix cache layout used by judge_matrix.py:

  data/judge/matrix_emb/<construction>-gpt-5.4-mini.openai_text-embedding-3-small.npz
      E        float32 (M systems, 20 tasks, k sections, 1536); k = 1 for blob, 6 otherwise
      systems, q20   row / column labels (sorted, as in every other script)

Text loading is identical to judge_matrix.py: a missing file becomes ' ', an
unparseable JSON becomes six blank sections. Existing files are skipped.
Cost: ~0.9M tokens per construction (a few cents). Key: OPENAI_API_KEY in the
environment or in ./.env.

Usage: python scripts/embed_judge_texts.py [--cells blob,generic,verdict,qubric] [--dry-run]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import requests

SECTIONS = ('understanding', 'localization', 'reproduction', 'editing', 'verification', 'final_state')
CELLS = {'blob': ('data/judge/gpt-5.4-mini', 'txt'),
         'generic': ('data/judge/structured-fixed', 'json'),
         'verdict': ('data/judge/structured-questions', 'json'),
         'qubric': ('data/judge/structured-qspec', 'json')}
MODEL = 'text-embedding-3-small'
TAG = 'openai_text-embedding-3-small'
OUT_DIR = 'data/judge/matrix_emb'


def load_env(path='.env'):
    if os.path.exists(path):
        for line in open(path):
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))


def load_texts(d, kind, systems, q20):
    n_bad = 0
    if kind == 'txt':
        T = np.empty((len(systems), len(q20), 1), object)
        for i, s in enumerate(systems):
            for j, q in enumerate(q20):
                p = os.path.join(d, s, f'{q}.txt')
                T[i, j, 0] = open(p).read() if os.path.exists(p) else ' '
    else:
        T = np.empty((len(systems), len(q20), len(SECTIONS)), object)
        for i, s in enumerate(systems):
            for j, q in enumerate(q20):
                try:
                    dd = json.loads(open(os.path.join(d, s, f'{q}.json')).read())
                    if isinstance(dd, list) and dd:
                        dd = dd[0]
                    if not isinstance(dd, dict):
                        dd, n_bad = {}, n_bad + 1
                except (json.JSONDecodeError, FileNotFoundError):
                    dd, n_bad = {}, n_bad + 1
                for k, sec in enumerate(SECTIONS):
                    T[i, j, k] = str(dd.get(sec, '') or ' ')
    return T, n_bad


def embed(texts, key, batch=128):
    out = []
    for b0 in range(0, len(texts), batch):
        chunk = [t if t.strip() else ' ' for t in texts[b0:b0 + batch]]
        for attempt in range(8):
            r = requests.post('https://api.openai.com/v1/embeddings',
                              json={'model': MODEL, 'input': chunk},
                              headers={'Authorization': f'Bearer {key}'}, timeout=180)
            if r.status_code == 200:
                out.extend(d['embedding'] for d in sorted(r.json()['data'], key=lambda d: d['index']))
                break
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(float(r.headers.get('retry-after', 0) or 2 ** attempt))
                continue
            sys.exit(f'embeddings API {r.status_code}: {r.text[:200]}')
        else:
            sys.exit('embeddings API: retries exhausted')
        print(f'  {min(b0 + batch, len(texts))}/{len(texts)}', end='\r', flush=True)
    return np.asarray(out, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cells', default=','.join(CELLS))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    load_env()
    key = os.environ.get('OPENAI_API_KEY')
    if not key and not args.dry_run:
        sys.exit('OPENAI_API_KEY not set (put it in .env)')

    labels = json.load(open('data/leaderboard/verified_labels.json'))
    ref = 'data/judge/structured-qspec'
    systems = sorted(s for s in os.listdir(ref) if 'resolved' in labels.get(s, {}))
    q20 = sorted(f[:-5] for f in os.listdir(os.path.join(ref, systems[0])))
    os.makedirs(OUT_DIR, exist_ok=True)

    for name in args.cells.split(','):
        d, kind = CELLS[name]
        p = os.path.join(OUT_DIR, f'{name}-gpt-5.4-mini.{TAG}.npz')
        if os.path.exists(p):
            print(f'{name}: exists, skipping ({p})')
            continue
        T, n_bad = load_texts(d, kind, systems, q20)
        flat = [T[i, j, k] for i in range(T.shape[0]) for j in range(T.shape[1]) for k in range(T.shape[2])]
        n_tok = sum(len(t) for t in flat) // 4
        print(f'{name}: {len(flat)} strings, ~{n_tok / 1e6:.2f}M tokens, {n_bad} unparseable cells')
        if args.dry_run:
            continue
        E = embed(flat, key).reshape(T.shape[0], T.shape[1], T.shape[2], -1)
        np.savez_compressed(p, E=E, systems=np.array(systems), q20=np.array(q20))
        print(f'\n  wrote {p} E{E.shape}')


if __name__ == '__main__':
    main()
