"""Extraction-only qubric judging via OpenRouter (open-weight judges).

Migration step 1 (PAPER.md): re-run q20 extraction with open judges on the
IDENTICAL cached gpt-5.4-mini-written rubrics, so only the extractor changes.
Caches to data/judge/structured-qspec-<safe_model>/<sys>/<q>.json -- the same
layout judge_matrix.py consumes, so new judges appear in the matrix by adding
a CELLS entry. No embedding here (local nomic happens in judge_matrix.py).

Reasoning models: effort pinned low -- extraction is transcription, not
reasoning, and low effort keeps cost/latency near non-reasoning models.

Usage:
  python scripts/judge_openrouter.py --judge-model openai/gpt-oss-20b [--limit 20]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import requests
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from judge_structured import (EXTRACT_PROMPT, SECTIONS, _load_one)  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))


def chat(key, model, content, max_tokens=2500):
    body = {'model': model, 'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_tokens,
            'response_format': {'type': 'json_object'},
            'reasoning': {'effort': 'low'}}
    delay = 2.0
    for _ in range(10):
        r = requests.post('https://openrouter.ai/api/v1/chat/completions', json=body,
                          headers={'Authorization': f'Bearer {key}'}, timeout=300)
        if r.status_code == 200:
            j = r.json()
            if 'choices' not in j:            # provider-level error in 200 body
                time.sleep(delay); delay = min(delay * 2, 60); continue
            return j['choices'][0]['message']['content'] or '{}'
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(max(delay, float(r.headers.get('retry-after', 0) or 0)))
            delay = min(delay * 2, 60)
            continue
        raise RuntimeError(f'{r.status_code}: {r.text[:200]}')
    raise RuntimeError('chat retries exhausted')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge-model', required=True)
    ap.add_argument('--root', default='data/leaderboard/verified')
    ap.add_argument('--labels', default='data/leaderboard/verified_labels.json')
    ap.add_argument('--rubrics', default='data/judge/qspec_rubrics')
    ap.add_argument('--ref-cache', default='data/judge/structured-qspec')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0,
                    help='only run this many calls (pilot mode)')
    args = ap.parse_args()

    load_dotenv()
    key = os.environ['OPENROUTER_API_KEY']

    # systems and q20 exactly as cached for the reference judge
    q20 = sorted(f[:-5] for f in os.listdir(
        os.path.join(args.ref_cache, sorted(os.listdir(args.ref_cache))[0])))
    labels = json.load(open(args.labels))
    systems = sorted(s for s in os.listdir(args.ref_cache)
                     if 'resolved' in labels.get(s, {}))
    rubrics = {q: json.loads(open(os.path.join(args.rubrics, f'{q}.json')).read())
               for q in q20}

    safe = args.judge_model.replace('/', '_')
    jdir = f'data/judge/structured-qspec-{safe}'

    def jpath(s, q):
        return os.path.join(jdir, s, f'{q}.json')

    todo = [(s, q) for s in systems for q in q20 if not os.path.exists(jpath(s, q))]
    if args.limit:
        todo = todo[:args.limit]
    print(f'{len(systems)} systems x {len(q20)} instances; {len(todo)} calls todo')
    if not todo:
        return

    need = sorted({s for s, _ in todo})
    with Pool(16) as pool:
        texts = dict(pool.map(_load_one, [(os.path.join(args.root, s), set(q20))
                                          for s in need]))

    def extract(pair):
        s, q = pair
        rub = '\n'.join(f'- {k}: {rubrics[q].get(k, "")}' for k in SECTIONS)
        base = EXTRACT_PROMPT.format(rubric=rub, keys=SECTIONS)
        content = base + '\n\n=== TRACE ===\n' + texts[s].get(q, '(empty trace)')
        try:
            out = chat(key, args.judge_model, content)
        except RuntimeError as e:
            print(f'FAIL {s}/{q}: {e}', file=sys.stderr)
            return
        os.makedirs(os.path.dirname(jpath(s, q)), exist_ok=True)
        open(jpath(s, q), 'w').write(out)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(extract, todo), total=len(todo), desc=safe))

    n_ok = n_parse = 0
    for s, q in todo:
        if os.path.exists(jpath(s, q)):
            n_ok += 1
            try:
                d = json.loads(open(jpath(s, q)).read())
                if isinstance(d, list) and d:
                    d = d[0]
                if isinstance(d, dict) and any(d.get(k) for k in SECTIONS):
                    n_parse += 1
            except json.JSONDecodeError:
                pass
    print(f'written {n_ok}/{len(todo)}; parseable-with-content {n_parse}')


if __name__ == '__main__':
    main()
