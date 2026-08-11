"""TraceEmbedder: orchestrates channels, text embedding, and the on-disk cache.

Text embedding is batched across the whole corpus (not per trace): all texts
from cache-missing traces are gathered into one length-sorted stream, which
keeps the GPU fed and makes padding cheap. Cache layout: one .npz per trace at
    cache_dir/<model>/<replicate>/<query>.<config_hash>.npz
holding that trace's step embeddings, outcome embedding, and (if the 'whole'
channel is active) the whole-trace embedding. Trivially resumable; the config
hash covers the embed model name and canonicalization settings.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from tqdm import tqdm

from .channels import (CHANNEL_CLASSES, OutcomeChannel, StepTextChannel,
                       WholeTraceChannel)

DEFAULT_EMBED_MODEL = 'nomic-ai/nomic-embed-text-v1.5'


def make_sentence_transformer_embed_fn(model_name=DEFAULT_EMBED_MODEL,
                                       batch_size=128, device=None, prefix=None,
                                       long_char_threshold=6_000, long_batch_size=4):
    """Default embed_fn factory. The model is imported and loaded on first call,
    so fully-cached runs never touch sentence-transformers. nomic models expect
    a task prefix.

    Texts longer than long_char_threshold are encoded in much smaller batches:
    near-context-length sequences at full batch size OOM the GPU (and on hosts
    with a broken NVML the allocator's OOM path asserts instead of recovering).
    """
    if prefix is None:
        prefix = 'clustering: ' if 'nomic' in model_name else ''
    state = {}

    def embed_fn(texts):
        if 'model' not in state:
            from sentence_transformers import SentenceTransformer
            state['model'] = SentenceTransformer(model_name, trust_remote_code=True,
                                                 device=device)
        model = state['model']
        texts = [prefix + (t or ' ') for t in texts]
        short = [i for i, t in enumerate(texts) if len(t) <= long_char_threshold]
        long = [i for i, t in enumerate(texts) if len(t) > long_char_threshold]
        out = [None] * len(texts)
        for idxs, bs in ((short, batch_size), (long, long_batch_size)):
            if idxs:
                emb = _encode_with_backoff(model, [texts[i] for i in idxs], bs)
                for i, v in zip(idxs, emb):
                    out[i] = v
        return np.asarray(out)
    embed_fn.model_name = model_name
    return embed_fn


def load_dotenv(path='.env'):
    """Minimal .env loader: KEY=VALUE lines into os.environ (no overwrite)."""
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))


def make_openai_embed_fn(model='text-embedding-3-small', batch_size=128,
                         max_workers=8, max_tokens_per_input=8000,
                         max_tokens_per_batch=100_000,
                         tokens_per_minute=4_500_000,
                         api_key=None, base_url=None):
    """Embed via the OpenAI embeddings API. Batched, parallel, order-preserving.

    A shared token bucket paces dispatch to tokens_per_minute (set slightly
    under the account's TPM limit) so parallel workers never stampede the rate
    limiter; 429/5xx still back off (honoring Retry-After) as a second line."""
    import threading
    import time
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    import requests
    import tiktoken

    load_dotenv()
    key = api_key or os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY not set (and no .env found)')
    url = (base_url or os.environ.get('OPENAI_BASE_URL')
           or 'https://api.openai.com/v1').rstrip('/') + '/embeddings'
    enc = tiktoken.get_encoding('cl100k_base')

    lock = threading.Lock()
    spent = deque()          # (timestamp, tokens) within the last 60s

    def _acquire(tokens):
        while True:
            with lock:
                now = time.time()
                while spent and now - spent[0][0] > 60:
                    spent.popleft()
                if sum(t for _, t in spent) + tokens <= tokens_per_minute:
                    spent.append((now, tokens))
                    return
                wait = 60 - (now - spent[0][0]) + 0.1 if spent else 1.0
            time.sleep(min(max(wait, 0.1), 5.0))

    def _truncate(t):
        t = t or ' '
        toks = enc.encode(t, disallowed_special=())
        return enc.decode(toks[:max_tokens_per_input]) if len(toks) > max_tokens_per_input else t

    def _post(batch):
        est = sum(len(t) // 3 + 1 for t in batch)
        delay = 2.0
        last = ''
        for attempt in range(12):
            _acquire(est)
            r = requests.post(url, json={'model': model, 'input': batch},
                              headers={'Authorization': f'Bearer {key}'},
                              timeout=180)
            if r.status_code == 200:
                data = r.json()['data']
                return [d['embedding'] for d in sorted(data, key=lambda d: d['index'])]
            last = f'{r.status_code}: {r.text[:200]}'
            if r.status_code in (429, 500, 502, 503, 529):
                retry_after = float(r.headers.get('retry-after', 0) or 0)
                time.sleep(max(delay, retry_after))
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f'embeddings API {last}')
        raise RuntimeError(f'embeddings API: retries exhausted (last: {last})')

    def embed_fn(texts):
        texts = [_truncate(t) for t in texts]
        batches, cur, cur_tok = [], [], 0
        for t in texts:
            est = len(t) // 3 + 1
            if cur and (len(cur) >= batch_size or cur_tok + est > max_tokens_per_batch):
                batches.append(cur)
                cur, cur_tok = [], 0
            cur.append(t)
            cur_tok += est
        if cur:
            batches.append(cur)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_post, batches))
        return np.asarray([v for batch in results for v in batch])

    embed_fn.model_name = f'openai/{model}'
    return embed_fn


def _encode_with_backoff(model, texts, batch_size, min_batch=1):
    """encode() with halve-the-batch retries. On this host CUDA OOM surfaces
    as an NVML INTERNAL ASSERT RuntimeError (broken NVML makes the allocator's
    OOM-recovery path assert), so any RuntimeError triggers backoff."""
    while True:
        try:
            return model.encode(texts, batch_size=batch_size,
                                show_progress_bar=False)
        except RuntimeError:
            if batch_size <= min_batch:
                raise
            import torch
            torch.cuda.empty_cache()
            batch_size = max(min_batch, batch_size // 2)


def hash_embed_fn(texts, dim=64):
    """Deterministic stub embedder for tests: sha1 -> pseudo-random unit vector."""
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        seed = int.from_bytes(hashlib.sha1((t or '').encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim)
        out[i] = v / np.linalg.norm(v)
    return out
hash_embed_fn.model_name = 'hash-stub'


def rms_scale(block):
    """Scale a feature block to unit RMS entry. Preserves within-block geometry
    (unlike per-dimension z-scoring, which distorts embedding spaces)."""
    rms = np.sqrt(np.mean(block ** 2))
    return block / rms if rms > 0 else block


class TraceEmbedder:
    def __init__(self, channels=('action', 'step_text', 'outcome', 'scalar', 'whole'),
                 channel_weights=None, embed_fn=None, cache_dir=None,
                 channel_kwargs=None, embed_chunk_size=4096):
        self.channel_names = list(channels)
        self.channel_weights = dict(channel_weights or {})
        self._embed_fn = embed_fn
        self.cache_dir = cache_dir
        self.embed_chunk_size = embed_chunk_size
        channel_kwargs = channel_kwargs or {}
        self.channels = {name: CHANNEL_CLASSES[name](**channel_kwargs.get(name, {}))
                         for name in self.channel_names}
        self.cache_hits = self.cache_misses = 0

    @property
    def embed_fn(self):
        if self._embed_fn is None:
            self._embed_fn = make_sentence_transformer_embed_fn()
        return self._embed_fn

    # -- text plumbing ----------------------------------------------------
    @property
    def _text_channel_names(self):
        return [n for n in ('step_text', 'outcome', 'whole') if n in self.channels]

    def _trace_texts(self, trace):
        """{key: list_of_texts} for every text channel, for one trace."""
        out = {}
        if 'step_text' in self.channels:
            out['steps'] = self.channels['step_text'].step_texts(trace)
        if 'outcome' in self.channels:
            out['outcome'] = [self.channels['outcome'].outcome_text(trace)]
        if 'whole' in self.channels:
            out['whole'] = [self.channels['whole'].whole_text(trace)]
        return out

    # -- cache ------------------------------------------------------------
    def _config_hash(self):
        cfg = {'embed_model': getattr(self.embed_fn, 'model_name', 'custom')}
        step = self.channels.get('step_text')
        if isinstance(step, StepTextChannel):
            cfg['step'] = [step.max_chars_per_step, step.include_tool_output_chars]
        outcome = self.channels.get('outcome')
        if isinstance(outcome, OutcomeChannel):
            cfg['outcome'] = outcome.max_chars
        whole = self.channels.get('whole')
        if isinstance(whole, WholeTraceChannel):
            cfg['whole'] = [whole.max_chars, whole.include_tool_output_chars,
                            whole.max_chars_per_step, whole.diff_chars]
        return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]

    def _cache_path(self, trace):
        safe_q = trace.query_id.replace(os.sep, '_')
        return os.path.join(self.cache_dir, trace.model_id, str(trace.replicate),
                            f'{safe_q}.{self._config_hash()}.npz')

    def _load_cached(self, trace):
        """Return {key: array} for one trace, or None if absent/incomplete."""
        path = self._cache_path(trace)
        if not os.path.exists(path):
            return None
        needed = {'step_text': 'steps', 'outcome': 'outcome', 'whole': 'whole'}
        keys = [needed[n] for n in self._text_channel_names]
        with np.load(path) as z:
            if any(k not in z for k in keys):
                return None
            return {k: z[k] for k in keys}

    # -- batched embedding ------------------------------------------------
    def _embed_missing(self, traces, progress=True):
        """Embed all texts for traces not in the cache, in corpus-level chunks.
        Returns {trace_key: {key: array}} for the missing traces."""
        missing = []
        for tr in traces:
            if self.cache_dir is None or self._load_cached(tr) is None:
                missing.append(tr)
        if not missing:
            return {}

        texts, index = [], []   # index[i] = (trace_pos, key, row_within_key)
        per_trace_texts = [self._trace_texts(tr) for tr in missing]
        for pos, tmap in enumerate(per_trace_texts):
            for key, tlist in tmap.items():
                for row, t in enumerate(tlist):
                    texts.append(t)
                    index.append((pos, key, row))

        chunks = range(0, len(texts), self.embed_chunk_size)
        if progress:
            chunks = tqdm(chunks, desc=f'embedding {len(texts)} texts '
                                       f'({len(missing)} traces)')
        emb = np.vstack([self.embed_fn(texts[i:i + self.embed_chunk_size])
                         for i in chunks]) if texts else np.zeros((0, 1))

        dim = emb.shape[1]
        results = {}
        for pos, tr in enumerate(missing):
            tmap = per_trace_texts[pos]
            results[tr.key] = {
                'steps' if key == 'steps' else key:
                    np.zeros((len(tmap[key]), dim)) for key in tmap
            }
        for (pos, key, row), vec in zip(index, emb):
            results[missing[pos].key][key][row] = vec

        for tr in missing:
            r = results[tr.key]
            flat = {k: (v[0] if k in ('outcome', 'whole') else v) for k, v in r.items()}
            results[tr.key] = flat
            if self.cache_dir is not None:
                path = self._cache_path(tr)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.savez_compressed(path, **flat)
        return results

    # -- public API -------------------------------------------------------
    def transform_channels(self, traces, progress=True):
        """Compute each channel separately: {channel_name: (n_traces, d_ch)}."""
        step_embs = outcome_embs = whole_embs = None
        if self._text_channel_names:
            fresh = self._embed_missing(traces, progress=progress)
            self.cache_misses += len(fresh)
            step_embs, outcome_embs, whole_embs = [], [], []
            for tr in traces:
                r = fresh.get(tr.key)
                if r is None:
                    r = self._load_cached(tr)
                    self.cache_hits += 1
                if 'step_text' in self.channels:
                    step_embs.append(r['steps'])
                if 'outcome' in self.channels:
                    outcome_embs.append(r['outcome'])
                if 'whole' in self.channels:
                    whole_embs.append(r['whole'])

        out = {}
        for name, ch in self.channels.items():
            if name == 'step_text':
                out[name] = ch.transform(traces, step_embeddings=step_embs)
            elif name == 'outcome':
                out[name] = ch.transform(traces, outcome_embeddings=outcome_embs)
            elif name == 'whole':
                out[name] = ch.transform(traces, whole_embeddings=whole_embs)
            else:
                out[name] = ch.transform(traces)
        return out

    def transform(self, traces, progress=True):
        """Single concatenated vector per trace: RMS-scale each channel block,
        apply channel weights, concatenate."""
        blocks = self.transform_channels(traces, progress=progress)
        parts = [rms_scale(blocks[name]) * self.channel_weights.get(name, 1.0)
                 for name in self.channel_names]
        return np.hstack(parts)
