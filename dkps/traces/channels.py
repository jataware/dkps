"""Per-trace feature channels.

Each channel maps a list of Traces to an (n_traces, d_channel) array. Text
channels take an embed_fn: list[str] -> ndarray; the other channels are
deterministic. Channels are intentionally independent so any combination
scheme (concatenation, per-channel DKPS, weighted distances) can be explored
downstream.
"""
from __future__ import annotations

import numpy as np

from .canonicalize import (canonicalize_diff, canonicalize_step, diff_stats,
                           head_tail)

START, END = '<start>', '<end>'

# fixed for the swe-agent tool suite; pass vocab=None to infer from the corpus
DEFAULT_TOOL_VOCAB = ('bash', 'str_replace_editor', 'search_dir', 'search_file',
                      'find_file', 'submit')

EXIT_STATUS_VOCAB = ('submitted', 'submitted (exit_cost)', 'exit_cost',
                     'exit_format', 'exit_error', 'other')


def _tool_sequence(trace):
    return [s.tool_name for s in trace.steps if s.tool_name is not None]


class ActionSequenceChannel:
    """Tool-name unigram/bigram counts (log1p) + row-normalized transition matrix."""
    name = 'action'

    def __init__(self, vocab=DEFAULT_TOOL_VOCAB, include_transition_matrix=True):
        self.vocab = list(vocab) if vocab is not None else None
        self.include_transition_matrix = include_transition_matrix

    def transform(self, traces, embed_fn=None):
        if self.vocab is None:
            self.vocab = sorted({t for tr in traces for t in _tool_sequence(tr)})
        vocab = self.vocab
        idx = {t: i for i, t in enumerate(vocab)}
        n_tok = len(vocab) + 2  # + start/end for the transition matrix
        rows = []
        for tr in traces:
            seq = [t for t in _tool_sequence(tr) if t in idx]
            uni = np.zeros(len(vocab))
            for t in seq:
                uni[idx[t]] += 1
            padded = [len(vocab)] + [idx[t] for t in seq] + [len(vocab) + 1]
            trans = np.zeros((n_tok, n_tok))
            for a, b in zip(padded[:-1], padded[1:]):
                trans[a, b] += 1
            row_sums = trans.sum(axis=1, keepdims=True)
            trans = np.divide(trans, row_sums, out=np.zeros_like(trans), where=row_sums > 0)
            feats = [np.log1p(uni)]
            if self.include_transition_matrix:
                feats.append(trans.ravel())
            rows.append(np.concatenate(feats))
        return np.array(rows)


class StepTextChannel:
    """Embed each step's canonical text, pool over steps.

    pooling='mean' or 'segments' (mean over first/middle/last thirds, concatenated).
    Zero-step traces get a zero vector: degenerate runs land far from everything,
    which is the desired geometry.
    """
    name = 'step_text'

    def __init__(self, pooling='mean', n_position_segments=3, max_chars_per_step=1000,
                 include_tool_output_chars=0):
        self.pooling = pooling
        self.n_position_segments = n_position_segments
        self.max_chars_per_step = max_chars_per_step
        self.include_tool_output_chars = include_tool_output_chars

    def step_texts(self, trace):
        return [canonicalize_step(s, self.max_chars_per_step, self.include_tool_output_chars)
                for s in trace.steps]

    def transform(self, traces, embed_fn=None, step_embeddings=None):
        """step_embeddings: optional precomputed list of (n_steps_i, d) arrays
        (from the cache); otherwise embed_fn is called on all step texts."""
        if step_embeddings is None:
            if embed_fn is None:
                raise ValueError('StepTextChannel requires embed_fn or step_embeddings')
            texts, spans = [], []
            for tr in traces:
                st = self.step_texts(tr)
                spans.append((len(texts), len(texts) + len(st)))
                texts.extend(st)
            flat = embed_fn(texts) if texts else np.zeros((0, 1))
            step_embeddings = [flat[a:b] for a, b in spans]

        dim = next((e.shape[1] for e in step_embeddings if len(e)), None)
        if dim is None:
            raise ValueError('no trace has any steps')

        rows = []
        for emb in step_embeddings:
            if self.pooling == 'segments':
                segs = []
                for chunk in np.array_split(emb, self.n_position_segments) if len(emb) else []:
                    segs.append(chunk.mean(axis=0) if len(chunk) else np.zeros(dim))
                if not segs:
                    segs = [np.zeros(dim)] * self.n_position_segments
                rows.append(np.concatenate(segs))
            else:
                rows.append(emb.mean(axis=0) if len(emb) else np.zeros(dim))
        return np.array(rows)


class OutcomeChannel:
    """Embed the canonicalized final diff (stats header + truncated diff text)."""
    name = 'outcome'

    def __init__(self, max_chars=4000):
        self.max_chars = max_chars

    def outcome_text(self, trace):
        return canonicalize_diff(trace.final_output, self.max_chars)

    def transform(self, traces, embed_fn=None, outcome_embeddings=None):
        if outcome_embeddings is not None:
            return np.array(outcome_embeddings)
        if embed_fn is None:
            raise ValueError('OutcomeChannel requires embed_fn or outcome_embeddings')
        return np.asarray(embed_fn([self.outcome_text(tr) for tr in traces]))


class WholeTraceChannel:
    """Naive baseline: the whole trajectory rendered as one text (assistant text,
    tool calls, truncated tool outputs, final diff), embedded in a single shot.
    The embedder truncates at its context limit -- that information loss is the
    point of the baseline."""
    name = 'whole'

    def __init__(self, max_chars=32_000, include_tool_output_chars=2_000,
                 max_chars_per_step=4_000, diff_chars=8_000):
        self.max_chars = max_chars
        self.include_tool_output_chars = include_tool_output_chars
        self.max_chars_per_step = max_chars_per_step
        self.diff_chars = diff_chars

    def whole_text(self, trace):
        parts = [canonicalize_step(s, self.max_chars_per_step,
                                   self.include_tool_output_chars)
                 for s in trace.steps]
        parts.append('FINAL PATCH:')
        parts.append(canonicalize_diff(trace.final_output, self.diff_chars))
        return head_tail('\n'.join(parts), self.max_chars)

    def transform(self, traces, embed_fn=None, whole_embeddings=None):
        if whole_embeddings is not None:
            return np.array(whole_embeddings)
        if embed_fn is None:
            raise ValueError('WholeTraceChannel requires embed_fn or whole_embeddings')
        return np.asarray(embed_fn([self.whole_text(tr) for tr in traces]))


class ScalarChannel:
    """Numeric behavioral features, z-scored across the corpus."""
    name = 'scalar'

    def __init__(self, tool_vocab=DEFAULT_TOOL_VOCAB, exit_vocab=EXIT_STATUS_VOCAB,
                 step_cap=50):
        self.tool_vocab = list(tool_vocab)
        self.exit_vocab = list(exit_vocab)
        self.step_cap = step_cap

    def _features(self, tr):
        seq = _tool_sequence(tr)
        n = len(seq)
        f = [n, float(n >= self.step_cap)]
        f += [(sum(1 for t in seq if t == v) / n if n else 0.0) for v in self.tool_vocab]
        f.append(sum(1 for s in tr.steps if s.tool_success is False) / n if n else 0.0)
        f.append(sum(1 for s in tr.steps if s.assistant_text) / len(tr.steps)
                 if tr.steps else 0.0)
        es = tr.exit_status or 'other'
        if es not in self.exit_vocab:
            es = 'other'
        f += [float(es == v) for v in self.exit_vocab]
        md = tr.metadata
        f.append(np.log1p(float(md.get('total_cost', 0) or 0)))
        f.append(np.log1p(float(md.get('total_tokens_sent', 0) or 0)))
        f.append(np.log1p(float(md.get('total_tokens_received', 0) or 0)))
        f.append(np.log1p(float(md.get('api_calls', 0) or 0)))
        f.append(np.log1p(sum(len(s.tool_output or '') for s in tr.steps)))
        ds = diff_stats(tr.final_output)
        f += [ds['n_files'], ds['n_hunks'], np.log1p(ds['plus_lines']),
              np.log1p(ds['minus_lines']), np.log1p(ds['size_bytes'])]
        return f

    def transform(self, traces, embed_fn=None):
        X = np.array([self._features(tr) for tr in traces], dtype=float)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd == 0] = 1.0
        return (X - mu) / sd


CHANNEL_CLASSES = {
    'action': ActionSequenceChannel,
    'step_text': StepTextChannel,
    'outcome': OutcomeChannel,
    'scalar': ScalarChannel,
    'whole': WholeTraceChannel,
}
