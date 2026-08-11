"""Rubric-anchored representation of long traces.

The whole-trace embedding truncates a median leaderboard trace to ~16% of its
content. Instead: chunk the rendered trace, embed every chunk, and pool chunks
into a fixed set of rubric sections by softmax similarity to embedded section
descriptions. Off-the-shelf embeddings only; no per-format parsing, no
classifier. Output per trace:

  - section vectors: (n_sections, d) similarity-weighted chunk means
  - mass: (n_sections,) fraction of the trace's chunks assigned to each section
    (a behavioral phase-profile that is interpretable on its own)
"""
from __future__ import annotations

import numpy as np

DEFAULT_RUBRIC = {
    'understanding': 'Reading and restating the problem. Reasoning about the '
                     'issue description, the expected behavior, and what needs '
                     'to change before taking any action.',
    'localization': 'Searching the repository to find relevant code: grep, '
                    'find, search, listing directories, opening and viewing '
                    'files to locate the source of the issue.',
    'reproduction': 'Writing and running a small script or test to reproduce '
                    'the reported bug and confirm the failure before fixing it.',
    'editing': 'Modifying source code to implement the fix: editing files, '
               'replacing code, applying a patch or diff to the repository.',
    'verification': 'Running the test suite or the reproduction script after '
                    'the change to check the fix works and nothing is broken; '
                    'reading test output, pass and fail results.',
    'submission': 'Wrapping up: summarizing the change that was made and '
                  'submitting the final patch.',
}


def chunk_text(text, chunk_chars=4000):
    """Split text into ~chunk_chars pieces on line boundaries. Lines longer
    than chunk_chars (minified blobs) are split mid-line so no chunk ever
    exceeds ~2x chunk_chars."""
    if not text:
        return []
    chunks, cur, n = [], [], 0
    for line in text.split('\n'):
        while len(line) > chunk_chars:
            if cur:
                chunks.append('\n'.join(cur))
                cur, n = [], 0
            chunks.append(line[:chunk_chars])
            line = line[chunk_chars:]
        cur.append(line)
        n += len(line) + 1
        if n >= chunk_chars:
            chunks.append('\n'.join(cur))
            cur, n = [], 0
    if cur:
        chunks.append('\n'.join(cur))
    return chunks


def embed_rubric(rubric, embed_fn):
    """(n_sections, d) row-normalized anchor matrix from section descriptions."""
    A = np.asarray(embed_fn([f'{k}: {v}' for k, v in rubric.items()]), dtype=float)
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def rubric_pool(chunk_embs, anchors, tau=0.05, hard=False):
    """Pool chunk embeddings into rubric sections.

    chunk_embs: (n_chunks, d); anchors: (n_sections, d) normalized.
    Returns (sections (n_sections, d), mass (n_sections,)).
    Zero-chunk traces get zero sections and uniform mass.
    """
    S, d = anchors.shape
    if len(chunk_embs) == 0:
        return np.zeros((S, d)), np.full(S, 1.0 / S)
    X = np.asarray(chunk_embs, dtype=float)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    sim = Xn @ anchors.T                       # (n_chunks, S) cosines
    if hard:
        W = np.zeros_like(sim)
        W[np.arange(len(sim)), sim.argmax(1)] = 1.0
    else:
        z = sim / tau
        z -= z.max(axis=1, keepdims=True)
        W = np.exp(z)
        W /= W.sum(axis=1, keepdims=True)      # per-chunk assignment probs
    mass = W.mean(axis=0)                      # (S,)
    denom = np.maximum(W.sum(axis=0), 1e-12)   # (S,)
    sections = (W.T @ X) / denom[:, None]      # (S, d)
    return sections, mass


def rubric_vector(chunk_embs, anchors, tau=0.05, hard=False, include_mass=True):
    """Flat per-trace vector: concatenated section vectors (+ mass block)."""
    sections, mass = rubric_pool(chunk_embs, anchors, tau=tau, hard=hard)
    parts = [sections.ravel()]
    if include_mass:
        parts.append(mass)
    return np.concatenate(parts)
