"""Trace-level evaluation metrics that don't require running a test harness.

Gold-patch file localization: the fraction of files touched by the reference
(gold) patch that the submitted patch also touches. Correctness-adjacent,
available per (model, query, replicate), and -- unlike submit rate -- not
saturated at the top of the model range.
"""
from __future__ import annotations

from .canonicalize import diff_stats


def file_localization(trace, gold_files):
    """Fraction of gold-patch files touched by this trace's submitted patch.

    gold_files: {query_id: set-of-paths}. Returns 0.0 for empty patches.
    """
    gold = set(gold_files[trace.query_id])
    if not gold:
        raise ValueError(f'no gold files for query {trace.query_id}')
    touched = set(diff_stats(trace.final_output)['paths'])
    return len(gold & touched) / len(gold)


def load_swebench_gold_files(query_ids, dataset='princeton-nlp/SWE-bench_Verified',
                             split='test'):
    """{query_id: set of file paths edited by the gold patch}, from HuggingFace.

    Lazy import; cache the result to disk yourself if calling repeatedly.
    """
    from datasets import load_dataset
    wanted = set(query_ids)
    out = {}
    for row in load_dataset(dataset, split=split):
        if row['instance_id'] in wanted:
            out[row['instance_id']] = set(diff_stats(row['patch'])['paths'])
    missing = wanted - out.keys()
    if missing:
        raise ValueError(f'instances not found in {dataset}: {sorted(missing)}')
    return out
