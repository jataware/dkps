"""Loader for Langfuse-style span trajectories (one JSON span per line).

Expected corpus layout (as produced by the swe-agent export):

    root/
      <model>-swe-bench-verified-test-YYYY-MM-DD-<r>of5/
        <query_id>_trajectory.jsonl
        ...

Span types: one AGENT span (final output + run metadata), GENERATION spans
(assistant turns; tool calls in llm.tool_calls, text in llm.response_text),
TOOL spans (action.name, action.input.data, result.output.data).
"""
from __future__ import annotations

import json
import os
import re
from glob import glob

from .schema import Step, Trace

# metadata keys hoisted out of the AGENT span's otel span_attributes
_METADATA_PREFIX = 'langfuse.trace.metadata.'

DEFAULT_DIRNAME_PATTERN = re.compile(
    r'^(?P<model>.+?)-swe-bench-verified-test-\d{4}-\d{2}-\d{2}-(?P<rep>\d+)of\d+$'
)


def default_parser(dirname):
    """Parse a run directory name into (model_id, replicate). Returns None to skip."""
    m = DEFAULT_DIRNAME_PATTERN.match(dirname)
    if m is None:
        return None
    return m.group('model'), int(m.group('rep')) - 1


def _as_dict(x, fallback_key='raw'):
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    if isinstance(x, str):
        try:
            parsed = json.loads(x)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return {fallback_key: x}
    return {fallback_key: str(x)}


def _as_text(x):
    if x is None or isinstance(x, str):
        return x
    return json.dumps(x)


def load_langfuse_trajectory(path, model_id='', query_id=None, replicate=0,
                             max_tool_output_chars=100_000):
    """Parse one trajectory .jsonl into a Trace.

    Tool outputs are truncated at load (max_tool_output_chars) so pathological
    multi-MB outputs never sit in memory corpus-wide; the pre-truncation size is
    kept on the step via metadata-free convention (output ends with a marker).
    """
    if query_id is None:
        query_id = os.path.basename(path).replace('_trajectory.jsonl', '')

    spans = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    spans.sort(key=lambda d: d.get('t_index', 0))

    trace = Trace(model_id=model_id, query_id=query_id, replicate=replicate)
    trace.metadata['path'] = path

    pending_text = None      # assistant text from the most recent GENERATION
    pending_tokens = (None, None)
    n_parse_fallbacks = 0

    for span in spans:
        obs_type = span.get('obs_type')
        action = span.get('action') or {}
        result = span.get('result') or {}
        output = (result.get('output') or {})

        if obs_type == 'AGENT':
            trace.final_output = _as_text(output.get('data'))
            attrs = ((span.get('otel') or {}).get('span_attributes') or {})
            for k, v in attrs.items():
                if k.startswith(_METADATA_PREFIX):
                    trace.metadata[k[len(_METADATA_PREFIX):]] = v
            trace.exit_status = trace.metadata.get('exit_status')

        elif obs_type == 'GENERATION':
            llm = span.get('llm') or {}
            pending_text = llm.get('response_text')
            pending_tokens = (llm.get('prompt_tokens'), llm.get('completion_tokens'))
            tool_calls = llm.get('tool_calls') or []
            if not isinstance(tool_calls, list):
                tool_calls = []
                n_parse_fallbacks += 1
            if pending_text and not tool_calls:
                trace.steps.append(Step(
                    index=len(trace.steps),
                    assistant_text=pending_text,
                    tokens_in=pending_tokens[0],
                    tokens_out=pending_tokens[1],
                ))
                pending_text = None

        elif obs_type == 'TOOL':
            args = _as_dict((action.get('input') or {}).get('data'))
            if 'raw' in args and len(args) == 1:
                n_parse_fallbacks += 1
            out_text = _as_text(output.get('data'))
            n_chars = len(out_text) if out_text else 0
            if out_text and n_chars > max_tool_output_chars:
                out_text = (out_text[:max_tool_output_chars]
                            + f'\n...[truncated, {n_chars} chars total]')
            status = result.get('status') or {}
            trace.steps.append(Step(
                index=len(trace.steps),
                assistant_text=pending_text,
                tool_name=action.get('name'),
                tool_args=args,
                tool_output=out_text,
                tool_success=status.get('success'),
                tokens_in=pending_tokens[0],
                tokens_out=pending_tokens[1],
            ))
            pending_text = None
            pending_tokens = (None, None)

    trace.metadata['n_parse_fallbacks'] = n_parse_fallbacks
    return trace


def load_langfuse_run_dir(run_dir, model_id, replicate, **kwargs):
    """Load all trajectories in one run directory. Returns {query_id: Trace}."""
    out = {}
    for path in sorted(glob(os.path.join(run_dir, '*_trajectory.jsonl'))):
        trace = load_langfuse_trajectory(path, model_id=model_id, replicate=replicate, **kwargs)
        out[trace.query_id] = trace
    return out


def load_langfuse_corpus(root, dirname_parser=default_parser, verbose=True, **kwargs):
    """Load every run directory under root that dirname_parser recognizes.

    Returns a flat list of Traces.
    """
    traces = []
    skipped = []
    for dirname in sorted(os.listdir(root)):
        run_dir = os.path.join(root, dirname)
        if not os.path.isdir(run_dir):
            continue
        parsed = dirname_parser(dirname)
        if parsed is None:
            skipped.append(dirname)
            continue
        model_id, replicate = parsed
        traces.extend(load_langfuse_run_dir(run_dir, model_id, replicate, **kwargs).values())
    if verbose and skipped:
        print(f'load_langfuse_corpus: skipped {len(skipped)} unrecognized dirs: {skipped[:5]}')
    return traces
