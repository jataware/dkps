"""Canonical text renderings of steps and outcomes for off-the-shelf embedders.

The guiding principle: embed short, structured, information-dense strings
rather than raw dumps. Bulk tool output is mostly noise (file listings, source
dumps); errors, commands, and diffs are signal. Truncation always keeps head
and tail -- conclusions and errors concentrate at the end.
"""
from __future__ import annotations

import re

NO_PATCH = 'NO PATCH SUBMITTED'


def head_tail(text, max_chars=2000, tail_frac=0.25):
    """Truncate to max_chars keeping the head and the tail with an omission marker."""
    if text is None:
        return ''
    if len(text) <= max_chars:
        return text
    n_tail = int(max_chars * tail_frac)
    n_head = max_chars - n_tail
    omitted = len(text) - n_head - n_tail
    return f'{text[:n_head]}\n...[{omitted} chars omitted]...\n{text[-n_tail:] if n_tail else ""}'


def canonicalize_args(tool_name, args, max_chars=500):
    """Render a tool call as a compact one-liner, most informative fields first."""
    args = args or {}
    if tool_name == 'bash':
        body = str(args.get('command', ''))
    elif tool_name == 'str_replace_editor':
        parts = [str(args.get('command', '')), str(args.get('path', ''))]
        for k in ('old_str', 'new_str', 'file_text', 'view_range', 'insert_line'):
            if k in args:
                parts.append(f'{k}={args[k]}')
        body = ' '.join(p for p in parts if p)
    else:
        body = ' '.join(f'{k}={v}' for k, v in args.items())
    return head_tail(f'{tool_name}: {body}', max_chars)


def summarize_result(step, max_error_chars=200):
    """One-line summary of a tool result: outcome + size, error text if failed."""
    if step.tool_name is None:
        return ''
    out = step.tool_output or ''
    if step.tool_success is False:
        return f'-> error: {head_tail(out, max_error_chars, tail_frac=0.5)}'
    return f'-> ok, {len(out)} chars'


def canonicalize_step(step, max_chars=1000, include_tool_output_chars=0):
    """Canonical text for one step: assistant text + tool call + result summary."""
    parts = []
    if step.assistant_text:
        parts.append(step.assistant_text)
    if step.tool_name is not None:
        parts.append(canonicalize_args(step.tool_name, step.tool_args))
        parts.append(summarize_result(step))
        if include_tool_output_chars and step.tool_output:
            parts.append(head_tail(step.tool_output, include_tool_output_chars))
    return head_tail('\n'.join(parts), max_chars)


_DIFF_FILE_RE = re.compile(r'^diff --git a/(\S+) b/(\S+)', re.M)
_HUNK_RE = re.compile(r'^@@', re.M)


def diff_stats(diff_text):
    """Cheap stats from a unified diff: files touched, hunks, +/- line counts."""
    if not diff_text:
        return {'n_files': 0, 'n_hunks': 0, 'plus_lines': 0, 'minus_lines': 0,
                'paths': [], 'size_bytes': 0}
    paths = [m.group(2) for m in _DIFF_FILE_RE.finditer(diff_text)]
    plus = minus = 0
    for line in diff_text.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            plus += 1
        elif line.startswith('-') and not line.startswith('---'):
            minus += 1
    return {'n_files': len(paths), 'n_hunks': len(_HUNK_RE.findall(diff_text)),
            'plus_lines': plus, 'minus_lines': minus, 'paths': paths,
            'size_bytes': len(diff_text)}


def canonicalize_diff(diff_text, max_chars=4000):
    """Header synthesized from diff stats + head/tail of the diff itself."""
    if not diff_text or not diff_text.strip():
        return NO_PATCH
    stats = diff_stats(diff_text)
    header = (f'Patch: {stats["n_files"]} files, {stats["n_hunks"]} hunks, '
              f'+{stats["plus_lines"]}/-{stats["minus_lines"]} lines. '
              f'Files: {", ".join(stats["paths"][:20])}')
    return header + '\n' + head_tail(diff_text, max_chars - min(len(header), 500))
