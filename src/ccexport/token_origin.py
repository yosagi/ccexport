# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 @yosagi
"""Per-origin token estimation (thinking / text / tool_use / prompt / tool_result).

Approximates how each API call's `output_tokens` and per-turn input growth
break down by content origin. Estimates are derived as follows:

- ``text``, ``tool_use``, ``prompt``, ``tool_result``: tiktoken cl100k_base
  applied to the relevant strings. Absolute values are *approximate* (English
  and code are typically under-counted by 20-30%, long Japanese is slightly
  over-counted) but relative comparisons within a session hold up well.

- ``thinking``: signature blob char count multiplied by 0.29 (empirical
  calibration, see ``scratch/token_calib.py`` and 2026-04-25 work history),
  plus tiktoken estimate of any plaintext ``thinking`` field. Three of four
  calibration sessions converged on alpha = 0.29 with R-squared > 0.99; the
  outlier (alpha = 0.26) is consistent with the tiktoken English/code
  under-count bias rather than a different signature density.

Only the *new* JSONL format (one ``output_tokens`` value shared across all
content blocks of a single API call) is supported. The legacy format, where
each block has its own ``output_tokens``, is detected and skipped.
"""

import json
from typing import Any, Dict, Iterable, List, Optional

try:
    import tiktoken
    _ENC = None  # lazy

    def _get_encoder():
        global _ENC
        if _ENC is None:
            _ENC = tiktoken.get_encoding('cl100k_base')
        return _ENC

except ImportError:  # pragma: no cover - tiktoken is a hard dependency
    def _get_encoder():
        return None


# Calibrated factor: thinking signature chars -> tokens (origin-passing fit).
THINKING_SIG_TOKENS_PER_CHAR = 0.29


def count_text_tokens(text: str) -> int:
    """Count tokens in a plain text string using cl100k_base."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is None:
        # Crude fallback if tiktoken unavailable: roughly 4 chars/token for
        # English-leaning text. This branch should not normally fire because
        # tiktoken is in the package's hard dependencies.
        return max(1, len(text) // 4)
    return len(enc.encode(text))


def count_tool_use_tokens(name: str, input_dict: Any) -> int:
    """Approximate tokens for a tool_use content block.

    Counts the serialized JSON of ``input`` plus the tool name. The actual
    wire format includes JSON keys, brackets, and the ``"type": "tool_use"``
    envelope — we approximate by the serialized payload length, which is the
    dominant term.
    """
    serialized = ''
    if input_dict is not None:
        try:
            serialized = json.dumps(input_dict, ensure_ascii=False)
        except (TypeError, ValueError):
            serialized = str(input_dict)
    return count_text_tokens(serialized) + count_text_tokens(name or '')


def estimate_thinking_tokens(signature_chars: int, thinking_text: str = '') -> int:
    """Estimate tokens contributed by a thinking block.

    ``signature`` is an opaque encrypted blob in the JSONL; we model it as
    ``signature_chars * 0.29``. If a plaintext ``thinking`` field is present
    (rare in practice — most production sessions only carry the signature),
    it is counted via tiktoken and added.
    """
    sig_tokens = int(round(signature_chars * THINKING_SIG_TOKENS_PER_CHAR))
    text_tokens = count_text_tokens(thinking_text) if thinking_text else 0
    return max(0, sig_tokens + text_tokens)


def compute_assistant_origin(content_items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Sum origin-categorized output tokens for one assistant message.

    Returns a dict with keys ``thinking``, ``text``, ``tool_use``. The sum
    should approximate the API's ``output_tokens`` for that message, modulo
    tiktoken's over/under-count bias.
    """
    thinking = 0
    text = 0
    tool_use = 0
    for ci in content_items or []:
        ctype = ci.get('type')
        if ctype == 'thinking':
            thinking += estimate_thinking_tokens(
                ci.get('signature_chars', 0),
                ci.get('text', ''),
            )
        elif ctype == 'text':
            text += count_text_tokens(ci.get('content', ''))
        elif ctype == 'tool_use':
            tool_use += count_tool_use_tokens(
                ci.get('name', ''),
                ci.get('input'),
            )
    return {'thinking': thinking, 'text': text, 'tool_use': tool_use}


def scale_assistant_origin(
    raw_origin: Dict[str, int], output_tokens: int,
) -> Dict[str, int]:
    """Scale tiktoken ratios so they sum to the API-reported ``output_tokens``.

    The raw tiktoken estimates are used only for relative proportions.
    Rounding remainder is assigned to ``tool_use`` (the most variable
    category, least sensitive to small rounding error).
    """
    raw_sum = sum(raw_origin.values())
    if raw_sum <= 0 or output_tokens <= 0:
        return dict(raw_origin)
    factor = output_tokens / raw_sum
    thinking = int(round(raw_origin.get('thinking', 0) * factor))
    text = int(round(raw_origin.get('text', 0) * factor))
    tool_use = output_tokens - thinking - text
    return {'thinking': thinking, 'text': text, 'tool_use': max(0, tool_use)}


def compute_user_origin(
    user_text: str,
    tool_results: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Sum origin-categorized input tokens contributed by a user turn.

    - ``prompt``: free-form user text (the typed message).
    - ``tool_result``: text content of any ``tool_result`` blocks attached
      to the same user turn.

    System prompt, memory, framework scaffolding, and image tokens are not
    included — they live in ``cache_creation`` / ``cache_read`` and are not
    decomposable from the JSONL.
    """
    prompt = count_text_tokens(user_text or '')
    tr = 0
    for item in tool_results or []:
        tr += count_text_tokens(item.get('content', '') or '')
    return {'prompt': prompt, 'tool_result': tr}


def detect_format(output_tokens_per_entry: List[int]) -> str:
    """Classify a per-message group of JSONL entries as ``new`` or ``old``.

    Older Claude Code recorded each content block on its own line with its
    own ``output_tokens`` (e.g. thinking = 9, tool_use = 184). Newer Claude
    Code repeats the same ``output_tokens`` value across every entry of a
    single API call. We use that distinction to gate origin estimation.
    """
    vals = [v for v in output_tokens_per_entry if v is not None]
    if len(vals) <= 1:
        return 'new'
    return 'new' if all(v == vals[0] for v in vals) else 'old'
