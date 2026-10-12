"""Selective Context baseline (Li et al., 2023, "Compressing Context to Enhance Inference Efficiency of
Large Language Models"): the self-information of a unit of text under a causal language model,
−Σ log p(token | everything before it), summed over the unit's tokens; the least informative units are
deleted first.

Here the units are the response's fragments and the language model reads the request as the user turn
and the response as its own reply (its chat template), so a fragment's self-information is conditioned
on the request and on the fragments before it. drybench.compare deletes, from the least informative
fragment up, as many characters as the filler map deletes on the same responses (equal budget).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from drybench.compare import locate


def fragment_self_information(backend, request: str, response: str, fragments: Sequence[str]) -> list[dict]:
    """[{self_info (nats, summed over the fragment's tokens), tokens}] per fragment. A token belongs to
    the fragment its middle character falls in (whitespace between fragments: the next fragment)."""
    if not fragments:
        return []
    ctx = backend.format_chat([{"role": "user", "content": request}])
    lps, offsets = backend.token_logprobs(ctx, response)
    bounds = locate(response, fragments)
    info = np.zeros(len(fragments))
    count = np.zeros(len(fragments), int)
    starts = np.array([a for a, _ in bounds])
    for lp, (a, b) in zip(np.asarray(lps, float), offsets):
        mid = (a + b) / 2.0
        k = int(np.searchsorted(starts, mid, side="right")) - 1
        k = min(max(k, 0), len(fragments) - 1)
        if mid >= bounds[k][1] and k + 1 < len(fragments):
            k += 1
        info[k] -= lp
        count[k] += 1
    return [{"self_info": float(x), "tokens": int(n)} for x, n in zip(info, count)]
