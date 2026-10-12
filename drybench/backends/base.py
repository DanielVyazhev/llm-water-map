"""Backend interface shared by the reader, the evaluated models and entropy signals.

All log-probabilities and entropies are in nats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np


def split_context_continuation(
    encode: Callable[[str], list[int]], context: str, continuation: str
) -> tuple[list[int], list[int], int]:
    """Tokenize (context, continuation) so that continuation tokens are exactly those that
    follow the context in the joint encoding (lm-eval-harness convention).

    Trailing spaces of the context are moved into the continuation ("Answer:" + " Paris").
    Newlines stay in the context (chat templates end with them).
    Returns (context_ids, continuation_ids, n_moved_chars).
    """
    n_moved = len(context) - len(context.rstrip(" "))
    if n_moved:
        continuation = context[-n_moved:] + continuation
        context = context[:-n_moved]
    whole = encode(context + continuation)
    ctx = encode(context)
    if whole[: len(ctx)] == ctx:
        return ctx, whole[len(ctx):], n_moved
    return ctx, encode(continuation), n_moved


def entropy_from_logits(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy (nats) of softmax(logits) along `axis`, numerically stable."""
    logits = np.asarray(logits, dtype=np.float64)
    m = logits.max(axis=axis, keepdims=True)
    z = logits - m
    ez = np.exp(z)
    s = ez.sum(axis=axis, keepdims=True)
    p = ez / s
    return (np.log(s) - (p * z).sum(axis=axis, keepdims=True)).squeeze(axis)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    m = logits.max(axis=axis, keepdims=True)
    z = logits - m
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def token_budget_batches(lengths: Sequence[int], max_batch_tokens: int) -> list[list[int]]:
    """Group request indices into batches (sorted by length, padded size <= budget)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and new_max * (len(cur) + 1) > max_batch_tokens:
            batches.append(cur)
            cur, new_max = [], lengths[i]
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


class LMBackend(ABC):
    """Minimal causal-LM interface used across drybench."""

    name: str = "backend"

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ...

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    @abstractmethod
    def continuation_logprobs(self, requests: Sequence[tuple[str, str]]) -> np.ndarray:
        """Sum of log p(continuation tokens | context) for each (context, continuation)."""

    def first_token_logmass(self, contexts: Sequence[str], strings: Sequence[str]) -> np.ndarray:
        """log p(the text after each context starts with the first token of one of `strings`).

        Default: sums continuation probabilities of the prefix-free closure of `strings`, which is
        exact for single-token strings. Backends with access to logits do it in one pass."""
        keep: list[str] = []
        for s in sorted(set(strings), key=len):
            if s and not any(s.startswith(b) for b in keep):
                keep.append(s)
        if not contexts or not keep:
            return np.full(len(contexts), -np.inf)
        lps = np.asarray(self.continuation_logprobs([(c, s) for c in contexts for s in keep]), float)
        return np.logaddexp.reduce(lps.reshape(len(contexts), len(keep)), axis=1)

    @abstractmethod
    def token_entropies(self, context: str, continuation: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Teacher-forced next-token entropy at every continuation token, and each token's
        (start, end) character offsets within `continuation`."""

    def token_logprobs(self, context: str, continuation: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Teacher-forced log p of every continuation token, and each token's (start, end) character
        offsets within `continuation` (self-information = minus these; drybench.selective)."""
        raise NotImplementedError(f"{type(self).__name__} does not give token log-probabilities")

    def choice_logprobs(self, contexts: Sequence[str], options: Sequence[str]) -> np.ndarray:
        """[contexts × options] log-probability of answering each context with each option.

        Default: the option's whole continuation. Backends with logits use the option's first token when
        the options start with different tokens (one pass per context); compare options per context
        (e.g. softmax over them), not across backends."""
        if not contexts or not options:
            return np.zeros((len(contexts), len(options)))
        lps = self.continuation_logprobs([(c, o) for c in contexts for o in options])
        return np.asarray(lps, float).reshape(len(contexts), len(options))

    @abstractmethod
    def next_token_entropy(self, contexts: Sequence[str]) -> np.ndarray:
        """Entropy of the next-token distribution right after each context."""

    @abstractmethod
    def generate(self, context: str, max_new_tokens: int = 256, temperature: float = 0.0,
                 seed: int | None = None) -> str:
        """Continuation text (greedy when temperature == 0)."""

    def format_chat(self, messages: list[dict[str, str]]) -> str:
        """Plain formatting for models without a chat template."""
        parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
        parts.append("Assistant:")
        return "\n\n".join(parts)

    @property
    def has_chat_template(self) -> bool:
        return False
