"""Shannon-entropy signals of the evaluated model.

H_ans    item level: entropy of the model's short answer to each probe, computed with
         the same report visible (does the model know the answer?). Two estimators:
         first-token entropy (one forward pass) and the entropy of the empirical
         distribution of N sampled normalized answers.
h_span   span level: mean teacher-forced next-token entropy of the model over the
         tokens of a response span (is the model on autopilot?).

Entropies live on model-specific scales; compare models via within-model percentiles.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from drybench.backends.base import LMBackend
from drybench.items import short_answer_messages
from drybench.types import Item, Span

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    """Lowercase, first line only, strip punctuation and articles, collapse spaces."""
    text = (text.strip().splitlines() or [""])[0].lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def empirical_entropy(values: Sequence[str]) -> float:
    counts = Counter(values)
    n = sum(counts.values())
    return float(-sum(c / n * math.log(c / n) for c in counts.values())) if n else float("nan")


def span_mean_entropy(token_entropies: np.ndarray, token_offsets: Sequence[tuple[int, int]],
                      spans: Sequence[Span]) -> np.ndarray:
    """Mean entropy of the tokens whose midpoint falls inside each span (NaN if none)."""
    mids = np.array([(a + b) / 2 for a, b in token_offsets]) if len(token_offsets) else np.zeros(0)
    out = np.full(len(spans), np.nan)
    for i, s in enumerate(spans):
        sel = (mids >= s.start) & (mids < s.end)
        if sel.any():
            out[i] = float(np.mean(np.asarray(token_entropies)[sel]))
    return out


def response_span_entropies(backend: LMBackend, context: str, response: str,
                            spans: Sequence[Span]) -> np.ndarray:
    """h_span for every span of `response`, teacher-forced after `context`."""
    ents, offs = backend.token_entropies(context, response)
    return span_mean_entropy(ents, offs, spans)


def answer_entropy_first_token(backend: LMBackend, item: Item) -> list[float]:
    """First-token entropy of the short answer for each required probe."""
    ctxs = [backend.format_chat(short_answer_messages(item, p)) for p in item.required_probes]
    return backend.next_token_entropy(ctxs).tolist()


def answer_entropy_sampled(backend: LMBackend, item: Item, n: int = 10, max_new_tokens: int = 16,
                           temperature: float = 1.0, seed: int = 0,
                           normalize: Callable[[str], str] = normalize_answer) -> list[float]:
    """Entropy of the empirical distribution of n sampled (normalized) short answers per probe."""
    out = []
    for j, p in enumerate(item.required_probes):
        ctx = backend.format_chat(short_answer_messages(item, p))
        gen = getattr(backend, "generate_batch", None)
        if gen is not None:
            samples = gen([ctx] * n, max_new_tokens=max_new_tokens, temperature=temperature,
                          seed=seed + j)
        else:
            samples = [backend.generate(ctx, max_new_tokens, temperature, seed=seed + j * n + s)
                       for s in range(n)]
        out.append(empirical_entropy([normalize(s) for s in samples]))
    return out


def item_answer_entropy(values: Sequence[float], reduce: str = "mean") -> float:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals) if reduce == "mean" else np.max(vals))


def within_group_percentile(df: pd.DataFrame, col: str, group: str = "model") -> pd.Series:
    """Percentile rank of `col` within each group (0..1, average ranks for ties)."""
    return df.groupby(group)[col].rank(pct=True, method="average")
