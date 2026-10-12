import math
import random

import numpy as np
import pandas as pd
import pytest

from drybench.attribution import attribute_spans, attribution_shares
from drybench.backends.base import entropy_from_logits, log_softmax, split_context_continuation, \
    token_budget_batches
from drybench.backends.mock import MockBackend
from drybench.entropy import (answer_entropy_first_token, answer_entropy_sampled, empirical_entropy,
                              normalize_answer, response_span_entropies, span_mean_entropy)
from drybench.items import make_item
from drybench.types import Span


def test_entropy_from_logits():
    assert entropy_from_logits(np.zeros(8)) == pytest.approx(math.log(8))
    p = np.array([0.7, 0.2, 0.1])
    assert entropy_from_logits(np.log(p)) == pytest.approx(-(p * np.log(p)).sum())
    big = np.array([[1000.0, 0.0], [0.0, 0.0]])
    assert np.allclose(entropy_from_logits(big), [0.0, math.log(2)], atol=1e-9)
    assert np.allclose(np.exp(log_softmax(np.array([1.0, 2.0, 3.0]))).sum(), 1.0)


def test_split_context_continuation_moves_trailing_spaces():
    enc = lambda s: [ord(c) for c in s]  # char-level toy tokenizer
    ctx, cont, moved = split_context_continuation(enc, "Answer: ", "Paris")
    assert moved == 1 and ctx == enc("Answer:") and cont == enc(" Paris")
    ctx, cont, moved = split_context_continuation(enc, "assistant\n", "Paris")
    assert moved == 0 and cont == enc("Paris")


def test_token_budget_batches_cover_everything_once():
    lengths = [5, 50, 7, 30, 30, 1, 64]
    batches = token_budget_batches(lengths, 100)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(len(lengths)))
    for b in batches:
        assert len(b) == 1 or max(lengths[i] for i in b) * len(b) <= 100


def test_span_mean_entropy_assigns_tokens_by_midpoint():
    text = "Great question! It is Paris."
    offsets = [(0, 5), (6, 15), (16, 18), (19, 21), (22, 28)]
    ents = np.array([0.1, 0.3, 1.0, 2.0, 3.0])
    spans = [Span(0, 15, "Great question!"), Span(16, 28, "It is Paris."), Span(28, 28, "")]
    out = span_mean_entropy(ents, offsets, spans)
    assert out[0] == pytest.approx(0.2) and out[1] == pytest.approx(2.0) and math.isnan(out[2])


def test_mock_span_entropies_separate_formulaic_words():
    backend = MockBackend()
    text = "Great question! Ostrell Freight."
    spans = [Span(0, 15, "Great question!"), Span(16, 32, "Ostrell Freight.")]
    h = response_span_entropies(backend, "ctx", text, spans)
    assert h[0] < h[1]


def test_normalize_and_empirical_entropy():
    assert normalize_answer("The  Ostrell Freight.\nextra") == "ostrell freight"
    assert empirical_entropy(["a", "a", "a"]) == 0.0
    assert empirical_entropy(["a", "b"]) == pytest.approx(math.log(2))


def test_answer_entropies_with_mock():
    it = make_item(random.Random(0), "x", k=2)
    backend = MockBackend(generate_fn=lambda ctx: "Ostrell Freight")
    first = answer_entropy_first_token(backend, it)
    assert len(first) == 2 and all(v >= 0 for v in first)
    sampled = answer_entropy_sampled(backend, it, n=5)
    assert sampled == [0.0, 0.0]


def test_attribution_cells_and_shares():
    df = pd.DataFrame({
        "model": ["m"] * 4,
        "H_ans_pct": [0.1, 0.1, 0.9, 0.9],
        "h_span_pct": [0.1, 0.9, 0.1, 0.9],
        "tokens": [10, 20, 30, 40],
    })
    df["filler_source"] = attribute_spans(df)
    assert df["filler_source"].tolist() == ["autopilot", "over_elaboration", "template_hedge", "compensation"]
    shares = attribution_shares(df)
    assert shares.loc["m"].sum() == pytest.approx(1.0)
    assert shares.loc["m", "compensation"] == pytest.approx(0.4)
