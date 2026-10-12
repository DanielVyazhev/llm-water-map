"""HF backend on a tiny randomly initialised GPT-2 with a freshly trained BPE tokenizer.

Checks the batching machinery (left padding, position ids, logits_to_keep) against a
plain unpadded forward pass. Needs torch + transformers; skipped otherwise.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from drybench.backends.hf import HFBackend  # noqa: E402
from drybench.items import generate_items  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny-gpt2")
    corpus = [it.document + "\n" + it.request + "\n" + it.gold_minimal for it in generate_items(3, seed=1)]
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=400, min_frequency=1, special_tokens=["<|endoftext|>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|endoftext|>")
    tok.save_pretrained(d)
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=len(tok), n_positions=512, n_embd=32, n_layer=2, n_head=2,
                     bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    GPT2LMHeadModel(cfg).save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def backend(tiny_model_dir):
    return HFBackend(str(tiny_model_dir), device="cpu", max_batch_tokens=64)


def _manual_logprob(backend, context, continuation):
    from drybench.backends.base import split_context_continuation

    ctx, cont, _ = split_context_continuation(backend.encode, context, continuation)
    ctx = backend._start_ids(context, ctx)
    ids = torch.tensor([ctx + cont])
    with torch.inference_mode():
        logits = backend.model(ids).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    return sum(lp[len(ctx) - 1 + j, t].item() for j, t in enumerate(cont))


def test_continuation_logprobs_match_unpadded_forward(backend):
    reqs = [("Shipment KX-1234 was handled by", " Ostrell Freight"),
            ("The delay of shipment", " was caused by a customs audit and a much longer tail of words"),
            ("Answer:", " Port Rosfen"),
            ("", "Hello")]
    got = backend.continuation_logprobs(reqs)  # budget 64 forces several padded batches
    for (c, k), g in zip(reqs, got):
        assert g == pytest.approx(_manual_logprob(backend, c, k), abs=1e-4)


def test_token_entropies_offsets_and_values(backend):
    ctx, cont = "Report:", " The carrier was Ostrell Freight."
    ents, offs = backend.token_entropies(ctx, cont)
    assert len(ents) == len(offs) > 0
    assert all(0 <= a <= b <= len(cont) for a, b in offs)
    assert offs[-1][1] == len(cont)
    assert np.all(ents >= 0) and np.all(ents <= math.log(backend.model.config.vocab_size) + 1e-6)


def test_next_token_entropy_matches_manual(backend):
    ctxs = ["The carrier was", "Shipment KX-1234 departed from Port"]
    got = backend.next_token_entropy(ctxs)
    for c, g in zip(ctxs, got):
        ids = torch.tensor([backend._start_ids(c, backend.encode(c))])
        with torch.inference_mode():
            logits = backend.model(ids).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        assert g == pytest.approx(-(p * torch.log(p)).sum().item(), abs=1e-4)


def test_batched_greedy_generation_matches_single(backend):
    ctxs = ["The carrier was", "Shipment KX-1234 departed from Port Rosfen and"]
    batch = backend.generate_batch(ctxs, max_new_tokens=6)
    single = [backend.generate(c, max_new_tokens=6) for c in ctxs]
    assert batch == single


def test_plain_chat_format_without_template(backend):
    s = backend.format_chat([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert s.endswith("Assistant:") and "User: U" in s


def test_first_token_logmass_matches_manual_softmax(backend):
    contexts = ["Question: Who was the carrier?\nAnswer:\n", "Report:\nShipment KX-1234 left.\n",
                "Short\n"]
    strings = ["unknown", "not", "Not stated", "Ostrell Freight"]
    got = backend.first_token_logmass(contexts, strings)  # budget 64 forces several batches
    for c, g in zip(contexts, got):
        ctx = backend._start_ids(c, backend.encode(c))
        firsts = sorted({backend._prepare(c, s)[1][0] for s in strings})
        with torch.inference_mode():
            lp = torch.log_softmax(backend.model(torch.tensor([ctx])).logits[0, -1].float(), -1)
        assert g == pytest.approx(torch.logsumexp(lp[firsts], 0).item(), abs=1e-4)
    # the generic fallback agrees for single-token strings
    single = [s for s in strings if len(backend._prepare(contexts[0], s)[1]) == 1]
    if single:
        from drybench.backends.base import LMBackend

        slow = LMBackend.first_token_logmass(backend, contexts[:1], single)
        assert backend.first_token_logmass(contexts[:1], single)[0] == pytest.approx(slow[0], abs=1e-4)
