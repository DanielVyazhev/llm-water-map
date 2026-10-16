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


# ---------------------------------------------------------------- v1.2

def test_generation_ignores_the_models_own_settings(tiny_model_dir, tmp_path):
    """Qwen2.5-0.5B/1.5B-Instruct ship repetition_penalty 1.1, which transformers applies even without
    sampling: the backend drops it, so greedy decoding is the same for every model."""
    import json
    import shutil

    d = tmp_path / "penalised"
    shutil.copytree(tiny_model_dir, d)
    (d / "generation_config.json").write_text(json.dumps(
        {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 3.0}))
    plain = HFBackend(str(tiny_model_dir), device="cpu")
    pen = HFBackend(str(d), device="cpu")
    assert pen.describe()["dropped_generation_config"]["repetition_penalty"] == 3.0
    ctx = ["Shipment KX-1234 departed from Port Rosfen and Shipment KX-1234 departed from Port"]
    assert pen.generate_batch(ctx, max_new_tokens=12) == plain.generate_batch(ctx, max_new_tokens=12)


def test_generation_reports_tokens_and_whether_the_model_ended(backend):
    out = backend.generate_batch_info(["The carrier was", "Shipment KX-1234 departed"], max_new_tokens=4)
    for g in out:
        assert g["n_tokens"] <= 4 and isinstance(g["finished"], bool)
        if not g["finished"]:
            assert g["n_tokens"] == 4


def test_terminated_logprobs_match_the_next_token_distribution(backend):
    ctx = "Question: How many containers?\nAnswer:"
    for cont in (" 7", " 12"):
        lp = backend.continuation_logprobs([(ctx, cont)])[0]
        got = backend.terminated_logprobs([(ctx, cont)])[0]
        ids = torch.tensor([backend._start_ids(ctx, backend.encode(ctx)) + backend.encode(cont)])
        with torch.inference_mode():
            nxt = torch.log_softmax(backend.model(ids).logits[0, -1].float(), -1)
        digits = [i for i in range(len(backend.tokenizer)) if backend.tokenizer.decode([i])[:1].isdigit()]
        p_digit = torch.logsumexp(nxt[digits], 0).exp().item()
        assert got == pytest.approx(lp + math.log1p(-p_digit), abs=1e-4)


def test_probe_models_on_a_tiny_chat_model(tmp_path):
    """v1.2.3: scripts/probe_models.py loads a local model folder and runs it as an evaluated model, a reader and
    a judge, one row per folder (a folder without config.json is reported, not fatal)."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    d = tmp_path / "tiny-chat"
    corpus = [it.document + "\n" + it.request + "\n" + it.gold_minimal for it in generate_items(3, seed=1)]
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus + ["needed filler wrong Needed Filler Wrong"], vocab_size=500, min_frequency=1,
                            special_tokens=["<|endoftext|>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|endoftext|>")
    tok.chat_template = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
                         "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    tok.save_pretrained(d)
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=1024, n_embd=32, n_layer=2, n_head=2,
                               bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)).save_pretrained(d)
    (tmp_path / "empty").mkdir()
    out = tmp_path / "probe.jsonl"
    script = Path(__file__).resolve().parents[1] / "scripts" / "probe_models.py"
    r = subprocess.run([sys.executable, str(script), str(d), str(tmp_path / "empty"), "--out", str(out),
                        "--dtype", "fp32", "--sha256", str(d)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert rows[0]["loads"] and rows[0]["chat_template"] and "facts_in_text" in rows[0]["answer"]
    assert len(rows[0]["reader"]["p_gold"]) == 2 and len(rows[0]["judge"]["labels"]) == 3
    assert rows[0]["sha256"] and rows[1]["error"].startswith("no config.json")
    assert rows[0]["load_mode"] == "one GPU" and rows[0]["machine"]["transformers"] and "this machine:" in r.stdout
    assert any(line.startswith("tiny-chat |") and "GPT2LMHeadModel" in line for line in r.stdout.splitlines())


def _bpe_tokenizer(texts, vocab_size=500):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(texts, vocab_size=vocab_size, min_frequency=1, special_tokens=["<|endoftext|>"])
    return PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|endoftext|>")


def test_vision_language_model_is_used_for_text(tmp_path):
    """v1.2.3: a Mistral 3 checkpoint (image-text-to-text, as Mistral Small 3.x and Ministral 3 ship) loads through
    its image-text-to-text class and works as a text model; its chat template may sit in the folder and want the
    content as a list of parts."""
    tf = pytest.importorskip("transformers")
    if not hasattr(tf, "Mistral3ForConditionalGeneration"):
        pytest.skip("no Mistral 3 in this transformers")
    from transformers import Mistral3Config, Mistral3ForConditionalGeneration, MistralConfig, PixtralVisionConfig

    d = tmp_path / "tiny-mistral3"
    corpus = [it.document + "\n" + it.request + "\n" + it.gold_minimal for it in generate_items(3, seed=1)]
    tok = _bpe_tokenizer(corpus + ["needed filler wrong Needed Filler Wrong"])
    tok.save_pretrained(d)
    (d / "chat_template.jinja").write_text(
        "{% for m in messages %}{% if m.content is string %}{{ raise_exception('parts only') }}{% endif %}"
        "<|{{ m.role }}|>{% for p in m.content %}{{ p.text }}{% endfor %}\n{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    v = len(tok)
    text = MistralConfig(vocab_size=v + 1, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=1024,
                         bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    vision = PixtralVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                 image_size=32, patch_size=16, head_dim=16)
    torch.manual_seed(0)
    Mistral3ForConditionalGeneration(Mistral3Config(text_config=text, vision_config=vision, image_token_index=v,
                                                    spatial_merge_size=1)).save_pretrained(d)
    be = HFBackend(str(d), device="cpu", max_batch_tokens=64)
    assert be.model_class == "AutoModelForImageTextToText" and be.describe()["class"] == "AutoModelForImageTextToText"
    assert be.max_length == 1024 and be.has_chat_template
    ctx = be.format_chat([{"role": "system", "content": "Answer."}, {"role": "user", "content": "Who?"}])
    assert ctx.startswith("<|system|>Answer.") and ctx.endswith("<|assistant|>")
    lp = be.choice_logprobs([ctx], ["needed", "filler", "wrong"])
    assert lp.shape == (1, 3) and np.isfinite(lp).all()
    out = be.generate_batch_info([ctx], max_new_tokens=5)[0]
    assert out["n_tokens"] >= 1


def test_device_map_auto_places_the_model(tiny_model_dir):
    """v1.2.3: device_map="auto" (a model spread over several GPUs) puts the inputs where the embeddings are."""
    pytest.importorskip("accelerate")
    be = HFBackend(str(tiny_model_dir), device_map="auto", max_batch_tokens=64)
    assert be.device == str(be.model.get_input_embeddings().weight.device) and be.describe()["device_map"] == "auto"
    assert np.isfinite(be.continuation_logprobs([("Shipment KR-4417 was handled by", " Dunmar")])).all()


def test_probe_sizes_and_load_modes(tmp_path):
    """v1.2.3: the probe counts the weights transformers loads, not Mistral's second copy, and says how a model
    fits: one GPU, several GPUs, 8-bit weights, or not."""
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("probe", Path(__file__).resolve().parents[1] / "scripts" / "probe_models.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": ["MistralForCausalLM"], "torch_dtype": "float32",
                                              "_name_or_path": "org/model"}))
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "consolidated.safetensors"):
        (d / name).write_bytes(b"0" * 10)
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    assert [f.name for f in probe.weight_files(d)] == ["model-00001-of-00002.safetensors",
                                                       "model-00002-of-00002.safetensors"]
    w = probe.weights(d)
    assert w["origin"] == "org/model" and w["second_copy"] and w["stored_dtype"] == "float32"
    one = {"gpus": [["A100", 42.4]], "accelerate": "1", "bitsandbytes": None}
    assert probe.load_mode({"bf16_gb": 28.0}, one)[0] is None
    assert probe.load_mode({"bf16_gb": 44.5}, one)[0] == "no"
    assert probe.load_mode({"bf16_gb": 44.5}, {**one, "gpus": [["A100", 42.4]] * 2})[0] == "auto"
    assert probe.load_mode({"bf16_gb": 44.5}, {**one, "bitsandbytes": "0.45"})[0] == "8bit"


def test_tokenizer_saved_by_a_newer_transformers(tiny_model_dir, tmp_path, monkeypatch):
    """v1.2.3: a tokenizer config naming a class this transformers lacks (a folder saved by transformers 5 and read
    by 4.x: "Tokenizer class TokenizersBackend does not exist") falls back to its tokenizer.json with the config's
    special tokens."""
    import json
    import shutil

    import transformers

    d = tmp_path / "newer"
    shutil.copytree(tiny_model_dir, d)
    cfg = json.loads((d / "tokenizer_config.json").read_text())
    cfg.update(tokenizer_class="TokenizersBackend", eos_token={"content": "<|endoftext|>", "special": True})
    (d / "tokenizer_config.json").write_text(json.dumps(cfg))
    ref = HFBackend(str(tiny_model_dir), device="cpu", max_batch_tokens=64)
    plain = transformers.AutoTokenizer.from_pretrained

    def older(path, *a, **k):
        if str(path) == str(d):
            raise ValueError("Tokenizer class TokenizersBackend does not exist or is not currently imported.")
        return plain(path, *a, **k)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", older)
    be = HFBackend(str(d), device="cpu", max_batch_tokens=64)
    text = "Shipment KR-4417 was handled by Dunmar Freight."
    assert be.encode(text) == ref.encode(text) and be.tokenizer.eos_token == "<|endoftext|>"
