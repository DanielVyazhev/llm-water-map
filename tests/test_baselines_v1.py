"""v1.0: the LLM judge and Selective Context baselines, the comparison of every method on one truth set,
the shared guidelines and the labelling page."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from drybench.backends.mock import MockBackend
from drybench.cli import main as cli
from drybench.compare import (compare, equal_budget, fill_unscored, locate, partial_spearman, planted_share, planted_truth,
                              shorter_pairs)
from drybench.guidelines import LABELS, guidelines_text, page_js
from drybench.judge import SYSTEM, judge_fragments, judge_messages
from drybench.report import synthetic_tables
from drybench.selective import fragment_self_information

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- guidelines and the page

def test_guidelines_text_and_judge_variant():
    full, judge = guidelines_text(), guidelines_text(judge=True)
    for lab in LABELS:
        assert f'"{lab}"' in full and f'"{lab}"' in judge
    assert "also_needed" in full and "unsure" in full
    assert "also_needed" not in judge and "unsure" not in judge and "Explanations" in judge
    assert "DRY" not in full and "DRY" not in SYSTEM


def test_labelling_page_carries_the_guidelines():
    page = (ROOT / "annotation" / "dry_span_labels.html").read_text(encoding="utf-8")
    assert page_js() in page, "run scripts/build_annotation_page.py"
    assert _load_script("build_annotation_page").main(["--check"]) == 0
    assert "<title>Span Labels</title>" in page and "DRY" not in page.split("// GUIDE:START")[0].split("<script>")[0]


# ---------------------------------------------------------------- judge and Selective Context

def test_judge_labels_each_fragment_with_probabilities():
    frags = ["Great question!", "The carrier was Dunmar Freight.", "It left on 4 May WRONG."]
    res = judge_fragments(MockBackend(), "Shipment KR-4417 was handled by Dunmar Freight.", "Carrier?", frags)
    assert [r["label"] for r in res] == ["filler", "needed", "wrong"]
    for r in res:
        assert abs(sum(r["p"]) - 1) < 1e-9 and max(r["p"]) == r["p"][LABELS.index(r["label"])]
    msgs = judge_messages("doc", "req", frags, 1, "Contract excerpt")
    assert "Contract excerpt:\ndoc" in msgs[1]["content"] and "fragment [2]" in msgs[1]["content"]
    assert "[3] It left on 4 May WRONG." in msgs[1]["content"]


def test_selective_context_self_information_per_fragment():
    resp = "Great question! The carrier was Dunmar Freight.\nI hope this helps!"
    frags = ["Great question!", "The carrier was Dunmar Freight.", "I hope this helps!"]
    res = fragment_self_information(MockBackend(), "Carrier?", resp, frags)
    assert [r["tokens"] for r in res] == [2, 5, 4]
    assert res[1]["self_info"] > res[0]["self_info"] and res[1]["self_info"] > res[2]["self_info"]
    assert locate("a  b c", ["a b", "c"]) == [(0, 4), (5, 6)]


# ---------------------------------------------------------------- comparison

def test_planted_truth_repeats_are_ambiguous():
    labels = ["filler:preamble", "fact", "fact", "filler:summary", "wrong"]
    facts = [[], ["f1"], ["f2"], ["f1", "f2"], []]
    assert planted_truth(labels, facts) == ["filler", None, None, None, "wrong"]
    assert planted_truth(labels[:3] + ["filler:closing", "wrong"], facts[:3] + [[], []]) == \
        ["filler", "needed", "needed", "filler", "wrong"]
    assert planted_truth(labels) == ["filler", None, None, None, "wrong"]  # rows before v1.0: no facts per fragment
    # the true share counts planted filler (the summary too: a later repeat) and wrong statements
    assert planted_share(labels, [5, 20, 20, 10, 5]) == pytest.approx(20 / 60)


def test_pairs_and_partial_correlation():
    truth, length = [0.8, 0.1, 0.5], [10, 50, 20]
    assert shorter_pairs(truth, length, length)["right"] == 0.0  # length never ranks the shorter one higher
    r = shorter_pairs(truth, length, truth)
    assert r["right"] == 1.0 and r["pairs"] == 3
    assert shorter_pairs(truth, length, truth, groups=["a", "b", "a"])["pairs"] == 1
    rng = np.random.default_rng(0)
    z = rng.normal(size=300)
    x, y = z + rng.normal(size=300), z + rng.normal(size=300)
    assert abs(partial_spearman(x, y, z)) < 0.15 < np.corrcoef(x, y)[0, 1]


def _case(i, truth, length, removable, score):
    return {"id": f"c{i}", "group": f"g{i % 2}", "chars": [10] * len(truth), "truth": truth, "counted": [True] * len(truth),
            "true_share": sum(t != "needed" for t in truth) / len(truth), "length": length, "yap": length / 2,
            "stopwords": 0.3, "methods": {"map:main": {"removable": removable, "score": [float(x) for x in removable]},
                                          "selective-x": {"removable": None, "score": score}}}


def test_compare_scores_every_method_on_the_same_cases():
    cases = [_case(0, ["needed", "filler"], 30, [False, True], [-5.0, -1.0]),
             _case(1, ["needed", "needed", "filler"], 60, [False, False, True], [-6.0, -4.0, -2.0]),
             _case(2, ["needed"], 10, [False], [-3.0]),
             _case(3, ["filler", "filler", "needed", "wrong"], 40, [True, True, False, True], [-1.0, -1.0, -7.0, -2.0])]
    methods = [("map", "map:main"), ("sc", "selective-x"), ("length", "length")]
    t, info = compare(cases, methods, budget_from="map:main", n_boot=0)
    assert info["responses"] == 4 and info["budget"] == pytest.approx(5 / 10)
    assert t.loc["map", "fragments: F1"] == 1.0 and t.loc["map", "shorter but more removable: ranked right"] == 1.0
    assert t.loc["sc", "fragments: AUC"] == 1.0  # lower self-information = more removable, ranks perfectly here
    assert t.loc["length", "shorter but more removable: ranked right"] == 0.0
    assert equal_budget(cases, "selective-x", 0.5) == -2.0
    cases[0]["methods"].pop("selective-x")  # a method that does not cover a case: the case is left out for all
    t2, info2 = compare(cases, methods, budget_from="map:main", n_boot=0)
    assert info2["responses"] == 3 and info2["of"] == 4 and info2["coverage"] == {"map": 4, "sc": 3}
    cases[1]["methods"].pop("map:main")  # the reader read nothing from case 1: with fill_unscored all of it is removable
    cases[1]["status"] = {"map:main": "no_signal"}
    cases[2]["methods"].pop("map:main")  # the reader cannot read case 2's gold: it stays out
    cases[2]["status"] = {"map:main": "reader_fail"}
    filled = fill_unscored(cases, ["map:main"])
    assert filled[1]["methods"]["map:main"]["removable"] == [True, True, True] and "map:main" not in cases[1]["methods"]
    assert "map:main" not in filled[2]["methods"]
    t3, info3 = compare(filled, [("map", "map:main"), ("length", "length")], n_boot=50)
    assert info3["responses"] == 3 and t3.loc["map", "fragments: F1"] < 1.0


# ---------------------------------------------------------------- backends

def test_hf_choice_and_token_logprobs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from drybench.backends.hf import HFBackend
    from drybench.items import generate_items
    import tempfile

    d = Path(tempfile.mkdtemp())
    corpus = [it.document + "\n" + it.request for it in generate_items(2, seed=1)] + ["needed filler wrong"] * 3
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=300, min_frequency=1, special_tokens=["<|endoftext|>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|endoftext|>")
    tok.save_pretrained(d)
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=256, n_embd=32, n_layer=2, n_head=2,
                               bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)).save_pretrained(d)
    be = HFBackend(str(d), device="cpu", max_batch_tokens=64)
    ctx, cont = "Shipment KR-4417 was handled by", " Dunmar Freight. It left."
    lps, offs = be.token_logprobs(ctx, cont)
    assert lps.sum() == pytest.approx(be.continuation_logprobs([(ctx, cont)])[0], abs=1e-4)
    assert offs[0][0] == 0 and offs[-1][1] == len(cont)
    opts = [" needed", " filler", " wrong"]
    got = be.choice_logprobs([ctx, ctx + " the"], opts)
    heads = [be.encode(ctx + o)[len(be.encode(ctx))] for o in opts]
    if len(set(heads)) == len(heads):  # one pass per context: first-token log-probabilities
        ids = torch.tensor([be._start_ids(ctx, be.encode(ctx))])
        lp = torch.log_softmax(be.model(ids).logits[0, -1].float(), -1)
        assert got[0] == pytest.approx([lp[h].item() for h in heads], abs=1e-4)
    same = be.choice_logprobs([ctx], [" Dunmar", " Dunmar Freight"])  # shared first token: whole continuations
    assert same[0] == pytest.approx(be.continuation_logprobs([(ctx, " Dunmar"), (ctx, " Dunmar Freight")]), abs=1e-4)


# ---------------------------------------------------------------- end to end on a mock run

def _synthetic_run(run: Path) -> None:
    items, resp = run / "items.jsonl", run / "responses.jsonl"
    cli(["items", "--out", str(items), "--n-per-cell", "1"])
    cli(["generate", "--items", str(items), "--out", str(resp), "--backend", "mock", "--model-name", "m1",
         "--variant", "default"])
    (run / "reader2_mock").mkdir()
    for sub in (run, run / "reader2_mock"):
        cli(["score", "--items", str(items), "--responses", str(resp), "--out", str(sub / "scored.jsonl"),
             "--reader-backend", "mock", "--mode", "loo"])
    run_real = _load_script("run_real")
    run_real.filler_check(argparse.Namespace(filler_check_n=1, backend="mock"), "unused", run / "filler_check", "auto")
    run_real.filler_check(argparse.Namespace(filler_check_n=1, backend="mock"), "unused",
                         run / "reader2_mock" / "filler_check", "auto")


def test_baselines_cli_and_tables(tmp_path):
    _synthetic_run(tmp_path)
    for method in ("judge", "selective"):
        cli(["baselines", "--run", str(tmp_path), "--set", "planted_synthetic", "--method", method, "--label", "mock",
             "--backend", "mock"])
    rows = [json.loads(x) for x in (tmp_path / "baselines" / "judge-mock" / "planted_synthetic.jsonl").read_text().splitlines()]
    assert rows and all(len(r["labels"]) == len(r["p"]) and r["seconds"] >= 0 for r in rows)
    n = len(rows)
    cli(["baselines", "--run", str(tmp_path), "--set", "planted_synthetic", "--method", "judge", "--label", "mock",
         "--backend", "mock"])  # resumes: nothing new
    assert len((tmp_path / "baselines" / "judge-mock" / "planted_synthetic.jsonl").read_text().splitlines()) == n
    scored = [json.loads(x) for x in (tmp_path / "scored.jsonl").read_text().splitlines()]
    assert all("seconds" in r for r in scored)
    text = synthetic_tables(tmp_path)
    for row in ("filler map (reader main)", "filler map (reader mock)", "LLM judge (mock)", "Selective Context (mock)",
                "shorter but more removable", "partial Spearman given length", "## T8. Cost per response"):
        assert row in text, row
    assert "DRY" not in text


def test_build_annotation_page_check_fails_on_a_stale_page(tmp_path, monkeypatch):
    mod = _load_script("build_annotation_page")
    page = (ROOT / "annotation" / "dry_span_labels.html").read_text(encoding="utf-8")
    stale = tmp_path / "p.html"
    stale.write_text(page.replace("Tricky cases", "Old cases"), encoding="utf-8")
    monkeypatch.setattr(mod, "PAGE", stale)
    monkeypatch.setattr(mod, "OFFLINE", tmp_path / "o.html")
    assert mod.main(["--check"]) == 1
    assert mod.main([]) == 0 and mod.main(["--check"]) == 0
    assert "Tricky cases" in stale.read_text(encoding="utf-8")


def test_chat_annotators_prompt_uses_the_guidelines(tmp_path):
    mod = _load_script("chat_annotators")
    text = mod.prompt()
    assert guidelines_text() in text and '"uid": "example"' in text
    sample = tmp_path / "sample.jsonl"
    sample.write_text(json.dumps({"uid": "r001", "document": "d", "request": "q", "response": "A. B.",
                                  "spans": [{"i": 0, "text": "A."}, {"i": 1, "text": "B."}]}) + "\n")
    reply = tmp_path / "gpt_1.txt"
    reply.write_text('Sure ```json [{"uid": "r001", "labels": [{"i": 0, "label": "needed"}, {"i": 1, "label": "filler"}]}]```')
    out = tmp_path / "gpt.jsonl"
    mod.main(["collect", str(sample), str(reply), "--model", "gpt", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert rec["labels"] == ["needed", "filler"] and rec["complete"]


def test_check_abstain_csv_mode_runs(tmp_path, capsys):
    from drybench.annotate import build_abstain, write_abstain

    _synthetic_run(tmp_path)
    rows, key, stats = build_abstain([tmp_path], n=5)
    assert rows and stats["answers"] == len(rows) <= 5
    write_abstain(tmp_path / "ann", rows, key, stats)
    import csv

    csv_path = tmp_path / "ann" / "abstain_sample.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    for r in got:
        r["label (A/B/V/O)"] = "v"  # case and spaces do not matter
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(got[0]))
        w.writeheader()
        w.writerows(got)
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_abstain.py"), "--csv", str(csv_path)],
                         capture_output=True, text=True, check=True)
    assert f"{len(got)} labelled answers" in out.stdout


def test_judge_reads_capitalised_labels_too():
    import math

    class Capital(MockBackend):
        """A judge that answers 'Filler' (capitalised), as the guidelines write the labels."""

        def continuation_logprobs(self, requests):
            return np.array([math.log(0.7 if c.strip() == "Filler" else 0.004 if c.strip() == "needed" else 0.002)
                             for _, c in requests])

    res = judge_fragments(Capital(), "doc", "req", ["Sure!"])
    assert res[0]["label"] == "filler" and len(res[0]["lp"]) == 6
    assert res[0]["mass"] == pytest.approx(0.7 + 0.004 + 4 * 0.002)
