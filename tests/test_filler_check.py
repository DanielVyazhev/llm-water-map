"""The filler-map check on simulated responses: pooled metrics, the runner step and the viewer script."""

import argparse
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from drybench.cli import main as cli
from drybench.items import generate_items
from drybench.report import _filler_check_section
from drybench.simulate import simulate, filler_check_metrics

ROOT = Path(__file__).resolve().parents[1]


def _row(item, cond, labels, kept, safety=1.0, precision=1.0, W=0.0, W_true=0.0):
    return {"item_id": item, "condition": cond, "labels": labels, "loo_kept": kept, "loo_status": "ok",
            "loo_safety": safety, "loo_filler_precision": precision, "loo_W": W, "W_true": W_true,
            "loo_filler_recall": 1.0}


def test_metrics_count_planted_filler_not_empty_responses():
    rows = [
        _row("a", "dense", ["fact"], [True]),
        _row("a", "heavy", ["filler:preamble", "fact", "filler:hedge", "filler:summary", "filler:closing"],
             [False, True, True, True, False], W=0.3, W_true=0.6),
        _row("b", "light", ["filler:restatement", "fact", "filler:summary"], [False, True, False], W=0.5, W_true=0.5),
        _row("b", "incomplete", ["fact", "fact", "filler:closing"], [False, True, False], safety=0.5, precision=0.5,
             W=0.6, W_true=0.3),
    ]
    m = filler_check_metrics(rows, n_boot=50)
    assert m["n_responses"] == 4 and m["n_with_filler"] == 3 and m["n_filler_spans"] == 5
    assert m["recall_pooled"] == pytest.approx(4 / 5)            # the hedge was left in
    assert m["recall_with_filler"] == pytest.approx((2 / 3 + 1 + 1) / 3)  # the dense response does not count
    assert m["responses_with_lost_fact"] == 1 and m["no_fact_lost"] == pytest.approx(0.75)
    assert m["precision_pooled"] == pytest.approx((2 + 2 + 1) / 6)
    assert m["repeats"] == 2 and m["repeats_resolved"] == pytest.approx(0.5)  # 'a' keeps summary and facts
    assert m["recall_by_type"]["hedge"] == {"removed": 0, "planted": 1}
    assert m["W_bias"] == pytest.approx((0 - 0.3 + 0 + 0.3) / 4)
    lo, hi = m["recall_pooled_ci"]
    assert 0 <= lo <= m["recall_pooled"] <= hi <= 1


def test_simulated_filler_is_never_planted_twice():
    items = [it for it in generate_items(3, seed=2026, cells=[{"k": 1, "d": 0}, {"k": 2, "d": 4}])]
    for it in items:
        sim = simulate(it, ["distractor", "unrequested", "distractor"], rng=random.Random(0))
        texts = [s.text for s in sim.spans]
        assert len(texts) == len(set(texts))
        if not it.meta.get("distractors"):  # no other shipment: only unrequested facts about this one
            assert "filler:distractor" not in [s.label for s in sim.spans]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_filler_check_and_viewer(tmp_path):
    run_real = _load_script("run_real")
    args = argparse.Namespace(filler_check_n=1, backend="mock")
    run_real.filler_check(args, "unused", tmp_path / "filler_check", "auto")
    summary = json.loads((tmp_path / "filler_check" / "summary.json").read_text())
    assert summary["n_responses"] == 32 and summary["loo"]["check_n_responses"] == 32
    row = json.loads((tmp_path / "filler_check" / "rows.jsonl").read_text().splitlines()[0])["row"]
    assert row["request"] and len(row["span_texts"]) == len(row["labels"])
    lines = _filler_check_section([("main", tmp_path)])
    assert lines[1].startswith("## R2") and any("recall" in l for l in lines)
    out = subprocess.run([sys.executable, ROOT / "scripts" / "show_checks.py", "filler",
                          tmp_path / "filler_check" / "rows.jsonl", "--examples", "1"],
                         capture_output=True, text=True, check=True).stdout
    assert "no fact lost" in out and "truth" in out and "request:" in out


def test_viewer_on_two_readers_and_examples(tmp_path):
    items, resp = tmp_path / "items.jsonl", tmp_path / "responses.jsonl"
    cli(["items", "--out", str(items), "--n-per-cell", "1"])
    cli(["generate", "--items", str(items), "--out", str(resp), "--backend", "mock", "--model-name", "m1",
         "--variant", "default", "concise"])
    gold = {json.loads(l)["item_id"]: json.loads(l)["gold_minimal"] for l in items.read_text().splitlines()}
    rows = [json.loads(l) for l in resp.read_text().splitlines()]
    for r in rows:
        r["response"] = f"Great question! {gold[r['item_id']]} I hope this helps!"
    resp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "reader2_mock").mkdir()
    for out in (tmp_path / "scored.jsonl", tmp_path / "reader2_mock" / "scored.jsonl"):
        cli(["score", "--items", str(items), "--responses", str(resp), "--out", str(out),
             "--reader-backend", "mock", "--mode", "loo"])
    out = subprocess.run([sys.executable, ROOT / "scripts" / "show_checks.py", "readers", tmp_path,
                          "--examples", "1"], capture_output=True, text=True, check=True).stdout
    assert "Kendall τ" in out and "same decision" in out and "Examples where both readers" in out
    out = subprocess.run([sys.executable, ROOT / "scripts" / "show_checks.py", "examples", tmp_path,
                          "--examples", "2", "--variant", "default"], capture_output=True, text=True, check=True).stdout
    assert out.count("MODEL'S ANSWER:") == 2 and "AFTER DELETING:" in out
    assert "Great question!" in out.split("AFTER DELETING:")[0]
    after = out.split("AFTER DELETING:")[1].split("DELETED:")[0]
    assert "Great question" not in after and "I hope this helps" not in after
    assert "Great question!" in out.split("DELETED:")[1]


def test_trace_matches_the_score(tmp_path):
    items, resp = tmp_path / "items.jsonl", tmp_path / "responses.jsonl"
    cli(["items", "--out", str(items), "--n-per-cell", "1"])
    cli(["generate", "--items", str(items), "--out", str(resp), "--backend", "mock", "--model-name", "m1"])
    rows = [json.loads(l) for l in resp.read_text().splitlines()]
    it = json.loads(items.read_text().splitlines()[3])
    for r in rows:
        if r["item_id"] == it["item_id"]:
            r["response"] = f"Great question! {it['gold_minimal']} I hope this helps!"
    resp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "trace.json"
    subprocess.run([sys.executable, ROOT / "scripts" / "trace_response.py", tmp_path, "--model", "m1", "--item",
                    it["item_id"], "--reader-backend", "mock", "--out", out], check=True, capture_output=True)
    tr = json.loads(out.read_text())
    frags = tr["fragments"]
    assert frags[0]["text"] == "Great question!" and not frags[0]["kept"] and frags[0]["candidate"]
    assert all(f["kept"] for f in frags if not f["candidate"])
    assert tr["queue"][: len(tr["read_alone"])] == [a["i"] for a in sorted(tr["read_alone"], key=lambda a: [f["step"] for f in frags if f["i"] == a["i"]][0])]
    passing = [c for c in tr["joint_checks"] if c["passes"]]
    assert passing and sorted(passing[-1]["removed"]) == sorted(a["i"] for a in tr["read_alone"])
    assert tr["facts"][0]["information_full"] > 1.0 and tr["result"]["tokens_kept"] < tr["result"]["tokens"]
