"""v1.2: the confirmatory tests of the protocol (drybench.confirm)."""

import json
from pathlib import Path

import numpy as np
import pytest

from drybench.confirm import H1_NULLS, TESTS, confirm, holm, markdown, p_value, ranking, validity


def test_holm_and_bootstrap_p():
    got = holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(got, [0.03, 0.06, 0.06, 0.02])
    assert np.isnan(holm([0.01, float("nan")])[1]) and holm([0.01, float("nan")])[0] == 0.01
    assert p_value([0.2] * 99, 0.0) == pytest.approx(1 / 100)            # never at or below the null
    assert p_value([-0.1] * 99, 0.0) == pytest.approx(1.0)
    assert p_value([0.01] * 99, 0.05, greater=False) == pytest.approx(1 / 100)


def _rows(n_items: int, lose_every: int = 0, wrong_delete_every: int = 0) -> list[dict]:
    """Planted responses: a fact sentence and 0-3 filler fragments; the map deletes the filler (and, every
    lose_every-th response, the fact; every wrong_delete_every-th, keeps one filler fragment)."""
    rows, k = [], 0
    for i in range(n_items):
        for n_fill in range(4):
            k += 1
            spans = ["The carrier was Dunmar Freight."] + ["Sure, happy to help with that question."] * n_fill
            labels = ["fact"] + ["filler:preamble"] * n_fill
            kept = [True] + [False] * n_fill
            lost = bool(lose_every and k % lose_every == 0)
            if lost:
                kept[0] = False
            if wrong_delete_every and n_fill and k % wrong_delete_every == 0:
                kept[1] = True
            removed = sum(not x for x in kept)
            prec = (removed - lost) / removed if removed else 1.0
            rows.append({"item_id": f"it{i}", "condition": f"c{n_fill}", "text": " ".join(spans), "span_texts": spans,
                         "labels": labels, "span_facts": [["f1"]] + [[]] * n_fill, "yap_excess": 0.0,
                         "stopword_share": 0.5, "chars": len(" ".join(spans)), "loo_status": "ok", "loo_kept": kept,
                         "loo_safety": 0.0 if lost else 1.0, "loo_filler_precision": prec,
                         "loo_cost": [3.0] + [0.0] * n_fill})
    return rows


def test_validity_and_ranking_on_planted_rows(tmp_path):
    from drybench.compare import fill_unscored, planted_cases

    good = validity(_rows(20), n_boot=300)
    assert good["lost"] == 0 and good["precision"] == 1.0 and good["p"] < 0.01
    bad = validity(_rows(20, lose_every=5), n_boot=300)  # a fact lost in 20% of the responses
    assert bad["lost"] == pytest.approx(0.2) and bad["p_lost"] > 0.5 and bad["p"] == max(bad["p_lost"], bad["p_precision"])
    sub = tmp_path / "syn"
    (sub / "filler_check").mkdir(parents=True)
    (sub / "filler_check" / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _rows(20)))
    cases = planted_cases(sub, "planted_synthetic")
    from drybench.compare import attach

    attach(cases, "map:main", {c["id"]: c["map"] for c in cases}, "map")
    r = ranking(fill_unscored(cases, ["map:main"]), n_boot=300)
    # the map deletes exactly the planted filler; length grows with it too, so both rank well: the map at least as well
    assert r["acc_map"] == pytest.approx(1.0) and r["acc_diff"] >= 0 and r["responses"] == 80


def test_h1_thresholds_per_truth_set():
    """T1: 5% and 0.95; R3: 15% and 0.85 (the development CI bounds rounded outward)."""
    rows = _rows(40, lose_every=10)  # a fact lost in 10% of the responses, precision 0.94
    t1 = validity(rows, n_boot=300, nulls=H1_NULLS["T1"])
    r3 = validity(rows, n_boot=300, nulls=H1_NULLS["R3"])
    assert t1["lost"] == pytest.approx(0.1) and t1["p"] > 0.5 and r3["p"] < 0.05
    assert (r3["max_lost"], r3["min_precision"]) == (0.15, 0.85)
    t, _ = confirm(n_boot=10)
    assert t.loc["H1 T1", "hypothesis"] == "lost < 5% and precision > 0.95"
    assert t.loc["H1 R3", "hypothesis"] == "lost < 15% and precision > 0.85"


def test_confirm_reports_eight_tests_and_holm(tmp_path):
    syn = tmp_path / "run" / "synthetic"
    (syn / "filler_check").mkdir(parents=True)
    (syn / "filler_check" / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _rows(15)))
    t, parts = confirm(synthetic=syn, n_boot=200)
    assert list(t.index) == [f"{h} {s}" for h, s in TESTS]
    assert t.loc["H1 T1", "result"] == "confirmed" and t.loc["H2 R3", "result"] == "–"
    text = markdown(t, parts, 200)
    assert "Provisional" in text and "| H1 T1 |" in text
    from drybench.cli import main as cli

    out = tmp_path / "c.md"
    cli(["confirm", "--run", str(tmp_path / "run"), "--n-boot", "100", "--out", str(out)])
    assert "Confirmatory tests" in out.read_text()


def test_confirm_on_human_labels(tmp_path):
    """HL: the annotation sample of a mock run against a page export."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ta", Path(__file__).with_name("test_annotation.py"))
    ta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ta)
    from drybench.annotate import build_sample, write_sample

    run = tmp_path / "syn"
    ta._run(run)
    sample, key, stats = build_sample([run], items_per_source=6, per_item=6, min_distinct=3)
    ann = tmp_path / "annotation"
    write_sample(ann, sample, key, stats)
    ta._export(tmp_path / "me.json", "local-aaaaaa", sample, name="Me")
    t, parts = confirm(annotation=ann, label_files=[tmp_path / "me.json"], n_boot=100)
    assert parts["HL"]["primary"] == "Me" and parts["HL"]["H2"]["responses"] == len(sample)
    assert t.loc["H2 HL", "p"] == t.loc["H2 HL", "p"]  # computed
