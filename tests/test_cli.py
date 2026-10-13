"""End-to-end CLI run with the mock backend (no model weights needed)."""

import json

import pandas as pd

from drybench.cli import main
from drybench.types import read_jsonl


def test_pipeline_end_to_end(tmp_path):
    items = tmp_path / "items.jsonl"
    resp = tmp_path / "resp.jsonl"
    scored = tmp_path / "scored.jsonl"
    signals = tmp_path / "signals.jsonl"
    report = tmp_path / "report"

    main(["items", "--out", str(items), "--n-per-cell", "2"])
    n_items = sum(1 for _ in read_jsonl(items))
    assert n_items > 20

    main(["generate", "--items", str(items), "--out", str(resp), "--backend", "mock",
          "--model-name", "mockA", "--batch-size", "4", "--variant", "default", "concise"])
    rows = list(read_jsonl(resp))
    assert len(rows) == 2 * n_items
    # resume: re-running generate adds nothing
    main(["generate", "--items", str(items), "--out", str(resp), "--backend", "mock", "--model-name", "mockA"])
    assert sum(1 for _ in read_jsonl(resp)) == 2 * n_items

    # make responses contain the gold answer + filler so scores are informative
    items_by_id = {r["item_id"]: r for r in read_jsonl(items)}
    patched = []
    for r in rows:
        gold = items_by_id[r["item_id"]]["gold_minimal"]
        r["response"] = f"Great question! {gold} I hope this helps!"
        patched.append(r)
    resp.write_text("".join(json.dumps(r) + "\n" for r in patched))

    main(["score", "--items", str(items), "--responses", str(resp), "--out", str(scored),
          "--reader-backend", "mock", "--mode", "loo", "--judge", "--ablation-abs-tol", "2.0"])
    assert all("judge_W" in r for r in read_jsonl(scored))
    main(["signals", "--items", str(items), "--responses", str(resp), "--out", str(signals),
          "--backend", "mock", "--model-name", "mockA", "--sampled", "3"])
    main(["analyze", "--items", str(items), "--scored", str(scored), "--signals", str(signals),
          "--out-dir", str(report), "--n-boot", "20"])

    main(["check-reader", "--items", str(items), "--reader-backend", "mock", "--n", "5",
          "--out", str(tmp_path / "reader_check.json")])
    rc = json.loads((tmp_path / "reader_check.json").read_text())
    assert rc["auc_present_vs_absent"] > 0.9 and rc["reader_fail_rate"] < 0.5

    for fig in ("fig_filler_vs_entropy.png", "fig_tokens_vs_k.png", "fig_attribution.png"):
        assert (report / fig).exists(), fig
    summary = json.loads((report / "summary.json").read_text())
    assert {"profiles", "groups"} <= set(summary)

    item_table = pd.read_csv(report / "item_table.csv")
    assert len(item_table) == 2 * n_items
    ok = item_table[item_table["status"] == "ok"]
    assert len(ok) > 0.8 * len(item_table)
    assert (ok["W"] > 0).all()  # preamble and closing are always filler here
    prof = pd.read_csv(report / "profiles.csv", index_col=0)
    assert set(prof.index) == {"mockA | default", "mockA | concise"}
    assert (report / "attribution.csv").exists()
    assert (report / "attribution_enrichment.csv").exists()
    assert all("ablation" in r and r["ablation"]["abs_tol"] == 2.0 for r in read_jsonl(scored))
    assert ok["W_ablation"].notna().all() and ok["facts"].notna().any()
    groups = pd.read_csv(report / "groups.csv", index_col=0)
    prod = groups["removable_gmean"] * groups["wordiness_gmean"]
    assert (abs(prod - groups["overhead_gmean"]) < 1e-9).all()


def test_items_update_probes_rewrites_answer_forms(tmp_path):
    items = tmp_path / "items.jsonl"
    main(["items", "--out", str(items), "--n-per-cell", "1"])
    fresh = list(read_jsonl(items))
    old = []
    for r in read_jsonl(items):  # as written by an older generator: one canonical answer, no value
        for p in r["probes"]:
            p["answers"] = [p.pop("value")]
        old.append(r)
    items.write_text("".join(json.dumps(r) + "\n" for r in old))
    main(["items", "--out", str(items), "--update-probes"])
    assert list(read_jsonl(items)) == fresh


def test_resume_refuses_other_settings(tmp_path):
    """v1.2: a run resumed with another model, token limit or scoring setting stops instead of mixing rows."""
    import pytest

    items, resp, scored = tmp_path / "items.jsonl", tmp_path / "resp.jsonl", tmp_path / "scored.jsonl"
    main(["items", "--out", str(items), "--n-per-cell", "1"])
    gen = ["generate", "--items", str(items), "--out", str(resp), "--backend", "mock", "--model-name", "m1"]
    main(gen)
    rows = list(read_jsonl(resp))
    assert all(r["gen"]["max_new_tokens"] == 320 and r["gen"]["decoding"] == "plain" for r in rows)
    assert all("truncated" in r and "n_tokens" in r for r in rows)
    first = rows[0]["item_id"]
    resp.write_text("".join(json.dumps(r) + "\n" for r in rows if r["item_id"] != first))
    with pytest.raises(SystemExit, match="max_new_tokens"):
        main(gen + ["--max-new-tokens", "10"])
    main(gen)  # same settings: resumes
    assert sum(1 for _ in read_jsonl(resp)) == len(rows)

    score = ["score", "--items", str(items), "--responses", str(resp), "--out", str(scored), "--reader-backend", "mock"]
    main(score)
    srows = list(read_jsonl(scored))
    assert all(r["settings"]["mode"] == "loo" and r["settings"]["keep_first"] for r in srows)
    scored.write_text("".join(json.dumps(r) + "\n" for r in srows[1:]))
    with pytest.raises(SystemExit, match="abs_tol"):
        main(score + ["--abs-tol", "2.0"])
    main(score)
    assert sum(1 for _ in read_jsonl(scored)) == len(srows)
