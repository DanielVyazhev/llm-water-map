"""v1.0: the item-wise annotation sample, the held-out abstention sample and the report on human labels."""

import json
from pathlib import Path

import pytest

from drybench.annotate import (abstain_check, build_abstain, build_sample, read_labels, report, sample_cases,
                               write_abstain, write_sample)
from drybench.cli import main as cli
from drybench.rescore import has_missing
from drybench.types import load_items

FILLER = {"m1": "", "m2": "Great question! ", "m3": "Sure, here is what the report says. "}
TAIL = {"default": " I hope this helps!", "concise": ""}


def _run(run: Path) -> None:
    """Synthetic items; three 'models' × two prompts answer with the gold answer and different filler."""
    run.mkdir(parents=True, exist_ok=True)
    items = run / "items.jsonl"
    cli(["items", "--out", str(items), "--n-per-cell", "2"])
    rows = []
    for it in load_items(items):
        for m, head in FILLER.items():
            for v, tail in TAIL.items():
                rows.append({"item_id": it.item_id, "model": m, "variant": v,
                             "response": f"{head}{it.gold_minimal}{tail}".strip()})
    (run / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "reader2_mock").mkdir(exist_ok=True)
    for sub in (run, run / "reader2_mock"):
        cli(["score", "--items", str(items), "--responses", str(run / "responses.jsonl"),
             "--out", str(sub / "scored.jsonl"), "--reader-backend", "mock", "--mode", "loo"])


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    r = tmp_path_factory.mktemp("ann") / "syn"
    _run(r)
    return r


def test_sample_is_item_wise_balanced_and_blind(run, tmp_path):
    sample, key, stats = build_sample([run], items_per_source=4, per_item=4, min_distinct=3, seed=1)
    items = {it.item_id: it for it in load_items(run / "items.jsonl")}
    assert stats["items"] == 4 and len(sample) == len(key) == stats["responses"]
    per_item = {}
    for k in key:
        per_item.setdefault(k["item_id"], []).append(k)
        assert not has_missing(items[k["item_id"]])  # missing-fact items are left out
        assert k["kinds_main"] is not None and k["kinds_mock"] is not None
    assert all(3 <= len(v) <= 4 for v in per_item.values())
    for s in sample:  # what annotators see carries no model, prompt or scores
        assert set(s) == {"uid", "source_label", "document", "request", "response", "spans"}
        assert all(sp["text"] in s["response"] for sp in s["spans"])
    # answers to one item come one after another (the source text is read once)
    order = [k["item_id"] for k in key]
    assert sum(a != b for a, b in zip(order, order[1:])) == len(per_item) - 1
    # the same seed gives the same sample
    assert build_sample([run], items_per_source=4, per_item=4, min_distinct=3, seed=1)[0] == sample
    # uids carry the sample's id: labels of one sample cannot name another sample's responses
    import re

    prefixes = {s["uid"].split("-")[0] for s in sample}
    assert len(prefixes) == 1 and all(re.fullmatch(r"s[0-9a-f]{6}-r\d{3}", s["uid"]) for s in sample)
    other = build_sample([run], items_per_source=4, per_item=4, min_distinct=3, seed=2)[0]
    assert {s["uid"].split("-")[0] for s in other} != prefixes
    write_sample(tmp_path, sample, key, stats)
    cases = sample_cases(tmp_path)
    assert len(cases) == len(sample) and cases[0]["group"].startswith("syn/")


def _export(path: Path, annotator: str, sample, flip_every: int = 0, fmt: str = "span-labels/1", name: str = "",
            ts: int = 1) -> None:
    """A page export whose labels follow the filler map of the key (flipping some to disagree)."""
    recs = []
    for n, s in enumerate(sample):
        labs = []
        for k, sp in enumerate(s["spans"]):
            lab = "filler" if any(w in sp["text"] for w in ("Great", "Sure", "hope")) else "needed"
            if flip_every and (n + k) % flip_every == 0:
                lab = "needed" if lab == "filler" else "filler"
            labs.append(lab)
        recs.append({"uid": s["uid"], "annotator": annotator, "span_i": [sp["i"] for sp in s["spans"]],
                     "labels": labs, "also_needed": [], "unsure": [], "complete": True, "ts": ts})
    doc = {"format": fmt, "records": recs}
    if name:
        doc["annotators"] = [{"id": annotator, "name": name, "self": True}]
    path.write_text(json.dumps(doc))


def test_exports_of_one_annotator_are_merged_and_other_samples_refused(run, tmp_path):
    from drybench.annotate import page_hash

    sample, key, stats = build_sample([run], items_per_source=4, per_item=3, min_distinct=3)
    # one annotator exports twice (a re-export after fixing labels): one annotator, the later labels
    _export(tmp_path / "anna_1.json", "local-3f9a2c", sample, name="Anna", ts=1)
    _export(tmp_path / "anna_2.json", "local-3f9a2c", sample, flip_every=4, name="Anna", ts=2)
    _export(tmp_path / "anna_b.json", "local-77aa00", sample, name="Anna")  # another person called Anna
    labels = read_labels([tmp_path / "anna_1.json", tmp_path / "anna_2.json", tmp_path / "anna_b.json"])
    assert set(labels) == {"Anna (local-3f9a2c)", "Anna (local-77aa00)"}
    flipped = json.loads((tmp_path / "anna_2.json").read_text())["records"][0]
    got = labels["Anna (local-3f9a2c)"][flipped["uid"]]["labels"]
    assert [got[i] for i in flipped["span_i"]] == flipped["labels"]
    assert set(read_labels([tmp_path / "anna_1.json", tmp_path / "anna_2.json"])) == {"Anna"}
    # labels of another sample: refused
    other = build_sample([run], items_per_source=4, per_item=3, min_distinct=3, seed=5)[0]
    _export(tmp_path / "other.json", "local-3f9a2c", other)
    with pytest.raises(ValueError, match="not in this sample"):
        read_labels([tmp_path / "other.json"], sample=sample)
    doc = json.loads((tmp_path / "anna_1.json").read_text())
    doc["dataset"] = {"hash": page_hash(other)}
    (tmp_path / "anna_1.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="another sample"):
        read_labels([tmp_path / "anna_1.json"], sample=sample)
    doc["dataset"] = {"hash": page_hash(sample)}
    (tmp_path / "anna_1.json").write_text(json.dumps(doc))
    assert set(read_labels([tmp_path / "anna_1.json"], sample=sample)) == {"Anna"}


def test_report_on_human_labels(run, tmp_path):
    sample, key, stats = build_sample([run], items_per_source=6, per_item=6, min_distinct=3)
    write_sample(tmp_path, sample, key, stats)
    _export(tmp_path / "a.json", "local", sample)
    _export(tmp_path / "b.json", "local", sample, flip_every=5, fmt="dry-span-labels/1")
    labels = read_labels([tmp_path / "a.json", tmp_path / "b.json"])
    assert set(labels) == {"local@a", "local@b"}  # two people labelling offline are told apart
    cli(["baselines", "--run", str(tmp_path), "--set", "sample", "--method", "judge", "--label", "mock",
         "--backend", "mock"])
    text = report(tmp_path, [tmp_path / "a.json", tmp_path / "b.json"], primary="local@a")
    for part in ("## Agreement between annotators", "local@a vs local@b", "### All pairs",
                 "### Pairs of answers to the same item", "filler map (reader main)", "LLM judge (mock)",
                 "### Annotator local@b on the responses both labelled", "## What the filler map did with each label"):
        assert part in text, part
    assert "DRY" not in text
    out = tmp_path / "rep.md"
    cli(["annotation-report", "--sample", str(tmp_path), "--labels", str(tmp_path / "a.json"), "--out", str(out)])
    assert "### All pairs" in out.read_text()


def test_abstain_sample_and_check(run, tmp_path):
    import csv

    rows, key, stats = build_abstain([run], n=8)
    assert len(rows) == 8 and all(k["item"]["knobs"]["answerable"] is False for k in key)
    write_abstain(tmp_path, rows, key, stats)
    assert "A = says the fact is not given" in (tmp_path / "abstain_legend.txt").read_text()
    path = tmp_path / "abstain_sample.csv"
    with open(path, newline="", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    for r in got:  # the toy answers give the gold, 'not stated in the report': all say it
        r["label (A/B/V/O)"] = "A"
    got[0]["label (A/B/V/O)"] = "x"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(got[0]))
        w.writeheader()
        w.writerows(got)
    res = abstain_check(path, tmp_path / "abstain_key.jsonl")
    assert res["labelled"] == 7 and res["unreadable_labels"] == [got[0]["uid"]]
    assert res["all"]["recall"] == 1.0 and res["all"]["precision"] == 1.0
    out = tmp_path / "r.md"
    cli(["annotation-report", "--sample", str(tmp_path), "--abstain-csv", str(path), "--out", str(out)])
    assert "## The abstention rule on held-out labels" in out.read_text()


def test_truth_is_a_person_and_needed_fact_is_kept(run, tmp_path):
    from drybench.annotate import page_hash, removable

    sample, key, stats = build_sample([run], items_per_source=4, per_item=3, min_distinct=3)
    write_sample(tmp_path, sample, key, stats)
    # a chat annotator alone cannot be the truth
    chat = tmp_path / "gpt.jsonl"
    chat.write_text("".join(json.dumps({"uid": s["uid"], "annotator": "gpt", "span_i": [x["i"] for x in s["spans"]],
                                        "labels": ["needed"] * len(s["spans"])}) + "\n" for s in sample))
    with pytest.raises(SystemExit, match="--primary"):
        report(tmp_path, [chat])
    # a person's export: '+ needed fact' keeps a wrong fragment out of the removable ones; the page's hash is checked
    _export(tmp_path / "me.json", "me", sample)
    doc = json.loads((tmp_path / "me.json").read_text())
    doc["dataset"] = {"hash": "deadbeef"}
    rec = doc["records"][0]
    rec["labels"][0] = "wrong"
    rec["also_needed"] = [rec["span_i"][0]]
    (tmp_path / "me.json").write_text(json.dumps(doc))
    labels = read_labels([tmp_path / "me.json"])
    assert not removable(labels["me"][rec["uid"]], rec["span_i"][0])
    with pytest.raises(SystemExit, match="another sample"):  # made on another version of the sample
        report(tmp_path, [tmp_path / "me.json", chat])
    doc["dataset"] = {"hash": page_hash(sample)}
    (tmp_path / "me.json").write_text(json.dumps(doc))
    text = report(tmp_path, [tmp_path / "me.json", chat])
    assert "me" in text and "the truth below" in text and "gpt" in text and "(chat LLM)" in text
    assert "## Cost on these responses" in text


def test_abstain_csv_from_spreadsheets(run, tmp_path):
    import csv

    rows, key, stats = build_abstain([run], n=4)
    write_abstain(tmp_path, rows, key, stats)
    with open(tmp_path / "abstain_sample.csv", newline="", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    for r in got:
        r["label (A/B/V/O)"] = " a "
    for name, kw in (("bom_semicolon.csv", {"delimiter": ";"}), ("tabs.csv", {"delimiter": "\t"})):
        path = tmp_path / name
        with open(path, "w", newline="", encoding="utf-8-sig") as f:  # Excel: BOM, ';' in a Russian locale
            w = csv.DictWriter(f, fieldnames=list(got[0]), **kw)
            w.writeheader()
            w.writerows(got)
        assert abstain_check(path, tmp_path / "abstain_key.jsonl")["labelled"] == len(got)
    (tmp_path / "wrong.csv").write_text("id,label\nx1,A\n")
    with pytest.raises(ValueError, match="no row has a uid"):
        abstain_check(tmp_path / "wrong.csv", tmp_path / "abstain_key.jsonl")


def test_baseline_rows_for_another_response_are_not_attached():
    from drybench.compare import attach, response_hash

    cases = [{"id": "r001", "fragments": ["a", "b"], "response": "a b", "chars": [1, 1]}]
    row = {"labels": ["needed", "filler"], "p": [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]]}
    attach(cases, "judge-x", {"r001": {**row, "hash": response_hash("c d")}}, "judge")
    assert "judge-x" not in cases[0]["methods"]
    attach(cases, "judge-x", {"r001": {**row, "hash": response_hash("a  b")}}, "judge")
    assert cases[0]["methods"]["judge-x"]["removable"] == [False, True]
