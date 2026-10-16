"""CUAD and Poddar et al. converters, and the checks on real texts (planted filler, human labels)."""

import importlib.util
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

from drybench.backends.mock import MockBackend
from drybench.cli import main as cli_main
from drybench.cuad import (CUAD_CELLS, GOVERNING, build_excerpt, contract_facts, cuad_items, date_forms,
                           find_date, governing_law, initial_term, load_cuad, parse_date)
from drybench.items import NOT_STATED
from drybench.poddar import (best_match, fragment_labels, labelled_cases, load_poddar, parse_record,
                             poddar_items)
from drybench.reader import Reader, ReaderConfig
from drybench.realcheck import (_same_value, _starts_any, human_label_check, planted_check, simulate_real,
                                summarize_planted, wrong_pool)
from drybench.realdata import build_real_items, squad_items
from drybench.filler import dry_score
from drybench.segment import segment
from drybench.types import load_items

ROOT = Path(__file__).resolve().parents[1]
MUSIQUE = Path(__file__).parent / "data" / "musique_sample.jsonl"
TOPICS = ["delivery", "pricing", "invoices", "warranty", "insurance", "audits", "notices", "assignment",
          "confidentiality", "records", "taxes", "force majeure", "subcontracting", "publicity"]


# ---------------------------------------------------------------- fixtures

def _contract(name, title, date, law, term, n_filler=12):
    paras = ["Exhibit 10.1", title,
             f"This {title.title()} is made as of {date} by and between Acme Widgets Inc., a Nevada corporation "
             f"(\"Acme\"), and Beta Supplies LLC (\"Beta\")."]
    if term:
        paras.append(f"1. Term. The term of this Agreement shall be {term} from the Effective Date, unless "
                     "terminated earlier as provided herein.")
    for i, topic in enumerate(TOPICS[:n_filler]):
        paras.append(f"{i + 2}. {topic.title()}. Each party shall handle {topic} in a commercially reasonable "
                     f"manner and shall keep complete records about {topic} for the duration of the business "
                     "relationship, making them available to the other party upon reasonable written request "
                     "during normal business hours.")
    if law:
        paras.append(f"20. Governing Law. This Agreement shall be governed by and construed in accordance with "
                     f"the laws of the State of {law}, without regard to its conflict of laws principles.")
    ctx = "\n\n".join(paras)
    qas = []

    def add(cat, text):
        i = ctx.find(text)
        assert i >= 0, text
        qas.append({"id": f"{name}__{cat}", "question": cat, "answers": [{"text": text, "answer_start": i}],
                    "is_impossible": False})

    add("Document Name", title)
    add("Agreement Date", date)
    if law:
        add("Governing Law", [p for p in paras if "Governing Law" in p][0].split(". ", 1)[1])
    if term:
        add("Expiration Date", [p for p in paras if p.startswith("1. Term")][0].split(". ", 1)[1])
    return {"title": name, "paragraphs": [{"context": ctx, "qas": qas}]}


@pytest.fixture
def cuad_file(tmp_path):
    specs = [("C1", "SUPPLY AGREEMENT", "6th day of April, 1999", "New York", "five (5) years"),
             ("C2", "DISTRIBUTION AGREEMENT", "March 3, 2005", "Delaware", "three (3) years"),
             ("C3", "LICENSE AGREEMENT", "1 July 2010", "California", None),
             ("C4", "HOSTING AGREEMENT", "January 15th, 2001", "Texas", "two (2) years"),
             ("C5", "SERVICES AGREEMENT", "October 9, 2012", None, "one (1) year")]
    data = {"version": "test", "data": [_contract(*s) for s in specs]}
    path = tmp_path / "CUADv1.json"
    path.write_text(json.dumps(data))
    return path


def _poddar_row(rid, header, passage, question, reference, parts):
    """parts: [(text, label or None)] joined into the response; labels get character offsets."""
    pre = f"{header}\n{passage}\n### Question: {question}\n### Response: {reference}"
    text = pre + "\n*#\n*#\n*#\n"
    labels = []
    for chunk, lab in parts:
        start = len(text)
        text += chunk
        if lab:
            labels.append([start, len(text), lab])
    return {"id": rid, "text": text, "label": labels, "Comments": []}


@pytest.fixture
def poddar_file(tmp_path):
    story = ("Lothair is a young nobleman. After the battle he is nursed by Clare Arundel, who tells him he was "
             "saved by an apparition of the Virgin Mary. Later he travels to Syria with Mr. Phoebus.")
    tweet = "Anyone who keeps a straight face while she sings deserves to be president. September 27, 2016"
    rows = [
        _poddar_row(1, "narrqa_gpt-4o-mini_17", story, "Who saved Lothair according to Clare?", "The Virgin Mary.",
                    [("Great question! ", "CONVERS"), ("Clare says he was saved by the Virgin Mary. ", "MINANS"),
                     ("Lothair later travels to Syria with Phoebus. ", "ADDINFO"), ("I hope this helps!", "CONVERS"),
                     ("\n### Prompt:\nWho is Phoebus?", None)]),
        _poddar_row(2, "narrqa_Llama-3.1-8B-Instruct_17", story, "Who saved Lothair according to Clare?",
                    "The Virgin Mary.", [("The Virgin Mary. ", "MINANS"), ("She was the one who saved him. ", "REDINFO")]),
        _poddar_row(3, "tweetqa_gpt-3.5-turbo_388", tweet, "what does this person deserve to be?", "president",
                    [("This person deserves to be president. ", "MINANS"),
                     ("The tweet is a joke about singing. ", "EXPLAIN")]),
        _poddar_row(4, "dolly_gpt-3.5-turbo_965", "", "Name a fruit.", "Apple", [("An apple.", "MINANS")]),
    ]
    rows[3]["text"] = rows[3]["text"].replace("\n\n### Question: Name a fruit.\n### Response:", "\nName a fruit.\n### Response:")
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _chat_reader(**kw):
    return Reader(MockBackend(**kw), ReaderConfig(use_chat=True))


# ---------------------------------------------------------------- CUAD

def test_cuad_values():
    assert parse_date("this 6th day of April, 1999") == (1999, 4, 6)
    assert parse_date("October 1,1996") == (1996, 10, 1)
    assert parse_date("1st  day of JUNE , 2006") == (2006, 6, 1)
    assert parse_date("[ ] day of [ ], 2020") is None and parse_date("29/3/18") is None
    forms = date_forms((1999, 4, 6), "the 6th day of April, 1999")
    assert {"April 6, 1999", "6th day of April, 1999", "4/6/1999"} <= set(forms)
    assert not any(a != b and b.startswith(a) for a in forms for b in forms)
    assert governing_law(["THIS AGREEMENT SHALL BE GOVERNED BY THE LAWS OF THE STATE OF NEW YORK, WITHOUT "
                          "REFERENCE TO ITS CONFLICT OF LAW PROVISIONS."])[:2] == ("New York", "NEW YORK")
    assert governing_law(["This Agreement is governed by English law."])[0] == "England"
    assert governing_law(["construed in accordance with the internal laws of the State of Nevada"])[0] == "Nevada"
    assert governing_law(["Each party shall comply with all policies."]) is None
    assert initial_term(["The initial term of this Agreement shall be three (3) years and shall renew for "
                         "successive one (1) year periods."])[0] == "three (3) years"
    assert initial_term(["The production period shall be two (2) months and the term of this Agreement shall be "
                         "twenty four (24) months."]) is None  # two durations: ambiguous
    # review fixes: renewal periods and sub-terms are not the initial term; joined jurisdictions
    assert initial_term(["This Agreement shall continue until December 31, 2010, and shall automatically renew "
                         "for a term of one (1) year."]) is None
    assert initial_term(["The Maintenance Term shall continue for 12 months after delivery."]) is None
    assert initial_term(["The Agreement shall have an initial term of ten (10) years."])[0] == "ten (10) years"
    assert governing_law(["governed by the laws of the United States and the State of Florida"])[0] == "Florida"
    assert governing_law(["governed by the laws of California and Hong Kong."]) is None
    assert governing_law(["governed by the laws of the State of New York, United States of America"])[0] == "New York"
    assert find_date('Consulting Agreement (the "Agreement") is made this 2nd day of January 2020')[1] == \
        "2nd day of January 2020"


def test_cuad_titles_cut_mid_label_are_repaired_or_dropped():
    ctx = "JOINT VENTURE AGREEMENT\n\nThis Joint Venture Agreement is made by the parties."
    cut = {"cid": "X", "context": ctx, "answers": {"Document Name": [("Joint Venture Agreemen", 0)]}}
    assert "title" not in contract_facts(cut)  # cut inside a word: no answer could match it
    comma = dict(cut, answers={"Document Name": [("JOINT VENTURE AGREEMENT, d", 0)]})
    f = contract_facts(comma)["title"]
    assert f["value"] == "Joint Venture Agreement" and {"JOINT VENTURE AGREEMENT", "Joint Venture Agreement"} <= set(f["forms"])


def test_cuad_facts_and_excerpts(cuad_file):
    contracts = load_cuad(cuad_file)
    facts = contract_facts(contracts[0])
    assert facts["title"]["value"] == "Supply Agreement" and "SUPPLY AGREEMENT" in facts["title"]["forms"]
    assert facts["date"]["value"] == "April 6, 1999" and facts["law"]["value"] == "New York"
    assert facts["term"]["value"] == "5 years" and "five (5) years" in facts["term"]["forms"]
    assert "law" not in contract_facts(contracts[4]) and "term" not in contract_facts(contracts[2])
    text = build_excerpt(contracts[0], facts, ["date", "law"], 2, random.Random(0), 900)
    assert "6th day of April, 1999" in text and "State of New York" in text and "[...]" in text
    miss = build_excerpt(contracts[0], facts, ["title", "law"], 2, random.Random(0), 900, without_law=True)
    assert not GOVERNING.search(miss) and "SUPPLY AGREEMENT" in miss


def test_cuad_items_follow_the_cells(cuad_file):
    items, stats = cuad_items(load_cuad(cuad_file), n_per_cell=2)
    assert [s["built"] for s in stats.values()] == [2] * len(CUAD_CELLS)
    by_cell = {}
    for it in items:
        by_cell.setdefault(it.meta["cell"], []).append(it)
        assert it.meta["doc_label"] == "Contract excerpt" and it.knobs["source"] == "cuad"
        assert len(it.required_probes) == it.knobs["k"]
        for p in it.required_probes:  # every present fact is in the excerpt as written
            if p.value != NOT_STATED:
                assert any(a in it.document for a in p.answers), (p.answers, it.item_id)
    assert all(it.request.startswith("Answer these questions about the contract:\n1. ") for it in by_cell[2])
    for it in by_cell[3]:  # long excerpts carry more distractors
        assert it.knobs["long"] and it.knobs["d"] == 8
    for it in by_cell[4]:
        law = [p for p in it.probes if p.fact_id.endswith(":law")][0]
        assert law.value == NOT_STATED and not GOVERNING.search(it.document)
        assert it.gold_minimal.count("Not stated in the excerpt") == 1


# ---------------------------------------------------------------- Poddar et al.

def test_poddar_records_items_and_labels(poddar_file):
    recs = load_poddar(poddar_file)
    assert [r["subset"] for r in recs] == ["narrqa", "narrqa", "tweetqa", "dolly"]
    assert recs[3]["passage"] is None and recs[0]["question"] == "Who saved Lothair according to Clare?"
    r0 = recs[0]
    assert r0["response"][r0["labels"][1][0]: r0["labels"][1][1]].startswith("Clare says")
    items, stats = poddar_items(recs, n_per_cell=5)
    assert [s["built"] for s in stats.values()] == [1, 1, 0]  # one question per subset; no MS MARCO here
    lothair = items[0]
    assert lothair.meta["labelled_responses"] == 2 and "Virgin Mary" in lothair.probes[0].answers
    assert items[1].request == "What does this person deserve to be?" and items[1].meta["doc_label"] == "Tweet"
    cases = labelled_cases(recs)
    assert len(cases) == 3 and cases[0]["cut_chars"] > 0 and "### Prompt" not in cases[0]["response"]
    spans = segment(cases[0]["response"])
    assert fragment_labels(spans, cases[0]["labels"]) == ["CONVERS", "MINANS", "ADDINFO", "CONVERS"]
    assert best_match("the women's U.S. hockey program", "womens us hockey program") == "women's U.S. hockey program"
    assert best_match("nothing alike here", "president") is None
    assert parse_record({"id": 9, "text": "no separator", "label": []}) is None


def test_human_label_check_scores_against_labels(poddar_file, tmp_path):
    cases = labelled_cases(load_poddar(poddar_file))
    s = human_label_check(_chat_reader(), cases, tmp_path / "human", quiet=True)
    assert s["n_responses"] == 3 and s["n_scored_ok"] == 3
    f = s["overall"]["fragments"]
    assert f["precision"] == 1.0 and f["recall"] == 1.0  # mock reader: exactly the answer sentences are kept
    assert s["overall"]["minimal_answer_kept"] == 1.0
    assert s["by_label"]["MINANS"]["deleted"] == 0.0 and s["by_label"]["CONVERS"]["deleted"] == 1.0
    again = human_label_check(_chat_reader(), cases, tmp_path / "human", quiet=True)  # resumes, same result
    assert again["overall"] == s["overall"]


# ---------------------------------------------------------------- planted filler on real texts

def _squad_paras():
    ctx = ("The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region "
           "in France. They were descended from Norse raiders led by Rollo. The Norman dynasty had a major "
           "political impact on medieval Europe. In 1066 William conquered England.")
    qa = lambda i, q, a: {"id": f"q{i}", "question": q, "is_impossible": False,  # noqa: E731
                          "answers": [{"text": a, "answer_start": 0}], "plausible_answers": []}
    return [{"pid": "Normans#0", "title": "Normans", "context": ctx,
             "qas": [qa(1, "In what country is Normandy located?", "France"),
                     qa(2, "Who led the Norse raiders?", "Rollo"),
                     qa(3, "When did William conquer England?", "1066")]}]


def test_simulate_real_labels_and_truth(cuad_file):
    items, _ = squad_items(_squad_paras(), n_per_cell=1, cells=[{"k": 2}])
    it = items[0]
    sim = simulate_real(it, ["preamble", "elaboration", "summary", "closing"], rng=random.Random(1))
    labels = [s.label for s in sim.spans]
    assert labels[0] == "filler:preamble" and labels[-1] == "filler:closing" and labels.count("fact") >= 1
    assert set(sim.present_facts) == {p.fact_id for p in it.required_probes}
    texts = [s.text for s in sim.spans]
    assert len(texts) == len(set(texts))
    for s in sim.spans:  # fact sentences come from the document; filler states no requested answer
        if s.label == "fact":
            assert s.text in it.document
        elif s.label != "filler:summary":
            assert not any(a in s.text for p in it.required_probes for a in p.answers)
    one = simulate_real(it, [], omit=[it.required_probes[0].fact_id], rng=random.Random(2))
    assert it.required_probes[0].fact_id not in one.present_facts
    cu, _ = cuad_items(load_cuad(cuad_file), n_per_cell=2)
    pools = wrong_pool(cu)
    miss = [i for i in cu if i.meta["cell"] == 4][0]
    sim = simulate_real(miss, ["hedge"], n_wrong=1, rng=random.Random(0), pool=pools["cuad"])
    assert any(s.text.startswith("Not stated in the contract excerpt:") for s in sim.spans)
    wrong = [s for s in sim.spans if s.label == "wrong"]
    assert wrong and wrong[0].text.startswith('The answer to "')


def test_wrong_answers_and_filler_filter_respect_equivalents():
    assert _same_value("24 months", "2 years", "term") and not _same_value("12 months", "2 years", "term")
    assert _same_value("England", "England and Wales", "law") and not _same_value("Texas", "New York", "law")
    assert _same_value("April 6, 1999", "6th day of April, 1999", "date")
    assert _starts_any("Most engines have six cylinders.", ["cylinder"])  # the answer covers 'cylinders'
    assert not _starts_any("A recylinder is not a word.", ["cylinder"])


def test_planted_check_end_to_end(cuad_file, tmp_path):
    squad, _ = squad_items(_squad_paras(), n_per_cell=1, cells=[{"k": 1}, {"k": 2}])
    cu, _ = cuad_items(load_cuad(cuad_file), n_per_cell=2)
    reader = _chat_reader(unknown_fn=lambda msg: 0.02 if "The answer to" in msg else 0.9)
    s = planted_check(reader, squad + cu, tmp_path / "planted", n_per_cell=1, quiet=True)
    assert s["n_scored_ok"] > 0 and set(s["by_source"]) == {"squad2", "cuad"}
    o = s["overall"]
    assert o["no_fact_lost"] == 1.0 and o["recall_pooled"] == 1.0  # the mock reader reads exact strings
    assert s["wrong"]["planted"] > 0 and s["wrong"]["deleted"] == 1.0 and s["wrong"]["labelled_wrong"] == 1.0
    assert s["filler_labelled_wrong"]["labelled_wrong"] == 0.0
    m = s["metric_vs_true_filler"]
    assert m["removable share (filler map)"]["spearman"] > m["length"]["spearman"]
    rows = [json.loads(line) for line in (tmp_path / "planted" / "rows.jsonl").read_text().splitlines()]
    assert summarize_planted(rows)["overall"] == o
    assert {r["condition"] for r in rows} >= {"dense", "light", "heavy", "incomplete", "wrong"}
    for r in rows:  # a left-out fact is really absent
        assert (r["n_omitted"] > 0) == (r["condition"] == "incomplete") and r["coverage"] <= 1.0
        if r["condition"] == "incomplete":
            assert r["coverage"] < 1.0


# ---------------------------------------------------------------- CLI, runner and report

def test_cli_builds_cuad_and_poddar_pools(cuad_file, poddar_file, tmp_path):
    cli_main(["real-items", "--source", "cuad", "--path", str(cuad_file), "--n-per-cell", "2",
              "--out", str(tmp_path / "cu.jsonl")])
    cli_main(["real-items", "--source", "poddar", "--path", str(poddar_file), "--n-per-cell", "2",
              "--out", str(tmp_path / "pd.jsonl")])
    cu, pd_ = load_items(tmp_path / "cu.jsonl"), load_items(tmp_path / "pd.jsonl")
    assert len(cu) == 2 * len(CUAD_CELLS) and len(pd_) == 2
    assert build_real_items("cuad", cuad_file, 2)[0][0].to_dict() == cu[0].to_dict()
    cli_main(["real-filler-check", "--items", str(tmp_path / "cu.jsonl"), "--out-dir", str(tmp_path / "rw"),
              "--n-per-cell", "1", "--reader-backend", "mock"])
    cli_main(["poddar-check", "--data", str(poddar_file), "--out-dir", str(tmp_path / "pc"),
              "--reader-backend", "mock"])
    assert json.loads((tmp_path / "rw" / "summary.json").read_text())["n_scored_ok"] > 0
    assert json.loads((tmp_path / "pc" / "summary.json").read_text())["n_responses"] == 3


def _squad_file(path):
    paras = _squad_paras()
    data = {"data": [{"title": "Normans", "paragraphs": [{"context": paras[0]["context"], "qas": paras[0]["qas"]}]}]}
    path.write_text(json.dumps(data))


def test_runner_on_four_suites_and_report(cuad_file, poddar_file, tmp_path):
    data = tmp_path / "data"
    for sub, src, name in (("squad2", None, "dev-v2.0.json"), ("musique", MUSIQUE, "musique_full_v1.0_dev.jsonl"),
                           ("cuad", cuad_file, "CUADv1.json"), ("poddar", poddar_file, "data.jsonl")):
        (data / sub).mkdir(parents=True)
        if src is None:
            _squad_file(data / sub / name)
        else:
            (data / sub / name).write_bytes(Path(src).read_bytes())
    run = tmp_path / "run"
    cmd = [sys.executable, ROOT / "scripts" / "run_real.py", "--run", run, "--data", data, "--backend", "mock",
           "--allow-cpu", "--models", "m1=echo", "--reader", "mock", "--reader2", "mock2", "--variants", "default",
           "--n-per-cell", "1", "--oversample", "1", "--filler-check-n", "1", "--real-check-n", "1",
           "--distractors", "1"]
    out = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    tables = (run / "tables_real.md").read_text()
    for head in ("## R0.", "## R3.", "## R4.", "## R5.", "## R6.", "## CUAD (contracts)", "## Poddar et al. (passages)",
                 "complete answers", "prefix ablation", "Missing-fact items: does the answer say", "filler or not"):
        assert head in tables, head
    for sub in (run, run / "reader2_mock2"):
        assert (sub / "filler_check_real" / "summary.json").exists() and (sub / "human_check" / "summary.json").exists()
    assert "DRY" not in tables and "v0.7.1" not in tables
    assert (tmp_path / "run.zip").exists()
    # an older filter (no scoring version) and a cell that lost its item: the rerun keeps the items in use
    # and tops the cell up from the pool (here with the same item, whose response is reused)
    items = [json.loads(x) for x in (run / "items.jsonl").read_text().splitlines()]
    dropped = items.pop(0)
    (run / "items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    rep = json.loads((run / "filter_report.json").read_text())
    rep.pop("scoring")
    (run / "filter_report.json").write_text(json.dumps(rep))
    n_resp = len((run / "responses.jsonl").read_text().splitlines())
    out = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    assert "re-checking the rest of the pools" in out.stdout
    again = [json.loads(x)["item_id"] for x in (run / "items.jsonl").read_text().splitlines()]
    assert dropped["item_id"] in again and set(x["item_id"] for x in items) <= set(again)
    assert len((run / "responses.jsonl").read_text().splitlines()) == n_resp
    # a run scored by version 9: v1.2.0 changed every score, so both readers rescore every response; the
    # responses are kept and the filter is re-checked
    import drybench

    n_scored = {sub: len((sub / "scored.jsonl").read_text().splitlines()) for sub in (run, run / "reader2_mock2")}
    (run / "versions.json").write_text(json.dumps({"scoring": "9", "signals": drybench.SIGNALS_VERSION}))
    rep = json.loads((run / "filter_report.json").read_text())
    (run / "filter_report.json").write_text(json.dumps({**rep, "scoring": "9"}))
    out = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    assert out.stdout.count("rescoring every score") == 2  # both readers
    assert "re-checking the rest of the pools" in out.stdout
    assert len((run / "responses.jsonl").read_text().splitlines()) == n_resp
    for sub, n in n_scored.items():
        assert len((sub / "scored.jsonl").read_text().splitlines()) == n
        assert (sub / "filler_check_real" / "summary.json").exists() and (sub / "report").exists()
    assert "## R3." in (run / "tables_real.md").read_text()
    # scored by 10: v1.2.1 changed synthetic reports only; the real-text scores stay (rows of version 10 next to
    # new ones), the synthetic planted-filler check is made again
    for sub in n_scored:
        rows = [json.loads(x) for x in (sub / "scored.jsonl").read_text().splitlines()]
        (sub / "scored.jsonl").write_text("".join(json.dumps({**r, "settings": {**r["settings"], "scoring": "10"}})
                                                  + "\n" for r in rows))
    (run / "versions.json").write_text(json.dumps({"scoring": "10", "signals": drybench.SIGNALS_VERSION}))
    out = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    assert out.stdout.count("rescoring scores of synthetic items") == 2
    for sub, n in n_scored.items():
        rows = [json.loads(x) for x in (sub / "scored.jsonl").read_text().splitlines()]
        assert len(rows) == n and {r["settings"]["scoring"] for r in rows} == {"10"}
        assert (sub / "filler_check" / "summary.json").exists()


def test_generation_halves_a_batch_on_out_of_memory():
    import argparse

    from drybench.cli import _generate_safe

    class Small:
        torch = None

        def generate_batch(self, ctxs, max_new_tokens, temperature, seed):
            if len(ctxs) > 3:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return [c.upper() for c in ctxs]

    args = argparse.Namespace(max_new_tokens=8, temperature=0.0, seed=0)
    got = _generate_safe(Small(), [f"c{i}" for i in range(10)], args)
    assert [g["text"] for g in got] == [f"C{i}" for i in range(10)]


def test_colab_real_notebook_is_built():
    spec = importlib.util.spec_from_file_location("make_colab", ROOT / "scripts" / "make_colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cell = mod.build_cell(template="real_template.py")
    assert "__PKG_B64__" not in cell and "run_real.py" in cell
    nb = mod.real_notebook(cell)
    assert nb["metadata"]["colab"]["gpuType"] == "T4" and len(nb["cells"]) == 7  # v1.2.1: + the cell "Package"
    miss = mod.build_cell(template="missing_check_template.py")
    assert "__PKG_B64__" not in miss and "missing-check" in miss and "DRY" not in miss.split("PKG_B64 = ")[0]
    assert len(mod.missing_notebook(miss)["cells"]) == 3


# ---------------------------------------------------------------- v0.7.1

def test_reader_asks_for_not_stated_explicitly():
    from drybench.reader import NOT_STATED_HINT
    from drybench.types import Probe

    seen = []

    class Recording(MockBackend):
        def continuation_logprobs(self, requests):
            seen.extend(c for c, _ in requests)
            return super().continuation_logprobs(requests)

    chat = Reader(Recording(), ReaderConfig(use_chat=True, missing_style="hint"))  # v0.7.1–v0.7.6
    missing = Probe("Who conquered Wales in 1066?", ["not stated", "Not stated"])
    fact = Probe("Who led the raiders?", ["Rollo"])
    assert chat.question(missing).endswith(NOT_STATED_HINT) and chat.question(fact) == fact.question
    assert "If there is no message, reply 'unknown'." in NOT_STATED_HINT
    assert chat.forms(missing) == ["not", "Not"] and chat.forms(fact) == ["Rollo"]  # 'not specified' counts too
    chat.probe_logprobs("req", ["Not stated in the text."], [missing, fact])
    assert sum(NOT_STATED_HINT.strip() in c for c in seen) == 2  # both aliases of the missing fact, not the fact
    plain = Reader(MockBackend(), ReaderConfig())
    assert plain.question(missing) == missing.question  # base-model readers keep the plain template


def test_skip_ablation_deletes_what_the_prefix_leaves():
    from drybench.types import Probe, Span
    from drybench.filler import filler_map

    texts = ["Please note this.", "The first fact is Alpha Centauri in the long version of the sentence.",
             "The first fact is Alpha Centauri again, said at some length here.",
             "Second: Beta.", "Again: Beta."]
    spans, pos = [], 0
    for t in texts:
        spans.append(Span(pos, pos + len(t), t))
        pos += len(t) + 1
    probes = [Probe("Which star?", ["Alpha Centauri"]), Probe("Which letter?", ["Beta"])]
    reader = Reader(MockBackend(), ReaderConfig(use_chat=True))
    # the search alone (keep_first=False, as up to v1.1)
    base = filler_map(reader, "req", spans, probes, mode="loo", loo_skip=False, keep_first=False)  # v0.7.1
    skip = filler_map(reader, "req", spans, probes, mode="loo", keep_first=False)  # since v0.7.2
    gone = lambda r: {i for i, s in enumerate(r.spans) if not s.kept}  # noqa: E731
    assert gone(base) == {0, 1}  # the prefix stops at the second copy of fact 1 and keeps everything after it
    assert gone(skip) == {0, 1, 3} and gone(base) <= gone(skip)  # one copy of each fact stays
    assert all(s.candidate for s in base.spans) and base.spans[0].loo_cost == pytest.approx(0.0, abs=1e-9)
    # v1.2 (default): of two statements of a fact the first one is kept
    assert gone(filler_map(reader, "req", spans, probes, mode="loo")) == {0, 2, 4}


def test_true_statement_excuse_counts_every_word():
    from drybench.types import Item, Probe
    from drybench.filler import misleading_excuse

    doc = ("SUPPLY AGREEMENT\n\nThis Supply Agreement is made by Acme Inc., a New Jersey corporation, and Beta LLC.\n\n"
           "Each party shall keep records of deliveries.")
    it = Item("cu-x", doc, "Which jurisdiction's law governs this agreement?",
              [Probe("Which jurisdiction's law governs this agreement?", ["not stated", "Not stated"])], "Not stated.",
              {"source": "cuad", "k": 1}, {"source": "cuad"})
    assert misleading_excuse("The law of New Jersey governs this agreement.", it) is None  # a made-up answer
    assert misleading_excuse("Acme Inc. is a New Jersey corporation.", it) == "true statement"
    assert misleading_excuse("Which jurisdiction's law governs this agreement?", it) == "no new words"


def test_filter_keeps_items_in_use_and_tops_up_short_cells():
    from drybench.realdata import filter_items

    items, _ = squad_items(_squad_paras(), n_per_cell=1, cells=[{"k": 1}, {"k": 2}])
    reader = _chat_reader()
    first, _ = filter_items(reader, items, n_per_cell=1)
    assert len(first) == 2
    kept, stats = filter_items(reader, items, n_per_cell=1, keep=first[:1])  # one cell lost its item
    assert [it.item_id for it in kept] == [it.item_id for it in first]
    st = list(stats.values())
    assert st[0]["kept_before"] == 1 and st[0]["checked"] == 0 and st[1]["kept_before"] == 0 and st[1]["checked"] == 1


def test_human_check_scores_only_responses_with_the_reference_answer(poddar_file, tmp_path):
    """v1.2: a response with another answer is left out; up to v1.1 the reader read its 'own answer' from the
    fragments people labelled minimal answer, so the labels entered the method."""
    rows = [json.loads(line) for line in Path(poddar_file).read_text().splitlines()]
    story = rows[0]["text"].split("\n")[1]
    rows.append(_poddar_row(5, "narrqa_gemma-2-2b-it_17", story, "Who saved Lothair according to Clare?",
                            "The Virgin Mary.", [("Clare thinks Saint Anne saved him. ", "MINANS"),
                                                 ("Lothair travels to Syria later. ", "ADDINFO")]))
    path = tmp_path / "data5.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cases = labelled_cases(load_poddar(path))
    reader = _chat_reader(generate_fn=lambda ctx: "Saint Anne" if "Saint Anne" in ctx else "unknown")
    s = human_label_check(reader, cases, tmp_path / "human", quiet=True)
    assert s["answer_source"] == {"reference": 3} and s["n_scored_ok"] == 3
    assert s["config"]["without_reference_answer"] == 1
    rows = [json.loads(x) for x in (tmp_path / "human" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all("Saint Anne" not in " ".join(r["texts"]) for r in rows)


def test_a_true_sentence_given_as_the_answer_is_a_wrong_answer():
    from drybench.analysis import build_tables
    from drybench.validate import check_reader

    items, _ = squad_items(_squad_paras(), n_per_cell=1, cells=[{"k": 2}])
    it = items[0]
    first, second = it.required_probes
    doc_sentence = "The Norman dynasty had a major political impact on medieval Europe."
    reader = _chat_reader(unknown_fn=lambda m: 0.02 if "dynasty" in m else 0.9)
    resp = f"1. {doc_sentence}\n2. {second.value}."
    res = dry_score(reader, it, resp, mode="loo")
    span = res.spans[0]
    assert not span.kept and span.kind == "misleading" and span.attempt  # numbered as question 1: an answer
    row = {"item_id": it.item_id, "model": "m", "variant": "default", **res.to_dict()}
    _, sp = build_tables([it], [row], [])
    assert sp.iloc[0]["kind"] == "misleading"  # the analysis keeps it: being true is no excuse here
    one, _ = squad_items(_squad_paras(), n_per_cell=1, cells=[{"k": 1}])
    res1 = dry_score(reader, one[0], f"{doc_sentence} {one[0].required_probes[0].value}.", mode="loo")
    assert all(s.kind != "misleading" for s in res1.spans)  # not an answer to it: a true unrequested sentence
    chk = check_reader(reader, items + one, n=5)
    assert {"n_missing", "missing_prior_p_median", "missing_gold_p_median"} <= set(chk)


# ---------------------------------------------------------------- v1.0: the confirmation run

def test_final_runner_both_suites_held_out_and_frozen(cuad_file, poddar_file, tmp_path):
    data = tmp_path / "data"
    for sub, src, name in (("squad2", None, "dev-v2.0.json"), ("musique", MUSIQUE, "musique_full_v1.0_dev.jsonl"),
                           ("cuad", cuad_file, "CUADv1.json"), ("poddar", poddar_file, "data.jsonl")):
        (data / sub).mkdir(parents=True)
        if src is None:
            _squad_file(data / sub / name)
        else:
            (data / sub / name).write_bytes(Path(src).read_bytes())
    # development items: one CUAD contract was used there, so the confirmation run must not use it
    pool = build_real_items("cuad", [data / "cuad" / "CUADv1.json"], 2)[0]
    dev = tmp_path / "dev_items.jsonl"
    dev.write_text(json.dumps(pool[0].to_dict()) + "\n")
    run = tmp_path / "final"
    cmd = [sys.executable, ROOT / "scripts" / "run_final.py", "--run", run, "--data", data, "--dev-items", dev,
           "--backend", "mock", "--allow-cpu", "--models", "m1=echo", "--reader", "mock",
           "--readers2", "r2=mock2", "big=mock3", "--reader-items-per-cell", "big=1", "--judges", "j=mock",
           "--selective", "s=mock", "--variants", "default", "--n-per-cell-synthetic", "1", "--n-per-cell-real", "1",
           "--filler-check-n", "1", "--real-check-n", "1", "--annotation-items", "1", "--annotation-per-item", "2",
           "--annotation-min-distinct", "1", "--abstain-n", "4"]
    out = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-3000:]
    from drybench.realdata import source_key

    real_items = load_items(run / "real" / "items.jsonl")
    assert source_key(pool[0]) not in {source_key(it) for it in real_items}  # held out
    syn = (run / "synthetic" / "tables_synthetic.md").read_text()
    real = (run / "real" / "tables_real.md").read_text()
    for text in (syn, real):
        for part in ("filler map (reader main)", "filler map (reader r2)", "LLM judge (j)", "Selective Context (s)",
                     "shorter but more removable", "Cost per response"):
            assert part in text, part
        assert "DRY" not in text
    assert "## R5." in real and "Main reader vs big:" in real
    for sub in (run / "synthetic", run / "real"):
        assert (sub / "missing_check_main.json").exists() and (sub / "missing_check_big.json").exists()
        assert (sub / "hardware.json").exists()
    ann = run / "annotation"
    for f in ("sample.jsonl", "key.jsonl", "abstain_sample.csv", "abstain_key.jsonl", "labelling_page.html",
              "ANNOTATION.md"):
        assert (ann / f).exists(), f
    key = [json.loads(x) for x in (ann / "key.jsonl").read_text().splitlines()]
    assert key and all(k["status_big"] is not None for k in key)  # the large reader scored the sampled answers
    assert (ann / "baselines" / "judge-j" / "sample.jsonl").exists()
    assert (tmp_path / "final.zip").exists() and (tmp_path / "final_annotation.zip").exists()
    conf = (run / "confirmatory.md").read_text()  # v1.2: the confirmatory tests, provisional without the labels
    assert "| H1 T1 |" in conf and "| H2 R3 |" in conf and "Provisional" in conf
    proto = json.loads((run / "protocol.json").read_text())
    import drybench

    assert proto["version"] == drybench.__version__ and proto["seed"] == 1
    # the protocol is frozen: another seed (or package version) stops the rerun
    again = subprocess.run([str(c) for c in cmd] + ["--seed", "2"], capture_output=True, text=True, timeout=300)
    assert again.returncode != 0 and "keeps its protocol" in again.stdout + again.stderr


def test_colab_final_notebook_is_built():
    spec = importlib.util.spec_from_file_location("make_colab", ROOT / "scripts" / "make_colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cell = mod.build_cell(template="final_template.py")
    assert "__PKG_B64__" not in cell and "run_final.py" in cell and "DRY" not in cell.split("PKG_B64 = ")[0]
    nb = mod.final_notebook(cell)
    assert nb["metadata"]["colab"]["gpuType"] == "A100" and len(nb["cells"]) == 5  # v1.2.1: + the cell "Package"
    import base64
    import io
    import zipfile

    blob = "".join(nb["cells"][1]["source"]).split('PKG_B64 = "', 1)[1].split('"', 1)[0]
    assert blob == cell.split('PKG_B64 = "', 1)[1].split('"', 1)[0]
    names = zipfile.ZipFile(io.BytesIO(base64.b64decode(blob))).namelist()
    for f in ("scripts/run_final.py", "scripts/run_synthetic.py", "drybench/judge.py", "docs/PROTOCOL.md",
              "annotation/dry_span_labels_offline.html"):
        assert f in names, f


# ---------------------------------------------------------------- v1.2: the truth of the planted check

def test_a_sentence_that_only_mentions_an_answer_does_not_state_its_fact():
    """'VerticalNet, Inc., a Pennsylvania corporation' mentions the governing law's answer but does not state
    it: deleting the governing-law clause loses the fact (up to v1.1 the truth counted the mention)."""
    from drybench.realcheck import document_sentences, fact_sentence
    from drybench.simulate import evaluate_filler_map
    from drybench.types import Item, Probe

    doc = ("SUPPLY AGREEMENT\n\nThis Agreement is dated July 1, 2020 and made by VerticalNet, Inc., a Pennsylvania "
           "corporation. The parties signed it on July 1, 2020 in Boston.\n\nThis Agreement is governed by the laws "
           "of the Commonwealth of Pennsylvania.")
    law = Probe("Which jurisdiction's law governs this agreement?", ["Pennsylvania"], fact_id="cuad:x:law")
    date = Probe("What is the date of this agreement?", ["July 1, 2020"], fact_id="cuad:x:date")
    it = Item("cu-x", doc, "Answer: date and law?", [date, law], "1. July 1, 2020.\n2. Pennsylvania.",
              {"source": "cuad", "k": 2}, {"source": "cuad"})
    sents = document_sentences(doc)
    # the date sentence that does not mention the law's answer is preferred
    assert "Pennsylvania" not in fact_sentence(it, date, sents, it.required_probes)[1]
    sim = simulate_real(it, [], rng=random.Random(0))
    facts = {s.text: s.fact_ids for s in sim.spans if s.label == "fact"}
    assert all(len(f) == 1 for f in facts.values())
    law_i = next(i for i, s in enumerate(sim.spans) if "cuad:x:law" in s.fact_ids)
    kept = [i != law_i for i in range(len(sim.spans))]
    assert evaluate_filler_map(sim, kept, [len(s.text) for s in sim.spans])["safety"] < 1.0


def test_planted_recall_counts_responses_without_a_map():
    from drybench.realcheck import _recall_with_unscored

    rows = [{"labels": ["fact", "filler:hedge"], "loo_status": "ok", "loo_kept": [True, False]},
            {"labels": ["fact", "filler:hedge", "filler:closing"], "loo_status": "no_signal"}]
    assert _recall_with_unscored(rows) == pytest.approx(1 / 3)


# ---------------------------------------------------------------- v1.2.2: the development baselines on a server

def test_dev_baselines_cell_on_a_server(cuad_file, poddar_file, tmp_path):
    """The A100 notebook of the development baselines outside Colab: the runs in a folder, nothing installed
    (PIP = False: the package runs from its folder), models named by path; it writes the baselines of every
    truth set, both runs' tables, the development values of the confirmatory tests and the zip."""
    import importlib.util
    import os

    data = tmp_path / "data"
    for sub, src, name in (("squad2", None, "dev-v2.0.json"), ("musique", MUSIQUE, "musique_full_v1.0_dev.jsonl"),
                           ("cuad", cuad_file, "CUADv1.json"), ("poddar", poddar_file, "data.jsonl")):
        (data / sub).mkdir(parents=True)
        if src is None:
            _squad_file(data / sub / name)
        else:
            (data / sub / name).write_bytes(Path(src).read_bytes())
    root = tmp_path / "runs"
    real, syn = root / "real_dev", root / "syn_dev"
    cmd = [sys.executable, ROOT / "scripts" / "run_real.py", "--run", real, "--data", data, "--backend", "mock",
           "--allow-cpu", "--models", "m1=echo", "--reader", "mock", "--reader2", "none", "--variants", "default",
           "--n-per-cell", "1", "--oversample", "1", "--filler-check-n", "1", "--real-check-n", "1",
           "--distractors", "1"]
    out = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    spec = importlib.util.spec_from_file_location("tb", Path(__file__).with_name("test_baselines_v1.py"))
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    syn.mkdir(parents=True)
    tb._synthetic_run(syn)
    models = tmp_path / "models"  # local model folders: no hub calls (the mock backend reads none of them)
    (models / "judge").mkdir(parents=True)
    (models / "small").mkdir()
    spec = importlib.util.spec_from_file_location("make_colab", ROOT / "scripts" / "make_colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    settings = (f'SYN_RUN = "syn_dev"\nREAL_RUN = "real_dev"\nJUDGES = {{"mock": "{models / "judge"}"}}\n'
                f'SELECTIVE = {{"mock": "Qwen/Qwen2.5-3B-Instruct"}}\nREADER = "Qwen/Qwen2.5-3B-Instruct"\n'
                f'COMPARE_N = 3\nCOMPARE_DTYPES = ["fp32"]\n'
                f'LOCAL = {{"Qwen/Qwen2.5-3B-Instruct": "{models / "small"}"}}\nPODDAR_DATA = None\nDTYPE = "fp32"\n'
                f'MAX_BATCH_TOKENS = 4096\nBACKEND = "mock"\nZIP_NAME = "dev_test"\nUSE_DRIVE = False\n'
                f'RUN_ROOT = "{root}"\nGPU = None\nPIP = False\n')
    work = tmp_path / "work"
    work.mkdir()
    (work / "cell.py").write_text(mod.build_cell(settings, template="dev_baselines_template.py"))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # drybench only from the cell's package
    out = subprocess.run([sys.executable, work / "cell.py"], cwd=work, capture_output=True, text=True, timeout=900,
                         env=env)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    assert "nothing installed: the package runs from" in out.stdout and "pip install" not in out.stdout
    for run, sets in ((syn, ["planted_synthetic"]), (real, ["planted_real", "human_poddar"])):
        for name in sets:
            for method in ("judge-mock", "selective-mock"):
                assert (run / "baselines" / method / f"{name}.jsonl").stat().st_size > 0, (run.name, name, method)
    assert "LLM judge (mock)" in (real / "tables_real.md").read_text()
    assert "LLM judge (mock)" in (syn / "tables_synthetic.md").read_text()
    assert "| H1 T1 |" in (root / "dev_test_confirmatory.md").read_text()
    import zipfile

    names = zipfile.ZipFile(root / "dev_test.zip").namelist()
    assert "dev_test_confirmatory.md" in names and any(n.startswith("real_dev/baselines/") for n in names)
    # v1.2.5: the main reader against the runs' scores (the mock reader scores as it did)
    assert "reader_equivalence/summary.md" in names
    summary = json.loads((root / "reader_equivalence" / "summary.json").read_text())
    for run_name in ("syn_dev", "real_dev"):
        s = summary["runs"][run_name]["fp32"]
        assert s["responses"] == 3 and s["same_status"] == 3 and s["same_kept"] == s["with_map_in_both"]
