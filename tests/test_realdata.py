import json
import math
import os
from pathlib import Path

import pytest

from drybench.backends.mock import MockBackend
from drybench.cli import main
from drybench.items import NOT_STATED, evaluated_model_messages, generate_items, short_answer_messages
from drybench.reader import Reader, ReaderConfig
from drybench.realdata import (MUSIQUE_CELLS, build_real_items, filter_items, load_musique, musique_items,
                               squad_items, text_answer_forms)
from drybench.types import load_items
from drybench.filler import dry_score, misleading_excuse

MUSIQUE = Path(__file__).parent / "data" / "musique_sample.jsonl"


def _squad_fixture():
    ctx = ("The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, "
           "a region in France. They were descended from Norse raiders led by Rollo. The Norman dynasty "
           "had a major political impact on medieval Europe. In 1066 William conquered England.")
    qa = lambda i, q, a, imp=False, pl=(): {  # noqa: E731
        "id": f"q{i}", "question": q, "is_impossible": imp,
        "answers": [] if imp else [{"text": t, "answer_start": 0} for t in a],
        "plausible_answers": [{"text": t, "answer_start": 0} for t in pl]}
    qas = [qa(1, "In what country is Normandy located?", ["France", "France", "France"]),
           qa(2, "Who led the Norse raiders?", ["Rollo", "Rollo"]),
           qa(3, "In what centuries did the Normans name Normandy?", ["10th and 11th centuries", "the 10th and 11th"]),
           qa(4, "When did William conquer England?", ["In 1066", "1066", "1066"]),
           qa(5, "Who conquered Wales in 1066?", [], imp=True, pl=["William"])]
    return [{"pid": "Normans#0", "title": "Normans", "context": ctx, "qas": qas}]


def test_answer_forms_are_prefix_free_with_variants():
    forms = text_answer_forms(["the Denver Broncos", "Denver Broncos", "Denver", "1,000 people"])
    assert "Denver" in forms and "Denver Broncos" not in forms  # "Denver" covers its continuations
    assert "the Denver Broncos" in forms and "The Denver Broncos" in forms
    assert "1000 people" in forms
    for a in forms:
        assert not any(b != a and b.startswith(a) for b in forms)
    assert text_answer_forms(["In 1066"])[:1] == ["1066"]
    assert text_answer_forms(["x"]) == []
    assert set(text_answer_forms(["eleven"])) == {"eleven", "Eleven", "11"}
    assert "six years" in text_answer_forms(["6 years"]) and "7" in text_answer_forms(["seven"])


def test_squad_items_follow_the_knobs():
    paras = _squad_fixture()
    cells = [{"k": 1}, {"k": 3}, {"k": 2, "answerable": False}]
    items, stats = squad_items(paras, n_per_cell=1, cells=cells)
    assert [s["built"] for s in stats.values()] == [1, 1, 1]
    one, three, miss = items
    assert one.request.endswith("?") and len(one.probes) == 1 and one.document.startswith("Normans\n\n")
    assert three.request.startswith("Answer these questions about the text:\n1. ")
    assert len(three.required_probes) == 3 and three.gold_minimal.count("\n") == 2
    for it in items:  # no answer is spelled out in the request
        for p in it.probes:
            if p.value != NOT_STATED:
                assert p.value.lower() not in it.request.lower()
    assert not miss.knobs["answerable"] and miss.knobs["source"] == "squad2"
    m = [p for p in miss.probes if p.value == NOT_STATED]
    assert len(m) == 1 and m[0].answers == [NOT_STATED, "Not stated"]
    assert "Not stated in the text." in miss.gold_minimal and miss.meta["plausible_answers"] == ["William"]
    four = [p for p in three.probes if p.fact_id == "squad2:q4"]
    if four:
        assert four[0].value == "1066"  # the span most annotators chose


def test_squad_real_dev_file_fills_every_cell():
    path = os.environ.get("DRYBENCH_SQUAD")
    if not path or not Path(path).exists():
        pytest.skip("set DRYBENCH_SQUAD to SQuAD 2.0 dev-v2.0.json to run this test")
    items, stats = build_real_items("squad2", path, n_per_cell=20)
    assert all(s["built"] == 20 for s in stats.values())
    assert len({it.item_id for it in items}) == len(items)


def test_musique_items_hops_explain_and_missing():
    inst = load_musique(MUSIQUE)
    cells = [{"hops": 2}, {"hops": 3, "explain": True}]
    items, stats = musique_items(inst, n_per_cell=1, cells=cells, distractors=3)
    assert [s["built"] for s in stats.values()] == [1, 1]
    # MuSiQue-Full's unanswerable version shares its question with the answerable one: a question
    # is used in one cell only, so the missing-fact cell gets it only when built on its own
    (miss,), _ = musique_items(inst, n_per_cell=1, cells=[{"hops": 2, "answerable": False}])
    two, three = items
    assert two.request == "Who is the spouse of the Green performer?"
    assert [p.required for p in two.probes] == [False, True]
    assert two.probes[0].question == "What is the performer of Green?"
    assert two.gold_minimal == "Miquette Giraudy." and two.document.count("Document ") == 5
    assert three.request.endswith("Explain how you found it.") and len(three.required_probes) == 3
    assert three.probes[1].question == "What is the country of Drohobycz?"
    assert three.probes[2].question.startswith("When did the country")
    assert three.gold_minimal == ("What is the place of birth of Kazimierz Wierzynski: Drohobycz. "
                                  "What is the country of Drohobycz: Ukraine. So the answer is 24 August 1991.")
    from drybench.segment import segment
    assert len(segment(three.gold_minimal)) == 3
    assert "August 24, 1991" in three.probes[-1].answers
    assert miss.required_probes[0].answers == [NOT_STATED, "Not stated"]
    assert miss.gold_minimal == "Not stated in the documents." and not miss.knobs["answerable"]
    with pytest.raises(ValueError):
        musique_items(inst, 1, cells=[{"hops": 2, "explain": True, "answerable": False}])


def test_musique_questions_are_used_once():
    inst = load_musique(MUSIQUE)
    items, _ = musique_items(inst, n_per_cell=5, cells=[{"hops": 2}, {"hops": 2, "explain": True}])
    assert len(items) == 1  # the only answerable 2-hop question goes to the first cell


def test_build_musique_without_unanswerable_drops_that_cell(tmp_path):
    ans = tmp_path / "ans.jsonl"
    ans.write_text("".join(l + "\n" for l in MUSIQUE.read_text().splitlines() if '"answerable": true' in l))
    items, stats = build_real_items("musique", ans, n_per_cell=1)
    assert len(stats) == len([c for c in MUSIQUE_CELLS if c.get("answerable", True)])


def test_filter_items_drops_known_and_unreadable():
    paras = _squad_fixture()
    items, _ = squad_items(paras, n_per_cell=1, cells=[{"k": 1}, {"k": 2}, {"k": 2, "answerable": False}])
    # the reader "knows" France without any message
    reader = Reader(MockBackend(prior_fn=lambda a: 0.3 if a == "France" else 1e-4), ReaderConfig())
    kept, stats = filter_items(reader, items, n_per_cell=5)
    has_france = [any("France" in p.answers for p in it.probes) for it in items]
    assert sum(s["known"] for s in stats.values()) == sum(has_france)
    assert all(not any("France" in p.answers for p in it.probes) for it in kept)
    assert all("filter" in it.meta for it in kept)
    # a gold answer that does not state a fact is unreadable
    bad = items[0]
    bad.gold_minimal = "I do not know."
    _, st = filter_items(Reader(MockBackend(), ReaderConfig()), [bad], n_per_cell=5)
    assert list(st.values())[0]["unreadable"] == 1


def test_filter_items_keeps_at_most_n_per_cell():
    inst = load_musique(MUSIQUE)
    items, _ = musique_items(inst, n_per_cell=1, cells=[{"hops": 2}, {"hops": 3}])
    kept, stats = filter_items(Reader(MockBackend(), ReaderConfig()), items + items, n_per_cell=1)
    assert len(kept) == 2 and all(s["kept"] == 1 and s["checked"] == 1 for s in stats.values())


def test_real_items_use_generic_prompts_and_synthetic_ones_do_not_change():
    it = squad_items(_squad_fixture(), 1, cells=[{"k": 2}])[0][0]
    msgs = evaluated_model_messages(it, "concise")
    assert msgs[0]["content"] == "You answer questions using the provided text. Be concise."
    assert msgs[1]["content"].startswith("Text:\nNormans")
    assert "If the answer is not there" in short_answer_messages(it, it.probes[0])[1]["content"]
    syn = generate_items(1)[0]
    assert evaluated_model_messages(syn)[0]["content"] == "You answer questions about internal operations reports."
    assert evaluated_model_messages(syn)[1]["content"].startswith("Report:\n")


def test_misleading_excuse_on_real_text():
    it = squad_items(_squad_fixture(), 1, cells=[{"k": 1}])[0][0]
    assert misleading_excuse("They were descended from Norse raiders led by Rollo.", it) == "true statement"
    assert misleading_excuse("The Normans came from Iceland under King Harald.", it) is None
    assert misleading_excuse(it.request, it) == "no new words"


def test_dry_score_runs_on_real_items():
    it = squad_items(_squad_fixture(), 1, cells=[{"k": 2}])[0][0]
    reader = Reader(MockBackend(p_absent=1e-9), ReaderConfig())
    vals = [p.value for p in it.probes]
    response = f"Sure, happy to help! 1. {vals[0]}. 2. {vals[1]}. Let me know if you need more."
    res = dry_score(reader, it, response, mode="loo")
    assert res.status == "ok" and res.facts == 1.0
    kept = [s.text for s in res.spans if s.kept]
    assert all("happy" not in t and "Let me know" not in t for t in kept)
    assert res.kappa == pytest.approx(1.0, abs=0.05)


def test_cli_real_items_and_filter(tmp_path):
    squad = tmp_path / "squad.json"
    paras = _squad_fixture()
    squad.write_text(json.dumps({"version": "v2.0", "data": [
        {"title": "Normans", "paragraphs": [{"context": paras[0]["context"], "qas": paras[0]["qas"]}]}]}))
    pool1, pool2, out, rep = (tmp_path / n for n in ("p1.jsonl", "p2.jsonl", "items.jsonl", "rep.json"))
    main(["real-items", "--source", "squad2", "--path", str(squad), "--n-per-cell", "2", "--out", str(pool1)])
    main(["real-items", "--source", "musique", "--path", str(MUSIQUE), "--n-per-cell", "2", "--out", str(pool2)])
    main(["filter-items", "--items", str(pool1), str(pool2), "--out", str(out), "--report", str(rep),
          "--n-per-cell", "1", "--reader-backend", "mock"])
    kept = load_items(out)
    report = json.loads(rep.read_text())
    assert kept and {it.knobs["source"] for it in kept} == {"squad2", "musique"}
    assert all(c["kept"] <= 1 for c in report["cells"].values())
    assert math.isclose(report["max_prior"], 0.05)
    # the rest of the pipeline accepts real items
    resp = tmp_path / "resp.jsonl"
    main(["generate", "--items", str(out), "--backend", "mock", "--out", str(resp), "--variant", "default"])
    scored = tmp_path / "scored.jsonl"
    main(["score", "--items", str(out), "--responses", str(resp), "--out", str(scored), "--reader-backend", "mock",
          "--mode", "loo"])
    assert sum(1 for _ in open(scored)) == len(kept)
