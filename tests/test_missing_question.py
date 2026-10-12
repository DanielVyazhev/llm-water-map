"""Ways of asking the reader about a fact the source does not state (missing-check)."""

import json
from dataclasses import replace

from drybench.backends.mock import MockBackend
from drybench.cli import main as cli
from drybench.items import generate_items
from drybench.reader import (MISSING_FORMS, NOT_STATED_HINT, SILENCE_HINT, YES_FORMS, Reader, ReaderConfig,
                             is_missing)
from drybench.validate import format_missing_check, missing_cases, missing_question_check
from drybench.filler import wrong_answer_mass


def _missing_item():
    return next(it for it in generate_items(n_per_cell=1, seed=0)
                if not it.knobs["answerable"] and it.knobs["k"] > 1)


def test_styles_change_the_information_question_only():
    it = _missing_item()
    probe = next(p for p in it.required_probes if is_missing(p))
    other = next(p for p in it.required_probes if not is_missing(p))
    base = Reader(MockBackend(), ReaderConfig(use_chat=True, missing_style="hint"))
    assert base.question(probe) == probe.question + NOT_STATED_HINT
    for style, tail, forms in (("plain", "", MISSING_FORMS), ("silence", SILENCE_HINT, MISSING_FORMS)):
        rd = Reader(base.backend, replace(base.config, missing_style=style))
        assert rd.question(probe) == probe.question + tail and rd.forms(probe) == forms
    yn = Reader(base.backend, replace(base.config, missing_style="yesno"))
    assert yn.question(probe).startswith("Does the message say that this is not stated") and yn.forms(probe) == YES_FORMS
    assert yn.question(other) == other.question and yn.forms(other) == list(other.answers)
    # a wrong answer is read with the content question whatever measures the information
    assert yn.content_question(probe) == base.question(probe)
    msgs = ["The inspector was Ilse Varn.", "Great question!"]
    assert (wrong_answer_mass(yn, it.request, msgs, [probe]) == wrong_answer_mass(base, it.request, msgs, [probe])).all()


def test_missing_question_check_on_gold_and_answers():
    it = _missing_item()
    gold = it.gold_minimal
    abstaining = {"item_id": it.item_id, "model": "m", "variant": "default", "status": "ok",
                  "spans": [{"text": s} for s in gold.split(". ")]}
    silent_only = {"item_id": it.item_id, "model": "m", "variant": "concise", "status": "ok",
                   "spans": [{"text": "Great question!"}]}
    cases = missing_cases([it], [abstaining, silent_only])
    assert [c["kind"] for c in cases] == ["gold", "abstaining answer", "other answer"]
    assert "not stated" in cases[0]["full"] and "not stated" not in cases[0]["silent"] and cases[0]["silent"]
    res = missing_question_check(Reader(MockBackend(), ReaderConfig(use_chat=True)), cases, ["hint", "yesno"])
    for style in ("hint", "yesno"):
        g, a, o = res[style]["gold"], res[style]["abstaining answer"], res[style]["other answer"]
        assert g["readable"] == 1.0 and g["protected"] == 1.0 and a["protected"] == 1.0
        assert o["readable"] == 0.0 and o["n_protected"] == 0
    assert "abstaining answer" in format_missing_check(res)


def test_missing_check_cli(tmp_path):
    out = tmp_path / "mc.json"
    cli(["missing-check", "--synthetic-n", "1", "--reader-backend", "mock", "--chat", "--styles", "hint", "plain",
         "--out", str(out)])
    res = json.loads(out.read_text())
    assert set(res) == {"hint", "plain"} and res["hint"]["gold"]["n"] == 2


def test_new_questions_and_the_verdict():
    from drybench.reader import MISSING_QUESTIONS
    from drybench.validate import missing_check_summary

    it = _missing_item()
    probe = next(p for p in it.required_probes if is_missing(p))
    base = Reader(MockBackend(), ReaderConfig(use_chat=True))
    for style in ("yesno2", "contains", "choice"):
        rd = Reader(base.backend, replace(base.config, missing_style=style))
        assert probe.question in rd.question(probe) and rd.forms(probe) == MISSING_QUESTIONS[style][1]
    abstaining = {"item_id": it.item_id, "model": "m", "variant": "default", "status": "ok",
                  "spans": [{"text": s} for s in it.gold_minimal.split(". ")]}
    res = missing_question_check(base, missing_cases([it], [abstaining]), ["choice", "contains"], show=1)
    assert set(res["choice"]["replies"][0]) == {"question", "full", "silent", "no message"}
    summary = missing_check_summary(res)
    assert summary["choice"]["pass"] and summary["contains"]["kept"] == 1.0
    bad = {"s": {"gold": {"readable": 1.0, "p_empty_median": 0.9},
                 "abstaining answer": {"readable": 1.0, "protected": 0.0}}}
    assert not missing_check_summary(bad)["s"]["pass"]  # abstentions deleted, 'yes' to no message


def test_each_reader_gets_its_missing_fact_question():
    from drybench.reader import DEFAULT_MISSING_STYLE, missing_style_for

    assert ReaderConfig().missing_style == DEFAULT_MISSING_STYLE == "yesno"
    assert missing_style_for("Qwen/Qwen2.5-3B-Instruct") == "yesno"
    assert missing_style_for("microsoft/Phi-3.5-mini-instruct") == "plain"
    assert missing_style_for("/content/models/Phi-3.5-mini-instruct/") == "plain"
    assert missing_style_for("ibm-granite/granite-3.3-2b-instruct") == "yesno"

    class Chat(MockBackend):
        name = "microsoft/Phi-3.5-mini-instruct"

        @property
        def has_chat_template(self):
            return True

    assert Reader.for_backend(Chat()).config.missing_style == "plain"


def test_check_reader_reports_the_silent_gold():
    from drybench.validate import check_reader

    items = [it for it in generate_items(n_per_cell=2, seed=0) if not it.knobs["answerable"]]
    res = check_reader(Reader(MockBackend(), ReaderConfig(use_chat=True)), items, n=4)
    assert res["missing_style"] == "yesno" and res["missing_gold_p_median"] > 0.3
    assert res["missing_silent_p_median"] < 0.01 and res["missing_prior_p_median"] < 0.01
