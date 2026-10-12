"""v1.1 (scoring version 9): a fact the response states in its own words is credited (drybench.forms)."""

import json

from drybench.backends.mock import MockBackend
from drybench.cli import main as cli
from drybench.forms import (MAX_KEEP, answer_spans, candidates, date_spans, prefix_free, stated_spans, term_spans,
                            with_own_wording)
from drybench.reader import Reader, ReaderConfig
from drybench.types import Item, Probe, read_jsonl
from drybench.filler import dry_score

QUESTION = "Who does he call out for a rematch?"


def _reader(chat: bool = False) -> Reader:
    return Reader(MockBackend(p_absent=1e-9), ReaderConfig(use_chat=chat))


def _item(answers=("matthew perry",)) -> Item:
    return Item("t1", "(tweet)", QUESTION, [Probe(QUESTION, list(answers), fact_id="poddar:tweetqa:1",
                                                  value=answers[0])], "matthew perry.")


def test_spans_ignore_case_dashes_quotes_digit_separators_and_articles():
    assert answer_spans("Matthew Perry", ["matthew perry"]) == [(0, 13)]
    r = "2. New York-based filmmakers were drawn to Jacksonville in the 1910s."
    # every form that matches gives its span, overlapping ones too (they end at the same place)
    assert [r[s:e] for s, e in answer_spans(r, ["filmmakers", "New York–based filmmakers"])] == \
        ["New York-based filmmakers", "filmmakers"]
    r = 'The title of this agreement is "Second Amendment to Co-branding Agreement".'
    (s, e), = answer_spans(r, ["SECOND AMENDMENT TO CO-BRANDING AGREEMENT"])
    assert r[s:e] == "Second Amendment to Co-branding Agreement"
    assert answer_spans("They shipped about 1,000 crates.", ["1000 crates"])
    assert answer_spans("Jim Harbaugh sang the National Anthem.", ["national anthem"])
    # whole words only: another answer that contains the letters is not the answer
    assert answer_spans("This tweet is talking about men.", ["women"]) == []
    assert answer_spans("It is about women.", ["men"]) == []


def test_candidates_end_with_the_answer_and_stay_in_its_clause():
    r = "3. The jurisdiction's law governing this agreement is the Internal Laws of the State of Delaware."
    c = [x for x, _ in candidates(r, answer_spans(r, ["Delaware"]))]
    assert c[0] == "Delaware" and all(x.endswith("Delaware") for x in c)
    assert {"State of Delaware", "the Internal Laws of the State of Delaware",
            "Internal Laws of the State of Delaware"} <= set(c)
    r = "Answer: Leonhard Euler."
    assert candidates(r, answer_spans(r, ["Euler"])) == [("Euler", 0), ("Leonhard Euler", 0)]  # ':' ends the phrase
    r = 'The title of this agreement is "Cooperation Agreement."'
    assert '"Cooperation Agreement' in [x for x, _ in candidates(r, answer_spans(r, ["Cooperation Agreement"]))]
    assert prefix_free(["Delaware", "Delaware corporation", "the State of Delaware", "Delaware"]) == \
        ["Delaware", "the State of Delaware"]


def test_cuad_dates_and_terms_match_by_value():
    r = "This Agreement is entered into as of the 1st day of July, 2020, by and between the parties."
    (s, e), = date_spans(r, "July 1, 2020")
    assert r[s:e] == "1st day of July, 2020"
    assert date_spans(r, "July 2, 2020") == []
    assert term_spans("It runs for a period of sixty (60) months.", "five (5) years")
    assert term_spans("It runs for a period of six (6) months.", "five (5) years") == []
    date = Probe("What is the date of this agreement?", ["July 1, 2020", "7/1/2020"], fact_id="cuad:0-1:date",
                 value="July 1, 2020")
    assert stated_spans(r, date) == [(s, e)]


def test_a_fact_in_the_responses_own_words_is_credited_and_kept():
    item, reader = _item(), _reader()
    text = "He calls out Matthew Perry for a rematch. Great question, I hope this helps!"
    old = dry_score(reader, item, text, mode="loo", own_wording=False)
    new = dry_score(reader, item, text, mode="loo")
    assert old.facts == 0.0  # v1.0.0: the gold says 'matthew perry', the reader copies 'Matthew Perry'
    assert new.facts == 1.0 and "Matthew Perry" in new.added_answers[0]
    assert 1 <= len(new.added_answers[0]) <= MAX_KEEP
    assert [s.kept for s in new.spans] == [True, False]  # the answer stays, the filler goes
    # the item passed in is not changed
    assert item.probes[0].answers == ["matthew perry"]


def test_a_different_answer_is_not_credited():
    res = dry_score(_reader(), _item(), "He calls out Matthew Broderick for a rematch.", mode="loo")
    assert res.facts == 0.0 and res.added_answers == [[]]


def test_facts_in_the_golds_words_are_scored_as_before():
    item, reader = _item(), _reader(chat=True)
    text = "Great question! He calls out matthew perry for a rematch. I hope this helps!"
    assert with_own_wording(reader, item, text) == (item, [[]])
    old = dry_score(reader, item, text, mode="loo", own_wording=False).to_dict()
    new = dry_score(reader, item, text, mode="loo").to_dict()
    assert new.pop("added_answers") == [[]] and old.pop("added_answers") == []
    assert json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True)


def test_score_cli_records_added_answers_and_can_score_like_v1_0(tmp_path):
    items, resp = tmp_path / "items.jsonl", tmp_path / "responses.jsonl"
    items.write_text(json.dumps(_item().to_dict()) + "\n")
    resp.write_text(json.dumps({"item_id": "t1", "model": "m", "variant": "default",
                                "response": "He calls out Matthew Perry for a rematch. I hope this helps!"}) + "\n")
    for out, extra in (("new.jsonl", []), ("old.jsonl", ["--gold-forms-only"])):
        cli(["score", "--items", str(items), "--responses", str(resp), "--out", str(tmp_path / out),
             "--reader-backend", "mock", "--mode", "loo"] + extra)
    new, = read_jsonl(tmp_path / "new.jsonl")
    old, = read_jsonl(tmp_path / "old.jsonl")
    assert new["facts"] == 1.0 and "Matthew Perry" in new["added_answers"][0]
    assert old["facts"] == 0.0 and old["added_answers"] == []


def test_runs_scored_before_v1_1_are_rescored_in_full():
    from drybench import SCORING_VERSION
    from drybench.rescore import plan

    assert SCORING_VERSION == "9"
    for reader in ("Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-3.5-mini-instruct", None):
        assert plan("8", "9", reader) == "all" and plan("7", "9", reader) == "all"


# ---------------------------------------------------------------- after review: what must not be credited

def test_the_lead_never_starts_with_another_answer():
    for r in ("He calls out David Schwimmer, not Matthew Perry.", "He calls out David Schwimmer or Matthew Perry.",
              "Rather David Schwimmer than Matthew Perry.", "It was David Schwimmer (Matthew Perry was busy)."):
        c = [x for x, _ in candidates(r, answer_spans(r, ["matthew perry"]))]
        assert c and all(not x.startswith(("David", "Rather", "(")) for x in c), (r, c)
    r = "The initial term is two (2) years, renewable for five (5) years."
    c = [x for x, _ in candidates(r, term_spans(r, "five (5) years"))]
    assert "five (5) years" in c and not any(x.startswith(("two", "renewable", "years")) for x in c), c


def test_numbers_keep_their_symbols_and_bare_numbers_are_not_offered():
    assert answer_spans("They have 5 children.", ["$5"]) == []
    assert answer_spans("It costs $5.", ["$5"])
    assert answer_spans("30% to 65% of the population", ["30% to 65%"])
    r = "A molecule of propane (C3H8) has 3 carbon atoms and 8 hydrogen atoms."
    assert [x for x, _ in candidates(r, answer_spans(r, ["8"]))] == []  # '8' alone would also cover '80'


def test_more_wordings_of_the_same_answer():
    assert answer_spans("It was Euler's proof.", ["Euler"])                       # possessive
    assert answer_spans("New York based filmmakers", ["New York–based filmmakers"])  # hyphen left out
    assert answer_spans("New York – based filmmakers", ["New York-based filmmakers"])
    assert answer_spans("Johnson & Johnson Agreement", ["Johnson and Johnson Agreement"])
    assert answer_spans("Café Rouge", ["Café Rouge"])                       # decomposed accent
    assert answer_spans("Strasse 5", ["Straße 5"])
    r = "THE AGREEMENT SHALL BE GOVERNED BY THE LAWS OF THE STATE OF DELAWARE."
    c = [x for x, _ in candidates(r, answer_spans(r, ["Delaware", "State of Delaware", "the laws of the State of Delaware"]))]
    assert {"DELAWARE", "STATE OF DELAWARE", "THE STATE OF DELAWARE", "LAWS OF THE STATE OF DELAWARE"} <= set(c)
    assert date_spans("Dated 07/01/2020 by the parties", "July 1, 2020") and date_spans("on 2020-07-01", "July 1, 2020")
    assert term_spans("a five-year term", "five (5) years") and term_spans("a 5-year term", "five (5) years")
    assert term_spans("a sixty-month term", "five (5) years")


def test_line_breaks_are_read_as_the_filler_map_reads_them():
    item, reader = _item(), _reader()
    res = dry_score(reader, item, "He calls out Matthew\nPerry for a rematch.\nI hope this helps!", mode="loo")
    assert res.facts == 1.0 and res.added_answers[0] and all("\n" not in a for a in res.added_answers[0])


def test_a_fact_restated_in_other_words_is_a_duplicate_not_a_wrong_statement():
    # a chat reader that answers the name it sees (and says 'unknown' otherwise)
    reader = Reader(MockBackend(p_absent=1e-9, unknown_fn=lambda msg: 0.02 if "Perry" in msg else 0.9),
                    ReaderConfig(use_chat=True))
    item = _item()
    text = "He calls out matthew perry for a rematch. In short: Matthew Perry!"
    old = dry_score(reader, item, text, mode="loo", own_wording=False)
    new = dry_score(reader, item, text, mode="loo")
    assert [s.kept for s in new.spans] == [True, False]
    assert old.spans[1].kind == "misleading" and old.W_misleading > 0  # v1.0.0
    assert new.spans[1].kind == "duplicate" and new.W_misleading == 0
