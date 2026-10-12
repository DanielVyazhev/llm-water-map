import math
import random

import numpy as np
import pytest

from drybench.items import make_item
from drybench.simulate import evaluate_filler_map, min_cover_tokens, simulate
from drybench.types import Item, Probe
from drybench.filler import dry_score, filler_map



def _sharp_reader():
    """Mock reader whose prior (1e-9 per answer form) lies below the 1e-6 floor, so every fact
    carries exactly log(0.8 / 1e-6) nats whatever its number of accepted forms."""
    from drybench.backends.mock import MockBackend
    from drybench.reader import Reader, ReaderConfig

    return Reader(MockBackend(p_absent=1e-9), ReaderConfig())


FLOOR_GAIN = math.log(0.8 / 1e-6)


def test_information_counts_present_answers(item_k2):
    probes = item_k2.required_probes
    msg_both = " ".join(f"It is {p.value}." for p in probes)
    msg_one = f"It is {probes[0].value}."
    info = _sharp_reader().information(item_k2.request, [msg_both, msg_one, ""], probes)
    assert info[0] == pytest.approx(2 * FLOOR_GAIN, abs=1e-6)
    assert info[1] == pytest.approx(FLOOR_GAIN, abs=1e-6)
    assert info[2] == pytest.approx(0.0)


@pytest.mark.parametrize("mode", ["greedy", "loo"])
def test_filler_map_removes_exactly_the_filler(mock_reader, item_k2, mode):
    sim = simulate(item_k2, ["preamble", "restatement", "hedge", "enumeration", "distractor", "closing"],
                   rng=random.Random(0))
    res = filler_map(mock_reader, item_k2.request, sim.spans, item_k2.required_probes, mode=mode)
    kept = [s.kept for s in res.spans]
    truth = [not s.is_filler_truth for s in sim.spans]
    assert kept == truth
    lengths = [s.tokens for s in res.spans]
    ev = evaluate_filler_map(sim, kept, lengths)
    assert ev["safety"] == 1.0 and ev["filler_f1"] == 1.0
    assert res.W == pytest.approx(ev["W_true"])
    assert res.status == "ok"


def test_redundant_summary_keeps_an_irredundant_cover(mock_reader, item_k2):
    sim = simulate(item_k2, ["preamble", "summary", "closing"], rng=random.Random(1))
    res = filler_map(mock_reader, item_k2.request, sim.spans, item_k2.required_probes)
    kept = [s.kept for s in res.spans]
    lengths = [s.tokens for s in res.spans]
    ev = evaluate_filler_map(sim, kept, lengths)
    assert ev["safety"] == 1.0 and ev["filler_recall"] == 1.0
    kept_spans = [s for s, k in zip(sim.spans, kept) if k]
    need = set(sim.present_facts)
    for drop in kept_spans:  # every kept span is needed for coverage
        rest = set().union(*(set(s.fact_ids) for s in kept_spans if s is not drop))
        assert not need <= rest
    assert res.tokens_kept >= min_cover_tokens(sim.spans, lengths, sim.present_facts)


def test_delta_zero_keeps_redundancy(mock_reader, item_k2):
    sim = simulate(item_k2, ["summary"], rng=random.Random(1))
    res = filler_map(mock_reader, item_k2.request, sim.spans, item_k2.required_probes, delta=0.0,
                    abs_tol=0.0)
    # with delta=0 no information may be lost, and a duplicate occurrence adds information
    assert all(s.kept for s in res.spans)


def test_no_signal_when_response_has_no_answers(mock_reader, item_k2):
    res = filler_map(mock_reader, item_k2.request, simulate(item_k2, ["preamble"], omit=[
        p.fact_id for p in item_k2.required_probes]).spans, item_k2.required_probes)
    assert res.status == "no_signal" and math.isnan(res.W)


def test_dry_score_kappa_and_reader_fail(item_k2):
    probes = item_k2.required_probes
    half = simulate(item_k2, ["closing"], omit=[probes[1].fact_id]).text
    reader = _sharp_reader()
    res = dry_score(reader, item_k2, half)
    assert res.kappa == pytest.approx(0.5, abs=1e-6)
    broken = Item("b", item_k2.document, item_k2.request, probes, gold_minimal="Nothing useful.")
    res2 = dry_score(reader, broken, half)
    assert res2.status == "reader_fail" and math.isnan(res2.kappa)


def test_density_and_token_accounting(mock_reader, item_k2):
    text = simulate(item_k2, ["preamble", "closing"]).text
    res = dry_score(mock_reader, item_k2, text)
    assert res.tokens == sum(s.tokens for s in res.spans)
    assert res.tokens_kept + res.removable_tokens == res.tokens
    assert res.density_bits_per_token == pytest.approx(res.info / math.log(2) / res.tokens)


def test_greedy_and_loo_agree_on_random_mixtures(mock_reader):
    rng = random.Random(5)
    for i in range(10):
        it = make_item(random.Random(i), f"r{i}", k=rng.randint(1, 4), d=2)
        filler = rng.sample(["preamble", "restatement", "hedge", "enumeration", "unrequested",
                            "distractor", "closing"], rng.randint(0, 5))
        sim = simulate(it, filler, rng=random.Random(i))
        g = filler_map(mock_reader, it.request, sim.spans, it.required_probes, mode="greedy")
        l = filler_map(mock_reader, it.request, sim.spans, it.required_probes, mode="loo")
        assert g.W == pytest.approx(l.W)
        assert l.n_evals <= 2 * len(sim.spans) + 1


@pytest.mark.parametrize("mode", ["greedy", "loo"])
def test_per_probe_constraint_protects_low_information_facts(item_k2, mode):
    """A fact the reader can half-guess carries few bits; a global budget would drop it."""
    from drybench.backends.mock import MockBackend
    from drybench.reader import Reader, ReaderConfig

    it = make_item(random.Random(3), "lowbit", k=6, d=0)
    guessable = it.required_probes[0].answers[0]
    # gain of the guessable fact: log(0.8 / 0.2) = 1.39 nats vs ~9 nats for the others
    reader = Reader(MockBackend(prior_fn=lambda a: 0.2 if a == guessable else 1e-4), ReaderConfig())
    sim = simulate(it, ["preamble", "closing"], rng=random.Random(0))
    per = filler_map(reader, it.request, sim.spans, it.required_probes, mode=mode)
    glob = filler_map(reader, it.request, sim.spans, it.required_probes, mode=mode, constraint="global",
                     delta=0.05)
    lengths = [s.tokens for s in per.spans]
    assert evaluate_filler_map(sim, [s.kept for s in per.spans], lengths)["safety"] == 1.0
    assert evaluate_filler_map(sim, [s.kept for s in glob.spans], lengths)["safety"] < 1.0


def test_removed_spans_split_into_filler_duplicate_misleading():
    from drybench.backends.mock import MockBackend
    from drybench.reader import Reader, ReaderConfig

    it = make_item(random.Random(7), "cls", k=2, d=0)
    a1, a2 = (p.answers[0] for p in it.required_probes)
    text = " ".join([f"It is {a1}.", f"It is {a2}.", f"In summary: {a1}; {a2}.",
                     "I hope this helps!", "The carrier was WRONG Lines."])
    res = dry_score(Reader(MockBackend(), ReaderConfig(use_chat=True)), it, text)
    kinds = {s.text: s.kind for s in res.spans}
    assert kinds["I hope this helps!"] == "uninformative"
    assert kinds["The carrier was WRONG Lines."] == "misleading"
    for t, k in kinds.items():  # deleted copies of delivered facts are duplicates
        if (a1 in t or a2 in t) and k != "kept":
            assert k == "duplicate"
    assert res.W == pytest.approx(res.W_filler + res.W_misleading)
    assert res.W_filler == pytest.approx(res.W_uninformative + res.W_duplicate)
    assert res.W_misleading > 0 and res.W_duplicate > 0 and res.W_uninformative > 0


def test_plain_reader_counts_all_removed_as_filler(mock_reader, item_k2):
    res = dry_score(mock_reader, item_k2, simulate(item_k2, ["preamble", "closing"]).text)
    assert {s.kind for s in res.spans} <= {"kept", "removed"}
    assert res.W_filler == pytest.approx(res.W)


def test_reader_check_reports_misleading_detection():
    from drybench.backends.mock import MockBackend
    from drybench.reader import Reader, ReaderConfig
    from drybench.validate import check_reader

    items = [make_item(random.Random(i), f"rc{i}", k=2, d=1) for i in range(4)]
    chat = check_reader(Reader(MockBackend(), ReaderConfig(use_chat=True)), items)
    assert chat["misleading_check_active"] == 1.0 and chat["p_unknown_empty_median"] > 0.5
    plain = check_reader(Reader(MockBackend(), ReaderConfig()), items)
    assert math.isnan(plain["misleading_check_active"])


def test_overhead_factorizes(mock_reader, item_k2):
    res = dry_score(mock_reader, item_k2, simulate(item_k2, ["preamble", "restatement", "closing"]).text)
    assert res.tokens_gold > 0 and res.wording == pytest.approx(res.tokens_kept / res.tokens_gold)
    assert res.overhead == pytest.approx(res.tokens / res.tokens_kept * res.wording / res.kappa)
    gold = dry_score(mock_reader, item_k2, item_k2.gold_minimal)
    assert gold.overhead == pytest.approx(1.0) and gold.W == 0


def test_wrong_answer_mass_counts_gold_and_absence_as_safe(mock_reader):
    from drybench.backends.mock import MockBackend
    from drybench.items import NOT_STATED
    from drybench.reader import Reader, ReaderConfig
    from drybench.filler import wrong_answer_mass

    reader = Reader(MockBackend(), ReaderConfig(use_chat=True))
    probes = [Probe("Who was the carrier?", ["Dralor Cargo"]), Probe("Who was the inspector?", [NOT_STATED])]
    msgs = ["The carrier was Dralor Cargo.", "I hope this helps!", "The carrier was WRONG Lines.", ""]
    p = wrong_answer_mass(reader, "Tell me.", msgs, probes)
    assert p[0, 0] < 0.2 and p[1, 0] < 0.2 and p[3, 0] < 0.2  # gold / 'unknown'
    assert p[2, 0] > 0.9                                       # a confident wrong answer
    assert p[1, 1] < 0.2 and p[2, 1] > 0.9  # gold 'not stated' lies inside the absence mass


def test_absolute_tolerance_does_not_scale_with_the_reader_prior():
    """A sentence that makes the reader 8x more sure of the answer is not filler, however small
    the reader's prior is; a relative tolerance (delta * I_j) would delete it when I_j is huge."""
    from drybench.backends.mock import MockBackend
    from drybench.reader import Reader, ReaderConfig
    from drybench.segment import segment

    class SupportReader(MockBackend):
        def continuation_logprobs(self, requests):
            out = []
            for ctx, cont in requests:
                msg = self._message_part(ctx)
                p = 1e-20 if cont.strip() not in msg else (0.8 if "SUPPORT" in msg else 0.1)
                out.append(math.log(p))
            return np.array(out)

    probe = Probe("Who was the carrier?", ["Dralor Cargo"])
    spans = segment("It was Dralor Cargo. SUPPORT confirms it. I hope this helps!")
    plain = Reader(SupportReader(), ReaderConfig(prior_floor=0.0))  # plain PMI: I_j = 45.8 nats
    strict = filler_map(plain, "Carrier?", spans, [probe], mode="greedy")
    loose = filler_map(plain, "Carrier?", spans, [probe], mode="greedy", delta=0.05)
    assert [s.kept for s in strict.spans] == [True, True, False]
    assert [s.kept for s in loose.spans] == [True, False, False]  # 2.1-nat loss < 0.05 * 45.8
    # with the prior floor a fact carries at most log(0.8 / 1e-6) = 13.6 nats, so the relative
    # tolerance stays below 0.7 nats and keeps the supporting sentence
    floored = filler_map(Reader(SupportReader(), ReaderConfig()), "Carrier?", spans, [probe],
                        mode="greedy", delta=0.05)
    assert [s.kept for s in floored.spans] == [True, True, False]


def test_facts_share_counts_answers_the_reader_would_give(mock_reader, item_k2):
    probes = item_k2.required_probes
    full = dry_score(mock_reader, item_k2, simulate(item_k2, ["closing"]).text)
    half = dry_score(mock_reader, item_k2, simulate(item_k2, ["closing"], omit=[probes[1].fact_id]).text)
    assert full.facts == 1.0 and half.facts == 0.5
    assert len(full.per_probe_p) == len(probes) and min(full.gold_per_probe_p) >= 0.5


def test_prior_floor_bounds_information_per_fact(item_k2):
    """Without the floor, a reader sure that the answer is unknown (p ~ 1e-20) makes every fact
    worth ~46 nats and a bare mention of the entity worth tens of nats."""
    from drybench.backends.mock import MockBackend
    from drybench.reader import Reader, ReaderConfig

    probes = item_k2.required_probes
    msg = " ".join(f"It is {p.value}." for p in probes)
    floored = Reader(MockBackend(p_absent=1e-20), ReaderConfig())
    plain = Reader(MockBackend(p_absent=1e-20), ReaderConfig(prior_floor=0.0))
    assert floored.information(item_k2.request, [msg], probes)[0] == pytest.approx(2 * math.log(0.8 / 1e-6), abs=1e-6)
    assert plain.information(item_k2.request, [msg], probes)[0] > 80
    # a message that makes the answer merely less unlikely carries nothing once floored
    faint = Reader(MockBackend(p_absent=1e-20, prior_fn=lambda a: 1e-12), ReaderConfig())
    assert faint.information(item_k2.request, [""], probes)[0] == pytest.approx(0.0)


def test_echoing_the_question_entity_is_not_a_wrong_statement():
    from drybench.backends.mock import MockBackend
    from drybench.reader import Reader, ReaderConfig
    from drybench.filler import ABSENCE_ALIASES, wrong_answer_mass

    it = make_item(random.Random(7), "echo", k=1, d=0)
    sid = it.meta["target"]["sid"]
    # this reader never says 'unknown' about a non-empty message; it copies what it sees
    reader = Reader(MockBackend(unknown_fn=lambda m: 0.9 if "(no message)" in m else 0.02),
                    ReaderConfig(use_chat=True))
    header = f"For shipment {sid}:"
    p = wrong_answer_mass(reader, it.request, [header, ""], it.required_probes)
    assert p[0, 0] < 0.5 and p[1, 0] < 0.5
    assert SHIPMENT_ID_IN(it.required_probes[0].question) == [sid]


def SHIPMENT_ID_IN(text):
    from drybench.filler import SHIPMENT_ID

    return SHIPMENT_ID.findall(text)


def test_misleading_excuse_rules():
    from drybench.filler import misleading_excuse

    it = make_item(random.Random(11), "ex", k=2, d=3)
    t = it.meta["target"]
    other = SHIPMENT_ID_IN(" ".join(it.meta["distractors"]))[0]
    cues = {"inspector": "It was inspected by {}.", "carrier": "It was handled by {}.",
            "origin": "It departed from {}.", "destination": "It was bound for {}."}
    # an attribute the report states although the request does not ask for it
    unrequested = [a for a in cues if a not in it.meta["requested"] and t[a] in it.document][0]
    cue = cues[unrequested]
    assert misleading_excuse(f"For shipment {t['sid']}:", it) == "no new words"
    assert misleading_excuse(f"Shipment {other} was delayed by Brenn Lines.", it) == "other shipment"
    assert misleading_excuse(cue.format(t[unrequested]), it) == "true statement"
    # a true value under another attribute's name is an attribute swap: not excused
    assert misleading_excuse(f"The inspector was {t[unrequested]}.", it) is None
    assert misleading_excuse("The carrier was Blorp Lines.", it) is None


def test_analysis_relabels_excused_misleading_spans():
    from drybench.analysis import build_tables

    it = make_item(random.Random(11), "rl", k=1, d=0)
    sid = it.meta["target"]["sid"]
    row = {"item_id": it.item_id, "model": "m", "variant": "default", "W": 0.5, "W_filler": 0.0,
           "W_misleading": 0.5, "tokens": 10, "tokens_kept": 5, "info": 5.0, "status": "ok",
           "spans": [{"text": f"For shipment {sid}:", "tokens": 5, "kept": False, "kind": "misleading"},
                     {"text": "The carrier was Blorp Lines.", "tokens": 5, "kept": True, "kind": "kept"}]}
    df, spans = build_tables([it], [row], [])
    assert df.loc[0, "W"] == pytest.approx(0.5) and df.loc[0, "W_misleading"] == pytest.approx(0.0)
    assert list(spans["kind"]) == ["uninformative", "kept"]
