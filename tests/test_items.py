import random
import re

import pytest

from drybench.items import ATTRS, NOT_STATED, generate_items, make_item
from drybench.types import Item, Probe


def test_probe_aliases_must_be_prefix_free():
    Probe("q", ["Port Vel", "Harbor"])
    with pytest.raises(ValueError):
        Probe("q", ["Port", "Port Vel"])
    with pytest.raises(ValueError):
        Probe("q", ["a", "a"])
    with pytest.raises(ValueError):
        Probe("q", [])


def test_generation_is_deterministic():
    a = [it.to_dict() for it in generate_items(2, seed=3)]
    b = [it.to_dict() for it in generate_items(2, seed=3)]
    c = [it.to_dict() for it in generate_items(2, seed=4)]
    assert a == b
    assert a != c


def test_item_roundtrip(items):
    for it in items:
        assert Item.from_dict(it.to_dict()) == it


@pytest.mark.parametrize("k", [1, 2, 3, 6])
def test_k_probes_and_answers_in_document(k):
    it = make_item(random.Random(k), "x", k=k, d=3)
    assert len(it.required_probes) == k
    for p in it.required_probes:
        assert p.value in it.document               # closed world: answer is in D
        assert p.value not in it.request            # ...but not in the request
        assert p.value in it.gold_minimal           # gold minimal answer states it
        assert all(not re.search(r"(?<![\w-])" + re.escape(a) + r"(?!\w)", it.request) for a in p.answers)
    assert len(it.meta["distractors"]) == 3
    for s in it.meta["distractors"]:
        assert s in it.document


def test_unanswerable_item_hides_one_fact():
    it = make_item(random.Random(1), "x", k=3, d=2, answerable=False)
    missing = it.meta["missing"]
    probes = {p.fact_id.split(":")[1]: p for p in it.probes}
    assert probes[missing].answers == [NOT_STATED]
    assert it.meta["target"][missing] not in it.document
    assert "not stated" in it.gold_minimal


def test_ambiguous_item_has_two_matching_shipments():
    it = make_item(random.Random(2), "x", k=1, d=2, ambiguous=True)
    assert len(it.required_probes) == 2
    origin = it.meta["target"]["origin"]
    assert it.document.count(f"departed from {origin}") == 2
    for p in it.required_probes:
        assert p.answers[0] in it.gold_minimal


@pytest.mark.parametrize("explain", [False, True])
def test_two_hop_intermediate_probe(explain):
    it = make_item(random.Random(3), "x", k=1, d=2, hops=2, explain=explain)
    carrier_probe = [p for p in it.probes if p.fact_id.endswith(":carrier")][0]
    assert carrier_probe.required is explain
    assert len(it.required_probes) == (2 if explain else 1)
    home = it.meta["carrier_home"]
    assert home in it.document and home in it.gold_minimal


def test_invalid_knob_combinations():
    with pytest.raises(ValueError):
        make_item(random.Random(0), "x", k=2, ambiguous=True)
    with pytest.raises(ValueError):
        make_item(random.Random(0), "x", k=0)
    with pytest.raises(ValueError):
        make_item(random.Random(0), "x", k=1, hops=2, answerable=False)


def test_all_attributes_have_templates():
    for spec in ATTRS.values():
        assert "{sid}" in spec.question and "{v}" in spec.doc and "{v}" in spec.resp


def test_numbers_do_not_repeat_across_facts():
    from drybench.items import NUMERIC_ATTRS, _numbers

    for it in generate_items(3, seed=5):
        target = it.meta["target"]
        seen = set()
        for a in NUMERIC_ATTRS:
            nums = _numbers(target[a])
            assert not nums & seen
            seen |= nums
        for sent in it.meta["distractors"]:
            assert not _numbers(re.sub(r"[A-Z]{2}-\d{4}", "", sent)) & seen  # ignore shipment IDs


def test_answer_forms_cover_how_readers_reply():
    from drybench.items import answer_forms

    assert answer_forms("delay_cause", "a crane failure") == [
        "a crane failure", "A crane failure", "crane failure", "Crane failure"]
    assert answer_forms("containers", "23 containers") == ["23"]  # also covers "23 containers"
    assert answer_forms("departure_date", "14 July") == ["14 July", "July 14"]
    assert answer_forms("departure_date", "2 July") == ["2 July"]  # "July 2" would cover "July 20"
    assert answer_forms("carrier", "Dralor Cargo") == ["Dralor Cargo"]
    for it in generate_items(3, seed=1):
        for p in it.probes:
            Probe(p.question, p.answers)  # prefix-free
            assert p.value is not None


def test_refresh_probes_keeps_documents_and_updates_answers(tmp_path):
    from drybench.items import refresh_probes

    old = generate_items(2, seed=0)
    for it in old:  # simulate items written by an older generator: single canonical answer
        for p in it.probes:
            p.answers, p.value = [p.value], None
    new = refresh_probes(old, seed=0)
    assert [i.document for i in new] == [i.document for i in old]
    assert new == generate_items(2, seed=0)
    with pytest.raises(ValueError):
        refresh_probes(old, seed=1)
