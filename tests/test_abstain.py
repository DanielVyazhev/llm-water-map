"""v0.8.2: whether an answer says a missing fact is not given is judged by a lexical rule."""

from drybench.abstain import ABSTAIN, abstains, says_not_given, topics
from drybench.items import generate_items
from drybench.types import Item, Probe

NS = ["not stated", "Not stated"]


def _item(questions, missing_j, request=None, gold=None):
    probes = [Probe(question=q, answers=NS if j == missing_j else ["X"], fact_id=f"f{j}",
                    value="not stated" if j == missing_j else "X") for j, q in enumerate(questions)]
    request = request or ("Answer these questions about the text:\n" + "\n".join(
        f"{i}. {q}" for i, q in enumerate(questions, 1)) if len(questions) > 1 else questions[0])
    gold = gold or "\n".join(f"{i}. {'Not stated in the text' if i - 1 == missing_j else 'X'}."
                             for i in range(1, len(questions) + 1))
    return Item("t-0000", "doc", request, probes, gold, {"k": len(questions), "answerable": False})


def test_answers_from_the_synthetic_run():  # model answers of t4_n20 (fictional names), hand-labelled
    one = next(it for it in generate_items(n_per_cell=1, seed=0) if not it.knobs["answerable"] and it.knobs["k"] == 1)
    q = one.required_probes[0].question.lower()  # e.g. 'which carrier handled shipment fc-2874?'
    noun = {"carrier": "carrier", "port": "origin port", "cargo": "cargo", "date": "departure date",
            "inspected": "inspector", "containers": "number of containers", "delay": "cause of the delay"}
    attr = next(v for k, v in noun.items() if k in q)
    assert abstains(f"The {attr} for this shipment is not mentioned in the provided operations report.", one)
    assert abstains("This information is not provided in the report.", one)  # one question: no topic needed
    assert abstains(f"I'm sorry, but I don't have any information about the {attr} in this report.", one)
    assert not abstains(f"The {attr} for this shipment is Bexyor Docks.", one)
    assert not abstains("The shipment departed on time and did not have any delays mentioned in this report."
                        if "date" not in q else "The shipment did not have any cargo mentioned.", one)
    three = next(it for it in generate_items(n_per_cell=1, seed=0) if not it.knobs["answerable"] and it.knobs["k"] == 3)
    assert abstains(three.gold_minimal, three)


def test_numbered_answers_and_topic():
    it = _item(["What is the title of this agreement?", "Which jurisdiction's law governs this agreement?"], 1)
    assert abstains("1. Endorsement Agreement\n2. The governing law is not specified in the excerpt.", it)
    assert abstains("1. Endorsement Agreement\n2. Not stated in the excerpt.", it)
    assert not abstains("1. The title is not stated.\n2. Delaware law.", it)  # abstains on the other question
    assert abstains("The title is Endorsement Agreement. The contract does not say which state's law governs it.", it)
    assert not abstains("The title is not mentioned anywhere. Delaware law governs.", it)
    tops = topics([p.question for p in it.required_probes], it.gold_minimal)
    assert "gover" in tops[1] and "agree" not in tops[1]  # shared words say nothing about which fact
    assert not says_not_given("2. Delaware law.", it.request, tops[1], 1, 2)


def test_phrasings_and_items_without_a_missing_fact():
    for text in ("The documents do not provide this.", "I couldn't find it in the documents.",
                 "None of the documents mention her birthplace.", "It cannot be determined from the text.",
                 "The cargo is unspecified.", "There is no specific mention of it.", "Unknown."):
        assert ABSTAIN.search(text), text
    for text in ("The shipment was not delayed.", "It is known for its harbour.", "No, it was Delaware."):
        assert not ABSTAIN.search(text), text
    answerable = next(it for it in generate_items(n_per_cell=1, seed=0) if it.knobs["answerable"])
    assert abstains("Not stated.", answerable) is None


def test_phrasings_added_in_v1_2():
    for text in ("I cannot determine the carrier.", "It is not possible to determine the date.",
                 "The date could not be determined.", "Unable to determine.", "The governing law isn't specified.",
                 "It has not been specified.", "There is no governing law clause in the excerpt.", "N/A",
                 "Not applicable."):
        assert ABSTAIN.search(text), text
    it = _item(["What is the title of this agreement?", "Which jurisdiction's law governs this agreement?"], 1)
    assert abstains("1. Endorsement Agreement\n2. The excerpt doesn’t mention the governing law.", it)  # ’
    # a number that is no question's number: the topic decides
    assert abstains("1. Endorsement Agreement\n3. The governing law is not specified in the excerpt.", it)


def test_the_phrase_and_the_topic_stand_in_one_clause():
    two = _item(["Which carrier handled shipment AB-1234?", "What caused the delay of shipment AB-1234?"], 0)
    assert not abstains("The report gives no details about the delay; the carrier was Dralor Cargo.", two)
    assert not abstains("We cannot determine which type of ship it was, but the carrier is Dralor Cargo.", two)
    assert abstains("The carrier is not stated; the delay was caused by a storm.", two)
