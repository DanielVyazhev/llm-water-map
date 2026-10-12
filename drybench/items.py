"""Closed-world item generator with controlled complexity knobs.

Items describe fictional shipments in an operations report. All names are
synthetic, so a reader model cannot know the answers without the response:
the information in the response is the only route from the document to the
reader. This is what makes the reader-based information measure valid.

Knobs
-----
k          number of facts the request needs (1..6)
d          number of distractor sentences about other shipments
answerable if False, one requested fact is missing from the document; the
           ideal response says it is not stated
ambiguous  the request matches two shipments; the ideal response covers both
           briefly (or asks which one)
hops       1 = direct lookup; 2 = lookup through the carrier's record
explain    for hops=2: whether the intermediate fact is required
"""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass
from typing import Any

from drybench.types import Item, Probe

SYLLABLES = [
    "ka", "ve", "lor", "dra", "mi", "sto", "rel", "van", "tu", "bex", "ol", "qui", "zan",
    "pel", "ros", "thi", "gor", "nal", "vi", "dun", "mar", "set", "ul", "ob", "fen", "rak",
    "sil", "yor", "esk", "ta",
]
CARRIER_SUFFIX = ["Freight", "Lines", "Logistics", "Shipping", "Cargo"]
PORT_PATTERNS = ["Port {n}", "{n} Harbor", "{n} Docks"]
UNITS = ["crates", "pallets", "drums", "bales", "coils"]
MATERIALS = [
    "copper wire", "tin ingots", "glass panels", "rubber sheets", "cotton yarn",
    "steel bolts", "olive oil", "ceramic tiles",
]
DELAY_CAUSES = [
    "a customs audit", "a crane failure", "a labour dispute", "a missing bill of lading",
    "a ballast pump fault", "a quarantine inspection", "a berth reassignment", "a fuel shortage",
]
MONTHS = [
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
]
BOILERPLATE = [
    "All times are local.",
    "Figures are provisional until the quarterly audit.",
    "This report covers shipments processed by the terminal.",
]
NOT_STATED = "not stated"


@dataclass(frozen=True)
class AttrSpec:
    noun: str
    question: str
    doc: str
    resp: str


ATTRS: dict[str, AttrSpec] = {
    "carrier": AttrSpec(
        "carrier", "Which carrier handled shipment {sid}?",
        "Shipment {sid} was handled by {v}.", "The carrier was {v}."),
    "origin": AttrSpec(
        "origin port", "From which port did shipment {sid} depart?",
        "Shipment {sid} departed from {v}.", "It departed from {v}."),
    "destination": AttrSpec(
        "destination port", "To which port was shipment {sid} bound?",
        "Shipment {sid} was bound for {v}.", "It was bound for {v}."),
    "cargo": AttrSpec(
        "cargo", "What cargo did shipment {sid} carry?",
        "Shipment {sid} carried {v}.", "It carried {v}."),
    "delay_cause": AttrSpec(
        "cause of the delay", "What caused the delay of shipment {sid}?",
        "The delay of shipment {sid} was caused by {v}.", "The delay was caused by {v}."),
    "inspector": AttrSpec(
        "inspector", "Who inspected shipment {sid}?",
        "Shipment {sid} was inspected by {v}.", "It was inspected by {v}."),
    "departure_date": AttrSpec(
        "departure date", "On what date did shipment {sid} depart?",
        "Shipment {sid} left on {v}.", "It left on {v}."),
    "containers": AttrSpec(
        "number of containers", "How many containers did shipment {sid} use?",
        "Shipment {sid} used {v}.", "It used {v}."),
}
ATTR_NAMES = list(ATTRS)
MAX_K = len(ATTR_NAMES) - 2  # keep two target attributes available as unrequested info


class _Namer:
    """Generates unique synthetic proper names within one item."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used: set[str] = set()

    def word(self, n_syll: int | None = None) -> str:
        for _ in range(1000):
            n = n_syll or self.rng.choice([2, 3])
            w = "".join(self.rng.choice(SYLLABLES) for _ in range(n)).capitalize()
            if w not in self.used:
                self.used.add(w)
                return w
        raise RuntimeError("name space exhausted")

    def shipment_id(self) -> str:
        for _ in range(1000):
            sid = "".join(self.rng.choice("BCDFGHJKLMNPRSTVXZ") for _ in range(2)) + "-" + str(
                self.rng.randint(1000, 9999))
            if sid not in self.used:
                self.used.add(sid)
                return sid
        raise RuntimeError("id space exhausted")


def _value(attr: str, rng: random.Random, namer: _Namer) -> str:
    if attr == "carrier":
        return f"{namer.word()} {rng.choice(CARRIER_SUFFIX)}"
    if attr in ("origin", "destination"):
        return rng.choice(PORT_PATTERNS).format(n=namer.word())
    if attr == "cargo":
        return f"{rng.randint(12, 97)} {rng.choice(UNITS)} of {rng.choice(MATERIALS)}"
    if attr == "delay_cause":
        return rng.choice(DELAY_CAUSES)
    if attr == "inspector":
        return f"{namer.word(2)} {namer.word()}"
    if attr == "departure_date":
        return f"{rng.randint(2, 28)} {rng.choice(MONTHS)}"
    if attr == "containers":  # unit included so the answer never collides with ID digits
        return f"{rng.randint(12, 64)} containers"
    raise KeyError(attr)


NUMERIC_ATTRS = ("cargo", "departure_date", "containers")


def answer_forms(attr: str, value: str) -> list[str]:
    """Prefix-free surface forms a reader may reply with. Readers drop articles, change case
    and give bare numbers, so a single canonical string under-counts delivered facts."""
    if attr == "delay_cause":
        base = value[2:] if value.startswith("a ") else value
        forms = [f"a {base}", f"A {base}", base, base[:1].upper() + base[1:]]
    elif attr == "containers":
        forms = [value.split()[0]]  # "23" also covers "23 containers"
    elif attr == "departure_date":
        day, month = value.split()
        forms = [value] + ([f"{month} {day}"] if int(day) >= 10 else [])  # "July 2" would cover "July 20"
    else:
        forms = [value]
    out: list[str] = []
    for f in forms:
        if f not in out:
            out.append(f)
    return out


def _numbers(value: str) -> set[str]:
    return set(re.findall(r"\d+", value))


def _shipment(rng: random.Random, namer: _Namer, avoid_numbers: set[str] | None = None) -> dict[str, str]:
    """A shipment record. Numbers never repeat across its numeric attributes (nor across
    `avoid_numbers`), so a reader cannot pick up a number cue from an unrelated fact."""
    rec = {"sid": namer.shipment_id()}
    for a in ATTR_NAMES:
        rec[a] = _value(a, rng, namer)
    used = set(avoid_numbers or ())
    for a in NUMERIC_ATTRS:
        for _ in range(200):
            if not _numbers(rec[a]) & used:
                break
            rec[a] = _value(a, rng, namer)
        used |= _numbers(rec[a])
    return rec


def _join_nouns(nouns: list[str]) -> str:
    nouns = [f"the {n}" for n in nouns]
    if len(nouns) == 1:
        return nouns[0]
    return ", ".join(nouns[:-1]) + " and " + nouns[-1]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def make_item(
    rng: random.Random,
    item_id: str,
    k: int = 1,
    d: int = 0,
    answerable: bool = True,
    ambiguous: bool = False,
    hops: int = 1,
    explain: bool = False,
) -> Item:
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be in [1, {MAX_K}]")
    if hops not in (1, 2):
        raise ValueError("hops must be 1 or 2")
    if (ambiguous or hops == 2) and k != 1:
        raise ValueError("ambiguous and 2-hop items use k=1")
    if ambiguous and hops == 2:
        raise ValueError("ambiguous and 2-hop cannot be combined")

    namer = _Namer(rng)
    target = _shipment(rng, namer)
    sid = target["sid"]
    doc_sents: list[str] = []
    probes: list[Probe] = []
    gold_parts: list[str] = []
    meta: dict[str, Any] = {"target": target}

    if ambiguous:
        twin = _shipment(rng, namer, avoid_numbers=set().union(*(_numbers(target[a]) for a in NUMERIC_ATTRS)))
        twin["origin"] = target["origin"]
        while twin["delay_cause"] == target["delay_cause"]:
            twin["delay_cause"] = rng.choice(DELAY_CAUSES)
        request = f"What caused the delay of the shipment that departed from {target['origin']}?"
        gold_bits = []
        for rec in (target, twin):
            for a in ("origin", "delay_cause", "carrier"):
                doc_sents.append(ATTRS[a].doc.format(sid=rec["sid"], v=rec[a]))
            probes.append(Probe(
                question=ATTRS["delay_cause"].question.format(sid=rec["sid"]),
                answers=answer_forms("delay_cause", rec["delay_cause"]),
                fact_id=f"{rec['sid']}:delay_cause", value=rec["delay_cause"]))
            gold_bits.append(f"{rec['sid']} ({rec['delay_cause']})")
        gold_parts.append(
            f"Two shipments departed from {target['origin']}: " + " and ".join(gold_bits) + ".")
        meta["twin"] = twin
        requested = ["delay_cause"]
    elif hops == 2:
        home = rng.choice(PORT_PATTERNS).format(n=namer.word())
        meta["carrier_home"] = home
        request = f"What is the home port of the carrier that handled shipment {sid}?"
        if explain:
            request += " Explain how you found it."
        doc_sents.append(ATTRS["carrier"].doc.format(sid=sid, v=target["carrier"]))
        doc_sents.append(f"{target['carrier']} is based in {home}.")
        probes.append(Probe(
            question=f"What is the home port of the carrier that handled shipment {sid}?",
            answers=[home], fact_id=f"{sid}:carrier_home", value=home))
        probes.append(Probe(
            question=ATTRS["carrier"].question.format(sid=sid),
            answers=[target["carrier"]], fact_id=f"{sid}:carrier", required=explain,
            value=target["carrier"]))
        if explain:
            gold_parts.append(f"The carrier was {target['carrier']}, which is based in {home}.")
        else:
            gold_parts.append(f"Home port: {home}.")
        requested = ["carrier"]
        if not answerable:
            raise ValueError("unanswerable 2-hop items are not supported")
    else:
        requested = rng.sample(ATTR_NAMES, k)
        missing = rng.choice(requested) if not answerable else None
        meta["missing"] = missing
        request = f"For shipment {sid}, what was {_join_nouns([ATTRS[a].noun for a in requested])}?"
        if k > 1:
            request = request.replace("what was", "what were", 1)
        for a in requested:
            spec = ATTRS[a]
            if a == missing:
                answer = NOT_STATED
                gold_parts.append(f"{_cap(spec.noun)}: not stated in the report.")
            else:
                answer = target[a]
                doc_sents.append(spec.doc.format(sid=sid, v=target[a]))
                gold_parts.append(f"{_cap(spec.noun)}: {target[a]}.")
            forms = [answer] if answer == NOT_STATED else answer_forms(a, answer)
            probes.append(Probe(question=spec.question.format(sid=sid), answers=forms,
                                fact_id=f"{sid}:{a}", value=answer))

    # Unrequested facts about the target (available for over-elaboration).
    extra = [a for a in ATTR_NAMES if a not in requested and a != meta.get("missing")]
    unrequested: list[str] = []
    for a in rng.sample(extra, min(2, len(extra))):
        s = ATTRS[a].doc.format(sid=sid, v=target[a])
        if s not in doc_sents:
            doc_sents.append(s)
            unrequested.append(s)
    meta["unrequested"] = unrequested

    # Distractors: facts about other shipments.
    distractors: list[str] = []
    taken = {v for rec in (target, meta.get("twin", {})) for key, v in rec.items() if key != "sid"}
    taken_numbers = set().union(*(_numbers(v) for v in taken))
    for _ in range(d):
        other = _shipment(rng, namer)
        a = rng.choice(ATTR_NAMES)
        for _ in range(200):  # never repeat a target value or number (e.g. same delay cause)
            if other[a] not in taken and not _numbers(other[a]) & taken_numbers:
                break
            other[a] = _value(a, rng, namer)
        distractors.append(ATTRS[a].doc.format(sid=other["sid"], v=other[a]))
    meta["distractors"] = distractors
    doc_sents += distractors
    doc_sents.append(rng.choice(BOILERPLATE))
    rng.shuffle(doc_sents)
    header = f"Operations report, {namer.word()} terminal."
    document = header + "\n" + "\n".join(doc_sents)

    knobs = {"k": k, "d": d, "answerable": answerable, "ambiguous": ambiguous,
             "hops": hops, "explain": explain}
    meta["requested"] = requested
    return Item(item_id=item_id, document=document, request=request, probes=probes,
                gold_minimal=" ".join(gold_parts), knobs=knobs, meta=meta)


DEFAULT_CELLS: list[dict[str, Any]] = (
    [{"k": k, "d": d} for k, d in itertools.product([1, 2, 3, 4, 6], [0, 4])]
    + [{"k": k, "d": 4, "answerable": False} for k in (1, 3)]
    + [{"k": 1, "d": 4, "ambiguous": True}]
    + [{"k": 1, "d": 4, "hops": 2, "explain": e} for e in (False, True)]
)


def refresh_probes(items: list[Item], seed: int = 0,
                   cells: list[dict[str, Any]] | None = None) -> list[Item]:
    """Regenerate items by id with the current generator and take their probes (answer forms).

    Documents, requests and gold answers must come out identical, so responses generated for
    the old items stay valid; anything else raises."""
    cells = DEFAULT_CELLS if cells is None else cells
    out = []
    for it in items:
        m = re.fullmatch(r"c(\d+)-(\d+)", it.item_id)
        if not m:
            raise ValueError(f"cannot regenerate item {it.item_id!r}")
        ci, j = int(m.group(1)), int(m.group(2))
        new = make_item(random.Random(f"{seed}-{ci}-{j}"), item_id=it.item_id, **cells[ci])
        if (new.document, new.request, new.gold_minimal) != (it.document, it.request, it.gold_minimal):
            raise ValueError(f"item {it.item_id} differs from the current generator (seed {seed})")
        out.append(Item(it.item_id, it.document, it.request, new.probes, it.gold_minimal, it.knobs, it.meta))
    return out


def generate_items(n_per_cell: int, seed: int = 0,
                   cells: list[dict[str, Any]] | None = None) -> list[Item]:
    """Generate n_per_cell items for each knob configuration (deterministic in seed)."""
    cells = DEFAULT_CELLS if cells is None else cells
    items = []
    for ci, cell in enumerate(cells):
        for j in range(n_per_cell):
            rng = random.Random(f"{seed}-{ci}-{j}")
            items.append(make_item(rng, item_id=f"c{ci:02d}-{j:04d}", **cell))
    return items


def _doc_words(item: Item) -> tuple[str, str | None]:
    """(label, noun) of the document; noun is None for synthetic operations reports."""
    noun = item.meta.get("doc_noun")
    return (item.meta.get("doc_label") or "Report", noun) if noun else ("Report", None)


def evaluated_model_messages(item: Item, variant: str = "default") -> list[dict[str, str]]:
    """Chat messages for the evaluated model. `variant` controls the style instruction."""
    label, noun = _doc_words(item)
    system = ("You answer questions about internal operations reports." if noun is None
              else f"You answer questions using the provided {noun}.")
    if variant == "concise":
        system += " Be concise."
    elif variant != "default":
        raise ValueError(f"unknown prompt variant {variant!r}")
    user = f"{label}:\n{item.document}\n\nRequest: {item.request}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def short_answer_messages(item: Item, probe: Probe) -> list[dict[str, str]]:
    """Messages for the short-answer pass used to compute H_ans (same document the model saw)."""
    label, noun = _doc_words(item)
    if noun is None:
        system = "You answer questions about internal operations reports. Reply with a few words only."
        tail = f"If the report does not say, reply '{NOT_STATED}'."
    else:
        system = f"You answer questions using the provided {noun}. Reply with a few words only."
        tail = f"If the answer is not there, reply '{NOT_STATED}'."
    user = f"{label}:\n{item.document}\n\nQuestion: {probe.question}\n{tail}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
