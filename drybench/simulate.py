"""Response simulator with ground-truth filler labels (metric validation).

A simulated response = fact sentences for the required probes + typed filler spans.
Every span carries a label ("fact" or "filler:<type>") and the fact ids it states, so
the true minimal response y* (minimum-token subset of spans that still states every
present fact) and the true filler fraction W* are computable exactly.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from drybench.items import ATTRS, NOT_STATED
from drybench.types import Item, Probe, Span

FILLER_TYPES = ("preamble", "restatement", "elaboration", "hedge", "enumeration",
               "unrequested", "distractor", "summary", "closing")
# Expected to be formulaic (low generation entropy) vs content-like.
FORMULAIC = {"preamble", "restatement", "elaboration", "hedge", "summary", "closing"}

TEMPLATES = {
    "preamble": [
        "Great question!", "Sure, let me look at the report for you.",
        "Thanks for asking, here is what the report says.", "Certainly, I checked the report.",
    ],
    "restatement": [
        "You asked about {nouns} of shipment {sid}.",
        "Here is the information you requested about shipment {sid}.",
    ],
    "elaboration": [
        "Shipment {sid} is one of the shipments covered in this report.",
        "The report lists several details for this shipment.",
        "This information comes directly from the operations report.",
    ],
    "hedge": [
        "Please note that the report may not capture every detail.",
        "It is always a good idea to double-check this with the logistics team.",
        "Keep in mind that operational records can occasionally be incomplete.",
    ],
    "enumeration": [
        "In general, delays can be caused by weather, port congestion or paperwork issues.",
        "Carriers, ports and cargo types can vary a lot between shipments.",
        "Shipping schedules often depend on factors such as crew availability and tides.",
    ],
    "closing": [
        "I hope this helps!", "Let me know if you need anything else.",
        "Feel free to ask if you have more questions.",
    ],
}


@dataclass
class SimResponse:
    text: str
    spans: list[Span]
    present_facts: list[str]
    filler_types: list[str] = field(default_factory=list)


def fact_sentence(item: Item, probe: Probe) -> str:
    sid, _, attr = (probe.fact_id or "::").partition(":")
    answer = probe.value or probe.answers[0]
    if attr in ATTRS:
        spec = ATTRS[attr]
        if answer == NOT_STATED:
            return f"The report does not state the {spec.noun}."
        sent = spec.resp.format(v=answer)
        if item.knobs.get("ambiguous"):
            sent = f"For shipment {sid}, " + sent[0].lower() + sent[1:]
        return sent
    if attr == "carrier_home":
        return f"The carrier is based in {answer}."
    return f"{probe.question} {answer}."


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:]


def simulate(item: Item, filler: Sequence[str] = (), omit: Iterable[str] = (),
             rng: random.Random | None = None) -> SimResponse:
    """Build a response for `item` with the given filler span types (repeats allowed)."""
    rng = rng or random.Random(0)
    omit = set(omit)
    unknown = set(filler) - set(FILLER_TYPES)
    if unknown:
        raise ValueError(f"unknown filler types {unknown}")
    target = item.meta.get("target", {})
    sid = target.get("sid", "the shipment")
    nouns = " and ".join(f"the {ATTRS[a].noun}" for a in item.meta.get("requested", []) if a in ATTRS)
    facts = [(p, fact_sentence(item, p)) for p in item.required_probes if p.fact_id not in omit]
    present = [p.fact_id for p, _ in facts]

    def tmpl(kind: str) -> str:
        return rng.choice(TEMPLATES[kind]).format(sid=sid, nouns=nouns or "the details")

    head, body, tail = [], [], []
    body = [(s, "fact", (p.fact_id,)) for p, s in facts]
    distractors = list(item.meta.get("distractors", []))
    unrequested = list(item.meta.get("unrequested", []))
    rng.shuffle(distractors)
    rng.shuffle(unrequested)
    used: set[str] = set()
    for kind in filler:
        if kind in ("preamble", "restatement"):
            head.append((tmpl(kind), f"filler:{kind}", ()))
        elif kind in ("elaboration", "hedge", "enumeration"):
            body.insert(rng.randint(0, len(body)), (tmpl(kind), f"filler:{kind}", ()))
        elif kind in ("distractor", "unrequested"):
            # without another shipment in the report a distractor becomes an unrequested fact (and the
            # reverse), labelled by what it is; a sentence is never planted twice in one response
            other = "unrequested" if kind == "distractor" else "distractor"
            src = kind if (distractors if kind == "distractor" else unrequested) else other
            fresh = [s for s in (distractors if src == "distractor" else unrequested) if s not in used]
            if not fresh:
                continue
            used.add(fresh[0])
            tail.append(("The report also mentions that " + _lower_first(fresh[0]), f"filler:{src}", ()))
        elif kind == "summary":
            vals = "; ".join(p.value or p.answers[0] for p, _ in facts)
            if vals:
                tail.append((f"In summary: {vals}.", "filler:summary", tuple(present)))
        elif kind == "closing":
            tail.append((tmpl(kind), f"filler:{kind}", ()))
    # preamble before restatement; summaries before closings
    head.sort(key=lambda x: x[1] != "filler:preamble")
    tail_rank = {"filler:unrequested": 0, "filler:distractor": 0, "filler:summary": 1, "filler:closing": 2}
    tail.sort(key=lambda x: tail_rank[x[1]])
    parts = head + body + tail

    spans, pos, pieces = [], 0, []
    for text, label, fids in parts:
        if pieces:
            pos += 1
        spans.append(Span(pos, pos + len(text), text, label=label, fact_ids=tuple(fids)))
        pieces.append(text)
        pos += len(text)
    return SimResponse(text=" ".join(pieces), spans=spans, present_facts=present,
                       filler_types=list(filler))


def min_cover_tokens(spans: Sequence[Span], lengths: Sequence[int], facts: Sequence[str]) -> int:
    """Minimum total length of a span subset that states every fact in `facts`."""
    need = set(facts)
    if not need:
        return 0
    carriers = [i for i, s in enumerate(spans) if set(s.fact_ids) & need]
    best = None
    for r in range(1, len(carriers) + 1):
        for combo in itertools.combinations(carriers, r):
            covered = set().union(*(set(spans[i].fact_ids) for i in combo))
            if need <= covered:
                cost = sum(lengths[i] for i in combo)
                best = cost if best is None else min(best, cost)
    if best is None:
        raise ValueError("facts not coverable by spans")
    return best


def true_filler_fraction(sim: SimResponse, lengths: Sequence[int]) -> float:
    total = sum(lengths)
    return 1.0 - min_cover_tokens(sim.spans, lengths, sim.present_facts) / total


def evaluate_filler_map(sim: SimResponse, kept: Sequence[bool], lengths: Sequence[int]) -> dict:
    """Compare a predicted kept-mask (aligned with sim.spans) to the ground truth.

    safety       share of present facts still stated by kept spans (1.0 = no information lost)
    token_recall removed tokens / removable tokens (removable = total - min cover)
    filler_*     precision / recall / F1 over pure filler spans (spans stating no fact);
                 removing a fact span counts as correct only if its facts stay covered.
    """
    if len(kept) != len(sim.spans):
        raise ValueError("kept mask must align with simulated spans")
    total = sum(lengths)
    need = set(sim.present_facts)
    covered = set().union(*(set(s.fact_ids) for s, k in zip(sim.spans, kept) if k)) if any(kept) else set()
    safety = len(need & covered) / len(need) if need else 1.0
    removable = total - min_cover_tokens(sim.spans, lengths, sim.present_facts)
    removed = sum(l for l, k in zip(lengths, kept) if not k)
    pure = [not s.fact_ids for s in sim.spans]
    removed_idx = [i for i, k in enumerate(kept) if not k]
    ok_removed = [i for i in removed_idx if pure[i] or set(sim.spans[i].fact_ids) <= covered]
    precision = len(ok_removed) / len(removed_idx) if removed_idx else 1.0
    n_pure = sum(pure)
    recall = sum(1 for i in removed_idx if pure[i]) / n_pure if n_pure else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "safety": safety,
        "token_recall": removed / removable if removable else 1.0,
        "filler_precision": precision,
        "filler_recall": recall,
        "filler_f1": f1,
        "W_true": removable / total if total else 0.0,
    }


def _is_pure_filler(label: str) -> bool:
    """A planted span that states no fact (a summary repeats facts, so it is not pure filler)."""
    return label.startswith("filler:") and label != "filler:summary"


def filler_check_metrics(rows: Sequence[dict], mode: str = "loo", n_boot: int = 2000, seed: int = 0) -> dict:
    """Filler-map check pooled over simulated responses (rows written by scripts/pilot_validation.py).

    Per-response recall is 1.0 by definition for a response without filler, so its plain mean over
    all responses overstates recall; this reports recall over the planted filler spans themselves
    (pooled, with a 95% bootstrap CI over items) and over responses that contain filler.
    Repeats (the summary span) are counted apart: a repeat is resolved when, after deletion, no
    fact is stated twice.
    """
    import numpy as np

    ok = [r for r in rows if r.get(f"{mode}_status") == "ok"]
    if not ok:
        return {"n_responses": 0}
    per_item: dict[str, list] = {}
    by_type: dict[str, list[int]] = {}
    lost = removed_n = removed_ok = 0
    repeats = resolved = 0
    for r in ok:
        labels, kept = r["labels"], r[f"{mode}_kept"]
        pure = [i for i, lab in enumerate(labels) if _is_pure_filler(lab)]
        gone = [i for i in pure if not kept[i]]
        for i in pure:
            t = labels[i].split(":", 1)[1]
            by_type.setdefault(t, [0, 0])
            by_type[t][0] += not kept[i]
            by_type[t][1] += 1
        n_removed = sum(1 for k in kept if not k)
        removed_n += n_removed
        removed_ok += round(r[f"{mode}_filler_precision"] * n_removed)
        lost += r[f"{mode}_safety"] < 1.0
        summ = [i for i, lab in enumerate(labels) if lab == "filler:summary"]
        if summ:
            repeats += 1
            facts_kept = any(kept[i] for i, lab in enumerate(labels) if lab == "fact")
            resolved += all(not kept[i] for i in summ) or not facts_kept
        per_item.setdefault(r["item_id"], []).append((len(gone), len(pure)))

    def pooled(groups):
        g = sum(a for grp in groups for a, _ in grp)
        n = sum(b for grp in groups for _, b in grp)
        return g / n if n else float("nan")

    def with_filler(groups):
        v = [a / b for grp in groups for a, b in grp if b]
        return sum(v) / len(v) if v else float("nan")

    groups = list(per_item.values())
    rng = np.random.default_rng(seed)
    boots = [[groups[j] for j in rng.integers(0, len(groups), len(groups))] for _ in range(n_boot)]

    def ci(fn):
        vals = np.array([fn(b) for b in boots], float)
        vals = vals[np.isfinite(vals)]
        return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if len(vals) else [np.nan, np.nan]

    W = np.array([r[f"{mode}_W"] for r in ok], float)
    Wt = np.array([r["W_true"] for r in ok], float)
    n_filler_spans = sum(b for grp in groups for _, b in grp)
    # why planted filler stayed: deleting it alone already costs more than the tolerance (the reader
    # uses it), or it could go alone but not together with the spans queued before it
    cand = f"{mode}_candidate"
    kept_w = [r[cand][i] for r in ok if r.get(cand) for i, lab in enumerate(r["labels"])
              if _is_pure_filler(lab) and r[f"{mode}_kept"][i]]
    diag = {}
    if any(r.get(cand) for r in ok):
        diag["kept_filler"] = len(kept_w)
        if kept_w:
            diag["kept_filler_not_candidate"] = float(np.mean([not c for c in kept_w]))
            diag["kept_filler_blocked"] = float(np.mean([bool(c) for c in kept_w]))
    return {**diag,
        "n_responses": len(ok),
        "n_with_filler": sum(1 for grp in groups for _, b in grp if b),
        "n_filler_spans": n_filler_spans,
        "no_fact_lost": 1.0 - lost / len(ok),
        "responses_with_lost_fact": lost,
        "precision_pooled": removed_ok / removed_n if removed_n else 1.0,
        "spans_removed": removed_n,
        "recall_pooled": pooled(groups),
        "recall_pooled_ci": ci(pooled),
        "recall_with_filler": with_filler(groups),
        "recall_with_filler_ci": ci(with_filler),
        "recall_by_type": {t: {"removed": a, "planted": b} for t, (a, b) in sorted(by_type.items())},
        "repeats": repeats,
        "repeats_resolved": resolved / repeats if repeats else float("nan"),
        "W_mae": float(np.mean(np.abs(W - Wt))),
        "W_bias": float(np.mean(W - Wt)),
    }
