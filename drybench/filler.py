"""The score: information, completeness (kappa) and filler (W) with a span-level map.

Filler: greedily delete the span whose removal costs the least information, as long as
every delivered probe keeps its information up to a tolerance of min(abs_tol, I_j(y) / 2)
nats. Since I_j(y') - I_j(y) = log p_j(y') - log p_j(y), this bounds how much the reader's
probability of the right answer may fall (by at most e^-0.5 = 0.61x for abs_tol = 0.5),
whatever the reader's prior. A relative tolerance delta * I_j(y) (ablation) grows with
-log p_j(empty), i.e. with how sure the reader is that the answer is unknown, not with the
answer itself. The kept spans form y*; deleted spans are the filler map.

Deleted spans are then read one by one (chat readers) and split into
  duplicate      restates a fact the response delivers elsewhere        -> filler
  uninformative  the reader learns nothing about any probe from it      -> filler
  misleading     the reader confidently answers something else (a wrong fact or another
                 entity's fact)                                        -> not filler, an error
so W_filler = W_uninformative + W_duplicate and W = W_filler + W_misleading.

Complexity: greedy needs O(n^2) message evaluations for n spans; "loo" needs 2n
(leave-one-out drops, then the longest admissible removal prefix in that order).
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import numpy as np

from drybench.items import ATTR_NAMES, answer_forms
from drybench.reader import Reader
from drybench.segment import join_spans, segment
from drybench.types import Item, Probe, Span

LN2 = math.log(2.0)


@dataclass
class SpanResult:
    start: int
    end: int
    text: str
    tokens: int
    kept: bool = True
    removal_rank: int | None = None   # order of deletion (0 = first)
    drop: float | None = None         # information lost when it was deleted (nats)
    kind: str | None = None           # kept | uninformative | duplicate | misleading | removed
    loo_cost: float | None = None     # loo: information lost by deleting this span alone (nats, all probes)
    candidate: bool | None = None     # loo: deleting it alone keeps every probe within tolerance
    attempt: bool | None = None       # misleading: it answers the question itself (numbered as it, or
                                      # in its words), so being true elsewhere in the document is no excuse


@dataclass
class DryResult:
    info: float                      # I(y), nats
    info_kept: float                 # I(y*), nats
    tokens: int
    tokens_kept: int
    W: float                         # removable fraction 1 - |y*|/|y| (NaN if no signal)
    density_bits_per_token: float    # I(y) / ln2 / tokens
    kappa: float | None = None       # sum_j max(I_j(y), 0) / I(y0)  (completeness)
    info_gold: float | None = None
    status: str = "ok"               # ok | no_signal | reader_fail
    per_probe: list[float] = field(default_factory=list)      # I_j(y), nats (floored prior)
    per_probe_lp: list[float] = field(default_factory=list)   # log p_R(gold_j | y), no floor
    prior_floor: float = 0.0
    spans: list[SpanResult] = field(default_factory=list)
    n_evals: int = 0
    mode: str = "greedy"
    delta: float = 0.0
    constraint: str = "per_probe"
    abs_tol: float = 0.5
    # split of the removable fraction (chat readers): filler = uninformative + duplicate
    W_filler: float = float("nan")
    W_uninformative: float = float("nan")
    W_duplicate: float = float("nan")
    W_misleading: float = float("nan")
    # token overhead vs the minimal gold answer y0 (needs kappa):
    #   overhead = |y| / (kappa |y0|) = cost per bit of y / cost per bit of y0
    #            = 1/(1 - W)  [removable sentences]  x  |y*| / (kappa |y0|)  [wordiness per delivered bit]
    # wording = |y*| / |y0| is stored as is (not per bit)
    tokens_gold: int | None = None
    wording: float = float("nan")
    overhead: float = float("nan")
    # accuracy view: reader's probability of each gold answer given y and given y0;
    # facts = share of answerable facts (readable in y0, p >= 0.5) the reader gets from y (p >= 0.5)
    per_probe_p: list[float] = field(default_factory=list)
    gold_per_probe_p: list[float] = field(default_factory=list)
    facts: float = float("nan")
    # v1.1: the response's own wordings of a fact that were added to its accepted answers, per required
    # probe (drybench.forms; empty lists: every fact read in the gold's words, or the option was off)
    added_answers: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def removable_tokens(self) -> int:
        return self.tokens - self.tokens_kept


def _feasible(pp_new: np.ndarray, pp_full: np.ndarray, delta: float, abs_tol: float,
              constraint: str) -> np.ndarray:
    """Which candidate messages preserve the information (rows of pp_new: per-probe gains)."""
    pp_new = np.atleast_2d(pp_new)
    if constraint == "global":
        full = pp_full.sum()
        return pp_new.sum(axis=1) >= full - max(delta * abs(full), abs_tol) - 1e-12
    gain = np.abs(pp_full)
    # absolute slack absorbs reader noise but never exceeds half of a probe's own gain,
    # so deleting the only statement of a fact is always infeasible
    slack = np.maximum(delta * gain, np.minimum(abs_tol, 0.5 * gain))
    active = pp_full >= abs_tol  # probes the response does not deliver impose no constraint
    return np.all((pp_new >= pp_full - slack - 1e-12) | ~active[None, :], axis=1)


def filler_map(
    reader: Reader,
    request: str,
    spans: Sequence[Span],
    probes: Sequence[Probe],
    delta: float = 0.0,
    mode: str = "greedy",
    length_fn: Callable[[str], int] | None = None,
    min_info: float = 1.0,
    constraint: str = "per_probe",
    abs_tol: float = 0.5,
    tie_tol: float = 1e-3,
    loo_skip: bool = True,
) -> DryResult:
    """Compute I(y), W and the per-span filler map for a pre-segmented message.

    constraint="per_probe" (default): every delivered probe (I_j(y) >= abs_tol) must keep its
        gain up to max(delta * I_j, min(abs_tol, I_j / 2)) nats (delta = 0 by default), so a
        low-information fact (an answer the reader could half-guess) is never traded away for
        other facts' bits.
    constraint="global": only the total I(y) must stay within max(delta * |I(y)|, abs_tol).

    mode="loo": the candidates (spans whose deletion alone is feasible) are queued by that cost and
    the longest feasible prefix of the queue is deleted; then (loo_skip=True, the default since
    v0.7.2) the candidate that ended the prefix is skipped and the rest of the queue continues,
    round by round, so one span that cannot go with the others does not keep every span queued
    after it. Each round is checked jointly; loo_skip=False (v0.7.1 and before) stops after the
    first round and deletes a subset of the same spans.
    """
    if mode not in ("greedy", "loo"):
        raise ValueError("mode must be 'greedy' or 'loo'")
    if constraint not in ("per_probe", "global"):
        raise ValueError("constraint must be 'per_probe' or 'global'")
    length_fn = length_fn or reader.backend.count_tokens
    lengths = [max(1, length_fn(s.text)) for s in spans]
    total = int(sum(lengths))
    res_spans = [SpanResult(s.start, s.end, s.text, lengths[i]) for i, s in enumerate(spans)]
    n_evals = 1
    full_msg = join_spans(list(spans))
    per_probe = reader.per_probe_information(request, [full_msg], probes)[0]
    per_probe_lp = reader.probe_logprobs(request, [full_msg], probes)[0]  # cached
    info_full = float(per_probe.sum())
    density = info_full / LN2 / total if total else 0.0
    base = dict(info=info_full, tokens=total, density_bits_per_token=density,
                per_probe=per_probe.tolist(), per_probe_lp=per_probe_lp.tolist(),
                prior_floor=reader.config.prior_floor, spans=res_spans, mode=mode, delta=delta,
                constraint=constraint, abs_tol=abs_tol)
    if not spans or info_full < min_info:
        return DryResult(info_kept=info_full, tokens_kept=total, W=float("nan"), status="no_signal",
                         n_evals=n_evals, **base)

    tol = tie_tol * abs(info_full)
    keep = [True] * len(spans)

    def msg_without(removed: set[int]) -> str:
        return join_spans(list(spans), [i not in removed and keep[i] for i in range(len(spans))])

    def feasible(pp: np.ndarray) -> np.ndarray:
        return _feasible(pp, per_probe, delta, abs_tol, constraint)

    if mode == "greedy":
        cur, rank = info_full, 0
        while any(keep):
            cand = [i for i in range(len(spans)) if keep[i]]
            pp = reader.per_probe_information(request, [msg_without({i}) for i in cand], probes)
            n_evals += len(cand)
            ok = feasible(pp)
            if not ok.any():
                break
            infos = pp.sum(axis=1)
            drops = np.where(ok, cur - infos, np.inf)
            best = drops.min()
            tied = [j for j in range(len(cand)) if drops[j] <= best + tol]
            j = max(tied, key=lambda t: lengths[cand[t]])
            i = cand[j]
            keep[i] = False
            res_spans[i].kept, res_spans[i].removal_rank = False, rank
            res_spans[i].drop = float(cur - infos[j])
            cur, rank = float(infos[j]), rank + 1
        info_kept = cur
    else:
        loo = reader.per_probe_information(request, [msg_without({i}) for i in range(len(spans))], probes)
        n_evals += len(spans)
        drops = info_full - loo.sum(axis=1)
        single_ok = feasible(loo)
        for i, sr in enumerate(res_spans):
            sr.loo_cost, sr.candidate = float(drops[i]), bool(single_ok[i])
        queue = sorted((i for i in range(len(spans)) if single_ok[i]),
                       key=lambda i: (round(drops[i] / tol) if tol else drops[i], -lengths[i]))
        removed: list[int] = []
        info_kept = info_full
        while queue:
            prefixes = [set(removed) | set(queue[: r + 1]) for r in range(len(queue))]
            pp = reader.per_probe_information(request, [msg_without(p) for p in prefixes], probes)
            n_evals += len(prefixes)
            ok = np.flatnonzero(feasible(pp))
            r_best = int(ok.max()) if len(ok) else -1
            if r_best >= 0:
                removed += queue[: r_best + 1]
                info_kept = float(pp[r_best].sum())
            if not loo_skip:
                break
            queue = queue[r_best + 2:]  # skip the span that ended the prefix
        for rank, i in enumerate(removed):
            keep[i] = False
            res_spans[i].kept, res_spans[i].removal_rank = False, rank
            res_spans[i].drop = float(drops[i])

    kept_tokens = int(sum(l for l, k in zip(lengths, keep) if k))
    return DryResult(info_kept=float(info_kept), tokens_kept=kept_tokens,
                     W=1.0 - kept_tokens / total, n_evals=n_evals, **base)


# Replies meaning "the message does not say" ("not" covers not stated / specified / mentioned).
ABSENCE_ALIASES = ["unknown", "Unknown", "not", "Not"]
SHIPMENT_ID = re.compile(r"\b[A-Z]{2}-\d{4}\b")


def non_answers(probe: Probe) -> list[str]:
    """Replies that are not a wrong fact: absence replies, and echoes of the question's entity
    ("HS-5423", "Shipment HS-5423", "For shipment HS-5423" when the message is a bare header)."""
    ids = SHIPMENT_ID.findall(probe.question)
    return ABSENCE_ALIASES + ids + (["Shipment", "shipment", "For"] if ids else [])


def wrong_answer_mass(reader: Reader, request: str, messages: Sequence[str],
                      probes: Sequence[Probe]) -> np.ndarray:
    """[n_messages, n_probes] probability that the reader answers something that is neither the
    gold answer nor a non-answer ('unknown', 'not stated', an echo of the question)."""
    safe = [non_answers(p) for p in probes]
    p_safe = np.exp(reader.absence_logprobs(request, messages, probes, safe))
    for j, p in enumerate(probes):
        # gold answers that begin like a non-answer ('not stated') are already counted
        rest = [a for a in p.answers if not any(a.startswith(b) for b in safe[j])]
        if rest:
            p_safe[:, j] += np.exp(reader.probe_logprobs(request, messages, [Probe(p.question, rest)])[:, 0])
    return np.clip(1.0 - p_safe, 0.0, 1.0)


WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
FUNCTION_WORDS = {"a", "an", "the", "for", "of", "is", "was", "were", "are", "and", "to", "in", "on", "it",
                  "its", "this", "that", "here", "as", "with", "by", "be", "been", "has", "have", "had", "s"}
# words that name an attribute; a true value next to its own attribute's words is a true statement
ATTR_CUES = {
    "carrier": r"\bcarrier|\bhandl",
    "origin": r"\bdepart|\borigin|\bfrom\b",
    "destination": r"\bbound\b|\bdestin|\bheaded|\barriv|\bto\b",
    "cargo": r"\bcargo|\bcarr(?:y|ied|ying)\b|\bconsist|\bload|\bgoods",
    "delay_cause": r"\bdelay|\bcaus|\bdue\b",
    "inspector": r"\binspect",
    "departure_date": r"\bleft\b|\bdepart|\bdate\b|\bon\b",
    "containers": r"\bcontainer",
    "carrier_home": r"\bhome\b|\bbased\b|\bheadquarter|\bbase\b",
}


def _true_values(item: Item) -> list[tuple[str, list[str]]]:
    """(attribute, surface forms) of every fact the report states about the requested shipment(s)."""
    out = []
    for rec in (item.meta.get("target"), item.meta.get("twin")):
        for attr in ATTR_NAMES if rec else ():
            value = rec.get(attr)
            if value and value in item.document:
                forms = answer_forms(attr, value)
                if attr == "cargo" and " of " in value:  # "52 crates of glass panels" -> "glass panels"
                    forms = forms + [value.split(" of ", 1)[1]]
                out.append((attr, forms))
    if item.meta.get("carrier_home"):
        out.append(("carrier_home", [item.meta["carrier_home"]]))
    return out


def _mentions(text: str, form: str) -> bool:
    return re.search(r"(?<![\w-])" + re.escape(form.lower()) + r"(?![\w-])", text) is not None


@lru_cache(maxsize=4096)
def _document_sentence_words(document: str) -> tuple[frozenset, ...]:
    return tuple(frozenset(w.lower() for w in WORD.findall(s.text)) for s in segment(document))


def supported_by_document(words: set[str], document: str, min_cover: float = 0.8) -> bool:
    """At least `min_cover` of `words` occur in one sentence of the document."""
    return bool(words) and any(len(words & sw) >= min_cover * len(words)
                               for sw in _document_sentence_words(document))


def misleading_excuse(text: str, item: Item) -> str | None:
    """Why a deleted span cannot be a wrong statement about the request, or None.

    The reader reads the span alone and may misattribute what it says; three patterns are not
    errors of the response: a span with no words beyond the request (a header "For shipment
    KJ-3299:"), a span about another shipment (a correctly attributed distractor), and a true
    value of the requested shipment next to its own attribute ("It was inspected by X" when the
    inspector was not requested). Real-text items have no records to check against, so there a
    true statement is one whose words (80% of them, the request's words included) occur in one
    sentence of the document: "The law of New Jersey governs this agreement" is not excused by
    "a New Jersey corporation" (v0.7.0 counted only the words new to the request and excused it)."""
    request_words = {w.lower() for w in WORD.findall(item.request)}
    words = {w.lower() for w in WORD.findall(text)} - FUNCTION_WORDS
    if not words - request_words:
        return "no new words"
    if item.knobs.get("source", "synthetic") != "synthetic":
        return "true statement" if supported_by_document(words, item.document) else None
    own = {rec["sid"] for rec in (item.meta.get("target"), item.meta.get("twin")) if rec}
    if set(SHIPMENT_ID.findall(text)) - own:
        return "other shipment"
    low = text.lower()
    for attr, forms in _true_values(item):
        if any(_mentions(low, f) for f in forms) and re.search(ATTR_CUES[attr], low):
            return "true statement"
    return None


_NUMBERED = re.compile(r"^\s*\(?(\d{1,2})[.):]\s")
_NUMBERED_REQUEST = re.compile(r"(?m)^\s*1[.)]\s")


def _content_words(text: str) -> set[str]:
    from drybench.baselines import EN_STOPWORDS

    return {w for w in (x.lower() for x in WORD.findall(text)) if len(w) > 2 and w not in EN_STOPWORDS
            and w not in FUNCTION_WORDS}


def answer_attempts(text: str, request: str, probes: Sequence[Probe]) -> np.ndarray:
    """[n_probes] True where the span answers probe j itself: it carries question j's number in a
    numbered request ("2. ..."), or at least 60% (and 2) of the content words of question j."""
    out = np.zeros(len(probes), bool)
    m = _NUMBERED.match(text)
    if m and _NUMBERED_REQUEST.search(request):
        j = int(m.group(1)) - 1
        if 0 <= j < len(probes):
            out[j] = True
    words = _content_words(text)
    for j, p in enumerate(probes):
        q = _content_words(p.question)
        if len(q) >= 2 and len(words & q) >= max(2, 0.6 * len(q)):
            out[j] = True
    return out


def classify_removed(reader: Reader, request: str, spans: Sequence[Span], probes: Sequence[Probe],
                     res: DryResult, dup_frac: float = 0.5, other_mass: float = 0.5,
                     item: Item | None = None, own_wording: bool = False) -> None:
    """Label every span of `res` (in place) and fill the W_* split.

    Each deleted span is read on its own:
      duplicate   I_j(span) >= max(abs_tol, dup_frac * I_j(y)) for a probe the response delivers
      misleading  the reader puts >= other_mass on answers that are neither gold nor an absence
                  reply ('unknown', 'not stated', ...), while with an empty message it does not,
                  and the span is not excused by misleading_excuse (needs `item`); being a true
                  statement of the document is no excuse for a span that answers that question
                  itself (answer_attempts): a true sentence can be the wrong answer
      uninformative  otherwise
    Needs a chat reader told to answer 'unknown' (ReaderConfig.use_chat); with other readers
    deleted spans are labelled 'removed' and all count as filler.
    own_wording (v1.1): a span read alone also gets its own wording of a fact accepted
    (drybench.forms), so a fact restated in other words is a duplicate, not a wrong statement.
    """
    removed = [i for i, sr in enumerate(res.spans) if not sr.kept]
    for sr in res.spans:
        sr.kind = "kept" if sr.kept else "removed"
    total = max(res.tokens, 1)
    if not removed or not reader.config.use_chat:
        removed_share = sum(res.spans[i].tokens for i in removed) / total
        res.W_filler, res.W_uninformative, res.W_duplicate = removed_share, float("nan"), float("nan")
        res.W_misleading = float("nan") if removed else 0.0
        if not removed:
            res.W_uninformative = res.W_duplicate = 0.0
        return
    msgs = [spans[i].text for i in removed] + [""]
    if own_wording:  # each span with its own probes: its own wording of a fact accepted
        from drybench.forms import own_wording_probes

        gain_rows, other_rows = [], []
        for text in msgs[:-1]:
            ps, _ = own_wording_probes(reader, request, probes, text)
            gain_rows.append(reader.per_probe_information(request, [text], ps)[0])
            other_rows.append(wrong_answer_mass(reader, request, [text, ""], ps))
        gain = np.array(gain_rows)
        p_other = np.array([r[0] for r in other_rows])
        p_empty = np.array([r[1] for r in other_rows])
    else:
        gain = reader.per_probe_information(request, msgs[:-1], probes)  # each span read alone
        both = wrong_answer_mass(reader, request, msgs, probes)
        p_other, p_empty = both[:-1], np.repeat(both[-1:], len(removed), axis=0)
    full = np.asarray(res.per_probe, float)
    delivered = full >= res.abs_tol
    shares = {"uninformative": 0, "duplicate": 0, "misleading": 0}
    for row, i in enumerate(removed):
        dup = np.any(delivered & (gain[row] >= np.maximum(res.abs_tol, dup_frac * full)))
        wrong = (p_other[row] >= other_mass) & (p_empty[row] < other_mass)
        excuse = misleading_excuse(spans[i].text, item) if item is not None else None
        attempt = bool(np.any(wrong & answer_attempts(spans[i].text, request, probes)))
        mis = bool(np.any(wrong)) and (excuse is None or (excuse == "true statement" and attempt))
        kind = "duplicate" if dup else ("misleading" if mis else "uninformative")
        res.spans[i].kind = kind
        if kind == "misleading":
            res.spans[i].attempt = attempt
        shares[kind] += res.spans[i].tokens
    res.W_uninformative = shares["uninformative"] / total
    res.W_duplicate = shares["duplicate"] / total
    res.W_misleading = shares["misleading"] / total
    res.W_filler = res.W_uninformative + res.W_duplicate


def dry_score(
    reader: Reader,
    item: Item,
    response: str,
    delta: float = 0.0,
    mode: str = "greedy",
    segment_mode: str = "sentence",
    spans: Sequence[Span] | None = None,
    length_fn: Callable[[str], int] | None = None,
    min_info: float = 1.0,
    with_kappa: bool = True,
    constraint: str = "per_probe",
    abs_tol: float = 0.5,
    classify: bool = True,
    loo_skip: bool = True,
    own_wording: bool = True,
) -> DryResult:
    """Full scoring of one response to one item (required probes only).

    own_wording (v1.1, scoring version 9): a fact the response states in its own words ("Matthew Perry"
    for the gold "matthew perry", "the State of Delaware" for "Delaware") is read with that wording too
    (drybench.forms); False scores with the gold forms only, as v1.0.0 did."""
    spans = list(spans) if spans is not None else segment(response, segment_mode)
    added: list[list[str]] = []
    if own_wording:
        from drybench.forms import with_own_wording

        item, added = with_own_wording(reader, item, join_spans(spans))
    probes = item.required_probes
    res = filler_map(reader, item.request, spans, probes, delta=delta, mode=mode,
                    length_fn=length_fn, min_info=min_info, constraint=constraint, abs_tol=abs_tol,
                    loo_skip=loo_skip)
    res.added_answers = added
    if classify and res.status == "ok":
        classify_removed(reader, item.request, spans, probes, res, item=item, own_wording=own_wording)
    if with_kappa:
        gold = float(reader.information(item.request, [item.gold_minimal], probes)[0])
        res.info_gold = gold
        p_resp = np.exp(np.asarray(res.per_probe_lp, float))
        p_gold = np.exp(reader.probe_logprobs(item.request, [item.gold_minimal], probes)[0])
        res.per_probe_p, res.gold_per_probe_p = p_resp.tolist(), p_gold.tolist()
        readable = p_gold >= 0.5  # a missing fact counts too: the response should say it is not stated
        res.facts = float(np.mean(p_resp[readable] >= 0.5)) if readable.any() else float("nan")
        count = length_fn or reader.backend.count_tokens  # same token count as the response spans
        res.tokens_gold = int(sum(max(1, count(s.text)) for s in segment(item.gold_minimal, segment_mode)))
        if gold < min_info:
            res.status = "reader_fail"
            res.kappa = float("nan")
        else:
            res.kappa = completeness(res.per_probe, gold)
            if res.status == "ok" and res.kappa > 0 and res.tokens_gold:
                res.wording = res.tokens_kept / res.tokens_gold
                res.overhead = res.tokens / (res.kappa * res.tokens_gold)
    return res


def completeness(per_probe: Sequence[float], info_gold: float) -> float:
    """kappa = sum_j max(I_j(y), 0) / I(y0): the share of the gold answer's information the response
    delivers. A fact the response contradicts counts as not delivered (0), not as negative: with a
    reader that has a prior for it (say p('not stated' | no message) = 0.08), a made-up value would
    otherwise cost ~11 nats and drive kappa of a one-fact answer to -4.5, a reader-dependent penalty
    (v0.7.3 and before). Wrong statements are measured separately (W_misleading, facts)."""
    if info_gold is None or not np.isfinite(info_gold) or info_gold <= 0:
        return float("nan")
    return float(np.clip(np.asarray(per_probe, float), 0.0, None).sum() / info_gold)


def cost_per_bit(tokens: int, info_nats: float, price_per_token: float) -> float:
    """Price of one transmitted bit of task-relevant information."""
    bits = info_nats / LN2
    return float("inf") if bits <= 0 else price_per_token * tokens / bits
