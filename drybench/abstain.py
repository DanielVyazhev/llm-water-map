"""Does an answer say that a requested fact is not given? A lexical rule for missing facts.

A reader cannot be asked this reliably (seven wordings for up to three readers were compared by
`drybench missing-check`). Hand labels of every missing-fact answer of the two T4 runs
(600 answers: synthetic 240, real 360; scripts/check_abstain.py, results/abstain_labels.json): the
main reader's yes/no question (Qwen2.5-3B, "Does the message say that this is not stated: ...?")
separates the answers that say the fact is not given from the others with AUC 0.53 on the synthetic
answers and 0.79 on the real ones — 1.00 on CUAD, where `drybench missing-check` chose it, but 0.44 on
MuSiQue: it says 'yes' to "The destination port for shipment ST-3904 is Bexyor Docks." (0.93) and
'no' to "The origin port for shipment SD-6300 is not mentioned in the report." (2·10⁻⁶). Phi-3.5's
content question ranks them better (AUC 0.94 / 0.89) but reads only 13% of the synthetic abstentions
with p ≥ 0.5. So from v0.8.2 missing facts are judged by this rule, and items with a missing fact are
left out of the filler-map tables.

A fragment says that question j's answer is not given when it has an abstention phrase (ABSTAIN) and is
about question j: it carries j's number in a numbered request; or it shares a word (5-letter stem) with
question j or j's gold sentence that the other questions do not have; or, when the item asks one
question, it has no content words besides the abstention itself ("This information is not provided.").
Against the hand labels: precision 0.99 (86/87), recall 1.00 (86/86); an answer that says so and then
guesses a value counts as saying so (3 of the 86).
"""

from __future__ import annotations

import re
from typing import Sequence

from drybench.segment import segment
from drybench.types import Item

_VERB = r"(?:say|state|mention|specify|provide|include|contain|indicate|give|list|name|identify|report|show)"
_PARTICIPLE = (r"(?:stated|specified|mentioned|provided|given|available|included|listed|indicated|disclosed|found|"
               r"known|clear|identified|reported|recorded|named|described|defined|determined)")
# phrases that say something is not given, in any wording (v0.7.4–v0.8.1: the first eleven alternatives;
# v1.2: "cannot determine", "isn't specified", "no governing law clause", "N/A", curly apostrophes)
ABSTAIN = re.compile(
    r"\b(?:not (?:\w+ )?" + _PARTICIPLE + r"|unspecified|unknown|"
    r"undisclosed|no (?:\w+ ){0,3}(?:information|mention|details?|data|record|indication|reference|clause|"
    r"provision|section|statement)|"
    r"none of the (?:\w+ ){0,3}" + _VERB + r"s?|"
    r"(?:does|do|did|is|are|was|were) not (?:\w+ )?" + _VERB + r"|"
    r"(?:doesn't|don't|didn't|isn't|aren't|wasn't|weren't) (?:\w+ )?" + _VERB + r"|"
    r"(?:isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't) (?:been )?(?:\w+ )?" + _PARTICIPLE + r"|"
    r"(?:don't|do not|doesn't|does not|didn't|did not) have (?:any |the |enough )?(?:information|details?|data|record)|"
    r"(?:cannot|can't|could not|couldn't|unable to|impossible to|not possible to) (?:\w+ )?(?:be )?"
    r"(?:determin\w*|state|say|tell|confirm|identify|know|answer)|"
    r"(?:cannot|can't|could not|couldn't|unable to|did not|didn't) (?:\w+ )?find|"
    r"(?:cannot|can't) be (?:\w+ )?(?:found|identified|determined|answered)|"
    r"N/A|not applicable)(?!\w)", re.I)
_APOSTROPHES = str.maketrans({c: "'" for c in "\u2019\u2018\u02bc"})
_CLAUSE = re.compile(r";|\bbut\b|\bhowever\b", re.I)

_NUMBERED = re.compile(r"^\s*\(?(\d{1,2})[.):]\s")
_NUMBERED_REQUEST = re.compile(r"(?m)^\s*1[.)]\s")
_ID = re.compile(r"\b[A-Z]{2}-\d{4}\b")  # synthetic shipment ids
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
# words that say nothing about which fact is meant: the source, the abstention itself
_GENERIC = {"shipment", "report", "text", "passage", "contract", "excerpt", "document", "documents", "agreement",
            "message", "information", "details", "detail", "data", "answer", "question", "provided", "given", "stated",
            "state", "specified", "specify", "mentioned", "mention", "unknown", "unspecified", "available", "explicitly",
            "specific", "determine", "determined", "found", "find", "included", "include", "indicate", "listed",
            "context", "operations", "based", "however", "therefore", "cannot", "unable", "possible", "impossible"}


_GENERIC_STEMS = {w[:5] for w in _GENERIC}


def _stems(text: str) -> set[str]:
    """5-letter stems of the content words (stop words, the generic words above and shipment ids left out)."""
    from drybench.baselines import EN_STOPWORDS

    words = (w.lower().strip("'-") for w in _WORD.findall(_ID.sub(" ", text)))
    return {w[:5] for w in words if len(w) > 2 and w not in EN_STOPWORDS} - _GENERIC_STEMS


def topics(questions: Sequence[str], gold: str = "") -> list[set[str]]:
    """Per question: the stems of the question and of its gold sentence (when the gold has one sentence
    per question) that the other questions do not have."""
    sents = [s.text for s in segment(gold)] if gold else []
    own = [_stems(q) | (_stems(sents[j]) if len(sents) == len(questions) else set()) for j, q in enumerate(questions)]
    out = []
    for j, w in enumerate(own):
        rest = set().union(*(own[i] for i in range(len(own)) if i != j)) if len(own) > 1 else set()
        out.append((w - rest) or w)
    return out


def says_not_given(fragment: str, request: str, topic: set[str], j: int, n_questions: int) -> bool:
    """Does this fragment say that question j's answer is not given? The abstention phrase and the topic
    must stand in the same clause (split at semicolons, "but", "however"): "The report gives no details about
    the delay; the carrier was Dralor Cargo." says nothing about the carrier (v1.2). A number that is not a question's
    number ("3." in a request of two questions) falls back to the topic words (v1.2)."""
    fragment = fragment.translate(_APOSTROPHES)
    if not ABSTAIN.search(fragment):
        return False
    m = _NUMBERED.match(fragment)
    if m and _NUMBERED_REQUEST.search(request) and 1 <= int(m.group(1)) <= n_questions:
        return int(m.group(1)) - 1 == j
    clauses = [c for c in _CLAUSE.split(fragment) if ABSTAIN.search(c)]
    words = set().union(*(_stems(c) for c in clauses))
    return bool(words & topic) or (n_questions == 1 and not _stems(fragment))


def abstains(response: str, item: Item) -> bool | None:
    """True when the response says that every missing fact of the item is not given; None for an item
    without a missing fact."""
    from drybench.reader import is_missing

    probes = item.required_probes
    missing = [j for j, p in enumerate(probes) if is_missing(p)]
    if not missing:
        return None
    tops = topics([p.question for p in probes], item.gold_minimal)
    frags = [s.text for s in segment(response)]
    return all(any(says_not_given(f, item.request, tops[j], j, len(probes)) for f in frags) for j in missing)
