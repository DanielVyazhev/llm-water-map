"""Accepted answers in the response's own words (scoring version 9, package v1.1.0).

The reader scores a fact by the probability that its reply starts with one of the probe's accepted
answers (Reader.probe_logprobs), and those come from the gold answer. A reader told to copy a short
phrase from the message copies the response's wording instead: "Matthew Perry" for the gold
"matthew perry", "New York-based" for "New York–based", "the Internal Laws of the State of Delaware"
for "Delaware". Up to v1.0.0 such a correct answer counted as not delivered, and the filler map could
delete the sentence that gave it, even as a wrong statement.

own_wording_probes() looks for the answer in a message: the same words up to case, dashes and hyphens,
quotes, apostrophes and possessive 's, digit separators, '&' for 'and', accents written either way and
articles (symbols of the form such as '$' or '%' must be there too); for CUAD dates and terms, the same date
or duration however written. Each place gives candidate replies that end with the answer: the words as the
message writes them, the same with up to MAX_LEAD words of the clause in front ("King George VI", "USC
Trojans", "the Internal Laws of the State of Delaware"), and with the opening quote when the message has
one. The lead stops at punctuation that separates phrases (, ( ) [ ] : ; . ! ? and a spaced dash) and at
words that can join or oppose another answer ("not", "or", "and", "than", "renewable", ...), so a candidate
never starts with a different answer ("David Schwimmer, not Matthew Perry", "two years, renewable for five
years"). Bare numbers and candidates of two characters or fewer are not offered (with "starts with" scoring,
"5" would also cover "50 dollars").

The reader keeps the candidates it replies with: the best one of each place where the answer stands, then
the most likely others, each with p >= MIN_P given the message and at most MAX_KEEP per fact. They join the
fact's accepted answers for this message. Only facts the gold forms leave undelivered (p < DELIVERED_P) are
looked at, so a message that states every fact in the gold's words is scored exactly as before.

dry_score applies it to the whole response (as the filler map reads it: drybench.segment.join_spans), and
classify_removed again to each deleted fragment read alone, so a fact restated in other words is a duplicate,
not a wrong statement.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from typing import Sequence

import numpy as np

from drybench.types import Item, Probe

MAX_LEAD = 8          # words of the same clause in front of the answer ("the Internal Laws of the State of")
MIN_P = 0.02          # a wording the reader replies with at least this probability, given the message
MAX_KEEP = 3          # wordings kept per fact
MAX_PLACES = 6        # places in the message where the answer stands, per fact (first ones)
MAX_CANDIDATES = 40   # candidate replies scored per fact
DELIVERED_P = 0.5     # facts the gold forms deliver at this probability are left as they are

_MARK = "̀-ͯ"  # combining accents (a decomposed é is one word)
_TOKEN = re.compile(rf"&|[$€£¥#]?[\w{_MARK}]+(?:['’.,/][\w{_MARK}]+)*%?")
_ARTICLES = {"a", "an", "the"}
_APOSTROPHES = str.maketrans({c: "'" for c in "‘’ʼ"})
_DASH = "-‐‑‒–—―−"
_PHRASE_BREAK = re.compile(r"[.!?;:\n,()\[\]{}]")
_SPACED_DASH = re.compile(rf"\s[{_DASH}]|[{_DASH}]\s")
# words that join or oppose another answer: a lead across them could start with a different answer
_STOP_WORDS = {"not", "no", "nor", "or", "and", "but", "than", "rather", "instead", "unlike", "except", "vs",
               "versus", "either", "neither", "never", "without", "plus", "minus", "additional", "successive",
               "thereafter", "another", "other", "between", "from", "to"}
_STOP_PREFIXES = ("renew", "extend", "extension")
_OPEN_QUOTES = "\"'“‘«"
_BARE_NUMBER = re.compile(r"[\d\s.,/%$€£¥#&+\-–—]+")
# Before a capitalised answer, a title keeps the name ("King George VI" is "George VI"); a word of this second
# list makes another one ("West Virginia" is not "Virginia", "New Mexico" not "Mexico").
TITLES = {"king", "queen", "prince", "princess", "emperor", "empress", "pope", "sir", "dame", "lord",
          "lady", "duke", "duchess", "earl", "count", "countess", "baron", "president", "vice", "senator",
          "governor", "mayor", "chancellor", "minister", "secretary", "general", "admiral", "captain", "colonel",
          "major", "lieutenant", "sergeant", "commander", "chief", "judge", "justice", "bishop", "archbishop",
          "cardinal", "reverend", "rev", "father", "sister", "brother", "doctor", "dr", "professor", "prof", "mr",
          "mrs", "ms", "miss", "sultan", "shah", "tsar", "czar", "pharaoh", "khan", "rabbi", "imam", "coach"}
NAME_MODIFIERS = {"north", "south", "east", "west", "northern", "southern", "eastern", "western", "northeast",
                  "northwest", "southeast", "southwest", "new", "old", "upper", "lower", "central", "greater",
                  "little", "great", "grand", "mount", "lake", "fort", "port", "cape", "saint", "st", "san", "santa",
                  "los", "las", "el", "la", "le"}
_NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                 "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
                 "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
                 "thousand", "million", "billion"}
_HYPHEN_ONLY = re.compile(rf"[{_DASH}]")


def _norm_token(token: str) -> str:
    if token == "&":
        return "and"
    t = unicodedata.normalize("NFKC", token).casefold().translate(_APOSTROPHES)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)  # 1,000 -> 1000
    return t[:-2] if t.endswith("'s") and len(t) > 3 else t  # possessive: Euler's -> euler


def _tokens(text: str) -> list[tuple[str, int, int]]:
    """(normalized word, start, end) for every word of `text`; a hyphen or dash splits words."""
    return [(_norm_token(m.group(0)), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def _content(tokens: Sequence[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    return [t for t in tokens if t[0] not in _ARTICLES]


def answer_spans(message: str, forms: Sequence[str]) -> list[tuple[int, int]]:
    """Character spans of `message` that state one of `forms` (same words up to case, dashes, apostrophes,
    quotes, digit separators, '&', accents and articles; whole words only). Overlapping matches of different
    forms are all kept ("the laws of the State of Delaware" and "Delaware")."""
    words = _content(_tokens(message))
    seq = [w for w, _, _ in words]
    found: set[tuple[int, int]] = set()
    for form in {f for f in forms if f}:
        want = [w for w, _, _ in _content(_tokens(form))]
        m = len(want)
        if not m:
            continue
        for i in range(len(seq) - m + 1):
            if seq[i:i + m] == want:
                found.add((words[i][1], words[i + m - 1][2]))
    return sorted(found)


def _months(key: tuple[int, str] | None) -> int | None:
    return None if key is None else key[0] * (12 if key[1] == "year" else 1)


_DATE_NUMERIC = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b|\b(\d{4})-(\d{1,2})-(\d{1,2})\b")


def date_spans(message: str, value: str) -> list[tuple[int, int]]:
    """Spans of `message` that write the calendar date of `value`: "1st day of July, 2020", "July 1, 2020",
    "07/01/2020" (month first, as in US contracts) or "2020-07-01" all equal "July 1, 2020"."""
    from drybench.cuad import _DATE_DMY, _DATE_MDY, _MONTH_NUM, parse_date

    target = parse_date(value)
    if target is None:
        return []
    out = []
    for rx, order in ((_DATE_MDY, "mdy"), (_DATE_DMY, "dmy")):
        for m in rx.finditer(message):
            a, b, y = m.group(1), m.group(2), int(m.group(3))
            mon, day = (a, b) if order == "mdy" else (b, a)
            if (y, _MONTH_NUM.get(mon.lower()), int(day)) == target:
                out.append((m.start(), m.end()))
    for m in _DATE_NUMERIC.finditer(message):
        g = m.groups()
        ymd = (int(g[2]), int(g[0]), int(g[1])) if g[0] else (int(g[3]), int(g[4]), int(g[5]))
        if ymd == target:
            out.append((m.start(), m.end()))
    return out


def term_spans(message: str, value: str) -> list[tuple[int, int]]:
    """Spans of `message` that give the duration of `value`: "sixty (60) months", "five-year" and "5-year"
    all equal "five (5) years"."""
    from drybench.cuad import _DUR_RX, _NUMW, _duration_key

    target = _months(_duration_key(value))
    if target is None:
        return []
    # "twenty-one-year" is 21 years: a number word joined to the one before it is not matched alone
    hyphenated = re.compile(rf"(?<![\w-])(?:(?:{_NUMW})(?:-(?:{_NUMW}))?|\d+)-(?:years?|months?)\b", re.I)
    return [(m.start(), m.end()) for rx in (_DUR_RX, hyphenated) for m in rx.finditer(message)
            if _months(_duration_key(m.group(0))) == target]


def _stops_lead(word: str) -> bool:
    return word in _STOP_WORDS or word.startswith(_STOP_PREFIXES)


def _is_number(word: str) -> bool:
    return word in _NUMBER_WORDS or word[:1].isdigit()


def _joins_answer(message: str, word: tuple[str, int, int], s: int, keep_words: set[str],
                  long_gold: bool = False) -> bool:
    """Does the word right before an answer starting at `s` make one word or name with it, so that the two
    state a different answer? "twenty-five" or "twenty five" for "five"; "West Virginia" for "Virginia";
    "John Quincy Adams" for "Adams" when the gold also names "John Adams" (long_gold: a gold form of two or
    more words). Not a title ("King George VI"), a word of the gold forms or the question, or a first name
    before a gold surname ("Leonhard Euler" for "Euler")."""
    w, _, b = word
    gap = message[b:s]
    if not gap or _HYPHEN_ONLY.fullmatch(gap):          # written together or hyphenated: one word
        return True
    if gap.strip():                                     # punctuation in between: no fusion
        return False
    first = message[s:s + 1]
    raw = message[word[1]:b]
    nxt = _TOKEN.match(message, s)
    if _is_number(w) and (first.isdigit() or (nxt is not None and _norm_token(nxt.group(0)) in _NUMBER_WORDS)):
        return True
    if not (first.isupper() and raw[:1].isupper()) or w in TITLES or w in keep_words:
        return False
    return w in NAME_MODIFIERS or long_gold


def candidates(message: str, spans: Sequence[tuple[int, int]], keep_words: set[str] = frozenset(),
               exempt: Sequence[tuple[int, int]] = (), long_gold: bool = False) -> list[tuple[str, int]]:
    """(reply, place) pairs: replies that end with a stated answer — as written, with up to MAX_LEAD words of
    its phrase in front, and with the opening quote that precedes it. A place is where an answer ends in the
    message (spans of several forms that end together are one place); the first MAX_PLACES places count.
    An answer that is part of a longer word or name ("twenty-five", "West Virginia"; _joins_answer) gets no
    lead: only its own wording is offered. `exempt`: spans found by value (CUAD dates and terms), offered even
    as bare numbers."""
    words = _tokens(message)
    ends = sorted({e for _, e in spans})[:MAX_PLACES]
    exempt = set(exempt)
    out: dict[str, int] = {}
    for s, e in spans:
        if e not in ends:
            continue
        place = ends.index(e)
        starts, k = [s], max((i for i, w in enumerate(words) if w[2] <= s), default=-1)
        if k >= 0 and _joins_answer(message, words[k], s, keep_words, long_gold):
            k = -1
        left = s
        while k >= 0 and len(starts) <= MAX_LEAD:
            w, a, b = words[k]
            gap = message[b:left]
            if _PHRASE_BREAK.search(gap) or _SPACED_DASH.search(gap) or _stops_lead(w):
                break
            starts.append(a)
            left, k = a, k - 1
        for a in starts:
            text = " ".join(message[a:e].split())
            variants = [text] + ([message[a - 1] + text] if a > 0 and message[a - 1] in _OPEN_QUOTES else [])
            for v in variants:
                bare_ok = (a, e) == (s, e) and (s, e) in exempt
                if len(v) > 2 and (bare_ok or not _BARE_NUMBER.fullmatch(v)) and v not in out:
                    out[v] = place
                if len(out) >= MAX_CANDIDATES:
                    return list(out.items())
    return list(out.items())


def prefix_free(forms: Sequence[str]) -> list[str]:
    """Distinct forms, none a prefix of another (the shorter one covers it), shortest first."""
    out: list[str] = []
    for f in sorted(dict.fromkeys(f for f in forms if f), key=len):
        if not any(f.startswith(o) for o in out):
            out.append(f)
    return out


def stated_spans(message: str, probe: Probe) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Where `message` states the answer of `probe` (text forms; CUAD dates and terms by value), in order, and
    which of those spans were found by value."""
    spans = answer_spans(message, list(probe.answers) + ([probe.value] if probe.value else []))
    fid = probe.fact_id or ""
    kind = fid.rsplit(":", 1)[-1] if fid.startswith("cuad:") else ""
    by_value: list[tuple[int, int]] = []
    if probe.value and kind == "date":
        by_value = date_spans(message, probe.value)
    elif probe.value and kind == "term":
        by_value = term_spans(message, probe.value)
    return sorted(set(spans) | set(by_value)), sorted(set(by_value))


def _keep_words(probe: Probe) -> set[str]:
    """Words a lead may cross even when capitalised: those of the gold forms and of the question."""
    texts = list(probe.answers) + [probe.value or "", probe.question]
    return {w for t in texts for w, _, _ in _tokens(t)}


def _long_gold(probe: Probe) -> bool:
    """A gold form of two or more words that starts with a capital: the gold names a longer name."""
    return any(len(_content(_tokens(a))) >= 2 and a[:1].isupper() for a in list(probe.answers) + [probe.value or ""])


def own_wording_probes(reader, request: str, probes: Sequence[Probe], message: str) -> tuple[list[Probe], list[list[str]]]:
    """`probes` with the message's own wording of each fact it states but the gold forms miss added to that
    fact's accepted answers, and the added wordings per probe."""
    from drybench.reader import is_missing

    probes = list(probes)
    added: list[list[str]] = [[] for _ in probes]
    if not probes or not message.strip():
        return probes, added
    p_gold = np.exp(reader.probe_logprobs(request, [message], probes)[0])
    out = list(probes)
    for j, p in enumerate(probes):
        if p_gold[j] >= DELIVERED_P or is_missing(p):
            continue
        spans, by_value = stated_spans(message, p)
        cands = [(c, place) for c, place in candidates(message, spans, _keep_words(p), by_value, _long_gold(p))
                 if c not in p.answers]
        if not cands:
            continue
        ctx = reader.prompt(request, message, reader.question(p))
        lps = reader.reply_logprobs([(ctx, reader.config.prefix + c) for c, _ in cands])
        scored = sorted(((float(pr), c, place) for pr, (c, place) in zip(np.exp(np.asarray(lps, float)), cands)),
                        key=lambda x: -x[0])
        bests, seen = [], set()
        for pr, c, place in scored:  # the best wording of each place first ...
            if pr < MIN_P:
                break
            if place not in seen:
                seen.add(place)
                bests.append(c)
        keep = bests[:MAX_KEEP]
        for pr, c, _ in scored:      # ... then the most likely others
            if len(keep) >= MAX_KEEP or pr < MIN_P:
                break
            if c not in keep:
                keep.append(c)
        if not keep:
            continue
        answers = prefix_free(list(p.answers) + keep)
        out[j] = replace(p, answers=answers)
        added[j] = [c for c in keep if c in answers]
    return out, added


def with_own_wording(reader, item: Item, message: str) -> tuple[Item, list[list[str]]]:
    """`item` whose required probes accept the message's own wording of the facts it states (see
    own_wording_probes), and the added wordings per required probe."""
    req = item.required_probes
    new, added = own_wording_probes(reader, item.request, req, message)
    if not any(added):
        return item, added
    swap = {id(p): q for p, q in zip(req, new)}
    return replace(item, probes=[swap.get(id(p), p) for p in item.probes]), added
