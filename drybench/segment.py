"""Segmentation of responses into spans (sentences, list lines, optional clauses)."""

from __future__ import annotations

import re

from drybench.types import Span

# Sentence boundary: . ! or ? (optionally followed by closing quotes/brackets), whitespace,
# then something that looks like a sentence start.
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_SENT_START = re.compile(r"[\"'(\[*_#>\-•]*\s*[A-Z0-9]")
_CLAUSE = re.compile(r";\s+")
# A piece that is only a list enumerator ("2.", "b)", "iv.") belongs to the text after it.
_ENUM = re.compile(r"^\s*(?:\d{1,3}|[A-Za-z]|[ivxIVX]{1,4})[.)]\s*$")
# A full stop after an initial or a common abbreviation does not end a sentence: "Rajendra K. Pachauri",
# "Pizza Fusion Holdings, Inc. Franchise Agreement", "Nov. 02, 2019", "St. Peter's" (v1.2; v1.1 cut there).
ABBREVIATIONS = {"Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Mt", "Ft", "Jr", "Sr", "Inc", "Corp", "Co", "Ltd", "Bros",
                 "Gen", "Gov", "Sen", "Rep", "Rev", "Lt", "Col", "Sgt", "Capt", "Adm", "Hon", "Fig", "Vol", "Jan",
                 "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec", "vs", "approx",
                 "Ave", "Blvd", "Rd", "Dept", "Univ", "Ste", "Pty", "Plc", "Mfg", "Intl"}
_INITIALS = re.compile(r"(?:[A-Z]\.)*[A-Z]")  # "K", "U.S" (before the final full stop)
_LAST_WORD = re.compile(r"[\w.]+$")


def _abbreviation_before(line: str, end: int) -> bool:
    """Does the full stop that ends line[:end] close an initial or an abbreviation (not a sentence)? A single
    capital is an initial after a name or another initial ("Rajendra K.", "J. R."), not after other words
    ("the carrier was X. 2. ...")."""
    if not line[:end].endswith("."):
        return False
    m = _LAST_WORD.search(line, 0, end - 1)
    word = m.group(0) if m else ""
    if word == "No":  # "No. 5"
        return bool(re.match(r"\s*\d", line[end:]))
    if word in ABBREVIATIONS or (len(word) > 1 and _INITIALS.fullmatch(word)):
        return True
    if len(word) == 1 and word.isupper():
        prev = re.search(r"(\S+)\s+$", line[: m.start()]) if m else None
        return bool(prev) and prev.group(1)[:1].isupper() and prev.group(1)[-1:] not in ":;,"
    return False


def _pieces(line: str, mode: str) -> list[tuple[int, int]]:
    cuts = [0]
    for m in _BOUNDARY.finditer(line):
        if (_SENT_START.match(line, m.end()) and not _ENUM.match(line[cuts[-1]:m.end()])
                and not _abbreviation_before(line, m.start())):
            cuts.append(m.end())
    if mode == "clause":
        for m in _CLAUSE.finditer(line):
            cuts.append(m.end())
    cuts = sorted(set(cuts)) + [len(line)]
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if a < b]


def segment(text: str, mode: str = "sentence") -> list[Span]:
    """Split text into spans with character offsets such that text[s.start:s.end] == s.text."""
    if mode not in ("sentence", "clause"):
        raise ValueError("mode must be 'sentence' or 'clause'")
    spans: list[Span] = []
    for m in re.finditer(r"[^\n]+", text):
        line, base = m.group(0), m.start()
        for a, b in _pieces(line, mode):
            piece = line[a:b]
            stripped = piece.strip()
            if not stripped:
                continue
            lead = len(piece) - len(piece.lstrip())
            start = base + a + lead
            spans.append(Span(start=start, end=start + len(stripped), text=stripped))
    return spans


def join_spans(spans: list[Span], keep: list[bool] | None = None) -> str:
    """Rebuild a message from the kept spans (order preserved, single-space joined)."""
    if keep is None:
        keep = [True] * len(spans)
    return " ".join(s.text for s, k in zip(spans, keep) if k)
