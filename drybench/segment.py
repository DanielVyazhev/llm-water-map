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


def _pieces(line: str, mode: str) -> list[tuple[int, int]]:
    cuts = [0]
    for m in _BOUNDARY.finditer(line):
        if _SENT_START.match(line, m.end()) and not _ENUM.match(line[cuts[-1]:m.end()]):
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
