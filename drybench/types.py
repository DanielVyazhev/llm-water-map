"""Core data types and JSONL helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class Probe:
    """A piece of information the requester needs.

    `answers` are accepted surface forms. They must be prefix-free: the reader
    probability of a probe is the sum of alias probabilities, which is only a
    valid event probability when no alias is a prefix of another. A short form
    covers its continuations ("23" covers "23 containers").
    """

    question: str
    answers: list[str]
    fact_id: str | None = None
    required: bool = True
    value: str | None = None  # canonical wording (as in the report); answers are accepted forms

    def __post_init__(self) -> None:
        if not self.answers:
            raise ValueError("Probe needs at least one accepted answer")
        if len(set(self.answers)) != len(self.answers):
            raise ValueError(f"Duplicate aliases: {self.answers!r}")
        for a in self.answers:
            for b in self.answers:
                if a != b and b.startswith(a):
                    raise ValueError(f"Aliases must be prefix-free: {a!r} is a prefix of {b!r}")


@dataclass
class Item:
    """One closed-world benchmark item.

    document: context D the evaluated model sees (the reader never sees it).
    request: the user's request x.
    probes: what the requester needs to learn from the response.
    gold_minimal: a minimal sufficient response y0 (used for kappa and baselines).
    knobs: controlled complexity variables (k, d, answerable, ambiguous, hops, explain).
    """

    item_id: str
    document: str
    request: str
    probes: list[Probe]
    gold_minimal: str
    knobs: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def required_probes(self) -> list[Probe]:
        return [p for p in self.probes if p.required]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Item":
        d = dict(d)
        d["probes"] = [Probe(**p) for p in d["probes"]]
        return cls(**d)


@dataclass
class Span:
    """A contiguous piece of a response (character offsets into the response)."""

    start: int
    end: int
    text: str
    label: str | None = None
    fact_ids: tuple[str, ...] = ()

    @property
    def is_filler_truth(self) -> bool:
        return bool(self.label and self.label.startswith("filler"))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Rows of a JSONL file. A last line that an interrupted write left unfinished is skipped with a warning
    (append_jsonl cuts it before writing); a broken line elsewhere is an error."""
    with Path(path).open(encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    lines = [line for line in lines if line]
    for n, line in enumerate(lines):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if n < len(lines) - 1:
                raise
            import warnings

            warnings.warn(f"{path}: the last line is unfinished (an interrupted write); skipped", stacklevel=2)


_REPAIRED: set[str] = set()


def repair_jsonl(path: str | Path) -> int:
    """Cut a last line that an interrupted write left unfinished (no final newline, or not JSON), so that the
    next append starts on a line of its own. Returns the number of bytes cut."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0
    data = p.read_bytes()
    body, last = (data[:-1], b"") if data.endswith(b"\n") else (data, None)
    if last is None:  # no final newline: the last line was cut while being written
        keep = data.rfind(b"\n") + 1
    else:
        tail = body.rsplit(b"\n", 1)[-1]
        try:
            if tail.strip():
                json.loads(tail)
            return 0
        except ValueError:
            keep = len(body) - len(tail)
    with p.open("r+b") as f:
        f.truncate(keep)
    return len(data) - keep


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one row; the first append to a file in this process first cuts an unfinished last line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    key = str(p.resolve())
    if key not in _REPAIRED:
        cut = repair_jsonl(p)
        if cut:
            print(f"{p}: cut an unfinished last line ({cut} bytes) left by an interrupted write")
        _REPAIRED.add(key)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_items(path: str | Path) -> list[Item]:
    return [Item.from_dict(d) for d in read_jsonl(path)]
