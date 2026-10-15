"""What a new scoring version recomputes in a run scored by an older one.

Responses, items and entropy signals are always kept. A version listed in PARTIAL changed the scores
of some responses, for some readers, only: a run scored by the version just before it rescores those
alone (scripts/run_real.py, the Colab cells, `drybench prune-scores`). Any other change of
SCORING_VERSION rescores every response.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from drybench.types import Item

PARTIAL = {
    # v0.8.1: a floor of 0.1 for a missing fact asked as a yes/no question (reader.BINARY_FLOOR):
    # responses to items with a missing fact, for readers whose missing-fact question is yes/no
    "8": "missing_binary",
    # "9" (v1.1.0, the response's own wording of a fact is accepted) and "10" (v1.2.0: matching, the deletion
    # search, completeness, kinds, segmentation) are not partial: every score is redone
    # v1.2.1: synthetic reports only (a cargo answer without its quantity is accepted; a span whose every value
    # is true is never a wrong statement): responses to synthetic items, and the synthetic planted-filler check
    "11": "synthetic",
}
# what a reader's folder holds besides its inputs: scores, checks of the reader, the analysis
OUTPUTS = ("scored.jsonl", "reader_check.json", "report", "filler_check", "filler_check_real", "human_check")
# with a partial rescore: files whose rows are kept for unaffected items, and outputs made again (cheap)
ROW_FILES = ("scored.jsonl", "filler_check_real/rows.jsonl")
REDONE = ("reader_check.json", "filler_check_real/summary.json", "report")


def plan(old: str | None, new: str, reader: str | None) -> str:
    """'none', 'missing' (responses to items with a missing fact), 'synthetic' (responses to synthetic items)
    or 'all': what to rescore in a folder scored by version `old` when scoring is now `new`, for this reader
    (HF id or path; None: unknown)."""
    if old == new:
        return "none"
    try:
        lo, hi = int(str(old)), int(str(new))
    except ValueError:
        return "all"
    changes = [PARTIAL.get(str(v)) for v in range(lo + 1, hi + 1)]
    if lo > hi or not changes or None in changes:
        return "all"
    if "synthetic" in changes:
        return "all" if "missing_binary" in changes else "synthetic"
    if reader is None:
        return "all"
    from drybench.reader import BINARY_STYLES, missing_style_for

    if "missing_binary" in changes and missing_style_for(reader) in BINARY_STYLES:
        return "missing"
    return "none"


def is_synthetic(item: Item) -> bool:
    return item.knobs.get("source", "synthetic") == "synthetic"


def stale(version: str | None, item: Item, reader: str | None) -> bool:
    """Is a score made by `version` for a response to `item` out of date now (SCORING_VERSION)?"""
    from drybench import SCORING_VERSION

    what = plan(version, SCORING_VERSION, reader)
    if what == "missing":
        return has_missing(item)
    if what == "synthetic":
        return is_synthetic(item)
    return what == "all"


def has_missing(item: Item) -> bool:
    from drybench.reader import is_missing

    return any(is_missing(p) for p in item.required_probes)


def _keep_rows(path: Path, keep) -> tuple[int, int]:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [line for line in rows if keep(json.loads(line))]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    tmp.replace(path)
    return len(kept), len(rows) - len(kept)


def _remove(path: Path) -> bool:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return True
    if path.exists():
        path.unlink()
        return True
    return False


def prune(folder: str | Path, items_path: str | Path, what: str) -> list[str]:
    """Remove what `what` (plan()) makes stale in a reader's folder (a run folder or its reader2_*
    folder), so that the usual resumable steps redo exactly that; returns one line per change."""
    folder, log = Path(folder), []
    if what == "none":
        return log
    if what in ("missing", "synthetic") and not Path(items_path).exists():
        what = "all"  # no items to tell which scores changed
    if what == "all":
        return [f"removed {name}" for name in OUTPUTS if _remove(folder / name)]
    if what not in ("missing", "synthetic"):
        raise ValueError(f"unknown plan {what!r}")
    from drybench.types import load_items

    test = has_missing if what == "missing" else is_synthetic
    gone = {it.item_id for it in load_items(items_path) if test(it)}
    label = "items with a missing fact" if what == "missing" else "synthetic items"
    for rel in ROW_FILES:
        path = folder / rel
        if path.exists():
            kept, dropped = _keep_rows(path, lambda r: r.get("item_id") not in gone)
            log.append(f"{rel}: {dropped} rows of {label} removed, {kept} kept")
    redone = REDONE + (("filler_check",) if what == "synthetic" else ())  # the synthetic planted check
    log += [f"removed {rel}" for rel in redone if _remove(folder / rel)]
    return log
