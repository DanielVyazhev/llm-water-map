#!/usr/bin/env python3
"""Write the labelling guidelines (drybench/guidelines.py) into the labelling page.

    python scripts/build_annotation_page.py          # annotation/dry_span_labels.html (+ _offline.html)
    python scripts/build_annotation_page.py --check  # exit 1 if the page is out of date

dry_span_labels.html is the page as published (an artifact: labels are shared through the page's
storage); dry_span_labels_offline.html is the same page as a standalone file to open in a browser
(labels stay in that browser until exported).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PAGE = REPO / "annotation" / "dry_span_labels.html"
OFFLINE = REPO / "annotation" / "dry_span_labels_offline.html"
START, END = "// GUIDE:START", "// GUIDE:END"
SKELETON = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1,viewport-fit=cover"><style>:root{color-scheme:light}'
            'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>')


def build(page: str) -> str:
    from drybench.guidelines import page_js

    a = page.index(START)
    a = page.index("\n", a) + 1
    b = page.index(END)
    return page[:a] + page_js() + page[b:]


def offline(page: str) -> str:
    return SKELETON + page.rstrip("\n") + "\n</body></html>"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only check that both pages are up to date")
    args = ap.parse_args(argv)
    cur = PAGE.read_text(encoding="utf-8")
    new = build(cur)
    if args.check:
        stale = [p.name for p, text in ((PAGE, new), (OFFLINE, offline(new)))
                 if not p.exists() or p.read_text(encoding="utf-8") != text]
        if stale:
            print("out of date: " + ", ".join(stale) + " (run scripts/build_annotation_page.py)")
            return 1
        print("pages up to date")
        return 0
    PAGE.write_text(new, encoding="utf-8")
    OFFLINE.write_text(offline(new), encoding="utf-8")
    print(f"written: {PAGE}, {OFFLINE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
