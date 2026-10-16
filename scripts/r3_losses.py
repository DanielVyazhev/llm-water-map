#!/usr/bin/env python3
"""Why R3 counts a requested fact as lost, per reader of a real-text run.

    python scripts/r3_losses.py RUN [--examples 5] [--csv mentions.csv]

The truth of R3 (v1.2) keeps a fact only by the sentence chosen to state it or by the summary. A lost fact is then
  not read   the reader reads less than 0.5 nats of it from the whole response, so the filler map does not protect
             its sentence;
  mention    a kept fragment mentions one of its accepted answers and the reader reads the fact from it: either a
             restatement the truth does not count ("INTELLECTUAL PROPERTY AGREEMENT, dated as of ..." after the
             title line went) or a mere mention ("VerticalNet, Inc., a Pennsylvania corporation" for the governing
             law);
  other      neither.
--csv writes the mention cases (question, accepted answers, the kept fragments that mention them) with an empty
'states' column, for a person to judge whether a kept fragment states the fact. Rows of v1.0 (no facts per
fragment) get the v1.2 truth first.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def fragment_facts(row: dict, item) -> list[set]:
    """The facts each fragment states by the v1.2 truth: the row's own, or (v1.0 rows) the chosen sentences'."""
    if row.get("span_facts") is not None:
        return [set(f) for f in row["span_facts"]]
    from drybench.realcheck import NOT_STATED, document_sentences, fact_sentence

    sents = document_sentences(item.document)
    chosen: dict[str, set] = {}
    for p in item.required_probes:
        if NOT_STATED not in p.answers:
            chosen.setdefault(fact_sentence(item, p, sents)[1], set()).add(p.fact_id)
    facts = [chosen.get(t, set()) if lab == "fact" else set() for t, lab in zip(row["span_texts"], row["labels"])]
    present = set().union(*facts)
    return [present if lab == "filler:summary" else f for f, lab in zip(facts, row["labels"])]


def losses(row: dict, item) -> list[dict]:
    """One record per lost fact of a scored planted response."""
    from drybench.realcheck import _mentions_any

    facts, kept, texts = fragment_facts(row, item), row["loo_kept"], row["span_texts"]
    covered = set().union(*(f for f, k in zip(facts, kept) if k))
    out = []
    for j, p in enumerate(item.required_probes):
        if p.fact_id not in set().union(*facts) or p.fact_id in covered:
            continue
        info = float(row["per_probe"][j])
        mention = [t for t, k in zip(texts, kept) if k and _mentions_any(t, p.answers)]
        out.append({"item_id": row["item_id"], "condition": row["condition"], "source": row.get("source", ""),
                    "question": p.question, "answers": p.answers, "I": info,
                    "kind": "not read" if info < 0.5 else ("mention" if mention else "other"),
                    "deleted": [t for t, k, f in zip(texts, kept, facts) if not k and p.fact_id in f],
                    "kept_mentions": mention})
    return out


def main(argv: list[str] | None = None) -> None:
    from drybench.compare import read_rows
    from drybench.report import _readers
    from drybench.rescore import has_missing
    from drybench.types import load_items

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="real-text run folder (items.jsonl, filler_check_real/, reader2_*/)")
    ap.add_argument("--examples", type=int, default=0, help="mention cases to print per reader")
    ap.add_argument("--csv", help="write the mention cases here, to be judged by a person")
    args = ap.parse_args(argv)
    run = Path(args.run)
    items = {i.item_id: i for i in load_items(run / "items.jsonl")}
    subs, _ = _readers(run, list(items.values()))
    table = []
    for label, sub in subs:
        rows = [r for r in read_rows(sub / "filler_check_real" / "rows.jsonl")
                if r.get("loo_status") == "ok" and not has_missing(items[r["item_id"]])]
        if not rows:
            continue
        per = [losses(r, items[r["item_id"]]) for r in rows]
        lost = [x for xs in per for x in xs]
        kinds = Counter(x["kind"] for x in lost)
        print(f"\n{label}: {len(rows)} responses with a filler map; {sum(map(bool, per))} lose a fact "
              f"({100 * sum(map(bool, per)) / len(rows):.1f}%); lost facts: "
              + ", ".join(f"{k} {kinds[k]}" for k in ("not read", "mention", "other")))
        by_src = Counter((x["source"], x["kind"]) for x in lost)
        for src in sorted({s for s, _ in by_src}):
            print(f"   {src}: " + ", ".join(f"{k} {by_src[(src, k)]}" for k in ("not read", "mention", "other")
                                           if by_src[(src, k)]))
        for x in [x for x in lost if x["kind"] == "mention"][:args.examples]:
            print(f"\n   {x['item_id']} · {x['condition']} · Q: {x['question']} · A: {x['answers'][:3]} · "
                  f"I = {x['I']:.1f}")
            for t in x["deleted"]:
                print(f"     deleted: {t[:200]}")
            for t in x["kept_mentions"]:
                print(f"     kept:    {t[:200]}")
        table += [{"reader": label, **x} for x in lost if x["kind"] == "mention"]
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["reader", "item_id", "condition", "source", "question", "answers", "kept fragments",
                        "states (yes/no)"])
            for x in table:
                w.writerow([x["reader"], x["item_id"], x["condition"], x["source"], x["question"],
                            " | ".join(x["answers"]), " || ".join(x["kept_mentions"]), ""])
        print(f"\n{len(table)} mention cases → {args.csv}")


if __name__ == "__main__":
    main()
