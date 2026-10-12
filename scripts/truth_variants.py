#!/usr/bin/env python3
"""Two checks of the truth behind the development results (docs/results.md, "Checks" and "Baselines").

    python scripts/truth_variants.py

1. Planted filler on real texts (R3) with a stricter truth. R3 counts a fact as kept while any kept fragment
   mentions one of its accepted answers ("VerticalNet, Inc., a Pennsylvania corporation" keeps the governing law).
   Here a fact is kept only by the sentence chosen to state it or by the summary that repeats it.
2. Human labels of Poddar et al. (R4, R5). For part of the responses the fact is what the reader reads from the
   fragments people labelled minimal answer, so whether a response is scored depends on those labels. The
   comparison with length again: without the responses for which the reader has neither the reference answer
   nor such an own answer (answer source 'none'), and on the responses whose fact is the reference answer.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def strict_rows(rows: list[dict], items: dict) -> list[dict]:
    """R3 rows with loo_safety and loo_filler_precision recomputed: a fact is stated only by the sentence chosen
    for it (realcheck.fact_sentence) and by the summary."""
    from drybench.realcheck import NOT_STATED, document_sentences, fact_sentence

    out = []
    for r in rows:
        it = items[r["item_id"]]
        sents = document_sentences(it.document)
        chosen: dict[str, set] = {}
        for p in it.required_probes:
            if NOT_STATED not in p.answers:
                chosen.setdefault(fact_sentence(it, p, sents)[1], set()).add(p.fact_id)
        labels, kept = r["labels"], r["loo_kept"]
        facts = [chosen.get(t, set()) if lab == "fact" else set() for t, lab in zip(r["span_texts"], labels)]
        present = set().union(*facts)
        facts = [present if lab == "filler:summary" else f for f, lab in zip(facts, labels)]
        covered = set().union(*(f for f, k in zip(facts, kept) if k))
        removed = [i for i, k in enumerate(kept) if not k]
        fine = [i for i in removed if facts[i] <= covered]
        out.append({**r, "loo_safety": len(present & covered) / len(present) if present else 1.0,
                    "loo_filler_precision": len(fine) / len(removed) if removed else 1.0})
    return out


def planted(real: Path) -> None:
    from drybench.compare import read_rows
    from drybench.realcheck import summarize_planted
    from drybench.report import _readers
    from drybench.rescore import has_missing
    from drybench.types import load_items

    items = {i.item_id: i for i in load_items(real / "items.jsonl")}
    subs, _ = _readers(real, list(items.values()))
    print("R3, planted filler on real texts: truth by mention (as published) -> by the chosen sentence")
    print("reader | responses | no fact lost | responses with a lost fact | precision | responses that change")
    for label, sub in subs:
        rows = [r for r in read_rows(sub / "filler_check_real" / "rows.jsonl")
                if not has_missing(items[r["item_id"]]) and r.get("loo_status") == "ok"]
        new = strict_rows(rows, items)
        a, b = summarize_planted(rows)["overall"], summarize_planted(new)["overall"]
        changed = sum((x["loo_safety"], x["loo_filler_precision"]) != (y["loo_safety"], y["loo_filler_precision"])
                      for x, y in zip(rows, new))
        print(f"{label} | {a['n_responses']} | {a['no_fact_lost']:.3f} -> {b['no_fact_lost']:.3f} | "
              f"{a['responses_with_lost_fact']} -> {b['responses_with_lost_fact']} | "
              f"{a['precision_pooled']:.3f} -> {b['precision_pooled']:.3f} | {changed}")


def human(real: Path) -> None:
    from drybench.compare import cases_for, compare, fill_unscored, method_list, read_rows
    from drybench.report import _readers
    from drybench.types import load_items

    subs, _ = _readers(real, load_items(real / "items.jsonl"))
    labels = [label for label, _ in subs]
    maps = [f"map:{label}" for label in labels]
    source = {(label, r["case"]): r["answer_source"] for label, sub in subs
              for r in read_rows(sub / "human_check" / "rows.jsonl")}
    cases = cases_for("human_poddar", subs, real)

    def where(cond):
        return [c for c in cases if all(cond(source.get((label, c["id"]))) for label in labels)]

    variants = [("scored by every reader (R5, first table)", cases, False),
                ("also the responses a reader reads no fact from, all removable (R5, second table)", cases, True),
                ("the same without answer source 'none'", where(lambda s: s != "none"), True),
                ("the same, fact = reference answer only", where(lambda s: s == "reference"), True)]
    print("\nR4/R5, human labels: filler map (main reader) vs length")
    print("responses | n | Spearman: filler map, length | pairs 'shorter but more removable' ranked right (tied) "
          "| partial Spearman given length")
    for name, cs, fill in variants:
        if fill:
            cs = fill_unscored(cs, maps)
        t, info = compare(cs, method_list(cs, labels), budget_from=maps[0])
        m = t.loc[f"filler map (reader {labels[0]})"]
        print(f"{name} | {info['responses']} | {m['responses: Spearman']:.2f}, "
              f"{t.loc['length (characters)', 'responses: Spearman']:.2f} | "
              f"{m['shorter but more removable: ranked right']:.2f} ({m['… tied']:.2f}) | "
              f"{m['partial Spearman given length']}")


def main() -> None:
    warnings.filterwarnings("ignore")
    real = REPO / "results" / "dev_v1.0" / "real"
    planted(real)
    human(real)


if __name__ == "__main__":
    main()
