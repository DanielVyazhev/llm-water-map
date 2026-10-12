#!/usr/bin/env python3
"""Missing-fact answers of a run: the lexical rule (drybench.abstain) against hand labels, and how well
each reader's own reading of 'not stated' separates the answers that say it from the others.

    python scripts/check_abstain.py RUN_DIR [--labels results/abstain_labels.json] [--name t4_n20]
    python scripts/check_abstain.py --csv abstain_sample.csv --key abstain_key.jsonl   # held-out labels

RUN_DIR holds items.jsonl, responses.jsonl, scored.jsonl and optionally reader2_*/scored.jsonl. The
labels file holds, per run name (default: the folder's name), one letter per answer in the file's
'order' of model × prompt (A = says the fact is not given, B = says so and guesses a value).
--csv: the sample of `drybench abstain-sample` with its label column filled in (A/B/V/O), checked against
the rule as it is (held-out: labelled without looking at the rule's output).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> None:
    from drybench.abstain import abstains
    from drybench.reader import is_missing
    from drybench.report import auc
    from drybench.types import load_items, read_jsonl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?")
    ap.add_argument("--labels", default=str(REPO / "results" / "abstain_labels.json"))
    ap.add_argument("--name", default=None, help="run name in the labels file (default: the folder's name)")
    ap.add_argument("--csv", default=None, help="filled-in abstain_sample.csv (held-out labels)")
    ap.add_argument("--key", default=None, help="its abstain_key.jsonl (default: next to the csv)")
    args = ap.parse_args(argv)
    if args.csv:
        from drybench.annotate import abstain_check

        res = abstain_check(args.csv, args.key or Path(args.csv).with_name("abstain_key.jsonl"))
        print(f"{res['labelled']} labelled answers {res['labels']}")
        for name, v in res.items():
            if isinstance(v, dict) and "precision" in v:
                print(f"  {name}: {v['n']} answers, {v['say_it']} say the fact is not given; rule precision "
                      f"{v['precision']:.3f} ({v['tp']}/{v['tp'] + v['fp']}), recall {v['recall']:.3f} "
                      f"({v['tp']}/{v['tp'] + v['fn']})")
        if res["disagreements"]:
            print("  rule and label disagree on: " + ", ".join(res["disagreements"]))
        return
    if not args.run:
        ap.error("give RUN_DIR, or --csv")
    run = Path(args.run)
    items = {it.item_id: it for it in load_items(run / "items.jsonl")}
    resp = {(r["model"], r.get("variant", "default"), r["item_id"]): r["response"]
            for r in read_jsonl(run / "responses.jsonl")}
    lab_file = json.loads(Path(args.labels).read_text()) if Path(args.labels).exists() else {}
    order = [tuple(x) for x in lab_file.get("order", [])]
    labels = lab_file.get("runs", {}).get(args.name or run.name, {})
    readers = {"main": run / "scored.jsonl"} | {p.name.replace("reader2_", ""): p / "scored.jsonl"
                                                for p in sorted(run.glob("reader2_*")) if (p / "scored.jsonl").exists()}
    rows = {}
    for label, path in readers.items():
        for s in read_jsonl(path):
            it = items[s["item_id"]]
            j = next((i for i, p in enumerate(it.required_probes) if is_missing(p)), None)
            if j is None:
                continue
            key = (s["model"], s.get("variant", "default"), s["item_id"])
            row = rows.setdefault(key, {"source": it.knobs.get("source", "synthetic"),
                                        "rule": abstains(resp[key], it), "hand": None})
            if labels.get(key[2]) and key[:2] in order:
                row["hand"] = labels[key[2]][order.index(key[:2])] in "AB"
            row[label] = float(np.exp(s["per_probe_lp"][j]))
    if not rows:
        raise SystemExit("no missing-fact items in this run")
    hand = [r for r in rows.values() if r["hand"] is not None]
    print(f"{len(rows)} missing-fact answers; the rule says the fact is not given in "
          f"{np.mean([r['rule'] for r in rows.values()]):.3f}")
    if hand:
        tp = sum(r["rule"] and r["hand"] for r in hand)
        fp = sum(r["rule"] and not r["hand"] for r in hand)
        fn = sum(r["hand"] and not r["rule"] for r in hand)
        print(f"rule vs hand labels ({len(hand)} answers, {tp + fn} say it): precision {tp / max(tp + fp, 1):.3f} "
              f"({tp}/{tp + fp}), recall {tp / max(tp + fn, 1):.3f} ({tp}/{tp + fn})")
    truth = "hand" if hand else "rule"
    pool = hand or list(rows.values())
    for src in sorted({r["source"] for r in pool}) + ["all"]:
        sub = [r for r in pool if src in ("all", r["source"])]
        parts = []
        for label in readers:
            yes = [r[label] for r in sub if r[truth] and label in r]
            no = [r[label] for r in sub if not r[truth] and label in r]
            if yes and no:
                parts.append(f"{label} AUC {auc(yes, no):.2f} (p ≥ 0.5: {np.mean(np.array(yes) >= 0.5):.2f} of those "
                             f"that say it, {np.mean(np.array(no) >= 0.5):.2f} of the others)")
        print(f"  {src} ({len(sub)}; against the {truth}): " + "; ".join(parts))


if __name__ == "__main__":
    main()
