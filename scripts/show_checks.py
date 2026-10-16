#!/usr/bin/env python3
"""Look inside the two validation checks of a run: the numbers and the examples behind them.

  python scripts/show_checks.py filler RUN/filler_check/rows.jsonl [RUN/reader2_*/filler_check/rows.jsonl]
      Simulated responses with planted filler (known truth for every fragment), read by the reader:
      no fact lost, precision, recall over the planted filler, repeats, and fragment-by-fragment
      examples: the truth next to what the filler map deleted.
  python scripts/show_checks.py readers RUN
      The same real responses scored by the main reader and by the second reader (RUN/reader2_*):
      per-response rank agreement, the ranking of model × prompt groups by each reader, and
      examples of responses where the two readers delete the same or different fragments.

  python scripts/show_checks.py examples RUN [--model M] [--variant default] [--design "2-hop+explain"]
      Real responses of the evaluated models, before and after the filler is deleted: the request,
      the model's answer, what is left, and every deleted fragment with its kind (uninformative, duplicate,
      wrong statement), plus tokens before → after and the facts the reader gets.

Options: --examples N (default 3), --seed S (which examples), --mode loo|loo_t1|loo_t2 (filler);
examples: --order filler|random (default filler: the most removable first), --reader main|<reader2 label>.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the drybench next to this script

FILLER_KINDS = ("uninformative", "duplicate", "removed", "misleading")


def _clip(text: str, n: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ---------------------------------------------------------------- filler-map check (simulated filler)

def _requests_from_config(rows_path: Path) -> dict[str, str]:
    """Requests of old rows files (before rows stored them): rebuild the items from summary.json."""
    summ = rows_path.parent / "summary.json"
    try:
        from drybench.items import generate_items

        cfg = json.loads(summ.read_text())["config"]
        cells = cfg["cells"]
        items = generate_items(cfg["items"] // len(cells), seed=cfg["seed"], cells=cells)
        return {it.item_id: it.request for it in items}
    except Exception:
        return {}


def _span_texts(row: dict) -> list[str] | None:
    if row.get("span_texts"):
        return row["span_texts"]
    if row.get("segmenter_aligned"):
        from drybench.segment import segment

        return [s.text for s in segment(row["text"])]
    return None


def _print_response(row: dict, mode: str, request: str | None) -> None:
    labels, kept = row["labels"], row[f"{mode}_kept"]
    texts = _span_texts(row) or [""] * len(labels)
    print(f"\n  [{row['item_id']} · {row['condition']} · true filler {row['W_true']:.2f}, "
          f"measured {row[f'{mode}_W']:.2f}]")
    if request:
        print(f"  request: {request}")
    print(f"    {'truth':<19} {'filler map':<9} fragment")
    for lab, k, t in zip(labels, kept, texts):
        mark = "keep" if k else "DELETE"
        flag = ""
        if lab == "fact" and not k:
            flag = "   ← fact deleted"
        elif lab.startswith("filler:") and lab != "filler:summary" and k:
            flag = "   ← filler left in"
        print(f"    {lab:<19} {mark:<9} {_clip(t)}{flag}")


def cmd_filler(args: argparse.Namespace) -> None:
    from drybench.simulate import filler_check_metrics

    for path in map(Path, args.rows):
        rows = [json.loads(line)["row"] for line in path.read_text().splitlines() if line.strip()]
        ok = [r for r in rows if r.get(f"{args.mode}_status") == "ok"]
        m = filler_check_metrics(rows, args.mode)
        print(f"\n=== {path}  ({args.mode}) ===")
        if not m["n_responses"]:
            print("no scored responses")
            continue
        old = sum(r[f"{args.mode}_filler_recall"] for r in ok) / len(ok)
        lo, hi = m["recall_pooled_ci"]
        wlo, whi = m["recall_with_filler_ci"]
        print(f"responses: {m['n_responses']} ({m['n_with_filler']} with planted filler, "
              f"{m['n_filler_spans']} planted filler fragments; the rest have none)")
        print(f"no fact lost: {m['n_responses'] - m['responses_with_lost_fact']} of {m['n_responses']} responses")
        print(f"precision: {m['precision_pooled']:.3f} of {m['spans_removed']} deleted fragments were filler "
              "(or a repeat of a fact kept elsewhere)")
        print(f"recall over planted filler fragments: {m['recall_pooled']:.3f} [{lo:.3f}, {hi:.3f}]; "
              f"mean per response with filler: {m['recall_with_filler']:.3f} [{wlo:.3f}, {whi:.3f}]")
        print(f"   (mean over all responses, as in the old T1 'filler recall': {old:.3f} — "
              "a response without filler counts as 1.0 there)")
        print("by type (deleted / planted): " + ", ".join(
            f"{t} {v['removed']}/{v['planted']}" for t, v in m["recall_by_type"].items()))
        if m["repeats"]:
            print(f"repeats (a summary restating the facts): resolved in {m['repeats_resolved']:.2f} "
                  f"of {m['repeats']} responses (after deletion no fact is stated twice)")
        print(f"filler share W vs the truth: mean |error| {m['W_mae']:.3f}, mean error {m['W_bias']:+.3f} "
              "(negative = the map leaves some filler in)")

        requests = {r["item_id"]: r["request"] for r in ok if r.get("request")} or _requests_from_config(path)
        rng = random.Random(args.seed)
        heavy = [r for r in ok if r["condition"] == "heavy"]
        shown = rng.sample(heavy, min(args.examples, len(heavy)))
        print(f"\nExamples (random '{'heavy'}' responses: all facts + 5 planted filler fragments):")
        for r in shown:
            _print_response(r, args.mode, requests.get(r["item_id"]))
        misses = [r for r in ok if r not in shown and (
            r[f"{args.mode}_safety"] < 1.0 or any(
                lab.startswith("filler:") and lab != "filler:summary" and k
                for lab, k in zip(r["labels"], r[f"{args.mode}_kept"])))]
        if misses:
            print(f"\nMisses ({len(misses)} responses lose a fact or leave planted filler in; showing "
                  f"{min(args.examples, len(misses))}):")
            for r in rng.sample(misses, min(args.examples, len(misses))):
                _print_response(r, args.mode, requests.get(r["item_id"]))


# ---------------------------------------------------------------- two readers on the same responses

def _reader_tables(run: Path, sub: Path, items):
    from drybench.analysis import build_tables

    scored = [json.loads(line) for line in (sub / "scored.jsonl").read_text().splitlines() if line.strip()]
    return build_tables(items, scored, [])


def cmd_readers(args: argparse.Namespace) -> None:
    import warnings

    import numpy as np

    warnings.filterwarnings("ignore")  # constant columns in tiny runs

    from drybench.types import load_items

    run = Path(args.run)
    seconds = sorted(p for p in run.glob("reader2_*") if (p / "scored.jsonl").exists())
    if not (run / "scored.jsonl").exists() or not seconds:
        raise SystemExit(f"need {run}/scored.jsonl and {run}/reader2_*/scored.jsonl")
    second = seconds[0]
    items = load_items(run / "items.jsonl")
    requests = {it.item_id: it.request for it in items}
    ia, sa = _reader_tables(run, run, items)
    ib, sb = _reader_tables(run, second, items)
    lab2 = second.name.replace("reader2_", "")
    m = ia.merge(ib, on=["model", "variant", "item_id"], suffixes=("_1", "_2"))
    both = m[(m["status_1"] == "ok") & (m["status_2"] == "ok")]
    print(f"=== main reader vs {lab2}: {len(both)} responses scored by both ===")
    print("\nPer response, Spearman ρ between the two readers' numbers (does the same response get the "
          "same place in the order from least to most filler?):")
    for c, name in (("W_removable", "removable share"), ("overhead", "overhead"), ("W", "filler"),
                    ("W_misleading", "wrong statements"), ("facts", "facts")):
        if f"{c}_1" in both:
            print(f"  {name:<17} ρ = {both[f'{c}_1'].corr(both[f'{c}_2'], method='spearman'):.3f}")

    def gmean(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x) & (x > 0)]
        return float(np.exp(np.log(x).mean())) if len(x) else float("nan")

    def groups(t):  # each reader over its own scored responses, as in the paper tables
        t = t[t["status"] == "ok"]
        return t.groupby(t["model"] + " | " + t["variant"]).agg(rem=("W_removable", "mean"), ovh=("overhead", gmean))

    g = groups(ia).join(groups(ib), lsuffix="_1", rsuffix="_2")
    g["rank_1"] = g["rem_1"].rank(ascending=False).astype(int)
    g["rank_2"] = g["rem_2"].rank(ascending=False).astype(int)
    g = g.sort_values("rem_1", ascending=False)
    print("\nGroups (model | prompt) by mean removable share — each reader's value and rank "
          "(1 = most filler); overhead as a geometric mean:")
    print(g.rename(columns={"rem_1": "removable main", "rem_2": f"removable {lab2}",
                            "rank_1": "rank main", "rank_2": f"rank {lab2}",
                            "ovh_1": "overhead main", "ovh_2": f"overhead {lab2}"}).round(3).to_string())
    print(f"  Kendall τ of the group ranking: removable {g['rem_1'].corr(g['rem_2'], method='kendall'):.2f}, "
          f"overhead {g['ovh_1'].corr(g['ovh_2'], method='kendall'):.2f}")

    s = sa.merge(sb, on=["model", "variant", "item_id", "span_idx"], suffixes=("_1", "_2"))
    d1, d2 = (s["kind_1"] != "kept").to_numpy(), (s["kind_2"] != "kept").to_numpy()
    po = float((d1 == d2).mean())
    pe = d1.mean() * d2.mean() + (1 - d1.mean()) * (1 - d2.mean())
    print(f"\nFragments: {len(s)}; same decision (delete / keep) for {100 * po:.1f}%, Cohen's κ = "
          f"{(po - pe) / (1 - pe):.2f}")
    print(f"  both delete {int((d1 & d2).sum())}, both keep {int((~d1 & ~d2).sum())}, "
          f"only main deletes {int((d1 & ~d2).sum())}, only {lab2} deletes {int((~d1 & d2).sum())}")

    s["diff"] = d1 != d2
    per = s.groupby(["model", "variant", "item_id"]).agg(n_diff=("diff", "sum"), n_del=("kind_1", lambda k: (k != "kept").sum()))
    same = per[(per["n_diff"] == 0) & (per["n_del"] > 0)].index.tolist()
    diff = per[per["n_diff"] > 0].index.tolist()
    print(f"  responses where the two maps are identical: {100 * (per['n_diff'] == 0).mean():.1f}% of {len(per)}")
    rng = random.Random(args.seed)
    for title, keys in (("Examples where both readers delete the same fragments", same),
                        ("Examples where the readers differ", diff)):
        pick = rng.sample(keys, min(args.examples, len(keys)))
        if not pick:
            continue
        print(f"\n{title}:")
        for key in pick:
            d = s[(s["model"] == key[0]) & (s["variant"] == key[1]) & (s["item_id"] == key[2])].sort_values("span_idx")
            print(f"\n  [{key[0]} | {key[1]} | {key[2]}] request: {_clip(requests.get(key[2], ''), 140)}")
            print(f"    {'main':<11} {lab2:<11} fragment")
            for _, r in d.iterrows():
                mark = "   ← differ" if r["diff"] else ""
                print(f"    {r['kind_1']:<11} {r['kind_2']:<11} {_clip(r['text_1'])}{mark}")


# ---------------------------------------------------------------- real responses before / after

KIND_NOTE = {"uninformative": "filler", "duplicate": "filler: repeats a fact", "removed": "filler",
             "misleading": "wrong statement"}


def _after(response: str, spans: list[dict]) -> str:
    """The response with the deleted fragments cut out (layout of the kept text preserved)."""
    out, pos = [], 0
    for sp in spans:
        if not sp["kept"] and "start" in sp:
            out.append(response[pos:sp["start"]])
            pos = sp["end"]
    out.append(response[pos:])
    lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in "".join(out).split("\n")]
    keep = [l for l in lines if l.strip(" -*•:#0123456789.")]  # drop lines left with bullets or numbers only
    return "\n".join(keep).strip()


def _block(text: str, mark: str = "│") -> str:
    return "\n".join(f"    {mark} {line}" for line in str(text).strip().splitlines()) or f"    {mark}"


def cmd_examples(args: argparse.Namespace) -> None:
    import warnings

    from drybench.report import load_reader
    from drybench.types import load_items, read_jsonl

    warnings.filterwarnings("ignore")
    run = Path(args.run)
    sub = run if args.reader == "main" else run / f"reader2_{args.reader}"
    if not (sub / "scored.jsonl").exists():
        raise SystemExit(f"no scores in {sub}; readers: main" + "".join(
            f", {p.name.replace('reader2_', '')}" for p in sorted(run.glob("reader2_*"))))
    items = load_items(run / "items.jsonl")
    requests = {it.item_id: it.request for it in items}
    tables = load_reader(run, sub, args.reader, items)
    it, sp = tables["items"], tables["spans"]
    key = lambda r: (r["model"], r.get("variant", "default"), r["item_id"])  # noqa: E731
    scored = {key(r): r for r in read_jsonl(sub / "scored.jsonl")}
    responses = {key(r): r["response"] for r in read_jsonl(run / "responses.jsonl")} \
        if (run / "responses.jsonl").exists() else {}
    d = it[it["status"] == "ok"]
    if not args.design and "missing" in d:  # as in the tables: missing-fact items only when asked for by design
        d = d[~d["missing"].astype(bool)]
    for col, val in (("model", args.model), ("variant", args.variant), ("design", args.design),
                     ("source", args.source)):
        if val:
            d = d[d[col] == val]
    if d.empty:
        raise SystemExit("no scored responses match; models: " + ", ".join(sorted(it["model"].unique()))
                         + "; designs: " + ", ".join(sorted(it["design"].unique()))
                         + "; sources: " + ", ".join(sorted(it["source"].astype(str).unique())))
    if args.order == "filler":
        d = d[d["W_removable"] > 0].sort_values(["W_removable", "tokens"], ascending=False)
    else:
        d = d.sample(frac=1.0, random_state=args.seed)
    print(f"{len(d)} responses match (reader {args.reader}); showing {min(args.examples, len(d))}, "
          f"{'most removable first' if args.order == 'filler' else 'in random order'}.")
    for _, row in d.head(args.examples).iterrows():
        k = (row["model"], row["variant"], row["item_id"])
        r = scored[k]
        kinds = sp[(sp["model"] == k[0]) & (sp["variant"] == k[1]) & (sp["item_id"] == k[2])] \
            .sort_values("span_idx")["kind"].tolist()
        spans = [dict(s, kind=kinds[i] if i < len(kinds) else s.get("kind")) for i, s in enumerate(r["spans"])]
        resp = responses.get(k, " ".join(s["text"] for s in spans))
        print("\n" + "=" * 100)
        f2 = lambda v: "n/a" if v != v else f"{v:.2f}"  # noqa: E731
        print(f"{k[0]} | {k[1]} | {k[2]} ({row['design']}): tokens {int(row['tokens'])} → {int(row['tokens_kept'])}, "
              f"removable {f2(row['W_removable'])} (filler {f2(row['W'])}, wrong {f2(row['W_misleading'])}), "
              f"facts the reader gets: {f2(row['facts'])}")
        print(f"REQUEST: {requests.get(k[2], '')}")
        print("MODEL'S ANSWER:")
        print(_block(resp))
        print("AFTER DELETING:")
        print(_block(_after(resp, spans) or "(nothing left)"))
        print("DELETED:")
        for s in spans:
            if not s["kept"]:
                print(f"    {s['kind']:<11} ({KIND_NOTE.get(s['kind'], s['kind'])}): {_clip(s['text'], 150)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("filler", help="simulated responses with known filler")
    w.add_argument("rows", nargs="+", help="filler_check/rows.jsonl of one or more readers")
    w.add_argument("--mode", default="loo", choices=["loo", "loo_t1", "loo_t2", "loo_global"])
    w.set_defaults(fn=cmd_filler)
    r = sub.add_parser("readers", help="the same responses scored by two readers")
    r.add_argument("run", help="run folder with scored.jsonl, items.jsonl and reader2_*/scored.jsonl")
    r.set_defaults(fn=cmd_readers)
    e = sub.add_parser("examples", help="real responses before and after deleting the filler")
    e.add_argument("run", help="run folder with items.jsonl, responses.jsonl and scored.jsonl")
    e.add_argument("--model")
    e.add_argument("--variant", help="default or concise")
    e.add_argument("--design", help='e.g. "k=4-6", "2-hop+explain", "missing fact" (real texts: "3-hop+explain")')
    e.add_argument("--source", help="real texts: squad2, musique, cuad or poddar")
    e.add_argument("--order", default="filler", choices=["filler", "random"])
    e.add_argument("--reader", default="main", help="main, or the label of a reader2_* folder")
    e.set_defaults(fn=cmd_examples)
    for p in (w, r, e):
        p.add_argument("--examples", type=int, default=3)
        p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
