"""Human labels of the benchmark's own answers: the blind samples to label and the report on the labels.

build_sample     item by item: several answers to the same item (different models and prompts), so the
                 length-confound pair test can compare answers to the same question (where the Yap excess
                 ranks like length); items with a missing fact are left out, and items whose gold answer the
                 reader cannot read; an answer from which the reader reads none of the requested facts stays in
                 (the filler map's verdict on it: it delivers nothing, all of it is removable); identical answers
                 are labelled once.
                 Writes sample.jsonl (what annotators see: source text, request, response, fragments) and
                 key.jsonl (run, item, models and prompts, the filler map of every reader per fragment).
build_abstain    answers to items with a missing fact, for held-out labels of the lexical abstention rule
                 (drybench.abstain): abstain_sample.csv (request, the question the source does not answer,
                 answer, an empty label column) and abstain_key.jsonl.
report           the labels against each other (agreement between annotators) and against the filler map,
                 the LLM judge, Selective Context and the cheap baselines (drybench.compare), over all
                 answers and within items.
"""

from __future__ import annotations

import csv
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from drybench.guidelines import LABELS

PAGE_FORMATS = ("span-labels/1", "dry-span-labels/1")
ABSTAIN_LEGEND = {"A": "says the fact is not given", "B": "says so and also guesses a value",
                  "V": "gives a value", "O": "neither (answers something else)"}


def _source(item) -> str:
    return item.knobs.get("source", "synthetic")


def _design(item) -> str:
    from drybench.report import design

    return design({**item.knobs, "source": _source(item), "subset": item.knobs.get("subset")})


def _norm(text: str) -> str:
    return " ".join(text.split())


def _readers(run: Path) -> list[tuple[str, Path]]:
    return [("main", run)] + [(p.name.replace("reader2_", ""), p) for p in sorted(run.glob("reader2_*"))
                              if (p / "scored.jsonl").exists() or (p / "scored_sample.jsonl").exists()]


def _scored(path: Path) -> dict[tuple, dict]:
    from drybench.types import read_jsonl

    return {(r["model"], r.get("variant", "default"), r["item_id"]): r for r in read_jsonl(path)} if path.exists() else {}


def _run_answers(run: Path) -> tuple[dict, dict[tuple, str], list[tuple[str, dict]]]:
    from drybench.types import load_items, read_jsonl

    items = {it.item_id: it for it in load_items(run / "items.jsonl")}
    resp = {(r["model"], r.get("variant", "default"), r["item_id"]): r["response"]
            for r in read_jsonl(run / "responses.jsonl")}
    # scored_sample.jsonl: answers of the annotation sample a reader that scores part of the run scored only
    # for the sample (kept apart, so that its tables stay on its own part of the items)
    return items, resp, [(label, {**_scored(sub / "scored_sample.jsonl"), **_scored(sub / "scored.jsonl")})
                         for label, sub in _readers(run)]


def build_sample(runs: Sequence[str | Path], sources: Sequence[str] | None = None, items_per_source: int = 6,
                 per_item: int = 8, min_distinct: int = 3, seed: int = 0) -> tuple[list[dict], list[dict], dict]:
    """(sample rows, key rows, stats). sources: e.g. ["synthetic", "squad2", "musique", "cuad"] (default:
    every source of the runs, in that order)."""
    from drybench.baselines import stopword_share, yap_excess_chars
    from drybench.items import _doc_words
    from drybench.rescore import has_missing

    rng = random.Random(seed)
    cands: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))  # source -> design -> items
    for run in map(Path, runs):
        items, resp, scored = _run_answers(run)
        main = scored[0][1]
        by_item: dict[str, dict[str, dict]] = defaultdict(dict)
        for key, s in main.items():
            it = items.get(key[2])
            if (it is None or s.get("status") not in ("ok", "no_signal") or not s.get("spans") or key not in resp
                    or has_missing(it)):
                continue
            text = resp[key]
            d = by_item[key[2]].setdefault(_norm(text), {"response": text, "groups": [], "scored": s})
            d["groups"].append(list(key[:2]))
        for iid, answers in by_item.items():
            if len(answers) >= min_distinct:
                it = items[iid]
                cands[_source(it)][_design(it)].append((run, it, list(answers.values()), scored))
    order = list(sources) if sources else sorted(cands, key=lambda s: (s != "synthetic", s))
    usage: Counter = Counter()
    chosen = []
    for src in order:
        pools = {d: sorted(v, key=lambda x: x[1].item_id) for d, v in cands.get(src, {}).items()}
        designs = sorted(pools)
        rng.shuffle(designs)
        for d in designs:
            rng.shuffle(pools[d])
        picked: list = []
        while len(picked) < items_per_source and any(pools.values()):
            for d in designs:
                if pools[d] and len(picked) < items_per_source:
                    picked.append(pools[d].pop())
        for run, it, answers, scored in picked:
            pool, take = list(answers), []
            while pool and len(take) < per_item:  # the least used model × prompt first
                best = min(pool, key=lambda a: (min(usage[tuple(g)] for g in a["groups"]), rng.random()))
                pool.remove(best)
                take.append(best)
                for g in best["groups"]:
                    usage[tuple(g)] += 1
            rng.shuffle(take)
            chosen.append((run, it, take, scored))
    rng.shuffle(chosen)
    sample, key = [], []
    for run, it, take, scored in chosen:
        label, _ = _doc_words(it)
        for a in take:
            uid = f"r{len(sample) + 1:03d}"
            spans = a["scored"]["spans"]
            sample.append({"uid": uid, "source_label": label, "document": it.document, "request": it.request,
                           "response": a["response"], "spans": [{"i": k, "text": s["text"]} for k, s in enumerate(spans)]})
            g0 = tuple(a["groups"][0]) + (it.item_id,)
            row = {"uid": uid, "run": run.name, "source": _source(it), "design": _design(it),
                   "item_id": it.item_id, "model": a["groups"][0][0], "variant": a["groups"][0][1],
                   "groups": a["groups"], "chars": [len(s["text"]) for s in spans],
                   "yap_excess": yap_excess_chars(a["response"], it.gold_minimal),
                   "stopword_share": stopword_share(a["response"])}
            for lab, sc in scored:
                s = sc.get(g0)
                same = s is not None and [x["text"] for x in s["spans"]] == [x["text"] for x in spans]
                row[f"status_{lab}"] = s.get("status") if same else None
                row[f"kinds_{lab}"] = [x["kind"] for x in s["spans"]] if same and s.get("status") == "ok" else None
                row[f"seconds_{lab}"] = s.get("seconds") if same else None
            key.append(row)
    stats = {"responses": len(sample), "items": len(chosen),
             "by_source": dict(Counter(k["source"] for k in key)),
             "items_by_source": dict(Counter(_source(it) for _, it, _, _ in chosen)),
             "by_group": {f"{m} | {v}": n for (m, v), n in sorted(usage.items())},
             "fragments": sum(len(s["spans"]) for s in sample)}
    return sample, key, stats


def write_sample(out_dir: str | Path, sample: list[dict], key: list[dict], stats: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample), encoding="utf-8")
    (out / "key.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in key), encoding="utf-8")
    (out / "sample_stats.json").write_text(json.dumps(stats, indent=1))


def build_abstain(runs: Sequence[str | Path], n: int = 150, seed: int = 0) -> tuple[list[dict], list[dict], dict]:
    """(csv rows, key rows, stats): answers to items with a missing fact, balanced over sources and then
    models; identical answers to the same item once."""
    from drybench.reader import is_missing

    rng = random.Random(seed)
    by_src: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for run in map(Path, runs):
        items, resp, _ = _run_answers(run)
        seen = set()
        for (model, variant, iid), text in sorted(resp.items()):
            it = items.get(iid)
            if it is None or not any(is_missing(p) for p in it.required_probes):
                continue
            if (run.name, iid, _norm(text)) in seen:
                continue
            seen.add((run.name, iid, _norm(text)))
            by_src[_source(it)][model].append((run, it, model, variant, text))
    srcs = sorted(by_src)
    for s in srcs:
        for m in by_src[s]:
            rng.shuffle(by_src[s][m])
    picked: list = []
    while len(picked) < n and any(lst for d in by_src.values() for lst in d.values()):
        for s in srcs:
            models = [m for m in sorted(by_src[s]) if by_src[s][m]]
            if not models or len(picked) >= n:
                continue
            m = min(models, key=lambda x: (sum(1 for p in picked if p[0] == s and p[1][2] == x), rng.random()))
            picked.append((s, by_src[s][m].pop()))
    rng.shuffle(picked)
    rows, key = [], []
    for k, (src, (run, it, model, variant, text)) in enumerate(picked, 1):
        uid = f"a{k:03d}"
        miss = [p.question for p in it.required_probes if is_missing(p)]
        rows.append({"uid": uid, "request": it.request, "question the source does not answer": " | ".join(miss),
                     "answer": text, "label (A/B/V/O)": ""})
        key.append({"uid": uid, "run": run.name, "source": src, "model": model, "variant": variant,
                    "item": it.to_dict(), "answer": text})
    stats = {"answers": len(rows), "by_source": dict(Counter(k["source"] for k in key)),
             "by_model": dict(Counter(k["model"] for k in key))}
    return rows, key, stats


def write_abstain(out_dir: str | Path, rows: list[dict], key: list[dict], stats: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "abstain_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["uid"])
        w.writeheader()
        w.writerows(rows)
    (out / "abstain_key.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in key),
                                          encoding="utf-8")
    legend = "\n".join(f"{k} = {v}" for k, v in ABSTAIN_LEGEND.items())
    (out / "abstain_legend.txt").write_text(
        "Each row: a request about a source text that does NOT answer one of the requested questions (the column "
        "'question the source does not answer'). Read the answer and fill 'label (A/B/V/O)' for that question:\n"
        + legend + "\nDecide from the answer alone; do not look at abstain_key.jsonl.\n", encoding="utf-8")
    (out / "abstain_stats.json").write_text(json.dumps(stats, indent=1))


def abstain_check(csv_path: str | Path, key_path: str | Path) -> dict:
    """Precision and recall of the lexical rule against filled-in A/B/V/O labels (A and B = says the fact
    is not given)."""
    from drybench.abstain import abstains
    from drybench.types import Item

    key = {r["uid"]: r for r in map(json.loads, Path(key_path).read_text(encoding="utf-8").splitlines()) if r}
    labelled, bad = [], []
    text = Path(csv_path).read_text(encoding="utf-8-sig")  # spreadsheets add a byte-order mark ...
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")  # ... and save with ';' or tabs
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    if not any((r.get("uid") or "").strip() in key for r in rows):
        raise ValueError(f"{csv_path}: no row has a uid of {key_path} (columns: {list(rows[0]) if rows else []})")
    for row in rows:
        uid = (row.get("uid") or "").strip()
        lab = (next((v for k, v in row.items() if k and k.strip().startswith("label")), "") or "").strip().upper()[:1]
        if uid not in key:
            continue
        if lab not in ABSTAIN_LEGEND:
            if lab:
                bad.append(uid)
            continue
        k = key[uid]
        rule = abstains(k["answer"], Item.from_dict(k["item"]))
        labelled.append({"uid": uid, "source": k["source"], "model": k["model"], "hand": lab in "AB",
                         "label": lab, "rule": bool(rule)})
    out: dict = {"labelled": len(labelled), "unreadable_labels": bad,
                 "labels": dict(Counter(x["label"] for x in labelled))}
    for name, sub in [("all", labelled)] + [(s, [x for x in labelled if x["source"] == s])
                                            for s in sorted({x["source"] for x in labelled})]:
        tp = sum(x["rule"] and x["hand"] for x in sub)
        fp = sum(x["rule"] and not x["hand"] for x in sub)
        fn = sum(x["hand"] and not x["rule"] for x in sub)
        out[name] = {"n": len(sub), "say_it": tp + fn, "precision": tp / (tp + fp) if tp + fp else float("nan"),
                     "recall": tp / (tp + fn) if tp + fn else float("nan"), "tp": tp, "fp": fp, "fn": fn}
    out["disagreements"] = [x["uid"] for x in labelled if x["rule"] != x["hand"]]
    return out


# ---------------------------------------------------------------- labels and the report

def read_labels(paths: Sequence[str | Path], with_origin: bool = False):
    """annotator -> uid -> {labels, also_needed, unsure, ts}, from page exports (span-labels/1) or JSONL files
    of chat annotators (scripts/chat_annotators.py collect). An annotator id that occurs in two files (two
    people labelling offline are both 'local') is told apart by the file name. with_origin: also annotator ->
    {"kind": "page" | "chat", "dataset": the page's hash of the sample it labelled}."""
    per_file = []
    for p in map(Path, paths):
        text = p.read_text(encoding="utf-8-sig").strip()
        if text.startswith("{") and '"records"' in text[:2000]:
            doc = json.loads(text)
            if doc.get("format") not in PAGE_FORMATS:
                raise ValueError(f"{p}: not a labels export ({doc.get('format')!r})")
            per_file.append((p, doc["records"], "page", (doc.get("dataset") or {}).get("hash")))
        else:
            per_file.append((p, [json.loads(line) for line in text.splitlines() if line.strip()], "chat", None))
    files_with = defaultdict(set)
    for p, recs, _, _ in per_file:
        for r in recs:
            files_with[r.get("annotator") or "local"].add(p)
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    origin: dict[str, dict] = {}
    for p, recs, kind, dataset in per_file:
        for r in recs:
            a = r.get("annotator") or "local"
            name = a if len(files_with[a]) == 1 else f"{a}@{p.stem}"
            origin.setdefault(name, {"kind": kind, "dataset": dataset})
            labs = {int(i): lab for i, lab in zip(r["span_i"], r["labels"]) if lab in LABELS}
            prev = out[name].get(r["uid"])
            if prev is None or (r.get("ts") or 0) >= (prev.get("ts") or 0):
                out[name][r["uid"]] = {"labels": labs, "also_needed": {int(i) for i in r.get("also_needed") or []},
                                       "unsure": {int(i) for i in r.get("unsure") or []}, "ts": r.get("ts") or 0}
    return (dict(out), origin) if with_origin else dict(out)


def removable(entry: dict, k: int) -> bool:
    """A fragment's label counts as removable: filler, or wrong without a correctly given requested fact
    ('+ needed fact' marks a wrong fragment that also delivers one: deleting it would lose that fact)."""
    lab = entry["labels"].get(k)
    return lab == "filler" or (lab == "wrong" and k not in entry["also_needed"])


def page_hash(sample: Sequence[dict]) -> str:
    """The labelling page's hash of a sample (FNV-1a over the UTF-16 code units of uid + response), to check
    that an export was made on this sample."""
    s = "\u0002".join(str(r["uid"]) + "\u0001" + str(r["response"]) for r in sample)
    h = 0x811C9DC5
    raw = s.encode("utf-16-le")
    for i in range(0, len(raw), 2):
        h ^= raw[i] | (raw[i + 1] << 8)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def sample_cases(sample_dir: str | Path) -> list[dict]:
    """Cases of the annotation sample (drybench.compare), without truth: one per response."""
    from drybench.baselines import length_chars

    d = Path(sample_dir)
    key = {r["uid"]: r for r in map(json.loads, (d / "key.jsonl").read_text(encoding="utf-8").splitlines()) if r} \
        if (d / "key.jsonl").exists() else {}
    out = []
    for line in (d / "sample.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        k = key.get(s["uid"], {})
        frags = [x["text"] for x in sorted(s["spans"], key=lambda x: int(x["i"]))]
        out.append({"id": s["uid"], "group": f"{k.get('run', '')}/{k.get('item_id', s['uid'])}",
                    "source": k.get("source", ""), "design": k.get("design", ""),
                    "source_label": s.get("source_label") or "Source text", "document": s["document"],
                    "request": s["request"], "response": s["response"], "fragments": frags,
                    "chars": [len(f) for f in frags], "length": float(length_chars(s["response"])),
                    "yap": float(k.get("yap_excess", np.nan)), "stopwords": float(k.get("stopword_share", np.nan)),
                    "kinds": {lab[len("kinds_"):]: v for lab, v in k.items() if lab.startswith("kinds_")},
                    "status": {lab[len("status_"):]: v for lab, v in k.items() if lab.startswith("status_")},
                    "seconds": {lab[len("seconds_"):]: v for lab, v in k.items() if lab.startswith("seconds_")}})
    return out


def _kappa(a: Sequence[str], b: Sequence[str]) -> float:
    a, b = list(a), list(b)
    if not a:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = float(np.mean([x == y for x, y in zip(a, b)]))
    pe = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def agreement(labels: dict[str, dict[str, dict]], uids_spans: dict[str, list[int]]) -> pd.DataFrame:
    """Every pair of annotators on the responses both labelled completely: fragments, Cohen's κ over the
    three labels and over removable (filler or wrong) vs needed."""
    rows = {}
    names = sorted(labels)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            xa, xb, ra, rb = [], [], [], []
            for uid, idx in uids_spans.items():
                la, lb = labels[a].get(uid), labels[b].get(uid)
                if la and lb and all(k in la["labels"] for k in idx) and all(k in lb["labels"] for k in idx):
                    xa += [la["labels"][k] for k in idx]
                    xb += [lb["labels"][k] for k in idx]
                    ra += [removable(la, k) for k in idx]
                    rb += [removable(lb, k) for k in idx]
            if xa:
                rows[f"{a} vs {b}"] = {"fragments": len(xa), "same label": float(np.mean([x == y for x, y in zip(xa, xb)])),
                                       "κ, 3 labels": _kappa(xa, xb), "κ, removable or not": _kappa(ra, rb)}
    t = pd.DataFrame(rows).T
    t.index.name = "annotators"
    return t


def report(sample_dir: str | Path, label_files: Sequence[str | Path], primary: str | None = None) -> str:
    """Markdown: annotators and their agreement, then every method against the primary annotator's labels
    (all pairs, and pairs of answers to the same item), per source, the wrong-statement label, and the cost
    per response of each method on these responses. The primary annotator is a person (a page export); chat
    LLM annotators are compared like the other methods."""
    from drybench.compare import attach, baseline_outputs, binary, compare, labelled_case, method_list
    from drybench.report import md

    d = Path(sample_dir)
    cases = sample_cases(d)
    labels, origin = read_labels(label_files, with_origin=True)
    if not labels:
        raise SystemExit("no labels in " + ", ".join(map(str, label_files)))
    spans = {c["id"]: list(range(len(c["fragments"]))) for c in cases}
    complete = {a: sum(1 for u, idx in spans.items() if v.get(u) and all(k in v[u]["labels"] for k in idx))
                for a, v in labels.items()}
    people = [a for a in labels if origin[a]["kind"] == "page"]
    if primary is None:
        if not people:
            raise SystemExit("no labels from the labelling page: name the annotator whose labels are the truth "
                             "with --primary")
        primary = max(people, key=lambda a: complete[a])
    if primary not in labels:
        raise SystemExit(f"--primary {primary!r} is not among the annotators: {', '.join(labels)}")
    with open(d / "sample.jsonl", encoding="utf-8") as f:
        expected = page_hash([json.loads(x) for x in f if x.strip()])
    doc = [f"# Human labels of the annotation sample — {d.name}", "",
           f"{len(cases)} responses; annotators (fully labelled responses): "
           + ", ".join(f"{a} {complete[a]}" + (" (chat LLM)" if origin[a]["kind"] == "chat" else "")
                       + (" — the truth below" if a == primary else "") for a in labels) + "."]
    stale = [a for a in labels if origin[a]["dataset"] and origin[a]["dataset"] != expected]
    if stale:
        doc += ["", "**Warning:** labels of " + ", ".join(stale) + " were made on another version of the sample "
                "(the page's hash differs): their uids may name other responses."]
    ag = agreement(labels, spans)
    if not ag.empty:
        doc += ["", "## Agreement between annotators", "",
                "Responses both labelled completely; removable = filler, or wrong without '+ needed fact'.", "", md(ag)]
    truth = labels[primary]
    lab_cases = []
    for c in cases:
        got = truth.get(c["id"])
        n = len(c["fragments"])
        if not got or not all(k in got["labels"] for k in range(n)):
            continue
        # '+ needed fact': the fragment also delivers a requested fact, so it is not removable
        t_lab = ["needed" if (got["labels"][k] == "wrong" and k in got["also_needed"]) else got["labels"][k]
                 for k in range(n)]
        lab_cases.append({**labelled_case(c, t_lab), "wrong": [got["labels"][k] == "wrong" for k in range(n)]})
    def verdict(c: dict, r: str) -> list | None:
        """The filler map's removable fragments; an answer that delivers nothing to the reader: all of them."""
        kinds = c["kinds"].get(r)
        if kinds is not None:
            return [k != "kept" for k in kinds]
        return [True] * len(c["fragments"]) if c.get("status", {}).get(r) == "no_signal" else None

    every = sorted({r for c in lab_cases for r in set(c["kinds"]) | set(c.get("status", {}))
                    if verdict(c, r) is not None}, key=lambda r: (r != "main", r))
    readers = [r for r in every if all(verdict(c, r) is not None for c in lab_cases)]
    partial = [r for r in every if r not in readers]  # scored part of the sample only: own tables below
    for r in every:
        attach(lab_cases, f"map:{r}", {c["id"]: verdict(c, r) for c in lab_cases if verdict(c, r) is not None}, "map")
    for name, rows in baseline_outputs(d, "sample").items():
        attach(lab_cases, name, rows, "judge" if name.startswith("judge-") else "selective")
    others = [a for a in labels if a != primary]
    for a in others:  # other annotators (people or chat LLMs) as methods
        attach(lab_cases, f"annotator-{a}",
               {c["id"]: [removable(labels[a][c["id"]], k) for k in range(len(c["fragments"]))]
                for c in lab_cases if labels[a].get(c["id"])
                and all(k in labels[a][c["id"]]["labels"] for k in range(len(c["fragments"])))}, "map")
    base = method_list(lab_cases, readers)
    budget = f"map:{readers[0]}" if readers else None
    doc += ["", f"## Methods against {primary}'s labels", "",
            "Removable = filler, or wrong without '+ needed fact'. Response-level shares by characters; pairs: the "
            "more removable response (by ≥ 0.2) is the shorter one; partial Spearman given length [95% CI, bootstrap over "
            "items]; Selective Context deletes as many characters as the filler map of reader "
            f"{readers[0] if readers else '—'} (equal budget). Other annotators, and readers that scored part of the "
            "sample, are scored on the responses they labelled or scored, in tables of their own below."]
    for title, within in (("All pairs", False), ("Pairs of answers to the same item (Yap ranks like length)", True)):
        t, info = compare(lab_cases, base, budget_from=budget, within_group=within)
        if not t.empty:
            doc += ["", f"### {title}", "", f"{info['responses']} responses, {info['fragments']} fragments, "
                    f"{info['pairs']} pairs.", "", md(t)]
    for src in sorted({c["source"] for c in lab_cases}):
        t, info = compare([c for c in lab_cases if c["source"] == src], base, budget_from=budget)
        if not t.empty:
            doc += ["", f"### {src}", "", f"{info['responses']} responses, {info['fragments']} fragments, "
                    f"{info['pairs']} pairs.", "", md(t)]
    for a in others:
        t, info = compare(lab_cases, [(f"annotator {a}", f"annotator-{a}")] + base[: len(readers)], budget_from=None)
        if not t.empty:
            doc += ["", f"### Annotator {a} on the responses both labelled", "",
                    f"{info['responses']} responses.", "", md(t)]
    for r in partial:
        t, info = compare(lab_cases, [(f"filler map (reader {r})", f"map:{r}")] + base[: len(readers)], budget_from=None)
        if not t.empty:
            doc += ["", f"### Reader {r} on the responses it scored", "", f"{info['responses']} responses.", "", md(t)]
    lines = []
    for r in every:  # the wrong-statement label: the annotator's 'wrong' vs the map's
        t_w, p_w = [], []
        for c in lab_cases:
            kinds = c["kinds"].get(r)
            if kinds is not None:
                t_w += c["wrong"]
                p_w += [k == "misleading" for k in kinds]
        if t_w:
            b = binary(t_w, p_w)
            lines.append(f"- reader {r}: {sum(t_w)} fragments labelled wrong, {sum(p_w)} labelled wrong by the map — "
                         f"precision {b['precision']:.2f}, recall {b['recall']:.2f}, κ {b['kappa']:.2f}")
    if lines:
        doc += ["", "## Wrong statements: the annotator's 'wrong' vs the map's wrong-statement label", ""] + lines
    conf: dict = defaultdict(Counter)
    for c in lab_cases:
        kinds = c["kinds"].get(readers[0]) if readers else None
        if kinds is None and readers and c.get("status", {}).get(readers[0]) == "no_signal":
            kinds = ["answer delivers nothing"] * len(c["truth"])
        if kinds is not None:
            for t_, k in zip(c["truth"], kinds):
                conf[t_][k] += 1
    if conf:
        tc = pd.DataFrame(conf).T.fillna(0).astype(int)
        tc.index.name = f"label \\ map ({readers[0]})"
        doc += ["", "## What the filler map did with each label", "", md(tc)]
    cost = {}
    for r in every:
        secs = [c["seconds"].get(r) for c in cases if c["seconds"].get(r) is not None]
        if secs:
            cost[f"filler map (reader {r})"] = secs
    for name, rows in baseline_outputs(d, "sample").items():
        secs = [v["seconds"] for v in rows.values() if v.get("seconds") is not None]
        if secs:
            cost[("LLM judge" if name.startswith("judge-") else "Selective Context") + f" ({name.split('-', 1)[1]})"] = secs
    if cost:
        tc = pd.DataFrame({k: {"responses": len(v), "seconds per response (median)": float(np.median(v)),
                               "seconds per response (mean)": float(np.mean(v))} for k, v in cost.items()}).T
        tc["responses"] = tc["responses"].astype(int)
        tc.index.name = "method"
        doc += ["", "## Cost on these responses", "",
                "Wall-clock seconds per response on the run's GPU (the map: its whole score of the answer).", "", md(tc)]
    return "\n".join(doc) + "\n"
