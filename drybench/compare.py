"""Filler map, LLM judge, Selective Context and cheap verbosity metrics against the same truth, fragment
by fragment and response by response.

A case is one response split into fragments, with
  truth       per fragment 'needed' / 'filler' / 'wrong', or None (unlabelled, or either copy of a repeated
              fact may go: such fragments are left out of the fragment-level metrics),
  counted     which fragments the response-level shares count (all of a planted response; the labelled
              ones of a human-labelled response),
  true_share  the true removable share by characters: planted filler and planted wrong statements (a summary
              that repeats the facts is planted filler, as the guidelines' "later repeats are filler"), or
              the fragments labelled filler or wrong,
and, per method, a removable decision and/or a score per fragment (higher = more removable).
Truth sets: planted filler in synthetic reports (T1/R2 rows), planted filler in real texts (R3 rows),
Poddar et al.'s human span labels (R4 rows), and the human labels of the annotation sample
(drybench.annotate). Every method is scored on the same cases: those every method covers. fill_unscored puts
back the responses a reader's filler map has no map for because the reader reads none of the requested facts
from them: the method's verdict on such an answer is that it delivers nothing, so all of it is removable (the
results tables count it as a failed answer). Responses whose gold answer the reader cannot read stay out.

Response level: Spearman with the true removable share; the length-confound pair test (pairs whose true
share differs by ≥ 0.2 where the more removable response is the SHORTER one: how often the method ranks it
higher, ties apart); the partial Spearman correlation with the true share controlling for length.
Fragment level: F1 and Cohen's κ of 'removable' (filler or wrong) and the AUC of the method's score.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from drybench.guidelines import REMOVABLE

PLANTED_SETS = {"planted_synthetic": "filler_check", "planted_real": "filler_check_real"}
SETS = ("planted_synthetic", "planted_real", "human_poddar", "sample")
RESPONSE_ONLY = {"length": "length (characters)", "yap": "Yap excess (characters)", "stopwords": "stop-word share"}
COLUMNS = ["fragments: F1", "fragments: κ", "fragments: AUC", "responses: Spearman",
           "shorter but more removable: ranked right", "… tied", "partial Spearman given length"]


# ---------------------------------------------------------------- truth

def read_rows(path: Path) -> list[dict]:
    """rows.jsonl of a check; pilot_validation.py wraps each row as {"row": ..., "spans": ...}."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.append(r["row"] if isinstance(r.get("row"), dict) else r)
    return out


def planted_truth(labels: Sequence[str], span_facts: Sequence[Sequence[str]] | None = None) -> list[str | None]:
    """Fragment truth of a planted response: planted filler and wrong statements as planted; a fragment
    that states facts is needed when it is the only one stating one of them, and ambiguous (None) when
    every fact it states is stated elsewhere too (a summary and the sentences it repeats: either may go).
    Rows written before v1.0 carry no facts per fragment: then a response with a summary has its summary
    and fact sentences ambiguous."""
    out: list[str | None] = []
    summary = "filler:summary" in labels
    for i, lab in enumerate(labels):
        if span_facts is not None and span_facts[i]:
            mine = set(span_facts[i])
            others = set().union(*(set(f) for j, f in enumerate(span_facts) if j != i))
            out.append(None if mine <= others else "needed")
        elif span_facts is None and summary and lab in ("fact", "filler:summary"):
            out.append(None)
        elif lab == "fact":
            out.append("needed")
        elif lab.startswith("filler"):
            out.append("filler")
        else:
            out.append("wrong" if lab in ("wrong", "misleading") else None)
    return out


def planted_share(labels: Sequence[str], chars: Sequence[int]) -> float:
    """Characters of planted filler and wrong statements / all characters."""
    rem = np.array([lab.startswith("filler") or lab in ("wrong", "misleading") for lab in labels])
    ch = np.asarray(chars, float)
    return float(ch[rem].sum() / ch.sum()) if ch.sum() else np.nan


def human_truth(label: str) -> str | None:
    """Poddar et al.'s labels: the minimal answer is needed, everything else is extra (no 'wrong' label)."""
    if label == "MINANS":
        return "needed"
    return None if label == "NONE" else "filler"


# ---------------------------------------------------------------- cases

def _synthetic_documents(sub: Path) -> dict[str, tuple[str, str, str]]:
    """item_id -> (document, request, gold) of the simulated synthetic check, regenerated from its settings."""
    from drybench.items import generate_items

    path = sub / "filler_check" / "summary.json"
    if not path.exists():
        return {}
    cfg = json.loads(path.read_text()).get("config") or {}
    cells, n_items, seed = cfg.get("cells"), cfg.get("items"), cfg.get("seed", 2026)
    if not cells or not n_items:
        return {}
    items = generate_items(max(1, -(-int(n_items) // len(cells))), seed=seed, cells=cells)
    return {it.item_id: (it.document, it.request, it.gold_minimal) for it in items}


def planted_cases(sub: Path, set_name: str, items: dict | None = None) -> list[dict]:
    """Cases of a planted-filler check (one reader's rows; the simulated responses are the same for every
    reader). `items`: item_id -> Item of the run (planted_real)."""
    from drybench.baselines import length_chars

    rows = read_rows(sub / PLANTED_SETS[set_name] / "rows.jsonl")
    docs = _synthetic_documents(sub) if set_name == "planted_synthetic" and rows and "document" not in rows[0] else {}
    out = []
    for r in rows:
        it = (items or {}).get(r["item_id"])
        doc, req, _ = docs.get(r["item_id"], ("", "", ""))
        frags = list(r["span_texts"])
        chars = [len(f) for f in frags]
        facts = r.get("span_facts")
        label = (it.meta.get("doc_label") if it is not None else None) or (
            "Report" if set_name == "planted_synthetic" else "Source text")
        out.append({"id": f"{r['item_id']}|{r['condition']}", "group": r["item_id"],
                    "source": r.get("source", "synthetic"), "source_label": label,
                    "document": r.get("document") or (it.document if it is not None else doc),
                    "request": r.get("request") or (it.request if it is not None else req),
                    "response": r["text"], "fragments": frags, "chars": chars,
                    "truth": planted_truth(r["labels"], facts), "counted": [True] * len(frags),
                    "true_share": planted_share(r["labels"], chars),
                    "length": float(r.get("chars") or length_chars(r["text"])),
                    "yap": float(r["yap_excess"]), "stopwords": float(r["stopword_share"]),
                    "map": [not k for k in r["loo_kept"]] if r.get("loo_status") == "ok" else None,
                    "map_status": r.get("loo_status")})
    return out


def labelled_case(case: dict, truth: Sequence[str | None]) -> dict:
    """A case whose truth is a label per fragment (human labels): shares over the labelled fragments."""
    t = list(truth)
    ch = np.asarray(case["chars"], float)
    lab = np.array([x is not None for x in t])
    rem = np.array([x in REMOVABLE for x in t])
    tot = ch[lab].sum()
    return {**case, "truth": t, "counted": list(lab), "true_share": float(ch[lab & rem].sum() / tot) if tot else np.nan}


def human_cases(sub: Path, documents: dict[str, str] | None = None) -> list[dict]:
    """Cases of the human-label check (Poddar et al.). `documents` (case -> passage) fills in rows written
    before v1.0, which do not carry the passage and the response."""
    out = []
    for r in read_rows(sub / "human_check" / "rows.jsonl"):
        frags = list(r["texts"])
        base = {"id": r["case"], "group": r["case"], "source": f"poddar:{r.get('subset', '')}",
                "source_label": r.get("source_label") or "Source text",
                "document": r.get("document") or (documents or {}).get(r["case"], ""),
                "request": r["request"], "response": r.get("response") or " ".join(frags),
                "fragments": frags, "chars": [len(f) for f in frags], "has_minimal": "MINANS" in r["human"],
                "length": float(r["resp_chars"]), "yap": float(r["yap_excess"]),
                "stopwords": float(r["stopword_share"]),
                "map": [not k for k in r["kept"]] if r.get("status") == "ok" else None, "map_status": r.get("status")}
        out.append(labelled_case(base, [human_truth(x) for x in r["human"]]))
    return out


def baseline_path(root: Path, method: str, label: str, set_name: str) -> Path:
    """Where `drybench baselines` writes one method's output on one truth set."""
    return Path(root) / "baselines" / f"{method}-{label}" / f"{set_name}.jsonl"


def baseline_outputs(root: Path, set_name: str) -> dict[str, dict[str, dict]]:
    """'judge-<label>' / 'selective-<label>' -> case id -> output row, for every baseline run on this set."""
    out = {}
    base = root / "baselines"
    for d in sorted(p for p in base.glob("*") if p.is_dir()) if base.exists() else []:
        path = d / f"{set_name}.jsonl"
        if path.exists():
            out[d.name] = {r["id"]: r for r in read_rows(path)}
    return out


def response_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()[:16]


def fill_unscored(cases: list[dict], keys: Sequence[str]) -> list[dict]:
    """Copies of the cases where a filler map key ('map:<reader>') that has no map because its reader reads
    none of the requested facts from the response ('no_signal') marks every fragment removable: the answer
    delivers nothing requested. A response whose gold the reader cannot read ('reader_fail') stays without."""
    out = []
    for c in cases:
        m = dict(c.get("methods", {}))
        for k in keys:
            if k not in m and (c.get("status") or {}).get(k) == "no_signal":
                n = len(c["chars"])
                m[k] = {"removable": [True] * n, "score": [1.0] * n}
        out.append({**c, "methods": m})
    return out


def attach(cases: list[dict], key: str, values: dict | None, kind: str) -> None:
    """Put a method's output on each case: kind 'map' (removable bools per fragment), 'judge' (rows with
    labels and p = [needed, filler, wrong] per fragment), 'selective' (rows with self_info per fragment)."""
    for c in cases:
        c.setdefault("methods", {})
        v = (values or {}).get(c["id"])
        if v is None:
            continue
        if kind == "map":
            if len(v) == len(c["fragments"]):
                c["methods"][key] = {"removable": [bool(x) for x in v], "score": [float(x) for x in v]}
        elif v.get("hash") and c.get("response") and v["hash"] != response_hash(c["response"]):
            continue  # made for another response with this id (a rebuilt sample): not this one's
        elif kind == "judge":
            if len(v.get("p", [])) == len(c["fragments"]):
                p = np.asarray(v["p"], float)
                c["methods"][key] = {"removable": [lab in REMOVABLE for lab in v["labels"]],
                                     "score": list(p[:, 1] + p[:, 2]), "soft": True}
        elif kind == "selective":
            if len(v.get("self_info", [])) == len(c["fragments"]):
                c["methods"][key] = {"removable": None, "score": [-float(x) for x in v["self_info"]]}


def cases_for(set_name: str, readers: Sequence[tuple[str, Path]], root: Path, items: dict | None = None,
              documents: dict[str, str] | None = None) -> list[dict]:
    """A planted or Poddar truth set with every method attached: the filler map of each reader (label,
    folder; the first one defines the cases) and the judges and Selective Context runs in root/baselines."""
    if not readers:
        return []
    load = (lambda sub: human_cases(sub, documents)) if set_name == "human_poddar" else (
        lambda sub: planted_cases(sub, set_name, items))
    cases = load(readers[0][1])
    for label, sub in readers:
        got = cases if sub == readers[0][1] else load(sub)
        attach(cases, f"map:{label}", {c["id"]: c["map"] for c in got if c["map"] is not None}, "map")
        status = {c["id"]: c.get("map_status") for c in got}
        for c in cases:
            c.setdefault("status", {})[f"map:{label}"] = status.get(c["id"])
    for name, rows in baseline_outputs(root, set_name).items():
        attach(cases, name, rows, "judge" if name.startswith("judge-") else "selective")
    return cases


def method_list(cases: list[dict], readers: Sequence[str]) -> list[tuple[str, str]]:
    """Display names and keys of the methods the cases carry: the filler map per reader, the judges,
    Selective Context, then the response-level metrics."""
    keys = {k for c in cases for k in c.get("methods", {})}
    out = [(f"filler map (reader {r})", f"map:{r}") for r in readers if f"map:{r}" in keys]
    out += [(f"LLM judge ({k.split('-', 1)[1]})", k) for k in sorted(keys) if k.startswith("judge-")]
    out += [(f"Selective Context ({k.split('-', 1)[1]})", k) for k in sorted(keys) if k.startswith("selective-")]
    return out + [(v, k) for k, v in RESPONSE_ONLY.items()]


# ---------------------------------------------------------------- metrics

def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy()


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan
    return float(np.corrcoef(_rank(x[m]), _rank(y[m]))[0, 1])


def partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y controlling for z (ranks; residuals of a linear fit on z's ranks)."""
    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 5:
        return np.nan
    rx, ry, rz = _rank(x[m]), _rank(y[m]), _rank(z[m])
    Z = np.c_[np.ones(int(m.sum())), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def shorter_pairs(truth, length, value, groups=None, min_gap: float = 0.2) -> dict:
    """Pairs whose true share differs by ≥ min_gap where the more removable response is the shorter one: share
    the method ranks right (strictly higher value), share of ties, number of pairs."""
    w, L, v = (np.asarray(a, float) for a in (truth, length, value))
    sel = (w[:, None] - w[None, :] >= min_gap) & (L[:, None] < L[None, :])
    sel &= np.isfinite(v)[:, None] & np.isfinite(v)[None, :]
    if groups is not None:
        g = np.asarray(groups, dtype=object)
        sel &= g[:, None] == g[None, :]
    n = int(sel.sum())
    if not n:
        return {"right": np.nan, "ties": np.nan, "pairs": 0}
    d = v[:, None] - v[None, :]
    return {"right": float((d > 0)[sel].mean()), "ties": float((d == 0)[sel].mean()), "pairs": n}


def auc(pos, neg) -> float:
    """P(score of a positive > score of a negative), ties count 1/2."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    r = _rank(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def binary(t, p) -> dict:
    t, p = np.asarray(t, bool), np.asarray(p, bool)
    tp, fp, fn = float((t & p).sum()), float((~t & p).sum()), float((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else np.nan
    po = float((t == p).mean()) if len(t) else np.nan
    pe = t.mean() * p.mean() + (1 - t.mean()) * (1 - p.mean()) if len(t) else np.nan
    return {"precision": prec, "recall": rec, "f1": f1, "kappa": (po - pe) / (1 - pe) if pe == pe and pe < 1 else np.nan}


def _counted_chars(c: dict) -> np.ndarray:
    return np.asarray(c["chars"], float) * np.asarray(c["counted"], float)


def share(c: dict, values) -> float:
    """Share of the counted characters weighted by `values` (0/1 decisions, or probabilities)."""
    w = _counted_chars(c)
    return float((w * np.asarray(values, float)).sum() / w.sum()) if w.sum() else np.nan


def equal_budget(cases: list[dict], key: str, target: float) -> float:
    """Score threshold (higher = more removable) at which a method deletes `target` of the counted
    characters of all cases."""
    sc = np.concatenate([np.asarray(c["methods"][key]["score"], float) for c in cases])
    ch = np.concatenate([_counted_chars(c) for c in cases])
    if target <= 0 or not ch.sum():
        return np.inf
    order = np.argsort(-sc, kind="stable")
    cum = np.cumsum(ch[order]) / ch.sum()
    return float(sc[order][min(int(np.searchsorted(cum, target - 1e-12)), len(sc) - 1)])


def compare(cases: list[dict], methods: Sequence[tuple[str, str]], budget_from: str | None = None,
            within_group: bool = False, min_gap: float = 0.2, n_boot: int = 500,
            seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """One row per method, given as (display name, key): a key of case['methods'] or of RESPONSE_ONLY.
    Cases: those every fragment-level method covers, with a true share. A method with scores only
    (Selective Context) deletes, from the highest score down, as many characters as `budget_from` does
    (equal budget). within_group: pairs only between responses to the same item."""
    frag_keys = [k for _, k in methods if k not in RESPONSE_ONLY]
    with_truth = [c for c in cases if c["true_share"] == c["true_share"]]
    cs = [c for c in with_truth if all(k in c.get("methods", {}) for k in frag_keys)]
    info = {"responses": len(cs), "fragments": int(sum(sum(t is not None for t in c["truth"]) for c in cs)),
            "of": len(with_truth),
            "coverage": {name: sum(k in c.get("methods", {}) for c in with_truth) for name, k in methods
                         if k not in RESPONSE_ONLY}}
    if len(cs) < 3:
        return pd.DataFrame(), info
    truth = np.array([c["true_share"] for c in cs])
    length = np.array([c["length"] for c in cs], float)
    groups = [c["group"] for c in cs] if within_group else None
    target = None
    if budget_from in frag_keys:
        w = np.concatenate([_counted_chars(c) for c in cs])
        d = np.concatenate([np.asarray(c["methods"][budget_from]["removable"], float) for c in cs])
        target = float((w * d).sum() / w.sum()) if w.sum() else None
        info["budget"] = target
    t_mask = [np.array([t is not None for t in c["truth"]]) for c in cs]
    t_frag = np.concatenate([np.array([t in REMOVABLE for t in c["truth"]])[m] for c, m in zip(cs, t_mask)])
    rng = np.random.default_rng(seed)
    by_group: dict = {}
    for i, c in enumerate(cs):  # bootstrap by item: answers to one item are not independent
        by_group.setdefault(c["group"], []).append(i)
    blocks = list(by_group.values())
    boots = [np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]) for _ in range(n_boot)]
    rows, pairs = {}, 0
    for name, key in methods:
        row: dict = {}
        if key in RESPONSE_ONLY:
            value = np.array([c[key] for c in cs], float)
        else:
            meth = [c["methods"][key] for c in cs]
            scores = [np.asarray(m["score"], float) for m in meth]
            removable = [m["removable"] for m in meth]
            if removable[0] is None and target is not None:
                thr = equal_budget(cs, key, target)
                removable = [list(s >= thr) for s in scores]
            if removable[0] is not None:
                p_frag = np.concatenate([np.asarray(r, bool)[m] for r, m in zip(removable, t_mask)])
                b = binary(t_frag, p_frag)
                row.update({"fragments: F1": b["f1"], "fragments: κ": b["kappa"]})
            s_frag = np.concatenate([s[m] for s, m in zip(scores, t_mask)])
            row["fragments: AUC"] = auc(s_frag[t_frag], s_frag[~t_frag])
            if meth[0].get("soft"):  # a judge's expected removable share: no ties at zero
                value = np.array([share(c, s) for c, s in zip(cs, scores)])
            elif removable[0] is not None:
                value = np.array([share(c, r) for c, r in zip(cs, removable)])
            else:
                value = np.full(len(cs), np.nan)
        pr = shorter_pairs(truth, length, value, groups, min_gap)
        pairs = max(pairs, pr["pairs"])
        part = partial_spearman(value, truth, length)
        bs = np.array([partial_spearman(value[b], truth[b], length[b]) for b in boots]) if n_boot else np.array([])
        bs = bs[np.isfinite(bs)]
        if part != part:
            ps = "n/a"
        elif len(bs) > 10:
            ps = f"{part:+.2f} [{np.quantile(bs, 0.025):+.2f}, {np.quantile(bs, 0.975):+.2f}]"
        else:
            ps = f"{part:+.2f}"
        row.update({"responses: Spearman": spearman(value, truth), "shorter but more removable: ranked right": pr["right"],
                    "… tied": pr["ties"], "partial Spearman given length": ps})
        rows[name] = row
    info["pairs"] = pairs
    t = pd.DataFrame(rows).T
    t = t[[c for c in COLUMNS if c in t.columns]]
    if "… tied" in t and not (pd.to_numeric(t["… tied"], errors="coerce").fillna(0) > 0.005).any():
        t = t.drop(columns="… tied")
    t.index.name = "method"
    return t, info


# ---------------------------------------------------------------- helpers for the baselines

def locate(response: str, fragments: Sequence[str]) -> list[tuple[int, int]]:
    """(start, end) of each fragment in the response, searched in order; whitespace may differ (fragments
    rejoined with spaces). A fragment not found gets the position right after the previous one."""
    out, cur = [], 0
    for f in fragments:
        pos = response.find(f, cur)
        if pos >= 0:
            end = pos + len(f)
        else:
            pat = r"\s+".join(re.escape(w) for w in f.split())
            m = re.compile(pat).search(response, cur) if pat else None
            pos, end = (m.start(), m.end()) if m else (cur, cur)
        out.append((pos, end))
        cur = end
    return out
