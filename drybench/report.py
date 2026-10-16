"""Markdown tables for a run on real texts (SQuAD 2.0, MuSiQue, CUAD, Poddar et al.) or on the
synthetic reports, with the same definitions and columns for both.

    python -m drybench.report RUN_DIR        # writes RUN_DIR/tables_real.md (or tables_synthetic.md)

A real-text RUN_DIR is the folder scripts/run_real.py writes: items.jsonl, scored.jsonl,
signals.jsonl, filter_report.json, reader_check.json, filler_check/, filler_check_real/, human_check/
and, for a second reader, reader2_<label>/ with the same checks. A synthetic RUN_DIR is the folder of
the T4 notebook's main cell (items.jsonl, scored.jsonl, signals.jsonl, reader_check.json,
filler_check/, reader2_<label>/ from the second-reader cell); its tables keep the T1–T7 numbering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

FILLER = ("uninformative", "duplicate", "removed")
SUITES = {"squad2": "SQuAD 2.0", "musique": "MuSiQue", "cuad": "CUAD (contracts)",
          "poddar": "Poddar et al. (passages)"}
SUBSETS = {"narrqa": "NarrativeQA", "tweetqa": "TweetQA", "msmarco": "MS MARCO"}
DESIGNS = {
    "squad2": ["k=1", "k=2", "k=3", "k=4", "missing (k=1)", "missing (k=3)"],
    "musique": ["2-hop", "2-hop+explain", "3-hop", "3-hop+explain", "4-hop", "4-hop+explain", "missing (2-hop)"],
    "cuad": ["k=1", "k=2", "k=3", "k=1, long", "missing law (k=2)"],
    "poddar": list(SUBSETS.values()),
    "synthetic": ["k=1", "k=2-3", "k=4-6", "missing fact", "ambiguous", "2-hop", "2-hop+explain"],
}
PREFIX = "prefix ablation"  # the deletion search that stops at the first fragment that cannot go
HUMAN_LABELS ={"MINANS": "minimal answer", "ADDINFO": "additional information", "EXPLAIN": "explanation",
                "CONVERS": "conversational", "REDINFO": "redundant", "IRREL": "irrelevant", "NONE": "unlabelled"}


def md(df: pd.DataFrame, digits: int = 3) -> str:
    """Markdown table without extra dependencies."""
    df = df.copy()
    for c in df.columns:  # floats also in mixed (object) columns
        df[c] = df[c].map(lambda v: ("" if v != v else f"{v:.{digits}f}") if isinstance(v, float) else v)
    cell = lambda v: str(v).replace("|", "\\|")  # noqa: E731  (group labels contain "|")
    head = [cell(df.index.name or "")] + [cell(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([cell(idx)] + [cell(v) for v in row.tolist()]) + " |")
    return "\n".join(lines)


def ci(x, ratio: bool = False, n: int = 2000, seed: int = 0) -> str:
    """Mean with a bootstrap 95% CI (geometric mean of ratios when x are log-ratios)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return "n/a"
    b = np.random.default_rng(seed).choice(x, (n, len(x))).mean(axis=1)
    m, lo, hi = x.mean(), np.quantile(b, 0.025), np.quantile(b, 0.975)
    if ratio:
        return f"×{np.exp(m):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}]"
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def auc(pos, neg) -> float:
    """P(score of a positive > score of a negative), ties count 1/2."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cohen(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float((a == b).mean())
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan


def _flag(v) -> bool:
    return bool(v) and v == v  # NaN (a knob other suites do not have) is False


def design(r) -> str:
    src = r.get("source")
    if src == "cuad":
        if not _flag(r["answerable"]):
            return f"missing law (k={int(r['k'])})"
        return f"k={int(r['k'])}" + (", long" if _flag(r.get("long")) else "")
    if src == "poddar":
        return SUBSETS.get(r.get("subset"), str(r.get("subset")))
    if src == "squad2":
        return f"k={int(r['k'])}" if r["answerable"] else f"missing (k={int(r['k'])})"
    if src == "musique":
        if not r["answerable"]:
            return f"missing ({int(r['hops'])}-hop)"
        return f"{int(r['hops'])}-hop" + ("+explain" if r["explain"] else "")
    if not r["answerable"]:
        return "missing fact"
    if r.get("ambiguous"):
        return "ambiguous"
    if r["hops"] == 2:
        return "2-hop+explain" if r["explain"] else "2-hop"
    return "k=1" if r["k"] == 1 else ("k=2-3" if r["k"] <= 3 else "k=4-6")


def cell_design(key: str) -> str:
    """'squad2:4' -> 'missing (k=1)' from the cell definitions."""
    from drybench.cuad import CUAD_CELLS
    from drybench.poddar import PODDAR_CELLS
    from drybench.realdata import MUSIQUE_CELLS, SQUAD_CELLS

    src, _, idx = key.partition(":")
    cells = {"squad2": SQUAD_CELLS, "musique": MUSIQUE_CELLS, "cuad": CUAD_CELLS, "poddar": PODDAR_CELLS}.get(src)
    if cells is None or not idx.isdigit() or int(idx) >= len(cells):
        return key
    c = cells[int(idx)]
    row = {"source": src, "k": c.get("k", 1), "answerable": c.get("answerable", True),
           "hops": c.get("hops", 1), "explain": c.get("explain", False), "long": c.get("long", False),
           "subset": c.get("subset")}
    return f"{SUITES.get(src, src)}: {design(row)}"


def load_reader(run: Path, sub: Path, label: str, items) -> dict:
    from drybench.analysis import build_tables
    from drybench.types import read_jsonl

    signals = list(read_jsonl(run / "signals.jsonl")) if (run / "signals.jsonl").exists() else []
    it, sp = build_tables(items, read_jsonl(sub / "scored.jsonl"), signals)
    it["group"] = it["model"] + " | " + it["variant"]
    sp["group"] = sp["model"] + " | " + sp["variant"]
    src = {i.item_id: i.knobs.get("source", "synthetic") for i in items}
    it["source"] = it["item_id"].map(src)
    sp["source"] = sp["item_id"].map(src)
    it["design"] = it.apply(design, axis=1)
    rc = sub / "reader_check.json"
    return {"label": label, "items": it, "spans": sp,
            "reader_check": json.loads(rc.read_text()) if rc.exists() else None}


def _results(it: pd.DataFrame) -> pd.DataFrame:
    from drybench.analysis import group_summary

    g = group_summary(it)
    ok = it[it["status"] == "ok"]
    g["removable"] = ok.groupby("group")["W_removable"].mean()
    g["complete"] = g["complete_share"]  # over ok and no-signal responses: a failed answer is incomplete
    g["removable_complete"] = ok[ok["facts"] >= 1.0 - 1e-9].groupby("group")["W_removable"].mean()
    if "truncated" in it and it["truncated"].notna().any():  # v1.2: responses cut at the token limit
        g["cut"] = it.assign(cut=it["truncated"].astype(float)).groupby("group")["cut"].mean()
    cols = {"n": "n", "tokens_mean": "tokens", "removable": "removable", "W_mean": "filler",
            "W_misleading_mean": "wrong", "facts_mean": "facts", "complete": "complete answers",
            "removable_complete": "removable, complete answers", "kappa_mean": "κ",
            "overhead_gmean": "overhead", "removable_gmean": "×removable", "wordiness_gmean": "×wordiness",
            "cut": "cut at the token limit"}
    t = g[[c for c in cols if c in g]].rename(columns=cols)
    t.index.name = "model | prompt"
    return t


def _concise(it: pd.DataFrame) -> pd.DataFrame | None:
    vals = ["tokens", "W_removable", "W", "W_misleading", "facts", "overhead"]
    piv = it.pivot_table(index=["model", "item_id"], columns="variant", values=vals, aggfunc="first",
                         dropna=False)
    rows = {}
    for m, g in piv.groupby(level=0):
        if ("tokens", "concise") not in g or ("tokens", "default") not in g:
            continue
        d = lambda c: g[(c, "concise")].to_numpy(float) - g[(c, "default")].to_numpy(float)  # noqa: E731
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(g[("overhead", "concise")].to_numpy(float) / g[("overhead", "default")].to_numpy(float))
        rows[m] = {"Δtokens": ci(d("tokens")), "Δremovable": ci(d("W_removable")), "Δfiller": ci(d("W")),
                   "Δwrong": ci(d("W_misleading")), "Δfacts": ci(d("facts")), "overhead": ci(lr, ratio=True),
                   # removable shares and overhead need a filler map under both prompts (v1.2: shown)
                   "pairs with both maps": f"{int(np.isfinite(d('W_removable')).sum())} of {len(g)}"}
    if not rows:
        return None
    t = pd.DataFrame(rows).T
    t.index.name = "model"
    return t


def filler_view(r: dict) -> dict:
    """A reader's tables without the items that have a missing fact: the reader cannot read 'not stated'
    (drybench.abstain), so their filler maps are not reported; _missing_section reports them."""
    from drybench.analysis import without_missing

    items, spans = without_missing(r["items"], r["spans"])
    return {**r, "items": items, "spans": spans}


MISSING_NOTE = ("Items whose source does not give one of the requested facts; the right answer says so. Whether "
                "an answer does is judged by a lexical rule (an abstention phrase — 'not stated / specified / "
                "mentioned / provided', 'unknown', 'no information', 'cannot determine', … — about the asked fact; "
                "drybench.abstain), not by the reader: the reader's reading of 'not stated' does not separate "
                "answers that say it from answers that make a value up (AUC below, from the stored scores). These "
                "items are left out of every other table.")


def _missing_section(R: list[dict], suite: str, heads: dict) -> list[str]:
    """Missing-fact items: share of answers that say the fact is not given (lexical rule), and how well
    each reader's own reading of 'not stated' separates them."""
    it = R[0]["items"]
    it = it[(it["source"] == suite) & it.get("missing", pd.Series(False, index=it.index)).astype(bool)]
    if it.empty:
        return []
    ab = it.assign(abstains=it["abstains"].astype(float))
    if suite == "synthetic":  # one design, two cells: one question, or three with one of them missing
        ab = ab.assign(design=[f"missing fact, k={int(k)}" for k in ab["k"]])
    t = ab.pivot_table(index="group", columns="design", values="abstains", aggfunc="mean")
    order = sorted(set(ab["design"])) if suite == "synthetic" else DESIGNS[suite]
    t = t.reindex(columns=[d for d in order if d in t.columns])
    t["all"] = ab.groupby("group")["abstains"].mean()
    t["n"] = ab.groupby("group").size()
    t.index.name = "model | prompt"
    lines = []
    for r in R:
        x = r["items"]
        x = x[(x["source"] == suite) & x.get("missing", pd.Series(False, index=x.index)).astype(bool)]
        x = x[x["p_missing"].notna()]
        yes, no = x[x["abstains"].astype(bool)], x[~x["abstains"].astype(bool)]
        if len(yes) and len(no):
            lines.append(f"- reader {r['label']}: AUC {auc(yes['p_missing'], no['p_missing']):.2f}; read as 'not "
                         f"stated' (p ≥ 0.5): {(yes['p_missing'] >= 0.5).mean():.2f} of the answers that say it, "
                         f"{(no['p_missing'] >= 0.5).mean():.2f} of the others")
    head = heads["missing"].format(label=R[0]["label"])
    return (["", head, "", MISSING_NOTE, "", md(t)]
            + (["", "The reader's reading of 'not stated' on the same answers (its accepted reply to the missing "
                "fact's question):", ""] + lines if lines else []))


def span_origin(text: str, document: str) -> str:
    """'copied' when the span has at least 3 content words and 80% of them occur in one sentence of
    the document (copied or closely paraphrased), 'own words' with 3+ content words otherwise, 'short'
    below 3 content words."""
    from drybench.filler import _content_words, supported_by_document

    words = _content_words(str(text))
    if len(words) < 3:
        return "short"
    return "copied" if supported_by_document(words, document) else "own words"


def _entropy(sp: pd.DataFrame, docs: dict | None = None) -> pd.DataFrame:
    """AUC of span entropy for filler vs kept spans; with `docs` (item_id -> document) also within
    the spans copied from the document and within those in the model's own words."""
    rows = {}
    sp = sp[sp["kind"] != "misleading"]
    if "status" in sp:  # a response without a filler map has no filler: its spans are not 'kept' (v1.2)
        sp = sp[sp["status"] == "ok"]
    if docs is not None and not sp.empty:
        sp = sp.assign(origin=[span_origin(t, docs.get(i, "")) for t, i in zip(sp["text"], sp["item_id"])])
    for g, d in sp.groupby("group"):
        w = d["kind"].isin(FILLER)
        within = [auc(x.loc[x["kind"].isin(FILLER), "h_span"], x.loc[~x["kind"].isin(FILLER), "h_span"])
                  for _, x in d.groupby("item_id") if x["kind"].isin(FILLER).any() and (~x["kind"].isin(FILLER)).any()]
        rows[g] = {"filler spans": int(w.sum()), "AUC span entropy": auc(d.loc[w, "h_span"], d.loc[~w, "h_span"]),
                   "AUC within a response": float(np.nanmean(within)) if within else np.nan,
                   "AUC span length": auc(d.loc[w, "tokens"], d.loc[~w, "tokens"])}
        if "origin" in d:
            long_filler = w & (d["origin"] != "short")
            rows[g]["filler copied from the document"] = (
                float((d.loc[long_filler, "origin"] == "copied").mean()) if long_filler.any() else np.nan)
            for origin in ("copied", "own words"):
                x = d[d["origin"] == origin]
                xw = x["kind"].isin(FILLER)
                rows[g][f"AUC, {origin}"] = auc(x.loc[xw, "h_span"], x.loc[~xw, "h_span"])
    t = pd.DataFrame(rows).T
    if not t.empty:
        t["filler spans"] = t["filler spans"].astype(int)
    t.index.name = "model | prompt"
    return t


def _agreement(a: dict, b: dict, suite: str) -> list[str]:
    ia, ib = a["items"][a["items"]["source"] == suite], b["items"][b["items"]["source"] == suite]
    m = ia.merge(ib, on=["model", "variant", "item_id"], suffixes=("_1", "_2"))
    both = m[(m["status_1"] == "ok") & (m["status_2"] == "ok")]
    if both.empty:
        return []
    rows = {}
    for c, lab in (("W_removable", "removable"), ("W", "filler"), ("W_misleading", "wrong"), ("kappa", "κ"),
                   ("facts", "facts"), ("overhead", "overhead")):
        within = np.array([d[f"{c}_1"].corr(d[f"{c}_2"], method="spearman")
                           for _, d in both.groupby(["model", "variant"])], float)
        rows[lab] = {"Spearman, pooled": both[f"{c}_1"].corr(both[f"{c}_2"], method="spearman"),
                     "Spearman, within model × prompt": np.nanmean(within) if np.isfinite(within).any() else np.nan}
    # group rankings on the responses both readers scored (v1.2; before, each reader's own scored responses)
    keys = both[["model", "variant", "item_id"]]
    ca, cb = ia.merge(keys, on=["model", "variant", "item_id"]), ib.merge(keys, on=["model", "variant", "item_id"])
    ga, gb = _results(ca), _results(cb)
    ids = sorted(set(keys["item_id"]))
    rng = np.random.default_rng(0)
    metrics = [(c, lab) for c, lab in (("removable", "removable"), ("filler", "filler"), ("overhead", "overhead"),
                                       ("facts", "facts")) if c in ga and c in gb]
    draws: dict[str, list[float]] = {c: [] for c, _ in metrics}
    for _ in range(200):  # bootstrap over items (each drawn item with all its responses)
        pick = pd.DataFrame({"item_id": rng.choice(ids, len(ids))})
        xa, xb = _results(ca.merge(pick, on="item_id")), _results(cb.merge(pick, on="item_id"))
        for c, _ in metrics:
            if c in xa and c in xb:
                draws[c].append(xa[c].corr(xb[c], method="kendall"))
    for c, lab in metrics:
        tau = ga[c].corr(gb[c], method="kendall")
        bs = np.asarray(draws[c], float)
        bs = bs[np.isfinite(bs)]
        rows[lab]["Kendall τ, group ranking [95% CI]"] = (
            f"{tau:+.2f} [{np.quantile(bs, 0.025):+.2f}, {np.quantile(bs, 0.975):+.2f}]"
            if tau == tau and len(bs) > 10 else (f"{tau:+.2f}" if tau == tau else "n/a"))
    t = pd.DataFrame(rows).T
    t.index.name = f"{a['label']} vs {b['label']}"
    sa, sb = a["spans"][a["spans"]["source"] == suite], b["spans"][b["spans"]["source"] == suite]
    s = sa.merge(sb, on=["model", "variant", "item_id", "span_idx"], suffixes=("_1", "_2"))
    if "status_1" in s:  # spans of responses both readers mapped (v1.2: failed ones were counted as 'kept')
        s = s[(s["status_1"] == "ok") & (s["status_2"] == "ok")]
    po_r, k_r = cohen(s["kind_1"] != "kept", s["kind_2"] != "kept")
    po_w, k_w = cohen(s["kind_1"].isin(FILLER), s["kind_2"].isin(FILLER))
    po_m, k_m = cohen(s["kind_1"] == "misleading", s["kind_2"] == "misleading")
    return [f"{len(both)} responses scored by both readers.", "", md(t), "",
            f"Span level ({len(s)} spans): deleted or kept — same decision {100 * po_r:.1f}%, Cohen's κ = {k_r:.2f}; "
            f"filler or not — {100 * po_w:.1f}%, κ = {k_w:.2f}; "
            f"wrong statement or not — {100 * po_m:.1f}%, κ = {k_m:.2f}."]


def _kept_note(label: str, n: int, not_candidate: float) -> str:
    return (f"- {label}: of {n} planted filler fragments left in, {100 * not_candidate:.0f}% cost more than the "
            f"tolerance to delete even alone (without them the reader's answer probabilities move by more than τ); "
            f"{100 * (1 - not_candidate):.0f}% could go alone but not together with the fragments deleted before them.")


def _filler_check_values(sub: Path) -> dict | None:
    """The check's values per setting, recomputed from its stored rows when there are any (v1.2.2: the error of
    the removable share against the planted truth), else as its summary.json has them."""
    path, rows_path = sub / "filler_check" / "summary.json", sub / "filler_check" / "rows.jsonl"
    if not path.exists() and not rows_path.exists():
        return None
    ws = json.loads(path.read_text()) if path.exists() else {}
    if rows_path.exists():
        from drybench.compare import read_rows
        from drybench.simulate import filler_check_metrics

        rows = read_rows(rows_path)
        for key in ("loo", "loo_t2", "loo_prefix"):
            ok = [r for r in rows if r.get(f"{key}_status") == "ok" and r.get(f"{key}_W") == r.get(f"{key}_W")]
            if ok:
                ws.setdefault(key, {}).update({f"check_{k}": v for k, v in filler_check_metrics(ok, key).items()})
    return ws


def _filler_check_section(subs: list[tuple[str, Path]],
                         heading: str = "## R2. Filler map on simulated responses with known filler") -> list[str]:
    """R2 (T1 for the synthetic run): the filler map on simulated responses with planted filler, one
    row per reader × tolerance."""
    rows, types, kept_notes = {}, [], []
    for label, sub in subs:
        ws = _filler_check_values(sub)
        if ws is None:
            continue
        for key, tol in (("loo", "0.5 nats"), ("loo_t2", "2 nats"), ("loo_prefix", f"0.5 nats, {PREFIX}")):
            v = ws.get(key) or {}
            if "check_recall_pooled" not in v:
                continue
            lo, hi = v["check_recall_pooled_ci"]
            rows[f"{label}, τ = {tol}"] = {
                "responses (with filler)": f"{v['check_n_responses']} ({v['check_n_with_filler']})",
                "no fact lost": v["check_no_fact_lost"], "precision": v["check_precision_pooled"],
                "recall, planted fragments": f"{v['check_recall_pooled']:.3f} [{lo:.3f}, {hi:.3f}]",
                "recall per response with filler": v["check_recall_with_filler"],
                "repeats resolved": v["check_repeats_resolved"], "MAE of removable share": v["check_W_mae"],
                "mean error of removable share": v["check_W_bias"]}
            if key == "loo":
                types.append(f"{label}: " + ", ".join(f"{t} {c['removed']}/{c['planted']}"
                                                      for t, c in v["check_recall_by_type"].items()))
                if v.get("check_kept_filler"):
                    kept_notes.append(_kept_note(label, v["check_kept_filler"], v["check_kept_filler_not_candidate"]))
    if not rows:
        return []
    t = pd.DataFrame(rows).T
    t.index.name = "reader, tolerance"
    return ["", heading, "",
            "Synthetic reports; each response = the requested facts + planted filler of known type "
            "(none / 2 / 5 fragments, or 2 with one fact left out). Recall counts planted filler "
            "fragments (95% CI, bootstrap over items); a repeat is resolved when no fact is stated twice "
            "after deletion; error of the removable share = measured − true, by characters, true = the planted "
            "filler (a summary that repeats the facts included: of two statements of a fact the first stays; "
            "v1.2.2, up to v1.2.1 the shortest set of fragments stating every fact). "
            f"'{PREFIX}' = the deletion search stops at the first fragment that cannot go together with the "
            "ones before it (the default skips that fragment and continues).", "", md(t), "",
            "Deleted / planted by type (τ = 0.5): " + "; ".join(types) + "."] + \
        (["", "Planted filler left in (τ = 0.5):"] + kept_notes if kept_notes else [])


def _reader_check_table(checks: dict, index_name: str) -> str:
    t = pd.DataFrame({k: {"AUC present vs absent": v["auc_present_vs_absent"],
                          "gold unreadable": v["reader_fail_rate"],
                          "wrong-statement detection": v.get("misleading_check_active", np.nan),
                          "nats per fact": v["gain_present_median"]}
                      for k, v in checks.items()}).T
    t.index.name = index_name
    return md(t)


def _summary(sub: Path, name: str) -> dict | None:
    path = sub / name / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def _planted(sub: Path, missing: set) -> dict | None:
    """R3's summary over the items without a missing fact (recomputed from the stored rows)."""
    from drybench.realcheck import summarize_planted

    rows_path = sub / "filler_check_real" / "rows.jsonl"
    if not rows_path.exists():
        return _summary(sub, "filler_check_real")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    return summarize_planted([r for r in rows if r["item_id"] not in missing])


def _f(v, spec: str = ".3f") -> str:
    return "n/a" if v is None or v != v else format(v, spec)


def _planted_section(subs: list[tuple[str, Path]], missing: set = frozenset()) -> list[str]:
    """R3: the filler map on real-text responses with planted filler, per reader and suite (items without
    a missing fact)."""
    rows, types, notes = {}, [], []
    for label, sub in subs:
        ws = _planted(sub, missing)
        if not ws or "overall" not in ws:
            continue
        parts = [("all suites", ws["overall"], ws["wrong"], ws.get("n_responses"), ws.get("recall_with_unscored"))] + [
            (SUITES.get(src, src), v, v["wrong"], v.get("n_built"), v.get("recall_with_unscored"))
            for src, v in ws["by_source"].items()]
        pf = ws.get("overall_prefix") or {}
        if pf.get("n_responses"):
            parts.append((f"all suites, {PREFIX}", pf, {"deleted": float("nan"), "labelled_wrong": float("nan")},
                          None, None))
        for name, v, wr, built, rec_all in parts:
            lo, hi = v["recall_pooled_ci"]
            rows[f"{label} — {name}"] = {
                "responses scored (with filler)": f"{v['n_responses']} ({v['n_with_filler']})"
                + (f" of {built}" if built else ""),
                "no fact lost": v["no_fact_lost"], "precision": v["precision_pooled"],
                "recall, planted filler [95% CI]": f"{_f(v['recall_pooled'])} [{_f(lo)}, {_f(hi)}]",
                "… counting unscored responses": rec_all if rec_all is not None else float("nan"),
                "repeats resolved": v["repeats_resolved"],
                "wrong statements deleted": wr["deleted"], "… and labelled wrong": wr["labelled_wrong"],
                "MAE of removable share": v["W_mae"]}
        types.append(f"{label}: " + ", ".join(f"{t} {c['removed']}/{c['planted']}"
                                              for t, c in ws["overall"]["recall_by_type"].items()))
        t2, k, fw = ws.get("overall_t2") or {}, ws.get("kappa") or {}, ws.get("filler_labelled_wrong") or {}
        if ws["overall"].get("kept_filler"):
            notes.append(_kept_note(label, ws["overall"]["kept_filler"], ws["overall"]["kept_filler_not_candidate"]))
        by_pf = {SUITES.get(src, src): (v.get("prefix") or {}).get("recall_pooled") for src, v in ws["by_source"].items()}
        if any(x is not None for x in by_pf.values()):
            notes.append(f"- {label}, {PREFIX}: planted filler deleted by suite — "
                         + "; ".join(f"{k} {_f(x)}" for k, x in by_pf.items()) + ".")
        cp = ws.get("overall_complete") or {}
        if cp.get("n_responses"):
            notes.append(f"- {label}, responses whose facts the reader all reads (p ≥ 0.5): {cp['n_responses']}; "
                         f"no fact lost {_f(cp['no_fact_lost'])}, precision {_f(cp['precision_pooled'])}, planted filler "
                         f"deleted {_f(cp['recall_pooled'])}.")
        notes.append(f"- {label}: planted filler labelled as a wrong statement in {_f(fw.get('labelled_wrong'))} of "
                     f"{fw.get('filler_spans', 0)} fragments; κ {_f(k.get('complete_mean'), '.2f')} with every fact, "
                     f"{_f(k.get('incomplete_mean'), '.2f')} with one fact left out; at τ = 2 nats no fact lost in "
                     f"{_f(t2.get('no_fact_lost'))}, planted filler deleted {_f(t2.get('recall_pooled'))}.")
    if not rows:
        return []
    t = pd.DataFrame(rows).T
    t.index.name = "reader — suite"
    return ["", "## R3. Filler map on real-text responses with planted filler", "",
            "Responses built from each item's own document: the sentence that states each requested fact, plus "
            "planted filler — formulaic (preamble, restatement, hedge, generic remark, closing), real document "
            "sentences that answer nothing asked (elaboration: same paragraph as a fact; unrequested: another "
            "paragraph) or a summary repeating the facts. Conditions: none / 2 / 5 filler fragments, 2 with one "
            "fact left out, and 1 filler fragment + 1 wrong statement (another item's answer given as the answer). "
            "Truth is known per fragment: a fact is stated by the sentence chosen for it and by the summary, not by "
            "a sentence that only mentions its answer (v1.2). Recall counts planted filler fragments (95% CI, "
            "bootstrap over items); '… counting unscored responses' also counts the planted filler of the responses "
            "from which the reader reads no fact (no filler map) as not deleted. Error of the removable share: "
            "measured − true, by characters, true = the planted filler and wrong statements (v1.2.2; up to v1.2.1 "
            "the shortest set of fragments stating every fact: in a response with a summary, almost always the "
            "summary). "
            "Items with a missing fact, and "
            "items whose facts the document does not state in a sentence, are left out. "
            f"'{PREFIX}' = the deletion search stops at the first fragment that cannot go together with the ones "
            "before it (the default skips that fragment and continues).",
            "", md(t), "", "Deleted / planted by type (τ = 0.5 nats): " + "; ".join(types) + ".", ""] + notes


def _human_section(subs: list[tuple[str, Path]]) -> list[str]:
    """R4: deletions vs Poddar et al.'s human span labels."""
    rows, by_label, by_subset, taus, sources = {}, {}, {}, [], []
    for label, sub in subs:
        hs = _summary(sub, "human_check")
        if not hs or "fragments" not in (hs.get("overall") or {}):
            continue  # nothing scored, or no fragment carries a human label
        src = hs.get("scored_by_answer_source") or {}
        own = any(k != "reference" for k in src)  # rows of v1.1 and before: some facts read from MINANS
        variants = [("", hs["overall"])] + (
            [(", reference answer only", hs.get("overall_reference_answer") or {})] if own else []) + [
            (f", {PREFIX}", hs.get("overall_prefix") or {})]
        for suffix, o in variants:
            if "fragments" not in o:
                continue
            f, c = o["fragments"], o["characters"]
            rows[label + suffix] = {"responses scored": str(o["n_responses"]),
                                    "precision": f["precision"], "recall": f["recall"], "F1": f["f1"],
                                    "Cohen's κ": f["cohen_kappa"], "F1 by characters": c["f1"],
                                    "minimal answer kept": o["minimal_answer_kept"],
                                    "Spearman, share per response": o["share_vs_human"]["spearman"],
                                    "deleted share (mean)": o["deleted_share_mean"],
                                    "human share (mean)": o["human_share_mean"]}
        left = (hs.get("config") or {}).get("without_reference_answer")
        if own:
            sources.append(f"{label}: {hs['n_scored_ok']} of {hs['n_responses']} labelled responses scored ("
                           + ", ".join(f"{v} by the {k}" for k, v in src.items()) + ")")
        else:
            sources.append(f"{label}: {hs['n_scored_ok']} of the {hs['n_responses']} responses that state the reference "
                           f"answer scored" + (f"; {left} labelled responses without it are left out" if left else ""))
        kn = hs.get("kept_not_minimal")
        if kn:
            sources.append(f"{label}: of {kn['fragments']} kept fragments the annotators marked as not minimal, "
                           f"{100 * kn['not_candidate']:.0f}% cost more than the tolerance to delete alone, "
                           f"{100 * kn['blocked']:.0f}% were blocked in the queue")
        for h, v in hs["by_label"].items():
            by_label.setdefault(f"{HUMAN_LABELS.get(h, h)} ({h})", {})[label] = f"{v['deleted']:.2f} (n={v['fragments']})"
        for subset, v in hs["by_subset"].items():
            fr = v.get("fragments") or {}
            by_subset.setdefault(SUBSETS.get(subset, subset), {})[label] = (
                f"F1 {_f(fr.get('f1'), '.2f')}, κ {_f(fr.get('cohen_kappa'), '.2f')} (n={v['n_responses']})")
        tau = hs.get("model_ranking_kendall_tau")
        if tau:
            taus.append(f"{label}: Kendall τ = {tau['tau']:.2f} (p = {tau['p']:.3f}, {tau['models']} LLMs)")
    if not rows:
        return []
    order = [f"{HUMAN_LABELS[h]} ({h})" for h in HUMAN_LABELS]
    tl = pd.DataFrame(by_label).T
    tl = tl.reindex([o for o in order if o in tl.index] + [o for o in tl.index if o not in order])
    t, ts = pd.DataFrame(rows).T, pd.DataFrame(by_subset).T
    t.index.name, tl.index.name, ts.index.name = "reader", "human label: share deleted", "subset"
    out = ["", "## R4. Filler map vs human span labels (Poddar et al., 2025)", "",
           "Responses of 10 LLMs (GPT-3.5/4o-mini, Llama, Gemma, Mistral, Phi) to the passage questions of "
           "Poddar et al.'s data (NarrativeQA, TweetQA, MS MARCO; reference answer of at most 6 words), with "
           "human span labels. A fragment is 'not minimal' when most of its characters carry a label other than "
           "MINANS; precision and recall of deletion are measured against that (unlabelled fragments left out). "
           "The fact is the reference answer; only responses that state it are scored (annotators label minimal vs "
           "extra text whether or not the answer is right, so for another answer the labels and the filler map "
           "measure different things; up to v1.1 the reader was given the fragments marked MINANS to find such a "
           "response's own answer, so the labels entered the method). "
           f"'{PREFIX}' = the deletion search stops at the first fragment that cannot go together with "
           "the ones before it.", "", md(t), ""] + \
        ([f"- {x}." for x in sources] + [""] if sources else []) + [md(tl), "", md(ts)]
    if taus:
        out += ["", "Ranking of the 10 LLMs by mean deleted share vs by mean human non-minimal share: "
                + "; ".join(taus) + "."]
    return out


BASELINES_NOTE = (
    "Every method on the same responses: the filler map of each reader, the LLM judge and Selective Context when "
    "they were run (drybench baselines), and three cheap metrics. Fragment level, removable (filler or wrong "
    "statement) vs needed: F1, Cohen's κ, and the AUC of the method's score (the filler map: minus the fragment's "
    "leave-one-out cost, or its yes/no decision where no cost is stored; the judge's p(filler) + p(wrong); "
    "Selective Context: lower self-information = more removable). Response level, removable share by characters "
    "(95% CIs: bootstrap over items): pairs ranked right = over pairs whose true share differs by ≥ 0.2, the share "
    "where the method scores the response with more filler higher (ties count half; 0.5 = chance), over all pairs "
    "and over pairs of responses to the same item; Spearman with the true share; 'shorter but more removable' = "
    "such pairs where the more removable response is the shorter one, share the method ranks higher (ties apart; "
    "length: 0 by construction); partial Spearman with the true share given length. Selective Context deletes, least "
    "informative fragment first, as many characters as the first reader's filler map (equal budget); the judge's "
    "share is its expected removable share. Planted responses: the true share is the planted filler and wrong "
    "statements (a summary that repeats the facts is planted filler); a fact stated twice may go either way, so "
    "such fragments are left out of the fragment-level scores. First table: the responses every method scores; "
    "second: also the answers from which a reader reads none of the requested facts — the filler map's verdict "
    "on such an answer is that it delivers nothing, so all of it counts as removable (as in the results tables: "
    "a failed answer); answers whose gold the reader cannot read stay out.")


def comparison(set_name: str, subs: list[tuple[str, Path]], root: Path, items: dict | None = None,
               missing: set = frozenset(), subset=None, fill: bool = False) -> tuple[pd.DataFrame, dict]:
    """drybench.compare on one truth set: the filler map of every reader, the baselines in root/baselines;
    fill: a response from which a reader reads none of the requested facts counts as all removable for that
    reader's filler map (compare.fill_unscored); a response whose gold it cannot read stays out."""
    from drybench.compare import cases_for, compare, fill_unscored, method_list

    cases = cases_for(set_name, subs, root, items)
    cases = [c for c in cases if c["group"] not in missing and (subset is None or subset(c))]
    labels = [label for label, _ in subs]
    if fill:
        cases = fill_unscored(cases, [f"map:{label}" for label in labels])
    return compare(cases, method_list(cases, labels), budget_from=f"map:{labels[0]}")


def _coverage_line(info: dict) -> str:
    cov = "; ".join(f"{k} {v}" for k, v in info.get("coverage", {}).items())
    return (f"{info['responses']} of {info.get('of', info['responses'])} responses scored by every method "
            f"({cov}); {info['fragments']} fragments with a truth label; {info.get('pairs', 0)} pairs.")


def _comparison_block(title: str, t: pd.DataFrame, info: dict, filled: tuple | None = None) -> list[str]:
    if t.empty:
        return []
    out = ["", title, "", _coverage_line(info), "", md(t)]
    if filled is not None and not filled[0].empty and filled[1]["responses"] > info["responses"]:
        out += ["", f"With the answers that deliver nothing to a reader ({filled[1]['responses']} responses; all of "
                "such an answer counts as removable):", "", md(filled[0])]
    return out


def _baselines_section(subs: list[tuple[str, Path]], root: Path, items: dict, missing: set = frozenset()) -> list[str]:
    """R5: every method against the planted truth (R3 responses) and the human labels (R4 responses)."""
    out = []
    t, info = comparison("planted_real", subs, root, items, missing)
    out += _comparison_block("### Planted filler in real texts (the R3 responses)", t, info,
                             comparison("planted_real", subs, root, items, missing, fill=True))
    t, info = comparison("human_poddar", subs, root)
    out += _comparison_block("### Human span labels (the R4 responses, Poddar et al.)", t, info,
                             comparison("human_poddar", subs, root, fill=True))
    t2, info2 = comparison("human_poddar", subs, root, subset=lambda c: c.get("has_minimal"))
    if not t.empty and not t2.empty and info2["of"] < info["of"]:
        out += _comparison_block(
            f"Responses with a fragment labelled minimal answer ({info2['of']} of {info['of']}; in the others no "
            "fragment is mostly minimal answer — the answer, if any, sits inside a fragment that is mostly something "
            "else, and the filler map does not delete a fragment that carries the answer):", t2, info2)
    if not out:
        return []
    return ["", "## R5. Baselines on the same responses", "", BASELINES_NOTE] + out


def _cost_section(subs: list[tuple[str, Path]], root: Path, heading: str) -> list[str]:
    """Seconds per response: the filler map of each reader (all answers of the run), the judges and Selective
    Context (the responses of the checks), per GPU. drybench annotation-report compares them on the same
    responses (the annotation sample)."""
    from drybench.compare import read_rows

    rows = {}

    def add(name: str, recs: list[dict]) -> None:
        by_dev: dict = {}
        for r in recs:
            if r.get("seconds") is not None:
                by_dev.setdefault(r.get("device") or "GPU not recorded", []).append(r["seconds"])
        for dev, secs in sorted(by_dev.items()):
            rows[f"{name} — {dev}"] = {"responses timed": len(secs),
                                      "seconds per response (median)": float(np.median(secs)),
                                      "seconds per response (mean)": float(np.mean(secs))}

    for label, sub in subs:
        add(f"filler map, reader {label}", read_rows(sub / "scored.jsonl"))
    base = root / "baselines"
    for d in sorted(p for p in base.glob("*") if p.is_dir()) if base.exists() else []:
        kind, label = d.name.split("-", 1)
        add(f"{'LLM judge' if kind == 'judge' else 'Selective Context'} ({label})",
            [r for f in sorted(d.glob("*.jsonl")) for r in read_rows(f)])
    if not rows:
        return []
    t = pd.DataFrame(rows).T
    t["responses timed"] = t["responses timed"].astype(int)
    t.index.name = "method — GPU"
    return ["", heading, "",
            "Wall-clock seconds per response (logged from v1.0). The filler map: the whole score of a model's answer "
            "(deletion search, kinds of deleted fragments, the gold answer); its reader never reads the source text. "
            "The LLM judge: one pass per fragment over the source text, the request and the response. Selective "
            "Context: one pass over the request and the response. The map is timed on the run's answers, the "
            "baselines on the checks' responses; `drybench annotation-report` times all of them on the same "
            "responses.", "", md(t)]


def _cross_suite(R0: dict) -> list[str]:
    """R6: removable share and facts per model × suite, and how stable the model ranking is."""
    it = R0["items"][R0["items"]["status"] == "ok"]
    judged = R0["items"][R0["items"]["status"].isin(["ok", "no_signal"])]
    present = [s for s in SUITES if s in set(it["source"])]
    if len(present) < 2:
        return []
    out = ["", "## R6. Across suites", ""]
    for col, lab in (("W_removable", "Removable share"), ("facts", "Facts delivered")):
        t = (judged if col == "facts" else it).pivot_table(index="group", columns="source", values=col,
                                                            aggfunc="mean")
        t = t.reindex(columns=present).rename(columns=SUITES)
        t.index.name = "model | prompt"
        out += [f"{lab} (reader {R0['label']}):", "", md(t), ""]
    d = it[it["variant"] == "default"]
    t = d.pivot_table(index="model", columns="source", values="W_removable", aggfunc="mean").reindex(columns=present)
    if len(t) >= 3:
        taus = {}
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                taus[f"{SUITES[a]} vs {SUITES[b]}"] = t[a].corr(t[b], method="kendall")
        out += ["Kendall τ between suites of the model ranking by removable share (default prompt; "
                f"{len(t)} models): " + "; ".join(f"{k} {_f(v, '+.2f')}" for k, v in taus.items()) + "."]
    return out


RESULTS_NOTE = ("Items without a missing fact (those are reported under 'Missing-fact items'). "
                "removable = share of tokens the filler map deletes = filler (uninformative + repeated facts) + wrong "
                "statements; facts, κ and complete answers count every response the reader can judge, a response "
                "that delivers nothing included; complete answers = share of responses from which the reader reads "
                "every requested fact, and removable over those only — a failed answer makes almost everything "
                "removable; overhead = tokens per delivered bit vs the minimal correct answer (geometric mean) = "
                "×removable × ×wordiness.")
REAL_HEADS = {
    "results": "### Results per model × prompt — reader {label}",
    "concise": "### 'Be concise.' vs default, same items (concise − default, 95% CI) — reader {label}",
    "design": "### {what} by item design — reader {label}",
    "missing": "### Missing-fact items: does the answer say the fact is not given?",
    "entropy": "### Does the generating model's span entropy flag filler? — reader {label}",
    "agreement": "### Agreement between readers",
    "tolerance": "### Tolerance: removable share at 0.5 vs 2 nats",
    "examples": "### Examples of deleted spans (default prompt) — reader {label}",
}
REAL_ORDER = ("results", "concise", "design", "entropy", "agreement", "tolerance", "examples")
SYNTH_HEADS = {
    "results": "## T2. Results per model × prompt — reader {label}",
    "concise": "## T3. 'Be concise.' vs default, same items (concise − default, 95% CI) — reader {label}",
    "agreement": "## T4. Agreement between readers",
    "design": "## T5. {what} by item design — reader {label}",
    "missing": "## T5. Missing-fact items: does the answer say the fact is not given?",
    "entropy": "## T6. Does the generating model's span entropy flag filler? — reader {label}",
    "tolerance": "## T7. Tolerance: removable share at 0.5 vs 2 nats",
    "examples": "## Examples of deleted spans (default prompt) — reader {label}",
}
SYNTH_ORDER = ("results", "concise", "agreement", "design", "entropy", "tolerance", "examples")


def _suite_section(R: list[dict], suite: str, by_id: dict, heads: dict, order: Sequence[str]) -> list[str]:
    """Results, 'Be concise.', agreement, item design, entropy, tolerance and examples for one suite
    (real texts) or for the synthetic reports; every reader where the table has one. The filler-map
    tables leave out the items with a missing fact; _missing_section reports those."""
    from drybench.analysis import analyze

    full, R = R, [filler_view(r) for r in R]
    pick = lambda r, key: r[key][r[key]["source"] == suite]  # noqa: E731
    main_r = R[0]
    it, sp = pick(main_r, "items"), pick(main_r, "spans")
    head = lambda key, **kw: heads[key].format(label=main_r["label"], **kw)  # noqa: E731
    parts: dict[str, list[str]] = {}

    sec = ["", head("results"), "", RESULTS_NOTE, "", md(_results(it))]
    for r in R[1:]:
        sec += ["", f"Same, reader {r['label']}:", "", md(_results(pick(r, "items")))]
    parts["results"] = sec

    sec = []
    for r in R:
        t3 = _concise(pick(r, "items"))
        if t3 is not None:
            sec += (["", head("concise"), ""] if not sec else ["", f"Same, reader {r['label']}:", ""]) + [md(t3)]
    parts["concise"] = sec

    sec = []
    ok = it[it["status"] == "ok"]
    judged = it[it["status"].isin(["ok", "no_signal"])]  # facts: failed answers count as failed
    for c, lab in (("W_removable", "Removable share"), ("W", "Filler"), ("W_misleading", "Wrong statements"),
                   ("facts", "Facts delivered")):
        t = (judged if c == "facts" else ok).pivot_table(index="group", columns="design", values=c, aggfunc="mean")
        if t.empty:
            continue
        t = t.reindex(columns=[d for d in DESIGNS[suite] if d in t.columns])
        t.index.name = "model | prompt"
        sec += ["", head("design", what=lab), "", md(t)]
    if len(judged):  # e.g. a made-up value on a one-question missing-fact item: a failed answer, no filler map
        t = judged.assign(nothing=(judged["status"] == "no_signal").astype(float)).pivot_table(
            index="group", columns="design", values="nothing", aggfunc="mean")
        t = t.reindex(columns=[d for d in DESIGNS[suite] if d in t.columns])
        t.index.name = "model | prompt"
        sec += ["", head("design", what="Answers that deliver nothing"), "",
                "Share of answers from which the reader reads less than 1 nat (a failed answer): they have no "
                "filler map, so the removable, filler and wrong tables above leave them out; facts counts them as "
                "failed.", "", md(t)]
    parts["design"] = sec + _missing_section(full, suite, heads)

    sec = []
    te = _entropy(sp, {k: v.document for k, v in by_id.items()})
    if not te.empty and te["AUC span entropy"].notna().any():
        sec += ["", head("entropy"), "",
                "AUC of span entropy for filler vs kept spans (0.5 = no signal); span length as a baseline. "
                "'copied' = spans whose content words (3 or more, 80% of them) occur in one sentence of the "
                "document; 'own words' = the other spans with 3+ content words; the AUC is computed within each "
                "(filler vs kept), and 'filler copied from the document' is the copied share of such filler spans.",
                "", md(te)]
    an = analyze(it, sp, n_boot=500)
    if "attribution_enrichment" in an:
        enr = an["attribution_enrichment"].copy()
        enr.index.name = "model | prompt"
        sec += ["", "Filler tokens per entropy cell relative to the cell's share of all tokens "
                "(> 1: over-represented):", "", md(enr, 2)]
    prof = an.get("profiles")
    if prof is not None and not prof.empty:
        fmt = lambda p, s, f: "n/a" if p[s] != p[s] else f"{p[s]:{f}} [{p[s + '_lo']:{f}}, {p[s + '_hi']:{f}}]"  # noqa: E731
        pr = pd.DataFrame({g: {"β_ctrl (filler vs uncertainty, same design)": fmt(p, "beta_ctrl", "+.3f"),
                               "γ (substance tokens per fact)": fmt(p, "gamma", "+.2f"),
                               "γ_filler (filler tokens per fact)": fmt(p, "gamma_filler", "+.2f")}
                           for g, p in prof.iterrows()}).T
        pr.index.name = "model | prompt"
        sec += ["", "Profiles (item bootstrap 95% CI):", "", md(pr)]
    parts["entropy"] = sec

    agr = []
    for other in R[1:]:
        got = _agreement(R[0], other, suite)
        agr += (["", f"Main reader vs {other['label']}:", ""] if len(R) > 2 else []) + got if got else []
    parts["agreement"] = ["", head("agreement"), ""] + agr if agr else []

    rows = {}
    for r in R:
        x = pick(r, "items")
        for g, d in x[x["status"] == "ok"].groupby("group"):
            rows[f"{r['label']} — {g}"] = {"removable, 0.5 nats": d["W_removable"].mean(),
                                           "removable, 2 nats": d["W_removable_ablation"].mean()}
    parts["tolerance"] = []
    if rows:
        t7 = pd.DataFrame(rows).T
        t7.index.name = "reader — model | prompt"
        parts["tolerance"] = ["", head("tolerance"), "", md(t7)]

    dele = sp[(sp["kind"] != "kept") & (sp["variant"] == "default")]
    ex = []
    for kind in ("uninformative", "duplicate", "misleading"):
        d = dele[dele["kind"] == kind]
        for _, row in d.sample(min(4, len(d)), random_state=0).iterrows():
            req = " ".join(getattr(by_id.get(row["item_id"]), "request", "").split())
            ex.append(f"- **{kind}** [{row['model']}, {row['item_id']}] request: {req[:100]}"
                      f"{'…' if len(req) > 100 else ''} — span: «{' '.join(str(row['text']).split())[:180]}»")
    parts["examples"] = ["", head("examples"), ""] + ex if ex else []
    return [line for key in order for line in parts.get(key, [])]


def _readers(run: Path, items) -> tuple[list[tuple[str, Path]], list[dict]]:
    subs = [("main", run)] + [(p.name.replace("reader2_", ""), p) for p in sorted(run.glob("reader2_*"))
                              if (p / "scored.jsonl").exists()]
    return subs, [load_reader(run, sub, lab, items) for lab, sub in subs]


def real_tables(run: str | Path) -> str:
    from drybench import __version__
    from drybench.types import load_items

    run = Path(run)
    items = load_items(run / "items.jsonl")
    by_id = {i.item_id: i for i in items}
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    subs, R = _readers(run, items)
    doc = [f"# Filler in LLM answers on real texts — {run.name} (package v{__version__})", ""]
    if cfg:
        extra = cfg.get("readers2") or ({"second reader": cfg["reader2"]} if cfg.get("reader2") else {})
        doc.append(f"Reader: {cfg.get('reader')}; other readers: "
                   + (", ".join(f"{k} = {v}" for k, v in extra.items()) or "none") + "; models: "
                   + ", ".join(f"{k} = {v}" for k, v in (cfg.get("models") or {}).items()) + ".")

    fr = run / "filter_report.json"
    if fr.exists():
        rep = json.loads(fr.read_text())
        runs = (rep.get("earlier") or []) + [rep]  # the first filter and its top-ups (v1.2 keeps them all)

        def total(k: str, field: str) -> int:
            return int(sum(r.get("cells", {}).get(k, {}).get(field, 0) for r in runs))

        t0 = pd.DataFrame({cell_design(k): {"pool": max(r.get("cells", {}).get(k, {}).get("pool", 0) for r in runs),
                                            "checked": total(k, "checked"), "reader knew a fact": total(k, "known"),
                                            "gold unreadable": total(k, "unreadable"), "kept": v["kept"]}
                           for k, v in rep["cells"].items()}).T
        t0.index.name = "cell"
        lost = (not rep.get("earlier") and any(v.get("kept_before") for v in rep["cells"].values()))
        doc += ["", "## R0. Items after the leakage filter", "",
                f"An item is dropped when the reader answers one of its facts from an empty message with "
                f"p ≥ {rep['max_prior']} (it knows the fact), or reads a fact from the gold answer with "
                f"p < {rep['min_gold_p']}. Counts over the first filter and its top-ups."
                + (" This run's first counts were overwritten by a top-up (before v1.2): cells it did not top up "
                   "show 0 checked." if lost else ""), "", md(t0)]

    checks = {r["label"]: r["reader_check"] for r in R if r["reader_check"]}
    if checks:
        doc += ["", "## R1. Reader check on the real items", "", _reader_check_table(checks, "reader check (real items)")]

    from drybench.rescore import has_missing

    missing = {i.item_id for i in items if has_missing(i)}
    doc += _filler_check_section(subs)
    doc += _planted_section(subs, missing)
    doc += _human_section(subs)
    doc += _baselines_section(subs, run, by_id, missing)
    doc += _cross_suite(filler_view(R[0]))
    doc += _cost_section(subs, run, "## R7. Cost per response")
    for suite, name in SUITES.items():
        if (R[0]["items"]["source"] == suite).any():
            doc += ["", f"## {name}"] + _suite_section(R, suite, by_id, REAL_HEADS, REAL_ORDER)
    return "\n".join(doc) + "\n"


def synthetic_tables(run: str | Path) -> str:
    """The synthetic run's tables (T1–T7) with the definitions and columns of the real-text tables."""
    from drybench import __version__
    from drybench.types import load_items

    run = Path(run)
    items = load_items(run / "items.jsonl")
    by_id = {i.item_id: i for i in items}
    subs, R = _readers(run, items)
    names = {}
    for label, sub in subs:
        cfg = (_summary(sub, "filler_check") or {}).get("config") or {}
        names[label] = str(cfg.get("reader", label)).replace(" (hf)", "")
    doc = [f"# Filler in LLM answers on synthetic reports — {run.name} (package v{__version__})", "",
           "Readers: " + "; ".join(f"{k} = {v}" for k, v in names.items()) + "."]
    t1 = _filler_check_section(subs, heading="## T1. Filler map on simulated responses with known filler")
    doc += t1 if t1 else ["", "## T1. Filler map on simulated responses with known filler", "", "(no filler check)"]
    checks = {r["label"]: r["reader_check"] for r in R if r["reader_check"]}
    if checks:
        doc += ["", _reader_check_table(checks, "reader check")]
    t, info = comparison("planted_synthetic", subs, run)
    if not t.empty:
        doc += ["", "Baselines on the same simulated responses. " + BASELINES_NOTE] + _comparison_block(
            "", t, info, comparison("planted_synthetic", subs, run, fill=True))[2:]
    doc += _suite_section(R, "synthetic", by_id, SYNTH_HEADS, SYNTH_ORDER)
    doc += _cost_section(subs, run, "## T8. Cost per response")
    return "\n".join(doc) + "\n"


def main(argv: list[str] | None = None) -> None:
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty bootstrap slices in tiny groups
    argv = sys.argv[1:] if argv is None else argv
    quiet = "--quiet" in argv
    argv = [a for a in argv if a != "--quiet"]
    if len(argv) != 1:
        raise SystemExit("usage: python -m drybench.report RUN_DIR [--quiet]")
    run = Path(argv[0])
    from drybench.types import load_items

    real = any(i.knobs.get("source") in SUITES for i in load_items(run / "items.jsonl"))
    text = real_tables(run) if real else synthetic_tables(run)
    path = run / ("tables_real.md" if real else "tables_synthetic.md")
    path.write_text(text, encoding="utf-8")
    if not quiet:
        print(text)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
