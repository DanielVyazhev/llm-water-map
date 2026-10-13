"""Join scored responses, entropy signals and item knobs into analysis tables."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from drybench.abstain import abstains
from drybench.attribution import attribute_spans, attribution_shares
from drybench.entropy import item_answer_entropy
from drybench.profile import profiles
from drybench.types import Item
from drybench.filler import completeness, misleading_excuse


def _filler(res: dict | None) -> float:
    """Filler share of a scored result: uninformative + duplicates (all removable text for old scores)."""
    if not res:
        return np.nan
    w = res.get("W_filler")
    return res.get("W", np.nan) if w is None or w != w else w


def build_tables(items: Iterable[Item], scored: Iterable[dict], signals: Iterable[dict],
                 h_key: str = "H_first", reduce: str = "mean") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (item_df, span_df). Rows are keyed by (model, variant, item_id).

    Spans labelled misleading are checked again with misleading_excuse (headers, other
    shipments, true unrequested statements become uninformative), so scores made before that rule
    and scores made after it give the same tables. Items with a missing fact get `missing` = True,
    `abstains` (the answer says the fact is not given: drybench.abstain) and `p_missing` (the reader's
    p of the accepted reply to the missing fact's question); the filler-map summaries leave them out."""
    from drybench.reader import is_missing

    items = {it.item_id: it for it in items}
    miss_j = {k: next((j for j, p in enumerate(it.required_probes) if is_missing(p)), None) for k, it in items.items()}
    knobs = {k: it.knobs for k, it in items.items()}
    sig = {(s["model"], s.get("variant", "default"), s["item_id"]): s for s in signals}
    item_rows, span_rows = [], []
    for r in scored:
        key = (r["model"], r.get("variant", "default"), r["item_id"])
        s = sig.get(key, {})
        h_vals = s.get(h_key) or []
        w_filler = r.get("W_filler")
        if w_filler is None or w_filler != w_filler:  # older scores or non-chat readers
            w_filler = r["W"]
        mis = r.get("W_misleading")
        kinds = [sp.get("kind") or ("kept" if sp["kept"] else "removed") for sp in r["spans"]]
        for i, sp in enumerate(r["spans"]):
            if kinds[i] == "misleading":
                excuse = misleading_excuse(sp["text"], items[key[2]])
                if excuse and not (excuse == "true statement" and sp.get("attempt")):
                    kinds[i] = "uninformative"
        if mis is not None and mis == mis and r["tokens"]:  # recompute the split from the spans
            tok = {k: sum(sp["tokens"] for sp, kk in zip(r["spans"], kinds) if kk == k)
                   for k in ("uninformative", "duplicate", "misleading")}
            w_filler = (tok["uninformative"] + tok["duplicate"]) / r["tokens"]
            mis = tok["misleading"] / r["tokens"]
        ab = r.get("ablation") or {}
        kappa, overhead = r.get("kappa"), r.get("overhead", np.nan)
        if r.get("per_probe") and r.get("info_gold") and r["status"] != "reader_fail":
            # completeness from the stored gains (a contradicted fact counts 0, not negative), so
            # scores made before v0.7.4 give the same tables as new ones
            kappa = completeness(r["per_probe"], r.get("gold_per_probe") or r["info_gold"])
            tg = r.get("tokens_gold")
            overhead = r["tokens"] / (kappa * tg) if r["status"] == "ok" and kappa > 0 and tg else np.nan
        row = {"model": key[0], "variant": key[1], "item_id": key[2], **knobs[key[2]],
               "H_ans": item_answer_entropy(h_vals, reduce) if h_vals else np.nan,
               "W": w_filler, "W_removable": r["W"], "W_misleading": np.nan if mis is None else mis,
               "tokens": r["tokens"], "tokens_kept": r["tokens_kept"],
               "filler_tokens": w_filler * r["tokens"] if w_filler == w_filler else np.nan,
               "density": r.get("density_bits_per_token", np.nan),
               "tokens_gold": r.get("tokens_gold") or np.nan,
               "wording": r.get("wording", np.nan), "overhead": overhead,
               "facts": r.get("facts", np.nan),
               "W_ablation": _filler(ab or None), "W_removable_ablation": ab.get("W", np.nan),
               "kappa": kappa, "info": r["info"], "status": r["status"],
               # v1.2: the response was cut at the token limit (None: not recorded)
               "truncated": r.get("truncated")}
        j = miss_j[key[2]]
        row["missing"] = j is not None
        # one fragment per line, so a numbered answer keeps its numbers (joined by spaces, "2." would end up
        # inside the line before it)
        row["abstains"] = abstains("\n".join(sp["text"] for sp in r["spans"]), items[key[2]]) if j is not None else None
        lp = r.get("per_probe_lp") or []
        row["p_missing"] = float(np.exp(lp[j])) if j is not None and j < len(lp) else np.nan
        item_rows.append(row)
        h_span = s.get("h_span") or [np.nan] * len(r["spans"])
        if len(h_span) != len(r["spans"]):  # signals computed with another segmentation
            h_span = [np.nan] * len(r["spans"])
        for i, (sp, h) in enumerate(zip(r["spans"], h_span)):
            kind = kinds[i]
            span_rows.append({"model": key[0], "variant": key[1], "item_id": key[2], "span_idx": i,
                              "tokens": sp["tokens"], "kept": sp["kept"], "kind": kind,
                              "h_span": np.nan if h is None else h, "text": sp["text"], "status": r["status"]})
    return pd.DataFrame(item_rows), pd.DataFrame(span_rows)


def analyze(item_df: pd.DataFrame, span_df: pd.DataFrame, n_boot: int = 500,
            seed: int = 0) -> dict[str, pd.DataFrame]:
    """Profiles per model x variant and entropy-based attribution of filler spans (items without a
    missing fact: see build_tables)."""
    item_df, span_df = without_missing(item_df, span_df)
    df = item_df[item_df["status"] == "ok"].copy()
    df["group"] = df["model"] + " | " + df["variant"]
    prof = profiles(df.dropna(subset=["H_ans"]), by="group", n_boot=n_boot, seed=seed) \
        if df["H_ans"].notna().any() else pd.DataFrame()
    out = {"profiles": prof}
    if not span_df.empty and span_df["h_span"].notna().any() and df["H_ans"].notna().any():
        sp = span_df.merge(df[["model", "variant", "item_id", "H_ans"]], on=["model", "variant", "item_id"])
        sp["group"] = sp["model"] + " | " + sp["variant"]
        h_items = df[["group", "item_id", "H_ans"]].copy()
        h_items["H_ans_pct"] = h_items.groupby("group")["H_ans"].rank(pct=True)
        sp = sp.merge(h_items[["group", "item_id", "H_ans_pct"]], on=["group", "item_id"])
        sp["h_span_pct"] = sp.groupby("group")["h_span"].rank(pct=True)
        is_filler = sp["kind"].isin(["uninformative", "duplicate", "removed"])  # misleading = error, not filler
        sp = sp.dropna(subset=["h_span_pct", "H_ans_pct"]).copy()
        sp["filler_source"] = attribute_spans(sp) if not sp.empty else []
        filler = sp[is_filler.reindex(sp.index).fillna(False).astype(bool)].copy()
        if not filler.empty:
            out["attribution"] = attribution_shares(filler, by="group")
            # base rate: share of ALL response tokens in each cell; > 1 = filler is over-represented
            base = attribution_shares(sp, by="group").reindex(columns=out["attribution"].columns)
            out["attribution_enrichment"] = out["attribution"] / base
            out["filler_spans"] = filler
    return out


def group_summary(item_df: pd.DataFrame) -> pd.DataFrame:
    """Per model x variant: responses, share scored, mean W (filler), misleading share, kappa,
    density (bits per token), substance tokens and the geometric-mean token overhead (items without a
    missing fact)."""
    df = without_missing(item_df)[0].copy()
    df["group"] = df["model"] + " | " + df["variant"]
    ok = df[df["status"] == "ok"]
    g = df.groupby("group")
    out = pd.DataFrame({
        "n": g.size(),
        "share_ok": g["status"].apply(lambda s: float((s == "ok").mean())),
        "share_no_signal": g["status"].apply(lambda s: float((s == "no_signal").mean())),
        "tokens_mean": g["tokens"].mean(),
    })
    okg = ok.groupby("group")
    out["W_mean"] = okg["W"].mean()
    out["W_misleading_mean"] = okg["W_misleading"].mean()
    # completeness over every response the reader could judge: a response that delivers nothing has
    # no filler map (no_signal) but is a failed answer, not a missing value (v0.7.1 averaged ok only)
    judged = df[df["status"].isin(["ok", "no_signal"])].groupby("group")
    out["kappa_mean"] = judged["kappa"].mean()
    out["facts_mean"] = judged["facts"].mean()
    out["complete_share"] = judged["facts"].apply(
        lambda f: float((f.dropna() >= 1.0 - 1e-9).mean()) if f.notna().any() else float("nan"))
    out["density_mean"] = okg["density"].mean()
    out["substance_tokens_mean"] = okg["tokens_kept"].mean()
    # overhead = removable sentences |y|/|y*| x kept tokens per delivered bit |y*|/(kappa |y0|);
    # both on the same responses, so the geometric means multiply exactly
    dec = ok[np.isfinite(ok["overhead"]) & (ok["overhead"] > 0)].assign(
        removable=lambda d: d["tokens"] / d["tokens_kept"],
        wordiness=lambda d: d["tokens_kept"] / (d["kappa"] * d["tokens_gold"]))
    decg = dec.groupby("group")
    out["overhead_gmean"] = decg["overhead"].apply(gmean)
    out["removable_gmean"] = decg["removable"].apply(gmean)
    out["wordiness_gmean"] = decg["wordiness"].apply(gmean)
    out["W_ablation_mean"] = okg["W_ablation"].mean()
    return out


def without_missing(item_df: pd.DataFrame, span_df: pd.DataFrame | None = None):
    """The rows of items without a missing fact (the reader cannot read 'not stated': drybench.abstain);
    tables made before v0.8.2 have no `missing` column and are returned as they are."""
    if "missing" not in item_df:
        return item_df, span_df
    keep = item_df[~item_df["missing"].astype(bool)]
    if span_df is not None and not span_df.empty:
        span_df = span_df.merge(keep[["model", "variant", "item_id"]], on=["model", "variant", "item_id"])
    return keep, span_df


def gmean(x) -> float:
    """Geometric mean of the finite positive values (overhead factors multiply)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")
