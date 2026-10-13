"""Pilot: validate the filler map (score) on simulated responses with known filler.

Reader: GPT-2 (124M, ONNX) — the only LM available offline in the build environment.
Responses: drybench.simulate with ground-truth span labels, four conditions per item:
  dense       all facts, no filler               (length grows with k, W* = 0)
  light       all facts + 2 filler spans
  heavy       all facts + 5 filler spans
  incomplete  one fact omitted + 2 filler spans   (kappa check)

Outputs (in --out-dir): rows.jsonl, spans.csv, summary.json and three PNG figures.

  python scripts/pilot_validation.py --assets /path/to/assets --out-dir results/pilot
  python scripts/pilot_validation.py --reader-backend hf --reader Qwen/Qwen2.5-7B-Instruct \
      --out-dir results/pilot_qwen7b          # same validation with a strong reader (GPU)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the drybench next to this script
from drybench.backends.onnx_gpt2 import OnnxGPT2Backend  # noqa: E402
from drybench.baselines import length_chars, stopword_share, yap_excess_chars  # noqa: E402
from drybench.entropy import response_span_entropies  # noqa: E402
from drybench.items import evaluated_model_messages, generate_items  # noqa: E402
from drybench.reader import Reader, ReaderConfig  # noqa: E402
from drybench.forms import own_wording_probes  # noqa: E402
from drybench.segment import join_spans, segment  # noqa: E402
from drybench.simulate import (FILLER_TYPES, evaluate_filler_map, simulate, true_filler_fraction,  # noqa: E402
                               filler_check_metrics)
from drybench.filler import completeness, filler_map  # noqa: E402

CONDITIONS = {"dense": (0, 0), "light": (2, 0), "heavy": (5, 0), "incomplete": (2, 1)}
# name, search, constraint, delta (None = --delta), tolerance in nats. The first is the default.
SCORE_CONFIGS = [
    ("loo", "loo", "per_probe", None, 0.5),
    ("loo_t1", "loo", "per_probe", 0.0, 1.0),
    ("loo_t2", "loo", "per_probe", 0.0, 2.0),
    ("loo_global", "loo", "global", 0.05, 0.5),  # one budget for all facts (ablation)
    ("loo_prefix", "loo", "per_probe", None, 0.5),  # stop at the queue's first blocking span (v0.7.1, ablation)
]
PREFIX_CONFIGS = {"loo_prefix"}
MODE_LABELS = {"loo": "per-probe, τ = 0.5 nats (default)", "loo_t1": "per-probe, τ = 1 nat",
               "loo_t2": "per-probe, τ = 2 nats", "loo_global": "global budget",
               "loo_prefix": "per-probe, τ = 0.5 nats, queue prefix only (v0.7.1)",
               "greedy": "greedy, per-probe, τ = 0.5 nats (heavy only)"}
CELLS = [{"k": k, "d": d} for k in (1, 2, 3, 4) for d in (0, 4)]


def score_response(reader, backend, it, cond, n_filler, n_omit, delta, greedy):
    rng = random.Random(f"{it.item_id}-{cond}")
    probes = it.required_probes
    omit = [p.fact_id for p in rng.sample(probes, n_omit)] if n_omit and len(probes) > 1 else []
    kinds = rng.sample(FILLER_TYPES, n_filler)
    sim = simulate(it, kinds, omit=omit, rng=rng)
    # v1.1: as in dry_score, a fact the response states in other words than the gold is read in them too
    probes, added = own_wording_probes(reader, it.request, probes, join_spans(list(sim.spans)))
    lengths = [backend.count_tokens(s.text) for s in sim.spans]
    row = {"item_id": it.item_id, "condition": cond, "k": it.knobs.get("k"), "d": it.knobs.get("d"),
           "filler_types": kinds, "n_omitted": len(omit), "request": it.request, "text": sim.text,
           "span_texts": [s.text for s in sim.spans], "span_facts": [list(s.fact_ids) for s in sim.spans],
           "document": it.document,
           "omitted_idx": [i for i, p in enumerate(probes) if p.fact_id in omit],
           "coverage": len(sim.present_facts) / len(probes),
           "W_true": true_filler_fraction(sim, lengths),
           "segmenter_aligned": [(s.start, s.end) for s in segment(sim.text)]
                                == [(s.start, s.end) for s in sim.spans],
           "tokens": sum(lengths), "chars": length_chars(sim.text),
           "yap_excess": yap_excess_chars(sim.text, it.gold_minimal),
           "stopword_share": stopword_share(sim.text), "added_answers": added}
    gold_info = float(reader.information(it.request, [it.gold_minimal], probes)[0])
    configs = [(name, algo, cons, delta if d is None else d, tol) for name, algo, cons, d, tol in SCORE_CONFIGS]
    if greedy:
        configs.append(("greedy", "greedy", "per_probe", delta, 0.5))
    row["labels"] = [s.label for s in sim.spans]
    for mode, algo, constraint, dlt, tol in configs:
        res = filler_map(reader, it.request, sim.spans, probes, delta=dlt, mode=algo, abs_tol=tol,
                        constraint=constraint, length_fn=backend.count_tokens, loo_skip=mode not in PREFIX_CONFIGS)
        row[f"{mode}_kept"] = [s.kept for s in res.spans]
        if mode == "loo":
            row["loo_candidate"] = [s.candidate for s in res.spans]
            row["loo_cost"] = [s.loo_cost for s in res.spans]
        ev = evaluate_filler_map(sim, [s.kept for s in res.spans], lengths) if res.status == "ok" else {}
        row.update({f"{mode}_{k}": v for k, v in ev.items() if k != "W_true"})
        row.update({f"{mode}_W": res.W, f"{mode}_status": res.status, f"{mode}_evals": res.n_evals})
        if mode == "loo":
            row.update({"info": res.info, "info_gold": gold_info,
                        "kappa": completeness(res.per_probe, gold_info) if gold_info >= 1.0 else float("nan"),
                        "per_probe": res.per_probe, "kept": [s.kept for s in res.spans]})
    ctx = backend.format_chat(evaluated_model_messages(it))
    h = response_span_entropies(backend, ctx, sim.text, sim.spans)
    spans = [{"item_id": it.item_id, "condition": cond, "label": s.label, "tokens": l,
              "h_span": float(e), "kept_loo": k}
             for s, l, e, k in zip(sim.spans, lengths, h, row["kept"])]
    return row, spans


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return {"spearman": float("nan"), "pearson": float("nan"), "n": int(m.sum())}
    return {"spearman": float(spearmanr(x[m], y[m])[0]), "pearson": float(pearsonr(x[m], y[m])[0]),
            "n": int(m.sum())}


def _kept_by_label(df, mode):
    kept, total = {}, {}
    for labels, mask in zip(df["labels"], df[f"{mode}_kept"]):
        for lab, k in zip(labels, mask):
            total[lab] = total.get(lab, 0) + 1
            kept[lab] = kept.get(lab, 0) + bool(k)
    return {lab: kept[lab] / total[lab] for lab in sorted(total)}


def _pairwise(df, metric, min_gap=0.2):
    """Among pairs where the more removable response is SHORTER, how often does `metric` rank it higher?"""
    w, L, v = df["W_true"].to_numpy(), df["tokens"].to_numpy(), df[metric].to_numpy(float)
    hits = total = 0
    for i in range(len(df)):
        for j in range(len(df)):
            if w[i] - w[j] >= min_gap and L[i] < L[j] and not (np.isnan(v[i]) or np.isnan(v[j])):
                total += 1
                hits += v[i] > v[j]
    return {"accuracy": hits / total if total else float("nan"), "pairs": total}


def summarize(rows: pd.DataFrame, spans: pd.DataFrame, greedy: bool) -> dict:
    ok = rows[rows["loo_status"] == "ok"]
    out = {"n_responses": int(len(rows)), "n_scored_ok": int(len(ok)),
           "segmenter_alignment": float(rows["segmenter_aligned"].mean())}
    for mode in [c[0] for c in SCORE_CONFIGS] + (["greedy"] if greedy else []):
        m = rows[rows[f"{mode}_status"] == "ok"] if f"{mode}_status" in rows else rows.iloc[0:0]
        m = m.dropna(subset=[f"{mode}_W"])
        if m.empty:
            continue
        out[mode] = {
            "n": int(len(m)),
            "W_vs_true": _corr(m[f"{mode}_W"], m["W_true"]),
            "mae": float(np.mean(np.abs(m[f"{mode}_W"] - m["W_true"]))),
            "safety_mean": float(m[f"{mode}_safety"].mean()),
            "safety_all_ok": float((m[f"{mode}_safety"] == 1.0).mean()),
            "token_recall_mean": float(m[f"{mode}_token_recall"].mean()),
            "filler_precision": float(m[f"{mode}_filler_precision"].mean()),
            "filler_recall": float(m[f"{mode}_filler_recall"].mean()),
            "filler_f1": float(m[f"{mode}_filler_f1"].mean()),
            "evals_per_response": float(m[f"{mode}_evals"].mean()),
            "kept_share_by_label": _kept_by_label(m, mode),
            # recall over planted filler spans (pooled) and over responses that contain filler
            **{f"check_{k}": v for k, v in filler_check_metrics(m.to_dict("records"), mode).items()},
            "by_condition": {c: {"W": float(g[f"{mode}_W"].mean()), "W_true": float(g["W_true"].mean()),
                                 "f1": float(g[f"{mode}_filler_f1"].mean())}
                             for c, g in m.groupby("condition")},
        }
    metrics = {"removable share (filler map)": "loo_W", "length (tokens)": "tokens", "Yap excess (chars)": "yap_excess",
               "stop-word share": "stopword_share"}
    out["metric_vs_true_filler"] = {name: _corr(rows[col], rows["W_true"]) for name, col in metrics.items()}
    out["length_confound_pairs"] = {name: _pairwise(rows, col) for name, col in metrics.items()}
    comp = ok.dropna(subset=["kappa"])
    out["kappa"] = {"vs_coverage": _corr(comp["kappa"], comp["coverage"]),
                    "complete_mean": float(comp[comp["n_omitted"] == 0]["kappa"].mean()),
                    "incomplete_mean": float(comp[comp["n_omitted"] > 0]["kappa"].mean()),
                    "incomplete_coverage_mean": float(comp[comp["n_omitted"] > 0]["coverage"].mean())}
    present, absent = [], []
    for _, r in rows.iterrows():
        for i, g in enumerate(r["per_probe"]):
            (absent if i in r["omitted_idx"] else present).append(g)
    present, absent = np.array(present), np.array(absent)
    auc = float(np.mean(present[:, None] > absent[None, :])) if len(absent) else float("nan")
    out["reader_gain_nats"] = {"present_median": float(np.median(present)),
                               "present_min": float(present.min()),
                               "absent_median": float(np.median(absent)) if len(absent) else float("nan"),
                               "absent_max": float(absent.max()) if len(absent) else float("nan"),
                               "auc_present_vs_absent": auc,
                               "n_present": int(len(present)), "n_absent": int(len(absent))}
    sp = spans.dropna(subset=["h_span"])
    out["h_span_by_label"] = {lab: {"mean": float(g["h_span"].mean()),
                                    "ci95": float(1.96 * g["h_span"].std(ddof=1) / math.sqrt(len(g)))
                                    if len(g) > 1 else float("nan"), "n": int(len(g))}
                              for lab, g in sp.groupby("label")}
    return out


def plots(rows: pd.DataFrame, spans: pd.DataFrame, summary: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surf, blue, orange = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2, "figure.facecolor": surf,
                         "axes.facecolor": surf, "savefig.facecolor": surf})

    # 1. removable share vs true filler share
    ok = rows[rows["loo_status"] == "ok"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=ink2, lw=1, ls="--", zorder=1)
    ax.scatter(ok["W_true"], ok["loo_W"], s=36, color=blue, edgecolor=surf, linewidth=1.5, zorder=2)
    n_origin = int(((ok["W_true"] == 0) & (ok["loo_W"] == 0)).sum())
    ax.annotate(f"{n_origin} dense responses\n(no filler, W = 0)", xy=(0, 0), xytext=(0.08, 0.12),
                color=ink2, fontsize=9, arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8))
    c = summary["loo"]["W_vs_true"]
    ax.set(xlim=(-0.03, 1), ylim=(-0.03, 1), xlabel="True filler share W*", ylabel="Removable share W, filler map (GPT-2 reader)")
    ax.set_title(f"The filler map recovers injected filler (Spearman {c['spearman']:.2f}, n={c['n']})", color=ink, fontsize=10.5)
    ax.grid(color=grid, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig1_W_vs_true.png", dpi=160)
    plt.close(fig)

    # 2. metric agreement with true filler, and length-confound pairs
    names = list(summary["metric_vs_true_filler"])
    rho = [summary["metric_vs_true_filler"][n]["spearman"] for n in names]
    acc = [summary["length_confound_pairs"][n]["accuracy"] for n in names]
    order = np.argsort(rho)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    for ax, vals, title, xlim in [
        (axes[0], rho, "Spearman with the true filler share W*", (0, 1)),
        (axes[1], acc, "Pairs where the more removable response is shorter:\nshare ranked correctly", (0, 1)),
    ]:
        y = np.arange(len(names))
        v = np.array(vals)[order]
        ax.barh(y, v, color=blue, height=0.55)
        for yi, vi in zip(y, v):
            ax.text(vi + (0.02 if vi >= 0 else -0.02), yi, f"{vi:.2f}", va="center",
                    ha="left" if vi >= 0 else "right", color=ink, fontsize=9)
        ax.axvline(0, color=ink2, lw=0.8)
        ax.set_xlim(*xlim)
        ax.set_title(title, color=ink, fontsize=10)
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(np.arange(len(names)), [names[i] for i in order])
    fig.tight_layout()
    fig.savefig(out / "fig2_metrics_vs_truth.png", dpi=160)
    plt.close(fig)

    # 3. span entropy by label
    tab = summary["h_span_by_label"]
    labs = sorted(tab, key=lambda l: tab[l]["mean"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, lab in enumerate(labs):
        is_fact = lab == "fact"
        ax.errorbar(tab[lab]["mean"], i, xerr=tab[lab]["ci95"], fmt="o", ms=8,
                    color=orange if is_fact else blue, ecolor=ink2, elinewidth=1, capsize=3)
    ax.set_yticks(range(len(labs)), [l.replace("filler:", "") + f"  (n={tab[l]['n']})" for l in labs])
    ax.set_xlabel("Mean token entropy of span, nats (GPT-2, teacher-forced)")
    ax.set_title("Negative control: GPT-2 is not the generator\n(orange = fact, blue = filler spans)", color=ink, fontsize=10.5)
    ax.grid(axis="x", color=grid, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig3_span_entropy.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=None, help="GPT-2 ONNX assets dir (reader-backend onnx-gpt2)")
    ap.add_argument("--reader-backend", default="onnx-gpt2", choices=["onnx-gpt2", "hf", "mock"],
                    help="mock: a smoke test without a model")
    ap.add_argument("--reader", default=None, help="HF model id/path when --reader-backend hf")
    ap.add_argument("--reader-dtype", default=None, help="hf: bf16 / fp16 / fp32 (default: bf16 if "
                    "the GPU supports it, else fp16; CPU: auto)")
    ap.add_argument("--out-dir", default="results/pilot")
    ap.add_argument("--n-per-cell", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--delta", type=float, default=0.0, help="relative tolerance of the default setting")
    ap.add_argument("--greedy-conditions", default="heavy", help="comma list, or 'none'")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--max-items", type=int, default=None, help="limit items (smoke tests)")
    ap.add_argument("--quiet", action="store_true", help="do not print the summary JSON")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.reader_backend == "hf":
        from drybench.backends.hf import HFBackend

        import torch

        dtype = args.reader_dtype or ("auto" if not torch.cuda.is_available() else
                                      "bf16" if torch.cuda.is_bf16_supported() else "fp16")
        backend = HFBackend(args.reader, dtype=dtype)
        reader = Reader.for_backend(backend)  # chat reader for instruct models
        reader_desc = f"{args.reader} (hf)"
    elif args.reader_backend == "mock":
        from drybench.backends.mock import MockBackend

        backend = MockBackend()
        reader = Reader(backend, ReaderConfig())
        reader_desc = "mock"
    else:
        if not args.assets:
            ap.error("--assets is required for the onnx-gpt2 reader")
        backend = OnnxGPT2Backend.from_dir(args.assets, max_batch_tokens=2048)
        reader = Reader(backend, ReaderConfig())  # zero-shot plain reader (as good as few-shot, 2x faster)
        reader_desc = "gpt2-124M (onnx), zero-shot plain template"
    items = generate_items(args.n_per_cell, seed=args.seed, cells=CELLS)[: args.max_items]
    greedy_conds = set() if args.greedy_conditions == "none" else set(args.greedy_conditions.split(","))

    rows_path = out / "rows.jsonl"
    done = set()
    rows, span_rows = [], []
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            rows.append(r["row"])
            span_rows += r["spans"]
            done.add((r["row"]["item_id"], r["row"]["condition"]))
    t0 = time.time()
    todo = [(it, c) for it in items for c in CONDITIONS if (it.item_id, c) not in done]
    for n, (it, cond) in enumerate(todo, 1):
        n_filler, n_omit = CONDITIONS[cond]
        row, sp = score_response(reader, backend, it, cond, n_filler, n_omit, args.delta,
                                 greedy=cond in greedy_conds)
        rows.append(row)
        span_rows += sp
        with rows_path.open("a") as f:
            f.write(json.dumps({"row": row, "spans": sp}) + "\n")
        if n % 10 == 0 or n == len(todo):
            print(f"[{n}/{len(todo)}] {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    rows_df, spans_df = pd.DataFrame(rows), pd.DataFrame(span_rows)
    spans_df.to_csv(out / "spans.csv", index=False)
    summary = summarize(rows_df, spans_df, greedy=bool(greedy_conds))
    summary["config"] = {"reader": reader_desc, "delta": args.delta,
                         "items": len(items), "cells": CELLS, "conditions": CONDITIONS,
                         "greedy_conditions": sorted(greedy_conds), "seed": args.seed}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if not args.no_plots:
        plots(rows_df, spans_df, summary, out)
    if not args.quiet:
        print(json.dumps({k: summary[k] for k in ("n_responses", "loo", "metric_vs_true_filler", "kappa")},
                         indent=1)[:3000])


if __name__ == "__main__":
    main()
