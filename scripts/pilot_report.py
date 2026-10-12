"""Render the pilot section of README.md from results/pilot/summary.json (no hand-copied numbers).

  python scripts/pilot_report.py --summary results/pilot/summary.json --readme README.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

START, END = "<!-- PILOT:START -->", "<!-- PILOT:END -->"
MODE_LABELS = {"loo": "per-probe, τ = 0.5 nats (default)", "loo_t1": "per-probe, τ = 1 nat",
               "loo_t2": "per-probe, τ = 2 nats", "loo_global": "global budget, δ = 0.05"}


def f(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}"


def render(s: dict, fig_dir: str) -> str:
    cfg = s["config"]
    n_items = cfg["items"]
    lines = [
        f"Setup: {n_items} items (k ∈ {{1,2,3,4}} × d ∈ {{0,4}}), four simulated responses each "
        f"({s['n_responses']} total): *dense* (facts only), *light* (+2 filler spans), *heavy* (+5), "
        "*incomplete* (one fact omitted, +2 filler). Filler spans are sampled from nine types "
        "(preamble, restatement, elaboration, hedge, enumeration, unrequested fact, distractor, "
        "summary, closing), so the true minimal response and the true filler fraction W* are known. "
        "Reader: GPT-2 124M (ONNX), zero-shot prompt; search: leave-one-out.",
        "",
        "**Filler map vs cheap baselines** (all responses)",
        "",
        "| Metric | Spearman with W* | Pairs where the more removable response is shorter: ranked correctly |",
        "|---|---|---|",
    ]
    for name, c in s["metric_vs_true_filler"].items():
        p = s["length_confound_pairs"][name]
        lines.append(f"| {name} | {f(c['spearman'])} | {f(p['accuracy'])} ({p['pairs']} pairs) |")
    lines += [
        "",
        "**Constraint ablation and tolerance sweep**",
        "",
        "| Setting | Responses with no fact lost | Filler precision | Filler recall | MAE vs W* | Spearman | Preambles kept |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, label in MODE_LABELS.items():
        m = s.get(mode)
        if not m:
            continue
        lines.append(
            f"| {label} | {f(100 * m['safety_all_ok'], 1)}% | {f(m['filler_precision'])} | "
            f"{f(m['filler_recall'])} | {f(m['mae'], 3)} | {f(m['W_vs_true']['spearman'])} | "
            f"{f(100 * m['kept_share_by_label'].get('filler:preamble', float('nan')), 0)}% |")
    m0, gl = s["loo"], s["loo_global"]
    sweep = "; ".join(
        f"{lab}: {f(100 * (1 - s[k]['safety_all_ok']), 1)}% / {f(100 * s[k]['filler_recall'], 0)}%"
        for k, lab in (("loo", "0.5 nats"), ("loo_t1", "1 nat"), ("loo_t2", "2 nats")) if k in s)
    lines += [
        "",
        "Responses that lose a fact / share of filler found, by tolerance: " + sweep + ". With this weak "
        "reader a looser tolerance buys recall with safety: GPT-2 gives a stated answer only ~10⁻³–10⁻², "
        "so its facts carry few nats and its confidence moves with context (with 0.5 nats it keeps "
        f"{f(100 * m0['kept_share_by_label'].get('filler:preamble', float('nan')), 0)}% of preambles and "
        f"{f(100 * m0['kept_share_by_label'].get('filler:summary', float('nan')), 0)}% of summaries that repeat "
        "the facts, because an opener or a repetition makes it surer of the answer). One budget for all "
        f"facts loses a fact in {f(100 * (1 - gl['safety_all_ok']), 1)}% of responses (facts the reader "
        "could partly guess carry few nats). The Colab cell runs the same check with the actual reader.",
    ]
    lost = s.get("loo", {}).get("safety_all_ok", 1.0)
    if lost < 1.0:
        lines.append("")
        lines.append(f"The {f(100 * (1 - lost), 1)}% of per-probe cases that lose a fact are facts the "
                     "reader registers with less than τ = 0.5 nats; such probes are treated as "
                     "undelivered and cannot be protected, so reader quality bounds safety.")
    g = s.get("greedy")
    if g:
        lines += ["", f"Greedy search on the *heavy* responses (n={g['n']}): no fact lost in "
                      f"{f(100 * g['safety_all_ok'], 1)}%, filler F1 {f(g['filler_f1'])}, MAE {f(g['mae'], 3)}, "
                      f"{f(g['evals_per_response'], 0)} reader evaluations per response vs "
                      f"{f(s['loo']['evals_per_response'], 0)} for leave-one-out."]
    r, k = s["reader_gain_nats"], s["kappa"]
    lines += [
        "",
        f"**Reader.** Information gain of a probe when its fact is present: median {f(r['present_median'], 1)} nats; "
        f"when absent: median {f(r['absent_median'], 2)}; AUC present vs absent {f(r['auc_present_vs_absent'], 3)}.",
        "",
        f"**Completeness.** κ = {f(k['complete_mean'])} for complete responses and {f(k['incomplete_mean'])} "
        f"for incomplete ones (true coverage {f(k['incomplete_coverage_mean'])}).",
        "",
    ]
    h = s["h_span_by_label"]
    order = sorted(h, key=lambda lab: h[lab]["mean"])
    lo, hi = order[0], order[-1]
    lines += [
        "**Negative control for h_span.** Scored by GPT-2 (not the model that wrote the text), "
        f"formulaic chat filler is *not* low-entropy: the lowest mean span entropy is "
        f"`{lo}` ({f(h[lo]['mean'])} nats), the highest `{hi}` ({f(h[hi]['mean'])}); facts: "
        f"{f(h['fact']['mean'])}. A base model does not expect assistant boilerplate, so `h_span` "
        "must come from the generating model itself — which is how `drybench signals` computes it.",
        "",
        f"![Filler map vs true filler]({fig_dir}/fig1_W_vs_true.png)",
        f"![Metrics vs truth]({fig_dir}/fig2_metrics_vs_truth.png)",
        f"![Span entropy]({fig_dir}/fig3_span_entropy.png)",
        "",
        "Caveats: responses are template-built, and GPT-2 is a weak reader. The pilot checks that the "
        "score behaves as designed; real claims need instruct generators, a strong reader and the "
        "human labels in `docs/ANNOTATION.md`.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/pilot/summary.json")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--fig-dir", default="results/pilot")
    args = ap.parse_args()
    s = json.loads(Path(args.summary).read_text())
    body = f"{START}\n{render(s, args.fig_dir)}\n{END}"
    readme = Path(args.readme).read_text()
    if START in readme:
        pre, rest = readme.split(START, 1)
        readme = pre + body + rest.split(END, 1)[1]
    else:
        readme = readme.replace("PILOT_RESULTS_PLACEHOLDER", body)
    Path(args.readme).write_text(readme)
    print(body)


if __name__ == "__main__":
    main()
