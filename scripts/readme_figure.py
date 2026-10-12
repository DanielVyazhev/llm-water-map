#!/usr/bin/env python3
"""The README figure: how often the filler map and the cheap metrics tell which of two responses has more filler,
per dataset, on the planted-filler checks of the development runs.

    python scripts/readme_figure.py          # → docs/img/filler_vs_baselines.png, prints the numbers

Pairwise accuracy: over all pairs of responses of one dataset whose true filler shares differ by at least MIN_GAP,
the share of pairs where the method gives the response with more filler the higher score (ties count half). 50% is
chance; with a two-class truth this is ROC-AUC. Responses: those every method scores, as in the baselines tables
(T1, R5). Also printed: the filler map's ROC-AUC per fragment (filler or wrong vs needed) and the same numbers on
the human labels of Poddar et al.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MIN_GAP = 0.2
SUITES = [("squad2", "SQuAD 2.0"), ("musique", "MuSiQue"), ("cuad", "CUAD"), ("poddar", "Poddar et al.")]
METHODS = [("map", "filler map (reader Qwen2.5-3B)", "#1f6fd1"),
           ("length", "length", "#5f5f5f"),
           ("yap", "Yap excess", "#a3a3a3"),
           ("stopwords", "stop-word share", "#d6d6d6")]


def pair_accuracy(truth, value, min_gap: float = MIN_GAP) -> tuple[float, int]:
    """Share of pairs (true shares differ by ≥ min_gap) where `value` is higher for the response with more filler."""
    t, v = np.asarray(truth, float), np.asarray(value, float)
    more = (t[:, None] - t[None, :]) >= min_gap - 1e-12
    n = int(more.sum())
    if not n:
        return float("nan"), 0
    d = v[:, None] - v[None, :]
    return float(((d > 0) & more).sum() + 0.5 * ((d == 0) & more).sum()) / n, n


def dataset(cases: list[dict], labels: list[str]) -> dict:
    from drybench.compare import compare, method_list, share

    maps = [f"map:{label}" for label in labels]
    cs = [c for c in cases if c["true_share"] == c["true_share"] and all(k in c.get("methods", {}) for k in maps)]
    truth = [c["true_share"] for c in cs]
    values = {"map": [share(c, c["methods"][maps[0]]["removable"]) for c in cs]}
    values.update({k: [c[k] for c in cs] for k, _, _ in METHODS[1:]})
    acc = {k: pair_accuracy(truth, v) for k, v in values.items()}
    t, _ = compare(cases, method_list(cases, labels), budget_from=maps[0], n_boot=0)
    return {"responses": len(cs), "pairs": acc["map"][1], "acc": {k: a for k, (a, _) in acc.items()},
            "fragment_auc": float(t.loc[f"filler map (reader {labels[0]})", "fragments: AUC"])}


def numbers(root: Path) -> tuple[list[tuple[str, dict]], dict]:
    from drybench.compare import cases_for
    from drybench.report import _readers
    from drybench.rescore import has_missing
    from drybench.types import load_items

    syn = root / "synthetic"
    subs, _ = _readers(syn, load_items(syn / "items.jsonl"))
    rows = [("Synthetic reports", dataset(cases_for("planted_synthetic", subs, syn), [l for l, _ in subs]))]
    real = root / "real"
    items = load_items(real / "items.jsonl")
    subs, _ = _readers(real, items)
    labels = [label for label, _ in subs]
    missing = {i.item_id for i in items if has_missing(i)}
    planted = [c for c in cases_for("planted_real", subs, real, {i.item_id: i for i in items})
               if c["group"] not in missing]
    for key, name in SUITES:
        rows.append((name, dataset([c for c in planted if c["source"] == key], labels)))
    human = dataset(cases_for("human_poddar", subs, real), labels)
    return rows, human


def draw(rows: list[tuple[str, dict]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surf = "#161616", "#555555", "#e4e4e4", "#ffffff"
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": ink2, "xtick.color": ink, "ytick.color": ink2,
                         "figure.facecolor": surf, "axes.facecolor": surf, "savefig.facecolor": surf})
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(rows))
    w = 0.2
    for j, (key, label, color) in enumerate(METHODS):
        vals = [100 * r["acc"][key] for _, r in rows]
        bars = ax.bar(x + (j - 1.5) * w, vals, w * 0.92, color=color, label=label,
                      edgecolor=ink2 if color == "#d6d6d6" else color, linewidth=0.6, zorder=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=9,
                    color=ink if j == 0 else ink2, fontweight="bold" if j == 0 else "normal", zorder=4,
                    bbox=dict(facecolor=surf, edgecolor="none", pad=0.5))
    ax.axhline(50, color=ink2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(1.008, 50, "coin\nflip", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9.5,
            color=ink2, clip_on=False)
    ax.axvline(0.5, color=grid, lw=1.2)
    ax.set_xticks(x, [f"{name}\n{r['responses']} responses" for name, r in rows])
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("Pairs ranked correctly, %")
    ax.text(0, 106, "synthetic", ha="center", va="bottom", fontsize=10, color=ink2)
    ax.text(2.5, 106, "real texts", ha="center", va="bottom", fontsize=10, color=ink2)
    ax.set_title("Of two responses, which has more filler? How often each metric is right", color=ink,
                 fontsize=13.5, pad=20)
    ax.grid(axis="y", color=grid, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore")
    rows, human = numbers(REPO / "results" / "dev_v1.0")
    path = REPO / "docs" / "img" / "filler_vs_baselines.png"
    draw(rows, path)
    head = " | ".join(label for _, label, _ in METHODS)
    print(f"dataset | responses | pairs | {head} | filler map, fragment ROC-AUC")
    for name, r in rows + [("Human labels (Poddar et al.)", human)]:
        print(f"{name} | {r['responses']} | {r['pairs']} | "
              + " | ".join(f"{100 * r['acc'][k]:.0f}%" for k, _, _ in METHODS) + f" | {r['fragment_auc']:.2f}")
    print(f"written: {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
