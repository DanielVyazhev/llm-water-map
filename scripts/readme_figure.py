#!/usr/bin/env python3
"""The README figure: how well the filler map and the cheap metrics order responses by their true filler share,
per dataset, on the planted-filler checks of the development runs.

    python scripts/readme_figure.py          # → docs/img/filler_vs_baselines.png, prints the numbers

Spearman correlation between each method's removable share and the true share, on the responses every method
scores (as in the baselines tables T1 and R5), the real-text check split by suite.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SUITES = [("squad2", "SQuAD 2.0"), ("musique", "MuSiQue"), ("cuad", "CUAD"), ("poddar", "Poddar et al.")]
METHODS = [("filler map (reader main)", "filler map (reader Qwen2.5-3B)", "#1f6fd1"),
           ("length (characters)", "length", "#5f5f5f"),
           ("Yap excess (characters)", "Yap excess", "#a3a3a3"),
           ("stop-word share", "stop-word share", "#d6d6d6")]
COL = "responses: Spearman"


def numbers(root: Path) -> list[tuple[str, int, dict]]:
    from drybench.report import _readers, comparison
    from drybench.rescore import has_missing
    from drybench.types import load_items

    out = []
    syn = root / "synthetic"
    subs, _ = _readers(syn, load_items(syn / "items.jsonl"))
    t, info = comparison("planted_synthetic", subs, syn)
    out.append(("Synthetic reports", info["responses"], {m: float(t.loc[m, COL]) for m, _, _ in METHODS}))
    real = root / "real"
    items = load_items(real / "items.jsonl")
    subs, _ = _readers(real, items)
    missing = {i.item_id for i in items if has_missing(i)}
    for key, name in SUITES:
        t, info = comparison("planted_real", subs, real, {i.item_id: i for i in items}, missing,
                             subset=lambda c, k=key: c["source"] == k)
        out.append((name, info["responses"], {m: float(t.loc[m, COL]) for m, _, _ in METHODS}))
    return out


def draw(rows: list[tuple[str, int, dict]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ink, ink2, grid, surf = "#161616", "#555555", "#e4e4e4", "#ffffff"
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": ink2, "xtick.color": ink, "ytick.color": ink2,
                         "figure.facecolor": surf, "axes.facecolor": surf, "savefig.facecolor": surf})
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(rows))
    w = 0.2
    for j, (key, label, color) in enumerate(METHODS):
        vals = [r[2][key] for r in rows]
        bars = ax.bar(x + (j - 1.5) * w, vals, w * 0.92, color=color, label=label,
                      edgecolor=ink2 if color == "#d6d6d6" else color, linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}".replace("-", "−"),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9,
                    color=ink if j == 0 else ink2, fontweight="bold" if j == 0 else "normal")
    ax.axhline(0, color=ink2, lw=0.8)
    ax.axvline(0.5, color=grid, lw=1.2)
    ax.set_xticks(x, [f"{name}\n{n} responses" for name, n, _ in rows])
    ax.set_ylim(-0.15, 1.08)
    ax.set_ylabel("Spearman ρ with the true filler share")
    ax.text(0, 1.065, "synthetic", ha="center", va="bottom", fontsize=10, color=ink2, transform=ax.transData)
    ax.text(2.5, 1.065, "real texts", ha="center", va="bottom", fontsize=10, color=ink2, transform=ax.transData)
    ax.set_title("Which metric orders responses by how much filler they contain?", color=ink, fontsize=13.5, pad=20)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore")
    rows = numbers(REPO / "results" / "dev_v1.0")
    path = REPO / "docs" / "img" / "filler_vs_baselines.png"
    draw(rows, path)
    print("dataset | responses | " + " | ".join(label for _, label, _ in METHODS))
    for name, n, v in rows:
        print(f"{name} | {n} | " + " | ".join(f"{v[k]:.2f}" for k, _, _ in METHODS))
    print(f"written: {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
