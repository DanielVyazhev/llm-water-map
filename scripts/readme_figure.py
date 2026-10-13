#!/usr/bin/env python3
"""The README figure: how often the filler map and the cheap metrics tell which of two responses has more filler,
per dataset, on the planted-filler checks of the development runs.

    python scripts/readme_figure.py          # → docs/img/filler_vs_baselines.png, prints the numbers

Pairwise accuracy: over all pairs of responses of one dataset whose true filler shares differ by at least MIN_GAP,
the share of pairs where the method gives the response with more filler the higher score (ties count half). 50% is
chance; with a two-class truth this is ROC-AUC. Responses: those every method scores, as in the baselines tables
(T1, R5). Also printed: the filler map's ROC-AUC per fragment (filler or wrong vs needed) and the same numbers on
the human labels of Poddar et al.

Second table, for the text next to the chart: the same with the responses from which a reader reads no fact
counted as all removable; pairs of responses to the same item only; the 95% CI of the filler map's accuracy minus
length's (bootstrap over items); the median length of the fact sentences.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MIN_GAP = 0.2
N_BOOT = 2000
SUITES = [("squad2", "SQuAD 2.0"), ("musique", "MuSiQue"), ("cuad", "CUAD"), ("poddar", "Poddar et al.")]
METHODS = [("map", "filler map (reader Qwen2.5-3B)", "#1f6fd1"),
           ("length", "length", "#5f5f5f"),
           ("yap", "Yap excess", "#a3a3a3"),
           ("stopwords", "stop-word share", "#d6d6d6")]


def pair_accuracy(truth, value, min_gap: float = MIN_GAP, groups=None) -> tuple[float, int]:
    """drybench.compare.pair_accuracy (share of pairs ranked right, ties half; groups: same-item pairs only)."""
    from drybench.compare import pair_accuracy as pa

    return pa(truth, value, min_gap, groups)


def _scored(cases: list[dict], maps: list[str]) -> list[dict]:
    return [c for c in cases if c["true_share"] == c["true_share"] and all(k in c.get("methods", {}) for k in maps)]


def _values(cs: list[dict], maps: list[str]) -> dict[str, np.ndarray]:
    from drybench.compare import share

    vals = {"map": [share(c, c["methods"][maps[0]]["removable"]) for c in cs]}
    vals.update({k: [c[k] for c in cs] for k, _, _ in METHODS[1:]})
    return {k: np.asarray(v, float) for k, v in vals.items()}


def diff_ci(cs: list[dict], vals: dict[str, np.ndarray], n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """95% CI of pair accuracy, filler map minus length, resampling items."""
    t = np.array([c["true_share"] for c in cs])
    blocks: dict[str, list[int]] = {}
    for i, c in enumerate(cs):
        blocks.setdefault(c["group"], []).append(i)
    groups = list(blocks.values())
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        d.append(pair_accuracy(t[idx], vals["map"][idx])[0] - pair_accuracy(t[idx], vals["length"][idx])[0])
    lo, hi = np.nanquantile(np.asarray(d, float), [0.025, 0.975])
    return float(lo), float(hi)


def dataset(cases: list[dict], labels: list[str], fact_chars: dict[str, list[int]] | None = None) -> dict:
    from drybench.compare import compare, fill_unscored, method_list

    maps = [f"map:{label}" for label in labels]
    cs = _scored(cases, maps)
    truth = [c["true_share"] for c in cs]
    vals = _values(cs, maps)
    acc = {k: pair_accuracy(truth, v) for k, v in vals.items()}
    groups = [c["group"] for c in cs]
    within = {k: pair_accuracy(truth, vals[k], groups=groups) for k in ("map", "length")}
    filled = _scored(fill_unscored(cases, maps), maps)
    fvals = _values(filled, maps)
    ftruth = [c["true_share"] for c in filled]
    t, _ = compare(cases, method_list(cases, labels), budget_from=maps[0], n_boot=0)
    facts = [n for c in cs for n in (fact_chars or {}).get(c["id"], [])]
    return {"responses": len(cs), "pairs": acc["map"][1], "acc": {k: a for k, (a, _) in acc.items()},
            "fragment_auc": float(t.loc[f"filler map (reader {labels[0]})", "fragments: AUC"]),
            "all": len(filled), "filled": {k: pair_accuracy(ftruth, fvals[k])[0] for k in ("map", "length")},
            "within_pairs": within["map"][1], "within": {k: a for k, (a, _) in within.items()},
            "diff_ci": diff_ci(cs, vals), "fact_median": float(np.median(facts)) if facts else float("nan")}


def fact_chars(sub: Path, folder: str) -> dict[str, list[int]]:
    """Case id -> characters of each fact sentence of a planted response."""
    from drybench.compare import read_rows

    return {f"{r['item_id']}|{r['condition']}": [len(t) for t, lab in zip(r["span_texts"], r["labels"]) if lab == "fact"]
            for r in read_rows(sub / folder / "rows.jsonl")}


def numbers(root: Path) -> tuple[list[tuple[str, dict]], dict]:
    from drybench.compare import cases_for
    from drybench.report import _readers
    from drybench.rescore import has_missing
    from drybench.types import load_items

    syn = root / "synthetic"
    subs, _ = _readers(syn, load_items(syn / "items.jsonl"))
    rows = [("Synthetic reports", dataset(cases_for("planted_synthetic", subs, syn), [l for l, _ in subs],
                                          fact_chars(subs[0][1], "filler_check")))]
    real = root / "real"
    items = load_items(real / "items.jsonl")
    subs, _ = _readers(real, items)
    labels = [label for label, _ in subs]
    missing = {i.item_id for i in items if has_missing(i)}
    planted = [c for c in cases_for("planted_real", subs, real, {i.item_id: i for i in items})
               if c["group"] not in missing]
    facts = fact_chars(subs[0][1], "filler_check_real")
    for key, name in SUITES:
        rows.append((name, dataset([c for c in planted if c["source"] == key], labels, facts)))
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
    everything = rows + [("Human labels (Poddar et al.)", human)]
    head = " | ".join(label for _, label, _ in METHODS)
    print(f"dataset | responses | pairs | {head} | filler map, fragment ROC-AUC")
    for name, r in everything:
        print(f"{name} | {r['responses']} | {r['pairs']} | "
              + " | ".join(f"{100 * r['acc'][k]:.0f}%" for k, _, _ in METHODS) + f" | {r['fragment_auc']:.2f}")
    print("\ndataset | responses: scored / also those the reader reads no fact from | the latter counted as all "
          "removable: filler map, length | pairs within one item: n, filler map, length | filler map − length, "
          "95% CI (points) | fact sentences, median characters")
    for name, r in everything:
        lo, hi = r["diff_ci"]
        within = (f"{r['within_pairs']}, {100 * r['within']['map']:.0f}%, {100 * r['within']['length']:.0f}%"
                  if r["within_pairs"] else "–")
        median = f"{r['fact_median']:.0f}" if r["fact_median"] == r["fact_median"] else "–"
        print(f"{name} | {r['responses']} / {r['all']} | {100 * r['filled']['map']:.0f}%, "
              f"{100 * r['filled']['length']:.0f}% | {within} | {100 * lo:+.1f} to {100 * hi:+.1f} | {median}")
    print(f"written: {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
