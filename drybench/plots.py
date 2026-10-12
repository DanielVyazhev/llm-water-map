"""Report figures for `drybench analyze` (matplotlib, static PNG).

fig_filler_vs_entropy   mean W by H_ans quintile (within model x variant): H2, left half
fig_tokens_vs_k        substance and filler tokens by k (answerable, unambiguous, 1-hop): H2, right half
fig_attribution        token shares of the four entropy cells per model x variant: H1
Colour = model (fixed categorical order), line style = prompt variant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
STYLES = {"default": "-", "concise": "--"}
CELL_ORDER = ["autopilot", "over_elaboration", "template_hedge", "compensation"]


def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    return plt


def _style(ax, title: str) -> None:
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=10.5)
    ax.grid(color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def report_figures(item_df: pd.DataFrame, results: dict, out_dir: str | Path) -> list[Path]:
    plt = _setup()
    out = Path(out_dir)
    paths: list[Path] = []
    df = item_df[item_df["status"] == "ok"].dropna(subset=["W"]).copy()
    if df.empty:
        return paths
    models = sorted(df["model"].unique())
    color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    # 1. filler vs answer-entropy quintile
    d = df.dropna(subset=["H_ans"]).copy()
    if not d.empty:
        d["H_pct"] = d.groupby(["model", "variant"])["H_ans"].rank(pct=True)
        d["H_q"] = np.minimum((d["H_pct"] * 5).apply(np.ceil), 5).astype(int)
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for (m, v), g in d.groupby(["model", "variant"]):
            agg = g.groupby("H_q")["W"].mean()
            ax.plot(agg.index, agg.values, STYLES.get(v, ":"), color=color[m], lw=2, marker="o", ms=5,
                    label=f"{m} ({v})")
        ax.set(xlabel="Answer entropy H_ans, quintile within model (1 = most certain)",
               ylabel="Mean filler share", xticks=[1, 2, 3, 4, 5], ylim=(0, 1))
        _style(ax, "Does filler grow with the model's uncertainty?")
        ax.legend(frameon=False, fontsize=8.5)
        fig.tight_layout()
        paths.append(out / "fig_filler_vs_entropy.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)

    # 2. substance vs filler tokens by k
    e = df[(df["answerable"].astype(bool)) & (~df["ambiguous"].astype(bool)) & (df["hops"] == 1)].copy()
    if not e.empty:
        if "filler_tokens" not in e:
            e["filler_tokens"] = e["tokens"] - e["tokens_kept"]
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
        for ax, col, title in [(axes[0], "tokens_kept", "Substance tokens |y*|"),
                               (axes[1], "filler_tokens", "Filler tokens (uninformative + repeats)")]:
            for (m, v), g in e.groupby(["model", "variant"]):
                agg = g.groupby("k")[col].mean()
                ax.plot(agg.index, agg.values, STYLES.get(v, ":"), color=color[m], lw=2, marker="o",
                        ms=5, label=f"{m} ({v})")
            ax.set(xlabel="Facts requested k")
            _style(ax, title)
        axes[0].set_ylabel("Tokens (reader tokenizer)")
        axes[1].legend(frameon=False, fontsize=8.5)
        fig.tight_layout()
        paths.append(out / "fig_tokens_vs_k.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)

    # 3. attribution shares
    att = results.get("attribution")
    if att is not None and not att.empty:
        att = att.reindex(columns=[c for c in CELL_ORDER if c in att.columns]).fillna(0)
        fig, ax = plt.subplots(figsize=(7.0, 0.55 * len(att) + 1.6))
        left = np.zeros(len(att))
        for i, c in enumerate(att.columns):
            ax.barh(range(len(att)), att[c].values, left=left, color=PALETTE[i], height=0.6,
                    edgecolor=SURFACE, linewidth=2, label=c.replace("_", " "))
            left += att[c].values
        ax.set_yticks(range(len(att)), att.index)
        ax.set(xlim=(0, 1), xlabel="Share of filler tokens")
        _style(ax, "Where the filler comes from (entropy cells)")
        ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
        fig.tight_layout()
        paths.append(out / "fig_attribution.png")
        fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
        plt.close(fig)
    return paths
