"""Distillation target selection by entropy bin (extension of complexity-aware fine-tuning).

For each prompt, N teacher samples are scored with the filler map (W, kappa). Easy bins (low answer
entropy) get the complete sample with the least filler; hard bins keep the most complete one
(reasoning allowed). This is data selection, not editing: targets stay fluent.
"""

from __future__ import annotations

import pandas as pd


def select_targets(samples: pd.DataFrame, prompt_col: str = "prompt_id", bin_col: str = "H_bin",
                   easy_bins: tuple[str, ...] = ("low",), kappa_min: float = 0.9,
                   w_col: str = "W", kappa_col: str = "kappa", tokens_col: str = "tokens") -> pd.DataFrame:
    picks = []
    for _, g in samples.groupby(prompt_col, sort=False):
        easy = g[bin_col].iloc[0] in easy_bins
        ok = g[g[kappa_col] >= kappa_min]
        if easy and len(ok):
            pick = ok.sort_values([w_col, tokens_col]).iloc[0]
        else:
            pick = g.sort_values([kappa_col, w_col], ascending=[False, True]).iloc[0]
        picks.append(pick)
    return pd.DataFrame(picks).reset_index(drop=True)


def selection_summary(selected: pd.DataFrame, samples: pd.DataFrame, prompt_col: str = "prompt_id",
                      cols: tuple[str, ...] = ("tokens", "W", "kappa")) -> pd.DataFrame:
    """Compare selected targets with 'first sample' and 'all samples' baselines."""
    first = samples.groupby(prompt_col, sort=False).head(1)
    rows = {"selected": selected, "first_sample": first, "all_samples": samples}
    return pd.DataFrame({name: {c: float(df[c].mean()) for c in cols} for name, df in rows.items()}).T
